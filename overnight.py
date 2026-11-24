"""Bounded, independently supervised 32K experiment. No GPU imports in supervisor."""

import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pilot_test as pilot
import quality_test as quality
import smoke_test as smoke

CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
)
VARIANTS = ("base", "pilot", "selected")


def read(path, default=None):
    return smoke.read_json(path) if path.exists() else default


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory():
    import torch

    return dict(
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )


def progress(args, **values):
    previous = read(args.run_dir / "progress.json", {})
    previous.update(values, updated_at=time.time(), memory=memory())
    smoke.write_json(args.run_dir / "progress.json", previous)
    print(json.dumps(previous), flush=True)


def isolated(records):
    groups, hashes, indices = set(), set(), set()
    for split, rows in records.items():
        for row in rows:
            if (
                row["task_group"] in groups
                or row["conversation_sha256"] in hashes
                or row["source_index"] in indices
            ):
                raise ValueError(f"Duplicate task/conversation/source in {split}")
            groups.add(row["task_group"])
            hashes.add(row["conversation_sha256"])
            indices.add(row["source_index"])


def encode(item, args, tokenizer, template, processor):
    prefix = pilot.complete_prefix(
        item["conversations"], tokenizer, template, args.seq_len
    )
    if not prefix:
        return None
    ids, labels = processor._encode_data_example(
        prefix[:-1], prefix[-1:], "", "", [], [], []
    )
    expected_ids, expected_labels = [], []
    for source, target in template.encode_multiturn(tokenizer, prefix, "", ""):
        expected_ids.extend(source + target)
        expected_labels.extend([-100] * len(source) + target)
    if template.efficient_eos:
        expected_ids.append(tokenizer.eos_token_id)
        expected_labels.append(tokenizer.eos_token_id)
    if ids != expected_ids or labels != expected_labels:
        raise ValueError("Preprocessing changed a complete turn or loss mask")
    if not any(x != -100 for x in labels[1:]):
        return None
    assert len(ids) <= args.seq_len
    row = dict(input_ids=ids, attention_mask=[1] * len(ids), labels=labels)
    record = {k: item[k] for k in ("source_index", "task_group", "conversation_sha256")}
    record.update(
        tokens=len(ids),
        supervised_tokens=sum(x != -100 for x in labels[1:]),
        messages=len(prefix),
        tokens_sha256=pilot.digest(dict(input_ids=ids, labels=labels)),
    )
    return row, record, prefix


def prepare(args):
    from datasets import Dataset, DatasetDict, load_dataset
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

    _, da, _, _, tm, template = pilot.components(args)
    tok = tm["tokenizer"]
    processor = SupervisedDatasetProcessor(template, tok, tm["processor"], da)
    reference = read(args.reference_run / "selected_trajectories.json")
    if reference["revisions"] != args.revisions or reference["seed"] != args.seed:
        raise ValueError("Reference revision/seed mismatch")
    selected = []
    for item in reference["trajectories"]:
        selected.append(
            dict(item, split="anchor" if item["split"] == "validation" else "train")
        )
    if (
        sum(x["split"] == "train" for x in selected) != 64
        or sum(x["split"] == "anchor" for x in selected) != 16
    ):
        raise ValueError("Expected original 64 training and 16 anchor task groups")
    groups = {x["task_group"] for x in selected}
    conversations = {x["conversation_sha256"] for x in selected}
    new = []
    needed = args.train_examples - 64 + args.val_examples + args.test_examples
    for index, row in enumerate(
        load_dataset(
            smoke.DATASET,
            split="train",
            streaming=True,
            revision=args.revisions["dataset_revision"],
        )
    ):
        if stopped(args):
            raise TimeoutError("Preparation interrupted by deadline")
        messages = row["conversations"]
        group, digest = pilot.task_key(messages), pilot.digest(messages)
        if group in groups or digest in conversations:
            continue
        item = dict(
            source_index=index,
            task_group=group,
            conversation_sha256=digest,
            conversations=messages,
        )
        if encode(item, args, tok, template, processor) is None:
            continue
        groups.add(group)
        conversations.add(digest)
        new.append(item)
        print(f"New eligible task groups: {len(new)}/{needed}", flush=True)
        if len(new) == needed:
            break
    if len(new) != needed:
        raise ValueError("Insufficient independent eligible task groups")
    random.Random(args.seed).shuffle(new)
    for i, item in enumerate(new):
        split = (
            "train"
            if i < args.train_examples - 64
            else "validation"
            if i < args.train_examples - 64 + args.val_examples
            else "test"
        )
        selected.append(dict(item, split=split))
    rows = {k: [] for k in ("train", "validation", "test", "anchor")}
    records = {k: [] for k in rows}
    cases = []
    for item in selected:
        result = encode(item, args, tok, template, processor)
        if result is None:
            raise ValueError("Reference example has no supervised complete prefix")
        row, record, prefix = result
        rows[item["split"]].append(row)
        records[item["split"]].append(record)
        if item["split"] == "test" and len(cases) < 4:
            history = prefix[:-1]
            while history:
                ids, _ = template.encode_oneturn(
                    tok, history + [dict(role="assistant", content="")], "", ""
                )
                if len(ids) + 2048 <= args.seq_len:
                    break
                history = history[:-2]
            if not history:
                raise ValueError("Test context leaves no generation budget")
            cases.append(
                dict(
                    case_id=f"test-{item['source_index']}",
                    kind="test_context",
                    source_index=item["source_index"],
                    messages=history,
                    input_ids=ids,
                    prompt_sha256=pilot.digest(ids),
                )
            )
    isolated(records)
    # The anchor and original train rows must be byte-identical to the pilot tokenization.
    ref_manifest = read(args.reference_run / "context-32768/data/manifest.json")
    for split, ref_split in (("train", "train"), ("anchor", "validation")):
        original = ref_manifest["records"][ref_split]
        for row, ref in zip(records[split], original):
            if (
                row["tokens_sha256"] != ref["tokens_sha256"]
                or row["source_index"] != ref["source_index"]
            ):
                raise ValueError("Original pilot examples changed")
    for case in quality.diagnostic_cases():
        ids, _ = template.encode_oneturn(
            tok, case["messages"] + [dict(role="assistant", content="")], "", ""
        )
        cases.append(dict(case, input_ids=ids, prompt_sha256=pilot.digest(ids)))
    for case in cases:
        assert len(case["input_ids"]) + 2048 <= args.seq_len
    # Interrupted preparation is safe to rebuild; training never rebuilds data.
    tokenized = args.attempt / "tokenized"
    if tokenized.exists():
        shutil.rmtree(tokenized)
    DatasetDict({k: Dataset.from_list(v) for k, v in rows.items()}).save_to_disk(
        str(tokenized)
    )
    smoke.write_json(
        args.run_dir / "selected_trajectories.json",
        dict(revisions=args.revisions, seed=args.seed, trajectories=selected),
    )
    smoke.write_json(args.run_dir / "cases.json", cases)
    manifest = dict(
        revisions=args.revisions,
        seq_len=args.seq_len,
        seed=args.seed,
        records=records,
        selection_sha256=pilot.digest(selected),
        counts={k: len(v) for k, v in rows.items()},
        loss_mask="assistant only, all complete turns, including reasoning",
        template="qwen3_8",
    )
    smoke.write_json(args.attempt / "data/manifest.json", manifest)
    checked_data(args)
    smoke.write_json(
        args.run_dir / "prepare.complete.json",
        dict(manifest_sha256=sha(args.attempt / "data/manifest.json")),
    )


def checked_data(args):
    from datasets import load_from_disk

    manifest = read(args.attempt / "data/manifest.json")
    assert (
        manifest["revisions"] == args.revisions and manifest["seq_len"] == args.seq_len
    )
    assert manifest["seed"] == args.seed
    assert manifest["counts"] == dict(
        train=args.train_examples,
        validation=args.val_examples,
        test=args.test_examples,
        anchor=16,
    )
    isolated(manifest["records"])
    data = load_from_disk(str(args.attempt / "tokenized"))
    for split, records in manifest["records"].items():
        assert len(data[split]) == len(records)
        for row, record in zip(data[split], records):
            assert 0 < len(row["input_ids"]) <= args.seq_len
            assert any(x != -100 for x in row["labels"][1:])
            assert (
                pilot.digest(dict(input_ids=row["input_ids"], labels=row["labels"]))
                == record["tokens_sha256"]
            )
    return manifest, data


def stopped(args, training=False):
    return (
        (args.run_dir / "STOP").exists()
        or time.time() >= args.budget["deadline"]
        or (
            training
            and (
                (args.run_dir / "STOP_TRAIN").exists()
                or time.time() >= args.budget["training_deadline"]
            )
        )
    )


def complete_checkpoint(path, identity=None, verify_files=True):
    """Only published checkpoints with checksummed completion markers are usable."""
    marker = read(path / "complete.json")
    if not marker or (identity is not None and marker["identity"] != identity):
        raise ValueError(f"Incomplete or incompatible checkpoint: {path}")
    for name in CHECKPOINT_FILES:
        if not (path / name).is_file() or (
            verify_files and sha(path / name) != marker["files"].get(name)
        ):
            raise ValueError(f"Incomplete/corrupt checkpoint: {path / name}")
    state = read(path / "trainer_state.json")
    step = state["global_step"]
    if [x["step"] for x in state["log_history"]] != list(range(1, step + 1)):
        raise ValueError("Checkpoint is missing per-step history")
    if not 0 < step <= state["max_steps"] <= 200 or state["examples_seen"] != step * 2:
        raise ValueError("Invalid checkpoint counters")
    return state


def checkpoints(args, verify_files=True):
    found = []
    for path in (args.attempt / "checkpoints").glob("checkpoint-*"):
        try:
            state = complete_checkpoint(path, args.identity, verify_files=verify_files)
            found.append((state["global_step"], path, state))
        except (ValueError, OSError, KeyError, json.JSONDecodeError):
            continue
    return sorted(found, key=lambda x: x[0])


def select_checkpoint(args):
    eligible = []
    for step, path, state in checkpoints(args):
        evaluation = read(path / "validation.json")
        if (
            evaluation
            and evaluation.get("complete")
            and math.isfinite(evaluation["loss"])
            and evaluation.get("checkpoint_sha256") == sha(path / "complete.json")
            and len(evaluation["per_example_losses"]) == args.val_examples
            and all(math.isfinite(x) for x in evaluation["per_example_losses"])
        ):
            eligible.append((evaluation["loss"], step, path))
    if not eligible:
        raise ValueError("No completed checkpoint validation; cannot select adapter")
    loss, step, path = min(eligible)
    result = dict(
        path=str(path),
        step=step,
        validation_loss=loss,
        selection="validation loss only",
    )
    smoke.write_json(args.run_dir / "selected_checkpoint.json", result)
    return result


def load_model(args, training=False, adapter=None):
    import torch
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file

    model, _, _, tm, template = pilot.load(args, training=training)
    if adapter:
        if training:
            incompatible = set_peft_model_state_dict(
                model, load_file(str(adapter / "adapter_model.safetensors"))
            )
            missing_lora = [key for key in incompatible.missing_keys if "lora_" in key]
            if incompatible.unexpected_keys or missing_lora:
                raise ValueError(
                    f"Adapter reload key mismatch: missing={missing_lora}, unexpected={incompatible.unexpected_keys}"
                )
        else:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    if training:
        model.config.use_cache = False
        assert model.is_gradient_checkpointing
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert params and all(
            "lora_" in n and "visual" not in n and "vision" not in n for n in params
        )
        assert all(
            p.dtype == torch.bfloat16
            for p in model.parameters()
            if not p.requires_grad and p.is_floating_point()
        )
        if adapter is None:
            assert all(
                torch.count_nonzero(p).item() == 0
                for n, p in params.items()
                if "lora_B" in n
            )
    return model, tm["tokenizer"], template


def optimizer_step(model, optimizer, rows):
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    for row in rows:
        batch = pilot.batch_for(row, next(model.parameters()).device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch, use_cache=False)
            loss = output.loss / 2
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite training loss")
        loss_value += loss.item()
        loss.backward()
        del loss, output, batch
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if not torch.isfinite(norm).item() or norm.item() <= 0:
        raise ValueError("Nonfinite or zero adapter gradients")
    optimizer.step()
    return loss_value, norm.item()


def probe(args):
    import torch

    _, data = checked_data(args)
    longest = max(
        (r for split in data.values() for r in split), key=lambda r: len(r["input_ids"])
    )
    torch.manual_seed(args.seed)
    model, _, _ = load_model(args, training=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0
    )
    start = time.monotonic()
    loss, norm = optimizer_step(model, optimizer, [longest, longest])
    torch.cuda.synchronize()
    smoke.write_json(
        args.run_dir / "probe.json",
        dict(
            complete=True,
            loss=loss,
            grad_norm=norm,
            example_tokens=len(longest["input_ids"]),
            seconds=time.monotonic() - start,
            memory=memory(),
        ),
    )
    # This process exits; its disposable adapter never enters the training process.


def score(args, model, rows, path, training=False):
    import torch

    results = []
    model.eval()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            if stopped(args, training):
                return None
            batch = pilot.batch_for(row, next(model.parameters()).device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**batch, use_cache=False)
            value = output.loss.item()
            if not math.isfinite(value):
                raise ValueError("Nonfinite evaluation loss")
            results.append(value)
            del output, batch
            smoke.write_json(
                path,
                dict(
                    complete=False,
                    per_example_losses=results,
                    expected_examples=len(rows),
                ),
            )
            progress(
                args,
                activity="evaluation",
                evaluation=str(path),
                evaluated_examples=i + 1,
            )
    result = dict(
        complete=True,
        loss=sum(results) / len(results),
        per_example_losses=results,
        examples=len(results),
        aggregation="mean per-example assistant-token cross entropy",
    )
    smoke.write_json(path, result)
    torch.cuda.empty_cache()
    return result


def save_checkpoint(args, model, tokenizer, optimizer, scheduler, state):
    import torch

    parent = args.attempt / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"checkpoint-{state['global_step']}"
    if path.exists():
        complete_checkpoint(path, args.identity)
        smoke.write_json(
            args.run_dir / "latest_checkpoint.json",
            dict(path=str(path), step=state["global_step"]),
        )
        return path
    temp = parent / f".checkpoint-{state['global_step']}-{os.getpid()}"
    temp.mkdir()
    model.save_pretrained(temp, safe_serialization=True)
    tokenizer.save_pretrained(temp)
    torch.save(optimizer.state_dict(), temp / "optimizer.pt")
    torch.save(scheduler.state_dict(), temp / "scheduler.pt")
    torch.save(
        dict(
            python=random.getstate(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(),
        ),
        temp / "rng_state.pth",
    )
    smoke.write_json(temp / "trainer_state.json", state)
    smoke.write_json(
        temp / "complete.json",
        dict(
            identity=args.identity, files={n: sha(temp / n) for n in CHECKPOINT_FILES}
        ),
    )
    # Flush artifacts before publishing the directory and its completion marker.
    for file in temp.iterdir():
        if file.is_file():
            with file.open("rb") as fp:
                os.fsync(fp.fileno())
    temp.rename(path)
    smoke.write_json(
        args.run_dir / "latest_checkpoint.json",
        dict(path=str(path), step=state["global_step"]),
    )
    torch.cuda.empty_cache()
    return path


def train(args):
    import torch

    manifest, data = checked_data(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    saved = checkpoints(args)
    resume = saved[-1] if saved else None
    model, tokenizer, _ = load_model(
        args, training=True, adapter=resume[1] if resume else None
    )
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    max_steps = min(200, args.train_examples // 2)
    state = dict(
        global_step=0,
        max_steps=max_steps,
        log_history=[],
        examples_seen=0,
        non_padding_tokens=0,
        supervised_tokens=0,
        evaluations=[],
        verified_updates=False,
    )
    if resume:
        state = resume[2]
        assert state["max_steps"] == max_steps
        optimizer.load_state_dict(
            torch.load(resume[1] / "optimizer.pt", weights_only=True)
        )
        scheduler.load_state_dict(
            torch.load(resume[1] / "scheduler.pt", weights_only=True)
        )
        rng = torch.load(resume[1] / "rng_state.pth", weights_only=False)
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
    order = list(range(args.train_examples))
    random.Random(args.seed).shuffle(order)
    while state["global_step"] < max_steps and not stopped(args, training=True):
        start = time.monotonic()
        offset = state["global_step"] * 2
        indices = order[offset : offset + 2]
        rows = [data["train"][i] for i in indices]
        loss, norm = optimizer_step(model, optimizer, rows)
        scheduler.step()
        state["global_step"] += 1
        step = state["global_step"]
        state["examples_seen"] += len(rows)
        state["non_padding_tokens"] += sum(len(row["input_ids"]) for row in rows)
        state["supervised_tokens"] += sum(
            manifest["records"]["train"][i]["supervised_tokens"] for i in indices
        )
        entry = dict(
            step=step,
            loss=loss,
            grad_norm=norm,
            seconds=time.monotonic() - start,
            examples_seen=state["examples_seen"],
            non_padding_tokens=state["non_padding_tokens"],
            memory=memory(),
        )
        state["log_history"].append(entry)
        progress(
            args,
            activity="training",
            **entry,
            latest_checkpoint=read(args.run_dir / "latest_checkpoint.json"),
        )
        final = step == max_steps or stopped(args, training=True)
        if step % 8 == 0 or final:
            changed = sum(
                not torch.equal(initial[n], p.detach().cpu()) for n, p in params.items()
            )
            if changed == 0:
                raise ValueError("Adapter weights did not change")
            state["verified_updates"] = True
            state["changed_adapter_tensors"] = changed
            checkpoint = save_checkpoint(
                args, model, tokenizer, optimizer, scheduler, state
            )
            if step % 16 == 0 or final:
                result = score(
                    args,
                    model,
                    data["validation"],
                    checkpoint / "validation.json",
                    training=True,
                )
                if result:
                    result["checkpoint_sha256"] = sha(checkpoint / "complete.json")
                    smoke.write_json(checkpoint / "validation.json", result)
                    select_checkpoint(args)
        if final:
            break
    # A deadline may arrive between steps. Save the last completed optimizer boundary.
    if state["global_step"]:
        changed = sum(
            not torch.equal(initial[n], p.detach().cpu()) for n, p in params.items()
        )
        if changed:
            state["verified_updates"] = True
            state["changed_adapter_tensors"] = changed
        if not state["verified_updates"]:
            raise ValueError("No verified adapter weight updates")
        checkpoint = save_checkpoint(
            args, model, tokenizer, optimizer, scheduler, state
        )
    smoke.write_json(
        args.run_dir / "training_summary.json",
        dict(
            state,
            complete=True,
            budget_limited=state["global_step"] < max_steps,
            memory=memory(),
        ),
    )


def evaluate(args, variant, generations=False):
    import torch

    _, data = checked_data(args)
    selection = select_checkpoint(args) if variant == "selected" else None
    adapter = (
        Path(selection["path"])
        if selection
        else args.reference_run / "context-32768/adapter"
        if variant == "pilot"
        else None
    )
    model, tokenizer, _ = load_model(args, adapter=adapter)
    if not generations:
        result_path = args.run_dir / f"{variant}-losses.json"
        results = read(result_path, {})
        # Selected validation verifies reload in a fresh process before held-out scoring.
        for split in (
            ("validation", "test", "anchor")
            if variant == "selected"
            else ("test", "anchor")
        ):
            if results.get(split, {}).get("complete"):
                continue
            result = score(
                args,
                model,
                data[split],
                args.run_dir / f"{variant}-{split}-partial.json",
            )
            if result is None:
                break
            results[split] = result
            if split == "validation":
                if not math.isclose(
                    result["loss"],
                    selection["validation_loss"],
                    rel_tol=0.005,
                    abs_tol=0.005,
                ):
                    raise ValueError(
                        "Fresh-process adapter reload changed validation loss"
                    )
                results["fresh_process_reload_verified"] = True
                results["checkpoint"] = selection
            smoke.write_json(result_path, results)
        return
    cases = read(args.run_dir / "cases.json")
    output_path = args.run_dir / f"{variant}-responses.json"
    records = read(output_path, [])
    model.eval()
    with torch.inference_mode():
        for case in cases[len(records) :]:
            if stopped(args):
                break
            start = time.monotonic()
            ids = torch.tensor(
                [case["input_ids"]], device=next(model.parameters()).device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=2048,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            tokens = output[0, ids.shape[1] :].tolist()
            response = tokenizer.decode(tokens, skip_special_tokens=False)
            parsed = quality.parse_response(response)
            record = dict(
                case_id=case["case_id"],
                kind=case["kind"],
                variant=variant,
                prompt_sha256=case["prompt_sha256"],
                response=response,
                token_ids=tokens,
                generated_tokens=len(tokens),
                seconds=time.monotonic() - start,
                token_cap_reached=len(tokens) == 2048
                and tokens[-1] != tokenizer.eos_token_id,
                **parsed,
            )
            if case["kind"] == "diagnostic":
                record["diagnostic_pass"] = quality.diagnostic_pass(case, parsed)
            records.append(record)
            smoke.write_json(output_path, records)
            del ids, output
            torch.cuda.empty_cache()
            progress(
                args,
                activity="generation",
                variant=variant,
                completed_responses=len(records),
            )


def validate(args):
    """Finish stop-time validation in a separate process within the final budget."""
    _, data = checked_data(args)
    saved = checkpoints(args)
    if not saved:
        raise ValueError("Training produced no completed checkpoint")
    _, checkpoint, _ = saved[-1]
    existing = read(checkpoint / "validation.json", {})
    if not existing.get("complete") or "checkpoint_sha256" not in existing:
        model, _, _ = load_model(args, adapter=checkpoint)
        result = score(args, model, data["validation"], checkpoint / "validation.json")
        if result:
            result["checkpoint_sha256"] = sha(checkpoint / "complete.json")
            smoke.write_json(checkpoint / "validation.json", result)
    select_checkpoint(args)


def bootstrap(left, right, seed=42, draws=4000):
    """Paired resampling of task examples; positive means lower right loss."""
    if len(left) != len(right) or not left:
        raise ValueError("Paired bootstrap needs equal nonempty task lists")
    rng = random.Random(seed)
    estimates, absolute = [], []
    for _ in range(draws):
        indices = rng.choices(range(len(left)), k=len(left))
        a = sum(left[i] for i in indices) / len(indices)
        b = sum(right[i] for i in indices) / len(indices)
        estimates.append((a - b) / a)
        absolute.append(a - b)
    estimates.sort()
    absolute.sort()
    a, b = sum(left) / len(left), sum(right) / len(right)
    return dict(
        relative_improvement=(a - b) / a,
        mean_loss_reduction=a - b,
        relative_95_interval=[
            estimates[int(draws * 0.025)],
            estimates[int(draws * 0.975)],
        ],
        loss_reduction_95_interval=[
            absolute[int(draws * 0.025)],
            absolute[int(draws * 0.975)],
        ],
        paired_tasks=len(left),
        bootstrap_draws=draws,
        seed=seed,
    )


def report(args, status, failure=None):
    """Always CPU-only and safe when preparation or evaluation is incomplete."""
    losses = {v: read(args.run_dir / f"{v}-losses.json", {}) for v in VARIANTS}
    responses = {v: read(args.run_dir / f"{v}-responses.json", []) for v in VARIANTS}
    comparisons = {}
    for split in ("test", "anchor"):
        comparisons[split] = {}
        for name, left, right in (
            ("base_to_selected", "base", "selected"),
            ("pilot_to_selected", "pilot", "selected"),
            ("base_to_pilot", "base", "pilot"),
        ):
            a, b = losses[left].get(split, {}), losses[right].get(split, {})
            if a.get("complete") and b.get("complete"):
                comparisons[split][name] = bootstrap(
                    a["per_example_losses"], b["per_example_losses"]
                )
    paired = []
    cases = read(args.run_dir / "cases.json", [])
    for case in cases:
        row = dict(
            case_id=case["case_id"],
            kind=case["kind"],
            prompt_sha256=case["prompt_sha256"],
            responses={},
        )
        for variant in VARIANTS:
            matching = [
                r for r in responses[variant] if r["case_id"] == case["case_id"]
            ]
            if matching:
                if matching[0]["prompt_sha256"] != case["prompt_sha256"]:
                    raise ValueError("Response prompt mismatch")
                row["responses"][variant] = matching[0]
        paired.append(row)
    smoke.write_json(args.run_dir / "paired_responses.json", paired)
    paired_text = [
        "# Paired responses",
        "",
        "Generated commands were not executed.",
        "",
    ]
    for row in paired:
        paired_text.extend([f"## {row['case_id']}", ""])
        for v in VARIANTS:
            paired_text.extend(
                [
                    f"### {v}",
                    "",
                    row["responses"].get(v, {}).get("response", "Unfinished"),
                    "",
                ]
            )
    (args.run_dir / "paired_responses.md").write_text("\n".join(paired_text))
    saved = checkpoints(args, verify_files=False)
    latest = dict(path=str(saved[-1][1]), step=saved[-1][0]) if saved else None
    training = read(
        args.run_dir / "training_summary.json", saved[-1][2] if saved else {}
    )
    selected = read(args.run_dir / "selected_checkpoint.json")
    skipped = []
    for v in VARIANTS:
        for split in ("test", "anchor"):
            if not losses[v].get(split, {}).get("complete"):
                skipped.append(f"{v} {split} loss unfinished")
        if len(responses[v]) != 8:
            skipped.append(f"{v} generations {len(responses[v])}/8 complete")
    verified = bool(
        training.get("verified_updates")
        and training.get("global_step", 0) > 0
        and losses["selected"].get("fresh_process_reload_verified")
    )
    test = comparisons["test"]
    promising = (
        verified
        and test.get("base_to_selected", {}).get("relative_improvement", -1) >= 0.05
        and test.get("pilot_to_selected", {}).get("relative_improvement", -1) > 0
    )
    quality_metrics = {
        v: dict(
            completed=len(rows),
            json_valid=sum(r["json_valid"] for r in rows),
            schema_valid=sum(r["schema_valid"] for r in rows),
            truncated=sum(r["token_cap_reached"] for r in rows),
            instruction_pass=sum(r.get("diagnostic_pass", False) for r in rows),
            instruction_total=sum(r["kind"] == "diagnostic" for r in rows),
        )
        for v, rows in responses.items()
    }
    attempts = read(args.run_dir / "attempts.json", [])
    summary = dict(
        status=status,
        failure=failure,
        budget=args.budget,
        elapsed_seconds=time.time() - args.budget["started_at"],
        run_dir=str(args.run_dir),
        training_verified=verified,
        promising_scaling_result=promising,
        latest_checkpoint=latest,
        selected_checkpoint=selected,
        training=training,
        comparisons=comparisons,
        losses=losses,
        quality=quality_metrics,
        attempts=attempts,
        last_observed_progress=read(args.run_dir / "progress.json", {}),
        skipped_work=skipped,
        configuration=read(args.run_dir / "experiment.json"),
        limitations="Loss and eight static generations do not establish task success; commands were not executed. Bootstrap is over task groups, not training seeds.",
    )
    smoke.write_json(args.run_dir / "summary.json", summary)
    history = training.get("log_history", [])
    with (args.run_dir / "training_curves.csv").open("w") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "step",
                "loss",
                "grad_norm",
                "seconds",
                "examples_seen",
                "non_padding_tokens",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(history)
    with (args.run_dir / "validation_curves.csv").open("w") as fp:
        writer = csv.DictWriter(fp, fieldnames=["step", "loss", "complete"])
        writer.writeheader()
        for step, path, _ in saved:
            validation = read(path / "validation.json", {})
            if validation:
                writer.writerow(
                    dict(
                        step=step,
                        loss=validation.get("loss"),
                        complete=validation.get("complete", False),
                    )
                )
    lines = [
        "# Overnight 32K experiment",
        "",
        f"Status: **{status}**. Training verified: {verified}.",
        "",
        f"Promising scaling result: {promising}.",
        "",
        f"Completed optimizer steps: {training.get('global_step', 0)}; examples: {training.get('examples_seen', 0)}; non-padding tokens: {training.get('non_padding_tokens', 0)}.",
        "",
        f"Selected checkpoint: {selected}",
        "",
        f"Latest resumable checkpoint: {latest}",
        "",
        "| Set | Model | Mean loss |",
        "| --- | --- | --- |",
    ]
    for split in ("test", "anchor"):
        for v in VARIANTS:
            value = losses[v].get(split, {}).get("loss", "unfinished")
            lines.append(f"| {split} | {v} | {value} |")
    lines.extend(["", "Relative improvements and paired bootstrap 95% intervals:", ""])
    for split, values in comparisons.items():
        for name, value in values.items():
            lines.append(
                f"- {split}, {name}: {value['relative_improvement']:.2%}, interval {value['relative_95_interval']}"
            )
    lines.extend(
        [
            "",
            "Output checks (counts include completed responses only):",
            "",
            "| Model | Responses | JSON | Schema | Truncated | Instructions |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for v, q in quality_metrics.items():
        lines.append(
            f"| {v} | {q['completed']}/8 | {q['json_valid']} | {q['schema_valid']} | {q['truncated']} | {q['instruction_pass']}/{q['instruction_total']} |"
        )
    lines.extend(
        [
            "",
            "## Runtime and limitations",
            "",
            f"Elapsed seconds: {summary['elapsed_seconds']:.1f}. Budget: {args.budget}",
            "",
            f"Training memory: {training.get('memory', {})}",
            "",
            "Phase timings, worker memory, losses, and failures are in summary.json, attempts.json, progress.json, and logs/. Training curves are in training_curves.csv.",
            "",
            summary["limitations"],
            "",
            f"Failure: {failure}",
            "",
            "Unfinished work:",
            "",
        ]
    )
    lines.extend(f"- {x}" for x in skipped)
    (args.run_dir / "report.md").write_text("\n".join(lines) + "\n")


def initialize(args):
    """Freeze experiment identity and absolute deadlines before any model work."""
    path = args.run_dir / "experiment.json"
    reference_manifest = args.reference_run / "context-32768/data/manifest.json"
    manifest = read(reference_manifest)
    lock = args.root / "requirements.lock"
    reference_summary = read(args.reference_run / "summary.json", {})
    expected_lock = reference_summary.get("dependency_lock_sha256")
    if expected_lock and sha(lock) != expected_lock:
        raise ValueError("Dependency lock differs from the reference pilot")
    selection = args.reference_run / "selected_trajectories.json"
    adapter = args.reference_run / "context-32768/adapter"
    identity = dict(
        revisions=manifest["revisions"],
        seq_len=args.seq_len,
        seed=42,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        test_examples=args.test_examples,
        reference_run=str(args.reference_run),
        reference_manifest_sha256=sha(reference_manifest),
        reference_selection_sha256=sha(selection),
        pilot_adapter_sha256=sha(adapter / "adapter_model.safetensors"),
        pilot_adapter_config_sha256=sha(adapter / "adapter_config.json"),
        dependency_lock_sha256=sha(lock),
        hours=args.hours,
        hyperparameters=dict(
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0,
            bf16=True,
            attention="sdpa",
            gradient_checkpointing=True,
            microbatch=1,
            accumulation=2,
            optimizer="AdamW",
            weight_decay=0,
            max_grad_norm=1,
            learning_rate=1e-4,
            scheduler="constant",
            max_steps=min(200, args.train_examples // 2),
            save_steps=8,
            eval_steps=16,
            save_total_limit=None,
        ),
    )
    if manifest["seq_len"] != 32768 or manifest["seed"] != 42:
        raise ValueError("Reference must be the approved 32K pilot")
    current = read(path)
    if current:
        if current["identity"] != identity:
            raise ValueError("Resume configuration/dependency/reference changed")
        if not args.resume and args.phase == "all":
            raise ValueError(
                "Run already exists; use --resume to preserve its deadline"
            )
    else:
        if args.resume or args.phase != "all":
            raise ValueError("No original experiment to resume")
        now = time.time()
        budget = dict(
            started_at=now,
            deadline=now + args.hours * 3600,
            training_deadline=now + (args.hours - 1.5) * 3600,
        )
        current = dict(identity=identity, budget=budget)
        smoke.write_json(path, current)
    args.budget = current["budget"]
    args.identity = pilot.digest(identity)
    args.revisions = identity["revisions"]
    args.seed, args.examples, args.eval_examples = (
        42,
        args.train_examples,
        args.val_examples,
    )
    return current


def heartbeat(args, phase, **extra):
    now = time.time()
    value = dict(
        status="running",
        phase=phase,
        timestamp=now,
        elapsed_seconds=now - args.budget["started_at"],
        remaining_seconds=max(0, args.budget["deadline"] - now),
        training_remaining_seconds=max(0, args.budget["training_deadline"] - now),
        progress=read(args.run_dir / "progress.json", {}),
        latest_checkpoint=read(args.run_dir / "latest_checkpoint.json"),
        **extra,
    )
    smoke.write_json(args.run_dir / "status.json", value)


def supervise_worker(
    args, phase, command=None, clock=time.time, sleep=time.sleep, popen=subprocess.Popen
):
    """Poll independently of worker stdout/CUDA; signal only this process group."""
    log_dir = args.run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = log_dir / f"{phase}-{stamp}.log"
    if command is None:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--root",
            str(args.root),
            "--run-dir",
            str(args.run_dir),
            "--reference-run",
            str(args.reference_run),
            "--hours",
            str(args.hours),
            "--seq-len",
            str(args.seq_len),
            "--train-examples",
            str(args.train_examples),
            "--val-examples",
            str(args.val_examples),
            "--test-examples",
            str(args.test_examples),
        ]
    worker_deadline = (
        args.budget["deadline"] - 15
    )  # Leave time for CPU-only partial reporting.
    start = clock()
    stop_at = None
    term_at = None
    next_heartbeat = start
    reason = None
    with log.open("x") as fp:
        process = popen(
            command, stdout=fp, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                now = clock()
                if now >= next_heartbeat:
                    heartbeat(args, phase, worker_pid=process.pid)
                    next_heartbeat = now + 30
                if phase == "train" and now >= args.budget["training_deadline"]:
                    (args.run_dir / "STOP_TRAIN").touch()
                    if stop_at is None:
                        stop_at, reason = now, "training deadline"
                if now >= worker_deadline - 60 or (args.run_dir / "STOP").exists():
                    (args.run_dir / "STOP").touch()
                    if stop_at is None:
                        stop_at, reason = now, "overall deadline or interruption"
                # Give an optimizer boundary/checkpoint up to two minutes, within total budget.
                if (
                    (stop_at is not None and now - stop_at >= 120)
                    or now >= worker_deadline - 10
                ) and term_at is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    term_at = now
                if (
                    term_at is not None and now - term_at >= 10
                ) or now >= worker_deadline:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                sleep(min(1, max(0.01, worker_deadline - now)))
            code = process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    failure = read(args.run_dir / f"{phase}-failure.json")
    return dict(
        phase=phase,
        exit_code=code,
        seconds=clock() - start,
        log=str(log),
        shutdown_reason=reason,
        forced=term_at is not None,
        failure=failure,
    )


def run(args):
    # An advisory lock is automatically released after a supervisor crash.
    with (args.run_dir / ".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another supervisor owns this run")
        if args.resume:
            (args.run_dir / "STOP").unlink(missing_ok=True)
            (args.run_dir / "STOP_TRAIN").unlink(missing_ok=True)
        (args.root / "latest_overnight_run.txt").write_text(str(args.run_dir) + "\n")
        attempts = read(args.run_dir / "attempts.json", [])
        failure = None
        status = "partial"

        def interrupt(signum, frame):
            (args.run_dir / "STOP").touch()

        old_handlers = {
            s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            phases = ["preflight", "prepare", "probe", "train", "validate"]
            phases += [f"loss-{v}" for v in VARIANTS] + [
                f"generate-{v}" for v in VARIANTS
            ]
            for phase in phases:
                if stopped(args):
                    break
                if (
                    phase == "prepare"
                    and (args.run_dir / "prepare.complete.json").exists()
                ):
                    if read(args.run_dir / "prepare.complete.json")[
                        "manifest_sha256"
                    ] != sha(args.attempt / "data/manifest.json"):
                        raise ValueError("Prepared manifest changed")
                    continue
                if phase == "probe" and read(args.run_dir / "probe.json", {}).get(
                    "complete"
                ):
                    continue
                if phase == "train" and (
                    time.time() >= args.budget["training_deadline"]
                    or read(args.run_dir / "training_summary.json", {}).get("complete")
                ):
                    continue
                if (
                    phase.startswith(("loss-", "generate-"))
                    and not (args.run_dir / "selected_checkpoint.json").exists()
                ):
                    raise ValueError("No validated selected adapter")
                print(f"OVERNIGHT phase={phase} run={args.run_dir}", flush=True)
                heartbeat(args, phase)
                result = supervise_worker(args, phase)
                attempts.append(result)
                smoke.write_json(args.run_dir / "attempts.json", attempts)
                # A watchdog kill is an explicit failure, never a fallback experiment.
                if result["exit_code"]:
                    failure = result
                    status = "failed"
                    break
                report(args, "running")
            else:
                status = "complete"
        except BaseException as exc:
            failure = dict(
                type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            status = "failed"
        finally:
            report(args, status, failure)
            summary = read(args.run_dir / "summary.json")
            if status == "complete" and summary["skipped_work"]:
                status = "partial"
                report(args, status, failure)
            heartbeat(args, "finished")
            final = read(args.run_dir / "status.json")
            final.update(
                status=status, failure=failure, report=str(args.run_dir / "report.md")
            )
            smoke.write_json(args.run_dir / "status.json", final)
            for s, previous in old_handlers.items():
                signal.signal(s, previous)
        return 1 if status == "failed" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("SMOKE_ROOT", smoke.DEFAULT_ROOT)),
    )
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--train-examples", type=int, default=400)
    parser.add_argument("--val-examples", type=int, default=32)
    parser.add_argument("--test-examples", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "preflight", "prepare", "probe", "train", "validate"]
        + [f"{p}-{v}" for p in ("loss", "generate") for v in VARIANTS],
    )
    args = parser.parse_args(argv)
    if (
        args.seq_len != 32768
        or not 1.5 < args.hours <= 12
        or not 64 <= args.train_examples <= 400
        or args.train_examples % 2
        or min(args.val_examples, args.test_examples) < 4
    ):
        parser.error(
            "Require 32K, 1.5 < hours <= 12, even 64..400 training examples, and >=4 validation/test examples"
        )
    args.root, args.run_dir, args.reference_run = (
        p.resolve() for p in (args.root, args.run_dir, args.reference_run)
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.attempt = args.run_dir / "context-32768"
    args.attempt.mkdir(exist_ok=True)
    smoke.environment(args.root)
    initialize(args)
    if args.phase == "all":
        return run(args)
    try:
        if args.phase == "preflight":
            smoke.preflight(args.root)
            smoke.write_json(
                args.run_dir / "preflight.json", read(args.root / "preflight.json")
            )
            if shutil.disk_usage(args.run_dir).free < 24 * 1024**3:
                raise RuntimeError(
                    "Require 24 GiB free for retained checkpoints and data"
                )
        elif args.phase.startswith("loss-"):
            evaluate(args, args.phase.removeprefix("loss-"))
        elif args.phase.startswith("generate-"):
            evaluate(args, args.phase.removeprefix("generate-"), generations=True)
        else:
            globals()[args.phase](args)
    except BaseException as exc:
        smoke.write_json(
            args.run_dir / f"{args.phase}-failure.json",
            dict(
                type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                oom="out of memory" in str(exc).lower(),
            ),
        )
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
