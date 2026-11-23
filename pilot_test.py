#!/usr/bin/env python3
"""Held-out long-context LoRA pilot; invoke through setup.sh --run pilot."""

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import smoke_test as smoke


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def task_key(messages):
    # The terminal screen contains container IDs and is not part of the task identity.
    prompt = messages[0]["content"].split("Current terminal state:", 1)[0]
    return digest(" ".join(prompt.split()))


def complete_prefix(messages, tokenizer, template, cutoff):
    if len(messages) < 2 or any(
        m.get("role") != ("user" if i % 2 == 0 else "assistant")
        or not isinstance(m.get("content"), str)
        for i, m in enumerate(messages)
    ):
        raise ValueError("Expected alternating user/assistant text messages")
    # With thinking preserved, each pair is independent of later turns.
    even_messages = messages[: len(messages) // 2 * 2]
    pairs = template.encode_multiturn(tokenizer, even_messages, "", "")
    size, end = int(template.efficient_eos), 0
    for source, target in pairs:
        if size + len(source) + len(target) > cutoff:
            break
        size += len(source) + len(target)
        end += 2
    return even_messages[:end]


def config(args, training=False, output=None):
    import yaml

    cfg = yaml.safe_load(Path(__file__).with_name("smoke_lora.yaml").read_text())
    cfg.update(
        model_revision=args.revisions["model_revision"],
        dataset="pilot_train",
        eval_dataset="pilot_eval",
        dataset_dir=str(args.attempt / "data"),
        tokenized_path=str(args.attempt / "tokenized"),
        output_dir=str(output or args.attempt / "adapter"),
        cutoff_len=args.seq_len,
        do_train=training,
        do_eval=training,
        max_steps=args.examples // 2,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=1,
        eval_strategy="steps" if training else "no",
        eval_steps=8,
        eval_on_start=False,
        prediction_loss_only=True,
        save_strategy="steps" if training else "no",
        save_steps=8,
        save_total_limit=2,
        save_only_model=False,
        load_best_model_at_end=False,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=1,
    )
    return cfg


def components(args, training=False, saved_tokenizer=None):
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from transformers import AutoTokenizer

    cfg = config(args, training=False)
    ma, da, ta, fa, ga = get_train_args(cfg)
    tm = load_tokenizer(ma)
    if saved_tokenizer is not None:
        tm["tokenizer"] = AutoTokenizer.from_pretrained(str(saved_tokenizer))
    template = get_template_and_fix_tokenizer(tm["tokenizer"], da)
    return ma, da, ta, fa, tm, template


def checked_data(args):
    from datasets import load_from_disk

    manifest = smoke.read_json(args.attempt / "data/manifest.json")
    assert (
        manifest["seq_len"] == args.seq_len and manifest["revisions"] == args.revisions
    )
    assert manifest["seed"] == args.seed
    assert manifest["train_examples"] == args.examples
    assert manifest["eval_examples"] == args.eval_examples
    data = load_from_disk(str(args.attempt / "tokenized"))
    for split in ("train", "validation"):
        records = manifest["records"][split]
        assert len(data[split]) == len(records)
        for row, record in zip(data[split], records):
            assert (
                digest({"input_ids": row["input_ids"], "labels": row["labels"]})
                == record["tokens_sha256"]
            )
            assert 0 < len(row["input_ids"]) <= args.seq_len
            assert any(x != -100 for x in row["labels"][1:])
    train_groups = {r["task_group"] for r in manifest["records"]["train"]}
    eval_groups = {r["task_group"] for r in manifest["records"]["validation"]}
    assert not train_groups & eval_groups, "Task groups cross splits"
    assert not (
        {r["conversation_sha256"] for r in manifest["records"]["train"]}
        & {r["conversation_sha256"] for r in manifest["records"]["validation"]}
    )
    return manifest, data


def prepare(args):
    from datasets import Dataset, DatasetDict, load_dataset
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

    _, da, _, _, tm, template = components(args)
    tokenizer = tm["tokenizer"]
    processor = SupervisedDatasetProcessor(template, tokenizer, tm["processor"], da)
    selected_path = args.run_dir / "selected_trajectories.json"
    if selected_path.exists():
        selection = smoke.read_json(selected_path)
        assert (
            selection["revisions"] == args.revisions and selection["seed"] == args.seed
        )
        selected = selection["trajectories"]
        assert len(selected) == args.examples + args.eval_examples
    else:
        selected, groups, conversations = [], set(), set()
        stream = load_dataset(
            smoke.DATASET,
            split="train",
            streaming=True,
            revision=args.revisions["dataset_revision"],
        )
        for index, row in enumerate(stream):
            messages = row["conversations"]
            group, conversation_hash = task_key(messages), digest(messages)
            if group in groups or conversation_hash in conversations:
                continue
            prefix = complete_prefix(messages, tokenizer, template, args.seq_len)
            if not prefix:
                continue
            _, labels = processor._encode_data_example(
                prefix[:-1], prefix[-1:], "", "", [], [], []
            )
            if not any(x != -100 for x in labels[1:]):
                continue
            groups.add(group)
            conversations.add(conversation_hash)
            selected.append(
                dict(
                    source_index=index,
                    task_group=group,
                    conversation_sha256=conversation_hash,
                    conversations=messages,
                )
            )
            print(
                f"Selected {len(selected)}/{args.examples + args.eval_examples} independent tasks",
                flush=True,
            )
            if len(selected) == args.examples + args.eval_examples:
                break
        if len(selected) != args.examples + args.eval_examples:
            raise RuntimeError("Insufficient eligible unique task groups")
        random.Random(args.seed).shuffle(selected)
        for i, item in enumerate(selected):
            item["split"] = "train" if i < args.examples else "validation"
        smoke.write_json(
            selected_path,
            dict(seed=args.seed, revisions=args.revisions, trajectories=selected),
        )

    rows = {"train": [], "validation": []}
    records = {"train": [], "validation": []}
    conversations = {"train": [], "validation": []}
    inspections, generation_inputs = [], []
    for item in selected:
        prefix = complete_prefix(
            item["conversations"], tokenizer, template, args.seq_len
        )
        if not prefix:
            raise RuntimeError(
                f"Selected trajectory {item['source_index']} has no intact prefix at {args.seq_len}"
            )
        ids, labels = processor._encode_data_example(
            prefix[:-1], prefix[-1:], "", "", [], [], []
        )
        pairs = template.encode_multiturn(tokenizer, prefix, "", "")
        expected_ids, expected_labels = [], []
        for source, target in pairs:
            expected_ids.extend(source + target)
            expected_labels.extend([-100] * len(source) + target)
        if template.efficient_eos:
            expected_ids.append(tokenizer.eos_token_id)
            expected_labels.append(tokenizer.eos_token_id)
        assert ids == expected_ids and labels == expected_labels, (
            "Preprocessing truncated or changed a turn"
        )
        assert len(ids) <= args.seq_len and any(v != -100 for v in labels[1:])
        split = item["split"]
        rows[split].append(
            dict(
                input_ids=ids,
                attention_mask=[1] * len(ids),
                labels=labels,
                images=None,
                videos=None,
                audios=None,
            )
        )
        record = {
            key: item[key]
            for key in ("source_index", "task_group", "conversation_sha256")
        }
        record.update(
            tokens=len(ids),
            supervised_tokens=sum(v != -100 for v in labels[1:]),
            messages=len(prefix),
            tokens_sha256=digest({"input_ids": ids, "labels": labels}),
        )
        records[split].append(record)
        conversations[split].append({"conversations": prefix})
        if len(rows[split]) <= 2:
            inspections.append(
                dict(
                    split=split,
                    **record,
                    formatted=tokenizer.decode(ids),
                    supervised=tokenizer.decode([x for x in labels if x != -100]),
                )
            )
        if split == "validation" and len(generation_inputs) < 4:
            history = prefix[:-1]
            while history:
                prompt_ids, _ = template.encode_oneturn(
                    tokenizer, history + [{"role": "assistant", "content": ""}], "", ""
                )
                if len(prompt_ids) + 256 <= args.seq_len:
                    break
                history = history[:-2]
            if not history:
                raise RuntimeError(
                    "Validation prompt leaves no room for 256 generation tokens"
                )
            generation_inputs.append(
                dict(
                    source_index=item["source_index"],
                    input_ids=prompt_ids,
                    prompt_messages=history,
                    prompt_sha256=digest(prompt_ids),
                )
            )
    data_dir = args.attempt / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, name in (("train", "pilot_train"), ("validation", "pilot_eval")):
        smoke.write_json(data_dir / f"{name}.json", conversations[split])
        info[name] = dict(
            file_name=f"{name}.json",
            formatting="sharegpt",
            columns={"messages": "conversations"},
            tags=dict(
                role_tag="role",
                content_tag="content",
                user_tag="user",
                assistant_tag="assistant",
            ),
        )
    smoke.write_json(data_dir / "dataset_info.json", info)
    tokenized_path = args.attempt / "tokenized"
    if tokenized_path.exists():
        raise RuntimeError(
            f"Prepared data already exists: {tokenized_path}; use a new run directory"
        )
    DatasetDict({k: Dataset.from_list(v) for k, v in rows.items()}).save_to_disk(
        str(tokenized_path)
    )
    manifest = dict(
        revisions=args.revisions,
        seed=args.seed,
        seq_len=args.seq_len,
        train_examples=args.examples,
        eval_examples=args.eval_examples,
        records=records,
        template="qwen3_8",
        model=smoke.MODEL,
        dataset=smoke.DATASET,
        llamafactory_commit=smoke.LF_COMMIT,
        selection_sha256=digest(selected),
        group_policy="one trajectory per normalized task prompt, excluding terminal state",
    )
    smoke.write_json(data_dir / "manifest.json", manifest)
    smoke.write_json(data_dir / "inspections.json", inspections)
    smoke.write_json(data_dir / "generation_inputs.json", generation_inputs)
    checked_data(args)
    print(
        json.dumps(
            {
                split: {
                    "examples": len(v),
                    "min_tokens": min(r["tokens"] for r in v),
                    "max_tokens": max(r["tokens"] for r in v),
                    "min_messages": min(r["messages"] for r in v),
                    "max_messages": max(r["messages"] for r in v),
                }
                for split, v in records.items()
            }
        ),
        flush=True,
    )


def load(args, training=False, adapter=False):
    from llamafactory.model import load_model
    from peft import PeftModel

    smoke.memory_check()
    if training:
        smoke.checkpoint_compatibility()
    output = args.attempt / "adapter"
    ma, da, ta, fa, tm, template = components(
        args, saved_tokenizer=output if adapter else None
    )
    model = load_model(tm["tokenizer"], ma, fa, is_trainable=training)
    if adapter:
        model = PeftModel.from_pretrained(model, str(output), is_trainable=False)
    return model, ma, da, tm, template


def batch_for(row, device):
    import torch

    length = len(row["input_ids"])
    padded = (length + 7) // 8 * 8
    return dict(
        input_ids=torch.tensor(
            [row["input_ids"] + [0] * (padded - length)], device=device
        ),
        attention_mask=torch.tensor(
            [[1] * length + [0] * (padded - length)], device=device
        ),
        labels=torch.tensor(
            [row["labels"] + [-100] * (padded - length)], device=device
        ),
    )


def probe(args):
    import torch

    _, data = checked_data(args)
    longest = max(
        (row for split in data.values() for row in split),
        key=lambda row: len(row["input_ids"]),
    )
    torch.cuda.reset_peak_memory_stats()
    model, _, _, _, _ = load(args, training=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    batch = batch_for(longest, next(model.parameters()).device)
    started, forward_times, losses = time.monotonic(), [], []
    optimizer.zero_grad(set_to_none=True)
    for microstep in range(2):
        torch.cuda.synchronize()
        t = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch, use_cache=False)
            loss = out.loss / 2
        torch.cuda.synchronize()
        forward_times.append(time.monotonic() - t)
        assert torch.isfinite(loss).item(), "Nonfinite probe loss"
        losses.append(loss.item() * 2)
        loss.backward()
        del out, loss
        print(f"Capacity probe: accumulated backward {microstep + 1}/2", flush=True)
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    assert torch.isfinite(norm).item() and norm.item() > 0, (
        "Invalid probe gradient norm"
    )
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    result = dict(
        status="passed",
        seq_len=args.seq_len,
        example_tokens=len(longest["input_ids"]),
        microsteps=2,
        optimizer_steps=1,
        grad_norm=norm.item(),
        losses=losses,
        step_seconds=elapsed,
        mean_forward_seconds=sum(forward_times) / 2,
        estimated_training_seconds=elapsed * (args.examples // 2),
        estimated_validation_seconds=(sum(forward_times) / 2)
        * args.eval_examples
        * (1 + math.ceil((args.examples // 2) / 8)),
        estimate_excludes_generation=True,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        trainable_parameters=sum(p.numel() for p in params),
    )
    smoke.write_json(args.attempt / "probe.json", result)
    print(json.dumps(result), flush=True)


def score(model, data, progress=None):
    import torch

    losses = []
    model.eval()
    with torch.inference_mode():
        for i, row in enumerate(data):
            batch = batch_for(row, next(model.parameters()).device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch, use_cache=False)
            loss = float(out.loss.item())
            assert math.isfinite(loss), "Nonfinite held-out loss"
            losses.append(loss)
            del out, batch
            if progress:
                print(f"{progress}: {i + 1}/{len(data)}, loss={loss:.6f}", flush=True)
    return dict(
        loss=sum(losses) / len(losses),
        per_example_losses=losses,
        aggregation="mean per-example assistant-token cross entropy",
        examples=len(losses),
    )


def generate(model, tokenizer, prompts, label):
    import torch

    results = []
    model.eval()
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            ids = torch.tensor(
                [prompt["input_ids"]], device=next(model.parameters()).device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            tokens = generated[0, ids.shape[1] :].tolist()
            assert tokens, "No generated tokens"
            results.append(
                dict(
                    source_index=prompt["source_index"],
                    prompt_sha256=prompt["prompt_sha256"],
                    input_tokens=ids.shape[1],
                    generated_tokens=len(tokens),
                    token_ids=tokens,
                    response=tokenizer.decode(tokens, skip_special_tokens=False),
                )
            )
            del generated, ids
            print(
                f"{label} generation {i + 1}/{len(prompts)}: {len(tokens)} tokens",
                flush=True,
            )
    return results


def baseline(args):
    _, data = checked_data(args)
    model, _, _, tm, _ = load(args)
    start = time.monotonic()
    metrics = score(model, data["validation"], "Base validation")
    prompts = smoke.read_json(args.attempt / "data/generation_inputs.json")
    generations = generate(model, tm["tokenizer"], prompts, "Base")
    smoke.write_json(
        args.attempt / "baseline.json",
        dict(
            **metrics,
            generations=generations,
            model_revision=args.revisions["model_revision"],
            elapsed_seconds=time.monotonic() - start,
        ),
    )
    print(f"Base held-out loss: {metrics['loss']:.6f}", flush=True)


def checkpoint_state(args):
    checkpoint = args.resume_from_checkpoint.resolve()
    if checkpoint.parent != (args.attempt / "adapter").resolve():
        raise ValueError("Resume checkpoint must belong to this run and context")
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Incomplete resume checkpoint: {checkpoint / name}")
    state = smoke.read_json(checkpoint / "trainer_state.json")
    step = state["global_step"]
    if not 0 < step < args.examples // 2 or state["max_steps"] != args.examples // 2:
        raise ValueError("Checkpoint step budget does not match this unfinished pilot")
    losses = [entry for entry in state["log_history"] if "loss" in entry]
    if [entry["step"] for entry in losses] != list(range(1, step + 1)):
        raise ValueError("Checkpoint is missing per-step training history")
    return state


def train(args):
    import torch
    from llamafactory.train.tuner import run_exp
    from transformers import TrainerCallback

    manifest, prepared = checked_data(args)
    baseline_result = smoke.read_json(args.attempt / "baseline.json")
    assert baseline_result["model_revision"] == args.revisions["model_revision"]
    smoke.memory_check()
    smoke.checkpoint_compatibility()
    cfg = config(args, training=True)
    output = Path(cfg["output_dir"])
    resume_state = checkpoint_state(args) if args.resume_from_checkpoint else None
    if resume_state is not None:
        cfg["resume_from_checkpoint"] = str(args.resume_from_checkpoint.resolve())
    elif output.exists():
        raise RuntimeError(f"Output already exists: {output}; refusing to overwrite")
    smoke.write_json(args.attempt / "run_config.json", cfg)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()

    class Audit(TrainerCallback):
        def __init__(self):
            self.logs, self.evaluations, self.hooks = [], [], []
            if resume_state is not None:
                self.logs = [
                    entry.copy()
                    for entry in resume_state["log_history"]
                    if "loss" in entry
                ]
                self.evaluations = [
                    entry.copy()
                    for entry in resume_state["log_history"]
                    if "eval_loss" in entry
                ]
            self.nonzero = False

        def progress(self, step):
            smoke.write_json(
                args.attempt / "progress.json",
                dict(
                    status="training",
                    step=step,
                    target_steps=args.examples // 2,
                    logs=self.logs,
                    evaluations=self.evaluations,
                ),
            )

        def on_train_begin(
            self, ta, state, control, model=None, train_dataloader=None, **kwargs
        ):
            self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            assert self.params and all(
                "lora_" in n and "visual" not in n and "vision" not in n
                for n in self.params
            )
            assert all(
                p.dtype == torch.bfloat16
                for n, p in model.named_parameters()
                if not p.requires_grad and p.is_floating_point()
            ), "Base weights must be BF16"
            if resume_state is None:
                assert all(
                    torch.count_nonzero(p).item() == 0
                    for n, p in self.params.items()
                    if "lora_B" in n
                ), "Pilot must start from a fresh adapter"
            else:
                assert state.global_step == resume_state["global_step"], (
                    "Trainer did not restore the requested step"
                )
            self.initial = {n: p.detach().cpu().clone() for n, p in self.params.items()}
            ds = train_dataloader.dataset
            assert len(ds) == args.examples
            for row, reference in zip(ds, prepared["train"]):
                assert (
                    row["input_ids"] == reference["input_ids"]
                    and row["labels"] == reference["labels"]
                )
            # Two short synthetic features test padding semantics without allocating
            # two long examples on GPU in this single-example microbatch run.
            features = []
            for n in (8, 16):
                row = prepared["train"][0]
                features.append(
                    dict(
                        input_ids=row["input_ids"][:n],
                        attention_mask=[1] * n,
                        labels=row["labels"][:n],
                        images=None,
                        videos=None,
                        audios=None,
                    )
                )
            padded = train_dataloader.collate_fn(features)
            assert torch.all(
                padded["labels"][padded["attention_mask"] == 0] == -100
            ).item()

            def watch(grad):
                if not torch.isfinite(grad).all().item():
                    raise RuntimeError("Nonfinite LoRA gradient")
                self.nonzero |= bool(torch.count_nonzero(grad).item())
                return grad

            self.hooks = [p.register_hook(watch) for p in self.params.values()]
            self.count = sum(p.numel() for p in self.params.values())
            self.progress(state.global_step)
            print(
                f"Pilot audit passed: {len(ds)} examples, {self.count:,} LoRA parameters",
                flush=True,
            )

        def on_log(self, ta, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                assert "grad_norm" in logs
                assert math.isfinite(float(logs["loss"])) and math.isfinite(
                    float(logs["grad_norm"])
                )
                self.logs.append(dict(step=state.global_step, **logs))
                self.progress(state.global_step)
                print(
                    f"PILOT_STEP {state.global_step}/{args.examples // 2}: loss={logs['loss']}, grad_norm={logs['grad_norm']}",
                    flush=True,
                )

        def on_evaluate(self, ta, state, control, metrics=None, **kwargs):
            assert math.isfinite(float(metrics["eval_loss"]))
            self.evaluations.append(dict(step=state.global_step, **metrics))
            self.progress(state.global_step)
            print(
                f"PILOT_EVAL step={state.global_step}, loss={metrics['eval_loss']}",
                flush=True,
            )

        def on_train_end(self, ta, state, control, **kwargs):
            assert (
                state.global_step == args.examples // 2
                and len(self.logs) == state.global_step
            )
            assert math.isclose(state.epoch, 1.0, abs_tol=1e-5), (
                "Pilot did not complete exactly one epoch"
            )
            assert (
                self.evaluations and self.evaluations[-1]["step"] == state.global_step
            )
            changed = sum(
                not torch.equal(self.initial[n], p.detach().cpu())
                for n, p in self.params.items()
            )
            assert changed > 0 and self.nonzero
            for hook in self.hooks:
                hook.remove()
            self.result = dict(
                status="training_passed",
                completed_steps=state.global_step,
                epoch=state.epoch,
                trainable_parameters=self.count,
                changed_adapter_tensors=changed,
                nonzero_adapter_gradient=self.nonzero,
                final_train_loss=self.logs[-1]["loss"],
                final_eval_loss=self.evaluations[-1]["eval_loss"],
                baseline_eval_loss=baseline_result["loss"],
                logs=self.logs,
                evaluations=self.evaluations,
                elapsed_seconds=time.monotonic() - start,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                revisions=args.revisions,
                seq_len=args.seq_len,
                compatibility_adjustment=smoke.CHECKPOINT_COMPAT,
                resumed_from_checkpoint=cfg.get("resume_from_checkpoint"),
                initial_step=resume_state["global_step"] if resume_state else 0,
            )

    audit = Audit()
    run_exp(cfg, callbacks=[audit])
    assert (output / "adapter_model.safetensors").exists() and (
        output / "tokenizer_config.json"
    ).exists()
    checkpoints = sorted(
        output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    assert 1 <= len(checkpoints) <= 2
    for checkpoint in checkpoints:
        for name in (
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "rng_state.pth",
            "adapter_model.safetensors",
        ):
            assert (checkpoint / name).exists(), (
                f"Missing resumable checkpoint artifact: {checkpoint / name}"
            )
    audit.result["checkpoints"] = [str(p) for p in checkpoints]
    smoke.write_json(args.attempt / "training_summary.json", audit.result)
    smoke.write_json(
        args.attempt / "progress.json",
        dict(
            status="training_passed",
            step=args.examples // 2,
            target_steps=args.examples // 2,
            evaluations=audit.evaluations,
        ),
    )
    print(json.dumps(audit.result), flush=True)


def verify(args):
    manifest, data = checked_data(args)
    summary = smoke.read_json(args.attempt / "training_summary.json")
    baseline_result = smoke.read_json(args.attempt / "baseline.json")
    model, _, _, tm, template = load(args, adapter=True)
    metrics = score(model, data["validation"], "Reloaded adapter validation")
    assert math.isclose(
        metrics["loss"], summary["final_eval_loss"], rel_tol=0.005, abs_tol=0.005
    ), "Reloaded adapter loss differs from training evaluation"
    prompts = smoke.read_json(args.attempt / "data/generation_inputs.json")
    generations = generate(model, tm["tokenizer"], prompts, "Adapter")
    assert [p["prompt_sha256"] for p in generations] == [
        p["prompt_sha256"] for p in baseline_result["generations"]
    ]
    improvement = (baseline_result["loss"] - metrics["loss"]) / baseline_result["loss"]
    summary.update(
        status="passed",
        training_success=True,
        learning_signal=improvement >= 0.05,
        relative_validation_loss_improvement=improvement,
        baseline_validation=baseline_result,
        final_validation=metrics,
        verification=dict(
            fresh_process=True,
            unmerged_adapter=True,
            template="qwen3_8",
            generations=generations,
        ),
        data_manifest=manifest,
        config=smoke.read_json(args.attempt / "run_config.json"),
        capacity_probe=smoke.read_json(args.attempt / "probe.json"),
        preflight=smoke.read_json(args.root / "preflight.json"),
        dependency_lock_sha256=hashlib.sha256(
            (args.root / "requirements.lock").read_bytes()
        ).hexdigest(),
    )
    smoke.write_json(args.attempt / "summary.json", summary)
    smoke.write_json(args.run_dir / "summary.json", summary)
    print(
        json.dumps(
            dict(
                status=summary["status"],
                learning_signal=summary["learning_signal"],
                baseline_loss=baseline_result["loss"],
                final_loss=metrics["loss"],
                relative_improvement=improvement,
            )
        ),
        flush=True,
    )


def child(args, phase):
    log = args.attempt / "logs" / f"{phase}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        log.rename(log.with_name(f"{phase}-{stamp}.log"))
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--root",
        str(args.root),
        "--run-dir",
        str(args.run_dir),
        "--seq-len",
        str(args.seq_len),
        "--fallback-seq-len",
        str(args.fallback_seq_len),
        "--examples",
        str(args.examples),
        "--eval-examples",
        str(args.eval_examples),
        "--seed",
        str(args.seed),
    ]
    if phase == "train" and args.resume_from_checkpoint:
        argv.extend(["--resume-from-checkpoint", str(args.resume_from_checkpoint)])
    with log.open("x") as fp:
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        try:
            for line in process.stdout:
                fp.write(line)
                fp.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    failure_path = args.attempt / f"{phase}-failure.json"
    error = smoke.read_json(failure_path) if failure_path.exists() else {}
    if code and not error:
        error = dict(
            error=f"Process exited with code {code} without a Python error report",
            oom=False,
        )
    return dict(
        phase=phase, seq_len=args.seq_len, exit_code=code, log=str(log), error=error
    )


def resume(args):
    checked_data(args)
    if (args.run_dir / "summary.json").exists():
        raise RuntimeError(
            "Pilot already verified; refusing to restart a completed run"
        )
    if args.resume_from_checkpoint is None:
        checkpoints = sorted(
            (args.attempt / "adapter").glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            raise RuntimeError("No saved checkpoint found for this context")
        args.resume_from_checkpoint = checkpoints[-1]
    checkpoint_state(args)
    for name in ("baseline.json", "probe.json"):
        if not (args.attempt / name).exists():
            raise RuntimeError(f"Missing completed pilot phase: {name}")
    attempts_path = args.run_dir / "attempts.json"
    attempts = smoke.read_json(attempts_path) if attempts_path.exists() else []
    started = time.monotonic()
    previous = smoke.read_json(args.run_dir / "run_status.json")
    previous_elapsed = previous.get("elapsed_seconds", 0)
    for phase in ("train", "verify"):
        smoke.write_json(
            args.run_dir / "run_status.json",
            dict(
                status="running",
                phase=phase,
                seq_len=args.seq_len,
                resumed_from=str(args.resume_from_checkpoint),
            ),
        )
        result = child(args, phase)
        attempts.append(result)
        smoke.write_json(attempts_path, attempts)
        if result["exit_code"]:
            smoke.write_json(
                args.run_dir / "run_status.json",
                dict(
                    status="failed",
                    phase=phase,
                    seq_len=args.seq_len,
                    error=result["error"],
                ),
            )
            raise RuntimeError(f"Resumed {phase} failed; see {result['log']}")
    summary = smoke.read_json(args.run_dir / "summary.json")
    summary.update(
        attempts=attempts,
        run_dir=str(args.run_dir),
        fallback_used=False,
        resumed_from_checkpoint=str(args.resume_from_checkpoint),
        total_elapsed_seconds=previous_elapsed + time.monotonic() - started,
    )
    smoke.write_json(args.run_dir / "summary.json", summary)
    smoke.write_json(
        args.run_dir / "run_status.json",
        dict(
            status="passed",
            phase="complete",
            seq_len=args.seq_len,
            learning_signal=summary["learning_signal"],
            summary=str(args.run_dir / "summary.json"),
        ),
    )
    print(f"PILOT COMPLETE: {args.run_dir / 'summary.json'}", flush=True)


def all_phases(args):
    started = time.monotonic()
    if (args.run_dir / "run_status.json").exists():
        raise RuntimeError(
            "Run directory already used; choose a new directory to preserve its artifacts"
        )
    attempts = []
    for attempt, cutoff in enumerate((args.seq_len, args.fallback_seq_len)):
        args.seq_len = cutoff
        args.attempt = args.run_dir / f"context-{cutoff}"
        args.attempt.mkdir(parents=True, exist_ok=True)
        phases = ["preflight", "prepare", "probe", "baseline", "train", "verify"]
        for phase in phases:
            smoke.write_json(
                args.run_dir / "run_status.json",
                dict(
                    status="running",
                    phase=phase,
                    seq_len=cutoff,
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            result = child(args, phase)
            attempts.append(result)
            smoke.write_json(args.run_dir / "attempts.json", attempts)
            if result["exit_code"]:
                if result["error"].get("oom") and attempt == 0:
                    print(
                        f"CUDA OOM during {phase} at {cutoff}: restarting from the base model at {args.fallback_seq_len}",
                        flush=True,
                    )
                    break
                smoke.write_json(
                    args.run_dir / "run_status.json",
                    dict(
                        status="failed",
                        phase=phase,
                        seq_len=cutoff,
                        attempts=attempts,
                        elapsed_seconds=time.monotonic() - started,
                    ),
                )
                raise RuntimeError(f"{phase} failed; see {result['log']}")
        else:
            summary = smoke.read_json(args.run_dir / "summary.json")
            summary.update(
                total_elapsed_seconds=time.monotonic() - started,
                attempts=attempts,
                fallback_used=attempt > 0,
                run_dir=str(args.run_dir),
            )
            smoke.write_json(args.run_dir / "summary.json", summary)
            smoke.write_json(
                args.run_dir / "run_status.json",
                dict(
                    status="passed",
                    phase="complete",
                    seq_len=cutoff,
                    learning_signal=summary["learning_signal"],
                    summary=str(args.run_dir / "summary.json"),
                    elapsed_seconds=summary["total_elapsed_seconds"],
                ),
            )
            print(f"PILOT COMPLETE: {args.run_dir / 'summary.json'}", flush=True)
            return
    raise RuntimeError("Both context lengths failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("SMOKE_ROOT", smoke.DEFAULT_ROOT)),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--fallback-seq-len", type=int, default=16384)
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--phase",
        choices=[
            "all",
            "preflight",
            "prepare",
            "probe",
            "baseline",
            "train",
            "verify",
            "resume",
        ],
        default="all",
    )
    args = parser.parse_args(argv)
    if args.resume_from_checkpoint and args.phase not in ("resume", "train"):
        parser.error("--resume-from-checkpoint is only supported for train/resume")
    if args.examples < 16 or args.examples % 16 or args.eval_examples < 4:
        parser.error(
            "training examples must be a positive multiple of 16, and validation examples at least 4"
        )
    if not 256 < args.fallback_seq_len <= args.seq_len or (
        args.phase == "all" and args.fallback_seq_len == args.seq_len
    ):
        parser.error(
            "require 256 < fallback cutoff < primary cutoff (equal is allowed in fallback child phases)"
        )
    args.root = args.root.resolve()
    smoke.environment(args.root)
    os.environ["PYTHONPYCACHEPREFIX"] = str(args.root / "cache/pycache")
    if args.run_dir is None:
        if args.phase != "all":
            parser.error("--run-dir is required for individual phases")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        args.run_dir = args.root / "runs" / f"pilot-{stamp}"
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.attempt = args.run_dir / f"context-{args.seq_len}"
    args.attempt.mkdir(parents=True, exist_ok=True)
    source = smoke.read_json(args.root / "data/manifest.json")
    args.revisions = {k: source[k] for k in ("model_revision", "dataset_revision")}
    print(
        f"PILOT_RUN_DIR={args.run_dir} PHASE={args.phase} CUTOFF={args.seq_len}",
        flush=True,
    )
    try:
        if args.phase == "all":
            all_phases(args)
        elif args.phase == "preflight":
            smoke.preflight(args.root)
        else:
            globals()[args.phase](args)
    except Exception as exc:
        error = dict(
            error=str(exc),
            type=type(exc).__name__,
            traceback=traceback.format_exc(),
            oom=type(exc).__name__ == "OutOfMemoryError"
            or "CUDA out of memory" in str(exc),
            time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        smoke.write_json(args.attempt / f"{args.phase}-failure.json", error)
        raise


if __name__ == "__main__":
    main()
