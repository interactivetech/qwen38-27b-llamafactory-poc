#!/usr/bin/env python3
"""Pinned LLaMA-Factory LoRA smoke test. Run via ./setup.sh --run all.

Artifacts, source revisions, token/mask inspections, and summaries live in scratch.
Dataset content is read as data and never executed.
"""

import argparse
import datetime
import getpass
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

DEFAULT_ROOT = f"/scratch1/{getpass.getuser()}/qwen38-27b-llamafactory-test"
MODEL = "Qwen/Qwen3.8-27B"
DATASET = "shunzou05/TerminalHorizon-3K"
LF_COMMIT = "8ebba0d07dd14fdb1d429c497f018c7dae2e44b7"
CHECKPOINT_COMPAT = "Accept Transformers 5.17 default every_n_layers=1, offload=False in LLaMA-Factory checkpointing"


def checkpoint_compatibility():
    from llamafactory.model.model_utils import checkpointing

    original = checkpointing._gradient_checkpointing_enable

    def enable(
        self,
        gradient_checkpointing_kwargs=None,
        use_unsloth_gc=False,
        *,
        every_n_layers=1,
        offload=False,
    ):
        if every_n_layers != 1 or offload:
            raise ValueError(
                "This compatibility wrapper supports all-layer checkpointing without offload only"
            )
        return original(
            self,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            use_unsloth_gc=use_unsloth_gc,
        )

    checkpointing._gradient_checkpointing_enable = enable
    print(CHECKPOINT_COMPAT, flush=True)


def environment(root):
    paths = {
        "UV_CACHE_DIR": root / "cache/uv",
        "UV_PYTHON_INSTALL_DIR": root / "python",
        "PIP_CACHE_DIR": root / "cache/pip",
        "HF_HOME": root / "cache/huggingface",
        "HF_HUB_CACHE": root / "cache/huggingface/hub",
        "HF_DATASETS_CACHE": root / "cache/huggingface/datasets",
        "XDG_CACHE_HOME": root / "cache",
        "TORCH_HOME": root / "cache/torch",
        "TORCH_EXTENSIONS_DIR": root / "cache/torch_extensions",
        "TRITON_CACHE_DIR": root / "cache/triton",
        "TMPDIR": root / "tmp",
    }
    for key, value in paths.items():
        os.environ[key] = str(value)
        value.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "8")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def preflight(root):
    import importlib.metadata as md

    import torch
    import torchaudio

    # Check real wheel ABI compatibility, not only installed metadata.
    import torchvision
    from llamafactory.data import get_template_and_fix_tokenizer  # noqa: F401

    subprocess.run(
        [str(root / "bin/uv"), "pip", "check", "--python", sys.executable], check=True
    )
    assert md.version("transformers") == "5.17.0"
    assert torch.__version__.split("+")[0] == "2.10.0"
    assert torchvision.__version__.split("+")[0] == "0.25.0"
    assert torchaudio.__version__.split("+")[0] == "2.10.0"
    assert torch.version.cuda == "13.0", torch.version.cuda
    actual_commit = subprocess.check_output(
        ["git", "-C", str(root / "LLaMA-Factory"), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual_commit == LF_COMMIT
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.device_count() == 1, (
        "Expose exactly one GPU using CUDA_VISIBLE_DEVICES"
    )
    assert torch.cuda.is_bf16_supported(), "BF16 is unsupported"
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    assert torch.isfinite(y).all().item(), "CUDA BF16 matrix multiplication failed"
    result = dict(
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        compiled_architectures=torch.cuda.get_arch_list(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        transformers=md.version("transformers"),
        llamafactory_commit=actual_commit,
        bf16_matmul=True,
    )
    write_json(root / "preflight.json", result)
    print(json.dumps(result), flush=True)


def memory_check():
    import torch

    free, total = torch.cuda.mem_get_info()
    # 27B BF16 weights need ~51 GiB; reserve room for activations/loading.
    required = 100 * 1024**3
    if free < required:
        raise RuntimeError(
            f"Insufficient free GPU memory: {free / 1024**3:.1f} GiB; require 100 GiB. No processes stopped."
        )
    print(
        f"GPU memory: {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB free", flush=True
    )


def config(args, manifest):
    import yaml

    cfg = yaml.safe_load(Path(__file__).with_name("smoke_lora.yaml").read_text())
    cfg.update(
        model_revision=manifest["model_revision"],
        dataset_dir=str(args.root / "data"),
        output_dir=str(args.root / "adapters" / f"lora-{args.seq_len}"),
        cutoff_len=args.seq_len,
        max_steps=args.steps,
    )
    return cfg


def components(args, manifest):
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    cfg = config(args, manifest)
    cfg["do_train"] = (
        False  # Tokenizer setup and verification do not start/resume training.
    )
    ma, da, ta, fa, ga = get_train_args(cfg)
    tm = load_tokenizer(ma)
    template = get_template_and_fix_tokenizer(tm["tokenizer"], da)
    return ma, da, ta, fa, ga, tm, template


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

    manifest_path = args.root / "data/manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        revisions = {k: old[k] for k in ("model_revision", "dataset_revision")}
    else:
        api = HfApi()
        revisions = dict(
            model_revision=api.model_info(MODEL).sha,
            dataset_revision=api.dataset_info(DATASET).sha,
        )
    ma, da, ta, fa, ga, tm, template = components(args, revisions)
    tokenizer = tm["tokenizer"]
    processor = SupervisedDatasetProcessor(template, tokenizer, tm["processor"], da)
    stream = load_dataset(
        DATASET, split="train", revision=revisions["dataset_revision"], streaming=True
    )
    rows, metadata, inspections = [], [], []
    for source_index, row in enumerate(stream):
        messages = row["conversations"]
        if not isinstance(messages, list):
            raise ValueError(f"Unexpected conversations schema at {source_index}")
        best = None
        # Every candidate is an unmodified initial prefix ending in an assistant turn.
        for end in range(2, len(messages) + 1, 2):
            prefix = messages[:end]
            if any(
                m["role"] != ("user" if i % 2 == 0 else "assistant")
                or not isinstance(m["content"], str)
                for i, m in enumerate(prefix)
            ):
                raise ValueError(f"Unexpected role/content at source {source_index}")
            pairs = template.encode_multiturn(tokenizer, prefix, "", "")
            length = sum(len(s) + len(t) for s, t in pairs) + int(
                template.efficient_eos
            )
            if length > args.seq_len:
                break
            ids, labels = processor._encode_data_example(
                prefix[:-1], prefix[-1:], "", "", [], [], []
            )
            expected_ids, expected_labels = [], []
            for source, target in pairs:
                expected_ids.extend(source + target)
                expected_labels.extend([-100] * len(source) + target)
            if template.efficient_eos:
                expected_ids.append(tokenizer.eos_token_id)
                expected_labels.append(tokenizer.eos_token_id)
            assert ids == expected_ids and labels == expected_labels, (
                "Template/preprocessor disagreement"
            )
            if not any(v != -100 for v in labels[1:]):
                continue
            best = (prefix, ids, labels)
        if best is None:
            continue
        prefix, ids, labels = best
        rows.append({"conversations": prefix})
        metadata.append(
            dict(
                source_index=source_index,
                message_count=len(prefix),
                tokens=len(ids),
                supervised_tokens=sum(v != -100 for v in labels),
                conversation_sha256=hashlib.sha256(
                    json.dumps(prefix, ensure_ascii=False).encode()
                ).hexdigest(),
            )
        )
        inspections.append(
            dict(
                source_index=source_index,
                input_ids=ids,
                labels=labels,
                formatted=tokenizer.decode(ids),
                supervised=tokenizer.decode([v for v in labels if v != -100]),
            )
        )
        if len(rows) == args.examples:
            break
    if len(rows) != args.examples:
        raise RuntimeError(
            f"Only {len(rows)} eligible trajectories found; requested {args.examples}"
        )
    data_dir = args.root / "data"
    write_json(data_dir / "terminal_smoke.json", rows)
    write_json(
        data_dir / "dataset_info.json",
        {
            "terminal_smoke": {
                "file_name": "terminal_smoke.json",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
        },
    )
    manifest = dict(
        **revisions,
        model=MODEL,
        dataset=DATASET,
        template="qwen3_8",
        llamafactory_commit=LF_COMMIT,
        examples=args.examples,
        seq_len=args.seq_len,
        records=metadata,
        data_sha256=hashlib.sha256(
            (data_dir / "terminal_smoke.json").read_bytes()
        ).hexdigest(),
    )
    write_json(manifest_path, manifest)
    write_json(data_dir / "inspections.json", inspections)
    print(
        f"Prepared {len(rows)} intact prefixes: {[m['tokens'] for m in metadata]}",
        flush=True,
    )


def checked_manifest(args):
    manifest = read_json(args.root / "data/manifest.json")
    assert (
        manifest["seq_len"] == args.seq_len and manifest["examples"] == args.examples
    ), "Run prepare with the same overrides first"
    data = args.root / "data/terminal_smoke.json"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == manifest["data_sha256"], (
        "Prepared data changed"
    )
    return manifest


def train(args):
    import torch
    from llamafactory.train.tuner import run_exp
    from transformers import TrainerCallback

    manifest = checked_manifest(args)
    memory_check()  # Deliberately before weight download/model construction.
    checkpoint_compatibility()
    cfg = config(args, manifest)
    output = Path(cfg["output_dir"])
    if (output / "adapter_config.json").exists():
        raise RuntimeError(
            f"Adapter already exists at {output}; select a new --root to preserve it"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", cfg)
    write_json(output / "data_manifest.json", manifest)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    class Audit(TrainerCallback):
        def __init__(self):
            self.logs, self.nonzero_gradient = [], False
            self.hooks = []

        def on_train_begin(
            self, args_, state, control, model=None, train_dataloader=None, **kwargs
        ):
            self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            assert self.params, "No trainable parameters"
            assert all(
                "lora_" in n and "visual" not in n and "vision" not in n
                for n in self.params
            ), "Non-language/LoRA parameters are trainable"
            self.initial = {n: p.detach().cpu().clone() for n, p in self.params.items()}

            def watch(grad):
                if not torch.isfinite(grad).all().item():
                    raise RuntimeError("Nonfinite adapter gradient")
                if torch.count_nonzero(grad).item():
                    self.nonzero_gradient = True
                return grad

            self.hooks = [p.register_hook(watch) for p in self.params.values()]
            # Check the actual LLaMA-Factory dataset, and padding from its collator.
            ds = train_dataloader.dataset
            assert len(ds) == manifest["examples"]
            expected = read_json(args.root / "data/inspections.json")
            for item, inspected in zip(ds, expected):
                assert item["input_ids"] == inspected["input_ids"]
                assert item["labels"] == inspected["labels"]
                assert any(v != -100 for v in item["labels"][1:])
            batch = train_dataloader.collate_fn([ds[0], ds[-1]])
            assert torch.all(
                batch["labels"][batch["attention_mask"] == 0] == -100
            ).item()
            self.count = sum(p.numel() for p in self.params.values())
            print(
                f"Audit passed: {len(ds)} examples, {self.count:,} trainable LoRA parameters",
                flush=True,
            )

        def on_log(self, args_, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                assert "grad_norm" in logs, "Gradient norm missing"
                assert math.isfinite(float(logs["loss"])) and math.isfinite(
                    float(logs["grad_norm"])
                )
                self.logs.append(dict(step=state.global_step, **logs))

        def on_train_end(self, args_, state, control, model=None, **kwargs):
            changed = [
                n
                for n, p in self.params.items()
                if not torch.equal(self.initial[n], p.detach().cpu())
            ]
            assert state.global_step == args.steps, (
                f"Completed {state.global_step} steps"
            )
            assert len(self.logs) == args.steps, "Missing per-step metrics"
            assert self.nonzero_gradient and changed, "Adapter did not learn"
            self.result = dict(
                status="training_passed",
                completed_steps=state.global_step,
                elapsed_seconds=time.monotonic() - started,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                trainable_parameters=self.count,
                nonzero_adapter_gradient=self.nonzero_gradient,
                changed_adapter_tensors=len(changed),
                final_loss=self.logs[-1]["loss"],
                logs=self.logs,
                config=cfg,
                model_revision=manifest["model_revision"],
                dataset_revision=manifest["dataset_revision"],
                compatibility_adjustment=CHECKPOINT_COMPAT,
            )
            for hook in self.hooks:
                hook.remove()

    audit = Audit()
    run_exp(cfg, callbacks=[audit])
    assert (output / "adapter_model.safetensors").exists(), "Missing saved adapter"
    assert (output / "tokenizer_config.json").exists(), "Missing saved tokenizer"
    write_json(output / "summary.json", audit.result)
    print(json.dumps(audit.result), flush=True)


def verify(args):
    import torch
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.model import load_model
    from peft import PeftModel
    from transformers import AutoTokenizer

    manifest = checked_manifest(args)
    memory_check()
    ma, da, ta, fa, ga, tm, template = components(args, manifest)
    output = Path(config(args, manifest)["output_dir"])
    summary = read_json(output / "summary.json")
    assert summary["model_revision"] == manifest["model_revision"]
    tm["tokenizer"] = AutoTokenizer.from_pretrained(str(output))
    template = get_template_and_fix_tokenizer(tm["tokenizer"], da)
    # Load through the same factory, then attach with PEFT to avoid the
    # factory's default inference-time merge_and_unload behavior.
    base = load_model(tm["tokenizer"], ma, fa, is_trainable=False)
    model = PeftModel.from_pretrained(base, str(output), is_trainable=False)
    model.eval()
    inspected = read_json(args.root / "data/inspections.json")[0]
    device = next(model.parameters()).device
    ids = torch.tensor([inspected["input_ids"]], device=device)
    labels = torch.tensor([inspected["labels"]], device=device)
    with torch.inference_mode():
        loss = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels
        ).loss.item()
        assert math.isfinite(loss), "Nonfinite reloaded loss"
        prompt, _ = template.encode_oneturn(
            tm["tokenizer"],
            [
                {"role": "user", "content": "Reply with the word hello."},
                {"role": "assistant", "content": ""},
            ],
        )
        prompt_ids = torch.tensor([prompt], device=device)
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tm["tokenizer"].pad_token_id,
            eos_token_id=tm["tokenizer"].eos_token_id,
        )
    new_ids = generated[0, len(prompt) :].tolist()
    assert new_ids, "No generated tokens"
    summary.update(
        status="passed",
        verification=dict(
            fresh_process=True,
            forward_loss=loss,
            generated_token_ids=new_ids,
            generation=tm["tokenizer"].decode(new_ids),
            model_revision=manifest["model_revision"],
            template="qwen3_8",
        ),
    )
    write_json(output / "summary.json", summary)
    write_json(args.root / "summary.json", summary)
    print(json.dumps(summary["verification"]), flush=True)


def child(args, command, seq_len=None):
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        command,
        "--root",
        str(args.root),
        "--examples",
        str(args.examples),
        "--steps",
        str(args.steps),
        "--seq-len",
        str(seq_len or args.seq_len),
    ]
    log = args.root / "logs" / f"{command}-{seq_len or args.seq_len}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        log.rename(log.with_name(f"{log.stem}-{stamp}.log"))
    with log.open("w") as fp:
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            fp.write(line)
            fp.flush()
            print(line, end="", flush=True)
        code = process.wait()
    return code, log


def all_commands(args):
    preflight(args.root)
    attempts_path = args.root / "attempts.json"
    attempts = read_json(attempts_path) if attempts_path.exists() else []
    for attempt in range(2):
        for command in ("prepare", "train", "verify"):
            code, log = child(args, command)
            if code:
                error = (
                    read_json(args.root / f"{command}-failure.json")
                    if (args.root / f"{command}-failure.json").exists()
                    else {}
                )
                attempts.append(
                    dict(
                        command=command,
                        seq_len=args.seq_len,
                        exit_code=code,
                        log=str(log),
                        error=error,
                    )
                )
                write_json(args.root / "attempts.json", attempts)
                if (
                    command == "train"
                    and attempt == 0
                    and args.seq_len > 1024
                    and error.get("oom")
                ):
                    args.seq_len = 1024
                    print(
                        "CUDA OOM: retrying once with regenerated 1024-token prefixes",
                        flush=True,
                    )
                    break
                raise RuntimeError(f"{command} failed; see {log}")
        else:
            attempts.append(dict(seq_len=args.seq_len, status="passed"))
            write_json(args.root / "attempts.json", attempts)
            return
    raise RuntimeError("Both training attempts failed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sft-demo":
        from sft_demo import main as demo_main

        return demo_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] in ("sft", "overnight"):
        if sys.argv[1] == "overnight":
            print("DEPRECATED: use ./setup.sh --run sft.", file=sys.stderr, flush=True)
        from sft_train import main as sft_main

        return sft_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "quality":
        from quality_test import main as quality_main

        return quality_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        from pilot_test import main as pilot_main

        return pilot_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "prepare",
            "train",
            "verify",
            "all",
            "pilot",
            "quality",
            "sft",
            "sft-demo",
            "overnight",
        ],
    )
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ.get("SMOKE_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if min(args.examples, args.steps, args.seq_len) < 1:
        parser.error("examples, steps and seq-len must be positive")
    environment(args.root)
    try:
        if args.command == "all":
            all_commands(args)
        elif args.command == "preflight":
            preflight(args.root)
        else:
            globals()[args.command](args)
    except Exception as exc:
        write_json(
            args.root / f"{args.command}-failure.json",
            dict(
                error=str(exc),
                type=type(exc).__name__,
                traceback=traceback.format_exc(),
                oom=type(exc).__name__ == "OutOfMemoryError"
                or "CUDA out of memory" in str(exc),
                time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
