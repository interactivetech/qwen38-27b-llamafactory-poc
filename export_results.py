#!/usr/bin/env python3
"""Export small, path-free training evidence for the public repository."""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path

from smoke_test import DEFAULT_ROOT


def read(path):
    return json.loads(path.read_text())


def select(value, fields):
    return {key: value[key] for key in fields if key in value}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export(root, pilot_run, output):
    output.mkdir(parents=True, exist_ok=True)
    smoke = read(root / "summary.json")
    smoke_result = select(
        smoke,
        (
            "status",
            "completed_steps",
            "elapsed_seconds",
            "peak_gpu_allocated_bytes",
            "peak_gpu_reserved_bytes",
            "trainable_parameters",
            "nonzero_adapter_gradient",
            "changed_adapter_tensors",
            "final_loss",
            "model_revision",
            "dataset_revision",
            "compatibility_adjustment",
            "logs",
        ),
    )
    smoke_result["verification"] = select(
        smoke["verification"],
        (
            "fresh_process",
            "forward_loss",
            "model_revision",
            "template",
        ),
    )
    smoke_result["verification"]["generated_tokens"] = len(
        smoke["verification"]["generated_token_ids"]
    )
    smoke_result["training_metrics"] = read(
        root / "adapters/lora-2048/train_results.json"
    )
    smoke_result["config"] = select(
        smoke["config"],
        (
            "model_name_or_path",
            "template",
            "cutoff_len",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "learning_rate",
            "max_steps",
            "bf16",
            "packing",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "seed",
        ),
    )
    write(output / "smoke_test.json", smoke_result)
    write(output / "environment.json", read(root / "preflight.json"))
    curves = [
        dict(
            experiment="smoke_2k",
            split="train",
            optimizer_step=row["step"],
            loss=row["loss"],
            grad_norm=row["grad_norm"],
        )
        for row in smoke["logs"]
    ]
    if pilot_run is not None:
        state = read(pilot_run / "run_status.json")
        context = pilot_run / f"context-{state['seq_len']}"
        final_path = pilot_run / "summary.json"
        final = read(final_path) if final_path.exists() else None
        progress = read(context / "progress.json")
        baseline = read(context / "baseline.json")
        cfg = read(context / "run_config.json")
        manifest = read(context / "data/manifest.json")
        logs = final["logs"] if final else progress["logs"]
        evaluations = final["evaluations"] if final else progress["evaluations"]
        latest = evaluations[-1] if evaluations else None
        improvement = (
            (baseline["loss"] - latest["eval_loss"]) / baseline["loss"]
            if latest
            else None
        )
        saved = [
            int(p.name.split("-")[-1])
            for p in (context / "adapter").glob("checkpoint-*")
            if p.is_dir()
        ]
        result = dict(
            status="completed" if final else "partial",
            run_state=state["status"],
            run_id=pilot_run.name,
            seq_len=state["seq_len"],
            completed_optimizer_steps=final["completed_steps"]
            if final
            else progress["step"],
            latest_saved_step=max(saved) if saved else None,
            target_optimizer_steps=cfg["max_steps"],
            revisions=manifest["revisions"],
            seed=manifest["seed"],
            group_policy=manifest["group_policy"],
            baseline_validation=select(
                baseline, ("loss", "examples", "aggregation", "per_example_losses")
            ),
            latest_validation=latest,
            relative_validation_loss_improvement=improvement,
            five_percent_threshold_met=improvement >= 0.05
            if improvement is not None
            else None,
            final_verification_completed=bool(
                final and final.get("verification", {}).get("fresh_process")
            ),
            capacity_probe=read(context / "probe.json"),
            logs=logs,
            evaluations=evaluations,
            config=select(
                cfg,
                (
                    "model_name_or_path",
                    "template",
                    "lora_rank",
                    "lora_alpha",
                    "lora_dropout",
                    "learning_rate",
                    "lr_scheduler_type",
                    "bf16",
                    "gradient_checkpointing",
                    "gradient_accumulation_steps",
                    "per_device_train_batch_size",
                    "save_steps",
                    "eval_steps",
                    "packing",
                ),
            ),
        )
        result["data"] = {}
        for split, records in manifest["records"].items():
            result["data"][split] = dict(
                examples=len(records),
                min_tokens=min(r["tokens"] for r in records),
                max_tokens=max(r["tokens"] for r in records),
                min_messages=min(r["messages"] for r in records),
                max_messages=max(r["messages"] for r in records),
            )
        if final:
            result["final_training_metrics"] = select(
                final,
                (
                    "completed_steps",
                    "epoch",
                    "trainable_parameters",
                    "changed_adapter_tensors",
                    "nonzero_adapter_gradient",
                    "peak_gpu_allocated_bytes",
                    "peak_gpu_reserved_bytes",
                    "elapsed_seconds",
                    "final_train_loss",
                    "final_eval_loss",
                ),
            )
        write(output / "pilot_32k.json", result)
        curves.extend(
            dict(
                experiment="pilot_32k",
                split="train",
                optimizer_step=row["step"],
                loss=row["loss"],
                grad_norm=row["grad_norm"],
            )
            for row in logs
        )
        curves.append(
            dict(
                experiment="pilot_32k",
                split="validation",
                optimizer_step=0,
                loss=baseline["loss"],
                grad_norm="",
            )
        )
        curves.extend(
            dict(
                experiment="pilot_32k",
                split="validation",
                optimizer_step=row["step"],
                loss=row["eval_loss"],
                grad_norm="",
            )
            for row in evaluations
        )
    with (output / "training_curves.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            lineterminator="\n",
            fieldnames=["experiment", "split", "optimizer_step", "loss", "grad_norm"],
        )
        writer.writeheader()
        writer.writerows(curves)
    write(
        output / "export_metadata.json",
        dict(
            exported_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            description="Allowlisted metrics exported from local run artifacts; no prompts, responses, weights, or absolute paths.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ.get("SMOKE_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--pilot-run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    export(args.root, args.pilot_run, args.output)
    print(f"Exported result snapshots to {args.output}")


if __name__ == "__main__":
    main()
