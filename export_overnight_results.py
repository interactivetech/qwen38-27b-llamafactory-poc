"""Export allowlisted overnight metrics and reproducible learning-curve figures."""

import argparse
import csv
import datetime
import hashlib
import re
from pathlib import Path

from export_results import read, select, write


def export(run, output, include_responses=False):
    summary = read(run / "summary.json")
    training = summary["training"]
    identity = read(run / "experiment.json")["identity"]
    output.mkdir(parents=True, exist_ok=True)
    result = select(
        summary,
        (
            "status",
            "elapsed_seconds",
            "training_verified",
            "promising_scaling_result",
            "comparisons",
            "quality",
            "skipped_work",
        ),
    )
    result.update(
        run_id=run.name,
        finished_at=datetime.datetime.fromtimestamp(
            summary["budget"]["started_at"] + summary["elapsed_seconds"],
            datetime.timezone.utc,
        ).isoformat(),
        selected_checkpoint=select(
            summary["selected_checkpoint"], ("step", "validation_loss", "selection")
        ),
        latest_checkpoint_step=summary["latest_checkpoint"]["step"],
        configuration=select(
            identity,
            (
                "revisions",
                "seq_len",
                "seed",
                "train_examples",
                "val_examples",
                "test_examples",
                "hours",
                "hyperparameters",
                "dependency_lock_sha256",
            ),
        ),
        training=select(
            training,
            (
                "global_step",
                "max_steps",
                "examples_seen",
                "non_padding_tokens",
                "supervised_tokens",
                "verified_updates",
                "changed_adapter_tensors",
                "complete",
                "budget_limited",
                "memory",
            ),
        ),
        losses={},
        phases=[],
        limitations=summary["limitations"],
    )
    for variant, splits in summary["losses"].items():
        result["losses"][variant] = {
            split: select(
                values,
                ("complete", "loss", "examples", "aggregation", "per_example_losses"),
            )
            for split, values in splits.items()
            if split in ("test", "anchor", "validation")
        }
    for phase in summary["attempts"]:
        result["phases"].append(
            select(
                phase, ("phase", "exit_code", "seconds", "shutdown_reason", "forced")
            )
        )
    result["failures"] = sum(phase["exit_code"] != 0 for phase in summary["attempts"])
    write(output / "overnight_32k.json", result)
    with (run / "training_curves.csv").open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    with (run / "validation_curves.csv").open(newline="") as fp:
        validation = [r for r in csv.DictReader(fp) if r["complete"] == "True"]
    for name, data, fields in (
        (
            "overnight_training_curves.csv",
            rows,
            [
                "step",
                "loss",
                "grad_norm",
                "seconds",
                "examples_seen",
                "non_padding_tokens",
            ],
        ),
        ("overnight_validation_curves.csv", validation, ["step", "loss", "complete"]),
    ):
        with (output / name).open("w", newline="") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=fields, lineterminator="\n", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(data)
    plot(rows, validation, result, output)
    plot_tokens(rows, output)
    if include_responses:
        export_responses(run, output)
    sources = [
        "summary.json",
        "experiment.json",
        "training_curves.csv",
        "validation_curves.csv",
    ]
    write(
        output / "overnight_export_metadata.json",
        dict(
            exported_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            run_id=run.name,
            source_sha256={
                name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                for name in sources
            },
            description="Allowlisted metrics and plots; no prompts, responses, model weights, checkpoint paths, or absolute paths.",
            plotting="matplotlib; raw per-step loss, trailing 10-step mean, validation checkpoints, pre-clipping adapter gradient L2 norm; per-step non-padding tokens and running mean",
        ),
    )


def plot(rows, validation, result, output):
    selected_step = result["selected_checkpoint"]["step"]
    cfg = result["configuration"]
    hp = cfg["hyperparameters"]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(r["step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    gradients = [float(r["grad_norm"]) for r in rows]
    smoothed = [
        sum(losses[max(0, i - 9) : i + 1]) / len(losses[max(0, i - 9) : i + 1])
        for i in range(len(losses))
    ]
    validation_steps = [int(r["step"]) for r in validation]
    validation_losses = [float(r["loss"]) for r in validation]
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "overnight-32k",
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
        fig.suptitle(
            "Qwen3.8-27B · overnight 32K LoRA training",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        axes[0].plot(
            steps,
            losses,
            color="#2474b5",
            alpha=0.3,
            linewidth=0.8,
            label="Training loss · each step",
        )
        axes[0].plot(
            steps,
            smoothed,
            color="#2474b5",
            linewidth=2,
            label="Training loss · trailing 10-step mean",
        )
        axes[0].plot(
            validation_steps,
            validation_losses,
            color="#d25b20",
            marker="o",
            markersize=4,
            linewidth=1.8,
            label="Validation loss · 32 held-out groups",
        )
        axes[0].set(ylabel="Assistant-token cross entropy", title="Learning curve")
        axes[0].legend(loc="upper right", frameon=False, fontsize=9)
        axes[1].plot(
            steps,
            gradients,
            color="#287c59",
            linewidth=1.2,
            label="Adapter gradient L2 norm · before clipping",
        )
        axes[1].set(
            xlabel="Optimizer step",
            ylabel="Gradient L2 norm",
            title="Gradient magnitude",
        )
        axes[1].set_ylim(bottom=0)
        axes[1].legend(loc="upper right", frameon=False, fontsize=9)
        for ax in axes:
            ax.axvline(selected_step, color="#555555", linestyle="--", linewidth=1)
            ax.grid(alpha=0.18)
            ax.set_xlim(0, max(200, steps[-1] + 5))
        axes[1].annotate(
            f"Selected / final: step {selected_step}",
            xy=(selected_step, 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(-8, 6),
            textcoords="offset points",
            ha="right",
            color="#444444",
            fontsize=9,
        )
        fig.text(
            0.08,
            0.025,
            f"{result['training']['examples_seen']} training examples · accumulation {hp['accumulation']} · "
            f"{hp['scheduler']} LR {hp['learning_rate']:g} · seed {cfg['seed']}\n"
            f"Training batches vary by step; validation uses fixed groups. Gradient clipping threshold: {hp['max_grad_norm']:.1f}.",
            fontsize=9,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.085, 1, 0.95))
        fig.savefig(output / "overnight_learning_curves.png", dpi=180)
        fig.savefig(output / "overnight_learning_curves.svg", metadata={"Date": None})
        plt.close(fig)


def token_usage(rows):
    """Difference cumulative counters; each optimizer step includes all microbatches."""
    result = []
    previous_tokens, previous_examples, previous_step = 0, 0, 0
    for row in rows:
        step = int(row["step"])
        cumulative_tokens = int(row["non_padding_tokens"])
        cumulative_examples = int(row["examples_seen"])
        tokens = cumulative_tokens - previous_tokens
        examples = cumulative_examples - previous_examples
        if step != previous_step + 1 or tokens <= 0 or examples <= 0:
            raise ValueError(
                "Token usage requires consecutive steps and increasing counters"
            )
        result.append(
            dict(
                step=step,
                non_padding_tokens_this_step=tokens,
                examples_this_step=examples,
                mean_tokens_per_example=tokens / examples,
                running_mean_tokens_per_step=cumulative_tokens / step,
            )
        )
        previous_tokens, previous_examples, previous_step = (
            cumulative_tokens,
            cumulative_examples,
            step,
        )
    return result


def plot_tokens(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    usage = token_usage(rows)
    with (output / "overnight_tokens_per_step.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(usage[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(usage)
    steps = [r["step"] for r in usage]
    tokens = [r["non_padding_tokens_this_step"] for r in usage]
    running_mean = [r["running_mean_tokens_per_step"] for r in usage]
    smooth = [
        sum(tokens[max(0, i - 9) : i + 1]) / len(tokens[max(0, i - 9) : i + 1])
        for i in range(len(tokens))
    ]
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "overnight-tokens",
            "savefig.facecolor": "white",
        }
    ):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(
            steps,
            tokens,
            color="#2474b5",
            alpha=0.3,
            linewidth=0.9,
            label="Tokens processed each optimizer step",
        )
        ax.plot(
            steps, smooth, color="#2474b5", linewidth=1.8, label="Trailing 10-step mean"
        )
        ax.plot(
            steps,
            running_mean,
            color="#d25b20",
            linewidth=2,
            label="Running mean tokens per step",
        )
        ax.set(
            title="Qwen3.8-27B · non-padding training tokens per optimizer step",
            xlabel="Optimizer step",
            ylabel="Non-padding input tokens",
            xlim=(0, steps[-1] + 3),
        )
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=0.18)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        mean = sum(tokens) / len(tokens)
        examples = int(rows[-1]["examples_seen"])
        fig.text(
            0.09,
            0.025,
            f"Run average: {mean:,.1f} tokens/step · {sum(tokens) / examples:,.1f} tokens/example\n"
            f"{sum(tokens):,} non-padding tokens across {len(steps)} optimizer steps and {examples} examples. "
            "Includes prompt + assistant tokens; excludes padding and evaluation.",
            fontsize=8.5,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(output / "overnight_tokens_per_step.png", dpi=180)
        fig.savefig(output / "overnight_tokens_per_step.svg", metadata={"Date": None})
        plt.close(fig)


def export_responses(run, output):
    """Explicitly publish model outputs, keeping prompt transcripts out of the export."""
    pairs = read(run / "paired_responses.json")
    lines = [
        "# Overnight model response comparison",
        "",
        "Exact decoded outputs from the completed 32K overnight experiment. "
        "Each case uses the same prompt for all three models, greedy decoding, and a 2,048-token output budget.",
        "",
        "**Models:** base = original Qwen3.8-27B; pilot = existing 32-step adapter; "
        "selected = overnight adapter at checkpoint 190.",
        "",
        "The first four cases are held-out terminal-task contexts; the last four are controlled instruction probes. "
        "Full prompt transcripts are omitted. The outputs include the model's emitted reasoning and final response verbatim. "
        "Generated commands were never executed; schema validity does not establish task correctness.",
        "",
    ]
    for pair in pairs:
        lines.extend(
            [
                f"## {pair['case_id']}",
                "",
                f"Case type: {pair['kind']}. "
                f"Prompt SHA-256: `{pair['prompt_sha256']}`.",
                "",
            ]
        )
        for variant in ("base", "pilot", "selected"):
            record = pair["responses"][variant]
            assert record["prompt_sha256"] == pair["prompt_sha256"]
            response = record["response"]
            # A fence longer than any emitted backtick sequence preserves all markup verbatim.
            fence = "`" * max(
                3, 1 + max((len(m) for m in re.findall(r"`+", response)), default=0)
            )
            checks = (
                f"Tokens: {record['generated_tokens']}; JSON valid: {record['json_valid']}; "
                f"schema valid: {record['schema_valid']}; truncated: {record['token_cap_reached']}."
            )
            if "diagnostic_pass" in record:
                checks += f" Instruction probe passed: {record['diagnostic_pass']}."
            lines.extend(
                [f"### {variant}", "", checks, "", f"{fence}text", response, fence, ""]
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "overnight_paired_responses.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--include-responses",
        action="store_true",
        help="Explicitly publish full model outputs as a paired Markdown comparison",
    )
    args = parser.parse_args()
    export(args.run_dir, args.output, include_responses=args.include_responses)
    print(f"Exported overnight metrics and plots to {args.output}")


if __name__ == "__main__":
    main()
