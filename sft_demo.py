"""CPU-only integration demonstration of SFT progress, supervision, and W&B."""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import sft_train as sft
from sft_progress import EventReader, Renderer, emit
from sft_tracking import TrackingController


def worker(args):
    import torch

    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(1, 4):
        start = time.monotonic()
        optimizer.zero_grad()
        prediction = model(torch.ones(1, 2))
        loss = (prediction - 1).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        time.sleep(0.4)
        emit(
            args,
            "train_step",
            step=step,
            target_steps=3,
            epoch=step / 3,
            loss=loss.item(),
            grad_norm=norm.item(),
            learning_rate=1e-4,
            seconds=time.monotonic() - start,
            tokens_per_second=16 / (time.monotonic() - start),
            tokens_this_step=16,
            mean_tokens_per_example=8,
            non_padding_tokens=step * 16,
            supervised_tokens=step * 8,
            examples_seen=step * 2,
            memory={},
        )
    emit(
        args,
        "evaluation_progress",
        variant="training",
        split="validation",
        checkpoint_step=3,
        evaluated_examples=1,
        expected_examples=1,
        running_loss=0.5,
    )
    emit(
        args,
        "evaluation_complete",
        variant="training",
        split="validation",
        checkpoint_step=3,
        loss=0.5,
    )
    emit(
        args,
        "generation_progress",
        variant="selected",
        completed_responses=1,
        expected_responses=1,
        generated_tokens=8,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="./setup.sh --run sft-demo", description=__doc__
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("SMOKE_ROOT", sft.smoke.DEFAULT_ROOT)),
    )
    parser.add_argument("--progress", choices=("auto", "lines", "off"), default="auto")
    parser.add_argument(
        "--wandb-mode",
        choices=("auto", "offline", "online", "disabled"),
        default="auto",
    )
    parser.add_argument("--phase", default="demo")
    parser.add_argument("--event-file", type=Path)
    parser.add_argument("--session-id")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    args.root = args.root.resolve()
    now = time.time()
    args.budget = dict(started_at=now, training_deadline=now + 45, deadline=now + 90)
    if args.phase == "worker":
        worker(args)
        return 0
    if args.run_dir.exists():
        raise ValueError("Use a new demo directory")
    args.run_dir.mkdir(parents=True)
    args.session_id = uuid.uuid4().hex
    args.event_file = args.run_dir / "telemetry" / args.session_id / "supervisor.jsonl"
    args._renderer, args._event_reader = Renderer(args), EventReader()
    args._tracker = TrackingController(args)
    args._tracker.ensure_dependencies()
    sft.smoke.write_json(
        args.run_dir / "experiment.json",
        dict(
            identity=dict(
                seed=42,
                seq_len=16,
                hyperparameters=dict(learning_rate=1e-4, optimizer="AdamW"),
            )
        ),
    )
    args._tracker.start()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(args.run_dir),
        "--phase",
        "worker",
    ]
    try:
        result = sft.supervise_worker(args, "train", command=command)
        # Allow real offline SDK initialization to complete before the bounded final flush.
        ready_by = time.monotonic() + 12
        while (
            args._tracker.status.get("state") == "starting"
            and time.monotonic() < ready_by
        ):
            args._tracker.safe_poll()
            time.sleep(0.1)
        emit(
            args,
            "final",
            status="complete" if result["exit_code"] == 0 else "failed",
            training_verified=False,
            quality={},
            losses={},
            comparisons={},
            selected_checkpoint=None,
            latest_checkpoint=None,
        )
        args._tracker.safe_finish()
        sft.smoke.write_json(
            args.run_dir / "status.json",
            dict(
                status="complete" if result["exit_code"] == 0 else "failed",
                phase="finished",
                worker=result,
                tracking=args._tracker.status,
                demonstration=True,
            ),
        )
        print(
            "SFT DEMO COMPLETE (CPU only; no model training experiment performed)",
            flush=True,
        )
        print(f"Tracking: {args._tracker.status}", flush=True)
        return int(result["exit_code"] != 0)
    finally:
        args._tracker.terminate()
        args._renderer.close()


if __name__ == "__main__":
    sys.exit(main())
