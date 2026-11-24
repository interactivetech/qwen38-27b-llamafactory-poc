"""Isolated, best-effort W&B aggregate reporter. No task text or trajectories are uploaded."""

import os
import sys
import time
from pathlib import Path

from sft_progress import EventReader
from sft_tracking import choose_mode, read, write


def metrics(summary):
    result = {
        "coverage/scored": summary["scored_trials"],
        "coverage/planned": 10,
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    for model, values in summary["models"].items():
        for field in ("solved", "scored", "planned"):
            result[f"{model}/{field}"] = values[field]
    for field in ("base_wins", "sft_wins", "ties"):
        result[f"paired/{field}"] = summary["paired"][field]
    for model in ("base", "sft"):
        trials = [t for t in summary["trials"] if t["model"] == model]
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "agent_turns",
            "json_parse_failures",
            "inference_seconds",
            "terminal_seconds",
            "verifier_seconds",
            "truncated_completions",
        ):
            result[f"{model}/{field}"] = sum(t.get(field) or 0 for t in trials)
    return result


def run(directory, requested):
    import wandb

    directory = Path(directory)
    tracking = directory / "tracking"
    tracking.mkdir(exist_ok=True)
    mode = choose_mode(requested)
    session = tracking / str(os.getpid())
    session.mkdir()
    # Configuration is fixed and allowlisted; never serialize arbitrary args or provenance.
    config = dict(
        harbor_commit="39c2219a41b3caa7a689cef1882263b5a93ae93f",
        vllm="0.34.0",
        context=65536,
        max_output=32768,
        checkpoint_step=190,
        seed=42,
        planned_trials=10,
    )
    write(
        tracking / "status.json",
        dict(state="starting", heartbeat=time.time(), mode=mode),
    )
    sdk = wandb.init(
        project="qwen38-terminalbench21-pilot",
        mode=mode,
        dir=str(session),
        config=config,
        settings=wandb.Settings(
            init_timeout=10,
            finish_timeout=3,
            console="off",
            disable_code=True,
            disable_git=True,
            disable_job_creation=True,
            x_disable_stats=True,
            x_disable_meta=True,
        ),
    )
    reader = EventReader()
    seen = set()
    try:
        while True:
            for event in reader.poll((directory / "events").glob("*.jsonl")):
                if event["event_id"] in seen:
                    continue
                seen.add(event["event_id"])
                # Only numeric metrics from the local event journal.
                values = {
                    f"progress/{k}": v
                    for k, v in event["values"].items()
                    if k in ("completed_trials", "elapsed_seconds", "remaining_seconds")
                    and isinstance(v, (int, float))
                }
                if values:
                    sdk.log(values)
            summary = read(directory / "summary.json")
            if summary:
                sdk.log(metrics(summary))
            write(
                tracking / "status.json",
                dict(state="running", heartbeat=time.time(), mode=mode),
            )
            if (tracking / "DONE").exists():
                if summary:
                    sdk.summary.update(metrics(summary))
                    sdk.summary["status"] = summary["status"]
                break
            time.sleep(3)
    finally:
        sdk.finish()
        write(
            tracking / "status.json",
            dict(
                state="finished",
                heartbeat=time.time(),
                mode=mode,
                sync_pending=mode == "offline",
            ),
        )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
