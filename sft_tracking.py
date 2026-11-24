"""Isolated W&B reporter and local journal recovery; never imports a model library."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from sft_progress import EventReader, metric_payload


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def choose_mode(requested, environ=None):
    environ = os.environ if environ is None else environ
    return (
        ("online" if environ.get("WANDB_API_KEY", "").strip() else "offline")
        if requested == "auto"
        else requested
    )


def safe_config(identity):
    result = {
        k: identity[k]
        for k in (
            "seq_len",
            "seed",
            "train_examples",
            "val_examples",
            "test_examples",
            "hours",
            "dependency_lock_sha256",
        )
        if k in identity
    }
    if "revisions" in identity:
        result["revisions"] = {
            k: identity["revisions"][k]
            for k in ("model_revision", "dataset_revision")
            if k in identity["revisions"]
        }
    if "hyperparameters" in identity:
        allowed = (
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "bf16",
            "attention",
            "gradient_checkpointing",
            "microbatch",
            "accumulation",
            "optimizer",
            "weight_decay",
            "max_grad_norm",
            "learning_rate",
            "scheduler",
            "max_steps",
            "save_steps",
            "eval_steps",
            "save_total_limit",
        )
        result["hyperparameters"] = {
            k: identity["hyperparameters"][k]
            for k in allowed
            if k in identity["hyperparameters"]
        }
    return result


def summary_payload(summary):
    result = {}
    for field in ("training_verified", "promising_scaling_result", "elapsed_seconds"):
        if field in summary:
            result[f"final/{field}"] = summary[field]
    for model in ("base", "pilot", "selected"):
        for split in ("validation", "test", "anchor"):
            record = summary.get("losses", {}).get(model, {}).get(split, {})
            if record.get("complete"):
                result[f"final/{model}/{split}/loss"] = record["loss"]
        for key in (
            "completed",
            "json_valid",
            "schema_valid",
            "truncated",
            "instruction_pass",
            "instruction_total",
        ):
            value = summary.get("quality", {}).get(model, {}).get(key)
            if isinstance(value, (int, float)):
                result[f"final/quality/{model}/{key}"] = value
    for split in ("test", "anchor"):
        for comparison in ("base_to_selected", "pilot_to_selected", "base_to_pilot"):
            values = summary.get("comparisons", {}).get(split, {}).get(comparison, {})
            for key in ("relative_improvement", "mean_loss_reduction"):
                if key in values:
                    result[f"final/{split}/{comparison}/{key}"] = values[key]
            for key in ("relative_95_interval", "loss_reduction_95_interval"):
                for label, value in zip(("low", "high"), values.get(key, [])):
                    result[f"final/{split}/{comparison}/{key}_{label}"] = value
    for kind in ("selected_checkpoint", "latest_checkpoint"):
        value = summary.get(kind) or {}
        if "step" in value:
            result[f"final/{kind}/step"] = value["step"]
    if summary.get("status") in ("complete", "partial", "failed"):
        result["final/status"] = summary["status"]
    return result


class TrackingController:
    """Only local filesystem/process polling runs in the model-worker supervisor."""

    def __init__(self, args, printer=print):
        self.args = args
        self.printer = printer
        self.requested = getattr(args, "wandb_mode", "disabled")
        self.mode = choose_mode(self.requested)
        self.process = None
        self.log_handle = None
        self.started = None
        self.warned = False
        self.fallback = False
        self.base = args.run_dir / "tracking"
        self.session = self.base / "sessions" / args.session_id
        self.status_path = self.session / "status.json"
        self.config_path = self.session / "config.json"
        self.python = args.root / "tracking-venv/bin/python"
        self.status = dict(
            requested_mode=self.requested,
            effective_mode=self.mode,
            state="starting",
            session_id=args.session_id,
            sync_pending=False,
        )

    def ensure_dependencies(self):
        if self.mode == "disabled":
            return
        ready = read(self.args.root / "tracking-venv/ready.json", {})
        lock = Path(__file__).with_name("requirements-tracking.lock")
        if (
            not self.python.is_file()
            or ready.get("lock_sha256") != hashlib.sha256(lock.read_bytes()).hexdigest()
        ):
            raise RuntimeError(
                "W&B tracking environment missing/outdated. Run ./setup.sh --setup-tracking, "
                "or launch with --wandb-mode disabled."
            )

    def publish(self):
        write(self.session / "controller.json", self.status)
        write(self.base / "status.json", self.status)

    def start(self):
        self.session.mkdir(parents=True, exist_ok=True)
        experiment = read(self.base / "experiment.json")
        if experiment is None:
            experiment = {"experiment_id": uuid.uuid4().hex}
            write(self.base / "experiment.json", experiment)
        identity = read(self.args.run_dir / "experiment.json", {}).get("identity", {})
        config = dict(
            mode=self.mode,
            project=getattr(self.args, "wandb_project", None)
            or os.environ.get("WANDB_PROJECT")
            or "qwen38-27b-llamafactory-poc",
            entity=getattr(self.args, "wandb_entity", None)
            or os.environ.get("WANDB_ENTITY")
            or None,
            name=getattr(self.args, "wandb_name", None)
            or os.environ.get("WANDB_NAME")
            or self.args.run_dir.name,
            experiment_id=experiment["experiment_id"],
            session_id=self.args.session_id,
            event_dir=str(self.args.event_file.parent),
            config=safe_config(identity),
            restored_checkpoint_step=getattr(self.args, "restored_step", 0),
        )
        write(self.config_path, config)
        if self.mode == "disabled":
            self.status.update(state="disabled", sync_pending=False)
            self.publish()
            return
        try:
            self.launch()
        except OSError:
            self.status.update(state="pending_recovery", sync_pending=True)
            self.publish()
            self.printer(
                "SFT WARNING: reporter failed to start; metrics retained locally for sync.",
                flush=True,
            )

    def launch(self):
        config = read(self.config_path)
        config["mode"] = self.mode
        write(self.config_path, config)
        self.status_path.unlink(missing_ok=True)
        self.log_handle = (self.session / f"reporter-{self.mode}.log").open("a")
        env = dict(os.environ)
        env.pop("WANDB_DISABLED", None)
        env["WANDB_MODE"] = self.mode
        env["WANDB_CONSOLE"] = "off"
        env["WANDB_SAVE_CODE"] = "false"
        # In offline mode, do not pass credentials to the SDK at all.
        if self.mode == "offline":
            env.pop("WANDB_API_KEY", None)
        self.process = subprocess.Popen(
            [
                str(self.python),
                str(Path(__file__).resolve()),
                "report",
                "--session",
                str(self.session),
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started = time.monotonic()
        self.status.update(
            effective_mode=self.mode,
            state="starting",
            sync_pending=True,
            local_directory=str(self.session),
        )
        self.publish()
        self.printer(f"SFT TRACKING mode={self.mode} local={self.session}", flush=True)

    def terminate(self, timeout=0.1):
        if self.process is not None:
            # Kill this reporter's whole group, including the SDK background service.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=max(0, timeout))
            except subprocess.TimeoutExpired:
                pass
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def safe_poll(self):
        try:
            self.poll()
        except Exception as exc:
            self.printer(
                f"SFT WARNING: reporter error ({type(exc).__name__}); metrics retained locally.",
                flush=True,
            )
            try:
                self.terminate()
            except Exception:
                pass
            self.process = None
            self.status.update(state="pending_recovery", sync_pending=True)
            try:
                self.publish()
            except OSError:
                pass

    def poll(self):
        if self.process is None:
            return
        state = read(self.status_path, {})
        if state.get("state") in ("ready", "finished"):
            if self.status.get("state") == "starting":
                self.printer(
                    f"SFT TRACKING {self.mode} ready "
                    + (state.get("url") or f"local={self.session}"),
                    flush=True,
                )
            self.status.update(state)
            self.status["effective_mode"] = self.mode
            self.publish()
        timeout = (
            time.monotonic() - self.started > 15
            and self.status.get("state") == "starting"
        )
        stalled = state.get("heartbeat_at") and time.time() - state["heartbeat_at"] > 30
        exited = self.process.poll() is not None
        if timeout or stalled or (exited and state.get("state") != "finished"):
            if self.mode == "online" and not self.fallback:
                self.printer(
                    "SFT WARNING: online W&B unavailable; continuing with local offline tracking. "
                    "Saved events will be replayed.",
                    flush=True,
                )
                self.terminate()
                self.mode, self.fallback = "offline", True
                self.status.update(
                    fallback_reason="online reporter unavailable", url=None
                )
                self.launch()
                return
            if not self.warned:
                self.printer(
                    f"SFT WARNING: tracking reporter unavailable; training continues. "
                    f"Local events retained. Sync with ./setup.sh --tracking sync --run-dir {self.args.run_dir}",
                    flush=True,
                )
                self.warned = True
            self.terminate()
            self.process = None
            self.status.update(state="pending_recovery", sync_pending=True)
            self.publish()

    def safe_finish(self, timeout=5):
        try:
            self.finish(timeout)
        except Exception as exc:
            self.printer(
                f"SFT WARNING: tracking flush incomplete ({type(exc).__name__}); local data retained.",
                flush=True,
            )
            self.status.update(state="pending_recovery", sync_pending=True)
            try:
                self.terminate(timeout=0)
            except Exception:
                pass
            try:
                self.publish()
            except OSError:
                pass

    def finish(self, timeout=5):
        if self.process is None:
            self.publish()
            return
        (self.session / "DONE").touch()
        cutoff = time.monotonic() + max(0, min(5, timeout))
        flush_cutoff = max(time.monotonic(), cutoff - 0.1)
        while self.process.poll() is None and time.monotonic() < flush_cutoff:
            time.sleep(0.05)
        state = read(self.status_path, {})
        code = self.process.poll()
        if code == 0 and state.get("state") == "finished":
            self.status.update(state)
        else:
            self.status.update(state="pending_recovery", sync_pending=True)
        self.terminate(timeout=max(0, cutoff - time.monotonic()))
        self.process = None
        self.publish()


def report_session(session):
    """Executed with the isolated tracking Python, never the training interpreter."""
    import wandb

    config = read(session / "config.json")
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_SAVE_CODE"] = "false"
    os.environ["WANDB_MODE"] = config["mode"]
    if config["mode"] == "offline":
        os.environ.pop("WANDB_API_KEY", None)
    state_path = session / "status.json"
    mode = config["mode"]
    # Offline SDK runs cannot be resumed. Each reporter invocation is a segment,
    # grouped under the same experiment/session. A recovery replays into a fresh segment.
    segment_id = uuid.uuid4().hex[:16]
    segment = session / "segments" / segment_id
    segment.mkdir(parents=True)
    settings = wandb.Settings(
        init_timeout=10,
        finish_timeout=3,
        console="off",
        disable_code=True,
        disable_git=True,
        disable_job_creation=True,
        x_disable_stats=True,
        x_disable_meta=True,
    )
    run = wandb.init(
        project=config["project"],
        entity=config["entity"],
        name=config["name"],
        group=config["experiment_id"],
        id=segment_id,
        mode=mode,
        dir=str(segment),
        config=dict(
            config["config"],
            tracking_session=config["session_id"],
            restored_checkpoint_step=config["restored_checkpoint_step"],
        ),
        settings=settings,
    )
    write(
        segment / "segment.json",
        dict(
            run_id=segment_id, mode=mode, project=config["project"], entity=run.entity
        ),
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("eval/training/validation/*", step_metric="train/step")
    run.define_metric("runtime/elapsed_seconds")
    for prefix in ("memory/*", "evaluation/*", "generation/*", "phase/*"):
        run.define_metric(prefix, step_metric="runtime/elapsed_seconds")
    state = dict(
        state="ready",
        effective_mode=mode,
        sync_pending=mode == "offline",
        url=run.url if mode == "online" else None,
        heartbeat_at=time.time(),
        segment_id=segment_id,
    )
    write(state_path, state)
    reader = EventReader()
    seen = set()
    count = 0
    last_sync_check = time.monotonic()
    # Persist consumed IDs/offsets for diagnosis. Recovered segments replay journals from
    # the beginning, rather than claim SDK resume support for an offline file.
    while True:
        events = reader.poll(Path(config["event_dir"]).glob("*.jsonl"))
        for event in events:
            if event["event_id"] in seen:
                continue
            seen.add(event["event_id"])
            if event["kind"] == "final":
                payload = summary_payload(event["values"])
                run.summary.update(payload)
            else:
                payload = metric_payload(event)
            run.log(payload, step=count)
            count += 1
        if mode == "online" and time.monotonic() - last_sync_check >= 5:
            sync_state = run.status()
            state["pending_uploads"] = sync_state.sync_items_pending
            state["sync_pending"] = sync_state.sync_items_pending > 0
            last_sync_check = time.monotonic()
        state.update(heartbeat_at=time.time(), consumed_events=count)
        write(state_path, state)
        write(
            segment / "cursor.json",
            dict(offsets=reader.offsets, event_ids=sorted(seen), sdk_step=count),
        )
        if (session / "DONE").exists() and not events:
            break
        time.sleep(0.2)
    run.finish()
    state.update(state="finished", heartbeat_at=time.time())
    write(state_path, state)


def sync_run(run_dir):
    with (run_dir / ".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Refusing sync while this run's supervisor is active")
        return _sync_run(run_dir)


def _sync_run(run_dir):
    """Sync SDK offline files; reconstruct missing reporter data from journals first."""
    import wandb  # Ensure the tracking interpreter was selected.

    del wandb
    failures = 0
    for session in sorted((run_dir / "tracking/sessions").glob("*")):
        controller = read(session / "controller.json", {})
        if controller.get("state") == "disabled":
            continue
        if not (session / "config.json").exists():
            continue
        if (
            controller.get("state") in ("starting", "ready")
            and not (session / "DONE").exists()
        ):
            raise RuntimeError(
                f"Refusing sync of an active tracking session: {session.name}"
            )
        before_failures = failures
        if controller.get("state") == "pending_recovery":
            previous_segments = list((session / "segments").glob("*/segment.json"))
            config = read(session / "config.json")
            config["mode"] = "offline"
            write(session / "config.json", config)
            (session / "DONE").touch()
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "report",
                    "--session",
                    str(session),
                ],
                check=True,
                timeout=60,
                env={**os.environ, "WANDB_MODE": "offline"},
            )
            for marker in previous_segments:
                write(marker.parent / "superseded.json", {"superseded_at": time.time()})
        for marker in sorted((session / "segments").glob("*/segment.json")):
            segment = marker.parent
            if (segment / "synced.json").exists() or (
                segment / "superseded.json"
            ).exists():
                continue
            offline_dirs = sorted(segment.glob("wandb/offline-run-*"))
            if not offline_dirs:
                continue
            command = [str(Path(sys.executable).with_name("wandb")), "sync"]
            for directory in offline_dirs:
                environment = dict(os.environ)
                environment.pop("WANDB_DISABLED", None)
                environment["WANDB_MODE"] = "online"
                result = subprocess.run(
                    command + [str(directory)], timeout=60, env=environment
                )
                if result.returncode:
                    failures += 1
                else:
                    write(segment / "synced.json", dict(synced_at=time.time()))
        if failures == before_failures:
            controller.update(sync_pending=False, state="synced")
            write(session / "controller.json", controller)
            latest = read(run_dir / "tracking/status.json", {})
            if latest.get("session_id") == session.name:
                write(run_dir / "tracking/status.json", controller)
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="./setup.sh --tracking", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report").add_argument("--session", type=Path, required=True)
    sync = sub.add_parser("sync")
    sync.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "report":
        report_session(args.session)
        return 0
    root = Path(os.environ.get("SMOKE_ROOT", smoke_root()))
    run = args.run_dir or latest_run(root)
    return sync_run(run.resolve())


def smoke_root():
    import getpass

    return f"/scratch1/{getpass.getuser()}/qwen38-27b-llamafactory-test"


def latest_run(root):
    for name in ("latest_sft_run.txt", "latest_overnight_run.txt"):
        path = root / name
        if path.exists() and path.read_text().strip():
            return Path(path.read_text().strip())
    raise ValueError("No SFT run pointer found; supply --run-dir")


if __name__ == "__main__":
    sys.exit(main())
