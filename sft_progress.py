"""Local structured telemetry and terminal rendering; no model or network imports."""

import json
import math
import sys
import time
from collections import deque
from pathlib import Path


def emit(args, kind, **values):
    path = getattr(args, "event_file", None)
    if path is None:
        return
    sequence = getattr(args, "_event_sequence", 0) + 1
    args._event_sequence = sequence
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(args, "budget"):
        values.setdefault("elapsed_seconds", time.time() - args.budget["started_at"])
    record = dict(
        event_id=f"{path.stem}:{sequence}",
        session_id=getattr(args, "session_id", ""),
        kind=kind,
        phase=values.pop("_phase", getattr(args, "phase", "all")),
        timestamp=time.time(),
        values=values,
    )
    # One producer owns each journal. Readers consume only complete newline-terminated records.
    with path.open("a") as fp:
        fp.write(json.dumps(record, allow_nan=False) + "\n")
        fp.flush()


class EventReader:
    """Bounded reads of regular files cannot block waiting for a model or stdout."""

    def __init__(self):
        self.offsets = {}
        self.pending = {}

    def poll(self, paths, limit=65536):
        result = []
        remaining = limit
        for path in sorted(map(Path, paths)):
            if remaining <= 0:
                break
            if not path.exists():
                continue
            offset = self.offsets.get(str(path), 0)
            with path.open("rb") as fp:
                fp.seek(offset)
                chunk = fp.read(remaining)
                self.offsets[str(path)] = fp.tell()
            remaining -= len(chunk)
            data = self.pending.pop(str(path), b"") + chunk
            lines = data.split(b"\n")
            tail = lines.pop()
            if len(tail) <= 131072:
                self.pending[str(path)] = tail
            for line in lines:
                try:
                    event = json.loads(line)
                    if (
                        isinstance(event, dict)
                        and {"event_id", "kind", "values", "timestamp"} <= event.keys()
                        and isinstance(event["values"], dict)
                        and isinstance(event["event_id"], str)
                        and isinstance(event["kind"], str)
                        and isinstance(event["timestamp"], (int, float))
                        and math.isfinite(event["timestamp"])
                    ):
                        result.append(event)
                except (ValueError, UnicodeDecodeError):
                    continue
        return sorted(result, key=lambda event: event["timestamp"])


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Renderer:
    def __init__(self, args, stream=None):
        self.args = args
        self.stream = stream if stream is not None else sys.stdout
        mode = getattr(args, "progress", "auto")
        self.enabled = mode != "off"
        self.interactive = self.enabled and mode == "auto" and self.stream.isatty()
        self.bar = None
        self.bar_key = None
        self.last_update = time.time()
        self.activity = "starting"
        self.durations = deque(maxlen=10)

    def line(self, message):
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.write(message, file=self.stream)
        else:
            print(message, file=self.stream, flush=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None
            self.bar_key = None

    def counter(self, key, count, total, description, postfix=""):
        if not self.interactive:
            self.line(
                f"SFT {description} {count}/{total}"
                + (f" {postfix}" if postfix else "")
            )
            return
        from tqdm import tqdm

        if key != self.bar_key:
            self.close()
            self.bar = tqdm(
                total=total,
                initial=count,
                desc=f"SFT {description}",
                file=self.stream,
                dynamic_ncols=True,
                mininterval=0.2,
            )
            self.bar_key = key
        else:
            self.bar.update(count - self.bar.n)
        self.bar.set_postfix_str(postfix, refresh=True)

    def event(self, event):
        try:
            self._event(event)
        except (KeyError, TypeError, ValueError):
            self.durations.clear()
            self.line(
                "SFT WARNING: ignored malformed progress event; worker supervision continues."
            )

    def _event(self, event):
        self.last_update = event["timestamp"]
        value = event["values"]
        kind = event["kind"]
        phase = event.get("phase", "")
        if kind == "operation":
            self.activity = value.get("operation", phase)
            if (
                self.activity.startswith("optimizer step")
                and self.interactive
                and "target_steps" in value
            ):
                self.counter(
                    "train",
                    value["step"],
                    value["target_steps"],
                    "TRAIN",
                    "computing optimizer step",
                )
            elif not self.activity.startswith("optimizer step"):
                self.line(f"SFT {phase.upper()} operation={self.activity}")
        elif kind == "train_step":
            self.activity = f"train step {value['step']}"
            self.durations.append(value["seconds"])
            total = value["target_steps"]
            remaining = (
                (total - value["step"]) * sum(self.durations) / len(self.durations)
            )
            budget = self.args.budget
            metrics = (
                f"({100 * value['step'] / total:.1f}%) epoch={value['epoch']:.3f} "
                f"loss={value['loss']:.5f} grad_norm={value['grad_norm']:.5f} "
                f"lr={value['learning_rate']:g} step={value['seconds']:.1f}s "
                f"tokens/s={value['tokens_per_second']:.1f} tokens={value['non_padding_tokens']:,}"
            )
            mem = value.get("memory", {})
            metrics += (
                f" allocated={mem.get('allocated_bytes', 0) / 1024**3:.1f}GiB"
                f" reserved={mem.get('reserved_bytes', 0) / 1024**3:.1f}GiB"
                f" elapsed={duration(event['timestamp'] - budget['started_at'])}"
                f" target_eta~{duration(remaining)}"
                f" training_budget_left={duration(budget['training_deadline'] - event['timestamp'])}"
            )
            self.counter("train", value["step"], total, "TRAIN", metrics)
        elif kind == "evaluation_progress":
            self.activity = f"evaluation {value['variant']} {value['split']}"
            label = f"EVAL {value['variant']} {value['split']} checkpoint={value.get('checkpoint_step') or '-'}"
            self.counter(
                (phase, label),
                value["evaluated_examples"],
                value["expected_examples"],
                label,
                f"running_loss={value['running_loss']:.5f}",
            )
        elif kind == "evaluation_complete":
            self.line(
                f"SFT EVAL COMPLETE model={value['variant']} split={value['split']} "
                f"checkpoint={value.get('checkpoint_step') or '-'} loss={value['loss']:.6f}"
            )
        elif kind == "generation_progress":
            self.activity = f"generation {value['variant']}"
            self.counter(
                (phase, "generation"),
                value["completed_responses"],
                value["expected_responses"],
                f"GENERATE {value['variant']}",
                f"tokens={value.get('generated_tokens', 0)}",
            )
        elif kind == "prepare_progress":
            self.counter(
                "prepare",
                value["selected_groups"],
                value["expected_groups"],
                "PREPARE groups",
            )
        elif kind == "checkpoint":
            self.line(f"SFT CHECKPOINT saved step={value['step']} resumable=yes")
        elif kind == "phase_start":
            self.close()
            self.activity = phase
            self.line(f"SFT PHASE START {phase} log={value.get('log', '-')}")
        elif kind == "phase_end":
            self.close()
            self.line(
                f"SFT PHASE END {phase} exit_code={value['exit_code']} duration={duration(value['seconds'])}"
            )

    def heartbeat(self, phase, now=None):
        now = time.time() if now is None else now
        self.line(
            f"SFT HEARTBEAT phase={phase} operation={self.activity} "
            f"last_update={max(0, now - self.last_update):.0f}s ago "
            f"budget_left={duration(self.args.budget['deadline'] - now)}"
        )


# This allowlist is shared by real tracking and tests; raw events never reach W&B.
def metric_payload(event):
    values = event["values"]
    kind = event["kind"]
    elapsed = values.get("elapsed_seconds", 0)
    payload = {
        "runtime/elapsed_seconds": elapsed
        if isinstance(elapsed, (float, int)) and math.isfinite(elapsed)
        else 0
    }
    numeric = {
        "train_step": (
            "train",
            (
                "loss",
                "grad_norm",
                "learning_rate",
                "epoch",
                "step",
                "seconds",
                "examples_seen",
                "non_padding_tokens",
                "supervised_tokens",
                "tokens_this_step",
                "mean_tokens_per_example",
                "tokens_per_second",
            ),
        ),
        "evaluation_progress": (
            "evaluation",
            ("evaluated_examples", "expected_examples", "running_loss"),
        ),
        "generation_progress": (
            "generation",
            ("completed_responses", "expected_responses", "generated_tokens"),
        ),
        "heartbeat": ("runtime", ("remaining_seconds", "training_remaining_seconds")),
        "checkpoint": ("checkpoint", ("step",)),
        "phase_end": ("phase", ("seconds", "exit_code")),
    }
    if kind in numeric:
        prefix, fields = numeric[kind]
        for field in fields:
            value = values.get(field)
            if isinstance(value, (float, int)) and math.isfinite(value):
                payload[f"{prefix}/{field}"] = value
    if kind == "evaluation_complete":
        variant, split = values.get("variant"), values.get("split")
        if variant in ("base", "pilot", "selected", "training") and split in (
            "validation",
            "test",
            "anchor",
        ):
            loss = values.get("loss")
            if isinstance(loss, (int, float)) and math.isfinite(loss):
                payload[f"eval/{variant}/{split}/loss"] = loss
            if variant == "training" and isinstance(values.get("checkpoint_step"), int):
                payload["train/step"] = values["checkpoint_step"]
    for name in (
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ):
        value = values.get("memory", {}).get(name)
        if isinstance(value, (int, float)) and math.isfinite(value):
            payload[f"memory/{name}"] = value
    phase = event.get("phase")
    if phase in (
        "preflight",
        "prepare",
        "probe",
        "train",
        "validate",
        "all",
        "finished",
    ) or phase in {
        f"{p}-{v}" for p in ("loss", "generate") for v in ("base", "pilot", "selected")
    }:
        payload["runtime/phase"] = phase
    return payload
