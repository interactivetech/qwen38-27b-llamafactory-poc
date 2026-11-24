"""CPU-only tests for terminal progress, isolated tracking, and compatibility."""

import contextlib
import fcntl
import io
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sft_progress as progress
import sft_tracking as tracking
import sft_train as sft
import smoke_test as smoke
from export_sft_results import PrefixedOutput


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        run = root / "runs/sft-test"
        run.mkdir(parents=True)
        now = time.time()
        self.args = SimpleNamespace(
            root=root,
            run_dir=run,
            event_file=run / "telemetry/session/supervisor.jsonl",
            session_id="session",
            phase="train",
            progress="auto",
            wandb_mode="disabled",
            resume=False,
            budget=dict(started_at=now, training_deadline=now + 90, deadline=now + 180),
        )
        self.stream = io.StringIO()
        self.renderer = progress.Renderer(self.args, self.stream)

    def event(self, kind="train_step", **values):
        defaults = dict(
            step=1,
            target_steps=200,
            loss=0.9,
            grad_norm=0.2,
            epoch=0.005,
            learning_rate=1e-4,
            seconds=2.0,
            tokens_per_second=30000.0,
            non_padding_tokens=60000,
            memory=dict(allocated_bytes=1024**3, reserved_bytes=2 * 1024**3),
        )
        if kind != "train_step":
            defaults = {}
        defaults.update(values)
        return dict(
            event_id="producer:1",
            kind=kind,
            phase="train",
            timestamp=time.time(),
            values=defaults,
        )

    def test_redirected_auto_prints_per_step_metrics_and_separate_eta(self):
        self.renderer.event(self.event())
        text = self.stream.getvalue()
        for field in (
            "TRAIN 1/200",
            "loss=0.90000",
            "grad_norm=0.20000",
            "epoch=",
            "lr=0.0001",
            "tokens/s=",
            "target_eta~",
            "training_budget_left=",
            "allocated=1.0GiB",
            "reserved=2.0GiB",
        ):
            self.assertIn(field, text)
        self.assertNotIn("\r", text)

    def test_interactive_auto_uses_updating_bar_and_lines_override(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        renderer = progress.Renderer(self.args, stream)
        self.assertTrue(renderer.interactive)
        with patch("tqdm.tqdm") as factory:
            bar = factory.return_value
            renderer.event(self.event())
            renderer.close()
            factory.assert_called_once()
            bar.set_postfix_str.assert_called_once()
            bar.close.assert_called_once()
        self.args.progress = "lines"
        self.assertFalse(progress.Renderer(self.args, stream).interactive)

    def test_malformed_progress_does_not_interrupt_supervision(self):
        self.renderer.event(self.event("train_step", loss="invalid metric"))
        self.renderer.event(self.event())
        self.assertIn("ignored malformed progress event", self.stream.getvalue())
        self.assertIn("TRAIN 1/200", self.stream.getvalue())

    def test_progress_off_suppresses_rendering(self):
        self.args.progress = "off"
        renderer = progress.Renderer(self.args, self.stream)
        renderer.event(self.event())
        renderer.heartbeat("train")
        self.assertEqual(self.stream.getvalue(), "")

    def test_eval_generation_checkpoint_and_phase_messages(self):
        for event in [
            self.event("phase_start", log="phase.log"),
            self.event(
                "evaluation_progress",
                variant="training",
                split="validation",
                checkpoint_step=16,
                evaluated_examples=12,
                expected_examples=32,
                running_loss=0.99,
            ),
            self.event(
                "evaluation_complete",
                variant="training",
                split="validation",
                checkpoint_step=16,
                loss=0.98,
            ),
            self.event(
                "generation_progress",
                variant="selected",
                completed_responses=2,
                expected_responses=8,
                generated_tokens=400,
            ),
            self.event("checkpoint", step=16),
            self.event("phase_end", seconds=10, exit_code=0),
        ]:
            self.renderer.event(event)
        text = self.stream.getvalue()
        for field in (
            "PHASE START",
            "12/32",
            "EVAL COMPLETE",
            "loss=0.980000",
            "GENERATE selected 2/8",
            "CHECKPOINT saved step=16 resumable=yes",
            "PHASE END",
        ):
            self.assertIn(field, text)

    def test_activity_change_clears_stale_training_metrics(self):
        with patch.object(sft, "memory", return_value={}), patch("builtins.print"):
            sft.progress(
                self.args, activity="training", loss=0.9, step=16, target_steps=200
            )
            sft.progress(
                self.args,
                activity="evaluation",
                variant="pilot",
                split="anchor",
                evaluated_examples=1,
                expected_examples=16,
                running_loss=1.0,
            )
        state = smoke.read_json(self.args.run_dir / "progress.json")
        self.assertNotIn("loss", state)
        self.assertNotIn("step", state)
        self.assertEqual(state["variant"], "pilot")

    def test_journal_partial_lines_dedup_offsets_and_bounded_reads(self):
        progress.emit(self.args, "checkpoint", step=8)
        reader = progress.EventReader()
        path = self.args.event_file
        data = path.read_bytes()
        path.write_bytes(data[:-1])
        self.assertEqual(reader.poll([path]), [])
        with path.open("ab") as fp:
            fp.write(b"\n")
        events = reader.poll([path])
        self.assertEqual(events[0]["values"]["step"], 8)
        self.assertEqual(reader.poll([path]), [])
        with path.open("ab") as fp:
            fp.write(b"x" * 200000)
        before = reader.offsets[str(path)]
        reader.poll([path], limit=4096)
        self.assertLessEqual(reader.offsets[str(path)] - before, 4096)

    def test_real_worker_step_is_visible_before_worker_exit(self):
        self.args._renderer = self.renderer
        self.args._event_reader = progress.EventReader()
        script = """
import argparse, time
from pathlib import Path
from sft_progress import emit
p=argparse.ArgumentParser()
p.add_argument('--phase'); p.add_argument('--event-file',type=Path); p.add_argument('--session-id')
a=p.parse_args()
emit(a,'train_step',step=1,target_steps=2,epoch=.5,loss=.9,grad_norm=.2,learning_rate=.0001,
     seconds=.1,tokens_per_second=100,non_padding_tokens=10,memory={})
release=a.event_file.parent/'release'
end=time.time()+10
while not release.exists() and time.time()<end: time.sleep(.01)
if not release.exists(): raise RuntimeError('No progress visible before exit')
"""
        observed = []

        def sleep(seconds):
            if "TRAIN 1/2" in self.stream.getvalue():
                observed.append(True)
                (self.args.event_file.parent / "release").touch()
            time.sleep(0.02)

        result = sft.supervise_worker(
            self.args,
            "train",
            command=[sys.executable, "-c", script, "--phase", "worker"],
            sleep=sleep,
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(observed)
        self.assertIn("PHASE END train exit_code=0", self.stream.getvalue())

    def test_auto_modes_and_overrides(self):
        self.assertEqual(tracking.choose_mode("auto", {}), "offline")
        self.assertEqual(
            tracking.choose_mode("auto", {"WANDB_API_KEY": "test-only-key"}), "online"
        )
        self.assertEqual(
            tracking.choose_mode("offline", {"WANDB_API_KEY": "test-only-key"}),
            "offline",
        )
        self.assertEqual(
            tracking.choose_mode("disabled", {"WANDB_API_KEY": "test-only-key"}),
            "disabled",
        )
        self.assertEqual(tracking.choose_mode("online", {}), "online")

    def test_missing_dependencies_fail_before_worker(self):
        self.args.wandb_mode = "offline"
        controller = tracking.TrackingController(self.args, printer=Mock())
        with self.assertRaisesRegex(RuntimeError, "--setup-tracking"):
            controller.ensure_dependencies()
        self.args.wandb_mode = "disabled"
        tracking.TrackingController(self.args).ensure_dependencies()

    def test_tracking_config_cli_precedence_and_no_key_persistence(self):
        self.args.wandb_project = "cli-project"
        self.args.wandb_entity = "cli-team"
        self.args.wandb_name = "cli-name"
        controller = tracking.TrackingController(self.args, printer=Mock())
        with patch.dict(
            os.environ, dict(WANDB_PROJECT="env-project", WANDB_API_KEY="test-only-key")
        ):
            controller.start()
        config = tracking.read(controller.config_path)
        self.assertEqual(config["project"], "cli-project")
        self.assertEqual(config["entity"], "cli-team")
        self.assertEqual(config["name"], "cli-name")
        self.assertNotIn("test-only-key", json.dumps(config))

    def test_payloads_are_allowlisted_and_have_step_axes_values(self):
        event = self.event(
            prompt="private prompt",
            response="private output",
            api_key="secret",
            checkpoint="/private/checkpoint",
            extra=float("nan"),
        )
        payload = progress.metric_payload(event)
        self.assertEqual(payload["train/step"], 1)
        self.assertEqual(payload["train/grad_norm"], 0.2)
        self.assertNotIn("private", json.dumps(payload))
        self.assertNotIn("secret", json.dumps(payload))
        payload = progress.metric_payload(
            self.event(
                "evaluation_complete",
                variant="training",
                split="validation",
                checkpoint_step=16,
                loss=0.98,
            )
        )
        self.assertEqual(payload["train/step"], 16)
        self.assertEqual(payload["eval/training/validation/loss"], 0.98)
        cfg = tracking.safe_config(
            dict(seed=42, api_key="secret", path="/private", prompts=["private"])
        )
        self.assertEqual(cfg, {"seed": 42})
        nested = tracking.safe_config(
            dict(
                hyperparameters=dict(learning_rate=0.0001, api_key="secret"),
                revisions=dict(model_revision="pinned", private="secret"),
            )
        )
        self.assertNotIn("secret", json.dumps(nested))
        summary = tracking.summary_payload(
            dict(
                status="complete",
                selected_checkpoint=dict(step=190, path="/private"),
                responses=["private"],
                config={"api_key": "secret"},
            )
        )
        self.assertEqual(summary["final/selected_checkpoint/step"], 190)
        self.assertNotIn("private", json.dumps(summary))

    def test_online_timeout_falls_back_to_fresh_offline_reporter(self):
        self.args.wandb_mode = "online"
        output = Mock()
        controller = tracking.TrackingController(self.args, printer=output)
        controller.session.mkdir(parents=True)
        controller.process = Mock()
        controller.process.poll.return_value = None
        controller.started = time.monotonic() - 16
        with (
            patch.object(controller, "terminate") as terminate,
            patch.object(controller, "launch") as launch,
        ):
            controller.poll()
        terminate.assert_called_once()
        launch.assert_called_once()
        self.assertEqual(controller.mode, "offline")
        self.assertTrue(controller.fallback)

    def test_stalled_online_reporter_falls_back_without_affecting_workers(self):
        self.args.wandb_mode = "online"
        controller = tracking.TrackingController(self.args, printer=Mock())
        controller.session.mkdir(parents=True)
        controller.process = Mock()
        controller.process.poll.return_value = None
        controller.started = time.monotonic()
        tracking.write(
            controller.status_path,
            dict(
                state="ready",
                heartbeat_at=time.time() - 31,
                url="https://example.invalid/run",
            ),
        )
        with (
            patch.object(controller, "terminate") as terminate,
            patch.object(controller, "launch") as launch,
        ):
            controller.safe_poll()
        terminate.assert_called_once()
        launch.assert_called_once()
        self.assertEqual(controller.mode, "offline")
        self.assertIsNone(controller.status["url"])

    def test_reporter_exception_never_escapes_watchdog(self):
        controller = tracking.TrackingController(self.args, printer=Mock())
        with (
            patch.object(controller, "poll", side_effect=OSError("tracker problem")),
            patch.object(controller, "terminate"),
        ):
            controller.safe_poll()
        self.assertEqual(controller.status["state"], "pending_recovery")

    def test_reporter_shutdown_is_bounded_and_only_own_group_is_signaled(self):
        self.args.wandb_mode = "offline"
        controller = tracking.TrackingController(self.args, printer=Mock())
        controller.session.mkdir(parents=True)
        process = Mock(pid=987654)
        process.poll.return_value = None
        controller.process = process
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        with (
            patch.object(tracking.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(tracking.time, "sleep", side_effect=sleep),
            patch.object(tracking.os, "killpg") as kill,
        ):
            controller.finish(timeout=0.2)
        self.assertLess(clock[0], 0.3)
        kill.assert_called_once_with(987654, signal.SIGKILL)
        self.assertTrue(controller.status["sync_pending"])

    def test_offline_reporter_replays_each_event_once_and_sets_metric_axes(self):
        session = self.args.run_dir / "tracking/sessions/session"
        tracking.write(
            session / "config.json",
            dict(
                mode="offline",
                project="test",
                entity=None,
                name="test",
                experiment_id="experiment",
                session_id="session",
                event_dir=str(self.args.event_file.parent),
                config=dict(seed=42),
                restored_checkpoint_step=8,
            ),
        )
        progress.emit(self.args, "train_step", step=9, loss=0.9, grad_norm=0.2)
        data = self.args.event_file.read_text()
        with self.args.event_file.open("a") as fp:
            fp.write(data)
        (session / "DONE").touch()
        run = Mock(entity="test", url=None)
        run.summary = {}
        sdk = SimpleNamespace(Settings=Mock(), init=Mock(return_value=run))
        with (
            patch.dict(os.environ, {}, clear=False),
            patch.dict(sys.modules, {"wandb": sdk}),
            patch.object(tracking.time, "sleep"),
        ):
            tracking.report_session(session)
        run.log.assert_called_once()
        run.define_metric.assert_any_call("train/*", step_metric="train/step")
        run.define_metric.assert_any_call(
            "eval/training/validation/*", step_metric="train/step"
        )
        kwargs = sdk.Settings.call_args.kwargs
        self.assertTrue(kwargs["disable_code"])
        self.assertTrue(kwargs["disable_git"])
        self.assertTrue(kwargs["disable_job_creation"])
        self.assertEqual(kwargs["console"], "off")
        cursor = next((session / "segments").glob("*/cursor.json"))
        self.assertEqual(tracking.read(cursor)["sdk_step"], 1)
        self.assertEqual(tracking.read(session / "status.json")["state"], "finished")

    def test_resume_sessions_group_under_same_experiment(self):
        first = tracking.TrackingController(self.args, printer=Mock())
        first.start()
        self.args.session_id = "resumed-session"
        self.args.restored_step = 8
        self.args.event_file = (
            self.args.run_dir / "telemetry/resumed-session/supervisor.jsonl"
        )
        second = tracking.TrackingController(self.args, printer=Mock())
        second.start()
        a, b = tracking.read(first.config_path), tracking.read(second.config_path)
        self.assertEqual(a["experiment_id"], b["experiment_id"])
        self.assertNotEqual(a["session_id"], b["session_id"])
        self.assertEqual(b["restored_checkpoint_step"], 8)

    def test_deadline_survives_event_flood_and_reporter_failure(self):
        args = self.args
        args.progress = "off"
        args.budget = dict(started_at=100.0, training_deadline=105.0, deadline=300.0)
        args._renderer = progress.Renderer(args, self.stream)
        args._event_reader = progress.EventReader()
        args.event_file.parent.mkdir(parents=True)
        (args.event_file.parent / "flood.jsonl").write_bytes(b"x" * 1000000)
        controller = tracking.TrackingController(args, printer=Mock())
        args._tracker = controller
        ticks = [100.0]
        sent = []

        class Process:
            pid = 987654
            code = None

            def poll(self):
                return self.code

            def wait(self):
                return self.code

        process = Process()

        def sleep(seconds):
            ticks[0] += seconds

        def kill(pid, sig):
            sent.append((pid, sig))
            if sig == signal.SIGKILL:
                process.code = -9

        with (
            patch.object(
                controller,
                "poll",
                side_effect=[OSError("tracking error")] + [None] * 300,
            ),
            patch.object(controller, "terminate"),
            patch.object(sft.os, "killpg", side_effect=kill),
        ):
            result = sft.supervise_worker(
                args,
                "train",
                command=["fake-worker"],
                clock=lambda: ticks[0],
                sleep=sleep,
                popen=lambda *a, **kw: process,
            )
        self.assertEqual(result["exit_code"], -9)
        self.assertLess(ticks[0], 300)
        self.assertEqual(sent, [(987654, signal.SIGTERM), (987654, signal.SIGKILL)])
        self.assertEqual(controller.status["state"], "pending_recovery")

    def test_sync_refuses_active_supervisor(self):
        with (self.args.run_dir / ".supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "supervisor is active"):
                tracking.sync_run(self.args.run_dir)

    def test_sync_skips_already_synced_and_updates_status(self):
        session = self.args.run_dir / "tracking/sessions/session"
        segment = session / "segments/segment"
        offline = segment / "wandb/offline-run-test"
        offline.mkdir(parents=True)
        (session / "DONE").touch()
        tracking.write(session / "config.json", {})
        tracking.write(
            session / "controller.json",
            dict(state="finished", sync_pending=True, session_id="session"),
        )
        tracking.write(segment / "segment.json", dict(mode="offline"))
        tracking.write(
            self.args.run_dir / "tracking/status.json", dict(session_id="session")
        )
        with (
            patch.dict(sys.modules, {"wandb": Mock()}),
            patch.object(tracking.subprocess, "run") as command,
        ):
            command.return_value.returncode = 0
            self.assertEqual(tracking.sync_run(self.args.run_dir), 0)
            self.assertEqual(tracking.sync_run(self.args.run_dir), 0)
        command.assert_called_once()
        self.assertFalse(
            tracking.read(self.args.run_dir / "tracking/status.json")["sync_pending"]
        )

    def test_finish_errors_do_not_change_training_outcome(self):
        controller = tracking.TrackingController(self.args, printer=Mock())
        with (
            patch.object(controller, "finish", side_effect=TimeoutError),
            patch.object(controller, "terminate"),
        ):
            controller.safe_finish()
        self.assertEqual(controller.status["state"], "pending_recovery")
        self.assertTrue(controller.status["sync_pending"])

    def test_legacy_entrypoints_and_artifact_prefixes(self):
        import overnight

        self.assertIs(overnight, sft)
        self.assertEqual(
            PrefixedOutput(Path("/tmp"), "sft") / "sft_32k.json",
            Path("/tmp/sft_32k.json"),
        )
        self.assertEqual(
            PrefixedOutput(Path("/tmp"), "overnight") / "sft_32k.json",
            Path("/tmp/overnight_32k.json"),
        )
        for command in ("sft", "overnight"):
            with (
                patch.object(sys, "argv", ["smoke_test.py", command, "--help"]),
                patch.object(sft, "main", return_value=0) as main,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(smoke.main(), 0)
                main.assert_called_once_with(["--help"])
        (self.args.root / "latest_overnight_run.txt").write_text("/legacy")
        self.assertEqual(tracking.latest_run(self.args.root), Path("/legacy"))
        (self.args.root / "latest_sft_run.txt").write_text("/canonical")
        self.assertEqual(tracking.latest_run(self.args.root), Path("/canonical"))


if __name__ == "__main__":
    unittest.main()
