"""CPU-only overnight control-plane tests; no dataset download or GPU allocation."""

import contextlib
import copy
import math
import signal
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import overnight as night
import smoke_test as smoke


class OvernightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        run = root / "runs/overnight"
        ref = root / "runs/reference"
        attempt = run / "context-32768"
        attempt.mkdir(parents=True)
        (root / "requirements.lock").write_text("pinned dependencies")
        smoke.write_json(
            ref / "context-32768/data/manifest.json",
            dict(
                revisions=dict(model_revision="model-sha", dataset_revision="data-sha"),
                seq_len=32768,
                seed=42,
            ),
        )
        smoke.write_json(ref / "selected_trajectories.json", {})
        adapter = ref / "context-32768/adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"pilot")
        (adapter / "adapter_config.json").write_text("{}")
        self.args = SimpleNamespace(
            root=root,
            run_dir=run,
            attempt=attempt,
            reference_run=ref,
            hours=12,
            seq_len=32768,
            train_examples=400,
            val_examples=32,
            test_examples=32,
            resume=False,
            phase="all",
        )
        night.initialize(self.args)

    def checkpoint(self, step, loss=None):
        args = self.args
        path = args.attempt / f"checkpoints/checkpoint-{step}"
        path.mkdir(parents=True)
        for name in night.CHECKPOINT_FILES:
            (path / name).write_bytes(b"fixture")
        state = dict(
            global_step=step,
            max_steps=200,
            log_history=[
                dict(step=i, loss=1.0, grad_norm=0.2) for i in range(1, step + 1)
            ],
            examples_seen=step * 2,
            non_padding_tokens=step * 100,
            verified_updates=True,
        )
        smoke.write_json(path / "trainer_state.json", state)
        smoke.write_json(
            path / "complete.json",
            dict(
                identity=args.identity,
                files={n: night.sha(path / n) for n in night.CHECKPOINT_FILES},
            ),
        )
        if loss is not None:
            smoke.write_json(
                path / "validation.json",
                dict(
                    complete=True,
                    loss=loss,
                    per_example_losses=[loss] * 32,
                    checkpoint_sha256=night.sha(path / "complete.json"),
                ),
            )
        return path

    def test_split_isolation_across_all_four_sets(self):
        rows = {
            k: [dict(source_index=i, task_group=f"g{i}", conversation_sha256=f"h{i}")]
            for i, k in enumerate(("train", "validation", "test", "anchor"))
        }
        night.isolated(rows)
        for field in ("task_group", "conversation_sha256", "source_index"):
            bad = copy.deepcopy(rows)
            bad["test"][0][field] = bad["anchor"][0][field]
            with self.assertRaises(ValueError):
                night.isolated(bad)
        rows["train"].append(rows["train"][0])
        with self.assertRaises(ValueError):
            night.isolated(rows)

    def test_encoding_keeps_complete_turns_reasoning_and_assistant_masks(self):
        class Template:
            efficient_eos = False

            def encode_multiturn(self, tokenizer, messages, system, tools):
                return [
                    (
                        [ord(c) for c in messages[i]["content"]],
                        [ord(c) for c in messages[i + 1]["content"]],
                    )
                    for i in range(0, len(messages), 2)
                ]

        template = Template()

        class Processor:
            def _encode_data_example(self, prompt, response, *args):
                ids, labels = [], []
                for source, target in template.encode_multiturn(
                    None, prompt + response, "", ""
                ):
                    ids.extend(source + target)
                    labels.extend([-100] * len(source) + target)
                return ids, labels

        messages = [
            dict(role="user", content="task"),
            dict(role="assistant", content='<think>reason</think>{"commands":[]}'),
            dict(role="user", content="observation"),
            dict(role="assistant", content="next"),
            dict(role="user", content="unfinished"),
        ]
        item = dict(
            source_index=1,
            task_group=night.pilot.task_key(messages),
            conversation_sha256=night.pilot.digest(messages),
            conversations=messages,
        )
        result = night.encode(item, self.args, None, template, Processor())
        row, record, prefix = result
        self.assertEqual(prefix, messages[:4])
        self.assertEqual(record["messages"], 4)
        self.assertEqual(
            "".join(chr(x) for x in row["labels"] if x != -100),
            messages[1]["content"] + messages[3]["content"],
        )
        self.assertEqual(row["labels"][:4], [-100] * 4)
        self.args.seq_len = len(messages[0]["content"]) + len(messages[1]["content"])
        self.assertEqual(
            night.encode(item, self.args, None, template, Processor())[2], messages[:2]
        )
        self.assertEqual(len(messages), 5)

    def test_completion_marker_and_hashes_required(self):
        checkpoint = self.checkpoint(8)
        self.assertEqual(
            night.complete_checkpoint(checkpoint, self.args.identity)["global_step"], 8
        )
        (checkpoint / "complete.json").unlink()
        self.assertEqual(night.checkpoints(self.args), [])
        checkpoint = self.checkpoint(16)
        (checkpoint / "optimizer.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            night.complete_checkpoint(checkpoint)
        self.assertEqual(night.checkpoints(self.args), [])

    def test_selection_uses_only_completed_validation_not_latest(self):
        best = self.checkpoint(8, 0.8)
        self.checkpoint(16, 0.9)
        latest = self.checkpoint(24)
        self.assertEqual(night.select_checkpoint(self.args)["path"], str(best))
        self.assertEqual(night.checkpoints(self.args)[-1][1], latest)
        smoke.write_json(latest / "validation.json", dict(complete=False, loss=0.1))
        self.assertEqual(night.select_checkpoint(self.args)["step"], 8)
        smoke.write_json(
            latest / "validation.json",
            dict(complete=True, loss=0.1, per_example_losses=[0.1] * 32),
        )
        self.assertEqual(night.select_checkpoint(self.args)["step"], 8)
        marker = night.read(best / "validation.json")
        marker["checkpoint_sha256"] = "wrong-checkpoint"
        smoke.write_json(best / "validation.json", marker)
        self.assertEqual(night.select_checkpoint(self.args)["step"], 16)

    def test_resume_preserves_absolute_deadlines_and_identity(self):
        original = dict(self.args.budget)
        self.args.resume = True
        with patch.object(
            night.time, "time", return_value=original["started_at"] + 7200
        ):
            night.initialize(self.args)
        self.assertEqual(self.args.budget, original)
        self.args.hours = 11
        with self.assertRaisesRegex(ValueError, "changed"):
            night.initialize(self.args)

    def test_resume_rejects_changed_dependencies_and_reference_adapter(self):
        self.args.resume = True
        (self.args.root / "requirements.lock").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            night.initialize(self.args)

    def test_resume_requires_complete_step_history(self):
        checkpoint = self.checkpoint(8)
        state = night.read(checkpoint / "trainer_state.json")
        state["log_history"].pop()
        smoke.write_json(checkpoint / "trainer_state.json", state)
        marker = night.read(checkpoint / "complete.json")
        marker["files"]["trainer_state.json"] = night.sha(
            checkpoint / "trainer_state.json"
        )
        smoke.write_json(checkpoint / "complete.json", marker)
        with self.assertRaisesRegex(ValueError, "history"):
            night.complete_checkpoint(checkpoint)

    def simulated_worker(self, phase, cooperative):
        args = self.args
        ticks = [100.0]
        args.budget = dict(started_at=100.0, training_deadline=105.0, deadline=300.0)
        sent = []

        class Process:
            pid = 987654
            code = None

            def poll(self):
                if cooperative and (args.run_dir / "STOP_TRAIN").exists():
                    self.code = 0
                return self.code

            def wait(self):
                return self.code

        process = Process()

        def kill(pid, sig):
            sent.append((pid, sig, ticks[0]))
            if sig == signal.SIGKILL:
                process.code = -9

        def sleep(seconds):
            ticks[0] += seconds

        with patch.object(night.os, "killpg", side_effect=kill):
            result = night.supervise_worker(
                args,
                phase,
                command=["fake-worker"],
                clock=lambda: ticks[0],
                sleep=sleep,
                popen=lambda *a, **k: process,
            )
        return result, sent, ticks[0]

    def test_graceful_shutdown_requests_optimizer_boundary(self):
        result, sent, end = self.simulated_worker("train", True)
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["forced"])
        self.assertEqual(sent, [])
        self.assertLess(end, 110)
        self.assertTrue((self.args.run_dir / "STOP_TRAIN").exists())
        self.assertTrue((self.args.run_dir / "status.json").exists())

    def test_stalled_worker_gets_term_then_kill_only_own_group(self):
        result, sent, end = self.simulated_worker("train", False)
        self.assertEqual(result["exit_code"], -9)
        self.assertEqual([x[1] for x in sent], [signal.SIGTERM, signal.SIGKILL])
        self.assertTrue(all(x[0] == 987654 for x in sent))
        self.assertTrue(result["forced"])
        self.assertLess(end, 300)

    def test_overall_deadline_kills_stalled_evaluation(self):
        result, sent, end = self.simulated_worker("loss-base", False)
        self.assertEqual(result["exit_code"], -9)
        self.assertLessEqual(end, self.args.budget["deadline"])
        self.assertTrue((self.args.run_dir / "STOP").exists())

    def test_worker_failure_creates_partial_report_without_fallback(self):
        calls = []

        def worker(args, phase):
            calls.append(phase)
            return dict(
                phase=phase, exit_code=1, failure=dict(error="CUDA out of memory")
            )

        with patch.object(night, "supervise_worker", side_effect=worker):
            self.assertEqual(night.run(self.args), 1)
        self.assertEqual(calls, ["preflight"])
        self.assertEqual(
            night.read(self.args.run_dir / "status.json")["status"], "failed"
        )
        summary = night.read(self.args.run_dir / "summary.json")
        self.assertFalse(summary["training_verified"])
        self.assertTrue(summary["skipped_work"])
        for name in (
            "report.md",
            "summary.json",
            "paired_responses.json",
            "paired_responses.md",
            "training_curves.csv",
        ):
            self.assertTrue((self.args.run_dir / name).exists())
        self.assertEqual(
            (self.args.root / "latest_overnight_run.txt").read_text().strip(),
            str(self.args.run_dir),
        )

    def test_expired_resume_never_launches_worker(self):
        self.args.resume = True
        self.args.budget["deadline"] = time.time() - 1
        with patch.object(night, "supervise_worker") as worker:
            night.run(self.args)
        worker.assert_not_called()
        self.assertEqual(
            night.read(self.args.run_dir / "status.json")["status"], "partial"
        )

    def test_paired_bootstrap_is_deterministic(self):
        result = night.bootstrap([1.0, 2.0, 3.0], [0.8, 1.6, 2.4])
        self.assertAlmostEqual(result["relative_improvement"], 0.2)
        for value in result["relative_95_interval"]:
            self.assertAlmostEqual(value, 0.2)
        self.assertEqual(result, night.bootstrap([1.0, 2.0, 3.0], [0.8, 1.6, 2.4]))

    def test_partial_report_does_not_claim_verified_training(self):
        self.checkpoint(8, 0.8)
        night.select_checkpoint(self.args)
        night.report(self.args, "partial")
        summary = night.read(self.args.run_dir / "summary.json")
        self.assertFalse(summary["training_verified"])
        self.assertFalse(summary["promising_scaling_result"])
        self.assertEqual(summary["training"]["global_step"], 8)

    def test_training_checkpoint_resume_restores_history_optimizer_and_order(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable; run with pinned environment")
        args = self.args
        args.train_examples = args.examples = 64

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_B = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, target=None, **kwargs):
                return SimpleNamespace(loss=(self.lora_B - target).square())

            def save_pretrained(self, path, **kwargs):
                torch.save(self.state_dict(), path / "adapter_model.safetensors")
                (path / "adapter_config.json").write_text("{}")

        class Tokenizer:
            def save_pretrained(self, path):
                (path / "tokenizer_config.json").write_text("{}")

        models = []

        def load(args, training=False, adapter=None):
            model = Tiny()
            if adapter:
                model.load_state_dict(
                    torch.load(adapter / "adapter_model.safetensors", weights_only=True)
                )
            models.append(model)
            return model, Tokenizer(), None

        data = dict(
            train=[dict(input_ids=[i], labels=[i]) for i in range(64)],
            validation=[{}] * 32,
        )
        manifest = dict(records=dict(train=[dict(supervised_tokens=1)] * 64))
        seen = []

        def batch(row, device):
            seen.append(row["input_ids"][0])
            return dict(target=float(row["input_ids"][0] + 1))

        def at_eight(args, training=False):
            return night.read(args.run_dir / "progress.json", {}).get("step", 0) >= 8

        with (
            patch.object(night, "checked_data", return_value=(manifest, data)),
            patch.object(night, "load_model", side_effect=load),
            patch.object(night.pilot, "batch_for", side_effect=batch),
            patch.object(night, "memory", return_value={}),
            patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
            patch.object(torch.cuda, "set_rng_state_all"),
            patch.object(torch.cuda, "empty_cache"),
            patch.object(
                torch, "autocast", side_effect=lambda *a, **k: contextlib.nullcontext()
            ),
            patch.object(
                night,
                "score",
                return_value=dict(
                    complete=True, loss=0.8, per_example_losses=[0.8] * 32
                ),
            ),
            patch("builtins.print"),
        ):
            with patch.object(night, "stopped", side_effect=at_eight):
                night.train(args)
            checkpoint = night.checkpoints(args)[-1]
            initial_history = copy.deepcopy(checkpoint[2]["log_history"])
            self.assertEqual(checkpoint[0], 8)
            self.assertTrue(checkpoint[2]["verified_updates"])
            self.assertEqual(len(seen), 16)
            first_weight = models[-1].lora_B.item()
            with patch.object(night, "stopped", return_value=False):
                night.train(args)
            latest = night.checkpoints(args)[-1]
            self.assertEqual(latest[0], 32)
            self.assertEqual(latest[2]["log_history"][:8], initial_history)
            self.assertEqual(latest[2]["examples_seen"], 64)
            self.assertEqual(latest[2]["non_padding_tokens"], 64)
            self.assertNotEqual(models[-1].lora_B.item(), first_weight)
            expected_order = list(range(64))
            import random

            random.Random(42).shuffle(expected_order)
            self.assertEqual(seen, expected_order)
            self.assertEqual(len(night.checkpoints(args)), 4)
            self.assertTrue((latest[1] / "scheduler.pt").exists())

    def test_optimizer_step_checks_finite_loss_gradients_and_changes_weights(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable; run with pinned environment")

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_B = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, **kwargs):
                return SimpleNamespace(loss=(self.lora_B - 1).square())

        model = Tiny()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        with (
            patch.object(night.pilot, "batch_for", return_value={}),
            patch.object(torch, "autocast", return_value=contextlib.nullcontext()),
        ):
            loss, norm = night.optimizer_step(model, optimizer, [{}, {}])
            self.assertGreater(norm, 0)
            self.assertTrue(math.isfinite(loss))
            self.assertNotEqual(model.lora_B.item(), 0)
            model.lora_B.data.fill_(1.0)
            with self.assertRaisesRegex(ValueError, "zero adapter gradients"):
                night.optimizer_step(model, optimizer, [{}, {}])
            model.lora_B.data.fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                night.optimizer_step(model, optimizer, [{}, {}])


if __name__ == "__main__":
    unittest.main()
