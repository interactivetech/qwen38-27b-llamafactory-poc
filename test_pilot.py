"""Tests for split identity, complete prefixes, and process-level fallback."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pilot_test as pilot
import smoke_test as smoke


class Template:
    efficient_eos = False

    def encode_multiturn(self, tokenizer, messages, system, tools):
        return [
            ([0] * len(messages[i]["content"]), [1] * len(messages[i + 1]["content"]))
            for i in range(0, len(messages), 2)
        ]


class PilotTests(unittest.TestCase):
    def checkpoint_fixture(self, directory):
        attempt = Path(directory) / "context-32768"
        checkpoint = attempt / "adapter/checkpoint-8"
        checkpoint.mkdir(parents=True)
        for name in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (checkpoint / name).touch()
        smoke.write_json(
            checkpoint / "trainer_state.json",
            dict(
                global_step=8,
                max_steps=32,
                log_history=[
                    dict(step=i, loss=1.0, grad_norm=0.2) for i in range(1, 9)
                ],
            ),
        )
        return SimpleNamespace(
            attempt=attempt, resume_from_checkpoint=checkpoint, examples=64
        )

    def test_resume_checkpoint_preserves_history_and_rejects_missing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.checkpoint_fixture(directory)
            state = pilot.checkpoint_state(args)
            self.assertEqual(state["global_step"], 8)
            self.assertEqual(len(state["log_history"]), 8)
            (args.resume_from_checkpoint / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                pilot.checkpoint_state(args)

    def test_resume_rejects_another_context(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.checkpoint_fixture(directory)
            args.attempt = Path(directory) / "context-16384"
            with self.assertRaisesRegex(ValueError, "belong"):
                pilot.checkpoint_state(args)

    def test_resume_runs_only_training_and_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.checkpoint_fixture(directory)
            args.run_dir = Path(directory)
            args.seq_len = 32768
            args.resume_from_checkpoint = None
            for name in ("baseline.json", "probe.json"):
                smoke.write_json(args.attempt / name, {})
            smoke.write_json(
                args.run_dir / "run_status.json",
                {"status": "stopped", "elapsed_seconds": 20},
            )
            calls = []

            def child(a, phase):
                calls.append(phase)
                if phase == "verify":
                    smoke.write_json(
                        a.run_dir / "summary.json", {"learning_signal": True}
                    )
                return dict(phase=phase, exit_code=0, log="test-only", error={})

            with (
                patch.object(pilot, "checked_data"),
                patch.object(pilot, "child", child),
            ):
                pilot.resume(args)
            self.assertEqual(calls, ["train", "verify"])
            self.assertEqual(
                smoke.read_json(args.run_dir / "run_status.json")["status"], "passed"
            )

    def test_task_group_ignores_terminal_container_but_preserves_task(self):
        def messages(task, screen):
            return [
                {
                    "role": "user",
                    "content": f"{task}\nCurrent terminal state:\n{screen}",
                }
            ]

        self.assertEqual(
            pilot.task_key(messages("solve task A", "root@abc")),
            pilot.task_key(messages("solve   task A", "root@xyz")),
        )
        self.assertNotEqual(
            pilot.task_key(messages("solve task A", "root@abc")),
            pilot.task_key(messages("solve task B", "root@abc")),
        )

    def test_longest_prefix_keeps_complete_turns_and_original_content(self):
        messages = [
            {"role": role, "content": content}
            for role, content in [
                ("user", "abc"),
                ("assistant", '<think>x</think>{"commands":[]}'),
                ("user", "terminal observation"),
                ("assistant", "next command"),
                ("user", "unanswered"),
            ]
        ]
        first_size = sum(len(x["content"]) for x in messages[:2])
        self.assertEqual(
            pilot.complete_prefix(messages, None, Template(), first_size), messages[:2]
        )
        self.assertEqual(
            pilot.complete_prefix(messages, None, Template(), first_size - 1), []
        )
        self.assertEqual(
            pilot.complete_prefix(messages, None, Template(), 1000), messages[:4]
        )
        self.assertEqual(len(messages), 5)

    def test_invalid_role_order_is_rejected(self):
        with self.assertRaises(ValueError):
            pilot.complete_prefix(
                [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
                None,
                Template(),
                50,
            )

    def run_fallback_case(self, fail_kind):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        args = SimpleNamespace(
            run_dir=Path(temp.name), seq_len=32768, fallback_seq_len=16384
        )
        calls = []

        def child(a, phase):
            calls.append((a.seq_len, phase, a.run_dir))
            fail = a.seq_len == 32768 and phase == "probe"
            if not fail and phase == "verify":
                smoke.write_json(a.run_dir / "summary.json", {"learning_signal": False})
            return dict(
                phase=phase,
                seq_len=a.seq_len,
                exit_code=int(fail),
                log="test-only",
                error={"oom": fail_kind == "oom"} if fail else {},
            )

        with patch.object(pilot, "child", child):
            if fail_kind == "oom":
                pilot.all_phases(args)
            else:
                with self.assertRaises(RuntimeError):
                    pilot.all_phases(args)
        return args, calls

    def test_oom_restarts_all_phases_once_with_same_run_directory(self):
        args, calls = self.run_fallback_case("oom")
        self.assertEqual(
            [phase for length, phase, _ in calls if length == 16384],
            ["preflight", "prepare", "probe", "baseline", "train", "verify"],
        )
        self.assertEqual({path for _, _, path in calls}, {args.run_dir})
        self.assertTrue(smoke.read_json(args.run_dir / "summary.json")["fallback_used"])
        self.assertEqual(
            smoke.read_json(args.run_dir / "run_status.json")["status"], "passed"
        )

    def test_non_oom_does_not_change_context_or_retry(self):
        args, calls = self.run_fallback_case("other")
        self.assertEqual({length for length, _, _ in calls}, {32768})
        self.assertEqual(
            smoke.read_json(args.run_dir / "run_status.json")["status"], "failed"
        )

    def test_both_contexts_oom_stop_after_one_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                run_dir=Path(directory), seq_len=32768, fallback_seq_len=16384
            )
            calls = []

            def child(a, phase):
                calls.append((a.seq_len, phase))
                return dict(
                    phase=phase,
                    seq_len=a.seq_len,
                    exit_code=int(phase == "probe"),
                    log="test-only",
                    error={"oom": True} if phase == "probe" else {},
                )

            with patch.object(pilot, "child", child), self.assertRaises(RuntimeError):
                pilot.all_phases(args)
            self.assertEqual(
                [length for length, phase in calls if phase == "probe"],
                [32768, 16384],
            )
            self.assertEqual(
                smoke.read_json(args.run_dir / "run_status.json")["status"], "failed"
            )


if __name__ == "__main__":
    unittest.main()
