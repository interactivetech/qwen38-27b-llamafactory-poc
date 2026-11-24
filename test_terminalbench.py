"""CPU-only evaluation contracts; mocked endpoints and simulated official Harbor results."""

import asyncio
import fnmatch
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import terminalbench as tb
import terminalbench_qwen as qwen
from export_terminalbench_results import export
from terminalbench_prepare import (
    cached_base,
    load_verified_adapter,
    verify_adapter_tensors,
)

HARBOR_AVAILABLE = importlib.util.find_spec("harbor") is not None


class FakeEndpoint(BaseHTTPRequestHandler):
    count = 120
    finish_reason = "stop"
    message = dict(
        content='{"analysis":"done","plan":"done","commands":[],"task_complete":true}',
        reasoning_content="Private reasoning stays separate.",
    )
    requests = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append((self.path, payload))
        if self.path == "/tokenize":
            result = dict(count=self.count, tokens=list(range(self.count)))
        else:
            result = dict(
                choices=[dict(message=self.message, finish_reason=self.finish_reason)],
                usage=dict(prompt_tokens=self.count, completion_tokens=12),
            )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = tb.parse_args(
            [
                "--root",
                str(self.root),
                "--sft-run",
                str(self.root / "overnight-20261124T022453Z"),
                "--run-dir",
                str(self.root / "eval"),
            ]
        )
        self.e = tb.Evaluation(self.args)
        self.addCleanup(self.e.lock.close)

    def result(self, entry, reward=None, **extra):
        path = self.e.directory / "harbor" / entry["id"] / "result.json"
        value = dict(
            task_name=entry["task"],
            verifier_result=None
            if reward is None
            else dict(rewards=dict(reward=reward)),
            **extra,
        )
        tb.write(path, value)
        self.e.state["trials"][entry["id"]] = dict(**entry, status="finished")
        return self.e.state["trials"][entry["id"]]

    def test_pairing_and_order(self):
        order = tb.trial_order(tb.TASKS)
        self.assertEqual(len(order), 10)
        self.assertEqual([o["task"] for o in order[::2]], tb.TASKS)
        self.assertEqual(
            [o["model"] for o in order],
            ["base", "sft", "sft", "base", "base", "sft", "sft", "base", "base", "sft"],
        )
        self.assertEqual(len({o["id"] for o in order}), 10)

    def test_exact_budget_boundaries(self):
        for count in (0, 120, 32512, 32513, 65279):
            limit = qwen.completion_limit(count)
            self.assertLessEqual(count + limit + 256, 65536)
            self.assertLessEqual(limit, 32768)
        self.assertEqual(qwen.completion_limit(32512), 32768)
        self.assertEqual(qwen.completion_limit(32513), 32767)
        self.assertEqual(qwen.completion_limit(65279), 1)
        with self.assertRaises(ValueError):
            qwen.completion_limit(65280)

    def test_reasoning_never_reaches_parser(self):
        self.assertEqual(
            qwen.split_reasoning(dict(content='<think>reason</think>{"commands":[]}')),
            ('{"commands":[]}', "reason"),
        )
        self.assertEqual(
            qwen.split_reasoning(dict(content="{}", reasoning="reason")),
            ("{}", "reason"),
        )
        with self.assertRaises(ValueError):
            qwen.split_reasoning(dict(content="<think>unfinished"))
        with self.assertRaises(ValueError):
            qwen.request("https://remote.example", "/v1/models")
        literal = '{"commands":[{"keystrokes":"echo <think>literal</think>"}]}'
        self.assertEqual(qwen.split_reasoning(dict(content=literal)), (literal, None))

    def test_reasoning_transport_matches_both_vllm_routes_without_mutating_history(
        self,
    ):
        messages = [
            dict(role="assistant", content="{}", reasoning_content="saved reasoning")
        ]
        payload = qwen.token_payload("base", messages)
        self.assertEqual(payload["messages"][0]["reasoning"], "saved reasoning")
        self.assertNotIn("reasoning_content", payload["messages"][0])
        self.assertEqual(messages[0]["reasoning_content"], "saved reasoning")
        with self.assertRaisesRegex(ValueError, "Conflicting reasoning"):
            qwen.api_messages(
                [
                    dict(
                        role="assistant",
                        reasoning="different",
                        reasoning_content="saved",
                    )
                ]
            )

    def test_rewards_missing_errors_and_paired_scores(self):
        order = self.e.state["order"]
        self.result(
            order[0],
            1,
            exception_info=dict(exception_type="AgentTimeoutError"),
            reward_valid=False,
        )
        self.result(order[1], 0)
        self.result(order[2], None)
        self.result(order[3], 1)
        self.e.report()
        summary = tb.read(self.e.directory / "summary.json")
        self.assertEqual(summary["scored_trials"], 3)
        self.assertEqual(summary["models"]["base"], dict(solved=2, scored=2, planned=5))
        self.assertEqual(summary["paired"]["base_wins"], 1)
        self.assertEqual(len(summary["paired"]["tasks"]), 1)
        self.assertEqual(summary["trials"][2]["reward"], None)
        self.assertIsNotNone(summary["trials"][0]["exception_info"])
        self.assertIn("unscored", (self.e.directory / "report.md").read_text())
        self.assertEqual(
            tb.reward_from_result(
                dict(verifier_result=dict(rewards=dict(reward=True)))
            ),
            None,
        )
        self.assertEqual(
            tb.reward_from_result(
                dict(verifier_result=dict(rewards=dict(reward=float("nan"))))
            ),
            None,
        )
        self.assertEqual(tb.reward_from_result(dict(completion_claim=True)), None)

    def test_resume_preserves_deadline_and_never_retries_interrupted(self):
        entry = self.e.state["order"][0]
        self.e.state["trials"][entry["id"]] = dict(**entry, status="started")
        self.e.save()
        deadline = self.e.state["deadline"]
        self.e.lock.close()
        self.args.resume = True
        resumed = tb.Evaluation(self.args)
        self.addCleanup(resumed.lock.close)
        self.assertEqual(resumed.state["deadline"], deadline)
        self.assertEqual(resumed.state["trials"][entry["id"]]["status"], "interrupted")
        with patch.object(
            resumed, "spawn", side_effect=AssertionError("must not repeat")
        ):
            self.assertEqual(
                resumed.trial(entry["task"], entry["model"], entry["id"])["status"],
                "interrupted",
            )

    def test_trial_started_is_persisted_before_spawn(self):
        self.e.endpoints = dict(base="http://127.0.0.1:8080")
        entry = self.e.state["order"][0]

        class Exited:
            pid = 999999999
            returncode = 0

            def poll(self):
                return 0

        def spawn(*a, **kw):
            saved = tb.read(self.e.state_path)
            self.assertEqual(saved["trials"][entry["id"]]["status"], "started")
            self.result(entry, 1)
            return Exited()

        with patch.object(self.e, "spawn", side_effect=spawn):
            self.e.trial(entry["task"], entry["model"], entry["id"])
        self.assertEqual(self.e.state["trials"][entry["id"]]["reward"], 1)

    def test_calibration_blocks_on_bad_or_missing_rewards(self):
        self.args.tasks = ["fix-git"]
        self.e.trial = lambda *a, **k: dict(reward=None, status="finished")
        with self.assertRaisesRegex(RuntimeError, "Calibration failure"):
            self.e.calibrate()
        calls = []

        def trial(task, model, *a, **k):
            calls.append((task, model))
            return dict(reward=1 if model == "oracle" else 0, status="finished")

        self.e.trial = trial
        self.e.calibrate()
        self.assertEqual(calls, [("fix-git", "oracle"), ("fix-git", "nop")])

    def test_official_timeouts_and_container_only_configuration(self):
        self.e.endpoints = dict(
            base="http://127.0.0.1:8080", sft="http://127.0.0.1:8081"
        )
        for model in ("base", "sft"):
            config = self.e.trial_config("fix-git", model, "trial")
            self.assertEqual(
                config["agent"]["base_url"], self.e.endpoints[model] + "/v1"
            )
            self.assertNotIn("override_timeout_sec", config["agent"])
            self.assertNotIn("timeout_multiplier", config)
            self.assertEqual(config["environment"]["type"], "singularity")
            mounts = config["environment"]["mounts"]
            self.assertEqual(
                {m["target"] for m in mounts},
                {
                    "/etc/resolv.conf",
                    "/etc/apt/apt.conf.d/99harbor-rootless",
                    "/logs/agent",
                    "/logs/verifier",
                    "/logs/artifacts",
                },
            )
            for mount in mounts:
                if mount["target"].startswith("/etc/"):
                    self.assertEqual(
                        Path(mount["source"]).parent,
                        self.e.directory / "runtime" / "trial",
                    )
                    self.assertTrue(mount["read_only"])
                    self.assertTrue(Path(mount["source"]).is_file())
                else:
                    self.assertTrue(
                        Path(mount["source"]).is_relative_to(
                            self.e.directory / "harbor/trial"
                        )
                    )
                    self.assertTrue(Path(mount["source"]).is_dir())
            self.assertEqual(
                config["agent"]["kwargs"]["proactive_summarization_threshold"], 33024
            )
            if HARBOR_AVAILABLE:
                from harbor.models.trial.config import TrialConfig

                TrialConfig.model_validate(config)

    def test_container_runtime_files_are_disposable_and_separate_per_trial(self):
        first = tb.runtime_mounts(self.e.directory / "runtime/first")
        dns = next(m for m in first if m["target"] == "/etc/resolv.conf")
        self.assertNotEqual(dns["source"], "/etc/resolv.conf")
        Path(dns["source"]).write_text("changed by task")
        second = tb.runtime_mounts(self.e.directory / "runtime/second")
        fresh = next(m for m in second if m["target"] == "/etc/resolv.conf")
        self.assertEqual(
            Path(fresh["source"]).read_text(), Path("/etc/resolv.conf").read_text()
        )
        self.assertEqual(self.e.env["APPTAINER_UNSQUASH"], "1")
        self.assertEqual(self.e.env["APPTAINER_WRITABLE"], "1")

    def test_failed_container_calibration_prevents_model_preparation(self):
        evaluation = Mock()
        evaluation.state = dict(phases={"preflight": True}, status="created")
        evaluation.calibrate.side_effect = RuntimeError("calibration failed")
        with (
            patch.object(tb, "Evaluation", return_value=evaluation),
            patch("signal.signal"),
        ):
            with self.assertRaisesRegex(RuntimeError, "calibration failed"):
                tb.main(
                    [
                        "--worker",
                        "--sft-run",
                        str(self.args.sft_run),
                        "--run-dir",
                        str(self.e.directory),
                    ]
                )
        evaluation.prepare.assert_not_called()
        evaluation.run.assert_not_called()
        evaluation.close.assert_called_once()

    def test_deadline_kills_stalled_worker_and_its_child_only(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        self.addCleanup(lambda: (unrelated.terminate(), unrelated.wait()))
        self.e.state["deadline"] = time.time() + 60 + 0.3
        self.e.save()
        with self.assertRaises(TimeoutError):
            self.e.command(
                [
                    sys.executable,
                    "-c",
                    'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); time.sleep(60)',
                ],
                "stalled",
            )
        self.assertIsNotNone(self.e.processes[-1].poll())
        self.assertIsNone(unrelated.poll())
        self.assertLess(time.time(), self.e.state["deadline"])

    def test_deadline_prevents_any_new_spawn(self):
        self.e.state["deadline"] = time.time() + 59
        with self.assertRaises(TimeoutError):
            self.e.spawn([sys.executable, "-c", "pass"], "too-late")
        self.assertFalse(self.e.processes)

    def test_merged_copy_tampering_blocks_serving(self):
        merged = self.root / "merged"
        merged.mkdir()
        (merged / "weights.safetensors").write_bytes(b"original")
        tb.write(
            self.e.directory / "prepared.json",
            dict(sft=str(merged), merged_hashes=tb.hashes(merged)),
        )
        self.e.state["phases"]["prepare"] = True
        (merged / "weights.safetensors").write_bytes(b"changed")
        with (
            patch.object(tb, "gpu_free"),
            patch.object(self.e, "spawn", side_effect=AssertionError("must not serve")),
        ):
            with self.assertRaisesRegex(ValueError, "Merged evaluation copy integrity"):
                self.e.start_servers()

    def test_old_preparation_without_verified_adapter_blocks_serving(self):
        weights = self.root / "weights"
        weights.mkdir()
        tb.write(self.e.directory / "reference.json", {})
        tb.write(
            self.e.directory / "prepared.json",
            dict(
                sft=str(weights),
                merged_hashes={},
                base=str(weights),
                base_hashes={},
                reference_sha256=tb.sha(self.e.directory / "reference.json"),
            ),
        )
        self.e.state["phases"]["prepare"] = True
        with patch.object(tb, "gpu_free"), patch.object(self.e, "spawn") as spawn:
            with self.assertRaisesRegex(ValueError, "Adapter loading was not verified"):
                self.e.start_servers()
            spawn.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires reference torch")
    def test_adapter_validation_rejects_initialized_missing_and_extra_tensors(self):
        import torch

        saved = {"lora_A": torch.tensor([1.0]), "lora_B": torch.tensor([0.5])}
        verify_adapter_tensors(saved, {k: v.clone() for k, v in saved.items()})
        for bad in (
            {},
            {"lora_A": saved["lora_A"]},
            {**saved, "unexpected": torch.tensor([1.0])},
            {**saved, "lora_B": torch.zeros(1)},
        ):
            with self.subTest(keys=list(bad)):
                with self.assertRaisesRegex(ValueError, "Adapter tensor"):
                    verify_adapter_tensors(saved, bad)

    @unittest.skipUnless(importlib.util.find_spec("peft"), "requires reference PEFT")
    def test_real_peft_loads_multimodal_checkpoint_into_text_model_exactly(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

        config = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["full_attention"],
        )
        model = get_peft_model(
            AutoModelForCausalLM.from_config(config),
            LoraConfig(
                r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"
            ),
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.125)
        checkpoint = self.root / "adapter"
        model.save_pretrained(checkpoint)
        path = checkpoint / "adapter_model.safetensors"
        tensors = load_file(path)
        save_file(
            {
                key.replace(
                    "base_model.model.model.layers.",
                    "base_model.model.model.language_model.layers.",
                ): value
                for key, value in tensors.items()
            },
            path,
            metadata={"format": "pt"},
        )
        loaded, proof = load_verified_adapter(
            AutoModelForCausalLM.from_config(config), checkpoint
        )
        self.assertTrue(proof["exact_match"])
        self.assertEqual(proof["tensor_count"], 2)
        for name, parameter in loaded.named_parameters():
            if "lora_B" in name:
                self.assertTrue(torch.all(parameter == 0.125))

    def test_serving_parity_also_checks_chat_route_prompt_counts(self):
        tb.write(
            self.e.directory / "reference.json",
            dict(
                probes=[
                    dict(
                        messages=[
                            dict(
                                role="assistant",
                                content="{}",
                                reasoning_content="saved",
                            )
                        ],
                        prompt=[1, 2],
                        target=[3],
                    )
                ],
                base=[[-1.0]],
            ),
        )
        chat_count = [2]

        def request(base, route, payload=None, **kw):
            if route == "/tokenize":
                self.assertEqual(payload["messages"][0]["reasoning"], "saved")
                return dict(tokens=[1, 2], count=2)
            if route == "/v1/completions":
                return dict(
                    choices=[
                        dict(logprobs=dict(token_logprobs=[None, -0.5, -2.0, -0.2]))
                    ]
                )
            self.assertEqual(route, "/v1/chat/completions")
            self.assertEqual(payload["messages"][0]["reasoning"], "saved")
            return dict(
                usage=dict(prompt_tokens=chat_count[0]),
                choices=[dict(finish_reason="stop")],
            )

        with patch.object(tb, "request", side_effect=request):
            self.e.serving_parity("base", "http://127.0.0.1:8000")
            advisory = tb.read(self.e.directory / "serving-parity-base.json")
            self.assertEqual(advisory["policy"], "advisory")
            self.assertFalse(advisory["within_tolerance"])
            self.e.report()
            summary = tb.read(self.e.directory / "summary.json")
            self.assertEqual(summary["numerical_checks"]["serving"]["base"], advisory)
            chat_count[0] = 3
            with self.assertRaisesRegex(
                ValueError, "chat/tokenizer prompt counts disagree"
            ):
                self.e.serving_parity("base", "http://127.0.0.1:8000")

    def test_corrupt_result_and_null_reward_remain_unscored(self):
        entry = self.e.state["order"][0]
        self.result(entry)
        path = self.e.directory / "harbor" / entry["id"] / "result.json"
        path.write_text('{"verifier_result":')
        self.e.report()
        summary = tb.read(self.e.directory / "summary.json")
        self.assertEqual(summary["scored_trials"], 0)
        self.assertEqual(summary["infrastructure_failures"], 1)
        self.assertIsNone(
            tb.reward_from_result(dict(verifier_result=dict(rewards=None)))
        )

    def test_resume_cleanup_requires_verified_process_ownership(self):
        p = self.e.spawn([sys.executable, "-c", "import time; time.sleep(60)"], "owned")
        try:
            # A forged/stale record must not make an unrelated process a cleanup target.
            identity = self.e.process_identity(p.pid)
            journal = self.e.directory / "processes.jsonl"
            original = journal.read_text()
            record = json.loads(original)
            record["start_ticks"] = "wrong"
            journal.write_text(json.dumps(record) + "\n")
            self.e.cleanup_abandoned_groups()
            self.assertIsNone(p.poll())
            journal.write_text(original)
            self.e.cleanup_abandoned_groups()
            p.wait(timeout=3)
            self.assertIsNotNone(p.returncode)
            self.assertIsNotNone(identity)
        finally:
            self.e.terminate(p)

    def test_busy_gpu_fails_without_killing_anything(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "12345\n", ""),
        ) as call:
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                tb.gpu_free()
            self.assertEqual(call.call_count, 1)

    def test_training_cache_does_not_require_repository_extras(self):
        files = {
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00001.safetensors",
            "tokenizer.json",
            "chat_template.jinja",
            "README.md",
            "LICENSE",
            ".gitattributes",
        }
        cached = files - {"README.md", "LICENSE", ".gitattributes"}

        def snapshot(repo, revision, local_files_only, allow_patterns=None):
            self.assertEqual(repo, "Qwen/Qwen3.8-27B")
            self.assertEqual(revision, "pinned-revision")
            self.assertTrue(local_files_only)
            requested = (
                files
                if allow_patterns is None
                else {
                    name
                    for name in files
                    if any(
                        fnmatch.fnmatchcase(name, pattern) for pattern in allow_patterns
                    )
                }
            )
            if requested - cached:
                raise ValueError("IncompleteSnapshotError: repository extras absent")
            self.assertIn("model-00001-of-00001.safetensors", requested)
            self.assertIn("config.json", requested)
            return str(self.root / "pinned-revision")

        with patch("huggingface_hub.snapshot_download", side_effect=snapshot):
            self.assertEqual(
                cached_base("pinned-revision"), self.root / "pinned-revision"
            )

    def test_worker_failure_surfaces_underlying_log(self):
        with self.assertRaisesRegex(RuntimeError, "specific preparation failure"):
            self.e.command(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('specific preparation failure\\n'); sys.exit(1)",
                ],
                "failing-prepare",
            )

    def test_stalled_reporter_is_optional_and_bounded(self):
        p = self.e.spawn(
            [sys.executable, "-c", "import time; time.sleep(60)"], "fake-reporter"
        )
        self.e.reporter = p
        self.e.reporter_started = time.time() - 100
        before = time.time()
        self.e.poll_reporter()
        self.assertIsNone(self.e.reporter)
        self.assertLess(time.time() - before, 4)
        self.assertIsNotNone(p.poll())

    def test_sanitized_export_drops_paths_errors_and_transcripts(self):
        record = self.result(self.e.state["order"][0], 1)
        record.update(error="secret", transcript="private task transcript")
        self.e.report()
        export(self.e.directory, self.root / "export")
        text = (self.root / "export/terminalbench21.json").read_text()
        self.assertNotIn("private task transcript", text)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("secret", text)

    def test_overlaps_block_exact_unicode_normalized_instructions(self):
        run = self.args.sft_run
        tb.write(
            run / "selected_trajectories.json",
            dict(
                trajectories=[
                    dict(
                        conversation_sha256="x",
                        conversations=[
                            dict(
                                content="Policy\nTask Description:\n FIX  Git\nCurrent terminal state:\nroot@id"
                            )
                        ],
                    )
                ]
            ),
        )
        tb.write(
            run / "context-32768/data/manifest.json",
            dict(records=dict(train=[dict(conversation_sha256="x")])),
        )
        result = tb.overlaps(run, {"fix-git": "fix git"})
        self.assertEqual(len(result["exact"]), 1)
        self.assertEqual(result["checked"], 1)
        with self.assertRaises(ValueError):
            tb.instruction_from_training("unknown framing")

    @unittest.skipUnless(HARBOR_AVAILABLE, "requires locked Harbor environment")
    def test_real_harbor_construction_preserves_native_parser_and_policy(self):
        from harbor.agents.terminus_2.terminus_2 import Terminus2
        from harbor.agents.terminus_2.terminus_json_plain_parser import (
            TerminusJSONPlainParser,
        )

        with patch.dict(
            os.environ, TB_ENDPOINT="http://127.0.0.1:8000", TB_MODEL="base"
        ):
            agent = qwen.agent_class()(
                logs_dir=self.root,
                model_name="openai/base",
                base_url="http://127.0.0.1:8000/v1",
                model_api_format="openai_chat_completions",
                skip_model_parsing=True,
                temperature=0,
                reasoning_effort="xhigh",
                interleaved_thinking=True,
                proactive_summarization_threshold=33024,
            )
        self.assertIsInstance(agent._parser, TerminusJSONPlainParser)
        self.assertEqual(agent._llm.model, "base")
        self.assertIs(qwen.agent_class()._run_agent_loop, Terminus2._run_agent_loop)


@unittest.skipUnless(HARBOR_AVAILABLE, "requires locked Harbor environment")
class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        FakeEndpoint.count = 120
        FakeEndpoint.finish_reason = "stop"
        FakeEndpoint.message = dict(
            content='{"commands":[]}', reasoning_content="private reasoning"
        )
        FakeEndpoint.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEndpoint)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.path = Path(self.temp.name) / "inference.jsonl"
        self.endpoint = qwen.Endpoint(
            f"http://127.0.0.1:{self.server.server_port}", "sft", self.path
        )

    def test_routing_exact_counts_reasoning_and_all_call_limits(self):
        for prompt in (
            "task action",
            "summary generation",
            "summary questions",
            "summary answers",
        ):
            response = asyncio.run(
                self.endpoint.call(
                    prompt,
                    message_history=[
                        dict(
                            role="assistant",
                            content="{}",
                            reasoning_content="old reasoning",
                        )
                    ],
                )
            )
            self.assertEqual(response.content, '{"commands":[]}')
            self.assertEqual(response.reasoning_content, "private reasoning")
        chats = [
            p for route, p in FakeEndpoint.requests if route == "/v1/chat/completions"
        ]
        tokens = [p for route, p in FakeEndpoint.requests if route == "/tokenize"]
        self.assertEqual(len(chats), 4)
        for chat, token in zip(chats, tokens):
            self.assertEqual(chat["messages"], token["messages"])
            self.assertEqual(chat["model"], "sft")
            self.assertEqual(chat["max_tokens"], 32768)
            self.assertEqual(chat["temperature"], 0)
            self.assertEqual(chat["top_p"], 1)
            self.assertEqual(chat["seed"], 42)
            self.assertEqual(
                chat["chat_template_kwargs"], token["chat_template_kwargs"]
            )
            self.assertEqual(chat["messages"][0]["reasoning"], "old reasoning")
            self.assertNotIn("reasoning_content", chat["messages"][0])
        self.assertEqual(len(self.path.read_text().splitlines()), 4)

    def test_reduced_budget_and_reasoning_truncation_are_recorded(self):
        from harbor.llms.base import OutputLengthExceededError

        FakeEndpoint.count = 65000
        FakeEndpoint.finish_reason = "length"
        FakeEndpoint.message = dict(content="<think>unfinished")
        with self.assertRaises(OutputLengthExceededError):
            asyncio.run(self.endpoint.call("summary"))
        record = json.loads(self.path.read_text())
        self.assertEqual(record["effective_limit"], 280)
        self.assertTrue(record["reasoning_truncated"])
        self.assertTrue(record["truncated"])

    def test_no_call_without_safe_context_budget(self):
        FakeEndpoint.count = 65500
        with self.assertRaises(ValueError):
            asyncio.run(self.endpoint.call("too big"))
        self.assertEqual([r for r, p in FakeEndpoint.requests], ["/tokenize"])

    def test_native_summary_trigger_includes_outgoing_prompt(self):
        from harbor.llms.chat import Chat

        with patch.dict(os.environ, TB_ENDPOINT=self.endpoint.base, TB_MODEL="sft"):
            agent = qwen.agent_class()(
                logs_dir=Path(self.temp.name),
                model_name="openai/sft",
                base_url=self.endpoint.base + "/v1",
                model_api_format="openai_chat_completions",
                skip_model_parsing=True,
                interleaved_thinking=True,
            )
        agent._summarize = AsyncMock(return_value=("handoff", []))
        chat = Chat(agent._llm, interleaved_thinking=True)
        FakeEndpoint.count = 32512  # Exactly 33,024 free: no summary.
        asyncio.run(agent._query_llm(chat, "outgoing prompt", "instruction", object()))
        agent._summarize.assert_not_awaited()
        FakeEndpoint.count = 32513
        asyncio.run(agent._query_llm(chat, "outgoing prompt", "instruction", object()))
        agent._summarize.assert_awaited_once()
        self.assertEqual(agent._pending_handoff_prompt, "handoff")

    def test_native_parser_errors_and_container_terminal_timing(self):
        from harbor.agents.terminus_2.terminus_2 import Command
        from harbor.llms.chat import Chat

        with patch.dict(os.environ, TB_ENDPOINT=self.endpoint.base, TB_MODEL="sft"):
            agent = qwen.agent_class()(
                logs_dir=Path(self.temp.name),
                model_name="openai/sft",
                base_url=self.endpoint.base + "/v1",
                model_api_format="openai_chat_completions",
                skip_model_parsing=True,
            )
        FakeEndpoint.message = dict(
            content="malformed JSON", reasoning_content="private"
        )
        result = asyncio.run(agent._handle_llm_interaction(Chat(agent._llm), "task"))
        self.assertTrue(result[2].startswith("ERROR:"))

        class Session:
            def __init__(self):
                self.sent = []

            async def send_keys(self, command, **kw):
                self.sent.append(command)

            async def get_incremental_output(self):
                return "container output"

        session = Session()
        self.assertEqual(
            asyncio.run(
                agent._execute_commands([Command("echo task\n", 0.1)], session)
            ),
            (False, "container output"),
        )
        self.assertEqual(session.sent, ["echo task\n"])
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertTrue(any(r.get("json_parse_failure") for r in records))
        self.assertTrue(any(r.get("kind") == "terminal" for r in records))


if __name__ == "__main__":
    unittest.main()
