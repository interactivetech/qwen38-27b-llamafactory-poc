#!/usr/bin/env python3
"""Bounded base/adapter output-quality screen. Generated commands are never executed."""

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import pilot_test as pilot
import smoke_test as smoke


def parse_response(text):
    """Validate the terminal-agent response after an optional model reasoning block."""
    if "<think>" in text:
        if "</think>" not in text:
            return dict(
                json_valid=False,
                schema_valid=False,
                errors=["Unclosed reasoning block"],
                parsed=None,
            )
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    for token in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(token):
            text = text[: -len(token)].rstrip()
    if text.startswith(("```json\n", "```\n")) and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except (ValueError, IndexError) as exc:
        return dict(
            json_valid=False, schema_valid=False, errors=[str(exc)], parsed=None
        )
    errors = []
    if not isinstance(value, dict):
        return dict(
            json_valid=True,
            schema_valid=False,
            errors=["Expected a JSON object"],
            parsed=value,
        )
    for key in ("analysis", "plan"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            errors.append(f"{key} must be a nonempty string")
    commands = value.get("commands")
    if not isinstance(commands, list):
        errors.append("commands must be an array")
    else:
        for i, command in enumerate(commands):
            if not isinstance(command, dict):
                errors.append(f"commands[{i}] must be an object")
                continue
            keys = command.get("keystrokes")
            if not isinstance(keys, str):
                errors.append(f"commands[{i}].keystrokes must be a string")
            elif keys and keys not in ("C-c", "C-d") and not keys.endswith("\n"):
                errors.append(f"commands[{i}] lacks a terminal newline")
            duration = command.get("duration", 1.0)
            if (
                type(duration) not in (int, float)
                or not math.isfinite(duration)
                or not 0 <= duration <= 60
            ):
                errors.append(
                    f"commands[{i}].duration must be a finite number in [0,60]"
                )
    if "task_complete" in value and not isinstance(value["task_complete"], bool):
        errors.append("task_complete must be Boolean")
    return dict(json_valid=True, schema_valid=not errors, errors=errors, parsed=value)


def diagnostic_pass(case, parsed):
    if not parsed["schema_valid"]:
        return False
    answer = parsed["parsed"]
    expected = case["expected"]
    if "task_complete" not in answer:
        return False
    if answer.get("task_complete", False) != expected["task_complete"]:
        return False
    commands = answer["commands"]
    if [row["keystrokes"] for row in commands] != expected["keystrokes"]:
        return False
    return (
        "duration" not in expected
        or commands[0].get("duration", 1.0) == expected["duration"]
    )


def diagnostic_cases():
    # These exact-action probes test instruction following and serialization,
    # not independent command planning or real terminal task success.
    instructions = """You are a terminal agent. Return one JSON object with nonempty string fields
"analysis" and "plan", an array "commands" of objects with string "keystrokes"
and numeric "duration", and a Boolean "task_complete". Ordinary shell commands
must end in a newline. An empty keystrokes string is a polling action. Durations
must be between 0 and 60 seconds. Do not claim a task is complete without evidence.
The terminal observations below are supplied data; no command will actually run here.
"""
    specs = [
        (
            "inspect_file",
            "The task is to inspect the first 40 lines of README.md. Observation: pwd printed /app/project and ls printed README.md. The inspection has not run. Your next action must be exactly head -n 40 README.md followed by a newline, with duration 0.1. Do not mark the task complete.",
            ["head -n 40 README.md\n"],
            False,
            0.1,
        ),
        (
            "wait_for_running_test",
            "The task is to run tests. Observation: pytest -q is still running; the shell prompt has not returned and no test result is available. Do not launch a second pytest. Submit exactly one polling action with an empty keystrokes string and duration 10. Do not mark the task complete.",
            [""],
            False,
            10,
        ),
        (
            "permission_denied",
            "The task is to inspect a protected file. Observation: cat /protected/config printed Permission denied. You must not use sudo, change permissions, or claim to have read the file. The only next action requested is id followed by a newline, duration 0.1. The original task is still incomplete.",
            ["id\n"],
            False,
            0.1,
        ),
        (
            "confirmed_completion",
            'The only task is to verify that /app/result.txt contains exactly OK followed by a newline. Observation: a completed Python verification printed MATCH and exited 0; it read the file and asserted that its bytes equal b"OK\\n". The task is complete. Return task_complete true and an empty commands array; no more actions are needed.',
            [],
            True,
            None,
        ),
    ]
    cases = []
    for name, prompt, keys, complete, duration in specs:
        expected = dict(keystrokes=keys, task_complete=complete)
        if duration is not None:
            expected["duration"] = duration
        cases.append(
            dict(
                case_id=name,
                kind="diagnostic",
                messages=[dict(role="user", content=instructions + prompt)],
                expected=expected,
            )
        )
    return cases


def model_args(args):
    manifest = smoke.read_json(args.pilot_run / "context-32768/data/manifest.json")
    return SimpleNamespace(
        root=args.root,
        run_dir=args.pilot_run,
        attempt=args.pilot_run / "context-32768",
        seq_len=32768,
        examples=manifest["train_examples"],
        eval_examples=manifest["eval_examples"],
        seed=manifest["seed"],
        revisions=manifest["revisions"],
    )


def prepare(args):
    pa = model_args(args)
    _, _, _, _, tm, template = pilot.components(pa)
    original = smoke.read_json(pa.attempt / "data/generation_inputs.json")
    selected = smoke.read_json(args.pilot_run / "selected_trajectories.json")[
        "trajectories"
    ]
    source = {row["source_index"]: row for row in selected}
    cases = []
    for i, row in enumerate(original):
        item = source[row["source_index"]]
        assert item["split"] == "validation"
        n = len(row["prompt_messages"])
        assert item["conversations"][:n] == row["prompt_messages"]
        reference = item["conversations"][n]
        assert reference["role"] == "assistant"
        cases.append(
            dict(
                case_id=f"heldout_{i + 1}",
                kind="heldout",
                source_index=row["source_index"],
                input_ids=row["input_ids"],
                prompt_sha256=row["prompt_sha256"],
                messages=row["prompt_messages"],
                reference_response=reference["content"],
            )
        )
    for case in diagnostic_cases():
        ids, _ = template.encode_oneturn(
            tm["tokenizer"],
            case["messages"] + [dict(role="assistant", content="")],
            "",
            "",
        )
        case.update(
            input_ids=ids,
            prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        )
        cases.append(case)
    smoke.write_json(args.output_dir / "cases.json", cases)
    smoke.write_json(
        args.output_dir / "config.json",
        dict(
            max_new_tokens=args.max_new_tokens,
            decoding="greedy",
            template="qwen3_8",
            reasoning_effort="xhigh",
            revisions=pa.revisions,
            seed=pa.seed,
            heldout_cases=4,
            diagnostic_cases=4,
            commands_executed=False,
            note="Same saved held-out input tokens for base and adapter; output budget does not truncate the input.",
        ),
    )
    print(f"Prepared {len(cases)} paired quality cases", flush=True)


def generate(args, variant):
    import torch

    pa = model_args(args)
    model, _, _, tm, _ = pilot.load(pa, adapter=variant == "adapter")
    tokenizer = tm["tokenizer"]
    cases = smoke.read_json(args.output_dir / "cases.json")
    records = []
    model.eval()
    with torch.inference_mode():
        for case in cases:
            ids = torch.tensor(
                [case["input_ids"]], device=next(model.parameters()).device
            )
            started = time.monotonic()
            print(
                f"QUALITY {variant}: starting {case['case_id']}, input_tokens={ids.shape[1]}",
                flush=True,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            tokens = output[0, ids.shape[1] :].tolist()
            response = tokenizer.decode(tokens, skip_special_tokens=False)
            parsed = parse_response(response)
            record = dict(
                case_id=case["case_id"],
                kind=case["kind"],
                variant=variant,
                prompt_sha256=case["prompt_sha256"],
                input_tokens=ids.shape[1],
                generated_tokens=len(tokens),
                response=response,
                elapsed_seconds=time.monotonic() - started,
                stopped_on_eos=bool(tokens and tokens[-1] == tokenizer.eos_token_id),
                token_cap_reached=len(tokens) == args.max_new_tokens
                and tokens[-1] != tokenizer.eos_token_id,
                reasoning_closed="</think>" in response,
                **parsed,
            )
            if case["kind"] == "diagnostic":
                record["diagnostic_pass"] = diagnostic_pass(case, parsed)
            records.append(record)
            smoke.write_json(args.output_dir / f"{variant}.json", records)
            print(
                f"QUALITY {variant}: {case['case_id']} tokens={len(tokens)}, capped={record['token_cap_reached']}, schema_valid={record['schema_valid']}",
                flush=True,
            )
            del ids, output
    return records


def report(args):
    from transformers import AutoTokenizer

    cases = smoke.read_json(args.output_dir / "cases.json")
    config = smoke.read_json(args.output_dir / "config.json")
    summary = dict(
        status="completed",
        config=config,
        commands_executed=False,
        scope="Four previously selected held-out contexts plus four exact-action instruction/format probes; not an executable task-success benchmark.",
        models={},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.pilot_run / "context-32768/adapter")
    )
    summary["heldout_reference_lengths"] = [
        dict(
            case_id=case["case_id"],
            tokens=len(
                tokenizer.encode(case["reference_response"], add_special_tokens=False)
            ),
        )
        for case in cases
        if case["kind"] == "heldout"
    ]
    summary["budget_caveat"] = (
        "Some held-out references exceed the generation budget. A capped response "
        "is not usable under this budget, but is not proof of incorrect reasoning or task failure."
    )
    markdown = [
        "# Output-quality comparison",
        "",
        "Generated commands were inspected as text and never executed.",
        "",
        f"Maximum output: {args.max_new_tokens} tokens per case, greedy decoding.",
        "",
    ]
    expected = [case["case_id"] for case in cases]
    hashes = [case["prompt_sha256"] for case in cases]
    for variant in ("base", "adapter"):
        records = smoke.read_json(args.output_dir / f"{variant}.json")
        assert [r["case_id"] for r in records] == expected
        assert [r["prompt_sha256"] for r in records] == hashes
        metrics = {}
        for kind in ("heldout", "diagnostic"):
            subset = [r for r in records if r["kind"] == kind]
            metrics[kind] = dict(
                cases=len(subset),
                json_valid=sum(r["json_valid"] for r in subset),
                schema_valid=sum(r["schema_valid"] for r in subset),
                token_cap_reached=sum(r["token_cap_reached"] for r in subset),
                completed_generation=sum(r["stopped_on_eos"] for r in subset),
                generated_tokens=[r["generated_tokens"] for r in subset],
            )
            if kind == "diagnostic":
                metrics[kind]["exact_action_pass"] = sum(
                    r["diagnostic_pass"] for r in subset
                )
        summary["models"][variant] = metrics
        markdown.extend(
            [f"## {variant}", "", "```json", json.dumps(metrics, indent=2), "```", ""]
        )
    markdown.extend(["## Complete responses", ""])
    by_variant = {
        name: {
            r["case_id"]: r for r in smoke.read_json(args.output_dir / f"{name}.json")
        }
        for name in ("base", "adapter")
    }
    summary["cases"] = []
    metric_fields = (
        "generated_tokens",
        "json_valid",
        "schema_valid",
        "token_cap_reached",
        "stopped_on_eos",
        "reasoning_closed",
        "diagnostic_pass",
    )
    for case in cases:
        summary["cases"].append(
            dict(
                case_id=case["case_id"],
                kind=case["kind"],
                prompt_sha256=case["prompt_sha256"],
                input_tokens=len(case["input_ids"]),
                **{
                    name: {key: row[key] for key in metric_fields if key in row}
                    for name in ("base", "adapter")
                    for row in (by_variant[name][case["case_id"]],)
                },
            )
        )
        markdown.extend([f"### {case['case_id']}", ""])
        if case["kind"] == "heldout":
            markdown.extend([f"Source index: {case['source_index']}.", ""])
        for variant in ("base", "adapter"):
            row = by_variant[variant][case["case_id"]]
            markdown.extend(
                [
                    f"**{variant}** ({row['generated_tokens']} tokens; capped: {row['token_cap_reached']})",
                    "",
                    "~~~~text",
                    row["response"],
                    "~~~~",
                    "",
                ]
            )
        if case["kind"] == "heldout":
            markdown.extend(
                [
                    "**Held-out reference next response (not shown to either model)**",
                    "",
                    "~~~~text",
                    case["reference_response"],
                    "~~~~",
                    "",
                ]
            )
    smoke.write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "comparison.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(summary["models"]), flush=True)
    print(f"QUALITY COMPLETE: {args.output_dir / 'comparison.md'}", flush=True)


def all_phases(args):
    for phase in ("prepare", "base", "adapter", "report"):
        smoke.write_json(
            args.output_dir / "status.json", dict(status="running", phase=phase)
        )
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--root",
            str(args.root),
            "--pilot-run",
            str(args.pilot_run),
            "--output-dir",
            str(args.output_dir),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        with (args.output_dir / f"{phase}.log").open("x") as fp:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                for line in process.stdout:
                    fp.write(line)
                    fp.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            smoke.write_json(
                args.output_dir / "status.json",
                dict(status="failed", phase=phase, exit_code=code),
            )
            raise RuntimeError(
                f"Quality {phase} failed; see {args.output_dir / (phase + '.log')}"
            )
    smoke.write_json(
        args.output_dir / "status.json",
        dict(status="completed", commands_executed=False),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("SMOKE_ROOT", smoke.DEFAULT_ROOT)),
    )
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--phase",
        choices=["all", "prepare", "base", "adapter", "report"],
        default="all",
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    args.root, args.pilot_run = args.root.resolve(), args.pilot_run.resolve()
    smoke.environment(args.root)
    if args.output_dir is None:
        if args.phase != "all":
            parser.error("--output-dir is required for individual phases")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        args.output_dir = args.pilot_run / "quality" / stamp
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final = smoke.read_json(args.pilot_run / "summary.json")
    assert final["status"] == "passed", "Quality comparison requires a completed pilot"
    print(f"QUALITY_OUTPUT_DIR={args.output_dir} PHASE={args.phase}", flush=True)
    try:
        if args.phase == "all":
            all_phases(args)
        elif args.phase in ("base", "adapter"):
            generate(args, args.phase)
        else:
            globals()[args.phase](args)
    except Exception as exc:
        smoke.write_json(
            args.output_dir / f"{args.phase}-failure.json",
            dict(error=str(exc), traceback=traceback.format_exc()),
        )
        raise


if __name__ == "__main__":
    main()
