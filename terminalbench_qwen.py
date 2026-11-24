"""Narrow endpoint adapter; Harbor retains its Terminus parser and execution policy."""

import asyncio
import json
import time
import urllib.request
from pathlib import Path

CONTEXT = 65536
OUTPUT = 32768
RESERVE = 256
SUMMARIZE = OUTPUT + RESERVE
TEMPLATE_KWARGS = dict(
    enable_thinking=True, reasoning_effort="xhigh", preserve_thinking=True
)


def request(base, route, payload=None, timeout=60):
    if not base.startswith("http://127.0.0.1:"):
        raise ValueError("Evaluation endpoints must be loopback-only")
    req = urllib.request.Request(
        base.rstrip("/") + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def api_messages(messages):
    # Harbor stores the legacy field. vLLM's chat route renames it, but its
    # tokenizer route does not; normalize before either route to preserve COT.
    result = []
    for original in messages:
        message = dict(original)
        legacy = message.pop("reasoning_content", None)
        if legacy is not None:
            if message.get("reasoning") is not None and message["reasoning"] != legacy:
                raise ValueError("Conflicting reasoning fields in chat history")
            message["reasoning"] = legacy
        result.append(message)
    return result


def token_payload(model, messages):
    return dict(
        model=model,
        messages=api_messages(messages),
        add_generation_prompt=True,
        chat_template_kwargs=TEMPLATE_KWARGS,
    )


def count_tokens(base, model, messages):
    result = request(base, "/tokenize", token_payload(model, messages))
    if not isinstance(result.get("count"), int) or result["count"] < 0:
        raise ValueError("vLLM /tokenize did not return an exact prompt count")
    return result["count"]


def completion_limit(count):
    limit = min(OUTPUT, CONTEXT - count - RESERVE)
    if limit <= 0:
        raise ValueError(f"No safe completion budget: prompt={count}")
    return limit


def split_reasoning(message):
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    # Fail closed on an unparsed reasoning prefix: it must never reach the action parser.
    has_prefix = content.lstrip().startswith("<think>")
    implicit_prefix = not reasoning and not content.lstrip().startswith(
        ("{", "[", "```")
    )
    if "</think>" in content and (has_prefix or implicit_prefix):
        prefix, content = content.split("</think>", 1)
        reasoning = (reasoning or "") + prefix.lstrip().removeprefix("<think>")
    elif has_prefix:
        raise ValueError("Unterminated reasoning in model content")
    return content.strip(), reasoning


class Endpoint:
    def __init__(self, base, model, journal):
        self.base, self.model, self.journal = base, model, Path(journal)

    def get_model_context_limit(self):
        return CONTEXT

    def get_model_output_limit(self):
        return OUTPUT

    def count(self, messages):
        return count_tokens(self.base, self.model, messages)

    def record(self, value):
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a") as stream:
            stream.write(json.dumps(value) + "\n")

    async def call(self, prompt, message_history=None, **kwargs):
        from harbor.llms.base import LLMResponse, OutputLengthExceededError
        from harbor.models.metric import UsageInfo

        messages = api_messages(
            list(message_history or []) + [{"role": "user", "content": prompt}]
        )
        count = await asyncio.to_thread(self.count, messages)
        limit = completion_limit(count)
        started = time.monotonic()
        result = await asyncio.to_thread(
            request,
            self.base,
            "/v1/chat/completions",
            dict(
                model=self.model,
                messages=messages,
                temperature=0,
                top_p=1,
                seed=42,
                max_tokens=limit,
                reasoning_effort="xhigh",
                chat_template_kwargs=TEMPLATE_KWARGS,
            ),
            900,
        )
        choice = result["choices"][0]
        usage = result["usage"]
        if usage["prompt_tokens"] != count:
            raise ValueError(
                f"Tokenizer/completion prompt disagreement: {count} vs {usage['prompt_tokens']}"
            )
        message = choice["message"]
        reasoning_present = bool(
            message.get("reasoning_content")
            or message.get("reasoning")
            or (message.get("content") or "").lstrip().startswith("<think>")
        )
        record = dict(
            kind="inference",
            at=time.time(),
            model=self.model,
            prompt_tokens=count,
            completion_tokens=usage["completion_tokens"],
            effective_limit=limit,
            remaining_context=CONTEXT - count,
            inference_seconds=time.monotonic() - started,
            finish_reason=choice["finish_reason"],
            truncated=choice["finish_reason"] == "length",
            reasoning_present=reasoning_present,
            reasoning_truncated=reasoning_present
            and choice["finish_reason"] == "length",
        )
        self.record(record)
        if (
            record["truncated"]
            and (message.get("content") or "").lstrip().startswith("<think>")
            and "</think>" not in message["content"]
        ):
            raise OutputLengthExceededError(
                "Qwen reasoning truncated", truncated_response=""
            )
        content, reasoning = split_reasoning(message)
        if record["truncated"]:
            raise OutputLengthExceededError(
                "Qwen completion limit reached", truncated_response=content
            )
        return LLMResponse(
            content=content,
            reasoning_content=reasoning,
            model_name=self.model,
            usage=UsageInfo(
                prompt_tokens=count,
                completion_tokens=usage["completion_tokens"],
                cache_tokens=0,
                cost_usd=0,
            ),
            extra=record,
        )


# Lazy factory keeps pure token-budget tests free of Harbor imports.
def agent_class():
    from harbor.agents.terminus_2.terminus_2 import Terminus2

    class QwenTerminus2(Terminus2):
        def _init_llm(self, **kwargs):
            import os

            return Endpoint(
                os.environ["TB_ENDPOINT"],
                os.environ["TB_MODEL"],
                self.logs_dir / "inference.jsonl",
            )

        def _count_total_tokens(self, chat):
            # The actual outgoing user prompt is added and counted again in every call,
            # including all three native summary calls and fallback summaries.
            return self._llm.count(chat.messages or [{"role": "user", "content": ""}])

        async def _query_llm(self, chat, prompt, original_instruction="", session=None):
            count = await asyncio.to_thread(
                self._llm.count, chat.messages + [{"role": "user", "content": prompt}]
            )
            if (
                CONTEXT - count < SUMMARIZE
                and self.options.enable_summarize
                and session is not None
            ):
                prompt, refs = await self._summarize(
                    chat, original_instruction, session
                )
                self._pending_subagent_refs = refs
                self._pending_handoff_prompt = prompt
            return await super()._query_llm(chat, prompt, original_instruction, session)

        async def _execute_commands(self, commands, session):
            started = time.monotonic()
            try:
                return await super()._execute_commands(commands, session)
            finally:
                self._llm.record(
                    dict(kind="terminal", terminal_seconds=time.monotonic() - started)
                )

        async def _handle_llm_interaction(self, *args, **kwargs):
            result = await super()._handle_llm_interaction(*args, **kwargs)
            self._llm.record(
                dict(kind="turn", json_parse_failure=result[2].startswith("ERROR:"))
            )
            return result

    return QwenTerminus2


# Harbor's import_path expects a module-level class.
def __getattr__(name):
    if name == "QwenTerminus2":
        return agent_class()
    raise AttributeError(name)
