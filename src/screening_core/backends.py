"""Pluggable LLM evaluation backends.

EvalBackend is the abstract interface; concrete implementations handle
Anthropic API, OpenAI API, Claude Code CLI, and a no-op null backend.

Apps assemble the system prompt and user message, then pass them to a backend.
No app-specific schema or business logic lives here.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any

import anthropic


class EvalBackend(ABC):
    """Abstract base for LLM backends.

    Concrete implementations return (tool_input_dict, input_tokens, output_tokens).
    System prompt and user message are assembled by the app-layer Evaluator and passed in.
    """

    token_accounting: str = "api_usage"  # "api_usage" | "estimated_chars" | "none"

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_message: str,
        tool: dict,
        max_tokens: int,
    ) -> tuple[dict, int, int]:
        """Send the evaluation prompt to the LLM; return (tool_input, in_tokens, out_tokens)."""
        ...

    def batch_call(
        self,
        system_prompt: str,
        user_message: str,
        batch_tool: dict,
        max_tokens: int,
    ) -> tuple[list[dict], int, int]:
        """Evaluate multiple items in one call; return (evaluations_list, in_tokens, out_tokens).

        Backends that do not support batch evaluation raise NotImplementedError.
        The app-layer batch_evaluate() falls back to sequential on NotImplementedError.
        """
        raise NotImplementedError


class ClaudeCodeBackend(EvalBackend):
    """Claude Code CLI (claude --print) via subprocess.

    Falls back to JSON-forced output instead of tool use, since the CLI does
    not expose tool_choice.

    Strips ANTHROPIC_API_KEY from the subprocess environment so the CLI
    authenticates via its subscription plan rather than the API key (PL-1).
    """

    token_accounting = "estimated_chars"

    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    _JSON_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._model = model

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tool: dict,
        max_tokens: int,
    ) -> tuple[dict, int, int]:
        schema_str = json.dumps(tool["input_schema"], ensure_ascii=False, indent=2)
        full_prompt = (
            f"{system_prompt}\n\n---\n\n{user_message}\n\n"
            "Output your evaluation as JSON matching the schema below exactly. "
            "Use a ```json ... ``` code block. No prose outside the block.\n\n"
            f"Schema:\n{schema_str}"
        )

        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        try:
            result = subprocess.run(
                ["claude", "--print", full_prompt, "--model", self._model],
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
                env=env,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "claude CLI not found. Install Claude Code and ensure it is in PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("claude CLI timed out (120 s)")

        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI error (exit {result.returncode}): {result.stderr[:300]}"
            )

        output = result.stdout.strip()
        tool_input = self._parse_json(output)

        in_tokens = len(full_prompt) // 4
        out_tokens = len(output) // 4
        return tool_input, in_tokens, out_tokens

    def batch_call(
        self,
        system_prompt: str,
        user_message: str,
        batch_tool: dict,
        max_tokens: int,
    ) -> tuple[list[dict], int, int]:
        schema_str = json.dumps(batch_tool["input_schema"], ensure_ascii=False, indent=2)
        full_prompt = (
            f"{system_prompt}\n\n---\n\n{user_message}\n\n"
            "Output your batch evaluation as JSON matching the schema below exactly. "
            "Use a ```json ... ``` code block. No prose outside the block.\n\n"
            f"Schema:\n{schema_str}"
        )
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        try:
            result = subprocess.run(
                ["claude", "--print", full_prompt, "--model", self._model],
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8",
                env=env,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "claude CLI not found. Install Claude Code and ensure it is in PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("claude CLI timed out (300 s) during batch evaluation")

        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI error (exit {result.returncode}): {result.stderr[:300]}"
            )

        output = result.stdout.strip()
        tool_input = self._parse_json(output)
        evaluations = tool_input.get("evaluations", [])
        in_tokens = len(full_prompt) // 4
        out_tokens = len(output) // 4
        return evaluations, in_tokens, out_tokens

    @classmethod
    def _parse_json(cls, text: str) -> dict:
        m = cls._JSON_BLOCK_RE.search(text)
        if m:
            return json.loads(m.group(1))
        m = cls._JSON_RE.search(text)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError(f"Could not find evaluation JSON in CLI output:\n{text[:400]}")


class AnthropicBackend(EvalBackend):
    """Anthropic Messages API with prompt caching and forced tool_use."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._client = client
        self._model = model

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tool: dict,
        max_tokens: int,
    ) -> tuple[dict, int, int]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user_message}],
        )
        tool_input = next(b.input for b in response.content if b.type == "tool_use")
        return tool_input, response.usage.input_tokens, response.usage.output_tokens

    def batch_call(
        self,
        system_prompt: str,
        user_message: str,
        batch_tool: dict,
        max_tokens: int,
    ) -> tuple[list[dict], int, int]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[batch_tool],
            tool_choice={"type": "tool", "name": batch_tool["name"]},
            messages=[{"role": "user", "content": user_message}],
        )
        tool_input = next(b.input for b in response.content if b.type == "tool_use")
        return tool_input["evaluations"], response.usage.input_tokens, response.usage.output_tokens


class NullBackend(EvalBackend):
    """No-op backend for --provider none / triage-only runs."""

    token_accounting = "none"

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tool: dict,
        max_tokens: int,
    ) -> tuple[dict, int, int]:
        raise RuntimeError(
            "NullBackend.call() must never be reached. "
            "The orchestrator should not create an Evaluator when provider=none."
        )


class OpenAIBackend(EvalBackend):
    """OpenAI Chat Completions API with Structured Outputs.

    The openai client is injected by the app-layer _build_backend() so that
    this module does not import openai at the top level — apps without the
    openai extra can still load this module.
    """

    token_accounting = "api_usage"

    def __init__(self, client: "Any", model: str = "gpt-4.1-mini") -> None:
        self._client = client
        self._model = model

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tool: dict,
        max_tokens: int,
    ) -> tuple[dict, int, int]:
        schema = dict(tool["input_schema"])
        schema["additionalProperties"] = False
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": tool["name"],
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        content = response.choices[0].message.content
        tool_input = json.loads(content)

        usage = response.usage
        if usage:
            return tool_input, usage.prompt_tokens, usage.completion_tokens

        self.__class__.token_accounting = "estimated_chars"
        return tool_input, (len(system_prompt) + len(user_message)) // 4, len(content) // 4
