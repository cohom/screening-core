"""Pluggable LLM evaluation backends.

EvalBackend is the abstract interface; concrete implementations handle
Anthropic API, OpenAI API, Claude Code CLI, Codex CLI, and a no-op null backend.

Apps assemble the system prompt and user message, then pass them to a backend.
No app-specific schema or business logic lives here.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
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


class CodexBackend(EvalBackend):
    """OpenAI Codex CLI (``codex exec``) via subprocess — the OpenAI-family CLI
    transport, symmetric to :class:`ClaudeCodeBackend`.

    Like the Claude Code backend it forces JSON output via the prompt rather than a
    structured schema: ``codex exec --output-schema`` demands strict OpenAI
    structured-output form (``additionalProperties: false`` on every object, every key
    required) that screening tool schemas — especially batch ones — do not satisfy, and
    in practice it returned empty objects for reasoning tasks. Instead the agent's
    **final message** is captured with ``--output-last-message`` (clean, no banner or
    token telemetry) and the JSON block is parsed out of it.

    **The prompt is fed on stdin (the ``-`` sentinel), never as an argv element.** On
    Windows the npm shim (``codex.CMD``) re-parses the command line and truncates long
    multi-line prompts, so the model sees partial input; stdin sidesteps argv limits.

    Codex is an *agentic* CLI (shell + web tools) and heavier than ``claude --print``.
    Reasoning effort therefore defaults to ``low``: the ``xhigh`` default burns ~14k
    tokens on a single trivial evaluation, prohibitive across a screening run.
    ``--sandbox read-only`` + ``--skip-git-repo-check`` keep it from touching the
    filesystem or requiring a git repo (emitting JSON needs neither).

    Strips ``OPENAI_API_KEY`` from the subprocess environment so the CLI authenticates
    via the ChatGPT subscription plan rather than the API key (mirrors
    :class:`ClaudeCodeBackend`, which drops ``ANTHROPIC_API_KEY``).

    ``model`` may be empty, in which case no ``--model`` flag is passed and codex uses
    its own configured default (``~/.codex/config.toml``). That is the robust default:
    the models a Codex subscription accepts are account- and CLI-version-dependent and
    differ from the OpenAI **API** model space, so a caller that cannot name a
    known-good id should delegate rather than guess.
    """

    token_accounting = "estimated_chars"

    # Reuse the proven JSON extractors from the Claude Code CLI backend.
    _JSON_BLOCK_RE = ClaudeCodeBackend._JSON_BLOCK_RE
    _JSON_RE = ClaudeCodeBackend._JSON_RE

    # ``codex exec`` prints a startup banner (``model: gpt-5-codex`` etc.). When no
    # --model is passed, the banner is the only place the actually-used model surfaces.
    _MODEL_BANNER_RE = re.compile(r"(?im)^\s*model:\s*(\S+)")

    def __init__(
        self,
        model: str = "",
        *,
        timeout: int = 120,
        batch_timeout: int = 300,
        reasoning_effort: str = "low",
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._batch_timeout = batch_timeout
        self._reasoning_effort = reasoning_effort
        # Model codex reported at run time (from the startup banner). None until the
        # first call; exposed via observed_model() so an app can record which model
        # actually served the run even when self._model is empty (delegated default).
        self._observed_model: str | None = None

    def observed_model(self) -> str | None:
        """Model codex reported using at run time, or None if not yet observed."""
        return self._observed_model

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
        output = self._run(full_prompt, self._timeout)
        tool_input = self._parse_json(output)
        return tool_input, len(full_prompt) // 4, len(output) // 4

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
        output = self._run(full_prompt, self._batch_timeout)
        tool_input = self._parse_json(output)
        evaluations = tool_input.get("evaluations", [])
        if not isinstance(evaluations, list):
            evaluations = []
        return evaluations, len(full_prompt) // 4, len(output) // 4

    def _run(self, full_prompt: str, timeout: int) -> str:
        """Run ``codex exec`` non-interactively and return the agent's final message.

        The prompt goes on **stdin** (with ``-`` as the prompt sentinel) to avoid Windows
        argv mangling of long multi-line prompts. The final message is read from
        ``--output-last-message`` (a temp file) so the banner and token telemetry printed
        to stdout never contaminate the JSON parse.
        """
        fd, last_path = tempfile.mkstemp(suffix=".txt", prefix="codex_last_")
        os.close(fd)
        # Resolve the executable path: CreateProcess cannot find the Windows npm shim
        # (codex.CMD) by bare name.
        codex_exe = shutil.which("codex") or "codex"
        cmd = [
            codex_exe, "exec",
            "-c", f'model_reasoning_effort="{self._reasoning_effort}"',
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--color", "never",
            "--output-last-message", last_path,
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.append("-")  # read the prompt from stdin

        # Force ChatGPT subscription auth (drop the API key), mirroring ClaudeCodeBackend.
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        try:
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"codex CLI error (exit {result.returncode}): {result.stderr[:300]}"
                )
            # Record the model codex actually used (banner → stdout, sometimes stderr).
            # Best-effort: never let a parse miss break the run.
            banner = f"{result.stdout or ''}\n{result.stderr or ''}"
            m = self._MODEL_BANNER_RE.search(banner)
            if m:
                self._observed_model = m.group(1).strip()
            output = Path(last_path).read_text(encoding="utf-8").strip()
            # Fall back to stdout only if the last-message file came back empty.
            return output or (result.stdout or "").strip()
        except FileNotFoundError:
            raise RuntimeError(
                "codex CLI not found. Install Codex CLI and ensure it is in PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"codex CLI timed out ({timeout} s)")
        finally:
            try:
                os.unlink(last_path)
            except OSError:
                pass

    @classmethod
    def _parse_json(cls, text: str) -> dict:
        return ClaudeCodeBackend._parse_json(text)


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
