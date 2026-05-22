"""OpenAI-compatible shim that forwards Hermes requests to `claude --print`.

The Claude CLI is not an HTTP API, but Hermes' agent loop expects a client with
`client.chat.completions.create(...)`. This adapter provides that small surface
by formatting the current Hermes message list into one prompt, invoking the
local Claude Code CLI in non-interactive print mode, and converting the result
back to the minimal OpenAI-style response shape used by Hermes.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import (
    _extract_tool_calls_from_text,
    _format_messages_as_prompt,
    _resolve_home_dir,
)

CLAUDE_CLI_MARKER_BASE_URL = "claude-cli://claude"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CLI_COMMAND", "").strip()
        or os.getenv("CLAUDE_CLI_PATH", "").strip()
        or "claude"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_CLAUDE_CLI_ARGS", "").strip()
    if raw:
        return shlex.split(raw)
    return ["--print", "--output-format", "json", "--no-session-persistence"]


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = _resolve_home_dir()
    return env


def _coerce_timeout(timeout: Any) -> float:
    if timeout is None:
        return _DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [
        getattr(timeout, attr, None)
        for attr in ("read", "write", "connect", "pool", "timeout")
    ]
    numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
    return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS


def _ensure_print_args(args: list[str], model: str | None) -> list[str]:
    final = list(args)
    if "--print" not in final and "-p" not in final:
        final.insert(0, "--print")
    if "--output-format" not in final:
        final.extend(["--output-format", "json"])
    if "--no-session-persistence" not in final:
        final.append("--no-session-persistence")
    if model and "--model" not in final:
        final.extend(["--model", model])
    return final


def _parse_claude_output(stdout: str) -> tuple[str, dict[str, Any]]:
    text = (stdout or "").strip()
    if not text:
        return "", {}
    try:
        obj = json.loads(text)
    except Exception:
        return text, {}
    if not isinstance(obj, dict):
        return text, {}
    result = obj.get("result")
    if isinstance(result, str):
        return result, obj
    message = obj.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content, obj
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts), obj
    if isinstance(obj.get("content"), str):
        return obj["content"], obj
    return text, obj


def _usage_from_payload(payload: dict[str, Any]) -> SimpleNamespace:
    usage_obj = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage_obj, dict):
        usage_obj = {}
    prompt_tokens = int(usage_obj.get("input_tokens") or usage_obj.get("prompt_tokens") or 0)
    completion_tokens = int(usage_obj.get("output_tokens") or usage_obj.get("completion_tokens") or 0)
    total_tokens = int(usage_obj.get("total_tokens") or (prompt_tokens + completion_tokens))
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )


class _ClaudeCLIChatCompletions:
    def __init__(self, client: "ClaudeCLIClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeCLIChatNamespace:
    def __init__(self, client: "ClaudeCLIClient"):
        self.completions = _ClaudeCLIChatCompletions(client)


class ClaudeCLIClient:
    """Minimal OpenAI-client-compatible facade for Claude Code CLI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        claude_command: str | None = None,
        claude_args: list[str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = claude_command or command or _resolve_command()
        self._args = list(claude_args or args or _resolve_args())
        self._cwd = str(Path(cwd or os.getcwd()).resolve())
        self.chat = _ClaudeCLIChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        cmd = [self._command] + _ensure_print_args(self._args, model)
        try:
            completed = subprocess.run(
                cmd,
                input=prompt_text,
                text=True,
                capture_output=True,
                timeout=_coerce_timeout(timeout),
                cwd=self._cwd,
                env=_build_subprocess_env(),
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Claude CLI command '{self._command}'. "
                "Install Claude Code or set HERMES_CLAUDE_CLI_COMMAND/CLAUDE_CLI_PATH."
            ) from exc
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"Claude CLI process failed: {detail}")

        response_text, payload = _parse_claude_output(completed.stdout or "")
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(
            choices=[choice],
            usage=_usage_from_payload(payload),
            model=model or "claude-cli",
        )
