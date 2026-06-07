from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp
from hermes_cli import auth as auth_mod


def test_resolve_provider_alias_claude_cli():
    assert auth_mod.resolve_provider("claude-cli") == "claude-cli"
    assert auth_mod.resolve_provider("claude-code-cli") == "claude-cli"
    assert auth_mod.resolve_provider("anthropic-cli") == "claude-cli"


def test_resolve_runtime_provider_claude_cli(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "claude-cli")
    monkeypatch.setattr(
        rp,
        "resolve_external_process_provider_credentials",
        lambda provider: {
            "provider": provider,
            "api_key": "claude-cli",
            "base_url": "claude-cli://claude",
            "command": "/usr/local/bin/claude",
            "args": ["--print", "--output-format", "json", "--no-session-persistence"],
            "source": "process",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="claude-cli")

    assert resolved["provider"] == "claude-cli"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == "claude-cli"
    assert resolved["base_url"] == "claude-cli://claude"
    assert resolved["command"] == "/usr/local/bin/claude"
    assert resolved["args"][:2] == ["--print", "--output-format"]
    assert resolved["requested_provider"] == "claude-cli"


def test_claude_cli_client_invokes_print_json(monkeypatch, tmp_path):
    from agent.claude_cli_client import ClaudeCLIClient

    calls = []

    class _Completed:
        returncode = 0
        stdout = '{"result":"hello from claude","usage":{"input_tokens":3,"output_tokens":4}}'
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    client = ClaudeCLIClient(command="claude", args=["--print", "--output-format", "json"], cwd=str(tmp_path))
    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Say hello"}],
        timeout=12,
    )

    cmd, kwargs = calls[0]
    assert cmd[:4] == ["claude", "--print", "--output-format", "json"]
    assert kwargs["input"]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["timeout"] == 12
    assert response.choices[0].message.content == "hello from claude"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 4


def test_claude_cli_client_passes_reasoning_effort(monkeypatch, tmp_path):
    from agent.claude_cli_client import ClaudeCLIClient

    calls = []

    class _Completed:
        returncode = 0
        stdout = '{"result":"ok"}'
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    client = ClaudeCLIClient(command="claude", args=["--print"], cwd=str(tmp_path))
    client.chat.completions.create(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "Say ok"}],
        reasoning_config={"enabled": True, "effort": "high"},
    )

    cmd, _kwargs = calls[0]
    assert cmd[-4:] == ["--model", "claude-opus-4-7", "--effort", "high"]


def test_claude_cli_client_extracts_tool_calls(monkeypatch, tmp_path):
    from agent.claude_cli_client import ClaudeCLIClient

    class _Completed:
        returncode = 0
        stdout = '<tool_call>{"id":"call_1","type":"function","function":{"name":"terminal","arguments":{"command":"pwd"}}}</tool_call>'
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Completed())

    client = ClaudeCLIClient(command="claude", args=["--print"], cwd=str(tmp_path))
    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "pwd"}],
    )

    msg = response.choices[0].message
    assert response.choices[0].finish_reason == "tool_calls"
    assert msg.tool_calls[0].function.name == "terminal"
    assert msg.tool_calls[0].function.arguments == '{"command": "pwd"}'
