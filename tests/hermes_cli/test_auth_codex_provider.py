"""Tests for Codex auth — tokens stored in Hermes auth store (~/.hermes/auth.json)."""

import json
import time
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    PROVIDER_REGISTRY,
    _read_codex_tokens,
    _save_codex_tokens,
    _import_codex_cli_tokens,
    _login_openai_codex,
    get_codex_auth_status,
    get_provider_auth_state,
    refresh_codex_oauth_pure,
    resolve_codex_runtime_credentials,
    resolve_provider,
)


def _setup_hermes_auth(hermes_home: Path, *, access_token: str = "access", refresh_token: str = "refresh"):
    """Write Codex tokens into the Hermes auth store."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "last_refresh": "2026-02-26T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps(auth_store, indent=2))
    return auth_file


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"h.{encoded}.s"


def test_read_codex_tokens_success(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == "access"
    assert data["tokens"]["refresh_token"] == "refresh"


def test_read_codex_tokens_missing(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Empty auth store
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        _read_codex_tokens()
    assert exc.value.code == "codex_auth_missing"


def test_resolve_codex_runtime_credentials_missing_access_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing_access_token"
    assert exc.value.relogin_required is True


def test_resolve_codex_runtime_credentials_refreshes_expiring_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    expiring_token = _jwt_with_exp(int(time.time()) - 10)
    _setup_hermes_auth(hermes_home, access_token=expiring_token, refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    called = {"count": 0}

    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "access-new", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()

    assert called["count"] == 1
    assert resolved["api_key"] == "access-new"


def test_resolve_codex_runtime_credentials_force_refresh(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="access-current", refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    called = {"count": 0}

    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "access-forced", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials(force_refresh=True, refresh_if_expiring=False)

    assert called["count"] == 1
    assert resolved["api_key"] == "access-forced"


def test_resolve_provider_explicit_codex_does_not_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_provider("openai-codex") == "openai-codex"


def test_save_codex_tokens_roundtrip(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens({"access_token": "at123", "refresh_token": "rt456"})
    data = _read_codex_tokens()

    assert data["tokens"]["access_token"] == "at123"
    assert data["tokens"]["refresh_token"] == "rt456"


def test_import_codex_cli_tokens(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-cli"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": "cli-at", "refresh_token": "cli-rt"},
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    tokens = _import_codex_cli_tokens()
    assert tokens is not None
    assert tokens["access_token"] == "cli-at"
    assert tokens["refresh_token"] == "cli-rt"


def test_import_codex_cli_tokens_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nonexistent"))
    assert _import_codex_cli_tokens() is None


def test_codex_tokens_not_written_to_shared_file(tmp_path, monkeypatch):
    """Verify _save_codex_tokens writes only to Hermes auth store, not ~/.codex/."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    hermes_home.mkdir(parents=True, exist_ok=True)
    codex_home.mkdir(parents=True, exist_ok=True)

    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    _save_codex_tokens({"access_token": "hermes-at", "refresh_token": "hermes-rt"})

    # ~/.codex/auth.json should NOT exist — _save_codex_tokens only touches Hermes store
    assert not (codex_home / "auth.json").exists()

    # Hermes auth store should have the tokens
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == "hermes-at"


def test_resolve_returns_hermes_auth_store_source(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    creds = resolve_codex_runtime_credentials()
    assert creds["source"] == "hermes-auth-store"
    assert creds["provider"] == "openai-codex"
    assert creds["base_url"] == DEFAULT_CODEX_BASE_URL


class _StubHTTPResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StubHTTPClient:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self._response


def _patch_httpx(monkeypatch, response):
    def _factory(*args, **kwargs):
        return _StubHTTPClient(response)

    monkeypatch.setattr("hermes_cli.auth.httpx.Client", _factory)


def test_refresh_parses_openai_nested_error_shape_refresh_token_reused(monkeypatch):
    """OpenAI returns {"error": {"code": "refresh_token_reused", "message": "..."}}
    — parser must surface relogin_required and the dedicated message.
    """
    response = _StubHTTPResponse(
        401,
        {
            "error": {
                "message": "Your refresh token has already been used to generate a new access token. Please try signing in again.",
                "type": "invalid_request_error",
                "param": None,
                "code": "refresh_token_reused",
            }
        },
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "refresh_token_reused"
    assert err.relogin_required is True
    # The existing dedicated branch should override the message with actionable guidance.
    assert "already consumed by another client" in str(err)


def test_refresh_parses_openai_nested_error_shape_generic_code(monkeypatch):
    """Nested error with arbitrary code still surfaces code + message."""
    response = _StubHTTPResponse(
        400,
        {
            "error": {
                "message": "Invalid client credentials.",
                "type": "invalid_request_error",
                "code": "invalid_client",
            }
        },
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "invalid_client"
    assert "Invalid client credentials." in str(err)


def test_refresh_parses_oauth_spec_flat_error_shape_invalid_grant(monkeypatch):
    """Fallback path: OAuth spec-shape {"error": "invalid_grant", "error_description": "..."}
    must still map to relogin_required=True via the existing code set.
    """
    response = _StubHTTPResponse(
        400,
        {
            "error": "invalid_grant",
            "error_description": "Refresh token is expired or revoked.",
        },
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "invalid_grant"
    assert err.relogin_required is True
    assert "Refresh token is expired or revoked." in str(err)


def test_refresh_falls_back_to_generic_message_on_unparseable_body(monkeypatch):
    """No JSON body → generic 'with status 401' message; 401 always forces relogin."""
    response = _StubHTTPResponse(401, ValueError("not json"))
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "codex_refresh_failed"
    # 401/403 from the token endpoint always means the refresh token is
    # invalid/expired — force relogin even without a parseable error body.
    assert err.relogin_required is True
    assert "status 401" in str(err)


def test_login_openai_codex_force_new_login_skips_existing_reuse_prompt(monkeypatch):
    called = {"device_login": 0}

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda: {"base_url": DEFAULT_CODEX_BASE_URL},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": "cli-at", "refresh_token": "cli-rt"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._codex_device_code_login",
        lambda: {
            "tokens": {"access_token": "fresh-at", "refresh_token": "fresh-rt"},
            "last_refresh": "2026-04-01T00:00:00Z",
            "base_url": DEFAULT_CODEX_BASE_URL,
        },
    )

    def _fake_save(tokens, last_refresh=None):
        called["device_login"] += 1
        called["tokens"] = dict(tokens)
        called["last_refresh"] = last_refresh

    monkeypatch.setattr("hermes_cli.auth._save_codex_tokens", _fake_save)
    monkeypatch.setattr("hermes_cli.auth._update_config_for_provider", lambda *args, **kwargs: "/tmp/config.yaml")
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": (_ for _ in ()).throw(AssertionError("force_new_login should not prompt for reuse/import")),
    )

    _login_openai_codex(SimpleNamespace(), PROVIDER_REGISTRY["openai-codex"], force_new_login=True)

    assert called["device_login"] == 1
    assert called["tokens"]["access_token"] == "fresh-at"


# ---------------------------------------------------------------------------
# Dual-file OAT read (HERMES_DUAL_FILE_OAT_READ flag)
# ---------------------------------------------------------------------------

def _setup_codex_cli_auth(codex_home: Path, *, access_token: str, refresh_token: str = "cli-refresh"):
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": access_token, "refresh_token": refresh_token},
    }))


def test_dual_file_oat_disabled_ignores_fresher_external(tmp_path, monkeypatch):
    """Flag off: even if Codex CLI has a fresher token, Hermes refreshes its own."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    expiring = _jwt_with_exp(int(time.time()) - 10)
    fresher = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=expiring, refresh_token="refresh-old")
    _setup_codex_cli_auth(codex_home, access_token=fresher)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.delenv("HERMES_DUAL_FILE_OAT_READ", raising=False)

    called = {"count": 0}
    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "refreshed-via-post", "refresh_token": "refresh-new"}
    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()
    assert called["count"] == 1, "refresh must fire when flag is off"
    assert resolved["api_key"] == "refreshed-via-post"


def test_dual_file_oat_enabled_adopts_fresher_external(tmp_path, monkeypatch):
    """Flag on + Codex CLI has fresher token → adopt, no refresh."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    expiring = _jwt_with_exp(int(time.time()) - 10)
    fresher = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=expiring, refresh_token="refresh-old")
    _setup_codex_cli_auth(codex_home, access_token=fresher)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("HERMES_DUAL_FILE_OAT_READ", "1")

    called = {"count": 0}
    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "should-not-be-called", "refresh_token": "x"}
    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()
    assert called["count"] == 0, "refresh must NOT fire when external is adopted"
    assert resolved["api_key"] == fresher


def test_dual_file_oat_enabled_ignores_older_external(tmp_path, monkeypatch):
    """Flag on but Codex CLI token is older → fall through to refresh."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    expiring = _jwt_with_exp(int(time.time()) - 10)
    even_older = _jwt_with_exp(int(time.time()) - 3600)
    _setup_hermes_auth(hermes_home, access_token=expiring, refresh_token="refresh-old")
    _setup_codex_cli_auth(codex_home, access_token=even_older)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("HERMES_DUAL_FILE_OAT_READ", "1")

    called = {"count": 0}
    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "refreshed-via-post", "refresh_token": "refresh-new"}
    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()
    assert called["count"] == 1, "refresh must fire when external is not strictly fresher"
    assert resolved["api_key"] == "refreshed-via-post"


def test_dual_file_oat_enabled_no_external_file(tmp_path, monkeypatch):
    """Flag on but ~/.codex/auth.json doesn't exist → fall through to refresh."""
    hermes_home = tmp_path / "hermes"
    expiring = _jwt_with_exp(int(time.time()) - 10)
    _setup_hermes_auth(hermes_home, access_token=expiring, refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nonexistent-codex"))
    monkeypatch.setenv("HERMES_DUAL_FILE_OAT_READ", "1")

    called = {"count": 0}
    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "refreshed-via-post", "refresh_token": "refresh-new"}
    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()
    assert called["count"] == 1, "refresh must fire when no external file"
    assert resolved["api_key"] == "refreshed-via-post"


def test_dual_file_oat_enabled_ignores_near_expired_external(tmp_path, monkeypatch):
    """Flag on, external is fresher than Hermes but itself near expiry → no adoption.

    Prevents ping-ponging: don't adopt a token that's about to need refresh anyway.
    """
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    expired = _jwt_with_exp(int(time.time()) - 100)
    # External is fresher than expired but has only 5s of life left — well under
    # the default refresh skew window — so should NOT be adopted.
    near_expired = _jwt_with_exp(int(time.time()) + 5)
    _setup_hermes_auth(hermes_home, access_token=expired, refresh_token="refresh-old")
    _setup_codex_cli_auth(codex_home, access_token=near_expired)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("HERMES_DUAL_FILE_OAT_READ", "1")

    called = {"count": 0}
    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "refreshed-via-post", "refresh_token": "refresh-new"}
    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()
    assert called["count"] == 1, "refresh must fire when external is also near-expired"
    assert resolved["api_key"] == "refreshed-via-post"


def test_dual_file_oat_adoption_persists_to_hermes_store(tmp_path, monkeypatch):
    """After adopting external tokens, next read from Hermes store sees them."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    expiring = _jwt_with_exp(int(time.time()) - 10)
    fresher = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=expiring, refresh_token="refresh-old")
    _setup_codex_cli_auth(codex_home, access_token=fresher, refresh_token="cli-refresh-xyz")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("HERMES_DUAL_FILE_OAT_READ", "1")

    def _fake_refresh(tokens, timeout_seconds):
        raise AssertionError("refresh must not be called")
    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolve_codex_runtime_credentials()

    # Hermes auth store should now contain the adopted tokens
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == fresher
    assert data["tokens"]["refresh_token"] == "cli-refresh-xyz"

