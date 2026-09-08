import httpx
import pytest
from openai import AuthenticationError, NotFoundError, RateLimitError

from minutegenius import check


def test_check_env_vars_passes_when_all_set(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    check.check_env_vars("FOO")  # ne doit pas lever


def test_check_env_vars_exits_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    with pytest.raises(SystemExit):
        check.check_env_vars("MISSING_VAR")


def _status_error(cls, status_code):
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": "boom"}})
    return cls("boom", response=response, body=None)


class _FakeChatCompletions:
    def __init__(self, error=None):
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return "ok"


class _FakeClient:
    def __init__(self, error=None):
        self.base_url = "https://api.example.com"
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions(error)})()


def test_check_ai_connection_passes_on_success():
    check.check_ai_connection(_FakeClient(), "some-model", "AI_API (vision)")  # ne doit pas lever


def test_check_ai_connection_exits_on_authentication_error():
    client = _FakeClient(_status_error(AuthenticationError, 401))
    with pytest.raises(SystemExit):
        check.check_ai_connection(client, "some-model", "AI_API (vision)")


def test_check_ai_connection_exits_on_not_found_error():
    client = _FakeClient(_status_error(NotFoundError, 404))
    with pytest.raises(SystemExit):
        check.check_ai_connection(client, "some-model", "AI_API (vision)")


def test_check_ai_connection_warns_but_continues_on_rate_limit_error(capsys):
    client = _FakeClient(_status_error(RateLimitError, 429))
    check.check_ai_connection(client, "some-model", "AI_API (vision)")  # ne doit pas lever
    assert "limite de débit" in capsys.readouterr().err


class _FakeResponse:
    def __init__(self, status_code, text="", ok=True):
        self.status_code = status_code
        self.text = text
        self.ok = ok


def test_check_pyannote_connection_passes_on_success(monkeypatch):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    monkeypatch.setattr(check.requests, "get", lambda url, headers=None, timeout=None: _FakeResponse(200))
    check.check_pyannote_connection()  # ne doit pas lever


def test_check_pyannote_connection_exits_on_missing_key(monkeypatch):
    monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        check.check_pyannote_connection()


def test_check_pyannote_connection_exits_on_401(monkeypatch):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    monkeypatch.setattr(check.requests, "get", lambda url, headers=None, timeout=None: _FakeResponse(401, ok=False))
    with pytest.raises(SystemExit):
        check.check_pyannote_connection()


def test_check_pyannote_connection_exits_on_request_exception(monkeypatch):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")

    def raise_connection_error(url, headers=None, timeout=None):
        raise check.requests.RequestException("network down")

    monkeypatch.setattr(check.requests, "get", raise_connection_error)
    with pytest.raises(SystemExit):
        check.check_pyannote_connection()
