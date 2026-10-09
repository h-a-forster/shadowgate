from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from shadowgate.backends.anthropic import AnthropicBackend, classify_error
from shadowgate.backends.base import RetryPolicy, TransientError
from shadowgate.errors import BackendError, ConfigError
from shadowgate.types import Request


def _response(
    content: list[Any] | None = None,
    stop_reason: str = "end_turn",
    model: str = "claude-haiku-5-5",
    usage: dict[str, Any] | None = None,
) -> SimpleNamespace:
    u = usage or {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
    }
    return SimpleNamespace(
        type="message",
        content=content if content is not None else [SimpleNamespace(type="text", text="hi")],
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(**u),
    )


class FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.messages = FakeMessages(outcomes)


# Duck-typed stand-ins shaped like the SDK's exceptions (no SDK needed).
class APIStatusError(Exception):
    def __init__(self, message: str, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status
        self.response = SimpleNamespace(headers=headers or {})


class RateLimitError(APIStatusError):
    pass


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    pass


def _backend(outcomes: list[Any], **kw: Any) -> tuple[AnthropicBackend, FakeClient, list[float]]:
    client = FakeClient(outcomes)
    sleeps: list[float] = []
    kw.setdefault("retry", RetryPolicy(max_attempts=3))
    b = AnthropicBackend("claude-haiku-5-5", client=client, sleep=sleeps.append, **kw)
    return b, client, sleeps


def test_name_and_basic_completion() -> None:
    b, client, _ = _backend([_response()])
    assert b.name == "anthropic:claude-haiku-5-5"
    c = b.complete(Request(prompt="hello"))
    assert c.text == "hi"
    assert c.stop_reason == "end"
    assert c.logprobs is None
    assert c.usage.input_tokens == 1000 and c.usage.output_tokens == 500
    assert c.usage.cache_read_tokens == 0
    assert c.cost_usd == pytest.approx((1000 * 0.10 + 500 * 0.50) / 1e6)
    assert c.latency_s >= 0
    call = client.messages.calls[0]
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["max_tokens"] == 2048
    for absent in ("temperature", "system", "stop_sequences", "output_config", "n_sample"):
        assert absent not in call


def test_name_override() -> None:
    b, _, _ = _backend([], name="fast")
    assert b.name == "fast"


def test_kwargs_mapping() -> None:
    b, client, _ = _backend([_response()])
    req = Request(
        prompt="p",
        system="sys",
        max_tokens=64,
        temperature=0.0,
        effort="low",
        stop=("END",),
        n_sample=3,
        extra={"output_config": {"format": {"type": "json_schema"}}, "metadata": {"x": 1}},
        tags={"task_id": "t1"},
    )
    b.complete(req)
    call = client.messages.calls[0]
    assert call["system"] == "sys"
    assert call["max_tokens"] == 64
    assert call["temperature"] == 0.0
    assert call["stop_sequences"] == ["END"]
    assert call["output_config"] == {"effort": "low", "format": {"type": "json_schema"}}
    assert call["metadata"] == {"x": 1}
    assert "tags" not in call and "n_sample" not in call


def test_thinking_blocks_ignored_and_text_concatenated() -> None:
    content = [
        SimpleNamespace(type="thinking", thinking="secret reasoning"),
        SimpleNamespace(type="text", text="part one "),
        {"type": "redacted_thinking", "data": "xx"},
        {"type": "text", "text": "part two"},
    ]
    b, _, _ = _backend([_response(content=content)])
    assert b.complete(Request(prompt="q")).text == "part one part two"


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("end_turn", "end"),
        ("max_tokens", "max_tokens"),
        ("stop_sequence", "stop_sequence"),
        ("refusal", "refusal"),
        ("pause_turn", "pause_turn"),
    ],
)
def test_stop_reason_normalised(raw: str, norm: str) -> None:
    b, _, _ = _backend([_response(stop_reason=raw)])
    assert b.complete(Request(prompt="q")).stop_reason == norm


def test_refusal_with_no_text() -> None:
    b, _, _ = _backend([_response(content=[], stop_reason="refusal")])
    c = b.complete(Request(prompt="q"))
    assert c.stop_reason == "refusal"
    assert c.text == ""


def test_cache_usage_and_cost() -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 100,
    }
    b, _, _ = _backend([_response(model="claude-opus-5-5", usage=usage)])
    c = b.complete(Request(prompt="q"))
    assert c.usage.cache_read_tokens == 1000
    assert c.usage.cache_write_tokens == 100
    expected = (10 * 4.0 + 20 * 20.0 + 1000 * 0.20 + 100 * 5.0) / 1e6
    assert c.cost_usd == pytest.approx(expected)


def test_unknown_model_cost_is_none() -> None:
    client = FakeClient([_response(model="claude-unknown-9")])
    b = AnthropicBackend("claude-unknown-9", client=client)
    assert b.complete(Request(prompt="q")).cost_usd is None


def test_pricing_override() -> None:
    from shadowgate.pricing import Pricing

    b, _, _ = _backend([_response()], pricing=Pricing(1.0, 1.0))
    assert b.complete(Request(prompt="q")).cost_usd == pytest.approx(1500 / 1e6)


def test_rate_limit_then_success_honours_retry_after() -> None:
    err = RateLimitError("slow down", 429, {"retry-after": "7"})
    b, client, sleeps = _backend([err, _response()])
    c = b.complete(Request(prompt="q"))
    assert c.text == "hi"
    assert sleeps == [7.0]
    assert len(client.messages.calls) == 2


def test_overloaded_exhaustion_is_retryable_backend_error() -> None:
    errs = [APIStatusError("overloaded", 529) for _ in range(3)]
    b, client, sleeps = _backend(errs)
    with pytest.raises(BackendError) as info:
        b.complete(Request(prompt="q"))
    assert info.value.retryable is True
    assert info.value.status == 529
    assert len(client.messages.calls) == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_client_errors_not_retried(status: int) -> None:
    b, client, sleeps = _backend([APIStatusError("bad", status)])
    with pytest.raises(BackendError) as info:
        b.complete(Request(prompt="q"))
    assert info.value.retryable is False
    assert info.value.status == status
    assert len(client.messages.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("exc", [APIConnectionError("down"), APITimeoutError("slow")])
def test_connection_and_timeout_retried(exc: Exception) -> None:
    b, client, _ = _backend([exc, _response()])
    assert b.complete(Request(prompt="q")).text == "hi"
    assert len(client.messages.calls) == 2


def test_unexpected_exception_not_retried() -> None:
    b, client, _ = _backend([ValueError("boom")])
    with pytest.raises(BackendError) as info:
        b.complete(Request(prompt="q"))
    assert info.value.retryable is False
    assert len(client.messages.calls) == 1


def test_classify_error_shapes() -> None:
    t = classify_error(APIStatusError("x", 500), backend="b")
    assert isinstance(t, TransientError) and t.status == 500
    e = classify_error(APIStatusError("x", 422), backend="b")
    assert isinstance(e, BackendError) and not e.retryable


def test_missing_sdk_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("no module named anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ConfigError, match=r"shadowgate-llm\[anthropic\]"):
        AnthropicBackend("claude-haiku-5-5")


def test_missing_custom_key_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.delenv("SHADOWGATE_TEST_MISSING_KEY", raising=False)
    with pytest.raises(ConfigError, match="SHADOWGATE_TEST_MISSING_KEY"):
        AnthropicBackend("claude-haiku-5-5", api_key_env="SHADOWGATE_TEST_MISSING_KEY")


def test_real_client_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.setenv("SHADOWGATE_TEST_KEY", "sk-test-not-a-real-key")
    b = AnthropicBackend(
        "claude-haiku-5-5",
        api_key_env="SHADOWGATE_TEST_KEY",
        base_url="http://127.0.0.1:9",
        timeout_s=5,
    )
    assert b._client.max_retries == 0
    assert "sk-test" not in repr(b)


# --------------------------------------------------------------------------- real SDK classes


def _sdk_objects() -> tuple[Any, Any]:
    anthropic = pytest.importorskip("anthropic")
    import importlib

    http_mod: Any = None
    for mod_name in ("httpx2", "httpx"):  # anthropic 1.x uses httpx2; 0.x used httpx
        try:
            http_mod = importlib.import_module(mod_name)
            break
        except ImportError:
            continue
    if http_mod is None:
        pytest.skip("no HTTP library found for building SDK exceptions")
    try:
        request = http_mod.Request("POST", "https://api.anthropic.com/v1/messages")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot build SDK request objects: {exc}")
    return anthropic, (http_mod, request)


def _status_exc(anthropic: Any, http_mod: Any, request: Any, cls_name: str, status: int,
                headers: dict[str, str] | None = None) -> Exception:
    try:
        response = http_mod.Response(status, headers=headers or {}, request=request)
        return getattr(anthropic, cls_name)("err", response=response, body=None)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot construct anthropic.{cls_name}: {exc}")


def test_real_sdk_exception_mapping() -> None:
    anthropic, (http_mod, request) = _sdk_objects()
    rl = _status_exc(anthropic, http_mod, request, "RateLimitError", 429, {"retry-after": "3"})
    t = classify_error(rl, backend="b")
    assert isinstance(t, TransientError) and t.status == 429 and t.retry_after == 3.0

    ise = _status_exc(anthropic, http_mod, request, "InternalServerError", 500)
    assert isinstance(classify_error(ise, backend="b"), TransientError)

    over = _status_exc(anthropic, http_mod, request, "APIStatusError", 529)
    assert isinstance(classify_error(over, backend="b"), TransientError)

    for name, status in [
        ("BadRequestError", 400),
        ("AuthenticationError", 401),
        ("PermissionDeniedError", 403),
        ("NotFoundError", 404),
    ]:
        e = classify_error(_status_exc(anthropic, http_mod, request, name, status), backend="b")
        assert isinstance(e, BackendError) and e.retryable is False and e.status == status

    try:
        conn = anthropic.APIConnectionError(request=request)
        tmo = anthropic.APITimeoutError(request=request)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot construct connection errors: {exc}")
    assert isinstance(classify_error(conn, backend="b"), TransientError)
    assert isinstance(classify_error(tmo, backend="b"), TransientError)


@pytest.mark.parametrize(
    "response",
    [
        "<html>bad gateway</html>",
        b"\x00",
        {"type": "error", "error": {"message": "x"}},
        SimpleNamespace(type="message", content="not a list"),
        SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")]),  # no type
    ],
)
def test_non_message_response_rejected(response: Any) -> None:
    b, client, sleeps = _backend([response])
    with pytest.raises(BackendError) as ei:
        b.complete(Request(prompt="x"))
    assert ei.value.retryable is False
    assert "unexpected response type" in str(ei.value)
    assert len(client.messages.calls) == 1 and sleeps == []


def test_dict_message_response_accepted() -> None:
    resp = {"type": "message", "model": "claude-haiku-5-5", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}
    b, _, _ = _backend([resp])
    assert b.complete(Request(prompt="x")).text == "ok"


def test_latency_is_final_attempt_only() -> None:
    import time

    class SlowFail:
        def __init__(self) -> None:
            self.n = 0

        def create(self, **kwargs: Any) -> Any:
            self.n += 1
            if self.n == 1:
                time.sleep(0.5)
                raise APIConnectionError("reset")
            return _response()

    client = SimpleNamespace(messages=SlowFail())
    sleeps: list[float] = []
    b = AnthropicBackend("claude-haiku-5-5", client=client, sleep=sleeps.append,
                         retry=RetryPolicy(max_attempts=3))
    c = b.complete(Request(prompt="x"))
    assert client.messages.n == 2 and len(sleeps) == 1
    assert 0 <= c.latency_s < 0.4
