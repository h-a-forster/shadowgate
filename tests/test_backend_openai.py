from __future__ import annotations

import contextlib
import json
import math
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from shadowgate.backends.base import RetryPolicy
from shadowgate.backends.openai_compat import OpenAICompatBackend, parse_retry_after
from shadowgate.errors import BackendError, ConfigError
from shadowgate.pricing import Pricing
from shadowgate.types import Request


class Script:
    """Scripted responses: (status, headers, body) tuples consumed in order; last one repeats."""

    def __init__(self) -> None:
        self.responses: list[tuple[int, dict[str, str], Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def next(self) -> tuple[int, dict[str, str], Any]:
        with self.lock:
            if len(self.responses) > 1:
                return self.responses.pop(0)
            return self.responses[0]


def _ok_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "id": "chatcmpl-1",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ANSWER: 4"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }
    body.update(overrides)
    return body


@pytest.fixture()
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, Script]]:
    # Keep system/registry proxy settings away from the loopback test server.
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    script = Script()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            script.requests.append(
                {"path": self.path, "headers": dict(self.headers), "body": json.loads(raw)}
            )
            status, headers, body = script.next()
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", script
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _backend(base_url: str, **kw: Any) -> tuple[OpenAICompatBackend, list[float]]:
    sleeps: list[float] = []
    kw.setdefault("retry", RetryPolicy(max_attempts=3))
    kw.setdefault("api_key_env", "SHADOWGATE_TEST_OPENAI_KEY")
    kw.setdefault("timeout_s", 10)
    return OpenAICompatBackend("test-model", base_url=base_url, sleep=sleeps.append, **kw), sleeps


def test_success(server: tuple[str, Script], monkeypatch: pytest.MonkeyPatch) -> None:
    url, script = server
    script.responses = [(200, {}, _ok_body())]
    monkeypatch.setenv("SHADOWGATE_TEST_OPENAI_KEY", "sk-test-123")
    b, sleeps = _backend(url + "/", pricing=Pricing(1.0, 2.0))
    assert b.name == "openai:test-model"
    c = b.complete(Request(prompt="2+2?", system="be brief", max_tokens=16, n_sample=2,
                           tags={"task_id": "t"}))
    assert c.text == "ANSWER: 4"
    assert c.stop_reason == "end"
    assert c.usage.input_tokens == 12 and c.usage.output_tokens == 5
    assert c.cost_usd == pytest.approx((12 * 1.0 + 5 * 2.0) / 1e6)
    assert c.logprobs is None
    assert c.latency_s > 0
    assert sleeps == []
    req = script.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-test-123"
    body = req["body"]
    assert body["model"] == "test-model"
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "2+2?"},
    ]
    assert body["max_tokens"] == 16
    for absent in ("temperature", "logprobs", "stop", "n_sample", "tags"):
        assert absent not in body


def test_no_key_no_auth_header_and_extra(
    server: tuple[str, Script], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, script = server
    script.responses = [(200, {}, _ok_body())]
    monkeypatch.delenv("SHADOWGATE_TEST_OPENAI_KEY", raising=False)
    b, _ = _backend(url, headers={"X-Extra": "1"}, name="local")
    assert b.name == "local"
    b.complete(Request(prompt="q", temperature=0.5, stop=("\n\n",),
                       extra={"max_completion_tokens": 99, "seed": 1}))
    req = script.requests[0]
    assert "Authorization" not in req["headers"]
    assert req["headers"]["X-Extra"] == "1"
    body = req["body"]
    assert body["temperature"] == 0.5
    assert body["stop"] == ["\n\n"]
    assert body["max_completion_tokens"] == 99 and "max_tokens" not in body
    assert body["seed"] == 1


def test_logprobs(server: tuple[str, Script]) -> None:
    url, script = server
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "4"},
        "finish_reason": "length",
        "logprobs": {"content": [{"token": "4", "logprob": -0.1, "top_logprobs": []},
                                 {"token": ".", "logprob": -0.5, "top_logprobs": []}]},
    }
    usage = {"prompt_tokens": 10, "completion_tokens": 2,
             "prompt_tokens_details": {"cached_tokens": 4}}
    script.responses = [(200, {}, _ok_body(choices=[choice], usage=usage))]
    b, _ = _backend(url)
    c = b.complete(Request(prompt="q", want_logprobs=True))
    assert script.requests[0]["body"]["logprobs"] is True
    assert c.logprobs == (-0.1, -0.5)
    assert math.exp(sum(c.logprobs) / 2) < 1
    assert c.stop_reason == "max_tokens"
    assert c.usage.input_tokens == 6 and c.usage.cache_read_tokens == 4
    assert c.cost_usd is None  # unknown model, no override


def test_429_retry_after_then_success(server: tuple[str, Script]) -> None:
    url, script = server
    script.responses = [
        (429, {"Retry-After": "2"}, {"error": {"message": "rate limited"}}),
        (200, {}, _ok_body()),
    ]
    b, sleeps = _backend(url)
    c = b.complete(Request(prompt="q"))
    assert c.text == "ANSWER: 4"
    assert sleeps == [2.0]
    assert len(script.requests) == 2


def test_500_exhaustion(server: tuple[str, Script]) -> None:
    url, script = server
    script.responses = [(500, {}, {"error": {"message": "internal"}})]
    b, sleeps = _backend(url)
    with pytest.raises(BackendError) as info:
        b.complete(Request(prompt="q"))
    assert info.value.retryable is True
    assert info.value.status == 500
    assert "internal" in str(info.value)
    assert len(script.requests) == 3
    assert len(sleeps) == 2


def test_400_not_retried(server: tuple[str, Script], monkeypatch: pytest.MonkeyPatch) -> None:
    url, script = server
    monkeypatch.setenv("SHADOWGATE_TEST_OPENAI_KEY", "sk-secret-xyz")
    script.responses = [(400, {}, {"error": {"message": "bad param"}})]
    b, sleeps = _backend(url)
    with pytest.raises(BackendError) as info:
        b.complete(Request(prompt="q"))
    assert info.value.retryable is False
    assert info.value.status == 400
    assert "bad param" in str(info.value)
    assert "sk-secret" not in str(info.value)
    assert len(script.requests) == 1
    assert sleeps == []


def test_invalid_json_response(server: tuple[str, Script]) -> None:
    url, script = server
    script.responses = [(200, {}, b"not json")]
    b, _ = _backend(url)
    with pytest.raises(BackendError) as info:
        b.complete(Request(prompt="q"))
    assert info.value.retryable is False


def test_refusal_and_provider_cost(server: tuple[str, Script]) -> None:
    url, script = server
    choice = {"index": 0, "message": {"role": "assistant", "content": None,
                                      "refusal": "cannot help"}, "finish_reason": "stop"}
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.0012}
    script.responses = [(200, {}, _ok_body(choices=[choice], usage=usage))]
    b, _ = _backend(url)
    c = b.complete(Request(prompt="q"))
    assert c.stop_reason == "refusal"
    assert c.text == "cannot help"
    assert c.cost_usd == pytest.approx(0.0012)


def test_connection_refused_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_PROXY", "*")
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    b, sleeps = _backend(f"http://127.0.0.1:{port}/v1", retry=RetryPolicy(max_attempts=2,
                                                                          base_delay_s=0))
    with pytest.raises(BackendError) as info:
        b.complete(Request(prompt="q"))
    assert info.value.retryable is True
    assert len(sleeps) == 1


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x", "not a url"])
def test_bad_base_url(url: str) -> None:
    with pytest.raises(ConfigError):
        OpenAICompatBackend("m", base_url=url)


def test_parse_retry_after() -> None:
    assert parse_retry_after({"retry-after": "3"}) == 3.0
    assert parse_retry_after({"Retry-After": "1.5"}) == 1.5
    assert parse_retry_after({"retry-after-ms": "250"}) == 0.25
    assert parse_retry_after({"retry-after": "garbage"}) is None
    assert parse_retry_after({}) is None
    assert parse_retry_after(None) is None
    past = parse_retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
    assert past == 0.0


# ---------------------------------------------------------------------- hardening regressions


class _Server:
    """Loopback server running ``handle(handler)`` for every request (GET and POST)."""

    def __init__(self, handle: Any) -> None:
        self.seen: list[tuple[str, str, str | None]] = []
        seen = self.seen

        class Handler(BaseHTTPRequestHandler):
            def _any(self) -> None:
                seen.append((self.command, self.path, self.headers.get("Authorization")))
                n = int(self.headers.get("Content-Length", "0") or 0)
                if n:
                    self.rfile.read(n)
                handle(self)

            do_GET = do_POST = _any  # noqa: N815

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def __enter__(self) -> _Server:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _send_json(h: BaseHTTPRequestHandler, body: Any, status: int = 200) -> None:
    payload = json.dumps(body).encode()
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(payload)))
    h.end_headers()
    h.wfile.write(payload)


@pytest.fixture()
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirect_refused_and_key_not_leaked(
    code: int, no_proxy: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHADOWGATE_TEST_OPENAI_KEY", "sk-secret-redirect")
    with _Server(lambda h: _send_json(h, _ok_body())) as other:
        target = f"http://localhost:{other.port}/collect"

        def redirect(h: BaseHTTPRequestHandler) -> None:
            h.send_response(code)
            h.send_header("Location", target)
            h.send_header("Content-Length", "0")
            h.end_headers()

        with _Server(redirect) as api:
            b, sleeps = _backend(f"http://127.0.0.1:{api.port}/v1")
            with pytest.raises(BackendError) as ei:
                b.complete(Request(prompt="hi"))
        assert other.seen == []  # the redirect target never got a request (or the key)
    err = ei.value
    assert err.retryable is False
    assert f"localhost:{other.port}" in str(err)
    assert "redirect" in str(err)
    assert "sk-secret-redirect" not in str(err)
    assert len(api.seen) == 1 and sleeps == []


def test_overall_timeout_on_trickling_body(no_proxy: None) -> None:
    import time

    def trickle(h: BaseHTTPRequestHandler) -> None:
        h.send_response(200)
        h.send_header("Content-Length", "1000")
        h.end_headers()
        try:
            for _ in range(10):
                h.wfile.write(b" ")
                h.wfile.flush()
                time.sleep(0.3)
        except OSError:
            pass

    with _Server(trickle) as api:
        b, _ = _backend(f"http://127.0.0.1:{api.port}/v1", timeout_s=1,
                        retry=RetryPolicy(max_attempts=1))
        t0 = time.monotonic()
        with pytest.raises(BackendError) as ei:
            b.complete(Request(prompt="hi"))
        elapsed = time.monotonic() - t0
    assert ei.value.retryable is True
    assert "timeout" in str(ei.value)
    assert elapsed < 2.0


@pytest.mark.parametrize("with_length", [True, False])
def test_response_size_cap(
    with_length: bool, no_proxy: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shadowgate.backends.openai_compat as oc

    monkeypatch.setattr(oc, "MAX_RESPONSE_BYTES", 1000)
    big = json.dumps(_ok_body(padding="x" * 5000)).encode()

    def respond(h: BaseHTTPRequestHandler) -> None:
        h.send_response(200)
        if with_length:
            h.send_header("Content-Length", str(len(big)))
        h.end_headers()
        with contextlib.suppress(OSError):
            h.wfile.write(big)

    with _Server(respond) as api:
        b, sleeps = _backend(f"http://127.0.0.1:{api.port}/v1")
        with pytest.raises(BackendError) as ei:
            b.complete(Request(prompt="hi"))
    assert ei.value.retryable is False
    assert "larger than" in str(ei.value)
    assert sleeps == []


def test_latency_is_final_attempt_only(no_proxy: None) -> None:
    import time

    calls = {"n": 0}

    def respond(h: BaseHTTPRequestHandler) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.5)
            _send_json(h, {"error": {"message": "busy"}}, status=503)
        else:
            _send_json(h, _ok_body())

    with _Server(respond) as api:
        b, sleeps = _backend(f"http://127.0.0.1:{api.port}/v1")
        c = b.complete(Request(prompt="hi"))
    assert calls["n"] == 2 and len(sleeps) == 1
    assert 0 < c.latency_s < 0.4
