"""OpenAI-compatible chat/completions backend (OpenAI, Ollama, vLLM, OpenRouter, ...).

Uses only the standard library (``urllib``). The API key is optional so local servers work
without one; it is read from the environment variable named by ``api_key_env`` and is never
logged or included in error messages.
"""

from __future__ import annotations

import contextlib
import email.utils
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from ..errors import BackendError, ConfigError
from ..pricing import Pricing, cost_of
from ..types import Completion, Request, Usage
from .base import RETRYABLE_STATUS, RetryPolicy, Timer, TransientError, call_with_retries

__all__ = ["MAX_RESPONSE_BYTES", "OpenAICompatBackend", "parse_retry_after"]

_FINISH = {
    "stop": "end",
    "length": "max_tokens",
    "content_filter": "refusal",
}

_MAX_ERROR_CHARS = 500
_MAX_ERROR_BODY_BYTES = 64 * 1024
_CHUNK = 64 * 1024

#: Responses larger than this are rejected (non-retryable BackendError).
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class _RedirectRefused(Exception):
    def __init__(self, code: int, target_host: str) -> None:
        super().__init__(f"redirect ({code}) to {target_host}")
        self.code = code
        self.target_host = target_host


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: following one would re-send the Authorization header (urllib
    keeps it for any host) and turn the POST into a GET whose answer is not ours."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        try:
            host = urllib.parse.urlsplit(urllib.parse.urljoin(req.full_url, newurl)).netloc
        except ValueError:
            host = ""
        host = host.rpartition("@")[2]  # never echo userinfo
        raise _RedirectRefused(code, host or "?")


def _opener() -> urllib.request.OpenerDirector:
    # Built per request so proxy environment variables are read at call time, as urlopen does.
    return urllib.request.build_opener(_NoRedirect)


def _set_read_timeout(resp: Any, seconds: float) -> None:
    """Best effort: shrink the socket timeout of an http.client response to ``seconds``."""
    with contextlib.suppress(AttributeError, OSError, ValueError):
        resp.fp.raw._sock.settimeout(max(0.001, seconds))


def parse_retry_after(headers: Any) -> float | None:
    """Seconds to wait from ``retry-after-ms`` / ``retry-after`` headers (seconds or HTTP date).

    ``headers`` is anything with a ``get`` method (``email.message.Message``, httpx headers, a
    dict). Returns None when absent or unparseable.
    """
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None

    def _get(name: str) -> Any:
        value = getter(name)
        if value is None:
            value = getter(name.title())
        return value

    ms = _get("retry-after-ms")
    if ms is not None:
        try:
            value = float(ms) / 1000.0
        except (TypeError, ValueError):
            value = -1.0
        if value >= 0:
            return value
    raw = _get("retry-after")
    if raw is None:
        return None
    text = str(raw).strip()
    try:
        value = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        value = when.timestamp() - time.time()
        return max(0.0, value)
    return value if value >= 0 else None


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_ERROR_CHARS else text[: _MAX_ERROR_CHARS - 3] + "..."


def _error_message(body: bytes) -> str:
    """Best-effort human message from an error response body."""
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return _truncate(body.decode("utf-8", errors="replace"))
    if isinstance(data, Mapping):
        err = data.get("error")
        if isinstance(err, Mapping) and err.get("message"):
            return _truncate(str(err["message"]))
        if isinstance(err, str):
            return _truncate(err)
        if data.get("message"):
            return _truncate(str(data["message"]))
    return _truncate(json.dumps(data))


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") in ("text", "output_text"):
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class OpenAICompatBackend:
    """POST ``{base_url}/chat/completions`` with stdlib urllib.

    ``Request.effort`` is ignored (not part of the common chat/completions surface); pass
    provider-specific fields such as ``reasoning_effort`` through ``Request.extra``.

    Redirects are never followed (non-retryable BackendError naming the target host), so the
    API key is only ever sent to ``base_url``. ``timeout_s`` bounds each attempt overall
    (connect, headers and body); bodies over ``MAX_RESPONSE_BYTES`` are rejected. ``latency_s``
    is the wall time of the final, successful attempt (failed attempts and backoff sleeps
    excluded).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout_s: float = 600,
        retry: RetryPolicy | None = None,
        pricing: Pricing | None = None,
        headers: Mapping[str, str] | None = None,
        name: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model:
            raise ConfigError("openai backend: model is required")
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ConfigError(f"openai backend: base_url must be an http(s) URL, got {base_url!r}")
        if timeout_s <= 0:
            raise ConfigError("openai backend: timeout_s must be positive")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)
        self.retry = retry or RetryPolicy()
        self.pricing = pricing
        self.headers = dict(headers or {})
        self.name = name or f"openai:{model}"
        self._sleep = sleep

    # ------------------------------------------------------------------ request building

    def build_body(self, request: Request) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if "max_completion_tokens" not in request.extra:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.stop:
            body["stop"] = list(request.stop)
        if request.want_logprobs:
            body["logprobs"] = True
        body.update(dict(request.extra))
        return body

    def _headers(self) -> dict[str, str]:
        hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if key:
                hdrs["Authorization"] = f"Bearer {key}"
        hdrs.update(self.headers)
        return hdrs

    # ------------------------------------------------------------------ transport

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers=self._headers(), method="POST")
        deadline = time.monotonic() + self.timeout_s
        try:
            with _opener().open(req, timeout=self.timeout_s) as resp:
                raw = self._read_body(resp, deadline)
        except _RedirectRefused as exc:
            raise BackendError(
                f"refusing to follow HTTP {exc.code} redirect to {exc.target_host!r}; set "
                "base_url to the final endpoint",
                backend=self.name,
                status=exc.code,
                retryable=False,
            ) from None
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                err_body = exc.read(_MAX_ERROR_BODY_BYTES)
            except OSError:
                err_body = b""
            msg = f"HTTP {status}: {_error_message(err_body) or exc.reason}"
            if status in RETRYABLE_STATUS or status >= 500:
                raise TransientError(
                    msg, status=status, retry_after=parse_retry_after(exc.headers)
                ) from None
            raise BackendError(msg, backend=self.name, status=status, retryable=False) from None
        except TimeoutError as exc:
            raise TransientError(f"timeout after {self.timeout_s:g}s: {exc}") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TransientError(f"timeout after {self.timeout_s:g}s") from None
            raise TransientError(f"connection error: {exc.reason}") from None
        except (ConnectionError, http.client.HTTPException) as exc:
            raise TransientError(f"connection error: {type(exc).__name__}: {exc}") from None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BackendError(
                f"invalid JSON in response: {exc}", backend=self.name, retryable=False
            ) from None
        if not isinstance(parsed, dict):
            raise BackendError("response is not a JSON object", backend=self.name)
        if "error" in parsed and "choices" not in parsed:
            raise BackendError(
                f"provider error: {_error_message(raw)}", backend=self.name, retryable=False
            )
        return parsed

    def _read_body(self, resp: Any, deadline: float) -> bytes:
        """Read the response body within the overall deadline and the size cap."""
        headers = getattr(resp, "headers", None)
        length = headers.get("Content-Length") if headers is not None else None
        try:
            too_large = length is not None and int(length) > MAX_RESPONSE_BYTES
        except ValueError:
            too_large = False
        if too_large:
            raise self._too_large()
        chunks: list[bytes] = []
        size = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("response body not received in time")
            _set_read_timeout(resp, remaining)
            chunk = resp.read1(_CHUNK) if hasattr(resp, "read1") else resp.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise self._too_large()
            chunks.append(chunk)
        return b"".join(chunks)

    def _too_large(self) -> BackendError:
        return BackendError(
            f"response larger than {MAX_RESPONSE_BYTES // (1024 * 1024)} MiB",
            backend=self.name,
            retryable=False,
        )

    # ------------------------------------------------------------------ parsing

    def parse(self, data: Mapping[str, Any], latency_s: float) -> Completion:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], Mapping):
            raise BackendError("response has no choices", backend=self.name)
        choice = choices[0]
        message = choice.get("message") or {}
        text = _content_text(message.get("content"))
        finish = choice.get("finish_reason")
        if message.get("refusal"):
            stop_reason = "refusal"
            if not text:
                text = str(message["refusal"])
        elif finish is None:
            stop_reason = "end"
        else:
            stop_reason = _FINISH.get(str(finish), str(finish))

        logprobs: tuple[float, ...] | None = None
        lp = choice.get("logprobs")
        if isinstance(lp, Mapping) and isinstance(lp.get("content"), list):
            vals = [
                float(tok["logprob"])
                for tok in lp["content"]
                if isinstance(tok, Mapping) and isinstance(tok.get("logprob"), (int, float))
            ]
            logprobs = tuple(vals)

        u = data.get("usage") or {}
        prompt_tokens = _int(u.get("prompt_tokens"))
        details = u.get("prompt_tokens_details") or {}
        cached = _int(details.get("cached_tokens")) if isinstance(details, Mapping) else 0
        cached = min(cached, prompt_tokens)
        usage = Usage(
            input_tokens=prompt_tokens - cached,
            output_tokens=_int(u.get("completion_tokens")),
            cache_read_tokens=cached,
        )
        cost: float | None
        if self.pricing is not None:
            cost = self.pricing.cost(usage)
        else:
            reported = u.get("cost")
            if isinstance(reported, (int, float)) and not isinstance(reported, bool):
                cost = float(reported)
            else:
                cost = cost_of(self.model, usage)
        return Completion(
            text=text,
            model=str(data.get("model") or self.model),
            usage=usage,
            cost_usd=cost,
            latency_s=latency_s,
            stop_reason=stop_reason,
            logprobs=logprobs,
        )

    # ------------------------------------------------------------------ public

    def complete(self, request: Request) -> Completion:
        body = self.build_body(request)

        def attempt() -> tuple[dict[str, Any], float]:
            with Timer() as timer:
                data = self._post(body)
            return data, timer.elapsed

        data, latency = call_with_retries(
            attempt, policy=self.retry, backend=self.name, sleep=self._sleep
        )
        return self.parse(data, latency)

    def __repr__(self) -> str:
        return f"OpenAICompatBackend(name={self.name!r}, url={self.url!r})"

