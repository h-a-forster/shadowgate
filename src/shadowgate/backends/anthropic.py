"""Anthropic Messages API backend using the official ``anthropic`` SDK (optional extra).

The SDK is imported lazily; when it is missing a ``ConfigError`` with an install hint is raised.
SDK-internal retries are disabled (``max_retries=0``) so shadowgate's ``RetryPolicy`` is the only
retry layer. API keys are read from the environment and never logged or put in error messages.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from ..errors import BackendError, ConfigError
from ..pricing import Pricing, cost_of
from ..types import Completion, Request, Usage
from .base import RETRYABLE_STATUS, RetryPolicy, Timer, TransientError, call_with_retries
from .openai_compat import parse_retry_after

__all__ = ["AnthropicBackend", "INSTALL_HINT"]

INSTALL_HINT = 'pip install "shadowgate-llm[anthropic]"'
_DEFAULT_KEY_ENV = "ANTHROPIC_API_KEY"
_MAX_ERROR_CHARS = 500

_STOP = {
    "end_turn": "end",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "refusal": "refusal",
}

# SDK exception class names that signal a transient transport failure (no HTTP status).
# APITimeoutError subclasses APIConnectionError in the SDK; both names are listed so that
# duck-typed stand-ins are recognized as well.
_TRANSIENT_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an SDK model object or a plain mapping."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _truncate(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _MAX_ERROR_CHARS else text[: _MAX_ERROR_CHARS - 3] + "..."


def _make_client(api_key_env: str, base_url: str | None, timeout_s: float) -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ConfigError(
            f"the anthropic backend needs the 'anthropic' package: {INSTALL_HINT}"
        ) from exc
    kwargs: dict[str, Any] = {"max_retries": 0, "timeout": timeout_s}
    key = os.environ.get(api_key_env) if api_key_env else None
    if key:
        kwargs["api_key"] = key
    elif api_key_env and api_key_env != _DEFAULT_KEY_ENV:
        raise ConfigError(f"anthropic backend: environment variable {api_key_env} is not set")
    # With the default variable unset, the SDK resolves credentials itself (e.g. other
    # supported environment variables or a stored login).
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return anthropic.Anthropic(**kwargs)
    except Exception as exc:  # noqa: BLE001 - SDK raises its own error types for bad config
        raise ConfigError(f"anthropic backend: could not create client: {type(exc).__name__}") \
            from None


def classify_error(exc: BaseException, *, backend: str) -> Exception:
    """Map an SDK (or SDK-shaped) exception to TransientError or a non-retryable BackendError."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    message = _truncate(getattr(exc, "message", None) or str(exc) or type(exc).__name__)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        text = f"{type(exc).__name__} (HTTP {status}): {message}"
        if status in RETRYABLE_STATUS or status >= 500 or "RateLimitError" in names:
            return TransientError(text, status=status, retry_after=parse_retry_after(headers))
        return BackendError(text, backend=backend, status=status, retryable=False)
    if names & _TRANSIENT_NAMES or isinstance(exc, (TimeoutError, ConnectionError)):
        return TransientError(f"{type(exc).__name__}: {message}")
    return BackendError(f"{type(exc).__name__}: {message}", backend=backend, retryable=False)


class AnthropicBackend:
    """Claude via ``client.messages.create``.

    ``temperature`` is sent only when set (current Claude models reject sampling parameters);
    ``effort`` goes in ``output_config``; ``Request.extra`` is merged into the call kwargs
    (``output_config`` dicts are merged key by key). Only ``text`` blocks contribute to the
    completion text; ``thinking`` and other blocks are ignored. Claude does not expose logprobs,
    so ``Completion.logprobs`` is always None.

    A response that is not message-shaped (``type == "message"`` with a list ``content``), such
    as the plain string the SDK returns for a non-JSON 200 from a misconfigured ``base_url``,
    raises a non-retryable BackendError. ``latency_s`` is the wall time of the final, successful
    attempt (failed attempts and backoff sleeps excluded).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = _DEFAULT_KEY_ENV,
        base_url: str | None = None,
        timeout_s: float = 600,
        retry: RetryPolicy | None = None,
        pricing: Pricing | None = None,
        client: Any = None,
        name: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model:
            raise ConfigError("anthropic backend: model is required")
        if timeout_s <= 0:
            raise ConfigError("anthropic backend: timeout_s must be positive")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.retry = retry or RetryPolicy()
        self.pricing = pricing
        self.name = name or f"anthropic:{model}"
        self._sleep = sleep
        self._client = (
            client if client is not None else _make_client(api_key_env, base_url, self.timeout_s)
        )

    def build_kwargs(self, request: Request) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = request.system
        if request.stop:
            kwargs["stop_sequences"] = list(request.stop)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.effort:
            kwargs["output_config"] = {"effort": request.effort}
        for key, value in request.extra.items():
            current = kwargs.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                kwargs[key] = {**current, **value}
            else:
                kwargs[key] = value
        return kwargs

    def _call(self, kwargs: dict[str, Any]) -> Any:
        try:
            return self._client.messages.create(**kwargs)
        except (TransientError, BackendError):
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            raise classify_error(exc, backend=self.name) from None

    def parse(self, response: Any, latency_s: float) -> Completion:
        content = None if isinstance(response, (str, bytes)) else _get(response, "content", None)
        if (
            isinstance(response, (str, bytes))
            or _get(response, "type", None) != "message"
            or not isinstance(content, (list, tuple))
        ):
            if isinstance(response, (str, bytes)):
                detail = f"{type(response).__name__} {_truncate(repr(response[:200]))}"
            else:
                detail = type(response).__name__
            raise BackendError(
                f"unexpected response type from the Messages API ({detail}); check base_url",
                backend=self.name,
                retryable=False,
            )
        parts: list[str] = []
        for block in content:
            if _get(block, "type") == "text":
                parts.append(str(_get(block, "text", "") or ""))
        raw_stop = _get(response, "stop_reason")
        stop_reason = "end" if raw_stop is None else _STOP.get(str(raw_stop), str(raw_stop))
        u = _get(response, "usage", None) or {}
        usage = Usage(
            input_tokens=_int(_get(u, "input_tokens")),
            output_tokens=_int(_get(u, "output_tokens")),
            cache_read_tokens=_int(_get(u, "cache_read_input_tokens")),
            cache_write_tokens=_int(_get(u, "cache_creation_input_tokens")),
        )
        model = str(_get(response, "model", None) or self.model)
        cost = cost_of(model, usage, self.pricing)
        if cost is None and model != self.model:
            cost = cost_of(self.model, usage)
        return Completion(
            text="".join(parts),
            model=model,
            usage=usage,
            cost_usd=cost,
            latency_s=latency_s,
            stop_reason=stop_reason,
            logprobs=None,
        )

    def complete(self, request: Request) -> Completion:
        kwargs = self.build_kwargs(request)

        def attempt() -> tuple[Any, float]:
            with Timer() as timer:
                response = self._call(kwargs)
            return response, timer.elapsed

        response, latency = call_with_retries(
            attempt, policy=self.retry, backend=self.name, sleep=self._sleep
        )
        return self.parse(response, latency)

    def __repr__(self) -> str:
        return f"AnthropicBackend(name={self.name!r})"
