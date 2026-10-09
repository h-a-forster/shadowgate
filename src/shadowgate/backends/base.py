"""Shared plumbing for backends: retry with backoff, timing, and request hashing."""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from ..errors import BackendError
from ..types import Request

T = TypeVar("T")

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class TransientError(Exception):
    """Raised inside a backend call to signal a retryable failure.

    ``retry_after`` (seconds) is honoured when the provider supplied one.
    """

    def __init__(self, message: str, *, status: int | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter, capped, honouring Retry-After."""

    max_attempts: int = 6
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    max_retry_after_s: float = 300.0

    def delay(self, attempt: int, retry_after: float | None, rng: random.Random) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_retry_after_s)
        cap = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        return rng.uniform(0, cap)


_rng_lock = threading.Lock()
_rng = random.Random()


def call_with_retries(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    backend: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``fn``; retry on TransientError per ``policy``; raise BackendError at the end.

    Any non-TransientError exception that is already a BackendError propagates unchanged
    (non-retryable). Other exceptions are wrapped as non-retryable BackendErrors.
    """
    last: TransientError | None = None
    for attempt in range(policy.max_attempts):
        try:
            return fn()
        except TransientError as exc:
            last = exc
            if attempt == policy.max_attempts - 1:
                break
            with _rng_lock:
                d = policy.delay(attempt, exc.retry_after, _rng)
            sleep(d)
        except BackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise unexpected failures
            raise BackendError(f"{type(exc).__name__}: {exc}", backend=backend) from exc
    assert last is not None
    raise BackendError(
        f"gave up after {policy.max_attempts} attempts: {last}",
        backend=backend,
        status=last.status,
        retryable=True,
    ) from last


def request_key(backend_name: str, request: Request) -> str:
    """Stable cache key over everything that affects the wire request (tags excluded)."""
    payload = request.to_dict()
    payload.pop("tags", None)
    payload["__backend__"] = backend_name
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Timer:
    """Context manager measuring wall-clock seconds with perf_counter."""

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0
