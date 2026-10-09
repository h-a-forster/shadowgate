"""FunctionBackend: wrap a Python callable as a Backend."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..errors import BackendError
from ..types import Completion, Request, Usage
from .base import RetryPolicy, Timer, call_with_retries

if TYPE_CHECKING:
    from ..pricing import Pricing

__all__ = ["FunctionBackend"]


class FunctionBackend:
    """Turn ``fn(request) -> str | Completion`` into a Backend.

    A returned string becomes a Completion with usage estimated as ``ceil(chars / 4)``,
    measured latency, and cost from ``pricing`` (None when no pricing is given). A returned
    Completion is passed through unchanged. ``fn`` may raise ``TransientError`` to be retried
    per ``retry``; other exceptions become BackendError. ``fn`` must be thread-safe.
    """

    def __init__(
        self,
        fn: Callable[[Request], str | Completion],
        *,
        name: str = "function",
        pricing: Pricing | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError("FunctionBackend needs a callable")
        self.fn = fn
        self.name = name
        self.pricing = pricing
        self.retry = retry or RetryPolicy(max_attempts=1)

    def complete(self, request: Request) -> Completion:
        with Timer() as timer:
            out = call_with_retries(
                lambda: self.fn(request), policy=self.retry, backend=self.name
            )
        if isinstance(out, Completion):
            return out
        if not isinstance(out, str):
            raise BackendError(
                f"function returned {type(out).__name__}, expected str or Completion",
                backend=self.name,
            )
        usage = Usage(
            input_tokens=math.ceil((len(request.system or "") + len(request.prompt)) / 4),
            output_tokens=math.ceil(len(out) / 4),
        )
        return Completion(
            text=out,
            model=self.name,
            usage=usage,
            cost_usd=None if self.pricing is None else self.pricing.cost(usage),
            latency_s=timer.elapsed,
        )
