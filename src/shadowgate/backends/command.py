"""Backend that runs an arbitrary command: prompt on stdin, completion text on stdout."""

from __future__ import annotations

import math
import os
import shlex
import subprocess
from collections.abc import Collection, Mapping, Sequence
from pathlib import PurePath
from typing import TYPE_CHECKING

from ..errors import BackendError, ConfigError
from ..types import Completion, Request, Usage
from .base import RetryPolicy, Timer, TransientError, call_with_retries

if TYPE_CHECKING:
    from ..pricing import Pricing

__all__ = ["CommandBackend", "estimate_tokens"]


def estimate_tokens(text: str) -> int:
    """Rough token estimate used when a command reports no usage: ceil(chars / 4)."""
    return math.ceil(len(text) / 4)


def _tail(text: str, limit: int = 500) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


class CommandBackend:
    """Run ``command`` once per request.

    The prompt is written to stdin as UTF-8 (the system prompt, if any, is prepended followed by
    a blank line); stdout, decoded as UTF-8 with replacement of invalid bytes and stripped of
    trailing newlines, is the completion text. A string ``command`` is split with POSIX shell
    rules on POSIX and passed to ``CreateProcess`` unchanged on Windows; no shell is involved.

    A timeout, or a non-zero exit code listed in ``retryable_exit_codes``, is retried; any other
    non-zero exit raises a non-retryable ``BackendError``. Usage is estimated from character
    counts. ``cost_usd`` comes from ``pricing`` when given, else None.
    """

    def __init__(
        self,
        command: Sequence[str] | str,
        *,
        name: str | None = None,
        timeout_s: float = 600,
        retry: RetryPolicy | None = None,
        pricing: Pricing | None = None,
        retryable_exit_codes: Collection[int] = (),
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if isinstance(command, str):
            if not command.strip():
                raise ConfigError("command backend: 'command' must not be empty")
            argv: list[str] | str = command if os.name == "nt" else shlex.split(command)
            first = shlex.split(command, posix=os.name != "nt")[0].strip("\"'")
        else:
            argv = [str(a) for a in command]
            if not argv:
                raise ConfigError("command backend: 'command' must not be empty")
            first = argv[0]
        self.command = argv
        self.name = name or f"command:{PurePath(first).stem or first}"
        self.timeout_s = float(timeout_s)
        self.retry = retry or RetryPolicy()
        self.pricing = pricing
        self.retryable_exit_codes = frozenset(int(c) for c in retryable_exit_codes)
        self.cwd = None if cwd is None else os.fspath(cwd)
        self.env = dict(env) if env is not None else None

    @staticmethod
    def render_input(request: Request) -> str:
        """The exact text written to the command's stdin."""
        if request.system:
            return f"{request.system}\n\n{request.prompt}"
        return request.prompt

    def _call_once(self, request: Request) -> Completion:
        stdin_text = self.render_input(request)
        env = None if self.env is None else {**os.environ, **self.env}
        with Timer() as timer:
            try:
                proc = subprocess.run(
                    self.command,
                    input=stdin_text.encode("utf-8"),
                    capture_output=True,
                    cwd=self.cwd,
                    env=env,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TransientError(f"command timed out after {self.timeout_s:g}s") from exc
            except OSError as exc:
                raise BackendError(
                    f"could not run command: {type(exc).__name__}: {exc}", backend=self.name
                ) from exc
        stdout = proc.stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            stderr = _tail(proc.stderr.decode("utf-8", errors="replace"))
            msg = f"command exited with code {proc.returncode}"
            if stderr:
                msg += f": {stderr}"
            if proc.returncode in self.retryable_exit_codes:
                raise TransientError(msg)
            raise BackendError(msg, backend=self.name)

        text = stdout.rstrip("\r\n")
        usage = Usage(
            input_tokens=estimate_tokens(stdin_text),
            output_tokens=estimate_tokens(text),
        )
        cost = None if self.pricing is None else float(self.pricing.cost(usage))
        return Completion(
            text=text,
            model=self.name,
            usage=usage,
            cost_usd=cost,
            latency_s=timer.elapsed,
            stop_reason="end",
        )

    def complete(self, request: Request) -> Completion:
        return call_with_retries(
            lambda: self._call_once(request), policy=self.retry, backend=self.name
        )

    def __repr__(self) -> str:
        return f"CommandBackend(name={self.name!r})"
