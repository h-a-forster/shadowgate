"""Backend that runs the Claude Code CLI in print mode.

This lets a cascade run on a Claude Code login (subscription or API key configured in the CLI)
without the ``anthropic`` SDK. Each call starts ``claude -p --output-format json`` in a fresh,
empty temporary directory with tools, settings files, slash commands and MCP servers disabled,
sends the prompt on stdin, and parses the single JSON result object.

Request fields the CLI has no flag for (``max_tokens``, ``temperature``, ``stop``,
``want_logprobs``, ``extra``) are ignored. ``effort`` is passed with ``--effort``.

When the executable resolves to a ``.cmd``/``.bat`` shim on Windows, cmd.exe re-parses the
command line (a newline ends the command, ``&``/``|``/``%`` are interpreted), so no free text is
ever put on it: the system prompt goes through ``--system-prompt-file`` (a temporary file outside
the working directory), or in-band at the top of stdin when even that path is not a plain token.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import BackendError, ConfigError
from ..pricing import Pricing, cost_with_cache_ttl, lookup
from ..types import Completion, Request, Usage
from ._proc import ProcessTimeout, run_process
from .base import RETRYABLE_STATUS, RetryPolicy, Timer, TransientError, call_with_retries

__all__ = ["ClaudeCodeBackend", "DEFAULT_ISOLATION_ARGS", "DEFAULT_SYSTEM"]

log = logging.getLogger("shadowgate.backends.claude_code")

#: Flags that keep a print-mode run self-contained: no built-in tools (so the answer is a single
#: turn), no user/project/local settings files, no saved session, no skills/slash commands and
#: no MCP servers beyond those passed explicitly (none). Override via ``isolation_args`` if a CLI
#: version lacks one of them.
DEFAULT_ISOLATION_ARGS: tuple[str, ...] = (
    "--tools", "",
    "--setting-sources", "",
    "--no-session-persistence",
    "--disable-slash-commands",
    "--strict-mcp-config",
)

#: Replaces the CLI's agentic default system prompt when a request has no system prompt.
DEFAULT_SYSTEM = "You are a helpful assistant."

# System prompts longer than this are prepended to the stdin prompt instead of being passed on
# the command line (Windows limits a command line to 32767 characters).
_MAX_ARGV_SYSTEM_CHARS = 8000

# Values placed on the command line must be plain tokens (always matched with fullmatch: "$"
# would accept a trailing newline): a .cmd/.bat shim on Windows routes arguments through cmd.exe,
# which would interpret metacharacters and stop at a newline.
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.:@/\[\]-]+")
# A temporary file path that may be passed through a shim (spaces are quoted by list2cmdline).
_SHIM_SAFE_PATH = re.compile(r"[A-Za-z0-9 _.:\\/~-]+")

# Cost bases reported in the CLI's ``modelUsage`` that mean "priced from a known rate card".
# The CLI reports "unknown" (and prints "[claude-code:unrecognized_model]" on stderr) when it
# does not recognise the model and falls back to a default, often much higher, price.
_UNTRUSTED_COST_BASES = frozenset({"", "unknown"})
_UNRECOGNIZED_MODEL_MARK = "unrecognized_model"

_STOP_REASONS = {
    "end_turn": "end",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "refusal": "refusal",
}

_AUTH_RE = re.compile(
    r"log ?in|logged in|authenticat|api key|credential|unauthori[sz]ed|forbidden|oauth",
    re.IGNORECASE,
)
_TRANSIENT_RE = re.compile(
    r"rate.?limit|overload|timed? ?out|timeout|temporarily|try again|econnreset|econnrefused|"
    r"connection|network|\b(?:408|409|425|429|500|502|503|504|529)\b",
    re.IGNORECASE,
)


def _tail(text: str, limit: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def _parse_result(stdout: str) -> dict[str, Any] | None:
    """Return the CLI's JSON result object, tolerating stray non-JSON lines around it."""
    stdout = stdout.strip()
    if not stdout:
        return None
    try:
        obj = json.loads(stdout)
    except json.JSONDecodeError:
        obj = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    return obj if isinstance(obj, dict) else None


def _int(d: Mapping[str, Any], key: str) -> int:
    try:
        return int(d.get(key) or 0)
    except (TypeError, ValueError):
        return 0


class ClaudeCodeBackend:
    """Run requests through the Claude Code CLI (``claude -p --output-format json``).

    ``executable`` is a program name resolved with ``shutil.which`` (``claude`` finds
    ``claude.exe`` or a ``claude.cmd`` shim on Windows) or an explicit argv prefix such as
    ``[sys.executable, "fake_claude.py"]``, used as-is. ``isolation_args`` and ``extra_args``
    are appended after the core flags. ``default_system`` replaces the CLI's own system prompt
    when a request has none (None keeps the CLI default).

    ``cost_usd`` is resolved in this order (the basis used is logged at debug level):

    1. ``pricing`` (explicit override), applied to the reported usage;
    2. shadowgate's ``PRICES`` table for the model the CLI reports in ``modelUsage`` (else the
       configured model);
    3. the CLI's ``total_cost_usd``, only when every ``modelUsage`` entry has a known
       ``costBasis`` (not "unknown") and stderr has no unrecognised-model marker;
    4. otherwise None (unknown).

    Cache writes with the 1-hour TTL (``usage.cache_creation.ephemeral_1h_input_tokens``) are
    priced at 2x base input in steps 1-2. ``latency_s`` is the wall time of the final, successful
    CLI run (failed attempts and backoff sleeps excluded). ``timeout_s`` bounds each run; on
    expiry the CLI and every process it started are killed.
    """

    def __init__(
        self,
        model: str,
        *,
        executable: str | Sequence[str] = "claude",
        timeout_s: float = 600,
        retry: RetryPolicy | None = None,
        extra_args: Sequence[str] = (),
        isolation_args: Sequence[str] = DEFAULT_ISOLATION_ARGS,
        default_system: str | None = DEFAULT_SYSTEM,
        name: str | None = None,
        env: Mapping[str, str] | None = None,
        pricing: Pricing | None = None,
    ) -> None:
        if not model or not _SAFE_TOKEN.fullmatch(model):
            raise ConfigError(f"claude-code backend: invalid model name {model!r}")
        if isinstance(executable, str):
            if not executable:
                raise ConfigError("claude-code backend: executable must not be empty")
        elif not executable or not all(isinstance(a, str) for a in executable):
            raise ConfigError("claude-code backend: executable must be a string or argv list")
        self.model = model
        self.executable = executable if isinstance(executable, str) else tuple(executable)
        self.timeout_s = float(timeout_s)
        self.retry = retry or RetryPolicy()
        self.extra_args = tuple(extra_args)
        self.isolation_args = tuple(isolation_args)
        self.default_system = default_system
        self.name = name or f"claude-code:{model}"
        self.env = dict(env) if env is not None else None
        self.pricing = pricing

    # ------------------------------------------------------------------ helpers

    def _resolve(self) -> tuple[list[str], bool]:
        """Return (argv prefix, routed-through-cmd.exe)."""
        if isinstance(self.executable, tuple):
            return list(self.executable), False
        path = shutil.which(self.executable)
        if path is None:
            raise BackendError(
                f"Claude Code CLI {self.executable!r} not found on PATH; install Claude Code "
                "or set the backend's 'executable' option",
                backend=self.name,
            )
        shim = os.name == "nt" and path.lower().endswith((".cmd", ".bat"))
        return [path], shim

    def build_command(
        self, request: Request, *, scratch_dir: str | None = None
    ) -> tuple[list[str], str]:
        """Return (argv, stdin text) for ``request``. Exposed for inspection and tests.

        ``scratch_dir`` is where a system-prompt file is written when the executable is a
        cmd.exe shim; without it (or when its path is not a plain token) the system prompt is
        sent in-band on stdin instead.
        """
        argv, shim = self._resolve()
        argv += ["-p", "--output-format", "json", "--model", self.model]
        prompt = request.prompt

        def argv_ok(text: str) -> bool:
            return len(text) <= _MAX_ARGV_SYSTEM_CHARS and "\x00" not in text

        # Through a cmd.exe shim no free text goes on the command line: the system prompt is
        # written to a file (--system-prompt-file) or, failing that, sent in-band on stdin.
        file_ok = (
            shim and scratch_dir is not None and bool(_SHIM_SAFE_PATH.fullmatch(scratch_dir))
        )
        system = request.system if request.system is not None else self.default_system
        if shim:
            if system is not None and file_ok:
                assert scratch_dir is not None
                argv += ["--system-prompt-file", self._write_system(scratch_dir, system)]
            elif request.system is not None:
                prompt = f"{request.system}\n\n{prompt}"
        else:
            if request.system is not None and not argv_ok(request.system):
                # Too long for a command line: send it in-band, keep the default on argv.
                prompt = f"{request.system}\n\n{prompt}"
                system = self.default_system
            if system is not None and argv_ok(system):
                argv += ["--system-prompt", system]
        if request.effort:
            if not _SAFE_TOKEN.fullmatch(request.effort):
                raise BackendError(f"invalid effort {request.effort!r}", backend=self.name)
            argv += ["--effort", request.effort]
        argv += list(self.isolation_args)
        argv += list(self.extra_args)
        return argv, prompt

    @staticmethod
    def _write_system(scratch_dir: str, text: str) -> str:
        path = os.path.join(scratch_dir, "system-prompt.txt")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return path

    def _error(self, message: str, status: int | None) -> Exception:
        msg = _tail(message) or "no output"
        if status in (401, 403) or _AUTH_RE.search(msg):
            return BackendError(
                f"Claude Code CLI authentication failed ({msg}); run `claude` and log in, "
                "or configure an API key for the CLI",
                backend=self.name,
                status=status,
            )
        if (status is not None and status in RETRYABLE_STATUS) or _TRANSIENT_RE.search(msg):
            return TransientError(f"Claude Code CLI: {msg}", status=status)
        return BackendError(f"Claude Code CLI error: {msg}", backend=self.name, status=status)

    def _call_once(self, request: Request) -> Completion:
        env = None if self.env is None else {**os.environ, **self.env}
        with tempfile.TemporaryDirectory(
            prefix="shadowgate-claude-", ignore_cleanup_errors=True
        ) as workdir, tempfile.TemporaryDirectory(
            prefix="shadowgate-claude-sys-", ignore_cleanup_errors=True
        ) as scratch:
            # The system-prompt file lives outside the (empty) working directory.
            argv, stdin_text = self.build_command(request, scratch_dir=scratch)
            with Timer() as timer:
                try:
                    proc = run_process(
                        argv,
                        input=stdin_text.encode("utf-8"),
                        timeout=self.timeout_s,
                        cwd=workdir,
                        env=env,
                    )
                except ProcessTimeout as exc:
                    raise TransientError(
                        f"Claude Code CLI timed out after {self.timeout_s:g}s"
                    ) from exc
                except FileNotFoundError as exc:
                    raise BackendError(
                        f"Claude Code CLI executable not found: {argv[0]!r}", backend=self.name
                    ) from exc
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        data = _parse_result(stdout)
        if data is None:
            detail = stderr or stdout or f"exit code {proc.returncode}"
            raise self._error(f"no JSON result (exit {proc.returncode}): {detail}", None)

        status_raw = data.get("api_error_status")
        status = status_raw if isinstance(status_raw, int) else None
        result = data.get("result")
        text = result if isinstance(result, str) else ""
        if data.get("is_error") or (proc.returncode != 0 and not text):
            detail = text or stderr or str(data.get("subtype") or "")
            raise self._error(detail, status)

        usage_d = data.get("usage")
        usage_d = usage_d if isinstance(usage_d, Mapping) else {}
        usage = Usage(
            input_tokens=_int(usage_d, "input_tokens"),
            output_tokens=_int(usage_d, "output_tokens"),
            cache_read_tokens=_int(usage_d, "cache_read_input_tokens"),
            cache_write_tokens=_int(usage_d, "cache_creation_input_tokens"),
        )
        creation = usage_d.get("cache_creation")
        write_1h = _int(creation, "ephemeral_1h_input_tokens") \
            if isinstance(creation, Mapping) else 0
        raw_stop = data.get("stop_reason")
        stop_reason = _STOP_REASONS.get(raw_stop, raw_stop) if raw_stop else "end"
        model = self.model
        model_usage = data.get("modelUsage")
        if isinstance(model_usage, Mapping) and len(model_usage) == 1:
            model = str(next(iter(model_usage)))
        cost, basis = self.resolve_cost(data, usage, model, write_1h=write_1h, stderr=stderr)
        log.debug("%s: cost_usd=%s (basis: %s)", self.name, cost, basis)
        return Completion(
            text=text,
            model=model,
            usage=usage,
            cost_usd=cost,
            latency_s=timer.elapsed,
            stop_reason=str(stop_reason),
        )

    def resolve_cost(
        self,
        data: Mapping[str, Any],
        usage: Usage,
        model: str,
        *,
        write_1h: int = 0,
        stderr: str = "",
    ) -> tuple[float | None, str]:
        """Return (cost_usd, basis) for a parsed CLI result; see the class docstring.

        ``basis`` is one of "override", "table", "cli:<costBasis>" or "unknown".
        """
        if self.pricing is not None:
            return cost_with_cache_ttl(
                model, usage, write_1h_tokens=write_1h, override=self.pricing
            ), "override"
        for candidate in dict.fromkeys((model, self.model)):
            if lookup(candidate) is not None:
                return cost_with_cache_ttl(candidate, usage, write_1h_tokens=write_1h), "table"
        cost_raw = data.get("total_cost_usd")
        if isinstance(cost_raw, bool) or not isinstance(cost_raw, (int, float)):
            return None, "unknown"
        if _UNRECOGNIZED_MODEL_MARK in stderr:
            log.debug("%s: CLI did not recognise the model; ignoring its cost", self.name)
            return None, "unknown"
        model_usage = data.get("modelUsage")
        bases: list[str] = []
        if isinstance(model_usage, Mapping):
            for entry in model_usage.values():
                raw = entry.get("costBasis") if isinstance(entry, Mapping) else None
                if raw is not None:
                    bases.append(str(raw).strip().lower())
        if any(b in _UNTRUSTED_COST_BASES for b in bases):
            log.debug("%s: CLI cost basis %s is not trusted; cost unknown", self.name, bases)
            return None, "unknown"
        return float(cost_raw), "cli:" + (",".join(sorted(set(bases))) or "unreported")

    # ------------------------------------------------------------------ Backend protocol

    def complete(self, request: Request) -> Completion:
        if request.temperature is not None or request.stop or request.want_logprobs:
            log.debug("%s: temperature/stop/logprobs are not supported and are ignored",
                      self.name)
        return call_with_retries(
            lambda: self._call_once(request), policy=self.retry, backend=self.name
        )

    def __repr__(self) -> str:
        return f"ClaudeCodeBackend(name={self.name!r}, model={self.model!r})"
