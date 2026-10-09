"""Model backends and the ``make_backend`` factory.

Backend classes are re-exported lazily (module ``__getattr__``) so importing this package never
imports an optional dependency or a provider module until it is used.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from ..errors import ConfigError
from .base import RetryPolicy

if TYPE_CHECKING:
    from ..types import Backend, Task
    from .anthropic import AnthropicBackend
    from .cache import CachedBackend, CacheStore
    from .claude_code import ClaudeCodeBackend
    from .command import CommandBackend
    from .function import FunctionBackend
    from .openai_compat import OpenAICompatBackend
    from .replay import ReplayBackend
    from .simulated import SimulatedBackend

__all__ = [
    "make_backend",
    "BACKEND_TYPES",
    "AnthropicBackend",
    "OpenAICompatBackend",
    "ClaudeCodeBackend",
    "CommandBackend",
    "SimulatedBackend",
    "ReplayBackend",
    "FunctionBackend",
    "CachedBackend",
    "CacheStore",
]

_LAZY: dict[str, str] = {
    "AnthropicBackend": ".anthropic",
    "OpenAICompatBackend": ".openai_compat",
    "ClaudeCodeBackend": ".claude_code",
    "CommandBackend": ".command",
    "SimulatedBackend": ".simulated",
    "ReplayBackend": ".replay",
    "FunctionBackend": ".function",
    "CachedBackend": ".cache",
    "CacheStore": ".cache",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


# Allowed spec keys per backend type ("type" itself is always allowed).
_KEYS: dict[str, frozenset[str]] = {
    "anthropic": frozenset(
        {"model", "name", "api_key_env", "base_url", "timeout_s", "max_attempts", "pricing"}
    ),
    "openai": frozenset(
        {"model", "name", "api_key_env", "base_url", "timeout_s", "max_attempts", "pricing",
         "headers"}
    ),
    "claude-code": frozenset(
        {"model", "name", "executable", "timeout_s", "max_attempts", "extra_args",
         "default_system"}
    ),
    "command": frozenset(
        {"command", "name", "timeout_s", "max_attempts", "pricing", "retryable_exit_codes", "cwd"}
    ),
    "simulated": frozenset(
        {"name", "model", "skill", "seed", "overconfidence", "confidence_noise",
         "discrimination", "systematic_error", "pricing", "latency_s", "latency_per_token_s",
         "emit_confidence", "emit_logprobs"}
    ),
    "replay": frozenset({"path", "name", "key_backend"}),
}
_REQUIRED: dict[str, tuple[str, ...]] = {
    "anthropic": ("model",),
    "openai": ("model",),
    "claude-code": ("model",),
    "command": ("command",),
    "simulated": ("skill",),
    "replay": ("path",),
}
BACKEND_TYPES: tuple[str, ...] = tuple(_KEYS)


def _number(spec: Mapping[str, Any], key: str, *, positive: bool = False) -> float:
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"backend key {key!r} must be a number, got {value!r}")
    if positive and value <= 0:
        raise ConfigError(f"backend key {key!r} must be > 0, got {value!r}")
    return float(value)


def _retry(spec: Mapping[str, Any]) -> RetryPolicy | None:
    if "max_attempts" not in spec:
        return None
    n = spec["max_attempts"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ConfigError(f"backend key 'max_attempts' must be an integer >= 1, got {n!r}")
    return RetryPolicy(max_attempts=n)


def make_backend(
    spec: Mapping[str, Any],
    *,
    cache: CacheStore | None = None,
    tasks: Iterable[Task] | None = None,
) -> Backend:
    """Build a backend from a config table such as ``{"type": "anthropic", "model": ...}``.

    Types: ``anthropic``, ``openai``, ``claude-code``, ``command``, ``simulated``, ``replay``.
    Common keys: ``model``, ``name`` (override), ``pricing`` (a table for
    ``pricing.pricing_from_spec``), ``timeout_s``, ``max_attempts``. ``simulated`` needs the task
    list through the ``tasks`` keyword argument (its answers are derived from task references).
    Unknown types or keys raise ConfigError naming the key; a literal ``api_key`` is rejected
    (put the key in an environment variable and name it with ``api_key_env``). When ``cache``
    is given the backend is wrapped in ``CachedBackend``.
    """
    if not isinstance(spec, Mapping):
        raise ConfigError(f"backend spec must be a table, got {type(spec).__name__}")
    if "api_key" in spec:
        raise ConfigError(
            "backend key 'api_key' is not allowed: store the key in an environment variable "
            "and set 'api_key_env' to its name"
        )
    kind = spec.get("type")
    if kind is None:
        raise ConfigError("backend spec is missing key 'type'")
    if kind not in _KEYS:
        raise ConfigError(
            f"unknown backend type {kind!r} (key 'type'); expected one of "
            f"{', '.join(BACKEND_TYPES)}"
        )
    keys = set(spec) - {"type"}
    unknown = sorted(keys - _KEYS[kind])
    if unknown:
        raise ConfigError(f"unknown key {unknown[0]!r} for backend type {kind!r}")
    for key in _REQUIRED[kind]:
        if spec.get(key) in (None, ""):
            raise ConfigError(f"backend type {kind!r} requires key {key!r}")

    kwargs: dict[str, Any] = {}
    if "timeout_s" in spec:
        kwargs["timeout_s"] = _number(spec, "timeout_s", positive=True)
    retry = _retry(spec)
    if retry is not None:
        kwargs["retry"] = retry
    if "pricing" in spec:
        from ..pricing import pricing_from_spec

        pricing = pricing_from_spec(spec["pricing"])
        if pricing is not None:
            kwargs["pricing"] = pricing
    for key in ("name", "api_key_env", "base_url", "executable", "default_system", "cwd",
                "key_backend"):
        if key in spec:
            kwargs[key] = spec[key]

    try:
        backend = _build(kind, spec, kwargs, tasks)
    except TypeError as exc:
        raise ConfigError(f"backend type {kind!r}: invalid configuration ({exc})") from exc
    if cache is not None:
        from .cache import CachedBackend

        return CachedBackend(backend, cache)
    return backend


def _build(
    kind: str, spec: Mapping[str, Any], kwargs: dict[str, Any], tasks: Iterable[Task] | None
) -> Backend:
    if kind == "anthropic":
        from .anthropic import AnthropicBackend

        return AnthropicBackend(str(spec["model"]), **kwargs)
    if kind == "openai":
        from .openai_compat import OpenAICompatBackend

        if "headers" in spec:
            if not isinstance(spec["headers"], Mapping):
                raise ConfigError("backend key 'headers' must be a table")
            kwargs["headers"] = dict(spec["headers"])
        return OpenAICompatBackend(str(spec["model"]), **kwargs)
    if kind == "claude-code":
        from .claude_code import ClaudeCodeBackend

        if "extra_args" in spec:
            kwargs["extra_args"] = tuple(_str_list(spec, "extra_args"))
        return ClaudeCodeBackend(str(spec["model"]), **kwargs)
    if kind == "command":
        from .command import CommandBackend

        if "retryable_exit_codes" in spec:
            codes = spec["retryable_exit_codes"]
            if not isinstance(codes, (list, tuple)) or not all(
                isinstance(c, int) and not isinstance(c, bool) for c in codes
            ):
                raise ConfigError("backend key 'retryable_exit_codes' must be a list of integers")
            kwargs["retryable_exit_codes"] = tuple(codes)
        command = spec["command"]
        if not isinstance(command, str):
            command = _str_list(spec, "command")
        return CommandBackend(command, **kwargs)
    if kind == "simulated":
        from .simulated import SimulatedBackend

        if tasks is None:
            raise ConfigError(
                "backend type 'simulated' needs the task list (make_backend(..., tasks=...))"
            )
        name = kwargs.pop("name", None) or spec.get("model") or "model"
        for key in ("skill", "overconfidence", "confidence_noise", "discrimination",
                    "systematic_error", "latency_s", "latency_per_token_s"):
            if key in spec:
                kwargs[key] = _number(spec, key)
        if "seed" in spec:
            seed = spec["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ConfigError(f"backend key 'seed' must be an integer, got {seed!r}")
            kwargs["seed"] = seed
        for key in ("emit_confidence", "emit_logprobs"):
            if key in spec:
                if not isinstance(spec[key], bool):
                    raise ConfigError(f"backend key {key!r} must be true or false")
                kwargs[key] = spec[key]
        return SimulatedBackend(str(name), tasks=tasks, **kwargs)
    # replay
    from .replay import ReplayBackend

    return ReplayBackend(spec["path"], **kwargs)


def _str_list(spec: Mapping[str, Any], key: str) -> list[str]:
    value = spec[key]
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"backend key {key!r} must be a list of strings")
    return list(value)
