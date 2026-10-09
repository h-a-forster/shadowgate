"""Per-token pricing and cost computation.

The built-in ``PRICES`` table is a convenience snapshot taken on ``PRICES_AS_OF``. Provider
prices change; the table goes stale. Any backend accepts a ``pricing`` override (config:
``pricing = {input = .., output = .., cache_read = .., cache_write = ..}`` in USD per million
tokens), which always wins over the table. Models missing from the table get ``cost_usd=None``
(unknown), never zero.

Anthropic cache-write prices in the table use the 5-minute TTL rate (1.25x base input). Writes
with the 1-hour TTL cost 2x base input; ``cost_with_cache_ttl`` prices them when the caller knows
how many of the cache-write tokens used the 1-hour TTL (``cost_of`` alone under-costs them).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ConfigError
from .types import Usage

__all__ = [
    "PRICES",
    "PRICES_AS_OF",
    "Pricing",
    "cost_of",
    "cost_with_cache_ttl",
    "lookup",
    "pricing_from_spec",
]

PRICES_AS_OF = "2026-10-06"

_MTOK = 1_000_000.0


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens.

    ``cache_read_per_mtok`` / ``cache_write_per_mtok`` default to the input price when None.
    ``long_context`` optionally replaces this rate card when the prompt (input + cache read +
    cache write tokens) exceeds ``long_context_threshold`` tokens (tiered pricing).
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    long_context: Pricing | None = None
    long_context_threshold: int = 100_000

    def card_for(self, usage: Usage) -> Pricing:
        """The rate card that applies to ``usage`` (the long-context card above the threshold)."""
        prompt = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        if self.long_context is not None and prompt > self.long_context_threshold:
            return self.long_context
        return self

    def cost(self, usage: Usage) -> float:
        card = self.card_for(usage)
        if card is not self:
            return card.cost(usage)
        read = self.input_per_mtok if self.cache_read_per_mtok is None else self.cache_read_per_mtok
        write = (
            self.input_per_mtok if self.cache_write_per_mtok is None else self.cache_write_per_mtok
        )
        total = (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
            + usage.cache_read_tokens * read
            + usage.cache_write_tokens * write
        )
        return total / _MTOK


def _anthropic(inp: float, out: float, read: float | None = None) -> Pricing:
    """Anthropic rate card: cache read 0.1x input unless given, cache write (5m TTL) 1.25x."""
    return Pricing(
        input_per_mtok=inp,
        output_per_mtok=out,
        cache_read_per_mtok=round(inp * 0.1, 6) if read is None else read,
        cache_write_per_mtok=round(inp * 1.25, 6),
    )


# Source: Anthropic model and pricing documentation as of PRICES_AS_OF. Cache reads are 0.1x
# base input except where listed explicitly; cache writes use the 5-minute TTL rate (1.25x).
PRICES: dict[str, Pricing] = {
    "claude-fable-5-1": _anthropic(10.0, 50.0, read=0.25),
    "claude-mythos-5-1": _anthropic(10.0, 50.0, read=0.25),
    "claude-fable-5": _anthropic(10.0, 50.0, read=1.0),
    "claude-mythos-5": _anthropic(10.0, 50.0, read=1.0),
    "claude-opus-5-5": _anthropic(4.0, 20.0, read=0.20),
    "claude-opus-5": _anthropic(5.0, 25.0),
    "claude-opus-4-8": _anthropic(5.0, 25.0),
    "claude-opus-4-7": _anthropic(5.0, 25.0),
    "claude-opus-4-6": _anthropic(5.0, 25.0),
    "claude-sonnet-5-5": _anthropic(2.0, 10.0, read=0.20),
    "claude-sonnet-5": _anthropic(2.0, 10.0),
    "claude-sonnet-4-6": _anthropic(3.0, 15.0),
    "claude-haiku-5-5": Pricing(
        input_per_mtok=0.10,
        output_per_mtok=0.50,
        cache_read_per_mtok=0.01,
        cache_write_per_mtok=0.125,
        long_context=_anthropic(0.50, 2.50),
        long_context_threshold=100_000,
    ),
    "claude-haiku-4-5": _anthropic(1.0, 5.0),
}

# Provider prefixes stripped by lookup(): "anthropic/claude-x", "anthropic:claude-x",
# "anthropic.claude-x" (Amazon Bedrock style), "openrouter/anthropic/claude-x", ...
# Bedrock cross-region inference profiles add "us.", "eu.", "apac.", "global." ... before it.
_PREFIX_RE = re.compile(r"^(?:[A-Za-z0-9_-]+[/:])+|^(?:[a-z]{2,6}\.)?anthropic\.")
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")
_VERSION_SUFFIX_RE = re.compile(r"-v\d+(?::\d+)?$")
# Vertex AI pins versions with "@": "claude-haiku-4-5@20251001".
_VERTEX_SUFFIX_RE = re.compile(r"@[A-Za-z0-9._-]+$")


def lookup(model: str) -> Pricing | None:
    """Find built-in pricing: exact id, then provider-prefix-stripped id (``anthropic/``,
    ``us.anthropic.``, ``global.anthropic.``, ...; Bedrock ``-v1:0`` and Vertex ``@date``
    suffixes dropped), then without a date suffix (``claude-haiku-4-5-20251001`` ->
    ``claude-haiku-4-5``)."""
    if not model:
        return None
    if model in PRICES:
        return PRICES[model]
    stripped = _VERSION_SUFFIX_RE.sub("", _VERTEX_SUFFIX_RE.sub("", model))
    for _ in range(4):
        new = _PREFIX_RE.sub("", stripped)
        if new == stripped:
            break
        stripped = new
    for cand in (stripped, _DATE_SUFFIX_RE.sub("", stripped)):
        if cand in PRICES:
            return PRICES[cand]
    return None


def cost_of(model: str, usage: Usage, override: Pricing | None = None) -> float | None:
    """Cost in USD of ``usage`` on ``model``; None when pricing is unknown."""
    pricing = override if override is not None else lookup(model)
    if pricing is None:
        return None
    return pricing.cost(usage)


def cost_with_cache_ttl(
    model: str,
    usage: Usage,
    *,
    write_1h_tokens: int = 0,
    override: Pricing | None = None,
) -> float | None:
    """Like ``cost_of`` but prices ``write_1h_tokens`` of ``usage.cache_write_tokens`` at the
    1-hour TTL rate (2x the applicable base input price) instead of the card's cache-write rate.

    ``write_1h_tokens`` is clamped to ``[0, usage.cache_write_tokens]``. None when pricing is
    unknown.
    """
    pricing = override if override is not None else lookup(model)
    if pricing is None:
        return None
    total = pricing.cost(usage)
    n_1h = max(0, min(int(write_1h_tokens), usage.cache_write_tokens))
    if n_1h:
        card = pricing.card_for(usage)
        write = (
            card.input_per_mtok if card.cache_write_per_mtok is None
            else card.cache_write_per_mtok
        )
        total += n_1h * (2.0 * card.input_per_mtok - write) / _MTOK
    return total


_SPEC_KEYS = {"input", "output", "cache_read", "cache_write", "long_context",
              "long_context_threshold"}


def _rate(spec: Mapping[str, Any], key: str, *, required: bool) -> float | None:
    if key not in spec or spec[key] is None:
        if required:
            raise ConfigError(f"pricing.{key} is required (USD per million tokens)")
        return None
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"pricing.{key} must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"pricing.{key} must be a finite non-negative number")
    return value


def pricing_from_spec(spec: Mapping[str, Any] | None) -> Pricing | None:
    """Build a Pricing from ``{"input", "output", "cache_read", "cache_write"}`` (USD/MTok).

    Optional ``long_context`` (a nested table of the same shape) and ``long_context_threshold``
    (prompt tokens) model tiered pricing. None or an empty mapping returns None.
    """
    if spec is None:
        return None
    if isinstance(spec, Pricing):
        return spec
    if not isinstance(spec, Mapping):
        raise ConfigError(f"pricing must be a table, got {type(spec).__name__}")
    if not spec:
        return None
    unknown = sorted(set(spec) - _SPEC_KEYS)
    if unknown:
        raise ConfigError(f"pricing: unknown key {unknown[0]!r}")
    long_spec = spec.get("long_context")
    long_context = None
    if long_spec is not None:
        if not isinstance(long_spec, Mapping) or not long_spec:
            raise ConfigError("pricing.long_context must be a non-empty table")
        if set(long_spec) & {"long_context", "long_context_threshold"}:
            raise ConfigError("pricing.long_context cannot be nested")
        long_context = pricing_from_spec(long_spec)
    threshold = spec.get("long_context_threshold", 100_000)
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise ConfigError("pricing.long_context_threshold must be a non-negative integer")
    inp = _rate(spec, "input", required=True)
    out = _rate(spec, "output", required=True)
    assert inp is not None and out is not None
    return Pricing(
        input_per_mtok=inp,
        output_per_mtok=out,
        cache_read_per_mtok=_rate(spec, "cache_read", required=False),
        cache_write_per_mtok=_rate(spec, "cache_write", required=False),
        long_context=long_context,
        long_context_threshold=threshold,
    )
