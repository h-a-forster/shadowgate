from __future__ import annotations

import re

import pytest

from shadowgate.errors import ConfigError
from shadowgate.pricing import PRICES, PRICES_AS_OF, Pricing, cost_of, lookup, pricing_from_spec
from shadowgate.types import Usage


def test_prices_as_of_is_iso_date() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", PRICES_AS_OF)


def test_basic_cost() -> None:
    p = Pricing(input_per_mtok=3.0, output_per_mtok=15.0)
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert p.cost(usage) == pytest.approx(18.0)


def test_cache_rates_default_to_input() -> None:
    p = Pricing(input_per_mtok=2.0, output_per_mtok=10.0)
    usage = Usage(cache_read_tokens=500_000, cache_write_tokens=500_000)
    assert p.cost(usage) == pytest.approx(2.0)


def test_explicit_cache_rates() -> None:
    p = Pricing(2.0, 10.0, cache_read_per_mtok=0.2, cache_write_per_mtok=2.5)
    usage = Usage(
        input_tokens=100, output_tokens=200, cache_read_tokens=1000, cache_write_tokens=10
    )
    expected = (100 * 2.0 + 200 * 10.0 + 1000 * 0.2 + 10 * 2.5) / 1e6
    assert p.cost(usage) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model", "inp", "out"),
    [
        ("claude-fable-5-1", 10.0, 50.0),
        ("claude-opus-5-5", 4.0, 20.0),
        ("claude-sonnet-5-5", 2.0, 10.0),
        ("claude-haiku-5-5", 0.10, 0.50),
        ("claude-opus-5", 5.0, 25.0),
        ("claude-sonnet-5", 2.0, 10.0),
        ("claude-haiku-4-5", 1.0, 5.0),
        ("claude-opus-4-8", 5.0, 25.0),
        ("claude-opus-4-7", 5.0, 25.0),
        ("claude-opus-4-6", 5.0, 25.0),
        ("claude-sonnet-4-6", 3.0, 15.0),
    ],
)
def test_table_values(model: str, inp: float, out: float) -> None:
    p = PRICES[model]
    assert p.input_per_mtok == inp
    assert p.output_per_mtok == out


def test_documented_cache_reads() -> None:
    assert PRICES["claude-fable-5-1"].cache_read_per_mtok == pytest.approx(0.25)
    assert PRICES["claude-opus-5-5"].cache_read_per_mtok == pytest.approx(0.20)
    assert PRICES["claude-sonnet-5-5"].cache_read_per_mtok == pytest.approx(0.20)
    assert PRICES["claude-opus-4-8"].cache_read_per_mtok == pytest.approx(0.50)
    assert PRICES["claude-opus-5-5"].cache_write_per_mtok == pytest.approx(5.0)
    assert PRICES["claude-fable-5-1"].cache_write_per_mtok == pytest.approx(12.5)


def test_haiku_5_5_tiered_pricing() -> None:
    p = PRICES["claude-haiku-5-5"]
    short = Usage(input_tokens=100_000, output_tokens=1_000_000)
    assert p.cost(short) == pytest.approx(0.01 + 0.50)
    long = Usage(input_tokens=60_000, cache_read_tokens=50_000, output_tokens=1_000_000)
    # prompt = 110K tokens > 100K -> long-context card ($0.50 / $2.50, cache read 0.05)
    expected = (60_000 * 0.50 + 50_000 * 0.05 + 1_000_000 * 2.50) / 1e6
    assert p.cost(long) == pytest.approx(expected)


@pytest.mark.parametrize(
    "model",
    [
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "anthropic/claude-haiku-4-5",
        "anthropic:claude-haiku-4-5",
        "openrouter/anthropic/claude-haiku-4-5",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
)
def test_lookup_variants(model: str) -> None:
    assert lookup(model) is PRICES["claude-haiku-4-5"]


def test_lookup_unknown() -> None:
    assert lookup("some-unknown-model") is None
    assert lookup("") is None
    assert cost_of("some-unknown-model", Usage(1, 1)) is None


def test_cost_of_override_wins() -> None:
    override = Pricing(1.0, 1.0)
    usage = Usage(input_tokens=1_000_000)
    assert cost_of("claude-opus-5-5", usage, override) == pytest.approx(1.0)
    assert cost_of("not-in-table", usage, override) == pytest.approx(1.0)
    assert cost_of("claude-opus-5-5", usage) == pytest.approx(4.0)


def test_pricing_from_spec() -> None:
    assert pricing_from_spec(None) is None
    assert pricing_from_spec({}) is None
    p = pricing_from_spec({"input": 1, "output": 2, "cache_read": 0.1, "cache_write": 1.25})
    assert p == Pricing(1.0, 2.0, 0.1, 1.25)


def test_pricing_from_spec_long_context() -> None:
    p = pricing_from_spec(
        {
            "input": 0.1,
            "output": 0.5,
            "long_context": {"input": 0.5, "output": 2.5},
            "long_context_threshold": 1000,
        }
    )
    assert p is not None and p.long_context == Pricing(0.5, 2.5)
    assert p.cost(Usage(input_tokens=2000)) == pytest.approx(2000 * 0.5 / 1e6)


@pytest.mark.parametrize(
    "spec",
    [
        {"input": 1},
        {"output": 1},
        {"input": 1, "output": 2, "bogus": 3},
        {"input": "1", "output": 2},
        {"input": -1, "output": 2},
        {"input": True, "output": 2},
        {"input": 1, "output": 2, "long_context_threshold": -5},
        {"input": 1, "output": 2, "long_context": {"input": 1}},
    ],
)
def test_pricing_from_spec_rejects(spec: dict) -> None:
    with pytest.raises(ConfigError):
        pricing_from_spec(spec)


def test_unknown_key_named() -> None:
    with pytest.raises(ConfigError, match="bogus"):
        pricing_from_spec({"input": 1, "output": 2, "bogus": 3})
