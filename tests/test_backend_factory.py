from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import shadowgate.backends as backends
from shadowgate.backends import make_backend
from shadowgate.backends.base import request_key
from shadowgate.backends.cache import CachedBackend, CacheStore
from shadowgate.backends.replay import ReplayBackend
from shadowgate.backends.simulated import SimulatedBackend
from shadowgate.errors import ConfigError
from shadowgate.pricing import Pricing
from shadowgate.types import Request, Task

TASKS = [Task("t1", "What is 1+1?", "2"), Task("t2", "What is 2+3?", "5")]


class Recorder:
    """Stand-in for a provider backend class: records constructor arguments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.name = kwargs.get("name") or f"fake:{args[0] if args else ''}"


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    classes = {}
    for mod, cls in [
        ("anthropic", "AnthropicBackend"),
        ("openai_compat", "OpenAICompatBackend"),
        ("claude_code", "ClaudeCodeBackend"),
        ("command", "CommandBackend"),
    ]:
        fake_cls = type(cls, (Recorder,), {})
        module = types.ModuleType(f"shadowgate.backends.{mod}")
        setattr(module, cls, fake_cls)
        monkeypatch.setitem(sys.modules, f"shadowgate.backends.{mod}", module)
        classes[cls] = fake_cls
    return classes


def test_import_is_lazy() -> None:
    code = (
        "import sys, shadowgate.backends as b; "
        "mods = [m for m in ('anthropic', 'shadowgate.backends.anthropic', "
        "'shadowgate.backends.openai_compat', 'shadowgate.pricing') if m in sys.modules]; "
        "print(mods)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_lazy_reexports() -> None:
    assert backends.SimulatedBackend is SimulatedBackend
    assert backends.CacheStore is CacheStore
    assert "AnthropicBackend" in dir(backends)
    with pytest.raises(AttributeError):
        _ = backends.NoSuchBackend


def test_anthropic_spec(fakes: dict[str, type]) -> None:
    b = make_backend(
        {
            "type": "anthropic",
            "model": "claude-x",
            "api_key_env": "MY_KEY",
            "timeout_s": 30,
            "max_attempts": 2,
            "pricing": {"input": 1.0, "output": 5.0},
            "name": "fast",
        }
    )
    assert isinstance(b, fakes["AnthropicBackend"])
    assert b.args == ("claude-x",)
    kw = b.kwargs
    assert kw["api_key_env"] == "MY_KEY" and kw["timeout_s"] == 30.0 and kw["name"] == "fast"
    assert kw["retry"].max_attempts == 2
    assert kw["pricing"] == Pricing(input_per_mtok=1.0, output_per_mtok=5.0)


def test_other_provider_specs(fakes: dict[str, type]) -> None:
    o = make_backend({"type": "openai", "model": "llama", "base_url": "http://localhost:11434/v1",
                      "headers": {"X-A": "1"}})
    assert isinstance(o, fakes["OpenAICompatBackend"])
    assert o.kwargs == {"base_url": "http://localhost:11434/v1", "headers": {"X-A": "1"}}
    c = make_backend({"type": "claude-code", "model": "haiku", "extra_args": ["--x"]})
    assert isinstance(c, fakes["ClaudeCodeBackend"]) and c.kwargs == {"extra_args": ("--x",)}
    cp = make_backend({"type": "claude-code", "model": "haiku",
                       "pricing": {"input": 1.0, "output": 5.0}})
    assert cp.kwargs == {"pricing": Pricing(input_per_mtok=1.0, output_per_mtok=5.0)}
    m = make_backend({"type": "command", "command": ["python", "s.py"],
                      "retryable_exit_codes": [75]})
    assert isinstance(m, fakes["CommandBackend"])
    assert m.args == (["python", "s.py"],) and m.kwargs == {"retryable_exit_codes": (75,)}
    s = make_backend({"type": "command", "command": "echo hi"})
    assert s.args == ("echo hi",)


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        ({"type": "anthropic", "model": "x", "api_key": "sk-123"}, "api_key_env"),
        ({"type": "nope"}, "nope"),
        ({"model": "x"}, "type"),
        ({"type": "anthropic", "model": "x", "temprature": 1}, "temprature"),
        ({"type": "anthropic"}, "model"),
        ({"type": "openai", "model": "x", "executable": "y"}, "executable"),
        ({"type": "anthropic", "model": "x", "max_attempts": 0}, "max_attempts"),
        ({"type": "anthropic", "model": "x", "timeout_s": "fast"}, "timeout_s"),
        ({"type": "anthropic", "model": "x", "pricing": {"input": 1, "outptu": 2}}, "outptu"),
        ({"type": "command", "command": [1, 2]}, "command"),
        ({"type": "replay"}, "path"),
        ({"type": "simulated", "skill": 1, "seed": "x"}, "seed"),
        ({"type": "simulated", "skill": 1, "emit_logprobs": "yes"}, "emit_logprobs"),
    ],
)
def test_config_errors(fakes: dict[str, type], spec: dict[str, Any], needle: str) -> None:
    with pytest.raises(ConfigError, match=needle):
        make_backend(spec, tasks=TASKS)


def test_api_key_message_does_not_leak_value() -> None:
    with pytest.raises(ConfigError) as info:
        make_backend({"type": "openai", "model": "x", "api_key": "sk-secret-value"})
    assert "sk-secret-value" not in str(info.value)


def test_simulated_spec() -> None:
    with pytest.raises(ConfigError, match="tasks"):
        make_backend({"type": "simulated", "skill": 1.0})
    b = make_backend(
        {"type": "simulated", "name": "small", "skill": 1, "seed": 3, "overconfidence": 0.1,
         "pricing": {"input": 2, "output": 4}, "emit_logprobs": False},
        tasks=TASKS,
    )
    assert isinstance(b, SimulatedBackend)
    assert b.name == "sim:small" and b.seed == 3 and b.overconfidence == 0.1
    assert b.pricing == Pricing(input_per_mtok=2.0, output_per_mtok=4.0)
    assert b.complete(Request(prompt="What is 1+1?")).cost_usd is not None
    assert make_backend({"type": "simulated", "model": "m2", "skill": 0},
                        tasks=TASKS).name == "sim:m2"


def test_replay_spec(tmp_path: Path) -> None:
    req = Request(prompt="p")
    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps({"key": request_key("orig", req), "completion": {"text": "x", "model": "m"}}),
        encoding="utf-8",
    )
    b = make_backend({"type": "replay", "path": str(path), "key_backend": "orig"})
    assert isinstance(b, ReplayBackend) and b.name == "replay:r"
    assert b.complete(req).text == "x"


def test_cache_wrapping(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "c.sqlite")
    b = make_backend({"type": "simulated", "name": "s", "skill": 0}, cache=store, tasks=TASKS)
    assert isinstance(b, CachedBackend) and b.name == "sim:s"
    req = Request(prompt="What is 2+3?")
    assert b.complete(req).cached is False
    # Simulated backends are not cacheable (deterministic; tags are not in the key): bypass.
    assert b.complete(req).cached is False
    assert len(store) == 0
    store.close()


@pytest.mark.parametrize(
    ("spec", "module"),
    [
        ({"type": "openai", "model": "llama3", "base_url": "http://localhost:1/v1",
          "api_key_env": "SHADOWGATE_TEST_UNSET_KEY"}, "openai_compat"),
        ({"type": "command", "command": ["python", "-c", "print(1)"]}, "command"),
    ],
)
def test_real_provider_classes_construct(spec: dict[str, Any], module: str) -> None:
    try:
        importlib.import_module(f"shadowgate.backends.{module}")
    except ImportError as exc:  # pragma: no cover - other modules may be in progress
        pytest.skip(f"shadowgate.backends.{module} not importable: {exc}")
    b = make_backend(spec)
    assert type(b).__module__ == f"shadowgate.backends.{module}"
    assert isinstance(b.name, str) and b.name
