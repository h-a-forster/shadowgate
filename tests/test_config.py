from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from shadowgate.backends.cache import CachedBackend
from shadowgate.cascade import Cascade
from shadowgate.config import DEFAULT_LEDGER, Config, RunSettings, load_config, parse_config
from shadowgate.errors import ConfigError
from shadowgate.types import Task

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
TASKS_FILE = EXAMPLES / "tasks" / "arithmetic-50.jsonl"
TASKS = [
    Task("t1", "What is 2 + 2?", "4", {"difficulty": 1}),
    Task("t2", "What is 10 * 3?", "30", {"difficulty": 3}),
    Task("t3", "What is 7 - 9?", "-2", {"difficulty": 5}),
]


def sim_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "backends": {
            "fast": {"type": "simulated", "skill": 2.0, "overconfidence": 0.1},
            "slow": {"type": "simulated", "skill": 8.0},
        },
        "tiers": [
            {"name": "fast", "backend": "fast", "threshold": 0.8, "confidence": {"type": "verbal"}},
            {"name": "slow", "backend": "slow"},
        ],
    }
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def err(data: dict[str, Any], **kw: Any) -> str:
    with pytest.raises(ConfigError) as info:
        cfg = parse_config(data, **kw)
        cfg.build(tasks=TASKS)
    return str(info.value)


# --------------------------------------------------------------------------- examples


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if building a backend opens a socket."""
    import socket

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access during config build")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy-key")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)


EXAMPLE_FILES = sorted(EXAMPLES.glob("*.toml"))


def test_expected_examples_exist() -> None:
    names = {p.name for p in EXAMPLE_FILES}
    assert {
        "basic.toml",
        "anthropic.toml",
        "claude-code.toml",
        "openai-compatible.toml",
        "simulated.toml",
    } <= names


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.stem)
def test_examples_parse_and_build(
    path: Path, no_network: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uses_anthropic = path.read_text(encoding="utf-8").count('type = "anthropic"')
    if uses_anthropic:
        pytest.importorskip("anthropic")
    monkeypatch.chdir(tmp_path)  # cache = true creates .shadowgate/ relative to the cwd
    from shadowgate.datasets import load_tasks

    cfg = load_config(path)
    assert cfg.path == path
    cascade, settings = cfg.build(tasks=load_tasks(TASKS_FILE))
    assert isinstance(cascade, Cascade)
    assert len(cascade.tiers) >= 2
    assert cascade.tiers[0].threshold is not None
    assert cascade.tiers[-1].threshold is None
    assert settings is cfg.settings
    json.dumps(cfg.snapshot())


def test_anthropic_example_details(
    no_network: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.chdir(tmp_path)
    cfg = load_config(EXAMPLES / "anthropic.toml")
    cascade, s = cfg.build()
    assert s.max_cost_usd == 5.0 and s.tolerance == 0.05
    assert s.cache_path == DEFAULT_LEDGER.parent / "cache.sqlite"
    fast, slow = cascade.tiers
    assert fast.effort == "low" and fast.threshold == 0.8
    assert "claude-haiku-5-5" in fast.backend.name
    assert "claude-opus-5-5" in slow.backend.name
    assert isinstance(fast.backend, CachedBackend)
    assert cascade.audit is not None
    assert cascade.audit.strata == ((0.8, 0.9, 0.3), (0.9, 1.0, 0.1))
    assert cascade.audit.audit_tier == "opus"


def test_claude_code_example_uses_full_model_ids(
    no_network: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cascade, _ = load_config(EXAMPLES / "claude-code.toml").build()
    names = [t.backend.name for t in cascade.tiers]
    assert names == ["claude-code:claude-haiku-5-5", "claude-code:claude-opus-5-5"]


def test_simulated_example_runs_offline() -> None:
    from shadowgate.datasets import load_tasks

    tasks = load_tasks(TASKS_FILE)
    cascade, settings = load_config(EXAMPLES / "simulated.toml").build(tasks=tasks)
    assert settings.cache_path is None
    decisions = [cascade.route(t, run_id="test") for t in tasks[:20]]
    assert all(d.error is None for d in decisions)
    assert any(d.escalated for d in decisions)
    assert any(not d.escalated for d in decisions)
    assert all(d.correct is not None for d in decisions)


# --------------------------------------------------------------------------- build behaviour


def test_minimal_build_and_defaults() -> None:
    cfg = parse_config(sim_config())
    assert cfg.settings == RunSettings()
    assert cfg.settings.ledger_path == Path(".shadowgate/ledger.sqlite")
    cascade, settings = cfg.build(tasks=TASKS)
    assert [t.name for t in cascade.tiers] == ["fast", "slow"]
    assert cascade.tiers[0].backend.name == "sim:fast"
    assert cascade.comparator is not None  # default comparator grades against references
    assert cascade.audit is not None and cascade.audit.mode == "inline"
    d = cascade.route(TASKS[0], run_id="r")
    assert d.answer


def test_simulated_needs_tasks() -> None:
    cfg = parse_config(sim_config())
    with pytest.raises(ConfigError, match=r"backends\.fast: simulated .*build\(tasks="):
        cfg.build()


def test_backend_instances_are_shared() -> None:
    data = sim_config()
    data["tiers"] = [
        {
            "name": "a",
            "backend": "fast",
            "threshold": 0.9,
            "confidence": {"type": "monitor", "backend": "slow"},
        },
        {"name": "b", "backend": "fast", "threshold": 0.5, "confidence": {"type": "verbal"}},
        {"name": "c", "backend": "slow"},
    ]
    data["audit"] = {"judge": {"type": "judge", "backend": "slow"}}
    cascade, _ = parse_config(data).build(tasks=TASKS)
    a, b, c = cascade.tiers
    assert a.backend is b.backend
    assert a.estimator.backend is c.backend  # type: ignore[union-attr]
    assert cascade.judge.backend is c.backend  # type: ignore[union-attr]


def test_unreferenced_backends_are_not_built() -> None:
    data = sim_config()
    data["backends"]["unused"] = {"type": "replay", "path": "does-not-exist.jsonl"}
    cascade, _ = parse_config(data).build(tasks=TASKS)
    assert len(cascade.tiers) == 2


def test_tier_extractor_overrides_answer_extractor() -> None:
    data = sim_config(answer={"extractor": "final_line", "comparator": "numeric"})
    data["tiers"][1]["extractor"] = {"type": "last_number"}
    cascade, _ = parse_config(data).build(tasks=TASKS)
    assert cascade.tiers[0].extractor is None
    assert cascade.tiers[1].extractor is not None
    assert cascade.tiers[1].extractor.name != cascade.extractor.name  # type: ignore[union-attr]
    assert "numeric" in cascade.comparator.name.lower()  # type: ignore[union-attr]
    assert cascade.judge is cascade.comparator


def test_audit_settings() -> None:
    data = sim_config(
        run={"seed": 7},
        audit={
            "rate": 0.3,
            "floor": 0.05,
            "mode": "deferred",
            "tier": "slow",
            "strata": [[0.8, 0.9, 0.5], [0.9, 1.0, 0.1]],
            "tolerance": 0.02,
        },
    )
    cfg = parse_config(data)
    assert cfg.settings.tolerance == 0.02 and cfg.settings.seed == 7
    cascade, _ = cfg.build(tasks=TASKS)
    policy = cascade.audit
    assert policy is not None
    assert (policy.rate, policy.floor, policy.mode, policy.audit_tier, policy.seed) == (
        0.3,
        0.05,
        "deferred",
        "slow",
        7,
    )
    assert policy.strata == ((0.8, 0.9, 0.5), (0.9, 1.0, 0.1))


def test_calibrated_and_combine_confidence() -> None:
    data = sim_config()
    data["tiers"][0]["confidence"] = {
        "type": "combine",
        "method": "mean",
        "members": [
            {"type": "verbal"},
            {"type": "calibrated", "base": {"type": "logprob"}, "points": [[0, 0], [1, 0.9]]},
        ],
    }
    cascade, _ = parse_config(data).build(tasks=TASKS)
    assert cascade.tiers[0].estimator is not None


def test_paths_resolve_against_config_dir(tmp_path: Path) -> None:
    (tmp_path / "cfg").mkdir()
    cfg_file = tmp_path / "cfg" / "c.toml"
    cfg_file.write_text(
        '[run]\nledger = "out/ledger.sqlite"\ncache = "out/cache.sqlite"\n'
        '[backends.r]\ntype = "replay"\npath = "rec.jsonl"\n'
        '[[tiers]]\nbackend = "r"\n',
        encoding="utf-8",
    )
    (tmp_path / "cfg" / "rec.jsonl").write_text("", encoding="utf-8")
    cfg = load_config(cfg_file)
    base = cfg_file.resolve().parent
    assert cfg.settings.ledger_path == base / "out" / "ledger.sqlite"
    assert cfg.settings.cache_path == base / "out" / "cache.sqlite"
    assert cfg.settings.name == "c"
    cascade, _ = cfg.build()
    assert len(cascade.tiers) == 1
    assert (base / "out" / "cache.sqlite").exists()


def test_cache_true_and_false() -> None:
    assert parse_config(sim_config(run={"cache": False})).settings.cache_path is None
    on = parse_config(
        sim_config(run={"cache": True, "ledger": "x/l.sqlite"}), base_dir=Path("/base")
    )
    assert on.settings.cache_path == Path("/base/x/cache.sqlite")


def test_snapshot_is_json_safe_and_redacted() -> None:
    import datetime

    cfg = Config(
        path=None,
        raw={
            "run": {"name": "x", "when": datetime.date(2026, 1, 2)},
            "b": {"api_key": "sk-123", "headers": {"Authorization": "y", "X-Other": "z"}},
        },
        settings=RunSettings(),
    )
    snap = cfg.snapshot()
    text = json.dumps(snap)
    assert "sk-123" not in text and '"y"' not in text
    assert snap["run"]["when"] == "2026-01-02"
    assert snap["b"]["headers"]["X-Other"] == "z"
    good = parse_config(sim_config())
    assert good.snapshot() == json.loads(json.dumps(sim_config()))


# --------------------------------------------------------------------------- validation errors


def test_unknown_backend_reference_names_path() -> None:
    data = sim_config()
    data["tiers"][1]["backend"] = "slwo"
    msg = err(data)
    assert msg.startswith('tiers[1].backend: unknown backend "slwo"')
    assert "(defined: fast, slow)" in msg
    assert 'did you mean "slow"' in msg


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d.update(rn={}), r'^rn: unknown key "rn" .*did you mean "run"'),
        (lambda d: d.update(run={"workrs": 2}), r'^run\.workrs: .*did you mean "workers"'),
        (lambda d: d["tiers"][0].update(treshold=0.5), r'^tiers\[0\]\.treshold: .*"threshold"'),
        (
            lambda d: d["backends"]["fast"].update(skil=1),
            r'^backends\.fast\.skil: .*did you mean "skill"',
        ),
        (lambda d: d.update(audit={"rte": 0.1}), r'^audit\.rte: .*did you mean "rate"'),
        (lambda d: d.update(answer={"comparater": "numeric"}), r'"comparator"'),
        (
            lambda d: d["tiers"][0].update(confidence={"type": "verbl"}),
            r'^tiers\[0\]\.confidence\.type: unknown confidence type "verbl".*"verbal"',
        ),
        (
            lambda d: d["tiers"][0].update(confidence={"type": "verbal", "instrction": "x"}),
            r'^tiers\[0\]\.confidence\.instrction: .*"instruction"',
        ),
        (
            lambda d: d.update(answer={"comparator": {"type": "numeric", "rel_tl": 1}}),
            r'^answer\.comparator\.rel_tl: .*"rel_tol"',
        ),
        (
            lambda d: d.update(answer={"extractor": "final_lin"}),
            r'^answer\.extractor\.type: .*"final_line"',
        ),
        (
            lambda d: d["backends"]["fast"].update(type="simulatd"),
            r'^backends\.fast\.type: unknown backend type "simulatd".*"simulated"',
        ),
    ],
)
def test_unknown_keys_suggest(mutate: Any, expected: str) -> None:
    data = sim_config()
    mutate(data)
    with pytest.raises(ConfigError, match=expected):
        parse_config(data)


@pytest.mark.parametrize("key", ["api_key", "token", "secret", "API_KEY", "password"])
def test_literal_secrets_rejected(key: str) -> None:
    data = sim_config()
    data["backends"]["fast"][key] = "sk-live-abc"
    msg = err(data)
    assert msg.startswith(f"backends.fast.{key}:")
    assert "api_key_env" in msg
    assert "sk-live-abc" not in msg


def test_secret_nested_anywhere_and_headers_rejected() -> None:
    data = sim_config(audit={"judge": {"type": "numeric", "token": "x"}})
    assert err(data).startswith("audit.judge.token:")
    data = sim_config()
    data["backends"]["h"] = {
        "type": "openai",
        "model": "m",
        "headers": {"Authorization": "Bearer x"},
    }
    assert err(data).startswith("backends.h.headers.Authorization:")


def test_api_key_env_must_be_a_name() -> None:
    data = sim_config()
    data["backends"]["o"] = {"type": "openai", "model": "m", "api_key_env": "sk-abc-123"}
    msg = err(data)
    assert msg.startswith("backends.o.api_key_env:") and "NAME" in msg


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d["tiers"][0].update(threshold="0.8"), r"^tiers\[0\]\.threshold: must be a n"),
        (lambda d: d["tiers"][0].update(threshold=True), r"^tiers\[0\]\.threshold: must be a n"),
        (lambda d: d["tiers"][0].update(threshold=80), r"^tiers\[0\]\.threshold: must be in"),
        (lambda d: d["tiers"][0].pop("threshold"), r"^tiers\[0\]\.threshold: missing"),
        (lambda d: d["tiers"][0].pop("confidence"), r"^tiers\[0\]\.confidence: missing"),
        (
            lambda d: d["tiers"][1].update(threshold=0.5),
            r"^tiers\[1\]\.threshold: final tier never escalates; remove threshold",
        ),
        (
            lambda d: d["tiers"][1].update(confidence={"type": "verbal"}),
            r"^tiers\[1\]\.confidence: ",
        ),
        (lambda d: d["tiers"][1].pop("backend"), r"^tiers\[1\]\.backend: missing"),
        (lambda d: d["tiers"][1].update(name="fast"), r"^tiers\[1\]\.name: duplicate"),
        (lambda d: d["tiers"][1].update(template="no token"), r"^tiers\[1\]\.template: must"),
        (lambda d: d["tiers"][1].update(max_tokens=0), r"^tiers\[1\]\.max_tokens: .*>= 1"),
        (lambda d: d.update(tiers=[]), r"^tiers: must be a non-empty"),
        (lambda d: d.pop("tiers"), r"^tiers: missing"),
        (lambda d: d.update(run={"workers": 0}), r"^run\.workers: .*>= 1"),
        (lambda d: d.update(run={"workers": 2.5}), r"^run\.workers: must be an integer"),
        (lambda d: d.update(run={"max_cost_usd": -1}), r"^run\.max_cost_usd: must be in"),
        (lambda d: d.update(run={"cache": 3}), r"^run\.cache: must be true, false or"),
        (lambda d: d.update(audit={"rate": 0}), r"^audit\.rate: must be in \(0, 1\]"),
        (lambda d: d.update(audit={"rate": 1.5}), r"^audit\.rate: must be in \(0, 1\]"),
        (lambda d: d.update(audit={"floor": 0}), r"^audit\.floor: "),
        (lambda d: d.update(audit={"tolerance": 1.0}), r"^audit\.tolerance: "),
        (lambda d: d.update(audit={"mode": "inlne"}), r'^audit\.mode: .*did you mean "inline"'),
        (lambda d: d.update(audit={"tier": "slwo"}), r'^audit\.tier: unknown tier "slwo"'),
        (lambda d: d.update(audit={"strata": [[0.8, 0.9]]}), r"^audit\.strata\[0\]: must be"),
        (lambda d: d.update(audit={"strata": [[0.9, 0.8, 0.1]]}), r"^audit\.strata\[0\]: lo"),
        (
            lambda d: d.update(audit={"strata": [[0.8, 0.9, 0]]}),
            r"^audit\.strata\[0\]\[2\] \(rate\): must be in",
        ),
        (
            lambda d: d.update(audit={"judge": {"type": "judge", "backend": "nope"}}),
            r'^audit\.judge\.backend: unknown backend "nope"',
        ),
        (lambda d: d.update(audit={"judge": {"type": "judge"}}), r"^audit\.judge\.backend: req"),
        (
            lambda d: d["tiers"][0].update(confidence={"type": "monitor", "backend": "x"}),
            r'^tiers\[0\]\.confidence\.backend: unknown backend "x"',
        ),
        (
            lambda d: d["tiers"][0].update(
                confidence={"type": "combine", "members": [{"type": "monitor", "backend": "x"}]}
            ),
            r'^tiers\[0\]\.confidence\.members\[0\]\.backend: unknown backend "x"',
        ),
        (
            lambda d: d["tiers"][0].update(
                confidence={"type": "calibrated", "base": {"type": "verbal"}}
            ),
            r"^tiers\[0\]\.confidence\.points: ",
        ),
        (
            lambda d: d["tiers"][0].update(confidence={"type": "callable"}),
            r"^tiers\[0\]\.confidence\.type: 'callable' .*API",
        ),
        (lambda d: d["backends"]["fast"].pop("type"), r"^backends\.fast\.type: missing"),
        (lambda d: d["backends"].update(o={"type": "openai"}), r"^backends\.o\.model: required"),
        (lambda d: d.update(backends=[]), r"^backends: must be a table"),
    ],
)
def test_validation_errors(mutate: Any, expected: str) -> None:
    data = sim_config()
    mutate(data)
    with pytest.raises(ConfigError, match=expected):
        parse_config(data)


def test_build_errors_are_prefixed_with_path() -> None:
    data = sim_config()
    data["backends"]["fast"]["confidence_noise"] = -1  # rejected by the backend itself
    with pytest.raises(ConfigError, match=r"^backends\.fast: "):
        parse_config(data).build(tasks=TASKS)
    data = sim_config()
    data["tiers"][0]["confidence"] = {"type": "logprob", "aggregate": "median"}
    with pytest.raises(ConfigError, match=r"^tiers\[0\]\.confidence: "):
        parse_config(data).build(tasks=TASKS)


# --------------------------------------------------------------------------- files


def test_toml_syntax_error_has_file_and_line(tmp_path: Path) -> None:
    f = tmp_path / "bad.toml"
    f.write_text('[run]\nname = "x"\nworkers = = 3\n', encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_config(f)
    msg = str(info.value)
    assert str(f) in msg and "line 3" in msg


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_no_env_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SG_TEST_NAME", "expanded")
    cfg = parse_config(sim_config(run={"name": "${SG_TEST_NAME}"}))
    assert cfg.settings.name == "${SG_TEST_NAME}"


def test_parse_does_not_import_providers() -> None:
    import subprocess

    code = (
        "import sys; from shadowgate.config import load_config; "
        f"load_config({str(EXAMPLES / 'anthropic.toml')!r}); "
        "print('anthropic' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
