"""Tests for the experiments' Claude Code budget guard (experiments/mmlu-pro-cascade)."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[1] / "experiments" / "mmlu-pro-cascade" / "claude_budget.py"
if not SCRIPT.exists():  # e.g. running the tests from an sdist, which omits experiments/
    pytest.skip("experiments/ not available", allow_module_level=True)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("claude_budget", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load()


def _cfg(tmp_path: Path, **env: str) -> Any:
    base = {"SHADOWGATE_COST_LOG": str(tmp_path / "costs.jsonl")}
    base.update(env)
    return cb.Config(base)


def _log(cfg: Any, *costs: float) -> None:
    with open(cfg.log, "a", encoding="utf-8") as f:
        for c in costs:
            f.write(json.dumps({"total_cost_usd": c}) + "\n")


def test_config_defaults_and_validation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert (cfg.budget, cfg.reserve, cfg.ttl_s) == (88.0, 0.25, 3600.0)
    with pytest.raises(ValueError):
        _cfg(tmp_path, SHADOWGATE_RESERVE_USD="-1")
    with pytest.raises(ValueError):
        _cfg(tmp_path, SHADOWGATE_RESERVATION_TTL_S="0")


def test_spent_sums_log_and_skips_bad_lines(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cb.spent(cfg.log) == 0.0
    _log(cfg, 1.5, 0.25)
    with open(cfg.log, "a", encoding="utf-8") as f:
        f.write('{"total_cost_usd": 2.0\n\n[1, 2]\n{"total_cost_usd": null}\n')
    assert cb.spent(cfg.log) == pytest.approx(1.75)


def test_reserve_counts_each_in_flight_call(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, SHADOWGATE_BUDGET_USD="1.0", SHADOWGATE_RESERVE_USD="0.25")
    _log(cfg, 0.3)
    ids = [cb.reserve(cfg)[0] for _ in range(3)]
    assert all(ids[:2]) and ids[2] is None  # 0.3 + 2*0.25 + 0.25 > 1.0
    rid, total, others = cb.reserve(cfg)
    assert rid is None and total == pytest.approx(0.3) and others == pytest.approx(0.5)


def test_settle_logs_cost_and_releases_reservation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, SHADOWGATE_BUDGET_USD="1.0", SHADOWGATE_RESERVE_USD="0.4")
    a, _, _ = cb.reserve(cfg)
    b, _, _ = cb.reserve(cfg)
    assert a and b and cb.reserve(cfg)[0] is None
    cb.settle(cfg, a, {"total_cost_usd": 0.1})
    assert cb.spent(cfg.log) == pytest.approx(0.1)
    held = json.loads(cfg.reservations_path.read_text(encoding="utf-8"))
    assert list(held) == [b]
    cb.settle(cfg, b, None)  # released without a log entry
    assert json.loads(cfg.reservations_path.read_text(encoding="utf-8")) == {}
    assert cb.spent(cfg.log) == pytest.approx(0.1)


def test_expired_and_corrupt_reservations_are_ignored(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, SHADOWGATE_BUDGET_USD="1.0", SHADOWGATE_RESERVATION_TTL_S="60")
    assert cb.reserve(cfg, now=1000.0)[0]
    # A crashed caller's reservation still counts within the TTL...
    assert cb.reserve(cfg, now=1030.0)[2] == pytest.approx(0.25)
    # ...and is dropped after it.
    assert cb.reserve(cfg, now=1075.0)[2] == pytest.approx(0.25)  # only the 1030 one
    cfg.reservations_path.write_text("not json", encoding="utf-8")
    assert cb.reserve(cfg)[2] == 0.0


def test_concurrent_reservations_never_exceed_budget(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, SHADOWGATE_BUDGET_USD="1.0", SHADOWGATE_RESERVE_USD="0.3")
    results: list[str | None] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        results.append(cb.reserve(cfg)[0])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(results) == 8
    assert sum(r is not None for r in results) == 3


def _fake_run(stdout: bytes, code: int = 0) -> Any:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        return SimpleNamespace(stdout=stdout, stderr=b"", returncode=code)

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_main_runs_cli_logs_cost_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    cfg = _cfg(tmp_path)
    result = json.dumps({"type": "result", "result": "ok", "total_cost_usd": 0.02}).encode()
    run = _fake_run(b"noise\n" + result + b"\n")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"prompt")))
    monkeypatch.setenv("SHADOWGATE_CLAUDE_BIN", "claude-fake")
    assert cb.main(["-p", "--model", "claude-haiku-4-5"], cfg) == 0
    entry = json.loads(cfg.log.read_text(encoding="utf-8"))
    assert entry["model"] == "claude-haiku-4-5" and entry["total_cost_usd"] == 0.02
    assert entry["parsed"] is True
    assert json.loads(cfg.reservations_path.read_text(encoding="utf-8")) == {}
    assert run.calls == [["claude-fake", "-p", "--model", "claude-haiku-4-5"]]
    assert capsysbinary.readouterr().out == b"noise\n" + result + b"\n"


def test_main_refuses_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _cfg(tmp_path, SHADOWGATE_BUDGET_USD="0.1")
    _log(cfg, 0.05)
    run = _fake_run(b"")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("SHADOWGATE_CLAUDE_BIN", "claude-fake")
    assert cb.main(["-p"], cfg) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["is_error"] is True and "call refused" in out["result"]
    assert run.calls == []


def test_main_version_passes_through_without_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, SHADOWGATE_BUDGET_USD="0")
    run = _fake_run(b"2.1.0 (Claude Code)\n")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("SHADOWGATE_CLAUDE_BIN", "claude-fake")
    assert cb.main(["--version"], cfg) == 0
    assert run.calls == [["claude-fake", "--version"]]
    assert not cfg.log.exists()
