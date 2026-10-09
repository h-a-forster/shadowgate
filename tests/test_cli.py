from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from shadowgate import __version__, cli
from shadowgate.cli import DEMO_CONFIG, STARTER_CONFIG, main
from shadowgate.config import parse_config
from shadowgate.datasets import arithmetic, load_tasks, save_tasks
from shadowgate.ledger import Ledger
from shadowgate.types import Decision

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

try:
    from shadowgate import report as _report  # noqa: F401

    HAVE_REPORT = True
except ImportError:  # pragma: no cover - report.py is developed separately
    HAVE_REPORT = False

needs_report = pytest.mark.skipif(not HAVE_REPORT, reason="shadowgate.report is not available")

N_TASKS = 80


def _deferred_config() -> str:
    return DEMO_CONFIG.replace('mode = "inline"', 'mode = "deferred"')


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A config, a task file and a ledger with one serve and one eval run."""
    d = tmp_path_factory.mktemp("cli")
    config = d / "sim.toml"
    config.write_text(DEMO_CONFIG, encoding="utf-8")
    tasks = d / "tasks.jsonl"
    save_tasks(arithmetic(N_TASKS, seed=1), tasks)
    ledger = d / "ledger.sqlite"
    common = ["-c", str(config), "-t", str(tasks), "--ledger", str(ledger), "-q"]
    assert main(["run", *common, "--run-id", "s1"]) == 0
    assert main(["run", *common, "--run-id", "e1", "--mode", "eval"]) == 0
    return {"dir": d, "config": config, "tasks": tasks, "ledger": ledger}


def _common(ws: dict[str, Path]) -> list[str]:
    return ["-c", str(ws["config"]), "-t", str(ws["tasks"]), "--ledger", str(ws["ledger"])]


# --------------------------------------------------------------------------- basics


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"shadowgate {__version__}"


def test_help_lists_commands_and_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    commands = ("init", "demo", "run", "audit", "sweep", "calibrate", "report", "runs", "export")
    for cmd in (*commands, "datasets"):
        assert cmd in out
    assert "exit codes" in out and "130" in out


def test_subcommand_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--help"]) == 0
    assert "--run-id" in capsys.readouterr().out


def test_no_command_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "required" in capsys.readouterr().err


def test_bad_argument_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "-c", "x.toml"]) == 2  # -t missing
    assert "-t/--tasks" in capsys.readouterr().err


def test_python_dash_m_version() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "shadowgate", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"shadowgate {__version__}"


def test_console_script_help() -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    proc = subprocess.run(
        [uv, "run", "--project", str(ROOT), "shadowgate", "--help"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=ROOT,
    )
    if proc.returncode != 0 and "shadowgate" not in proc.stdout:
        pytest.skip(f"uv run unavailable here: {proc.stderr.strip()[:200]}")
    assert proc.returncode == 0
    assert "usage: shadowgate" in proc.stdout


def test_jsonable_handles_nan_and_paths() -> None:
    from shadowgate.stats import Estimate

    obj = {"e": Estimate(float("nan"), None, float("inf"), 3, "x"), "p": Path("a"), "t": (1, 2)}
    text = json.dumps(cli._jsonable(obj), allow_nan=False)
    data = json.loads(text)
    assert data["e"]["value"] is None and data["e"]["hi"] is None
    assert data["p"] == "a" and data["t"] == [1, 2]


# --------------------------------------------------------------------------- init


def test_init_writes_runnable_starter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init", str(tmp_path / "proj")]) == 0
    cfg = tmp_path / "proj" / "shadowgate.toml"
    tasks = tmp_path / "proj" / "tasks.jsonl"
    assert cfg.read_text(encoding="utf-8") == STARTER_CONFIG
    assert len(load_tasks(tasks)) == 20
    parse_config(tomllib.loads(STARTER_CONFIG))
    capsys.readouterr()
    ledger = tmp_path / "l.sqlite"
    code = main(["run", "-c", str(cfg), "-t", str(tasks), "--ledger", str(ledger), "-q"])
    assert code == 0
    assert "20/20 done" in capsys.readouterr().out


def test_init_refuses_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "shadowgate.toml").write_text("# mine\n", encoding="utf-8")
    assert main(["init", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("shadowgate: error:") and "--force" in err
    assert (tmp_path / "shadowgate.toml").read_text(encoding="utf-8") == "# mine\n"
    assert not (tmp_path / "tasks.jsonl").exists()
    assert main(["init", str(tmp_path), "--force"]) == 0
    assert (tmp_path / "shadowgate.toml").read_text(encoding="utf-8") == STARTER_CONFIG


# --------------------------------------------------------------------------- demo


def test_demo_config_matches_example() -> None:
    example = EXAMPLES / "simulated.toml"
    if not example.is_file():
        pytest.skip("examples directory not available")
    assert tomllib.loads(DEMO_CONFIG) == tomllib.loads(example.read_text(encoding="utf-8"))


@needs_report
def test_demo_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--n", "120", "--seed", "2", "-q"]) == 0
    text = capsys.readouterr().out
    for name in ("report.html", "report.md", "ledger.sqlite", "tasks.jsonl", "shadowgate.toml"):
        assert (out / name).is_file(), name
    assert (out / "report.html").read_text(encoding="utf-8").lstrip().lower().startswith("<!doc")
    assert "report:" in text and "recommended threshold" in text
    assert "escalation rate" in text
    with Ledger(out / "ledger.sqlite") as led:
        runs = {r.run_id: r for r in led.runs()}
        assert set(runs) == {"demo-n120-s2-eval", "demo-n120-s2-serve"}
        assert runs["demo-n120-s2-eval"].mode == "eval"
        assert all(r.n_decisions == 120 for r in runs.values())
        assert led.latest_run_id() == "demo-n120-s2-serve"
    # The written config reproduces the demo cascade.
    raw = tomllib.loads((out / "shadowgate.toml").read_text(encoding="utf-8"))
    assert raw["run"]["seed"] == 2
    # Re-running into the same directory resumes (no duplicate work, same result).
    assert main(["demo", "--out", str(out), "--n", "120", "--seed", "2", "-q"]) == 0
    assert "120 resumed" in capsys.readouterr().out


@needs_report
def test_demo_open_uses_webbrowser(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import webbrowser

    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url) or True)
    assert main(["demo", "--out", str(tmp_path), "--n", "30", "--open", "-q"]) == 0
    assert len(opened) == 1 and opened[0].startswith("file:") and opened[0].endswith("report.html")


# --------------------------------------------------------------------------- run


def test_run_prints_run_id_and_summary(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", *_common(workspace), "--run-id", "r-basic", "--limit", "10"]) == 0
    out = capsys.readouterr().out
    assert "starting run r-basic" in out
    assert "run r-basic: 10/10 done" in out
    assert "next: shadowgate audit --run-id r-basic" in out


def test_run_default_run_id_and_resume(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = workspace["dir"] / "fresh.sqlite"
    args = ["-c", str(workspace["config"]), "-t", str(workspace["tasks"]), "--limit", "15"]
    assert main(["run", *args, "--ledger", str(ledger)]) == 0
    out = capsys.readouterr().out
    m = re.search(r"starting run (simulated-demo-serve-\d{8}-\d{6})\b", out)
    assert m, out
    run_id = m.group(1)
    # Without --run-id a new run starts (never an implicit resume) ...
    assert main(["run", *args, "--ledger", str(ledger)]) == 0
    assert "starting run" in capsys.readouterr().out
    # ... and the same --run-id resumes, skipping recorded tasks.
    assert main(["run", *args, "--ledger", str(ledger), "--run-id", run_id]) == 0
    out = capsys.readouterr().out
    assert f"resuming run {run_id}" in out and "15 resumed" in out
    with Ledger(ledger) as led:
        assert sum(1 for _ in led.decisions(run_id)) == 15
        assert len(led.runs()) == 2


def test_run_no_resume_reroutes(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["run", *_common(workspace), "--run-id", "r-nores", "--limit", "5", "-q"]
    assert main(base) == 0
    capsys.readouterr()
    assert main([*base, "--no-resume"]) == 0
    out = capsys.readouterr().out
    assert "5/5 done" in out and "resumed" not in out


def test_run_mode_mismatch_is_usage_error(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", *_common(workspace), "--run-id", "s1", "--mode", "eval", "-q"]) == 2
    assert "mode 'serve'" in capsys.readouterr().err


def test_run_budget_stop_exit_code(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["run", *_common(workspace), "--run-id", "r-budget", "--max-cost", "0.002",
         "--workers", "1", "-q"]
    )  # fmt: skip
    assert code == cli.EXIT_BUDGET == 4
    captured = capsys.readouterr()
    assert "(stopped: budget)" in captured.out
    assert "resume with" in captured.err and "--run-id r-budget" in captured.err


def test_run_config_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(DEMO_CONFIG.replace('backend = "slow"', 'backend = "slwo"'), encoding="utf-8")
    tasks = tmp_path / "t.jsonl"
    save_tasks(arithmetic(3), tasks)
    assert main(["run", "-c", str(bad), "-t", str(tasks), "--ledger", str(tmp_path / "l")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("shadowgate: error:") and "slwo" in err
    assert "Traceback" not in err
    assert not (tmp_path / "l").exists()


def test_run_invalid_toml(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[run\n", encoding="utf-8")
    assert main(["run", "-c", str(bad), "-t", str(tmp_path / "t.jsonl")]) == 2
    assert "invalid TOML" in capsys.readouterr().err


def test_run_missing_tasks_file(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", "-c", str(workspace["config"]), "-t", str(workspace["dir"] / "nope.jsonl")])
    assert code == 2
    assert "shadowgate: error:" in capsys.readouterr().err


def test_debug_env_shows_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHADOWGATE_DEBUG", "1")
    assert main(["run", "-c", str(tmp_path / "missing.toml"), "-t", "x.jsonl"]) == 2
    err = capsys.readouterr().err
    assert "Traceback" in err and "config file not found" in err


def test_interrupt_exit_code(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_cmd_runs", boom)
    parser_args = ["runs", "--ledger", str(workspace["ledger"])]
    # build_parser binds the function at parse time, so patch before calling main.
    assert main(parser_args) == 130
    assert "interrupted" in capsys.readouterr().err


def test_verbose_flag_accepted(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["-vv", "runs", "--ledger", str(workspace["ledger"])]) == 0


# --------------------------------------------------------------------------- audit


@needs_report
def test_audit_text(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", "--ledger", str(workspace["ledger"]), "--run-id", "s1"]) == 0
    out = capsys.readouterr().out
    assert "s1" in out and "Escalation rate" in out


def test_audit_json(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", "--ledger", str(workspace["ledger"]), "--run-id", "s1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == "s1" and data["mode"] == "serve"
    assert data["n_decisions"] == N_TASKS
    assert data["tolerance"] == 0.05  # taken from the run's stored config
    assert data["status"] in ("ok", "breach", "inconclusive", "no-data")


def test_audit_fail_on_breach(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # A permissive threshold keeps many wrong fast-tier answers: a clear breach.
    config = tmp_path / "loose.toml"
    config.write_text(DEMO_CONFIG.replace("threshold = 0.55", "threshold = 0.3"), encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    save_tasks(arithmetic(60, seed=1), tasks)
    ledger = tmp_path / "ledger.sqlite"
    args = ["-c", str(config), "-t", str(tasks), "--ledger", str(ledger), "--run-id", "b1", "-q"]
    assert main(["run", *args]) == 0
    capsys.readouterr()
    base = ["audit", "--ledger", str(ledger), "--run-id", "b1", "--json"]
    assert main([*base, "--tolerance", "0.01", "--fail-on-breach"]) == 3
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "breach"
    assert "breach" in captured.err
    # Without the flag a breach is reported but does not fail.
    assert main([*base, "--tolerance", "0.01"]) == 0
    capsys.readouterr()
    assert main([*base, "--tolerance", "1", "--fail-on-breach"]) == 0


def test_audit_missing_ledger_does_not_create_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "nope" / "ledger.sqlite"
    assert main(["audit", "--ledger", str(ledger)]) == 2
    err = capsys.readouterr().err
    assert "no ledger at" in err
    assert not ledger.exists() and not ledger.parent.exists()


def test_audit_default_ledger_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["runs"]) == 2
    assert "no ledger at" in capsys.readouterr().err
    assert not (tmp_path / ".shadowgate").exists()


def test_audit_unknown_run(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", "--ledger", str(workspace["ledger"]), "--run-id", "zzz"]) == 2
    assert "no run 'zzz'" in capsys.readouterr().err


def test_audit_run_pending(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "deferred.toml"
    config.write_text(_deferred_config(), encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    save_tasks(arithmetic(60, seed=3), tasks)
    ledger = tmp_path / "ledger.sqlite"
    args = ["-c", str(config), "-t", str(tasks), "--ledger", str(ledger), "--run-id", "d1"]
    assert main(["run", *args]) == 0
    out = capsys.readouterr().out
    assert "deferred audit(s) pending" in out
    with Ledger(ledger) as led:
        assert sum(1 for _ in led.pending_audits("d1")) > 0
    code = main(["audit", "--run-pending", "-c", str(config), "--ledger", str(ledger), "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_pending"] == 0 and data["n_audited"] > 0
    with Ledger(ledger) as led:
        assert sum(1 for _ in led.pending_audits("d1")) == 0


def test_audit_run_pending_needs_config(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["audit", "--run-pending", "--ledger", str(workspace["ledger"])]) == 2
    assert "-c/--config" in capsys.readouterr().err


# --------------------------------------------------------------------------- sweep


def test_sweep_text_defaults_to_latest_eval_run(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sweep", "--ledger", str(workspace["ledger"])]) == 0
    out = capsys.readouterr().out
    assert out.startswith("run e1")
    assert "Baselines" in out and "Recommendation" in out and "only:slow" in out


def test_sweep_json(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["sweep", "--ledger", str(workspace["ledger"]), "--run-id", "e1", "--json",
         "--objective", "min-accuracy", "--min-accuracy", "0.8", "--holdout", "0"]
    )  # fmt: skip
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == "e1" and data["n"] == N_TASKS
    assert data["tiers"] == ["fast", "slow"]
    assert data["recommendation"]["objective"] == "min-accuracy"
    assert data["recommendation"]["holdout"] is None


def test_sweep_on_serve_run_is_usage_error(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sweep", "--ledger", str(workspace["ledger"]), "--run-id", "s1"]) == 2
    assert "eval-mode" in capsys.readouterr().err


def test_sweep_min_accuracy_needs_value(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["sweep", "--ledger", str(workspace["ledger"]), "--objective", "min-accuracy"])
    assert code == 2
    assert "min_accuracy" in capsys.readouterr().err


# --------------------------------------------------------------------------- calibrate


def test_calibrate_text_defaults_to_latest_eval_run(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["calibrate", "--ledger", str(workspace["ledger"])]) == 0
    out = capsys.readouterr().out
    assert out.startswith("run e1 (eval), tier 'fast': 80 (confidence, correct) pairs")
    assert "before (raw scores)" in out and "2-fold cross-fit" in out
    assert '[[tiers]]  # replace the confidence line of tier "fast" with:' in out
    assert 'base = { type = "verbal" }' in out


def test_calibrate_json(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["calibrate", "--ledger", str(workspace["ledger"]), "--run-id", "e1", "--tier", "fast",
         "--max-knots", "5", "--json"]
    )  # fmt: skip
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == "e1" and data["tier"] == "fast" and data["truth"] == "reference"
    assert data["n"] == N_TASKS and data["base_spec"] == {"type": "verbal"}
    pts = data["points"]
    assert 1 <= len(pts) <= 5
    assert all(x1 > x0 and y1 >= y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False))
    assert all(round(v, 4) == v for p in pts for v in p)
    for key in ("before", "after", "cross_fitted"):
        assert set(data[key]) == {"ece", "brier"}
    # The isotonic map (even thinned) fits the sample at least as well as the raw scores.
    assert data["after"]["ece"] <= data["before"]["ece"] + 1e-9


def test_calibrate_snippet_round_trip(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    from shadowgate.confidence import Calibrated

    assert main(["calibrate", "--ledger", str(workspace["ledger"]), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    spec = tomllib.loads(data["snippet"])["tiers"][0]["confidence"]
    assert spec["type"] == "calibrated" and spec["base"] == {"type": "verbal"}
    assert [list(p) for p in spec["points"]] == data["points"]
    # The snippet drops into the config and builds a working cascade.
    raw = tomllib.loads(DEMO_CONFIG)
    raw["tiers"][0]["confidence"] = spec
    cascade, _ = parse_config(raw).build(tasks=arithmetic(5, seed=1))
    est = cascade.tiers[0].estimator
    assert isinstance(est, Calibrated)
    assert [list(p) for p in est.points] == data["points"]


def test_calibrate_bad_tier(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    base = ["calibrate", "--ledger", str(workspace["ledger"])]
    assert main([*base, "--tier", "nope"]) == 2
    assert "no tier 'nope'" in capsys.readouterr().err
    assert main([*base, "--tier", "slow"]) == 2
    assert "final tier" in capsys.readouterr().err
    assert main([*base, "--max-knots", "1"]) == 2
    assert "--max-knots" in capsys.readouterr().err


def test_calibrate_serve_run_without_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No references and no audits: a serve run has nothing to fit against.
    config = tmp_path / "noaudit.toml"
    config.write_text(DEMO_CONFIG.split("[audit]")[0], encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    save_tasks([dataclasses.replace(t, reference=None) for t in arithmetic(30, seed=1)], tasks)
    ledger = tmp_path / "ledger.sqlite"
    args = ["-c", str(config), "-t", str(tasks), "--ledger", str(ledger), "--run-id", "n1", "-q"]
    assert main(["run", *args]) == 0
    capsys.readouterr()
    assert main(["calibrate", "--ledger", str(ledger), "--run-id", "n1"]) == 2
    err = capsys.readouterr().err
    assert "serve mode" in err and "--mode eval" in err
    # Without --run-id the latest eval-mode run is required.
    assert main(["calibrate", "--ledger", str(ledger)]) == 2
    assert "no eval-mode run" in capsys.readouterr().err


# --------------------------------------------------------------------------- report / export


@needs_report
def test_report_eval_run_includes_sweep(
    workspace: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "eval.html"
    code = main(["report", "--ledger", str(workspace["ledger"]), "--run-id", "e1", "-o", str(out)])
    assert code == 0
    assert "with sweep of e1" in capsys.readouterr().out
    assert out.read_text(encoding="utf-8").lstrip().lower().startswith("<!doc")


@needs_report
def test_report_serve_run_with_sweep_run(
    workspace: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "serve.md"
    code = main(
        ["report", "--ledger", str(workspace["ledger"]), "--run-id", "s1",
         "--sweep-run-id", "e1", "-o", str(out)]
    )  # fmt: skip
    assert code == 0
    assert "run s1 with sweep of e1" in capsys.readouterr().out
    assert out.stat().st_size > 0
    plain = tmp_path / "plain.md"
    assert main(["report", "--ledger", str(workspace["ledger"]), "--run-id", "s1",
                 "-o", str(plain)]) == 0  # fmt: skip
    assert "with sweep" not in capsys.readouterr().out


def test_runs_text_and_json(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["runs", "--ledger", str(workspace["ledger"])]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^s1\s+serve\s+80\s", out, re.M)
    assert re.search(r"^e1\s+eval\s+80\s", out, re.M)
    assert main(["runs", "--ledger", str(workspace["ledger"]), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    by_id = {r["run_id"]: r for r in data}
    assert by_id["s1"]["mode"] == "serve" and by_id["e1"]["n_decisions"] == N_TASKS


def test_export_round_trip(
    workspace: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "s1.jsonl"
    code = main(["export", "--ledger", str(workspace["ledger"]), "--run-id", "s1", "-o", str(out)])
    assert code == 0
    assert f"wrote {N_TASKS} decisions of run s1" in capsys.readouterr().out
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == N_TASKS
    decisions = [Decision.from_dict(json.loads(line)) for line in lines]
    assert {d.run_id for d in decisions} == {"s1"}
    with Ledger(workspace["ledger"]) as led:
        assert [d.task.id for d in led.decisions("s1")] == [d.task.id for d in decisions]


# --------------------------------------------------------------------------- datasets


def test_datasets_make_and_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "d.jsonl"
    code = main(["datasets", "make", "arithmetic", "-n", "12", "--seed", "4", "-o", str(out),
                 "--min-steps", "2", "--max-steps", "3"])  # fmt: skip
    assert code == 0
    assert "wrote 12 arithmetic tasks" in capsys.readouterr().out
    tasks = load_tasks(out)
    assert len(tasks) == 12 and all(2 <= t.meta["steps"] <= 3 for t in tasks)
    assert main(["datasets", "show", str(out), "--limit", "2"]) == 0
    shown = capsys.readouterr().out
    assert "12 tasks, 12 with references" in shown
    assert shown.count("reference:") == 2


def test_datasets_unknown_generator(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["datasets", "make", "nope", "-n", "3", "-o", str(tmp_path / "x.jsonl")]) == 2
    assert "unknown dataset generator" in capsys.readouterr().err


# --------------------------------------------------------------------------- console encoding


def test_ascii_console_does_not_crash(workspace: dict[str, Path]) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "ascii"}
    env.pop("SHADOWGATE_DEBUG", None)
    proc = subprocess.run(
        [sys.executable, "-m", "shadowgate", "run", *_common(workspace),
         "--run-id", "r-ascii", "--limit", "3", "-q"],
        capture_output=True,
        env=env,
        timeout=120,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert b"3/3 done" in proc.stdout


# --------------------------------------------------------------------------- import / --level


def test_import_round_trip(
    workspace: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = str(workspace["ledger"])
    s1, e1 = tmp_path / "s1.jsonl", tmp_path / "e1.jsonl"
    assert main(["export", "--ledger", src, "--run-id", "s1", "-o", str(s1)]) == 0
    assert main(["export", "--ledger", src, "--run-id", "e1", "-o", str(e1)]) == 0
    capsys.readouterr()
    fresh = tmp_path / "sub" / "fresh.sqlite"
    assert main(["import", str(s1), str(e1), "--ledger", str(fresh)]) == 0
    out = capsys.readouterr().out
    assert f"imported {2 * N_TASKS} decisions (runs: s1, e1)" in out
    assert fresh.is_file()

    def audit_json(ledger: str) -> dict:
        args = ["audit", "--ledger", ledger, "--run-id", "s1", "--tolerance", "0.05", "--json"]
        assert main(args) == 0
        return json.loads(capsys.readouterr().out)

    assert audit_json(str(fresh)) == audit_json(src)
    # Re-importing upserts: no duplicates.
    assert main(["import", str(s1), "--ledger", str(fresh)]) == 0
    capsys.readouterr()
    with Ledger(fresh) as led:
        assert {r.run_id: r.n_decisions for r in led.runs()} == {"s1": N_TASKS, "e1": N_TASKS}


@needs_report
def test_import_round_trip_report_identical(
    workspace: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import datetime as dt

    from shadowgate import report
    from shadowgate.audit import summarize
    from shadowgate.sweep import sweep

    exported = tmp_path / "e1.jsonl"
    assert main(["export", "--ledger", str(workspace["ledger"]), "--run-id", "e1",
                 "-o", str(exported)]) == 0  # fmt: skip
    fresh = tmp_path / "fresh.sqlite"
    assert main(["import", str(exported), "--ledger", str(fresh)]) == 0
    now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    html = []
    for i, path in enumerate((workspace["ledger"], fresh)):
        with Ledger(path) as led:
            ds = list(led.decisions("e1"))
        out = report.write_report(
            tmp_path / f"r{i}.html", summarize(ds, tolerance=0.05), sweep(ds), now=now
        )
        html.append(out.read_text(encoding="utf-8"))
    assert html[0] == html[1]


def test_import_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger = tmp_path / "l.sqlite"
    assert main(["import", str(tmp_path / "nope.jsonl"), "--ledger", str(ledger)]) == 2
    assert "no such file" in capsys.readouterr().err
    assert not ledger.exists()


def test_import_invalid_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"run_id": "x"}\n', encoding="utf-8")
    assert main(["import", str(bad), "--ledger", str(tmp_path / "l.sqlite")]) == 1
    assert "invalid decision record" in capsys.readouterr().err


@pytest.mark.parametrize("cmd", ["audit", "sweep", "report"])
@pytest.mark.parametrize("bad", ["0.5", "1", "1.2", "x"])
def test_level_validation(
    workspace: dict[str, Path], tmp_path: Path, cmd: str, bad: str,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    extra = ["-o", str(tmp_path / "r.md")] if cmd == "report" else []
    args = [cmd, "--ledger", str(workspace["ledger"]), "--level", bad, *extra]
    assert main(args) == 2
    assert "level" in capsys.readouterr().err


def test_level_passed_through(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = str(workspace["ledger"])
    assert main(["audit", "--ledger", ledger, "--run-id", "s1", "--json"]) == 0
    base = json.loads(capsys.readouterr().out)
    assert main(["audit", "--ledger", ledger, "--run-id", "s1", "--json", "--level", "0.99"]) == 0
    strict = json.loads(capsys.readouterr().out)
    assert base["level"] == 0.95 and strict["level"] == 0.99
    b, s = base["escalation_rate"], strict["escalation_rate"]
    assert s["lo"] <= b["lo"] and s["hi"] >= b["hi"] and (s["hi"] - s["lo"]) > (b["hi"] - b["lo"])
    assert main(["sweep", "--ledger", ledger, "--json", "--level", "0.99"]) == 0
    sw = json.loads(capsys.readouterr().out)
    assert sw["baselines"]["only:slow"]["accuracy"]["level"] == 0.99


def test_level_help_mentions_repeated_checks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", "--help"]) == 0
    assert "0.99" in capsys.readouterr().out


def test_cli_output_is_ascii(workspace: dict[str, Path]) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    env.pop("SHADOWGATE_DEBUG", None)
    proc = subprocess.run(
        [sys.executable, "-m", "shadowgate", "run", *_common(workspace),
         "--run-id", "r-cp1252", "--limit", "3"],
        capture_output=True,
        env=env,
        timeout=120,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.isascii() and proc.stderr.isascii()
    assert b"3/3 done | " in proc.stdout
