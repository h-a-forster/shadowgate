"""The top-level ``shadowgate`` namespace: exports, laziness and version."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

import shadowgate

ROOT = Path(__file__).resolve().parents[1]
REPORT_NAMES = {"report", "render_text", "render_markdown", "render_html", "write_report"}
HAS_REPORT = importlib.util.find_spec("shadowgate.report") is not None


def test_all_has_no_duplicates() -> None:
    assert len(shadowgate.__all__) == len(set(shadowgate.__all__))


def test_all_names_resolve() -> None:
    for name in shadowgate.__all__:
        if name in REPORT_NAMES and not HAS_REPORT:
            continue
        assert getattr(shadowgate, name) is not None, name


@pytest.mark.skipif(not HAS_REPORT, reason="shadowgate.report is not available")
def test_report_names_resolve_lazily() -> None:
    from shadowgate import render_html, render_markdown, render_text, write_report

    for fn in (render_text, render_markdown, render_html, write_report):
        assert callable(fn)
    assert isinstance(shadowgate.report, types.ModuleType)


def test_no_private_names_exported() -> None:
    assert [n for n in shadowgate.__all__ if n.startswith("_")] == []


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        _ = shadowgate.does_not_exist  # type: ignore[attr-defined]


def test_submodules_and_key_names() -> None:
    for mod in ("backends", "compare", "confidence", "extract", "stats"):
        assert isinstance(getattr(shadowgate, mod), types.ModuleType), mod
    from shadowgate import runner, sweep

    assert shadowgate.run_sweep is sweep.sweep

    assert shadowgate.run_tasks is runner.run
    assert issubclass(shadowgate.ConfigError, shadowgate.ShadowgateError)
    assert issubclass(shadowgate.BackendError, shadowgate.ShadowgateError)


SUBMODULES = [
    p.stem if p.is_file() else p.name
    for p in sorted((ROOT / "src" / "shadowgate").iterdir())
    if (p.suffix == ".py" and not p.stem.startswith("_")) or (p / "__init__.py").is_file()
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_no_export_shadows_a_submodule(name: str) -> None:
    if name == "report" and not HAS_REPORT:
        pytest.skip(f"shadowgate.{name} is not imported by this check")
    module = importlib.import_module(f"shadowgate.{name}")
    assert isinstance(module, types.ModuleType)
    attr = getattr(shadowgate, name, module)
    assert attr is module, f"shadowgate.{name} is {attr!r}, not the submodule"


def test_dir_lists_public_names() -> None:
    assert set(shadowgate.__all__) <= set(dir(shadowgate))


def test_version_matches_pyproject() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = meta["project"]
    if "version" in project:
        assert shadowgate.__version__ == project["version"]
        return
    assert "version" in project.get("dynamic", []), "pyproject has no static or dynamic version"
    try:
        installed = importlib.metadata.version(project["name"])
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"distribution {project['name']!r} is not installed")
    assert shadowgate.__version__ == installed


def test_import_does_not_load_optional_or_lazy_modules() -> None:
    code = (
        "import sys, shadowgate\n"
        "bad = sorted(m for m in sys.modules if m == 'anthropic' or m.startswith('anthropic.')"
        " or m == 'shadowgate.report' or m == 'shadowgate.backends.anthropic')\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip() == ""
