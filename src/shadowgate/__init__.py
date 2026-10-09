"""shadowgate: confidence-gated model cascades that audit what they skip.

A :class:`Cascade` routes each :class:`Task` through model tiers: a cheap tier answers first,
a confidence estimator scores the answer, and low-confidence answers escalate to the next
tier. Accepted ("skipped") answers are shadow-audited by the reference tier with known
inclusion probabilities, so :func:`summarize` can estimate how often the fast path is wrong
on the cases it keeps, with confidence intervals. Eval-mode runs feed :func:`run_sweep`, which
draws the accuracy/cost Pareto curve and recommends thresholds.

Typical use::

    import shadowgate as sg

    cascade, settings = sg.load_config("cascade.toml").build()
    with sg.Ledger(settings.ledger_path) as ledger:
        sg.run_tasks(cascade, sg.load_tasks("tasks.jsonl"), ledger, run_id="r1")
        print(sg.summarize(ledger.decisions("r1")).disagreement)

Submodules ``backends``, ``confidence``, ``compare``, ``extract`` and ``stats`` hold the
pluggable pieces. Report helpers (:func:`render_text`, :func:`render_markdown`,
:func:`render_html`, :func:`write_report`) and the ``report`` module load on first access.
Importing this package never imports an optional provider SDK such as ``anthropic``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from . import backends, compare, confidence, extract, stats
from .audit import AuditSummary, BinSummary, summarize
from .backends import make_backend
from .cascade import AuditPolicy, Cascade, Tier
from .config import Config, RunSettings, load_config, parse_config
from .datasets import load_tasks, save_tasks
from .errors import (
    BackendError,
    BudgetExceeded,
    ConfigError,
    DatasetError,
    InsufficientData,
    LedgerError,
    ShadowgateError,
)
from .ledger import Ledger, RunInfo
from .runner import RunStats, run_pending_audits
from .runner import run as run_tasks
from .stats import Estimate
from .sweep import OperatingPoint, Recommendation, SweepResult, fit_isotonic
from .sweep import sweep as run_sweep
from .types import (
    Attempt,
    Backend,
    Comparator,
    Completion,
    ConfidenceEstimator,
    ConfidenceResult,
    Decision,
    Judgement,
    Request,
    ShadowResult,
    Task,
    Usage,
)

if TYPE_CHECKING:
    from . import report
    from .report import render_html, render_markdown, render_text, write_report

__version__ = "0.1.0"

__all__ = [
    # core records and protocols
    "Task",
    "Request",
    "Usage",
    "Completion",
    "Attempt",
    "ShadowResult",
    "Decision",
    "Judgement",
    "ConfidenceResult",
    "Backend",
    "ConfidenceEstimator",
    "Comparator",
    # routing
    "Cascade",
    "Tier",
    "AuditPolicy",
    "make_backend",
    # running and recording
    "Ledger",
    "RunInfo",
    "RunStats",
    "run_tasks",
    "run_pending_audits",
    "load_tasks",
    "save_tasks",
    # configuration
    "Config",
    "RunSettings",
    "load_config",
    "parse_config",
    # analysis
    "summarize",
    "AuditSummary",
    "BinSummary",
    "Estimate",
    "run_sweep",
    "SweepResult",
    "OperatingPoint",
    "Recommendation",
    "fit_isotonic",
    # reports (loaded on first access)
    "render_text",
    "render_markdown",
    "render_html",
    "write_report",
    # errors
    "ShadowgateError",
    "ConfigError",
    "BackendError",
    "BudgetExceeded",
    "LedgerError",
    "DatasetError",
    "InsufficientData",
    # submodules
    "backends",
    "compare",
    "confidence",
    "extract",
    "stats",
    "report",
]

# name -> (module, attribute or None for the module itself), resolved by __getattr__.
_LAZY: dict[str, tuple[str, str | None]] = {
    "report": ("shadowgate.report", None),
    "render_text": ("shadowgate.report", "render_text"),
    "render_markdown": ("shadowgate.report", "render_markdown"),
    "render_html": ("shadowgate.report", "render_html"),
    "write_report": ("shadowgate.report", "write_report"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'shadowgate' has no attribute {name!r}")
    module = importlib.import_module(target[0])
    value = module if target[1] is None else getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
