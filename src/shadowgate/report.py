"""Reports: Markdown, self-contained HTML and compact terminal text.

:func:`render_html` and :func:`render_markdown` turn an :class:`~shadowgate.audit.AuditSummary`
(and optionally a :class:`~shadowgate.sweep.SweepResult`) into a report with the same sections:

1. headline: status badge, skipped-case disagreement / error with confidence intervals, key
   numbers (escalation rate, cost per task, savings, audit overhead, expected wrong answers);
2. audit by confidence bin (table, plus a chart in HTML);
3. tier usage, audit coverage and costs;
4. threshold sweep (when given): Pareto chart, recommendation, baselines, frontier, calibration;
5. caveats and methodology;
6. footer (version, UTC timestamp, run id, mode).

:func:`render_text` is a compact plain-text summary for terminals (at most 100 columns, no
colour codes). :func:`write_report` picks the format from the file suffix.

All model- and dataset-derived strings are escaped for the output format. The HTML report is a
single file with inline CSS and inline SVG only: no scripts, fonts or other external requests.
"""

from __future__ import annotations

import html
import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from shadowgate import __version__
from shadowgate.audit import AuditSummary, BinSummary
from shadowgate.stats import Estimate
from shadowgate.svg import (
    CHART_CSS,
    Diagonal,
    ErrorBar,
    Marker,
    RefLine,
    Series,
    bar_with_ci,
    format_value,
    line_chart,
    scatter,
    xy_chart,
)
from shadowgate.sweep import NEVER_ACCEPT, OperatingPoint, SweepResult

__all__ = [
    "DASH",
    "fmt_ci",
    "fmt_count",
    "fmt_estimate",
    "fmt_pct",
    "fmt_usd",
    "render_html",
    "render_markdown",
    "render_text",
    "write_report",
]

DASH = "—"  # em dash: missing value
_NDASH = "–"  # en dash: ranges
_ARROW = " → "
TEXT_WIDTH = 100
MAX_FRONTIER_ROWS = 15
LOG_X_SPAN = 20.0  # use a log cost axis when costs span more than this factor

_FORMATS = {
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}
_FMT_ALIASES = {
    "html": "html",
    "htm": "html",
    "md": "markdown",
    "markdown": "markdown",
    "text": "text",
    "txt": "text",
}

_STATUS_LABEL = {
    "ok": "OK",
    "breach": "BREACH",
    "inconclusive": "INCONCLUSIVE",
    "no-data": "NO DATA",
    "n/a": "N/A",
}

METHODOLOGY = (
    "Disagreement is the share of skipped cases (answered by a non-final tier without "
    "escalating) whose answer differs from the reference tier's answer to the same task, as "
    "decided by the configured comparator or judge. In serve mode only a random sample of skipped "
    "cases is shadow-audited; each audited case is weighted by 1/π, the inverse of its known "
    "inclusion probability (a Hajek inverse-probability-weighted estimator), so the estimate "
    "describes all skipped cases rather than only the audited ones. In eval mode every tier "
    "answers every task, so every skipped case is observed (π = 1). The audit tier is a "
    "proxy, not ground truth: disagreement counts differences from its answers, not verified "
    "errors, unless reference answers exist, in which case the error rate against them is "
    "reported as well. Audits that are pending, failed or undecided are treated as "
    "nonresponse: within each inclusion-probability stratum, completed audits are up-weighted "
    "by the inverse of the stratum's response rate, which assumes audits are missing at random "
    "within their stratum; strata without any completed audit cannot be represented and are "
    "excluded from the estimate (a caveat says so). Intervals are Wilson score intervals for "
    "unweighted proportions and Korn-Graubard intervals (Clopper-Pearson at the effective sample "
    "size n_eff) for weighted ones; when a stratum holding at least 10% of the skipped cases has "
    "fewer than 10 completed audits, the status is never reported as ok. The status uses a "
    "fixed-sample interval: checking it repeatedly as audits accumulate and stopping at the "
    "first ok inflates the chance of a false ok, so fix the number of audits in advance or use a "
    "stricter confidence level for repeated checks. The savings interval reflects only the "
    "uncertainty in the cost of always using the reference tier. Sweep accuracies are Wilson "
    "intervals; recommended thresholds are chosen on a selection split and re-evaluated on a "
    "held-out split."
)


# --------------------------------------------------------------------------- number formatting


def _finite(x: object) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _strip_neg_zero(s: str) -> str:
    if s.startswith("-") and not any(c in "123456789" for c in s):
        return s[1:]
    return s


def _pct_digits(p: float) -> int:
    a = abs(p)
    if a == 0 or a == 100:
        return 0
    return 2 if a < 1 else 1


def fmt_pct(x: float | None, digits: int | None = None, *, none: str = DASH) -> str:
    """Format a fraction as a percentage: 0.074 -> "7.4%", 0.0086 -> "0.86%", None -> "—"."""
    v = _finite(x)
    if v is None:
        return none
    p = v * 100.0
    d = _pct_digits(p) if digits is None else digits
    return _strip_neg_zero(f"{p:.{d}f}") + "%"


def fmt_ci(lo: float | None, hi: float | None, *, none: str = DASH) -> str:
    """Format an interval of fractions: (0.049, 0.11) -> "4.9–11.0%"."""
    a, b = _finite(lo), _finite(hi)
    if a is None or b is None:
        return none
    m = max(abs(a), abs(b)) * 100.0
    d = 0 if m == 0 else 2 if m < 1 else 1
    return f"{_strip_neg_zero(f'{a * 100:.{d}f}')}{_NDASH}{_strip_neg_zero(f'{b * 100:.{d}f}')}%"


def _fmt_tol(x: float | None) -> str:
    """Tolerance without a needless trailing ".0": 0.03 -> "3%", 0.025 -> "2.5%"."""
    out = fmt_pct(x)
    return out.replace(".0%", "%") if out.endswith(".0%") else out


def _level_txt(level: float | None) -> str:
    lv = _finite(level)
    return f"{lv * 100:g}% CI" if lv is not None else "CI"


def fmt_estimate(est: Estimate | None, *, none: str = DASH, ci: bool = True) -> str:
    """ "7.4% (95% CI 4.9–11.0%)"; the interval is omitted when unknown."""
    if est is None or _finite(est.value) is None:
        return none
    out = fmt_pct(est.value)
    if ci and _finite(est.lo) is not None and _finite(est.hi) is not None:
        out += f" ({_level_txt(est.level)} {fmt_ci(est.lo, est.hi)})"
    return out


def fmt_usd(x: float | None, *, none: str = DASH) -> str:
    """Smart-precision currency: 0.0042 -> "$0.0042", 1.5 -> "$1.50", None -> "—"."""
    v = _finite(x)
    if v is None:
        return none
    return format_value(v, "currency")


def fmt_count(n: int | float | None, *, none: str = DASH) -> str:
    v = _finite(n)
    if v is None:
        return none
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.1f}" if abs(v) < 10 else f"{v:,.0f}"


def _fmt_thr(t: float) -> str:
    if t >= NEVER_ACCEPT - 1e-9:
        return "never accept"
    s = f"{t:.3f}".rstrip("0").rstrip(".")
    return s or "0"


def _fmt_thresholds(tiers: Sequence[str], thresholds: Sequence[float]) -> str:
    if not thresholds:
        return DASH
    parts = []
    for name, t in zip(tiers, thresholds, strict=False):
        if t >= NEVER_ACCEPT - 1e-9:
            parts.append(f"{name}: always escalate")
        else:
            parts.append(f"{name} ≥ {_fmt_thr(t)}")
    return ", ".join(parts)


def _fmt_pp(x: float | None) -> str:
    v = _finite(x)
    if v is None:
        return DASH
    return _strip_neg_zero(f"{v * 100:+.1f}") + " pp"


def _fmt_delta(est: Estimate | None) -> str:
    if est is None or _finite(est.value) is None:
        return DASH
    out = _fmt_pp(est.value)
    if _finite(est.lo) is not None and _finite(est.hi) is not None:
        lo = _strip_neg_zero(f"{est.lo * 100:+.1f}")  # type: ignore[operator]
        hi = _strip_neg_zero(f"{est.hi * 100:+.1f}")  # type: ignore[operator]
        out += f" ({_level_txt(est.level)} {lo} to {hi} pp)"
    return out


def _fmt_neff(est: Estimate) -> str | None:
    """n_eff text when the estimate is weighted (n_eff differs from n)."""
    ne = _finite(est.n_eff)
    if ne is None or abs(ne - est.n) < 0.5:
        return None
    return f"n_eff {ne:.0f}"


# --------------------------------------------------------------------------- document model


@dataclass
class _Table:
    caption: str
    headers: list[str]
    rows: list[list[str]]
    numeric: set[int] = field(default_factory=set)  # right-aligned column indices
    highlight: set[int] = field(default_factory=set)  # emphasised row indices
    note: str = ""


Segment = tuple[str, bool]  # (text, strong)


@dataclass
class _Block:
    kind: str  # p | cards | table | charts | rec | list | h3 | badge | muted
    data: object


@dataclass
class _Section:
    id: str
    title: str
    blocks: list[_Block]


def _p(*segs: Segment | str) -> _Block:
    return _Block("p", [(s, False) if isinstance(s, str) else s for s in segs])


# --------------------------------------------------------------------------- shared content


def _non_final(summary: AuditSummary) -> list[str]:
    return [t for t in summary.tiers if t != summary.final_tier]


def _ref_name(summary: AuditSummary) -> str:
    return summary.reference_tier or summary.final_tier or "the reference tier"


def _metric_name(summary: AuditSummary) -> str:
    if summary.status_metric == "skipped_error":
        return "error vs references"
    return "disagreement"


def _status_parts(summary: AuditSummary) -> tuple[str, str]:
    """(badge label, explanation)."""
    status = summary.status
    label = _STATUS_LABEL.get(status, str(status).upper())
    tol = _fmt_tol(summary.tolerance)
    metric = (
        summary.skipped_error
        if summary.status_metric == "skipped_error"
        else (summary.disagreement)
    )
    name = _metric_name(summary)
    if status == "ok" and metric is not None:
        why = (
            f"Skipped-case {name} is within tolerance {tol}: the upper bound "
            f"{fmt_pct(metric.hi)} is at or below it."
        )
    elif status == "breach" and metric is not None:
        why = (
            f"Skipped-case {name} exceeds tolerance {tol}: the lower bound "
            f"{fmt_pct(metric.lo)} is above it."
        )
    elif status == "inconclusive":
        why = f"The interval for skipped-case {name} straddles tolerance {tol}."
        tasks = summary.tasks_to_resolve
        if summary.audits_to_resolve:
            why += (
                f" About {fmt_count(summary.audits_to_resolve)} more audits would resolve it "
                "if the rate holds."
            )
        elif tasks:
            why += (
                " More audits cannot help (every skipped case is already graded or audited); "
                f"more tasks are needed to resolve it: about {fmt_count(tasks)} more skipped "
                "cases if the rate holds."
            )
    elif status == "no-data":
        why = "No completed audits or graded skipped cases, so the skipped-case rate is unknown."
    elif status == "n/a":
        why = "No tolerance set; the status is not evaluated."
    else:
        why = ""
    return label, why


def _headline_segments(summary: AuditSummary) -> list[Segment]:
    s = summary
    segs: list[Segment] = []
    if s.n_decisions == 0:
        return [("No decisions were recorded for this run.", False)]
    ref = _ref_name(s)
    lvl = _level_txt(s.level)
    non_final = _non_final(s)
    if len(s.tiers) <= 1 and s.final_tier is not None:
        return [
            (
                f"Single-tier run: every task was served by {s.final_tier}; nothing was kept "
                "early, so there is nothing to audit.",
                False,
            )
        ]
    share = s.n_skipped / s.n_decisions
    who = (
        f"The fast tier ({non_final[0]})"
        if len(non_final) == 1
        else f"The non-final tiers ({', '.join(non_final)})"
        if non_final
        else "The fast tiers"
    )
    segs.append((f"{who} answered ", False))
    segs.append((fmt_pct(share, 0 if share >= 0.1 or share == 0 else 1), True))
    segs.append((f" of {fmt_count(s.n_decisions)} tasks without escalating.", False))
    d = s.disagreement
    if s.n_skipped == 0:
        return segs
    if d is not None and _finite(d.value) is not None:
        extra = [f"{fmt_count(s.n_audited)} audit{'s' if s.n_audited != 1 else ''}"]
        ne = _fmt_neff(d)
        if ne:
            extra.append(ne)
        segs.append(
            (
                f" On those, {'it disagrees' if len(non_final) <= 1 else 'they disagree'} "
                f"with {ref} on an estimated ",
                False,
            )
        )
        segs.append((fmt_pct(d.value), True))
        segs.append((f" ({lvl} {fmt_ci(d.lo, d.hi)}, {', '.join(extra)}).", False))
    e = s.skipped_error
    if e is not None and _finite(e.value) is not None:
        lead = " Against reference answers, the kept answers are wrong on "
        segs.append((lead, False))
        segs.append((fmt_pct(e.value), True))
        segs.append(
            (f" ({lvl} {fmt_ci(e.lo, e.hi)}, {fmt_count(s.n_graded_skipped)} graded).", False)
        )
    if (d is None or _finite(d.value) is None) and (e is None or _finite(e.value) is None):
        segs.append(
            (
                " No completed audits yet, so how often those kept answers are wrong cannot "
                "be estimated.",
                False,
            )
        )
    w = s.expected_wrong_skipped
    if w is not None and all(_finite(v) is not None for v in w):
        what = (
            "wrong answers"
            if s.expected_wrong_source == "skipped_error"
            else f"answers that differ from {ref}"
        )
        segs.append(
            (
                f" That is about {_fmt_num(w[0], w[2])} {what} among the "
                f"{fmt_count(s.n_skipped)} kept ({_fmt_num(w[1], w[2])}{_NDASH}{_fmt_num(w[2])}).",
                False,
            )
        )
    return segs


def _fmt_num(x: float, scale: float | None = None) -> str:
    """One decimal below 100 (judged on ``scale``, default ``x``), none above."""
    return f"{x:,.1f}" if abs(x if scale is None else scale) < 100 else f"{x:,.0f}"


def _cards(summary: AuditSummary) -> list[tuple[str, str, str]]:
    s = summary
    ref = _ref_name(s)
    cards: list[tuple[str, str, str]] = []
    share = s.n_skipped / s.n_decisions if s.n_decisions else None
    cards.append(
        (
            "Kept by fast tier",
            fmt_pct(share),
            f"{fmt_count(s.n_skipped)} of {fmt_count(s.n_decisions)} tasks",
        )
    )
    esc = s.escalation_rate
    cards.append(
        (
            "Escalation rate",
            fmt_pct(esc.value if esc else None),
            f"{_level_txt(esc.level)} {fmt_ci(esc.lo, esc.hi)}" if esc else "",
        )
    )
    if s.served_accuracy is not None and _finite(s.served_accuracy.value) is not None:
        a = s.served_accuracy
        cards.append(
            (
                "Served accuracy",
                fmt_pct(a.value),
                f"{_level_txt(a.level)} {fmt_ci(a.lo, a.hi)}, {fmt_count(a.n)} graded",
            )
        )
    sub = f"{fmt_count(s.n_unknown_cost)} with unknown cost" if s.n_unknown_cost else ""
    cards.append(("Cost per task", fmt_usd(s.cost_per_task, none="unknown"), sub))
    sav = s.est_savings
    if sav is not None and _finite(sav.value) is not None:
        sub = f"{_level_txt(sav.level)} {fmt_ci(sav.lo, sav.hi)}"
        slow = s.est_all_slow_cost_per_task
        if slow is not None and _finite(slow.value) is not None:
            sub += f"; always {ref}: {fmt_usd(slow.value)}/task"
        cards.append((f"Savings vs always {ref}", fmt_pct(sav.value), sub))
    else:
        cards.append((f"Savings vs always {ref}", "unknown", ""))
    sub = f"{fmt_usd(s.cost_audit)} audit spend" if s.cost_audit is not None else ""
    cards.append(("Audit overhead", fmt_pct(s.audit_overhead, none="unknown"), sub))
    w = s.expected_wrong_skipped
    if w is not None and all(_finite(v) is not None for v in w):
        label = (
            "Expected wrong-but-kept"
            if s.expected_wrong_source == "skipped_error"
            else "Expected disagreeing-but-kept"
        )
        cards.append(
            (
                label,
                _fmt_num(w[0], w[2]),
                f"{_level_txt(s.level)} {_fmt_num(w[1], w[2])}{_NDASH}{_fmt_num(w[2])}",
            )
        )
    return cards


def _est_cell(est: Estimate | None) -> str:
    if est is None or _finite(est.value) is None:
        return DASH
    ci = fmt_ci(est.lo, est.hi)
    return f"{fmt_pct(est.value)} ({ci})" if ci != DASH else fmt_pct(est.value)


def _bins_all(summary: AuditSummary) -> list[BinSummary]:
    bins = list(summary.bins)
    if summary.no_score_bin is not None:
        bins.append(summary.no_score_bin)
    return bins


def _bins_table(summary: AuditSummary) -> _Table:
    bins = _bins_all(summary)
    has_err = any(b.error is not None for b in bins)
    headers = ["Confidence bin", "Accepted", "Audited", "Disagreement (CI)"]
    if has_err:
        headers.append("Error vs references (CI)")
    rows = []
    hi_rows: set[int] = set()
    tol = summary.tolerance
    for i, b in enumerate(bins):
        row = [
            b.label or f"[{b.lo:.2f}, {b.hi:.2f})",
            fmt_count(b.n_accepted),
            fmt_count(b.n_audited),
            _est_cell(b.disagreement if b.n_audited else None),
        ]
        if has_err:
            row.append(_est_cell(b.error))
        rows.append(row)
        lo = _finite(b.disagreement.lo) if b.n_audited else None
        if tol is not None and lo is not None and lo > tol:
            hi_rows.add(i)
    return _Table(
        caption=f"Skipped cases by served confidence ({_level_txt(summary.level)} in brackets)",
        headers=headers,
        rows=rows,
        numeric={1, 2, 3, 4},
        highlight=hi_rows,
        note="Highlighted bins have a disagreement interval entirely above the tolerance."
        if hi_rows
        else "",
    )


def _bins_chart(summary: AuditSummary, *, narrow: bool = False) -> str | None:
    bins = _bins_all(summary)
    use_dis = any(b.n_audited for b in bins)
    use_err = not use_dis and any(b.error is not None for b in bins)
    if not (use_dis or use_err):
        return None
    cats, vals, lows, highs, notes, styles = [], [], [], [], [], []
    tol = summary.tolerance
    for b in bins:
        if narrow and b.label != "no score" and math.isfinite(b.lo):
            cats.append(f"{b.lo:.2f}")
        else:
            cats.append(b.label or f"{b.lo:.2f}")
        est = (b.disagreement if b.n_audited else None) if use_dis else b.error
        if est is None or _finite(est.value) is None:
            vals.append(None)
            lows.append(None)
            highs.append(None)
        else:
            vals.append(est.value)
            lows.append(est.lo)
            highs.append(est.hi)
        notes.append(f"n={b.n_audited if use_dis else b.n_graded}")
        lo = _finite(est.lo) if est is not None and _finite(est.value) is not None else None
        styles.append("accent" if tol is not None and lo is not None and lo > tol else 1)
    metric = "disagreement" if use_dis else "error vs references"
    ref_line = RefLine("y", tol, f"tolerance {_fmt_tol(tol)}") if tol is not None else None
    return bar_with_ci(
        cats,
        vals,
        lows,
        highs,
        title=f"Skipped-case {metric} by confidence bin",
        x_label="served confidence (bin start)" if narrow else "served confidence",
        y_label=metric,
        width=360 if narrow else 640,
        height=300 if narrow else 320,
        y_format="percent",
        ref_line=ref_line,
        annotations=notes,
        styles=styles,
        desc=f"Bars show the {metric} per bin with {_level_txt(summary.level)} whiskers"
        + ("; the dashed line is the tolerance." if ref_line else "."),
    )


def _responsive(wide: str, narrow: str) -> str:
    """Two renderings of one chart; CSS shows the narrow one on small screens."""
    return f'<div class="chart-wide">{wide}</div><div class="chart-narrow">{narrow}</div>'


def _tier_table(summary: AuditSummary) -> _Table:
    s = summary
    names = list(s.tiers) + [t for t in s.tier_share if t not in s.tiers]
    served_total = sum(s.tier_share.values())
    rows = []
    for t in names:
        roles = []
        if t == s.final_tier:
            roles.append("final")
        elif t in s.tiers:
            roles.append("fast" if t == (s.tiers[0] if s.tiers else None) else "intermediate")
        if t == s.reference_tier:
            roles.append("reference")
        n = s.tier_share.get(t, 0)
        rows.append(
            [
                t,
                ", ".join(roles) or DASH,
                fmt_count(n),
                fmt_pct(n / served_total if served_total else None),
            ]
        )
    note = f"{fmt_count(s.n_errors)} decision(s) failed and are not counted." if s.n_errors else ""
    return _Table(
        "Decisions served per tier",
        ["Tier", "Role", "Served", "Share"],
        rows,
        numeric={2, 3},
        note=note,
    )


def _coverage_table(summary: AuditSummary) -> _Table:
    s = summary
    rows = [
        ["Skipped (kept early)", fmt_count(s.n_skipped)],
        ["Audited", fmt_count(s.n_audited)],
        ["Not selected for audit", fmt_count(s.n_not_selected)],
        ["Audit pending", fmt_count(s.n_pending)],
        ["Audit errors", fmt_count(s.n_audit_errors)],
        ["Undecided (judge could not decide)", fmt_count(s.n_undecided)],
        ["No audit record", fmt_count(s.n_no_audit_record)],
        ["Graded against references", fmt_count(s.n_graded_skipped)],
    ]
    if s.disagreement_unweighted is not None and s.mode != "eval":
        rows.append(["Unweighted disagreement (biased)", _est_cell(s.disagreement_unweighted)])
    if s.audit_tier_error is not None and _finite(s.audit_tier_error.value) is not None:
        rows.append(
            [f"Audit tier ({_ref_name(s)}) error vs references", _est_cell(s.audit_tier_error)]
        )
    return _Table("Audit coverage of skipped cases", ["Quantity", "Count"], rows, numeric={1})


def _cost_table(summary: AuditSummary) -> _Table:
    s = summary
    slow = s.est_all_slow_cost_per_task
    slow_txt = DASH
    if slow is not None and _finite(slow.value) is not None:
        slow_txt = fmt_usd(slow.value)
        if _finite(slow.lo) is not None and _finite(slow.hi) is not None:
            slow_txt += f" ({fmt_usd(slow.lo)}{_NDASH}{fmt_usd(slow.hi)})"
    rows = [
        ["Serving cost (total)", fmt_usd(s.cost_serving)],
        ["Audit cost (total)", fmt_usd(s.cost_audit)],
        ["Cost per task", fmt_usd(s.cost_per_task)],
        [f"Est. cost per task, always {_ref_name(s)}", slow_txt],
        ["Estimated savings", _est_cell(s.est_savings)],
        ["Audit overhead (audit / serving)", fmt_pct(s.audit_overhead)],
        ["Decisions with unknown serving cost", fmt_count(s.n_unknown_cost)],
        ["Decisions with unknown audit cost", fmt_count(s.n_unknown_audit_cost)],
    ]
    return _Table("Costs (USD)", ["Quantity", "Value"], rows, numeric={1})


# --------------------------------------------------------------------------- sweep content


def _acc_label(sweep: SweepResult) -> str:
    if sweep.truth == "audit-tier" and sweep.tiers:
        return f"agreement with {sweep.tiers[-1]}"
    return "accuracy"


def _rec_index(sweep: SweepResult) -> int | None:
    rec = sweep.recommendation
    if rec is None:
        return None
    if rec.point_index is not None and 0 <= rec.point_index < len(sweep.points):
        return rec.point_index
    for i, p in enumerate(sweep.points):
        if p.thresholds == rec.thresholds:
            return i
    return None


def _baseline_label(key: str) -> str:
    if key.startswith("only:"):
        return f"always {key[5:]}"
    if key == "oracle":
        return "oracle (lower bound)"
    return key


def _pareto_chart(sweep: SweepResult, *, narrow: bool = False) -> str:
    known = [
        p
        for p in sweep.points
        if _finite(p.cost_per_task) is not None and _finite(p.accuracy.value) is not None
    ]
    base_known = {
        k: p
        for k, p in sweep.baselines.items()
        if _finite(p.cost_per_task) is not None and _finite(p.accuracy.value) is not None
    }
    costs = [p.cost_per_task for p in known] + [p.cost_per_task for p in base_known.values()]
    pos = [c for c in costs if c is not None and c > 0]
    x_log = bool(pos) and len(pos) == len(costs) and max(pos) / min(pos) > LOG_X_SPAN
    cloud = [(p.cost_per_task, p.accuracy.value) for p in known]
    front = [sweep.points[i] for i in sweep.frontier if 0 <= i < len(sweep.points)]
    front_pts = sorted(
        (p.cost_per_task, p.accuracy.value)
        for p in front
        if _finite(p.cost_per_task) is not None and _finite(p.accuracy.value) is not None
    )
    series = [
        Series("operating points", cloud, style="muted", opacity=0.55, size=2.5),
        Series("Pareto frontier", front_pts, kind="line", style=1, show_points=True),
    ]
    markers: list[Marker] = []
    shapes = ["square", "triangle", "triangle-down", "circle", "cross", "x"]
    only = [k for k in sweep.baselines if k.startswith("only:")]
    for i, k in enumerate(only):
        if k in base_known:
            p = base_known[k]
            shape = shapes[i % len(shapes)]
            markers.append(
                Marker(
                    p.cost_per_task,
                    p.accuracy.value,
                    _baseline_label(k),
                    shape=shape,  # type: ignore[arg-type]
                    in_legend=True,
                )
            )
    for k, p in base_known.items():
        if not k.startswith("only:"):
            markers.append(
                Marker(
                    p.cost_per_task,
                    p.accuracy.value,
                    _baseline_label(k),
                    shape="diamond",
                    in_legend=True,
                )
            )
    ri = _rec_index(sweep)
    if ri is not None:
        rp = sweep.points[ri]
        markers.append(
            Marker(
                rp.cost_per_task,
                rp.accuracy.value,
                "recommended",
                shape="star",
                style="accent",
                emphasis=True,
                in_legend=True,
            )
        )
    acc = _acc_label(sweep)
    return scatter(
        series,
        title=f"{acc[0].upper()}{acc[1:]} vs cost per task",
        x_label="cost per task (USD" + (", log scale)" if x_log else ")"),
        y_label=acc,
        x_format="currency",
        y_format="percent",
        x_log=x_log,
        markers=markers,
        width=360 if narrow else 720,
        height=440 if narrow else 420,
        legend="bottom" if narrow else "auto",
        desc=f"{len(known)} simulated threshold settings (grey), the Pareto frontier (line), "
        "single-tier and oracle baselines, and the recommended point (star).",
        empty_message="No operating points with a known cost",
    )


def _rec_rows(sweep: SweepResult) -> tuple[list[tuple[str, str]], str]:
    rec = sweep.recommendation
    acc = _acc_label(sweep)
    if rec is None:
        why = next((n for n in sweep.notes if n.startswith("no recommendation")), "")
        return [], why or "No recommendation: no operating point met the objective."
    non_final = list(sweep.tiers[:-1])
    rows: list[tuple[str, str]] = [
        ("Objective", rec.objective),
        ("Thresholds", _fmt_thresholds(non_final, rec.thresholds)),
        (
            f"Selection {acc} ({fmt_count(rec.n_selection or rec.point.accuracy.n)} tasks)",
            fmt_estimate(rec.point.accuracy),
        ),
        ("Selection cost per task", fmt_usd(rec.point.cost_per_task)),
        ("Selection escalation rate", fmt_pct(rec.point.escalation_rate)),
    ]
    if rec.holdout is not None:
        rows += [
            (
                f"Held-out {acc} ({fmt_count(rec.n_holdout or rec.holdout.accuracy.n)} tasks)",
                fmt_estimate(rec.holdout.accuracy),
            ),
            ("Held-out cost per task", fmt_usd(rec.holdout.cost_per_task)),
            ("Held-out escalation rate", fmt_pct(rec.holdout.escalation_rate)),
        ]
    if rec.delta_vs_best is not None:
        best = rec.best_single_tier or "best single tier"
        rows.append((f"{acc[0].upper()}{acc[1:]} vs always {best}", _fmt_delta(rec.delta_vs_best)))
    return rows, rec.note


def _point_row(sweep: SweepResult, label: str, p: OperatingPoint) -> list[str]:
    return [
        label,
        _est_cell(p.accuracy),
        fmt_usd(p.cost_per_task),
        fmt_pct(p.escalation_rate),
        ", ".join(f"{t} {fmt_pct(s, 0)}" for t, s in zip(sweep.tiers, p.tier_share, strict=False))
        or DASH,
    ]


def _baselines_table(sweep: SweepResult) -> _Table:
    rows = []
    highlight: set[int] = set()
    for k, p in sweep.baselines.items():
        rows.append(_point_row(sweep, _baseline_label(k), p))
    ri = _rec_index(sweep)
    if ri is not None:
        highlight.add(len(rows))
        rows.append(_point_row(sweep, "recommended (all data)", sweep.points[ri]))
    acc = _acc_label(sweep)
    return _Table(
        f"Baselines on all {fmt_count(sweep.n)} usable tasks",
        ["Router", f"{acc[0].upper()}{acc[1:]} (CI)", "Cost / task", "Escalation", "Tier share"],
        rows,
        numeric={1, 2, 3},
        highlight=highlight,
        note="The oracle serves from the cheapest correct tier without paying for the tiers it "
        "skips; no real router can reach it."
        if "oracle" in sweep.baselines
        else "",
    )


def _frontier_rows(sweep: SweepResult) -> tuple[list[int], int]:
    idx = [i for i in sweep.frontier if 0 <= i < len(sweep.points)]
    idx.sort(key=lambda i: (_finite(sweep.points[i].cost_per_task) or 0.0, i))
    total = len(idx)
    if total <= MAX_FRONTIER_ROWS:
        return idx, total
    pick = {round(k * (total - 1) / (MAX_FRONTIER_ROWS - 1)) for k in range(MAX_FRONTIER_ROWS)}
    chosen = [idx[k] for k in sorted(pick)]
    ri = _rec_index(sweep)
    if ri is not None and ri in idx and ri not in chosen:
        chosen.append(ri)
        chosen.sort(key=lambda i: (_finite(sweep.points[i].cost_per_task) or 0.0, i))
    return chosen, total


def _frontier_table(sweep: SweepResult) -> _Table:
    chosen, total = _frontier_rows(sweep)
    non_final = list(sweep.tiers[:-1])
    ri = _rec_index(sweep)
    rows, highlight = [], set()
    for i in chosen:
        p = sweep.points[i]
        if i == ri:
            highlight.add(len(rows))
        rows.append(
            [
                _fmt_thresholds(non_final, p.thresholds),
                _est_cell(p.accuracy),
                fmt_usd(p.cost_per_task),
                fmt_pct(p.escalation_rate),
                _est_cell(p.skipped_error),
                "recommended" if i == ri else "",
            ]
        )
    acc = _acc_label(sweep)
    caption = "Pareto frontier"
    if total > len(chosen):
        caption += f" (showing {len(chosen)} of {total} points)"
    return _Table(
        caption,
        [
            "Thresholds",
            f"{acc[0].upper()}{acc[1:]} (CI)",
            "Cost / task",
            "Escalation",
            "Skipped error (CI)",
            "",
        ],
        rows,
        numeric={1, 2, 3, 4},
        highlight=highlight,
    )


def _calibration_table(sweep: SweepResult) -> _Table:
    def num(x: float | None, d: int = 3) -> str:
        v = _finite(x)
        return DASH if v is None else f"{v:.{d}f}"

    rows = [
        [
            name,
            fmt_count(c.n),
            num(c.ece),
            num(c.brier),
            num(c.auroc),
            num(c.aurc),
            fmt_count(c.score_missing),
        ]
        for name, c in sweep.calibration.items()
    ]
    return _Table(
        "Confidence calibration per non-final tier",
        ["Tier", "Scored", "ECE", "Brier", "AUROC", "AURC", "Missing scores"],
        rows,
        numeric={1, 2, 3, 4, 5, 6},
        note="ECE and Brier: lower is better. AUROC: how well the score separates right from "
        "wrong answers (0.5 = chance). AURC: area under the risk-coverage curve, lower is better.",
    )


def _calibration_charts(sweep: SweepResult, tolerance: float | None) -> list[tuple[str, str]]:
    acc = _acc_label(sweep)
    out: list[tuple[str, str]] = []
    for name, c in sweep.calibration.items():
        bins = [
            b
            for b in c.reliability
            if b.n > 0 and b.mean_conf is not None and b.accuracy is not None
        ]
        rel = xy_chart(
            [
                Series(
                    "observed",
                    [(b.mean_conf, b.accuracy) for b in bins],
                    style=1,
                    show_points=False,
                    in_legend=False,
                )
            ],
            title=f"Reliability: {name}",
            x_label="mean confidence",
            y_label=acc,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            x_format="percent",
            y_format="percent",
            error_bars=[ErrorBar(b.mean_conf, b.accuracy, b.ci_lo, b.ci_hi) for b in bins],
            diagonal=Diagonal("perfect calibration"),
            legend="none",
            width=440,
            height=360,
            desc=f"{acc[0].upper()}{acc[1:]} per confidence bin with Wilson intervals; points "
            "on the diagonal are perfectly calibrated.",
            empty_message="No scored decisions",
        )
        out.append((rel, f"Reliability diagram for {name} ({fmt_count(c.n)} scored tasks)."))
        rc = sorted(
            (cov, risk)
            for _, cov, risk in c.risk_coverage
            if _finite(cov) is not None and _finite(risk) is not None
        )
        refs = (
            [RefLine("y", tolerance, f"tolerance {_fmt_tol(tolerance)}")]
            if (tolerance is not None and rc)
            else []
        )
        risk_chart = line_chart(
            [Series("risk", rc, kind="step", style=1, in_legend=False)],
            title=f"Risk-coverage: {name}",
            x_label="coverage (share kept)",
            y_label="risk (error among kept)",
            x_range=(0.0, 1.0),
            x_format="percent",
            y_format="percent",
            ref_lines=refs,
            legend="none",
            width=440,
            height=360,
            desc="Error rate among the kept answers as the threshold is lowered and more "
            "answers are kept.",
            empty_message="No scored decisions",
        )
        out.append((risk_chart, f"Risk-coverage curve for {name}."))
    return out


# --------------------------------------------------------------------------- assemble


def _notes(summary: AuditSummary, sweep: SweepResult | None) -> list[str]:
    seen: dict[str, None] = {}
    for n in list(summary.notes) + (list(sweep.notes) if sweep is not None else []):
        text = str(n).strip()
        if text:
            first = text.split(" ", 1)[0]
            if first.isalpha() and first.islower():
                text = text[0].upper() + text[1:]
            seen.setdefault(text, None)
    return list(seen)


def _sections(
    summary: AuditSummary,
    sweep: SweepResult | None,
    *,
    charts: bool,
) -> list[_Section]:
    s = summary
    label, why = _status_parts(s)
    secs: list[_Section] = []
    head = [
        _Block("badge", (s.status, label, why)),
        _Block("p", _headline_segments(s)),
        _Block("cards", _cards(s)),
    ]
    secs.append(_Section("summary", "Summary", head))

    bins_blocks: list[_Block] = []
    if s.n_skipped == 0:
        bins_blocks.append(_p("No skipped cases: there is nothing to break down by confidence."))
    else:
        bins_blocks.append(_Block("table", _bins_table(s)))
        if charts:
            chart = _bins_chart(s)
            narrow = _bins_chart(s, narrow=True)
            if chart and narrow:
                chart = _responsive(chart, narrow)
                bins_blocks.append(
                    _Block(
                        "charts",
                        [
                            (
                                chart,
                                "Whiskers are "
                                f"{_level_txt(s.level)}s; n is the number of audits "
                                "per bin.",
                            )
                        ],
                    )
                )
    secs.append(_Section("bins", "Audit by confidence bin", bins_blocks))

    secs.append(
        _Section(
            "tiers",
            "Tier usage and costs",
            [
                _Block("table", _tier_table(s)),
                _Block("table", _coverage_table(s)),
                _Block("table", _cost_table(s)),
            ],
        )
    )

    if sweep is not None:
        blocks: list[_Block] = []
        acc = _acc_label(sweep)
        intro = (
            f"Offline threshold sweep over {fmt_count(sweep.n)} eval-mode tasks "
            f"({fmt_count(len(sweep.points))} threshold settings); {acc} is measured "
            + (
                "against reference answers."
                if sweep.truth == "reference"
                else f"as agreement with the last tier ({sweep.tiers[-1] if sweep.tiers else '?'})"
                ", which counts as correct by definition."
            )
        )
        blocks.append(_p(intro))
        if charts:
            blocks.append(
                _Block(
                    "charts",
                    [
                        (
                            _responsive(_pareto_chart(sweep), _pareto_chart(sweep, narrow=True)),
                            "Grey: every simulated threshold setting. Line: "
                            "Pareto frontier. Star: recommended thresholds.",
                        )
                    ],
                )
            )
        rows, note = _rec_rows(sweep)
        blocks.append(_Block("rec", (rows, note)))
        blocks.append(_Block("table", _baselines_table(sweep)))
        if sweep.frontier:
            blocks.append(_Block("table", _frontier_table(sweep)))
        if sweep.calibration:
            blocks.append(_Block("h3", "Calibration"))
            blocks.append(_Block("table", _calibration_table(sweep)))
            if charts:
                blocks.append(_Block("charts", _calibration_charts(sweep, s.tolerance)))
        secs.append(_Section("sweep", "Threshold sweep", blocks))

    notes = _notes(s, sweep)
    cav: list[_Block] = []
    if notes:
        cav.append(_Block("list", notes))
    cav.append(_Block("h3", "Methodology"))
    cav.append(_p(METHODOLOGY))
    secs.append(_Section("caveats", "Caveats", cav))
    return secs


def _utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _footer_text(summary: AuditSummary, now: datetime | None) -> str:
    ts = _utc(now).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"Generated by shadowgate {__version__} at {ts} · run {summary.run_id or DASH} "
        f"· mode {summary.mode}"
    )


def _default_title(summary: AuditSummary) -> str:
    return f"shadowgate report: {summary.run_id}" if summary.run_id else "shadowgate report"


# --------------------------------------------------------------------------- HTML


def _e(s: object) -> str:
    return html.escape(str(s), quote=True)


_PAGE_CSS = """
:root {
  color-scheme: light;
  --bg: #ffffff; --fg: #1f2937; --muted: #5b6472; --border: #e5e7eb; --card: #f8fafc;
  --accent: #0072b2; --hl: #fff7ed;
  --ok-bg: #dcfce7; --ok-fg: #14532d; --breach-bg: #fee2e2; --breach-fg: #7f1d1d;
  --warn-bg: #fef3c7; --warn-fg: #78350f; --na-bg: #e5e7eb; --na-fg: #374151;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #111827; --fg: #e5e7eb; --muted: #9ca3af; --border: #374151; --card: #1f2937;
    --accent: #56b4e9; --hl: #3b2a1a;
    --ok-bg: #14532d; --ok-fg: #dcfce7; --breach-bg: #7f1d1d; --breach-fg: #fee2e2;
    --warn-bg: #78350f; --warn-fg: #fef3c7; --na-bg: #374151; --na-fg: #e5e7eb;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #111827; --fg: #e5e7eb; --muted: #9ca3af; --border: #374151; --card: #1f2937;
  --accent: #56b4e9; --hl: #3b2a1a;
  --ok-bg: #14532d; --ok-fg: #dcfce7; --breach-bg: #7f1d1d; --breach-fg: #fee2e2;
  --warn-bg: #78350f; --warn-fg: #fef3c7; --na-bg: #374151; --na-fg: #e5e7eb;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.sg-chart { --sg-bg: var(--bg) !important; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 40px; }
header.top { margin-bottom: 8px; }
h1 { font-size: 1.6rem; line-height: 1.25; margin: 0 0 4px; overflow-wrap: anywhere; }
h2 { font-size: 1.3rem; margin: 32px 0 8px; padding-top: 8px; border-top: 1px solid var(--border); }
h3 { font-size: 1.05rem; margin: 24px 0 8px; }
p { margin: 8px 0; overflow-wrap: anywhere; }
.meta, .muted, figcaption, .table-note { color: var(--muted); font-size: 0.875rem; }
.lede { font-size: 1.08rem; }
.status { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin: 12px 0; }
.badge {
  display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700;
  font-size: 0.8rem; letter-spacing: 0.05em; white-space: nowrap;
}
.badge-ok { background: var(--ok-bg); color: var(--ok-fg); }
.badge-breach { background: var(--breach-bg); color: var(--breach-fg); }
.badge-inconclusive { background: var(--warn-bg); color: var(--warn-fg); }
.badge-no-data, .badge-na, .badge-other { background: var(--na-bg); color: var(--na-fg); }
.cards {
  display: grid; gap: 10px; margin: 16px 0; padding: 0;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 150px), 1fr));
}
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 12px; margin: 0; min-width: 0; }
.card dt { font-size: 0.8rem; color: var(--muted); overflow-wrap: anywhere; }
.card dd { margin: 2px 0 0; font-size: 1.35rem; font-weight: 650;
  font-variant-numeric: tabular-nums; }
.card dd.sub { font-size: 0.8rem; font-weight: 400; color: var(--muted); overflow-wrap: anywhere; }
.table-wrap {
  overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 12px 0 4px;
  border: 1px solid var(--border); border-radius: 8px; max-width: 100%;
}
table { border-collapse: collapse; width: 100%; font-size: 0.875rem;
  font-variant-numeric: tabular-nums; }
caption { caption-side: top; text-align: left; padding: 8px 10px; font-weight: 600; }
th, td { padding: 6px 10px; text-align: left; border-top: 1px solid var(--border);
  white-space: nowrap; vertical-align: top; }
thead th { background: var(--card); font-weight: 600; }
tbody th { font-weight: 400; }
.num { text-align: right; }
tr.hl td, tr.hl th { background: var(--hl); font-weight: 600; }
.table-note { margin: 2px 0 12px; }
figure { margin: 16px 0; min-width: 0; }
figure svg { width: 100%; }
.chart-narrow { display: none; }
@media (max-width: 560px) {
  .chart-wide { display: none; }
  .chart-narrow { display: block; }
}
.charts { display: grid; gap: 16px;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr)); }
.rec { border: 2px solid var(--accent); border-radius: 10px; padding: 4px 16px 12px;
  background: var(--card); margin: 16px 0; }
.rec h3 { margin-top: 12px; }
.rec dl { display: grid; grid-template-columns: minmax(0, 14rem) minmax(0, 1fr);
  gap: 4px 16px; margin: 8px 0; }
.rec dt { color: var(--muted); }
.rec dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
ul.notes { padding-left: 1.2rem; }
ul.notes li { margin: 4px 0; overflow-wrap: anywhere; }
footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--border);
  color: var(--muted); font-size: 0.8rem; overflow-wrap: anywhere; }
@media (max-width: 560px) {
  h1 { font-size: 1.3rem; }
  h2 { font-size: 1.15rem; }
  .rec dl { grid-template-columns: minmax(0, 1fr); gap: 0 0; }
  .rec dd { margin-bottom: 8px; }
  .card dd { font-size: 1.15rem; }
}
"""


def _html_table(t: _Table) -> str:
    out = ['<div class="table-wrap"><table>']
    if t.caption:
        out.append(f"<caption>{_e(t.caption)}</caption>")
    out.append("<thead><tr>")
    for j, h in enumerate(t.headers):
        cls = ' class="num"' if j in t.numeric else ""
        out.append(f'<th scope="col"{cls}>{_e(h)}</th>')
    out.append("</tr></thead><tbody>")
    for i, row in enumerate(t.rows):
        out.append('<tr class="hl">' if i in t.highlight else "<tr>")
        for j, cell in enumerate(row):
            cls = ' class="num"' if j in t.numeric else ""
            tag = "th" if j == 0 else "td"
            scope = ' scope="row"' if j == 0 else ""
            out.append(f"<{tag}{scope}{cls}>{_e(cell)}</{tag}>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    if t.note:
        out.append(f'<p class="table-note">{_e(t.note)}</p>')
    return "".join(out)


def _html_segments(segs: Sequence[Segment]) -> str:
    return "".join(f"<strong>{_e(t)}</strong>" if strong else _e(t) for t, strong in segs)


def _html_block(b: _Block) -> str:
    if b.kind == "p":
        return f"<p>{_html_segments(b.data)}</p>"  # type: ignore[arg-type]
    if b.kind == "badge":
        status, label, why = b.data  # type: ignore[misc]
        cls = {
            "ok": "ok",
            "breach": "breach",
            "inconclusive": "inconclusive",
            "no-data": "no-data",
            "n/a": "na",
        }.get(status, "other")
        return (
            f'<div class="status"><span class="badge badge-{cls}" role="status">'
            f"{_e(label)}</span><span>{_e(why)}</span></div>"
        )
    if b.kind == "cards":
        items = []
        for label, value, sub in b.data:  # type: ignore[attr-defined]
            subhtml = f'<dd class="sub">{_e(sub)}</dd>' if sub else ""
            items.append(
                f'<div class="card"><dt>{_e(label)}</dt><dd>{_e(value)}</dd>{subhtml}</div>'
            )
        return f'<dl class="cards">{"".join(items)}</dl>'
    if b.kind == "table":
        return _html_table(b.data)  # type: ignore[arg-type]
    if b.kind == "charts":
        figs = [f"<figure>{svg}<figcaption>{_e(cap)}</figcaption></figure>" for svg, cap in b.data]  # type: ignore[attr-defined]
        if len(figs) == 1:
            return figs[0]
        return f'<div class="charts">{"".join(figs)}</div>'
    if b.kind == "rec":
        rows, note = b.data  # type: ignore[misc]
        body = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in rows)
        dl = f"<dl>{body}</dl>" if rows else ""
        return (
            f'<section class="rec" aria-labelledby="rec-h"><h3 id="rec-h">Recommendation'
            f'</h3>{dl}<p class="muted">{_e(note)}</p></section>'
        )
    if b.kind == "list":
        lis = "".join(f"<li>{_e(x)}</li>" for x in b.data)  # type: ignore[attr-defined]
        return f'<ul class="notes">{lis}</ul>'
    if b.kind == "h3":
        return f"<h3>{_e(b.data)}</h3>"
    return f"<p>{_e(b.data)}</p>"


def render_html(
    summary: AuditSummary,
    sweep: SweepResult | None = None,
    *,
    title: str | None = None,
    now: datetime | None = None,
) -> str:
    """A single self-contained HTML page (inline CSS and SVG, no scripts or external requests).

    ``now`` fixes the footer timestamp (default: the current UTC time).
    """
    title = title if title is not None else _default_title(summary)
    secs = _sections(summary, sweep, charts=True)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'; img-src data:\">",
        '<meta name="color-scheme" content="light dark">',
        f'<meta name="generator" content="shadowgate {_e(__version__)}">',
        f"<title>{_e(title)}</title>",
        f"<style>\n{CHART_CSS}{_PAGE_CSS}</style>",
        "</head>",
        "<body>",
        "<main>",
        '<header class="top">',
        f"<h1>{_e(title)}</h1>",
        f'<p class="meta">Run <code>{_e(summary.run_id or DASH)}</code> · mode '
        f"{_e(summary.mode)} · {_e(fmt_count(summary.n_decisions))} decisions"
        + (f" · tiers {_e(_ARROW.join(summary.tiers))}" if summary.tiers else "")
        + "</p>",
        "</header>",
    ]
    for sec in secs:
        parts.append(f'<section id="{_e(sec.id)}" aria-labelledby="{_e(sec.id)}-h">')
        parts.append(f'<h2 id="{_e(sec.id)}-h">{_e(sec.title)}</h2>')
        for b in sec.blocks:
            if sec.id == "summary" and b.kind == "p":
                parts.append(f'<p class="lede">{_html_segments(b.data)}</p>')  # type: ignore[arg-type]
            else:
                parts.append(_html_block(b))
        parts.append("</section>")
    parts += [
        f"<footer><p>{_e(_footer_text(summary, now))}</p></footer>",
        "</main>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- Markdown


_MD_SPECIAL = str.maketrans({c: "\\" + c for c in "\\`*_[]|#"})


def _md(s: object) -> str:
    """Escape for Markdown text and table cells (raw HTML is neutralised as entities)."""
    raw = str(s)
    text = " ".join(raw.split())
    if text:
        text = (" " if raw[:1].isspace() else "") + text + (" " if raw[-1:].isspace() else "")
    return html.escape(text.translate(_MD_SPECIAL), quote=False)


def _md_table(t: _Table) -> str:
    lines = []
    if t.caption:
        lines.append(f"**{_md(t.caption)}**")
        lines.append("")
    lines.append("| " + " | ".join(_md(h) or " " for h in t.headers) + " |")
    lines.append(
        "| " + " | ".join("---:" if j in t.numeric else "---" for j in range(len(t.headers))) + " |"
    )
    for i, row in enumerate(t.rows):
        cells = [_md(c) for c in row]
        if i in t.highlight and cells:
            cells = [f"**{c}**" if c else c for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    if t.note:
        lines.append("")
        lines.append(f"_{_md(t.note)}_")
    return "\n".join(lines)


def _md_block(b: _Block) -> str:
    if b.kind == "p":
        return "".join(f"**{_md(t.strip())}**" if strong else _md(t) for t, strong in b.data)  # type: ignore[attr-defined]
    if b.kind == "badge":
        _, label, why = b.data  # type: ignore[misc]
        return f"**Status: {_md(label)}**" + (f" — {_md(why)}" if why else "")
    if b.kind == "cards":
        return "\n".join(
            f"- **{_md(label)}:** {_md(value)}" + (f" ({_md(sub)})" if sub else "")
            for label, value, sub in b.data
        )  # type: ignore[attr-defined]
    if b.kind == "table":
        return _md_table(b.data)  # type: ignore[arg-type]
    if b.kind == "charts":
        return "_Charts are included in the HTML report._"
    if b.kind == "rec":
        rows, note = b.data  # type: ignore[misc]
        lines = ["### Recommendation", ""]
        lines += [f"- **{_md(k)}:** {_md(v)}" for k, v in rows]
        if note:
            lines += ["", f"_{_md(note)}_"] if rows else [_md(note)]
        return "\n".join(lines)
    if b.kind == "list":
        return "\n".join(f"- {_md(x)}" for x in b.data)  # type: ignore[attr-defined]
    if b.kind == "h3":
        return f"### {_md(b.data)}"
    return _md(b.data)


def render_markdown(
    summary: AuditSummary,
    sweep: SweepResult | None = None,
    *,
    title: str | None = None,
    now: datetime | None = None,
) -> str:
    """Markdown report with the same sections as the HTML report (charts omitted, with a note)."""
    title = title if title is not None else _default_title(summary)
    secs = _sections(summary, sweep, charts=True)
    out = [f"# {_md(title)}", ""]
    meta = (
        f"Run {_md(summary.run_id or DASH)} · mode {_md(summary.mode)} "
        f"· {fmt_count(summary.n_decisions)} decisions"
    )
    if summary.tiers:
        meta += f" · tiers {_ARROW.join(_md(t) for t in summary.tiers)}"
    out += [meta, ""]
    for sec in secs:
        out += [f"## {_md(sec.title)}", ""]
        last_chart = False
        for b in sec.blocks:
            if b.kind == "charts":
                if last_chart:
                    continue
                last_chart = True
            out += [_md_block(b), ""]
    out += ["---", "", f"_{_md(_footer_text(summary, now))}_", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- text


_ASCII = str.maketrans(
    {
        "\u2014": "-",
        "\u2013": "-",
        "\u2212": "-",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00b7": "|",
        "\u2192": ">",
        "\u03c0": "pi",
        "\u03a3": "sum ",
        "\u00b2": "^2",
    }
)


def _ascii(text: str) -> str:
    """Replace typographic symbols with ASCII so legacy consoles can print the text."""
    return text.translate(_ASCII)


def _clean_text(s: object) -> str:
    """Strip control characters (incl. ANSI escapes) from model/dataset strings."""
    return "".join(ch if ch.isprintable() else " " for ch in str(s))


def _wrap(text: str, indent: str = "", sub: str | None = None) -> list[str]:
    return textwrap.wrap(
        _ascii(_clean_text(text)),
        width=TEXT_WIDTH,
        initial_indent=indent,
        subsequent_indent=sub if sub is not None else indent + "  ",
        break_long_words=True,
        break_on_hyphens=False,
    ) or [indent.rstrip()]


def _kv(key: str, value: str) -> list[str]:
    return _wrap(f"{key + ':':<32} {value}", "  ", " " * 35)


def _num3(x: float | None) -> str:
    v = _finite(x)
    return DASH if v is None else f"{v:.3f}"


def render_text(summary: AuditSummary, sweep: SweepResult | None = None) -> str:
    """Compact plain-text summary for terminals.

    Lines are at most 100 columns, there are no colour codes, and the report's own symbols are
    ASCII (dashes, ">=", "|") so legacy console encodings can print it.
    """
    s = summary
    lines: list[str] = []
    label, why = _status_parts(s)
    head = f"shadowgate | run {s.run_id or DASH} | mode {s.mode} | "
    head += f"{fmt_count(s.n_decisions)} decisions"
    if s.tiers:
        head += f" | tiers {' > '.join(s.tiers)}"
    lines += _wrap(head)
    lines += _wrap(f"Status: {label}" + (f" - {why}" if why else ""))
    lines.append("")
    lines += _wrap("".join(t for t, _ in _headline_segments(s)))
    lines.append("")
    for k, v, sub in _cards(s):
        lines += _kv(k, v + (f"  ({sub})" if sub else ""))
    bins = [b for b in _bins_all(s) if b.n_accepted]
    if bins:
        lines.append("")
        lines.append("By confidence bin:")
        has_err = any(b.error is not None for b in bins)
        hdr = f"  {'bin':<14} {'kept':>6} {'audited':>8}  {'disagreement (CI)':<26}"
        if has_err:
            hdr += f"  {'error vs refs (CI)':<26}"
        lines.append(hdr)
        for b in bins:
            row = (
                f"  {_clean_text(b.label)[:14]:<14} {b.n_accepted:>6} {b.n_audited:>8}  "
                f"{_est_cell(b.disagreement if b.n_audited else None):<26}"
            )
            if has_err:
                row += f"  {_est_cell(b.error):<26}"
            lines.append(_ascii(row)[:TEXT_WIDTH])
    if sweep is not None:
        lines.append("")
        acc = _acc_label(sweep)
        truth = "references" if sweep.truth == "reference" else "the last tier"
        lines += _wrap(
            f"Sweep: {fmt_count(sweep.n)} tasks, {fmt_count(len(sweep.points))} threshold "
            f"settings, {fmt_count(len(sweep.frontier))} on the Pareto frontier; {acc} vs "
            f"{truth}."
        )
        for k, p in sweep.baselines.items():
            lines += _kv(
                _baseline_label(k), f"{_est_cell(p.accuracy)}, {fmt_usd(p.cost_per_task)}/task"
            )
        rows, note = _rec_rows(sweep)
        if rows:
            lines.append("Recommendation:")
            for k, v in rows:
                lines += _kv(k, v)
        if note:
            lines += _wrap(f"Note: {note}", "  ")
        for name, c in sweep.calibration.items():
            lines += _wrap(
                f"Calibration {name}: ECE {_num3(c.ece)}, Brier {_num3(c.brier)}, AUROC "
                f"{_num3(c.auroc)}, AURC {_num3(c.aurc)}, {fmt_count(c.n)} scored, "
                f"{fmt_count(c.score_missing)} missing",
                "  ",
            )
    notes = _notes(s, sweep)
    if notes:
        lines.append("")
        lines.append("Notes:")
        for n in notes:
            lines += _wrap(n, "  - ", "    ")
    return "\n".join(line.rstrip() for line in lines) + "\n"


# --------------------------------------------------------------------------- files


def write_report(
    path: str | Path,
    summary: AuditSummary,
    sweep: SweepResult | None = None,
    *,
    fmt: str | None = None,
    title: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Write a report and return its path.

    ``fmt`` is ``"html"``, ``"markdown"`` (``"md"``) or ``"text"`` (``"txt"``); when None it is
    taken from the suffix (.html/.htm, .md/.markdown, .txt). Unknown formats raise ValueError.
    Parent directories are created as needed.
    """
    p = Path(path)
    if fmt is None:
        kind = _FORMATS.get(p.suffix.lower())
        if kind is None:
            raise ValueError(
                f"cannot infer report format from suffix {p.suffix!r}; use .html, .htm, .md, "
                ".markdown or .txt, or pass fmt"
            )
    else:
        kind = _FMT_ALIASES.get(fmt.lower())
        if kind is None:
            raise ValueError(f"unknown report format {fmt!r}; use 'html', 'markdown' or 'text'")
    if kind == "html":
        content = render_html(summary, sweep, title=title, now=now)
    elif kind == "markdown":
        content = render_markdown(summary, sweep, title=title, now=now)
    else:
        content = render_text(summary, sweep)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    return p
