from __future__ import annotations

import dataclasses
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import pytest

from shadowgate import __version__, compare, confidence, extract
from shadowgate.audit import AuditSummary, summarize
from shadowgate.backends.simulated import SimulatedBackend
from shadowgate.cascade import AuditPolicy, Cascade, Tier
from shadowgate.datasets import arithmetic
from shadowgate.pricing import Pricing
from shadowgate.report import (
    DASH,
    fmt_ci,
    fmt_count,
    fmt_estimate,
    fmt_pct,
    fmt_usd,
    render_html,
    render_markdown,
    render_text,
    write_report,
)
from shadowgate.stats import Estimate
from shadowgate.sweep import SweepResult, sweep
from shadowgate.types import Decision

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
EVIL = "<script>alert(1)</script>"


# --------------------------------------------------------------------------- fixtures


def _cascade(tasks, names=("fast", "slow"), skills=(9.0, 15.0), audit_rate=0.2, three=False):
    prices = [Pricing(0.1, 0.4), Pricing(0.8, 4.0), Pricing(3.0, 15.0)]
    tiers = []
    all_names = list(names)
    all_skills = list(skills)
    if three:
        all_names.insert(1, "mid")
        all_skills.insert(1, 12.0)
    for i, (name, skill) in enumerate(zip(all_names, all_skills, strict=True)):
        price = prices[0] if i == 0 else prices[-1] if i == len(all_names) - 1 else prices[1]
        backend = SimulatedBackend.from_tasks(
            tasks, name=f"b{i}", skill=skill, pricing=price, overconfidence=0.05 * (i == 0)
        )
        final = i == len(all_names) - 1
        tiers.append(
            Tier(
                name,
                backend,
                threshold=None if final else 0.8,
                estimator=None if final else confidence.Verbal(),
            )
        )
    return Cascade(
        tiers,
        extractor=extract.from_spec(None),
        comparator=compare.from_spec("numeric"),
        audit=AuditPolicy(rate=audit_rate, strata=((0.0, 0.9, 0.5),)),
    )


def _run(tasks, mode, run_id, **kw) -> list[Decision]:
    c = _cascade(tasks, **kw)
    return [c.route(t, run_id=run_id, mode=mode) for t in tasks]


@pytest.fixture(scope="module")
def tasks():
    return arithmetic(240, seed=3)


@pytest.fixture(scope="module")
def eval_run(tasks) -> tuple[AuditSummary, SweepResult]:
    ds = _run(tasks, "eval", "eval-1")
    return summarize(ds, tolerance=0.05), sweep(ds)


@pytest.fixture(scope="module")
def serve_summary(tasks) -> AuditSummary:
    return summarize(_run(tasks, "serve", "serve-1"), tolerance=0.05)


@pytest.fixture(scope="module")
def evil_run(tasks) -> tuple[AuditSummary, SweepResult]:
    ds = _run(tasks[:120], "eval", f"run-{EVIL}", names=(EVIL, "slow&<b>"))
    s = summarize(ds, tolerance=0.05)
    s = dataclasses.replace(s, notes=[*s.notes, f"note with {EVIL} and <img src=x onerror=1>"])
    return s, sweep(ds)


def _empty() -> AuditSummary:
    return summarize([])


# --------------------------------------------------------------------------- helpers


class _Checker(HTMLParser):
    VOID = {"meta", "br", "img", "hr", "input", "link", "col", "area", "base", "wbr", "source"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.attrs: list[tuple[str, str, str | None]] = []
        self.tags: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for k, v in attrs:
            self.attrs.append((tag, k, v))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append(tag)
        for k, v in attrs:
            self.attrs.append((tag, k, v))

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}>; open: {self.stack[-3:]}")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


def _check_html(doc: str) -> _Checker:
    p = _Checker()
    p.feed(doc)
    p.close()
    assert not p.errors, p.errors[:5]
    assert not p.stack, p.stack
    for svg in re.findall(r"<svg\b.*?</svg>", doc, flags=re.S):
        ET.fromstring(svg)  # every chart is well-formed XML
    return p


# --------------------------------------------------------------------------- formatting


def test_fmt_pct_precision_and_none() -> None:
    assert fmt_pct(0.074) == "7.4%"
    assert fmt_pct(0.63, 0) == "63%"
    assert fmt_pct(0.0086) == "0.86%"
    assert fmt_pct(0.0) == "0%"
    assert fmt_pct(1.0) == "100%"
    assert fmt_pct(-0.0001, 1) == "0.0%"
    assert fmt_pct(None) == DASH
    assert fmt_pct(float("nan")) == DASH
    assert fmt_pct(None, none="unknown") == "unknown"


def test_fmt_ci_and_estimate() -> None:
    assert fmt_ci(0.049, 0.11) == "4.9–11.0%"
    assert fmt_ci(0.001, 0.009) == "0.10–0.90%"
    assert fmt_ci(None, 0.1) == DASH
    est = Estimate(0.074, 0.049, 0.11, 212, "wilson")
    assert fmt_estimate(est) == "7.4% (95% CI 4.9–11.0%)"
    assert fmt_estimate(Estimate(None, 0.0, 1.0, 0, "wilson")) == DASH
    assert fmt_estimate(None) == DASH


def test_fmt_usd_and_count() -> None:
    assert fmt_usd(0.0042) == "$0.0042"
    assert fmt_usd(1.5) == "$1.50"
    assert fmt_usd(0) == "$0"
    assert fmt_usd(None) == DASH
    assert fmt_usd(None, none="unknown") == "unknown"
    assert fmt_count(12345) == "12,345"
    assert fmt_count(None) == DASH


# --------------------------------------------------------------------------- HTML


def test_html_well_formed_and_self_contained(eval_run) -> None:
    s, sw = eval_run
    doc = render_html(s, sw, now=NOW)
    assert doc.startswith("<!DOCTYPE html>")
    p = _check_html(doc)
    assert "script" not in p.tags
    for tag, k, v in p.attrs:
        if k in ("src", "href", "xlink:href", "action", "srcset"):
            assert not (v or "").lower().startswith(("http", "//")), (tag, k, v)
    assert not re.search(r"""(?:src|href)\s*=\s*["']?\s*(?:https?:)?//""", doc, re.I)
    assert "@import" not in doc
    assert all(u.startswith("#") for u in re.findall(r"url\(\s*['\"]?([^)'\"]*)", doc))
    assert "<link" not in doc


def test_html_structure_and_accessibility(eval_run) -> None:
    s, sw = eval_run
    doc = render_html(s, sw, now=NOW)
    assert '<html lang="en">' in doc
    assert 'name="viewport"' in doc
    assert doc.count("<h1>") == 1
    for sec in ("summary", "bins", "tiers", "sweep", "caveats"):
        assert f'<section id="{sec}"' in doc
    assert doc.count("<table>") == doc.count("<caption>")
    assert '<th scope="col"' in doc and '<th scope="row"' in doc
    assert 'class="table-wrap"' in doc
    assert 'role="img"' in doc


def test_html_theme_css(eval_run) -> None:
    s, sw = eval_run
    doc = render_html(s, sw, now=NOW)
    assert "prefers-color-scheme: dark" in doc
    assert ':root[data-theme="dark"]' in doc
    assert ':root:not([data-theme="light"])' in doc
    assert "--sg-bg: var(--bg)" in doc
    assert ".sg-chart" in doc  # CHART_CSS included
    assert "max-width: 100%" in doc
    assert "overflow-x: auto" in doc


def test_html_headline_numbers(eval_run) -> None:
    s, sw = eval_run
    doc = render_html(s, sw, now=NOW)
    share = fmt_pct(s.n_skipped / s.n_decisions, 0)
    assert f"answered <strong>{share}</strong>" in doc
    assert s.disagreement is not None
    assert f"<strong>{fmt_pct(s.disagreement.value)}</strong>" in doc
    assert fmt_ci(s.disagreement.lo, s.disagreement.hi) in doc
    assert f"{s.n_audited} audits" in doc
    assert "Escalation rate" in doc and fmt_pct(s.escalation_rate.value) in doc
    assert fmt_usd(s.cost_per_task) in doc
    assert "Savings vs always slow" in doc
    assert "Audit overhead" in doc
    assert "wrong" in doc  # expected wrong-but-kept


def test_html_status_badges(eval_run) -> None:
    s, _ = eval_run
    for status, label in [
        ("ok", "OK"),
        ("breach", "BREACH"),
        ("inconclusive", "INCONCLUSIVE"),
        ("no-data", "NO DATA"),
        ("n/a", "N/A"),
    ]:
        doc = render_html(dataclasses.replace(s, status=status, audits_to_resolve=42), now=NOW)
        assert f">{label}</span>" in doc
    doc = render_html(dataclasses.replace(s, status="inconclusive", audits_to_resolve=42), now=NOW)
    assert "42 more audits" in doc
    assert "tolerance 5%" in doc


def test_html_sweep_sections(eval_run) -> None:
    s, sw = eval_run
    doc = render_html(s, sw, now=NOW)
    assert "Recommendation" in doc
    assert sw.recommendation is not None
    assert "fast ≥" in doc
    assert "Held-out accuracy" in doc
    assert "always fast" in doc and "always slow" in doc and "oracle" in doc
    assert "Pareto frontier" in doc
    assert "Reliability: fast" in doc and "Risk-coverage: fast" in doc
    assert "ECE" in doc and "AUROC" in doc
    assert 'class="chart-wide"' in doc and 'class="chart-narrow"' in doc
    # frontier table capped at ~15 rows (+ the recommended row)
    frontier_table = doc[doc.index("<caption>Pareto frontier") :]
    frontier_table = frontier_table[: frontier_table.index("</table>")]
    assert frontier_table.count("<tr") - 1 <= 16


def test_html_escapes_model_strings_everywhere(evil_run) -> None:
    s, sw = evil_run
    doc = render_html(s, sw, now=NOW)
    _check_html(doc)
    assert "<script" not in doc.lower()
    assert "<img" not in doc
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in doc
    assert "slow&amp;&lt;b&gt;" in doc
    assert "<b>" not in doc
    md = render_markdown(s, sw, now=NOW)
    assert "<script" not in md.lower() and "<img" not in md and "<b>" not in md
    assert "&lt;script&gt;" in md


def test_title_escaped_and_custom(eval_run) -> None:
    s, _ = eval_run
    doc = render_html(s, title="A & B <i>", now=NOW)
    assert "<title>A &amp; B &lt;i&gt;</title>" in doc
    assert "<h1>A &amp; B &lt;i&gt;</h1>" in doc
    assert "<title>shadowgate report: eval-1</title>" in render_html(s, now=NOW)


def test_deterministic_with_fixed_now(eval_run) -> None:
    s, sw = eval_run
    a = render_html(s, sw, now=NOW)
    assert a == render_html(s, sw, now=NOW)
    assert render_markdown(s, sw, now=NOW) == render_markdown(s, sw, now=NOW)
    assert render_text(s, sw) == render_text(s, sw)
    assert "2026-01-02 03:04:05 UTC" in a
    assert f"shadowgate {__version__}" in a
    assert "run eval-1" in a and "mode eval" in a


def test_now_converted_to_utc(eval_run) -> None:
    s, _ = eval_run
    plus2 = datetime(2026, 1, 2, 5, 4, 5, tzinfo=timezone(timedelta(hours=2)))
    assert "2026-01-02 03:04:05 UTC" in render_html(s, now=plus2)
    naive = datetime(2026, 1, 2, 3, 4, 5)
    assert "2026-01-02 03:04:05 UTC" in render_markdown(s, now=naive)


def test_log_x_when_costs_span_widely(eval_run) -> None:
    s, sw = eval_run
    costs = [b.cost_per_task for b in sw.baselines.values() if b.cost_per_task]
    assert max(costs) / min(costs) > 20
    assert "log scale" in render_html(s, sw, now=NOW)
    flat = dataclasses.replace(
        sw,
        points=[dataclasses.replace(p, cost_per_task=0.001) for p in sw.points],
        baselines={
            k: dataclasses.replace(p, cost_per_task=0.0012) for k, p in sw.baselines.items()
        },
    )
    assert "log scale" not in render_html(s, flat, now=NOW)


# --------------------------------------------------------------------------- robustness


def test_serve_mode_without_sweep(serve_summary) -> None:
    s = serve_summary
    assert s.mode == "serve"
    doc = render_html(s, now=NOW)
    _check_html(doc)
    assert 'id="sweep"' not in doc
    assert "Threshold sweep" not in doc
    assert "mode serve" in doc
    md = render_markdown(s, now=NOW)
    assert "## Audit by confidence bin" in md
    assert "Threshold sweep" not in md
    d = s.disagreement
    assert d is not None and d.n_eff is not None
    assert abs(d.n_eff - d.n) >= 0.5  # stratified audit rates -> unequal weights
    assert f"n_eff {d.n_eff:.0f}" in doc
    assert "Unweighted disagreement (biased)" in doc


def test_empty_summary_renders() -> None:
    s = _empty()
    doc = render_html(s, now=NOW)
    _check_html(doc)
    assert "No decisions were recorded" in doc
    assert "NO DATA" in doc
    md = render_markdown(s, now=NOW)
    assert "No decisions were recorded" in md
    txt = render_text(s)
    assert "No decisions were recorded" in txt


def test_all_none_fields_render(eval_run) -> None:
    s, _ = eval_run
    blank = dataclasses.replace(
        s,
        disagreement=None,
        skipped_error=None,
        audit_tier_error=None,
        served_accuracy=None,
        expected_wrong_skipped=None,
        cost_serving=None,
        cost_audit=None,
        cost_per_task=None,
        audit_overhead=None,
        est_all_slow_cost_per_task=None,
        est_savings=None,
        tolerance=None,
        status="n/a",
        disagreement_unweighted=None,
        reference_tier=None,
        no_score_bin=None,
    )
    doc = render_html(blank, now=NOW)
    _check_html(doc)
    assert "unknown" in doc
    assert DASH in doc
    assert "cannot be estimated" in doc
    render_markdown(blank, now=NOW)
    render_text(blank)


def test_sweep_without_recommendation(eval_run) -> None:
    s, sw = eval_run
    no_rec = dataclasses.replace(
        sw,
        recommendation=None,
        notes=[*sw.notes, "no recommendation for objective 'min-accuracy': nothing fits"],
    )
    doc = render_html(s, no_rec, now=NOW)
    _check_html(doc)
    assert "nothing fits" in doc
    assert "recommended (all data)" not in doc
    assert "nothing fits" in render_markdown(s, no_rec, now=NOW)
    assert "nothing fits" in render_text(s, no_rec)


def test_single_tier_run(tasks) -> None:
    backend = SimulatedBackend.from_tasks(tasks[:30], name="solo", skill=12.0)
    c = Cascade(
        [Tier("solo", backend)],
        extractor=extract.from_spec(None),
        comparator=compare.from_spec("numeric"),
    )
    ds = [c.route(t, run_id="solo-run", mode="eval") for t in tasks[:30]]
    s = summarize(ds)
    sw = sweep(ds)
    doc = render_html(s, sw, now=NOW)
    _check_html(doc)
    assert "Single-tier run" in doc
    assert "No skipped cases" in doc
    render_markdown(s, sw, now=NOW)
    render_text(s, sw)


def test_three_tier_eval(tasks) -> None:
    ds = _run(tasks[:80], "eval", "three", three=True, names=("fast", "slow"))
    s = summarize(ds, tolerance=0.1)
    sw = sweep(ds)
    doc = render_html(s, sw, now=NOW)
    _check_html(doc)
    assert "fast, mid" in doc  # both non-final tiers named
    assert "Reliability: mid" in doc
    assert sw.recommendation is not None
    assert len(sw.recommendation.thresholds) == 2
    assert re.search(r"fast(?: ≥ [0-9.]+|: always escalate), mid", doc)


def test_audit_tier_truth_label(eval_run) -> None:
    s, sw = eval_run
    proxy = dataclasses.replace(sw, truth="audit-tier")
    doc = render_html(s, proxy, now=NOW)
    assert "agreement with slow" in doc


# --------------------------------------------------------------------------- markdown


def test_markdown_contents(eval_run) -> None:
    s, sw = eval_run
    md = render_markdown(s, sw, now=NOW)
    assert md.startswith("# shadowgate report: eval-1")
    assert "**Status: " in md
    assert s.disagreement is not None
    assert fmt_pct(s.disagreement.value) in md
    assert fmt_ci(s.disagreement.lo, s.disagreement.hi) in md
    assert fmt_usd(s.cost_per_task) in md
    for heading in (
        "## Summary",
        "## Audit by confidence bin",
        "## Tier usage and costs",
        "## Threshold sweep",
        "### Recommendation",
        "### Calibration",
        "## Caveats",
        "### Methodology",
    ):
        assert heading in md
    assert "Charts are included in the HTML report" in md
    assert "<svg" not in md
    assert "| ---" in md
    assert "2026-01-02 03:04:05 UTC" in md
    # bold numbers keep their surrounding spaces
    assert "answered **" in md and "** of " in md


def test_markdown_table_cells_escaped(evil_run) -> None:
    s, sw = evil_run
    s2 = dataclasses.replace(s, tier_share={**s.tier_share, "a|b": 1})
    md = render_markdown(s2, sw, now=NOW)
    assert "a\\|b" in md


# --------------------------------------------------------------------------- text


def test_text_summary(eval_run) -> None:
    s, sw = eval_run
    txt = render_text(s, sw)
    lines = txt.splitlines()
    assert all(len(line) <= 100 for line in lines), max(lines, key=len)
    assert "\x1b" not in txt
    assert txt.isascii()
    assert "Status:" in txt
    assert "eval-1" in txt
    assert "Recommendation:" in txt
    assert "Calibration fast" in txt
    assert "By confidence bin:" in txt
    assert s.disagreement is not None
    assert fmt_pct(s.disagreement.value) in txt


def test_text_strips_control_characters(evil_run) -> None:
    s, sw = evil_run
    s2 = dataclasses.replace(s, notes=["bad \x1b[31mred\x1b[0m note"])
    txt = render_text(s2, sw)
    assert "\x1b" not in txt
    assert all(len(line) <= 100 for line in txt.splitlines())


def test_text_serve_mode(serve_summary) -> None:
    txt = render_text(serve_summary)
    assert "mode serve" in txt
    assert "Sweep:" not in txt
    assert all(len(line) <= 100 for line in txt.splitlines())


# --------------------------------------------------------------------------- write_report


@pytest.mark.parametrize(
    ("name", "marker"),
    [
        ("r.html", "<!DOCTYPE html>"),
        ("r.HTM", "<!DOCTYPE html>"),
        ("r.md", "# shadowgate report"),
        ("r.markdown", "# shadowgate report"),
        ("r.txt", "shadowgate | run"),
    ],
)
def test_write_report_suffix(tmp_path: Path, eval_run, name: str, marker: str) -> None:
    s, sw = eval_run
    out = write_report(tmp_path / "sub" / name, s, sw, now=NOW)
    assert out == tmp_path / "sub" / name
    assert out.read_text(encoding="utf-8").startswith(marker)


def test_write_report_explicit_fmt_and_errors(tmp_path: Path, eval_run) -> None:
    s, _ = eval_run
    out = write_report(tmp_path / "report.out", s, fmt="md", now=NOW)
    assert out.read_text(encoding="utf-8").startswith("# ")
    out = write_report(tmp_path / "x.html", s, fmt="text")
    assert out.read_text(encoding="utf-8").startswith("shadowgate | run")
    with pytest.raises(ValueError, match="suffix"):
        write_report(tmp_path / "report.pdf", s)
    with pytest.raises(ValueError, match="suffix"):
        write_report(tmp_path / "report", s)
    with pytest.raises(ValueError, match="unknown report format"):
        write_report(tmp_path / "r.html", s, fmt="docx")
