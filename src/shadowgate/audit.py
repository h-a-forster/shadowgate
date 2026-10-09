"""Audit summaries: how often the fast path is wrong on the cases it kept.

:func:`summarize` turns the decisions of one run into an :class:`AuditSummary`. Its headline
number is the skipped-case disagreement rate: among decisions answered by a non-final tier
without escalation ("skipped" cases), the share whose answer differs from the reference (audit)
tier. In serve mode only a sample of skipped cases is shadow-audited, each with a known
inclusion probability pi, so the rate is a Hajek (inverse-probability weighted) estimate with
weights 1/pi. In eval mode every tier answers every task, so every skipped case is observed
(pi = 1).

Conventions
-----------
* **Tier order.** Decisions in serve mode only contain the attempts that were actually run, so the
  cascade's tier order is inferred (:func:`infer_tier_order`): attempt sequences are merged
  longest-first, and the final tier is the tier whose attempts carry ``threshold=None`` (the
  cascade never gives the final tier a threshold). If no final-tier attempt was ever observed
  (every decision was accepted early), the shadow audit tier is used when it is not one of the
  observed tiers; otherwise the final tier is unknown and every observed tier counts as
  non-final.
* **Skipped case.** A decision without ``error`` whose served attempt (the accepted attempt of
  ``final_tier``) is at a non-final tier. Decisions with ``error`` are counted in ``n_errors``
  and excluded from tier shares, skipped cases and cost-per-task figures they cannot inform.
* **Serve-mode audit states.** For each skipped case: ``shadow.status == "done"`` with a decided
  agreement -> audited; "done" with ``equivalent=None`` -> ``n_undecided``; "pending" ->
  ``n_pending``; "error" -> ``n_audit_errors``; "skipped" -> not selected (its pi still matters);
  no ShadowResult at all (audit off) -> ``n_no_audit_record``. Non-audited states are excluded
  from the estimate; this is unbiased only if they are missing at random given pi (see notes).
* **Eval mode.** The served attempt's ``agreement`` (vs the last tier) is the audit, pi = 1. A
  missing agreement because the last tier failed counts as an audit error.
* **Status.** ``skipped_error`` drives the status when every skipped case is graded against a
  reference; otherwise ``disagreement`` does. ``breach`` if lo > tolerance, ``ok`` if
  hi <= tolerance, else ``inconclusive``; ``no-data`` when the metric has no observations;
  ``n/a`` when no tolerance is given.
* **Confidence bins.** Edges ``bins`` define half-open bins ``[e_i, e_{i+1})``; the last bin is
  closed. Scores below the first edge fall into the first bin, scores above the last edge into
  the last bin. Skipped cases without a score (None/NaN, only possible with hand-built records)
  are summarised separately in ``no_score_bin`` (``lo = hi = nan``, label "no score").
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from shadowgate.stats import Estimate, mean_ci, weighted_proportion, wilson, z_for
from shadowgate.types import Attempt, Decision

__all__ = ["AuditSummary", "BinSummary", "infer_tier_order", "summarize"]

DEFAULT_BINS: tuple[float, ...] = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
LOW_N_EFF = 30.0  # effective sample size below which a note is added
TINY_PI = 0.05  # inclusion probabilities below this are flagged when few audits back them
MIN_AUDITS_PER_STRATUM = 5
HEAVY_WEIGHT_SHARE = 0.2  # one audit carrying more than this share of the total weight


@dataclass(frozen=True)
class BinSummary:
    """Skipped cases whose served confidence falls in ``[lo, hi)`` (the last bin is closed).

    ``disagreement`` is the weighted (Hajek) disagreement among audited cases in the bin (the
    vacuous interval with ``value=None`` when the bin has no audits). ``error`` is the unweighted
    Wilson error rate over the bin's graded skipped cases (None when none are graded).
    """

    lo: float
    hi: float
    n_accepted: int
    n_audited: int
    disagreement: Estimate
    error: Estimate | None
    label: str = ""
    n_graded: int = 0


@dataclass(frozen=True)
class AuditSummary:
    """Audit of one run; see the module docstring for the definitions used."""

    run_id: str
    mode: str  # "serve" | "eval" | "mixed" | "n/a" (empty input)
    n_decisions: int
    n_errors: int
    n_skipped: int  # accepted at a non-final tier (decisions without error)
    escalation_rate: Estimate  # share of decisions escalated past tier 0 (all decisions)
    tier_share: dict[str, int]  # decisions served per tier (decisions without error)
    n_audited: int
    n_pending: int
    n_audit_errors: int
    n_undecided: int
    disagreement: Estimate | None  # weighted (Hajek) skipped-case disagreement vs audit tier
    skipped_error: Estimate | None  # vs references, over all graded skipped cases
    audit_tier_error: Estimate | None  # audit tier's own error on audited cases (weighted)
    served_accuracy: Estimate | None  # accuracy of served answers over graded decisions
    expected_wrong_skipped: tuple[float, float, float] | None  # (est, lo, hi) count
    bins: list[BinSummary]
    cost_serving: float | None  # sum of known Decision.cost_usd
    cost_audit: float | None  # sum of known Decision.audit_cost_usd
    cost_per_task: float | None  # cost_serving / decisions with known serving cost
    audit_overhead: float | None  # cost_audit / cost_serving
    est_all_slow_cost_per_task: Estimate | None
    est_savings: Estimate | None  # 1 - cost_per_task / all-slow cost per task
    tolerance: float | None
    status: str  # "ok" | "breach" | "inconclusive" | "no-data" | "n/a"
    audits_to_resolve: int | None
    notes: list[str]
    # ---- additions beyond the design contract
    tiers: tuple[str, ...] = ()  # inferred tier order
    final_tier: str | None = None  # inferred final tier (None if never observed)
    reference_tier: str | None = None  # tier the disagreement is measured against
    disagreement_unweighted: Estimate | None = None  # naive share among audits (biased)
    n_graded_skipped: int = 0  # skipped cases with a decided reference grade
    n_not_selected: int = 0  # skipped cases not sampled for audit (status "skipped")
    n_no_audit_record: int = 0  # skipped cases with no ShadowResult (audit off)
    n_unknown_cost: int = 0  # decisions with cost_usd None
    n_unknown_audit_cost: int = 0  # decisions with audit_cost_usd None
    status_metric: str | None = None  # "skipped_error" | "disagreement" | None
    expected_wrong_source: str | None = None  # "skipped_error" | "disagreement" | None
    no_score_bin: BinSummary | None = None  # skipped cases without a confidence score
    level: float = 0.95


# --------------------------------------------------------------------------- tier order


def infer_tier_order(decisions: Iterable[Decision]) -> tuple[tuple[str, ...], str | None]:
    """Infer ``(tier order, final tier)`` from the attempts recorded in ``decisions``.

    Attempt sequences are merged longest-first: a tier not yet seen is inserted right after the
    tier that preceded it in its sequence. The final tier is the latest tier with an attempt
    whose ``threshold`` is None; failing that, a shadow audit tier never seen among attempts is
    appended as the final tier; failing that, the final tier is unknown (None).
    """
    decisions = list(decisions)
    order: list[str] = []
    seqs = [tuple(dict.fromkeys(a.tier for a in d.attempts)) for d in decisions]
    for seq in sorted(seqs, key=len, reverse=True):
        prev: str | None = None
        for name in seq:
            if name not in order:
                order.insert(0 if prev is None else order.index(prev) + 1, name)
            prev = name
    finals = {a.tier for d in decisions for a in d.attempts if a.threshold is None}
    final: str | None = None
    if finals:
        final = max(finals, key=order.index)
        order.remove(final)
        order.append(final)
    else:
        audit_tiers = sorted({d.shadow.audit_tier for d in decisions if d.shadow is not None})
        unseen = [t for t in audit_tiers if t not in order]
        if unseen:
            final = unseen[0]
            order.append(final)
    return tuple(order), final


def _served_attempt(d: Decision) -> Attempt | None:
    for a in d.attempts:
        if a.tier == d.final_tier and a.accepted and a.error is None:
            return a
    for a in reversed(d.attempts):
        if a.tier == d.final_tier:
            return a
    return None


def _attempt_cost(a: Attempt | None) -> float | None:
    """Model call plus confidence calls; None when unknown or the call failed."""
    if a is None or a.completion is None or a.completion.cost_usd is None:
        return None
    cost = a.completion.cost_usd
    if a.confidence is not None:
        for c in a.confidence.calls:
            if c.cost_usd is None:
                return None
            cost += c.cost_usd
    return cost


def _decided(j: object) -> bool | None:
    eq = getattr(j, "equivalent", None)
    return eq if isinstance(eq, bool) else None


# --------------------------------------------------------------------------- per-case record


@dataclass
class _Case:
    """One skipped case."""

    score: float | None
    pi: float | None  # None: no audit record / invalid
    state: str  # audited | undecided | pending | error | not-selected | none | invalid | unknown
    disagree: bool | None = None
    wrong: bool | None = None  # served answer vs reference
    ref_wrong: bool | None = None  # audit tier's answer vs reference (audited cases only)


def _weighted_mean(xs: Sequence[float], ws: Sequence[float], level: float) -> Estimate | None:
    """Hajek weighted mean with a linearised normal interval (mean_ci when weights are equal)."""
    if not xs:
        return None
    if all(w == ws[0] for w in ws):
        return mean_ci(xs, level)
    n = len(xs)
    sw = math.fsum(ws)
    m = math.fsum(w * x for w, x in zip(ws, xs, strict=True)) / sw
    n_eff = sw * sw / math.fsum(w * w for w in ws)
    if n == 1:
        return Estimate(m, None, None, 1, "hajek-normal", level, n_eff=n_eff)
    var = math.fsum(w * w * (x - m) ** 2 for w, x in zip(ws, xs, strict=True)) / (sw * sw)
    var *= n / (n - 1)
    half = z_for(level) * math.sqrt(var)
    return Estimate(m, m - half, m + half, n, "hajek-normal", level, n_eff=n_eff)


def _check_bins(bins: Sequence[float]) -> list[float]:
    edges = [float(b) for b in bins]
    if len(edges) < 2:
        raise ValueError("bins needs at least two edges")
    if any(not math.isfinite(e) for e in edges):
        raise ValueError("bin edges must be finite")
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError("bin edges must be strictly increasing")
    return edges


def _bin_of(score: float, edges: Sequence[float]) -> int:
    last = len(edges) - 2
    if score < edges[0]:
        return 0
    return min(last, bisect.bisect_right(edges, score) - 1)


def _bin_summary(
    lo: float, hi: float, label: str, cases: Sequence[_Case], level: float
) -> BinSummary:
    audited = [c for c in cases if c.state == "audited"]
    graded = [c for c in cases if c.wrong is not None]
    dis = weighted_proportion(
        [bool(c.disagree) for c in audited], [1.0 / c.pi for c in audited if c.pi], level
    )
    err = wilson(sum(1 for c in graded if c.wrong), len(graded), level) if graded else None
    return BinSummary(lo, hi, len(cases), len(audited), dis, err, label, len(graded))


def _fmt_edge(x: float) -> str:
    return f"{x:.2f}"


def _wilson_clears(p: float, n: float, z: float, tolerance: float) -> bool:
    """Whether the Wilson interval at proportion p and (effective) size n excludes tolerance
    on the side of p (lo > tolerance when p > tolerance, hi <= tolerance when p < tolerance)."""
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z / denom * math.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0))
    if p > tolerance:
        return centre - half > tolerance
    return centre + half <= tolerance


def _audits_to_resolve(est: Estimate, tolerance: float, level: float) -> int | None:
    """Extra observations for the interval to clear ``tolerance``, assuming p stays put.

    Finds the smallest effective size m at which the Wilson interval around the current point
    estimate excludes the tolerance (the same interval family that is reported; unlike the Wald
    planning formula of :func:`~shadowgate.stats.required_n` it handles p near 0 or 1 and the
    interval's asymmetry). m is converted to audits with the current design effect n / n_eff,
    assumed constant as audits are added (new audits drawn under the same sampling design).
    Returns None when p equals the tolerance or more than 10^7 audits would be needed.
    """
    p = est.value
    if p is None or p == tolerance:
        return None
    z = z_for(level)
    cap = 1e7
    lo, hi = 0.5, 1.0
    while not _wilson_clears(p, hi, z, tolerance):
        hi *= 2.0
        if hi > cap:
            return None
    while hi - lo > 1e-3 * hi:
        mid = (lo + hi) / 2.0
        if _wilson_clears(p, mid, z, tolerance):
            hi = mid
        else:
            lo = mid
    deff = 1.0
    if est.n_eff is not None and est.n_eff > 0 and est.n > 0:
        deff = max(1.0, est.n / est.n_eff)
    total = math.ceil(hi * deff - 1e-9)
    return max(1, total - est.n)


# --------------------------------------------------------------------------- summarize


def summarize(
    decisions: Iterable[Decision],
    *,
    tolerance: float | None = None,
    bins: Sequence[float] = DEFAULT_BINS,
    level: float = 0.95,
) -> AuditSummary:
    """Summarise one run's decisions; see the module docstring for definitions.

    Raises ValueError when the decisions come from more than one run, for malformed ``bins`` or
    a ``tolerance`` outside [0, 1]. Never raises on empty input or on decisions with errors.
    """
    decisions = list(decisions)
    edges = _check_bins(bins)
    if tolerance is not None and not (
        isinstance(tolerance, (int, float)) and 0.0 <= tolerance <= 1.0
    ):
        raise ValueError(f"tolerance must be in [0, 1], got {tolerance!r}")
    z_for(level)  # validates level
    run_ids = sorted({d.run_id for d in decisions})
    if len(run_ids) > 1:
        raise ValueError(f"decisions span {len(run_ids)} runs {run_ids[:5]}; summarise per run")
    notes: list[str] = []
    run_id = run_ids[0] if run_ids else ""
    modes = sorted({d.mode for d in decisions})
    if not modes:
        mode = "n/a"
    elif len(modes) == 1:
        mode = modes[0]
    else:
        mode = "mixed"
        notes.append(f"Run mixes modes {modes}; each decision is handled by its own mode.")

    tiers, final = infer_tier_order(decisions)
    n = len(decisions)
    ok = [d for d in decisions if d.error is None]
    n_errors = n - len(ok)
    if n_errors:
        notes.append(
            f"{n_errors} decision(s) failed (final tier error); they are excluded from skipped "
            "cases and tier shares."
        )
    escalation_rate = wilson(sum(1 for d in decisions if d.escalated), n, level)
    tier_share = dict(Counter(d.final_tier for d in ok))

    audit_tier_names = Counter(
        d.shadow.audit_tier for d in decisions if d.mode != "eval" and d.shadow is not None
    )
    if mode == "eval" or not audit_tier_names:
        reference_tier = final
    else:
        reference_tier = audit_tier_names.most_common(1)[0][0]

    # ---- classify skipped cases and gather reference-tier cost observations
    cases: list[_Case] = []
    ref_costs: list[float] = []
    ref_weights: list[float] = []
    n_ref_cost_unknown = 0
    for d in ok:
        served = _served_attempt(d)
        is_skipped = (
            served is not None
            and served.accepted
            and served.error is None
            and (final is None or d.final_tier != final)
        )
        if d.mode == "eval":
            last = next((a for a in reversed(d.attempts) if a.tier == final), None)
            if last is not None and last.error is None:
                c = _attempt_cost(last)
                if c is None:
                    n_ref_cost_unknown += 1
                else:
                    ref_costs.append(c)
                    ref_weights.append(1.0)
        elif not is_skipped and reference_tier == final and served is not None:
            if served.tier == reference_tier and served.error is None:
                c = _attempt_cost(served)
                if c is None:
                    n_ref_cost_unknown += 1
                else:
                    ref_costs.append(c)
                    ref_weights.append(1.0)
        if not is_skipped:
            continue
        assert served is not None
        score = served.confidence.score if served.confidence is not None else None
        if score is not None and math.isnan(score):
            score = None
        case = _Case(score=score, pi=None, state="none", wrong=_flip(_decided(served.correct)))
        if d.mode == "eval":
            case.pi = 1.0
            last = next((a for a in reversed(d.attempts) if a.tier == final), None)
            agreement = served.agreement
            if agreement is None:
                case.state = "error" if last is None or last.error is not None else "none"
            elif _decided(agreement) is None:
                case.state = "undecided"
            else:
                case.state = "audited"
                case.disagree = not _decided(agreement)
                if last is not None:
                    case.ref_wrong = _flip(_decided(last.correct))
        else:
            sh = d.shadow
            if sh is not None:
                pi = sh.inclusion_prob
                if not (isinstance(pi, (int, float)) and math.isfinite(pi) and pi > 0):
                    case.state = "invalid"
                else:
                    case.pi = min(1.0, float(pi))
                    if sh.status == "done":
                        agreement = sh.agreement if sh.agreement is not None else served.agreement
                        eq = _decided(agreement)
                        if eq is None:
                            case.state = "undecided"
                        else:
                            case.state = "audited"
                            case.disagree = not eq
                            if sh.attempt is not None:
                                case.ref_wrong = _flip(_decided(sh.attempt.correct))
                        if sh.attempt is not None and sh.attempt.error is None:
                            c = _attempt_cost(sh.attempt)
                            if c is None:
                                n_ref_cost_unknown += 1
                            elif sh.audit_tier == reference_tier:
                                ref_costs.append(c)
                                ref_weights.append(1.0 / case.pi)
                    elif sh.status == "pending":
                        case.state = "pending"
                    elif sh.status == "error":
                        case.state = "error"
                    elif sh.status == "skipped":
                        case.state = "not-selected"
                    else:
                        case.state = "unknown"
        cases.append(case)

    n_skipped = len(cases)
    states = Counter(c.state for c in cases)
    audited = [c for c in cases if c.state == "audited"]
    n_audited = len(audited)
    weights = [1.0 / c.pi for c in audited if c.pi]

    disagreement: Estimate | None = None
    disagreement_unweighted: Estimate | None = None
    audit_tier_error: Estimate | None = None
    if audited:
        ys = [bool(c.disagree) for c in audited]
        disagreement = weighted_proportion(ys, weights, level)
        disagreement_unweighted = wilson(sum(ys), len(ys), level)
        ref_graded = [
            (c.ref_wrong, 1.0 / c.pi) for c in audited if c.ref_wrong is not None and c.pi
        ]
        if ref_graded:
            audit_tier_error = weighted_proportion(
                [g for g, _ in ref_graded], [w for _, w in ref_graded], level
            )

    graded_skipped = [c for c in cases if c.wrong is not None]
    n_graded_skipped = len(graded_skipped)
    skipped_error = (
        wilson(sum(1 for c in graded_skipped if c.wrong), n_graded_skipped, level)
        if graded_skipped
        else None
    )
    graded_decisions = [_decided(d.correct) for d in decisions]
    graded_decisions = [g for g in graded_decisions if g is not None]
    served_accuracy = (
        wilson(sum(graded_decisions), len(graded_decisions), level) if graded_decisions else None
    )
    has_refs = bool(graded_decisions) or n_graded_skipped > 0

    # ---- notes on audit coverage
    if n == 0:
        notes.append("No decisions.")
    elif n_skipped == 0:
        notes.append("No skipped cases: every decision escalated to the final tier or failed.")
    if final is None and ok:
        notes.append(
            "The final tier was never observed; every served tier is treated as non-final."
        )
    if n_audited and not has_refs:
        notes.append(
            f"The audit tier ({reference_tier}) is a proxy, not ground truth: disagreement "
            "counts differences from its answers, not errors."
        )
    excluded = [
        (states["pending"], "pending"),
        (states["error"], "errored"),
        (states["undecided"], "undecided (judge could not decide)"),
    ]
    for count, what in excluded:
        if count:
            notes.append(
                f"{count} audit(s) {what} were excluded; if they differ systematically from "
                "completed audits the disagreement estimate is biased."
            )
    if states["none"]:
        notes.append(
            f"{states['none']} skipped case(s) have no audit record (audit off?); the "
            "disagreement estimate covers only skipped cases with a recorded inclusion "
            "probability."
        )
    if states["invalid"]:
        notes.append(
            f"{states['invalid']} skipped case(s) have an invalid inclusion probability and "
            "were excluded."
        )
    if states["unknown"]:
        notes.append(f"{states['unknown']} skipped case(s) have an unknown audit status.")
    if n_skipped and not audited and not graded_skipped:
        notes.append("No completed audits: the skipped-case disagreement cannot be estimated.")
    if disagreement is not None and (disagreement.n_eff or 0.0) < LOW_N_EFF:
        notes.append(
            f"Low effective sample size (n_eff = {disagreement.n_eff:.1f} from {n_audited} "
            "audits); the interval is wide and its coverage approximate."
        )
    # Strata with tiny inclusion probabilities and few audits.
    strata: dict[float, list[_Case]] = {}
    for c in cases:
        if c.pi is not None:
            strata.setdefault(round(c.pi, 6), []).append(c)
    for pi, members in sorted(strata.items()):
        k = sum(1 for c in members if c.state == "audited")
        if pi < TINY_PI and k < MIN_AUDITS_PER_STRATUM:
            notes.append(
                f"{len(members)} skipped case(s) were sampled at pi = {pi:g} but only {k} "
                f"audit(s) completed there; each carries weight {1 / pi:.0f}, so the estimate "
                "is sensitive to them."
            )
    if len(weights) >= 2 and max(weights) / math.fsum(weights) > HEAVY_WEIGHT_SHARE:
        notes.append(
            f"A single audit carries {max(weights) / math.fsum(weights):.0%} of the total "
            "weight; the estimate leans heavily on it."
        )

    # ---- status
    every_graded = n_skipped > 0 and n_graded_skipped == n_skipped
    metric_name: str | None
    metric: Estimate | None
    if every_graded:
        metric_name, metric = "skipped_error", skipped_error
    elif disagreement is not None:
        metric_name, metric = "disagreement", disagreement
        if skipped_error is not None:
            notes.append(
                f"Only {n_graded_skipped} of {n_skipped} skipped cases have references; status "
                "uses the audit disagreement."
            )
    else:
        metric_name, metric = None, None
    audits_to_resolve: int | None = None
    if metric is None or metric.value is None:
        status = "no-data"
        metric_name = None
    elif tolerance is None:
        status = "n/a"
    else:
        assert metric.lo is not None and metric.hi is not None
        if metric.lo > tolerance:
            status = "breach"
        elif metric.hi <= tolerance:
            status = "ok"
        else:
            status = "inconclusive"
            audits_to_resolve = _audits_to_resolve(metric, float(tolerance), level)

    # ---- expected wrong-but-kept answers
    expected_wrong: tuple[float, float, float] | None = None
    wrong_source: str | None = None
    rate, wrong_source = (
        (skipped_error, "skipped_error")
        if skipped_error is not None
        else (disagreement, "disagreement")
    )
    if rate is not None and rate.value is not None and rate.lo is not None and rate.hi is not None:
        expected_wrong = (rate.value * n_skipped, rate.lo * n_skipped, rate.hi * n_skipped)
        if wrong_source == "skipped_error" and n_graded_skipped < n_skipped:
            notes.append(
                "expected_wrong_skipped scales the error rate of graded skipped cases to all "
                f"{n_skipped} skipped cases."
            )
        elif wrong_source == "disagreement":
            notes.append(
                "expected_wrong_skipped counts disagreements with the audit tier, not "
                "verified errors."
            )
    else:
        wrong_source = None

    # ---- bins
    bin_cases: list[list[_Case]] = [[] for _ in range(len(edges) - 1)]
    no_score: list[_Case] = []
    for c in cases:
        if c.score is None:
            no_score.append(c)
        else:
            bin_cases[_bin_of(c.score, edges)].append(c)
    bin_list: list[BinSummary] = []
    for i, members in enumerate(bin_cases):
        lo, hi = edges[i], edges[i + 1]
        close = "]" if i == len(bin_cases) - 1 else ")"
        label = f"[{_fmt_edge(lo)}, {_fmt_edge(hi)}{close}"
        bin_list.append(_bin_summary(lo, hi, label, members, level))
    no_score_bin = (
        _bin_summary(math.nan, math.nan, "no score", no_score, level) if no_score else None
    )

    # ---- costs
    serving = [d.cost_usd for d in decisions if d.cost_usd is not None]
    n_unknown_cost = n - len(serving)
    audit_costs = [d.audit_cost_usd for d in decisions if d.audit_cost_usd is not None]
    n_unknown_audit_cost = n - len(audit_costs)
    cost_serving = math.fsum(serving) if serving else None
    cost_audit = math.fsum(audit_costs) if audit_costs else None
    cost_per_task = cost_serving / len(serving) if cost_serving is not None else None
    audit_overhead = (
        cost_audit / cost_serving
        if cost_audit is not None and cost_serving is not None and cost_serving > 0
        else None
    )
    if n_unknown_cost or n_unknown_audit_cost:
        notes.append(
            f"Unknown cost for {n_unknown_cost} decision(s) (serving) and "
            f"{n_unknown_audit_cost} (audit); cost totals sum known values only and "
            "cost_per_task averages over decisions with a known serving cost."
        )
    all_slow = _weighted_mean(ref_costs, ref_weights, level)
    if all_slow is not None and mode != "eval" and (states["none"] or states["invalid"]):
        all_slow = None
        notes.append(
            "No all-slow cost estimate: some skipped cases have no usable inclusion probability, "
            "so reference-tier costs cannot be reweighted to the whole run."
        )
    if n_ref_cost_unknown:
        notes.append(
            f"{n_ref_cost_unknown} reference-tier attempt(s) have unknown cost and were left out "
            "of the all-slow cost estimate."
        )
    if all_slow is not None and mode != "eval" and reference_tier != final:
        notes.append(
            f"The all-slow cost uses audit tier {reference_tier!r}, which is not the final tier; "
            "it reflects skipped cases only."
        )
    est_savings: Estimate | None = None
    if (
        all_slow is not None
        and all_slow.value is not None
        and all_slow.value > 0
        and cost_per_task is not None
    ):

        def sav(a: float | None) -> float | None:
            return None if a is None or a <= 0 else 1.0 - cost_per_task / a

        est_savings = Estimate(
            sav(all_slow.value),
            sav(all_slow.lo),
            sav(all_slow.hi),
            all_slow.n,
            "ratio-approx",
            level,
            n_eff=all_slow.n_eff,
        )
        notes.append(
            "est_savings treats serving cost per task as exact; its interval reflects only the "
            "uncertainty in the all-slow cost estimate, and excludes audit cost."
        )

    return AuditSummary(
        run_id=run_id,
        mode=mode,
        n_decisions=n,
        n_errors=n_errors,
        n_skipped=n_skipped,
        escalation_rate=escalation_rate,
        tier_share=tier_share,
        n_audited=n_audited,
        n_pending=states["pending"],
        n_audit_errors=states["error"],
        n_undecided=states["undecided"],
        disagreement=disagreement,
        skipped_error=skipped_error,
        audit_tier_error=audit_tier_error,
        served_accuracy=served_accuracy,
        expected_wrong_skipped=expected_wrong,
        bins=bin_list,
        cost_serving=cost_serving,
        cost_audit=cost_audit,
        cost_per_task=cost_per_task,
        audit_overhead=audit_overhead,
        est_all_slow_cost_per_task=all_slow,
        est_savings=est_savings,
        tolerance=None if tolerance is None else float(tolerance),
        status=status,
        audits_to_resolve=audits_to_resolve,
        notes=notes,
        tiers=tiers,
        final_tier=final,
        reference_tier=reference_tier,
        disagreement_unweighted=disagreement_unweighted,
        n_graded_skipped=n_graded_skipped,
        n_not_selected=states["not-selected"],
        n_no_audit_record=states["none"],
        n_unknown_cost=n_unknown_cost,
        n_unknown_audit_cost=n_unknown_audit_cost,
        status_metric=metric_name,
        expected_wrong_source=wrong_source,
        no_score_bin=no_score_bin,
        level=level,
    )


def _flip(x: bool | None) -> bool | None:
    return None if x is None else not x
