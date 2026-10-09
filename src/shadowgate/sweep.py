"""Offline threshold sweeps over eval-mode runs.

In eval mode every tier answers every task and every non-final tier records its confidence, so
any threshold vector can be simulated after the fact without new model calls. :func:`sweep`
simulates a grid of threshold vectors, draws the accuracy/cost Pareto frontier, compares it with
single-tier baselines and an oracle router, reports how well each confidence signal is calibrated
and recommends thresholds chosen on one split of the data and validated on a held-out split.

Semantics
---------
* **Usable records.** A decision is used when it has no ``error``, has one attempt per tier (in
  the tier order of the run), and no attempt has an ``error`` or a missing completion. Other
  decisions are skipped and counted in ``SweepResult.n_skipped_records``; skipping can bias the
  estimates if failures correlate with difficulty, so a note is added.
* **Truth.** ``truth="reference"`` grades each tier's answer with ``Attempt.correct`` (vs the
  task's reference answer). ``truth="audit-tier"`` grades non-final tiers by
  ``Attempt.agreement`` (agreement with the last tier) and counts the last tier as correct by
  definition, so "accuracy" then means *agreement with the reference (last) tier*, not
  correctness. ``truth="auto"`` picks "reference" when every usable decision has
  ``Attempt.correct`` on every tier, else "audit-tier".
* **Undecidable judgements** (``Judgement.equivalent is None``, or a missing agreement in
  audit-tier mode) are counted as incorrect; the number of decisions with at least one such
  judgement is ``SweepResult.n_undecided``.
* **Simulation.** Walk the tiers in order and serve from the first tier whose confidence is
  ``>=`` its threshold; a missing (None/NaN) score never accepts. The cost of a decision is the
  sum, over every tier walked, of the attempt's completion cost plus its confidence-call costs
  (grading/agreement judge calls are evaluation overhead and are excluded). Latency is summed the
  same way. An unknown (None) cost on any walked tier makes that point's ``cost_per_task`` None.
* **Points, frontier and baselines** are evaluated on all usable decisions. The recommendation's
  ``point`` is evaluated on the selection split and ``holdout`` on the held-out split.
* **Baselines.** ``"only:<tier>"`` always serves from that tier and pays only that tier's
  completion (no confidence calls). ``"oracle"`` serves from the cheapest tier whose answer is
  correct (by that tier's completion cost; unknown costs rank last, ties go to the earlier tier),
  or the last tier when none is; it pays only the chosen tier's completion, never the tiers it
  skipped, so its cost is a lower bound no real router can reach. Baselines have empty
  ``thresholds``.
"""

from __future__ import annotations

import bisect
import itertools
import logging
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from shadowgate import stats
from shadowgate.errors import InsufficientData
from shadowgate.stats import Bin, Estimate
from shadowgate.types import Attempt, Decision, Judgement

__all__ = [
    "OperatingPoint",
    "CalibrationReport",
    "Recommendation",
    "SweepResult",
    "OBJECTIVES",
    "TRUTHS",
    "MAX_COMBINATIONS",
    "NEVER_ACCEPT",
    "sweep",
    "simulate",
    "pareto_frontier",
    "fit_isotonic",
]

log = logging.getLogger("shadowgate.sweep")

OBJECTIVES = ("max-savings", "max-accuracy", "min-accuracy")
TRUTHS = ("auto", "reference", "audit-tier")
MAX_COMBINATIONS = 20_000
MAX_GRID_POINTS = 101
NEVER_ACCEPT = 1.0001  # threshold above any score in [0, 1]: always escalate
_EPS = 1e-12
_BOOTSTRAP_DRAWS = 1_000_000  # work budget (reps x n) for the paired bootstrap of delta_vs_best


def _bootstrap_reps(n: int) -> int:
    """2000 replicates for small samples, fewer (never below 200) for large ones."""
    return min(2000, max(200, _BOOTSTRAP_DRAWS // max(1, n)))


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class OperatingPoint:
    """The outcome of routing with one threshold vector (or of a baseline router)."""

    thresholds: tuple[float, ...]  # one per non-final tier; () for baselines
    accuracy: Estimate  # Wilson interval vs truth
    cost_per_task: float | None  # None when any walked tier had an unknown cost
    latency_per_task: float
    escalation_rate: float  # share not served by the first tier
    tier_share: tuple[float, ...]  # share served by each tier (sums to 1)
    skipped_error: Estimate | None  # error rate among cases served by a non-final tier


@dataclass(frozen=True)
class CalibrationReport:
    """How well one non-final tier's confidence predicts that tier being correct (vs truth)."""

    tier: str
    n: int  # decisions with a usable score
    ece: float | None
    brier: float | None
    auroc: float | None
    aurc: float | None
    reliability: list[Bin]
    risk_coverage: list[tuple[float, float, float]]  # (threshold, coverage, risk)
    score_missing: int  # decisions whose score was None/NaN (excluded from the metrics)


@dataclass(frozen=True)
class Recommendation:
    """Thresholds chosen on the selection split, re-evaluated on the held-out split."""

    objective: str
    thresholds: tuple[float, ...]
    point: OperatingPoint  # on the selection split
    holdout: OperatingPoint | None  # same thresholds on the held-out split (None: no holdout)
    note: str
    # accuracy(recommended) - accuracy(best single tier) on the held-out split (selection split
    # when there is none), paired percentile bootstrap; 2000 replicates up to n=500, then
    # max(200, 1e6 // n) to bound run time.
    delta_vs_best: Estimate | None = None
    best_single_tier: str | None = None  # best "only:<tier>" baseline on the selection split
    n_selection: int = 0
    n_holdout: int = 0
    point_index: int | None = None  # index into SweepResult.points (same thresholds, all data)


@dataclass(frozen=True)
class SweepResult:
    truth: str  # "reference" | "audit-tier"
    n: int  # usable decisions
    tiers: tuple[str, ...]
    points: list[OperatingPoint]
    frontier: list[int]  # indices into points, Pareto-optimal, sorted by cost
    baselines: dict[str, OperatingPoint]  # "only:<tier>" for each tier, "oracle"
    calibration: dict[str, CalibrationReport]  # per non-final tier
    recommendation: Recommendation | None
    n_skipped_records: int = 0
    n_undecided: int = 0
    notes: list[str] = field(default_factory=list)
    grids: tuple[tuple[float, ...], ...] = ()  # threshold grid used for each non-final tier


# --------------------------------------------------------------------------- per-decision rows


@dataclass(frozen=True)
class _Item:
    task_id: str
    scores: tuple[float | None, ...]  # per non-final tier
    correct: tuple[int, ...]  # per tier, 0/1
    cost: tuple[float | None, ...]  # per tier: completion + confidence calls
    base_cost: tuple[float | None, ...]  # per tier: completion only
    latency: tuple[float, ...]  # per tier: completion + confidence calls
    base_latency: tuple[float, ...]  # per tier: completion only


def _sum_known(values: Iterable[float | None]) -> float | None:
    total = 0.0
    for v in values:
        if v is None:
            return None
        total += v
    return total


def _score(att: Attempt) -> float | None:
    conf = att.confidence
    if conf is None or conf.score is None:
        return None
    s = float(conf.score)
    return None if math.isnan(s) else s


def _graded(j: Judgement | None) -> tuple[int, bool]:
    """(0/1 correctness, undecided?) from a judgement."""
    if j is None or j.equivalent is None:
        return 0, True
    return int(bool(j.equivalent)), False


def _usable(d: Decision, tiers: tuple[str, ...]) -> bool:
    if d.error is not None or len(d.attempts) != len(tiers):
        return False
    return all(a.error is None and a.completion is not None for a in d.attempts)


def _collect(
    decisions: Sequence[Decision],
) -> tuple[tuple[str, ...], list[Decision], int]:
    if not decisions:
        raise ValueError("sweep needs at least one decision")
    run_ids = sorted({d.run_id for d in decisions})
    if len(run_ids) > 1:
        raise ValueError(
            f"decisions come from {len(run_ids)} runs ({', '.join(map(repr, run_ids[:5]))}); "
            "sweep one run at a time (select it with --run-id)"
        )
    modes = sorted({d.mode for d in decisions})
    if modes != ["eval"]:
        raise ValueError(
            f"threshold sweeps need an eval-mode run (every tier answers every task), but this "
            f"run has mode {', '.join(map(repr, modes))}; re-run it with `--mode eval`"
        )
    longest = max(decisions, key=lambda d: len(d.attempts))
    tiers = tuple(a.tier for a in longest.attempts)
    if not tiers:
        raise ValueError("no decision has any attempts; nothing to sweep")
    for d in decisions:
        names = tuple(a.tier for a in d.attempts)
        if names != tiers[: len(names)]:
            raise ValueError(
                f"decision for task {d.task.id!r} has tiers {names}, inconsistent with {tiers}; "
                "a run must use a single cascade configuration"
            )
    usable = [d for d in decisions if _usable(d, tiers)]
    return tiers, usable, len(decisions) - len(usable)


def _items(decisions: Sequence[Decision], truth: str) -> tuple[list[_Item], int]:
    items: list[_Item] = []
    undecided = 0
    for d in decisions:
        atts = d.attempts
        last = len(atts) - 1
        correct: list[int] = []
        und = False
        for j, a in enumerate(atts):
            if truth == "reference":
                c, u = _graded(a.correct)
            elif j == last:
                c, u = 1, False
            else:
                c, u = _graded(a.agreement)
            correct.append(c)
            und = und or u
        undecided += und
        cost: list[float | None] = []
        base_cost: list[float | None] = []
        latency: list[float] = []
        base_latency: list[float] = []
        for a in atts:
            comp = a.completion
            assert comp is not None
            calls = a.confidence.calls if a.confidence is not None else ()
            base_cost.append(comp.cost_usd)
            cost.append(_sum_known([comp.cost_usd, *(c.cost_usd for c in calls)]))
            base_latency.append(float(comp.latency_s))
            latency.append(float(comp.latency_s) + sum(float(c.latency_s) for c in calls))
        items.append(
            _Item(
                task_id=d.task.id,
                scores=tuple(_score(a) for a in atts[:last]),
                correct=tuple(correct),
                cost=tuple(cost),
                base_cost=tuple(base_cost),
                latency=tuple(latency),
                base_latency=tuple(base_latency),
            )
        )
    return items, undecided


def _resolve_truth(truth: str, decisions: Sequence[Decision]) -> str:
    if truth not in TRUTHS:
        raise ValueError(f"truth must be one of {TRUTHS}, got {truth!r}")
    has_ref = all(a.correct is not None for d in decisions for a in d.attempts)
    if truth == "auto":
        return "reference" if has_ref else "audit-tier"
    if truth == "reference" and not has_ref:
        raise ValueError(
            "truth='reference' needs Attempt.correct on every tier of every usable decision "
            "(tasks with references and a comparator); use truth='audit-tier' or 'auto'"
        )
    return truth


# --------------------------------------------------------------------------- grids


def _thin(values: Sequence[float], k: int) -> list[float]:
    """At most ``k`` evenly spaced (by rank) values, keeping both ends."""
    if len(values) <= k:
        return list(values)
    if k <= 1:
        return [values[0]]
    idx = sorted({round(i * (len(values) - 1) / (k - 1)) for i in range(k)})
    return [values[i] for i in idx]


def _default_grid(scores: Iterable[float | None]) -> list[float]:
    uniq = sorted({s for s in scores if s is not None})
    grid = set(_thin(uniq, MAX_GRID_POINTS))
    grid.update((0.0, NEVER_ACCEPT))
    return sorted(grid)


def _per_tier_cap(m: int) -> int:
    if m <= 1:
        return MAX_COMBINATIONS
    k = max(2, int(MAX_COMBINATIONS ** (1.0 / m)))
    while (k + 1) ** m <= MAX_COMBINATIONS:
        k += 1
    while k > 2 and k**m > MAX_COMBINATIONS:
        k -= 1
    return k


def _grids(items: Sequence[_Item], m: int, grid: Sequence[float] | None) -> list[list[float]]:
    if grid is not None:
        vals = [float(g) for g in grid]
        if not vals or any(math.isnan(v) for v in vals):
            raise ValueError("grid must be a non-empty sequence of numbers (no NaN)")
        base = sorted(set(vals))
        grids = [list(base) for _ in range(m)]
    else:
        grids = [_default_grid(it.scores[j] for it in items) for j in range(m)]
    cap = _per_tier_cap(m)
    return [_thin(g, cap) for g in grids]


# --------------------------------------------------------------------------- fast evaluation


class _Table:
    """Multi-dimensional prefix sums so every threshold vector is evaluated in O(m).

    For tier ``j`` the bucket ``b_j = bisect_right(grid_j, score_j)`` (0 for a missing score)
    satisfies ``score_j >= grid_j[g]  <=>  b_j > g``. A decision walks to tier ``k`` iff
    ``b_i <= g_i`` for all ``i < k``, so every walked-to-``k`` sum is a dominance count over the
    first ``k`` bucket coordinates: an inclusive prefix sum of a ``(G_0+1) x ... x (G_{k-1}+1)``
    array. Building costs O(n m + prod(G+1)); each query costs O(m).
    """

    def __init__(self, items: Sequence[_Item], grids: Sequence[Sequence[float]]) -> None:
        m = len(grids)
        self.m = m
        self.n = len(items)
        sizes = [len(g) + 1 for g in grids]
        buckets = [
            tuple(
                0 if it.scores[j] is None else bisect.bisect_right(grids[j], it.scores[j])
                for j in range(m)
            )
            for it in items
        ]
        # Per walked-to level k: strides and arrays [count, cost, unknown, latency, c_k, c_{k-1}].
        self.strides: list[list[int]] = []
        self.arrays: list[list[list[float]]] = []
        for k in range(m + 1):
            dims = sizes[:k]
            strides = [1] * k
            for a in range(k - 2, -1, -1):
                strides[a] = strides[a + 1] * dims[a + 1]
            total = math.prod(dims)
            arrs = [[0.0] * total for _ in range(6)]
            cnt, cost, unk, lat, ck, cprev = arrs
            for it, b in zip(items, buckets, strict=True):
                idx = sum(b[a] * strides[a] for a in range(k))
                cnt[idx] += 1
                c = it.cost[k]
                if c is None:
                    unk[idx] += 1
                else:
                    cost[idx] += c
                lat[idx] += it.latency[k]
                ck[idx] += it.correct[k]
                if k:
                    cprev[idx] += it.correct[k - 1]
            for a in range(k):
                st, sz = strides[a], dims[a]
                live = [idx for idx in range(total) if (idx // st) % sz]
                for arr in arrs:
                    for idx in live:
                        arr[idx] += arr[idx - st]
            self.strides.append(strides)
            self.arrays.append(arrs)

    def query(self, g: Sequence[int]) -> tuple[int, list[int], float | None, float, int]:
        """(correct, served per tier, total cost or None, total latency, correct on skipped)."""
        m = self.m
        idxs = [sum(g[a] * self.strides[k][a] for a in range(k)) for k in range(m + 1)]
        walked = [int(round(self.arrays[k][0][idxs[k]])) for k in range(m + 1)]
        cost: float | None = 0.0
        latency = 0.0
        correct_skipped = 0
        for k in range(m + 1):
            arrs, i = self.arrays[k], idxs[k]
            if cost is not None:
                cost = None if arrs[2][i] > 0 else cost + arrs[1][i]
            latency += arrs[3][i]
            if k < m:
                served_ok = arrs[4][i] - self.arrays[k + 1][5][idxs[k + 1]]
                correct_skipped += int(round(served_ok))
        correct = correct_skipped + int(round(self.arrays[m][4][idxs[m]]))
        served = [walked[k] - walked[k + 1] for k in range(m)] + [walked[m]]
        return correct, served, cost, latency, correct_skipped


def _make_point(
    thresholds: tuple[float, ...],
    n: int,
    correct: int,
    served: Sequence[int],
    cost: float | None,
    latency: float,
    correct_skipped: int | None,
    level: float,
) -> OperatingPoint:
    n_skipped = n - served[-1]
    skipped_error = None
    if correct_skipped is not None and n_skipped > 0:
        skipped_error = stats.wilson(n_skipped - correct_skipped, n_skipped, level)
    return OperatingPoint(
        thresholds=thresholds,
        accuracy=stats.wilson(correct, n, level),
        cost_per_task=None if cost is None or n == 0 else cost / n,
        latency_per_task=latency / n if n else 0.0,
        escalation_rate=(n - served[0]) / n if n else 0.0,
        tier_share=tuple(s / n if n else 0.0 for s in served),
        skipped_error=skipped_error,
    )


# --------------------------------------------------------------------------- direct simulation


def _served_tier(it: _Item, thresholds: Sequence[float]) -> int:
    for j, t in enumerate(thresholds):
        s = it.scores[j]
        if s is not None and s >= t:
            return j
    return len(thresholds)


def _simulate_items(
    items: Sequence[_Item], thresholds: tuple[float, ...], level: float
) -> tuple[OperatingPoint, list[int]]:
    m = len(items[0].scores) if items else len(thresholds)
    if len(thresholds) != m:
        raise ValueError(f"expected {m} thresholds (one per non-final tier), got {len(thresholds)}")
    served = [0] * (m + 1)
    correct = correct_skipped = 0
    cost: float | None = 0.0
    latency = 0.0
    per_item: list[int] = []
    for it in items:
        j = _served_tier(it, thresholds)
        served[j] += 1
        c = it.correct[j]
        per_item.append(c)
        correct += c
        if j < m:
            correct_skipped += c
        walked_cost = _sum_known(it.cost[: j + 1])
        cost = None if cost is None or walked_cost is None else cost + walked_cost
        latency += sum(it.latency[: j + 1])
    point = _make_point(
        thresholds, len(items), correct, served, cost, latency, correct_skipped, level
    )
    return point, per_item


def _only(
    items: Sequence[_Item], k: int, n_tiers: int, level: float
) -> tuple[OperatingPoint, list[int]]:
    served = [0] * n_tiers
    served[k] = len(items)
    per_item = [it.correct[k] for it in items]
    cost = _sum_known(it.base_cost[k] for it in items)
    latency = sum(it.base_latency[k] for it in items)
    correct = sum(per_item)
    final = k == n_tiers - 1
    n = len(items)
    point = OperatingPoint(
        thresholds=(),
        accuracy=stats.wilson(correct, n, level),
        cost_per_task=None if cost is None or n == 0 else cost / n,
        latency_per_task=latency / n if n else 0.0,
        escalation_rate=0.0,
        tier_share=tuple(s / n if n else 0.0 for s in served),
        skipped_error=None if final or n == 0 else stats.wilson(n - correct, n, level),
    )
    return point, per_item


def _oracle(items: Sequence[_Item], n_tiers: int, level: float) -> OperatingPoint:
    served = [0] * n_tiers
    correct = correct_skipped = 0
    cost: float | None = 0.0
    latency = 0.0
    for it in items:
        ok = [j for j in range(n_tiers) if it.correct[j]]
        if ok:
            j = min(ok, key=lambda t: (math.inf if it.base_cost[t] is None else it.base_cost[t], t))
            correct += 1
            if j < n_tiers - 1:
                correct_skipped += 1
        else:
            j = n_tiers - 1
        served[j] += 1
        c = it.base_cost[j]
        cost = None if cost is None or c is None else cost + c
        latency += it.base_latency[j]
    return _make_point((), len(items), correct, served, cost, latency, correct_skipped, level)


# --------------------------------------------------------------------------- frontier & choice


def _cost_key(c: float) -> float:
    return float(f"{c:.12g}")


def pareto_frontier(points: Sequence[OperatingPoint]) -> list[int]:
    """Indices of non-dominated points (max accuracy, min cost), sorted by increasing cost.

    Points with an unknown cost are excluded. Among points with equal cost and accuracy only the
    lowest index is kept, so the result is deterministic.
    """
    known = [
        (i, p)
        for i, p in enumerate(points)
        if p.cost_per_task is not None and p.accuracy.value is not None
    ]
    known.sort(key=lambda ip: (_cost_key(ip[1].cost_per_task), -ip[1].accuracy.value, ip[0]))  # type: ignore[arg-type,operator]
    out: list[int] = []
    best = -math.inf
    for i, p in known:
        acc = p.accuracy.value
        assert acc is not None
        if acc > best + _EPS:
            out.append(i)
            best = acc
    return out


def _acc(p: OperatingPoint) -> float:
    return -math.inf if p.accuracy.value is None else p.accuracy.value


def _choose(
    points: Sequence[OperatingPoint],
    objective: str,
    *,
    target: float | None,
    budget: float | None,
) -> int | None:
    def cheap_key(i: int) -> tuple[float, float, float, tuple[float, ...]]:
        p = points[i]
        assert p.cost_per_task is not None
        return (_cost_key(p.cost_per_task), -_acc(p), p.latency_per_task, p.thresholds)

    if objective == "max-accuracy":
        cand = [
            i
            for i, p in enumerate(points)
            if budget is None or (p.cost_per_task is not None and p.cost_per_task <= budget + _EPS)
        ]
        if not cand:
            return None
        return min(
            cand,
            key=lambda i: (
                -_acc(points[i]),
                math.inf if points[i].cost_per_task is None else points[i].cost_per_task,
                points[i].latency_per_task,
                points[i].thresholds,
            ),
        )
    assert target is not None
    cand = [
        i for i, p in enumerate(points) if p.cost_per_task is not None and _acc(p) >= target - _EPS
    ]
    return min(cand, key=cheap_key) if cand else None


# --------------------------------------------------------------------------- calibration


def _calibration(tier: str, items: Sequence[_Item], j: int, level: float) -> CalibrationReport:
    scores: list[float] = []
    labels: list[bool] = []
    missing = 0
    for it in items:
        s = it.scores[j]
        if s is None:
            missing += 1
            continue
        scores.append(s)
        labels.append(bool(it.correct[j]))
    if not scores:
        return CalibrationReport(tier, 0, None, None, None, None, [], [], missing)
    in_range = all(0.0 <= s <= 1.0 for s in scores)
    clipped = [min(1.0, max(0.0, s)) for s in scores]
    return CalibrationReport(
        tier=tier,
        n=len(scores),
        ece=stats.ece(clipped, labels) if in_range else None,
        brier=stats.brier(clipped, labels) if in_range else None,
        auroc=stats.auroc(scores, labels),
        aurc=stats.aurc(scores, labels),
        reliability=stats.reliability(clipped, labels, level=level) if in_range else [],
        risk_coverage=stats.risk_coverage(scores, labels),
        score_missing=missing,
    )


# --------------------------------------------------------------------------- public API


def simulate(
    decisions: Iterable[Decision],
    thresholds: Sequence[float],
    *,
    truth: str = "auto",
    level: float = 0.95,
) -> OperatingPoint:
    """Evaluate one threshold vector on eval-mode decisions (same semantics as :func:`sweep`)."""
    tiers, usable, _ = _collect(list(decisions))
    if not usable:
        raise InsufficientData("no usable eval-mode decisions (all have errors or missing tiers)")
    items, _ = _items(usable, _resolve_truth(truth, usable))
    if len(thresholds) != len(tiers) - 1:
        raise ValueError(
            f"expected {len(tiers) - 1} thresholds (one per non-final tier), got {len(thresholds)}"
        )
    return _simulate_items(items, tuple(float(t) for t in thresholds), level)[0]


def sweep(
    decisions: Iterable[Decision],
    *,
    truth: str = "auto",
    grid: Sequence[float] | None = None,
    objective: str = "max-savings",
    max_accuracy_drop: float = 0.01,
    min_accuracy: float | None = None,
    budget_per_task: float | None = None,
    holdout: float = 0.3,
    seed: int = 0,
    level: float = 0.95,
) -> SweepResult:
    """Simulate every threshold vector on an eval-mode run and recommend one.

    ``grid`` (applied to every non-final tier) defaults per tier to the unique observed scores
    thinned to at most 101 rank-quantiles, plus ``0.0`` (never escalate) and ``1.0001`` (always
    escalate). With more than one non-final tier the joint grid is thinned per tier to at most
    ``MAX_COMBINATIONS`` (20,000) combinations.

    Objectives (evaluated on the selection split):

    * ``"max-savings"``: minimum cost with accuracy >= best single-tier accuracy (on the selection
      split) - ``max_accuracy_drop``;
    * ``"max-accuracy"``: maximum accuracy with cost <= ``budget_per_task`` (no budget: any cost);
    * ``"min-accuracy"``: minimum cost with accuracy >= ``min_accuracy``.

    Ties break towards higher accuracy / lower cost, lower latency, then lower thresholds.
    Decisions are ordered by task id and a ``random.Random(seed)`` permutation holds out
    ``round(holdout * n)`` of them; ``holdout=0`` selects on all data (in-sample, noted).
    When no point meets the objective's constraint the recommendation is None and a note says why.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective == "min-accuracy" and min_accuracy is None:
        raise ValueError("objective 'min-accuracy' needs min_accuracy")
    if not 0.0 <= holdout < 1.0:
        raise ValueError(f"holdout must be in [0, 1), got {holdout!r}")
    if max_accuracy_drop < 0:
        raise ValueError(f"max_accuracy_drop must be >= 0, got {max_accuracy_drop!r}")

    tiers, usable, n_skipped = _collect(list(decisions))
    notes: list[str] = []
    if n_skipped:
        notes.append(
            f"{n_skipped} decision(s) skipped (errors or missing tier attempts); if failures "
            "correlate with difficulty the estimates are biased"
        )
    if not usable:
        raise InsufficientData("no usable eval-mode decisions (all have errors or missing tiers)")
    usable.sort(key=lambda d: d.task.id)
    resolved = _resolve_truth(truth, usable)
    items, n_undecided = _items(usable, resolved)
    if resolved == "audit-tier":
        notes.append(
            f"no ground truth on every tier: accuracy is agreement with the reference tier "
            f"{tiers[-1]!r}, which counts as correct by definition"
        )
    if n_undecided:
        notes.append(f"{n_undecided} decision(s) had undecidable judgements, counted as incorrect")

    n_tiers = len(tiers)
    m = n_tiers - 1
    grids = _grids(items, m, grid)
    n = len(items)

    # Sweep on all usable data.
    table = _Table(items, grids)
    combos = list(itertools.product(*(range(len(g)) for g in grids)))
    points: list[OperatingPoint] = []
    for combo in combos:
        thr = tuple(grids[j][g] for j, g in enumerate(combo))
        correct, served, cost, latency, ok_skipped = table.query(combo)
        points.append(_make_point(thr, n, correct, served, cost, latency, ok_skipped, level))
    if any(p.cost_per_task is None for p in points):
        notes.append("some costs are unknown (unpriced model); those points have no cost")
    frontier = pareto_frontier(points)

    baselines: dict[str, OperatingPoint] = {
        f"only:{t}": _only(items, k, n_tiers, level)[0] for k, t in enumerate(tiers)
    }
    baselines["oracle"] = _oracle(items, n_tiers, level)
    calibration = {t: _calibration(t, items, j, level) for j, t in enumerate(tiers[:m])}

    # Selection / holdout split.
    order = list(range(n))
    random.Random(seed).shuffle(order)
    n_hold = min(n - 1, round(holdout * n)) if holdout > 0 else 0
    hold_idx = sorted(order[:n_hold])
    sel_idx = sorted(order[n_hold:])
    sel = [items[i] for i in sel_idx]
    hold = [items[i] for i in hold_idx]
    if holdout > 0 and n_hold == 0:
        notes.append("too few decisions for a held-out split; thresholds selected in-sample")

    sel_table = table if not hold else _Table(sel, grids)
    sel_points = (
        points
        if not hold
        else [
            _make_point(points[i].thresholds, len(sel), *sel_table.query(combo), level=level)
            for i, combo in enumerate(combos)
        ]
    )
    singles = [_only(sel, k, n_tiers, level)[0] for k in range(n_tiers)]
    best_k = max(range(n_tiers), key=lambda k: (_acc(singles[k]), -k))
    best_acc = _acc(singles[best_k])

    target: float | None = None
    if objective == "max-savings":
        target = best_acc - max_accuracy_drop
    elif objective == "min-accuracy":
        target = min_accuracy
    choice = _choose(sel_points, objective, target=target, budget=budget_per_task)

    recommendation: Recommendation | None = None
    if choice is None:
        if objective == "max-accuracy":
            why = f"no point has a known cost <= budget_per_task={budget_per_task}"
        else:
            why = f"no point with a known cost reaches accuracy >= {target:.4f}"
        notes.append(f"no recommendation for objective {objective!r}: {why}")
    else:
        thr = points[choice].thresholds
        if objective == "max-savings":
            parts = [
                f"cheapest thresholds with accuracy >= {target:.4f} (best single tier "
                f"{tiers[best_k]!r} at {best_acc:.4f} minus {max_accuracy_drop:g}) on the "
                f"selection split"
            ]
        elif objective == "min-accuracy":
            parts = [f"cheapest thresholds with accuracy >= {target:.4f} on the selection split"]
        else:
            budget_txt = "any cost" if budget_per_task is None else f"cost <= {budget_per_task}"
            parts = [f"most accurate thresholds with {budget_txt} on the selection split"]
        hold_point: OperatingPoint | None = None
        eval_items = sel
        if hold:
            hold_point, rec_items = _simulate_items(hold, thr, level)
            eval_items = hold
            parts.append(f"validated on {len(hold)} held-out decisions (seed {seed})")
            if (
                target is not None
                and hold_point.accuracy.value is not None
                and hold_point.accuracy.value < target - _EPS
            ):
                parts.append("held-out accuracy falls below the target; treat with caution")
        else:
            _, rec_items = _simulate_items(sel, thr, level)
            parts.append("no held-out split: estimates are in-sample and optimistic")
        _, best_items = _only(eval_items, best_k, n_tiers, level)
        delta = stats.paired_diff(
            [float(x) for x in rec_items],
            [float(x) for x in best_items],
            level=level,
            reps=_bootstrap_reps(len(eval_items)),
            seed=seed,
        )
        recommendation = Recommendation(
            objective=objective,
            thresholds=thr,
            point=sel_points[choice],
            holdout=hold_point,
            note="; ".join(parts),
            delta_vs_best=delta,
            best_single_tier=tiers[best_k],
            n_selection=len(sel),
            n_holdout=len(hold),
            point_index=choice,
        )

    return SweepResult(
        truth=resolved,
        n=n,
        tiers=tiers,
        points=points,
        frontier=frontier,
        baselines=baselines,
        calibration=calibration,
        recommendation=recommendation,
        n_skipped_records=n_skipped,
        n_undecided=n_undecided,
        notes=notes,
        grids=tuple(tuple(g) for g in grids),
    )


# --------------------------------------------------------------------------- isotonic fit


def fit_isotonic(
    scores: Sequence[float], labels: Sequence[bool | int]
) -> list[tuple[float, float]]:
    """Isotonic (monotone non-decreasing) fit of P(correct | score) by pool-adjacent-violators.

    Duplicate scores are merged first (weighted by count). Returns knots ``(x, y)`` with strictly
    increasing ``x`` and non-decreasing ``y`` in [0, 1]: for each pooled block, its first and last
    score with the block's mean label. Interpolating linearly between knots (as
    ``confidence.Calibrated(points=...)`` does) reproduces the fitted value at every observed
    score. Raises :class:`~shadowgate.errors.InsufficientData` on empty input.
    """
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels differ in length ({len(scores)} vs {len(labels)})")
    pairs: list[tuple[float, float]] = []
    for s, y in zip(scores, labels, strict=True):
        x = float(s)
        if not math.isfinite(x):
            raise ValueError(f"scores must be finite, got {s!r}")
        if y not in (0, 1):  # True/False compare equal to 1/0
            raise ValueError(f"labels must be booleans or 0/1, got {y!r}")
        pairs.append((x, float(y)))
    if not pairs:
        raise InsufficientData("fit_isotonic needs at least one observation")
    pairs.sort()
    # blocks: [x_first, x_last, sum_y, weight]
    blocks: list[list[float]] = []
    for x, y in pairs:
        if blocks and blocks[-1][1] == x:
            blocks[-1][2] += y
            blocks[-1][3] += 1
        else:
            blocks.append([x, x, y, 1.0])
    pooled: list[list[float]] = []
    for b in blocks:
        pooled.append(list(b))
        while len(pooled) > 1 and pooled[-2][2] / pooled[-2][3] >= pooled[-1][2] / pooled[-1][3]:
            top = pooled.pop()
            prev = pooled[-1]
            prev[1] = top[1]
            prev[2] += top[2]
            prev[3] += top[3]
    knots: list[tuple[float, float]] = []
    for x0, x1, sy, w in pooled:
        y = min(1.0, max(0.0, sy / w))
        knots.append((x0, y))
        if x1 > x0:
            knots.append((x1, y))
    return knots
