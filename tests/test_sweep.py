from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Sequence

import pytest

from shadowgate import sweep as sw
from shadowgate.confidence import Calibrated
from shadowgate.errors import InsufficientData
from shadowgate.stats import Estimate, brier
from shadowgate.sweep import (
    NEVER_ACCEPT,
    OperatingPoint,
    fit_isotonic,
    pareto_frontier,
    simulate,
    sweep,
)
from shadowgate.types import Attempt, Completion, ConfidenceResult, Decision, Judgement, Task

# ---------------------------------------------------------------------------- builders

_MISSING = object()


def _j(v: bool | None) -> Judgement:
    return Judgement(equivalent=v, comparator="exact")


def make_decision(
    task_id: str,
    tiers: Sequence[dict],
    *,
    run_id: str = "r1",
    mode: str = "eval",
    with_reference: bool = True,
    error: str | None = None,
) -> Decision:
    """Each tier dict: ok (bool|None), score (float|None, non-final), cost, conf_cost, lat,
    agree (bool|None; defaults to ok == last ok), err (str|None)."""
    attempts = []
    last = len(tiers) - 1
    for i, t in enumerate(tiers):
        name = t.get("name", f"t{i}")
        if t.get("err"):
            attempts.append(Attempt(name, "b", None, "", None, None, False, error=t["err"]))
            continue
        comp = Completion(
            text="x", model="m", cost_usd=t.get("cost", 1.0), latency_s=t.get("lat", 1.0)
        )
        conf = None
        if i < last:
            calls = ()
            if "conf_cost" in t:
                calls = (Completion(text="c", model="m", cost_usd=t["conf_cost"], latency_s=0.5),)
            conf = ConfidenceResult("verbal", t.get("score"), calls=calls)
        agree = None
        if i < last:
            agree_v = t.get("agree", _MISSING)
            if agree_v is _MISSING:
                agree_v = t["ok"] == tiers[-1]["ok"]
            agree = _j(agree_v)
        attempts.append(
            Attempt(
                tier=name,
                backend="b",
                completion=comp,
                answer="a",
                confidence=conf,
                threshold=0.5 if i < last else None,
                accepted=True,
                correct=_j(t["ok"]) if with_reference else None,
                agreement=agree,
            )
        )
    return Decision(
        run_id=run_id,
        task=Task(task_id, "p", "ref" if with_reference else None),
        answer="a",
        final_tier=attempts[-1].tier,
        escalated=False,
        attempts=tuple(attempts),
        cost_usd=None,
        audit_cost_usd=None,
        latency_s=0.0,
        mode=mode,
        error=error,
    )


def two_tier(score: float | None, ok_small: bool, ok_big: bool, tid: str, **kw) -> Decision:
    return make_decision(
        tid,
        [
            {"name": "small", "score": score, "ok": ok_small, "cost": 1.0, "conf_cost": 0.5},
            {"name": "big", "ok": ok_big, "cost": 10.0},
        ],
        **kw,
    )


def example2(**kw) -> list[Decision]:
    return [
        two_tier(0.9, True, True, "t1", **kw),
        two_tier(0.8, False, True, "t2", **kw),
        two_tier(0.6, True, True, "t3", **kw),
        two_tier(0.3, False, False, "t4", **kw),
    ]


def example3() -> list[Decision]:
    def d(tid, sa, oa, sb, ob, oc):
        return make_decision(
            tid,
            [
                {"name": "a", "score": sa, "ok": oa, "cost": 1.0},
                {"name": "b", "score": sb, "ok": ob, "cost": 2.0},
                {"name": "c", "ok": oc, "cost": 4.0},
            ],
        )

    return [
        d("t1", 0.9, True, 0.5, True, True),
        d("t2", 0.4, False, 0.9, True, True),
        d("t3", 0.2, False, 0.2, False, True),
    ]


def synthetic(n: int, seed: int = 0, tiers: int = 2, none_rate: float = 0.0) -> list[Decision]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        spec = []
        for k in range(tiers):
            skill = 0.55 + 0.4 * k / max(1, tiers - 1)
            ok = rng.random() < skill
            score = min(1.0, max(0.0, (0.75 if ok else 0.45) + rng.gauss(0, 0.2)))
            if rng.random() < none_rate:
                score = None
            spec.append(
                {
                    "name": f"tier{k}",
                    "score": round(score, 3) if score is not None else None,
                    "ok": ok,
                    "cost": 1.0 * 4**k,
                    "lat": 0.1 * (k + 1),
                }
            )
        out.append(make_decision(f"q{i:06d}", spec))
    return out


def point_for(res: sw.SweepResult, thresholds: tuple[float, ...]) -> OperatingPoint:
    matches = [p for p in res.points if p.thresholds == thresholds]
    assert len(matches) == 1
    return matches[0]


# ---------------------------------------------------------------------------- 2-tier by hand


def test_two_tier_default_grid():
    res = sweep(example2(), holdout=0)
    assert res.tiers == ("small", "big")
    assert res.truth == "reference"
    assert res.n == 4
    assert res.grids == ((0.0, 0.3, 0.6, 0.8, 0.9, NEVER_ACCEPT),)
    assert [p.thresholds for p in res.points] == [(t,) for t in res.grids[0]]


@pytest.mark.parametrize(
    ("thr", "acc", "cost", "share"),
    [
        (0.0, 0.5, 1.5, (1.0, 0.0)),
        (0.3, 0.5, 1.5, (1.0, 0.0)),
        (0.6, 0.5, 4.0, (0.75, 0.25)),
        (0.8, 0.5, 6.5, (0.5, 0.5)),
        (0.9, 0.75, 9.0, (0.25, 0.75)),
        (NEVER_ACCEPT, 0.75, 11.5, (0.0, 1.0)),
    ],
)
def test_two_tier_points_by_hand(thr, acc, cost, share):
    p = point_for(sweep(example2(), holdout=0), (thr,))
    assert p.accuracy.value == pytest.approx(acc)
    assert p.accuracy.n == 4
    assert p.accuracy.method == "wilson"
    assert p.cost_per_task == pytest.approx(cost)
    assert p.tier_share == pytest.approx(share)
    assert p.escalation_rate == pytest.approx(share[1])


def test_two_tier_latency_includes_confidence_calls():
    p = point_for(sweep(example2(), holdout=0), (0.9,))
    # every task walks small (1.0 + 0.5 confidence call); 3 of 4 also walk big (1.0)
    assert p.latency_per_task == pytest.approx(1.5 + 0.75)


def test_two_tier_skipped_error():
    res = sweep(example2(), holdout=0)
    p0 = point_for(res, (0.0,))
    assert p0.skipped_error is not None
    assert p0.skipped_error.value == pytest.approx(0.5)
    assert p0.skipped_error.n == 4
    assert point_for(res, (0.9,)).skipped_error.value == 0.0
    assert point_for(res, (NEVER_ACCEPT,)).skipped_error is None


def test_two_tier_frontier_by_hand():
    res = sweep(example2(), holdout=0)
    assert [res.points[i].thresholds for i in res.frontier] == [(0.0,), (0.9,)]


def test_two_tier_baselines():
    res = sweep(example2(), holdout=0)
    small, big = res.baselines["only:small"], res.baselines["only:big"]
    assert small.accuracy.value == 0.5 and small.cost_per_task == 1.0  # no confidence cost
    assert big.accuracy.value == 0.75 and big.cost_per_task == 10.0
    assert small.tier_share == (1.0, 0.0) and big.tier_share == (0.0, 1.0)
    assert small.thresholds == () and small.escalation_rate == 0.0
    assert small.skipped_error.value == 0.5 and big.skipped_error is None
    assert small.latency_per_task == 1.0


def test_oracle_pays_only_chosen_tier():
    o = sweep(example2(), holdout=0).baselines["oracle"]
    assert o.accuracy.value == 0.75
    assert o.cost_per_task == pytest.approx((1 + 10 + 1 + 10) / 4)
    assert o.tier_share == (0.5, 0.5)
    assert o.escalation_rate == 0.5


def test_oracle_picks_cheapest_correct_tier_by_cost():
    d = make_decision(
        "x",
        [
            {"name": "a", "score": 0.5, "ok": True, "cost": 5.0},
            {"name": "b", "ok": True, "cost": 2.0},
        ],
    )
    o = sweep([d], holdout=0).baselines["oracle"]
    assert o.cost_per_task == 2.0 and o.tier_share == (0.0, 1.0)


def test_max_savings_recommendation_in_sample():
    res = sweep(example2(), holdout=0, max_accuracy_drop=0.0)
    rec = res.recommendation
    assert rec is not None
    assert rec.thresholds == (0.9,)
    assert rec.point.cost_per_task == pytest.approx(9.0)
    assert rec.holdout is None
    assert rec.best_single_tier == "big"
    assert rec.n_selection == 4 and rec.n_holdout == 0
    assert res.points[rec.point_index].thresholds == (0.9,)
    assert "in-sample" in rec.note
    assert rec.delta_vs_best.value == pytest.approx(0.0)


def test_max_savings_with_drop_allows_cheaper_point():
    rec = sweep(example2(), holdout=0, max_accuracy_drop=0.25).recommendation
    assert rec.thresholds == (0.0,)


def test_max_accuracy_with_budget():
    rec = sweep(example2(), holdout=0, objective="max-accuracy", budget_per_task=5.0).recommendation
    assert rec.thresholds == (0.0,)  # 0.5 accuracy; cheapest among ties
    rec = sweep(example2(), holdout=0, objective="max-accuracy").recommendation
    assert rec.thresholds == (0.9,)


def test_min_accuracy_objective():
    rec = sweep(example2(), holdout=0, objective="min-accuracy", min_accuracy=0.7).recommendation
    assert rec.thresholds == (0.9,)
    with pytest.raises(ValueError, match="min_accuracy"):
        sweep(example2(), objective="min-accuracy")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"objective": "min-accuracy", "min_accuracy": 0.99},
        {"objective": "max-accuracy", "budget_per_task": 0.1},
    ],
)
def test_infeasible_objective_gives_none_with_note(kwargs):
    res = sweep(example2(), holdout=0, **kwargs)
    assert res.recommendation is None
    assert any("no recommendation" in n for n in res.notes)
    assert res.points and res.frontier  # the sweep itself is still reported


def test_bad_arguments():
    with pytest.raises(ValueError, match="objective"):
        sweep(example2(), objective="fastest")
    with pytest.raises(ValueError, match="holdout"):
        sweep(example2(), holdout=1.0)
    with pytest.raises(ValueError, match="truth"):
        sweep(example2(), truth="vibes")
    with pytest.raises(ValueError, match="grid"):
        sweep(example2(), grid=[])


def test_custom_grid():
    res = sweep(example2(), grid=[0.85, 0.0, 0.85], holdout=0)
    assert res.grids == ((0.0, 0.85),)
    assert point_for(res, (0.85,)).accuracy.value == 0.75


# ---------------------------------------------------------------------------- 3-tier by hand


def test_three_tier_points_by_hand():
    res = sweep(example3(), grid=[0.0, 0.5, NEVER_ACCEPT], holdout=0)
    assert len(res.points) == 9
    expected = {
        (0.5, 0.5): (1.0, 11 / 3, (1 / 3, 1 / 3, 1 / 3)),
        (0.5, NEVER_ACCEPT): (1.0, 5.0, (1 / 3, 0.0, 2 / 3)),
        (0.0, 0.5): (1 / 3, 1.0, (1.0, 0.0, 0.0)),
        (NEVER_ACCEPT, 0.0): (2 / 3, 3.0, (0.0, 1.0, 0.0)),
        (NEVER_ACCEPT, 0.5): (1.0, 13 / 3, (0.0, 2 / 3, 1 / 3)),
        (0.5, 0.0): (2 / 3, 7 / 3, (1 / 3, 2 / 3, 0.0)),
        (NEVER_ACCEPT, NEVER_ACCEPT): (1.0, 7.0, (0.0, 0.0, 1.0)),
    }
    for thr, (acc, cost, share) in expected.items():
        p = point_for(res, thr)
        assert p.accuracy.value == pytest.approx(acc), thr
        assert p.cost_per_task == pytest.approx(cost), thr
        assert p.tier_share == pytest.approx(share), thr


def test_three_tier_frontier_and_baselines():
    res = sweep(example3(), grid=[0.0, 0.5, NEVER_ACCEPT], holdout=0)
    front = [(res.points[i].cost_per_task, res.points[i].accuracy.value) for i in res.frontier]
    assert front == [
        pytest.approx((1.0, 1 / 3)),
        pytest.approx((7 / 3, 2 / 3)),
        pytest.approx((11 / 3, 1.0)),
    ]
    assert res.baselines["only:c"].accuracy.value == 1.0
    assert res.baselines["oracle"].cost_per_task == pytest.approx((1 + 2 + 4) / 3)
    assert set(res.calibration) == {"a", "b"}
    rec = res.recommendation
    assert rec.thresholds == (0.5, 0.5) and rec.best_single_tier == "c"


def test_fast_table_matches_direct_simulation():
    decisions = synthetic(300, seed=3, tiers=3, none_rate=0.1)
    # make some costs unknown on the last tier
    decisions[5] = make_decision(
        decisions[5].task.id,
        [
            {"name": "tier0", "score": 0.6, "ok": True, "cost": 1.0},
            {"name": "tier1", "score": None, "ok": False, "cost": 4.0},
            {"name": "tier2", "ok": True, "cost": None},
        ],
    )
    res = sweep(decisions, grid=[0.0, 0.3, 0.5, 0.62, 0.8, NEVER_ACCEPT], holdout=0)
    for p in res.points:
        q = simulate(decisions, p.thresholds)
        assert p.accuracy == q.accuracy
        assert p.tier_share == pytest.approx(q.tier_share)
        assert p.latency_per_task == pytest.approx(q.latency_per_task)
        assert (p.cost_per_task is None) == (q.cost_per_task is None)
        if p.cost_per_task is not None:
            assert p.cost_per_task == pytest.approx(q.cost_per_task)
        assert (p.skipped_error is None) == (q.skipped_error is None)
        if p.skipped_error is not None:
            assert p.skipped_error.value == pytest.approx(q.skipped_error.value)


def test_joint_grid_capped():
    res = sweep(synthetic(400, seed=1, tiers=4), holdout=0)
    assert len(res.points) <= sw.MAX_COMBINATIONS
    assert all(len(g) <= 27 for g in res.grids)
    assert len(res.points) == math.prod(len(g) for g in res.grids)
    for g in res.grids:
        assert g[0] == 0.0 and g[-1] == NEVER_ACCEPT


def test_default_grid_thinned_to_quantiles():
    res = sweep(synthetic(2000, seed=2), holdout=0)
    (g,) = res.grids
    assert len(g) <= 103 and g == tuple(sorted(set(g)))


# ---------------------------------------------------------------------------- truth & validation


def test_truth_auto_falls_back_to_audit_tier():
    ds = example2(with_reference=False)
    res = sweep(ds, holdout=0)
    assert res.truth == "audit-tier"
    assert any("agreement with the reference tier" in n for n in res.notes)
    assert res.baselines["only:big"].accuracy.value == 1.0  # last tier correct by definition
    # small agrees with big on t1, t3 (both right) and t4 (both wrong)
    assert res.baselines["only:small"].accuracy.value == 0.75
    with pytest.raises(ValueError, match="audit-tier"):
        sweep(ds, truth="reference")


def test_truth_audit_tier_can_be_forced():
    res = sweep(example2(), truth="audit-tier", holdout=0)
    assert res.truth == "audit-tier"
    assert res.baselines["only:big"].accuracy.value == 1.0


def test_undecided_counted_as_incorrect():
    ds = example2()
    ds[0] = two_tier(0.9, None, True, "t1")  # type: ignore[arg-type]
    res = sweep(ds, holdout=0)
    assert res.n_undecided == 1
    assert res.baselines["only:small"].accuracy.value == 0.25
    assert any("undecidable" in n for n in res.notes)


def test_rejects_serve_mode():
    with pytest.raises(ValueError, match="--mode eval"):
        sweep(example2(mode="serve"))


def test_rejects_mixed_runs():
    ds = example2()
    ds.append(two_tier(0.5, True, True, "t9", run_id="other"))
    with pytest.raises(ValueError, match="run"):
        sweep(ds)


def test_rejects_empty():
    with pytest.raises(ValueError):
        sweep([])


def test_skips_broken_records():
    ds = example2()
    ds.append(two_tier(0.5, True, True, "e1", error="final tier failed"))
    ds.append(
        make_decision(
            "e2", [{"name": "small", "err": "timeout"}, {"name": "big", "ok": True, "cost": 10.0}]
        )
    )
    ds.append(
        make_decision("e3", [{"name": "small", "score": 0.5, "ok": True, "cost": 1.0}])
    )  # missing the big tier
    res = sweep(ds, holdout=0)
    assert res.n == 4 and res.n_skipped_records == 3
    assert any("skipped" in n for n in res.notes)
    assert point_for(res, (0.9,)).cost_per_task == pytest.approx(9.0)


def test_all_records_broken():
    ds = [two_tier(0.5, True, True, "e1", error="boom")]
    with pytest.raises(InsufficientData):
        sweep(ds)


def test_none_costs():
    ds = example2()
    ds[3] = make_decision(
        "t4",
        [
            {"name": "small", "score": 0.3, "ok": False, "cost": 1.0, "conf_cost": 0.5},
            {"name": "big", "ok": False, "cost": None},
        ],
    )
    res = sweep(ds, holdout=0)
    assert point_for(res, (0.0,)).cost_per_task == pytest.approx(1.5)
    assert point_for(res, (0.3,)).cost_per_task == pytest.approx(1.5)
    assert point_for(res, (0.6,)).cost_per_task is None  # t4 walks to unpriced big
    assert res.baselines["only:big"].cost_per_task is None
    assert res.baselines["only:small"].cost_per_task == 1.0
    assert all(res.points[i].cost_per_task is not None for i in res.frontier)
    assert any("unknown" in n for n in res.notes)
    # max-savings needs a known cost reaching 0.75 accuracy: none exists
    assert sweep(ds, holdout=0, max_accuracy_drop=0.0).recommendation is None


def test_missing_scores_never_accept():
    ds = [two_tier(None, True, True, f"t{i}") for i in range(3)]
    res = sweep(ds, holdout=0)
    assert all(p.tier_share == (0.0, 1.0) for p in res.points)
    cal = res.calibration["small"]
    assert cal.n == 0 and cal.score_missing == 3 and cal.ece is None


def test_single_tier_run():
    ds = [make_decision(f"t{i}", [{"name": "only", "ok": i % 2 == 0}]) for i in range(4)]
    res = sweep(ds, holdout=0)
    assert res.tiers == ("only",) and len(res.points) == 1
    assert res.points[0].thresholds == () and res.points[0].accuracy.value == 0.5


# ---------------------------------------------------------------------------- holdout


def test_holdout_split_and_determinism():
    ds = synthetic(500, seed=7)
    a = sweep(ds, holdout=0.3, seed=11)
    b = sweep(list(reversed(ds)), holdout=0.3, seed=11)  # input order does not matter
    assert a.recommendation == b.recommendation
    rec = a.recommendation
    assert rec.n_holdout == 150 and rec.n_selection == 350
    assert rec.holdout is not None and rec.holdout.accuracy.n == 150
    assert rec.point.accuracy.n == 350
    assert rec.holdout.thresholds == rec.thresholds
    assert isinstance(rec.delta_vs_best, Estimate) and rec.delta_vs_best.n == 150
    assert rec.delta_vs_best.method == "paired-bootstrap"
    seeds = {sweep(ds, holdout=0.3, seed=s).recommendation.holdout.accuracy.value for s in range(6)}
    assert len(seeds) > 1


def test_points_use_all_data_regardless_of_holdout():
    ds = synthetic(200, seed=4)
    assert sweep(ds, holdout=0.5).points == sweep(ds, holdout=0).points


# ---------------------------------------------------------------------------- frontier helper


def _pt(acc_k: int, cost: float | None) -> OperatingPoint:
    return OperatingPoint(
        thresholds=(),
        accuracy=Estimate(acc_k / 10, 0, 1, 10, "wilson"),
        cost_per_task=cost,
        latency_per_task=0.0,
        escalation_rate=0.0,
        tier_share=(1.0,),
        skipped_error=None,
    )


def test_pareto_frontier_ties_and_dominance():
    pts = [
        _pt(5, 2.0),
        _pt(5, 2.0),
        _pt(7, 3.0),
        _pt(6, 3.0),
        _pt(7, 4.0),
        _pt(9, None),
        _pt(8, 1e9),
    ]
    assert pareto_frontier(pts) == [0, 2, 6]
    assert pareto_frontier([]) == []


def test_frontier_is_non_dominated_on_synthetic():
    res = sweep(synthetic(300, seed=5), holdout=0)
    known = [p for p in res.points if p.cost_per_task is not None]
    front = [res.points[i] for i in res.frontier]
    costs = [p.cost_per_task for p in front]
    assert costs == sorted(costs)
    for f in front:
        for q in known:
            dominates = (
                q.cost_per_task <= f.cost_per_task
                and q.accuracy.value >= f.accuracy.value
                and (q.cost_per_task < f.cost_per_task or q.accuracy.value > f.accuracy.value)
            )
            assert not dominates


# ---------------------------------------------------------------------------- calibration


def test_calibration_report():
    res = sweep(example2(), holdout=0)
    cal = res.calibration["small"]
    assert cal.tier == "small" and cal.n == 4 and cal.score_missing == 0
    # correct small answers have scores 0.9, 0.6; wrong ones 0.8, 0.3 -> AUROC 3/4
    assert cal.auroc == pytest.approx(0.75)
    assert cal.brier == pytest.approx(((0.1**2) + 0.8**2 + 0.4**2 + 0.3**2) / 4)
    assert len(cal.reliability) == 10
    assert cal.risk_coverage[-1][1] == 1.0
    assert cal.aurc is not None and cal.ece is not None


# ---------------------------------------------------------------------------- isotonic


def test_fit_isotonic_known_example():
    assert fit_isotonic([1, 2, 3, 4], [1, 0, 1, 1]) == [
        (1.0, 0.5),
        (2.0, 0.5),
        (3.0, 1.0),
        (4.0, 1.0),
    ]


def test_fit_isotonic_merges_duplicates():
    knots = fit_isotonic([0.5, 0.5, 0.5, 0.9], [True, False, False, True])
    assert knots == [(0.5, pytest.approx(1 / 3)), (0.9, 1.0)]


def test_fit_isotonic_properties_and_calibrated_compat():
    rng = random.Random(0)
    scores = [rng.random() for _ in range(500)]
    labels = [rng.random() < s**2 for s in scores]
    knots = fit_isotonic(scores, labels)
    xs = [x for x, _ in knots]
    ys = [y for _, y in knots]
    assert all(b > a for a, b in itertools.pairwise(xs))
    assert all(b >= a for a, b in itertools.pairwise(ys))
    assert all(0.0 <= y <= 1.0 for y in ys)

    class Base:
        name = "base"

    Calibrated(Base(), knots)  # type: ignore[arg-type]  # validates the knots


def test_fit_isotonic_reduces_brier_for_overconfident_score():
    rng = random.Random(1)

    def draw(n):
        s, y = [], []
        for _ in range(n):
            p = rng.uniform(0.2, 0.8)
            s.append(min(1.0, p + 0.2))  # overconfident by 0.2
            y.append(rng.random() < p)
        return s, y

    train_s, train_y = draw(3000)
    test_s, test_y = draw(3000)

    class Base:
        name = "base"

    cal = Calibrated(Base(), fit_isotonic(train_s, train_y))  # type: ignore[arg-type]
    raw = brier(test_s, test_y)
    fitted = brier([cal.apply(s) for s in test_s], test_y)
    assert fitted < raw - 0.02


def test_fit_isotonic_errors():
    with pytest.raises(InsufficientData):
        fit_isotonic([], [])
    with pytest.raises(ValueError):
        fit_isotonic([0.1], [True, False])
    with pytest.raises(ValueError):
        fit_isotonic([float("nan")], [True])
    with pytest.raises(ValueError):
        fit_isotonic([0.1], [2])


# ---------------------------------------------------------------------------- performance


def test_performance_20k_two_tier():
    ds = synthetic(20_000, seed=9)
    t0 = time.perf_counter()
    res = sweep(ds)
    elapsed = time.perf_counter() - t0
    assert res.n == 20_000 and len(res.points) <= 103
    assert res.recommendation is not None
    assert elapsed < 4.0, f"sweep took {elapsed:.2f}s"


def test_performance_three_tier_joint_grid():
    ds = synthetic(5_000, seed=10, tiers=3)
    t0 = time.perf_counter()
    res = sweep(ds)
    elapsed = time.perf_counter() - t0
    assert 5_000 < len(res.points) <= sw.MAX_COMBINATIONS
    assert elapsed < 10.0, f"sweep took {elapsed:.2f}s"
