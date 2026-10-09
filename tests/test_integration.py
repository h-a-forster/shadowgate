"""End-to-end, offline checks that the pieces add up to sound statistics.

Everything runs on simulated models over generated arithmetic tasks, so every number below is
deterministic. The simulated fast tier is deliberately overconfident (it reports ~0.15 more
confidence than its accuracy warrants) and its verbal confidence is only loosely informative,
which is the situation shadow audits exist to catch.

The large population runs (a, b, c, g) call ``Cascade.route`` directly to stay fast; the runner
and ledger are exercised end to end by the resume, deferred-audit and config tests.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import shadowgate as sg
from shadowgate import datasets, stats
from shadowgate.backends import SimulatedBackend
from shadowgate.compare import Numeric
from shadowgate.confidence import Calibrated, Combine, SelfConsistency, Verbal
from shadowgate.extract import FinalLine
from shadowgate.pricing import Pricing

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

N_POPULATION = 3000
FAST_SKILL = 7.0
THRESHOLD = 0.7
OVERCONFIDENCE = 0.15
FAST_PRICING = Pricing(input_per_mtok=0.5, output_per_mtok=2.5)
SLOW_PRICING = Pricing(input_per_mtok=5.0, output_per_mtok=25.0)
# Audit borderline acceptances heavily and confident ones lightly; the floor keeps every
# accepted case auditable, which is what makes the 1/pi-weighted estimate unbiased.
STRATIFIED = sg.AuditPolicy(
    rate=0.1, strata=((0.7, 0.85, 0.5), (0.85, 1.0, 0.1)), floor=0.02, seed=1
)


# --------------------------------------------------------------------------- builders


def make_cascade(
    tasks: Sequence[sg.Task],
    *,
    audit: sg.AuditPolicy | None = STRATIFIED,
    estimator: sg.ConfidenceEstimator | None = None,
    threshold: float = THRESHOLD,
) -> sg.Cascade:
    fast = SimulatedBackend(
        "fast",
        skill=FAST_SKILL,
        tasks=tasks,
        overconfidence=OVERCONFIDENCE,
        confidence_noise=0.1,
        pricing=FAST_PRICING,
    )
    slow = SimulatedBackend("slow", skill=16.0, tasks=tasks, pricing=SLOW_PRICING)
    return sg.Cascade(
        [
            sg.Tier("fast", fast, threshold=threshold, estimator=estimator or Verbal()),
            sg.Tier("slow", slow),
        ],
        extractor=FinalLine(),
        comparator=Numeric(),
        audit=audit,
    )


def route_all(
    cascade: sg.Cascade, tasks: Sequence[sg.Task], *, run_id: str, mode: str
) -> list[sg.Decision]:
    return [cascade.route(t, run_id=run_id, mode=mode) for t in tasks]


def fast_scores_and_labels(decisions: Sequence[sg.Decision]) -> tuple[list[float], list[bool]]:
    scores: list[float] = []
    labels: list[bool] = []
    for d in decisions:
        a = d.attempts[0]
        if a.confidence is None or a.confidence.score is None or a.correct is None:
            continue
        if a.correct.equivalent is None:
            continue
        scores.append(a.confidence.score)
        labels.append(bool(a.correct.equivalent))
    return scores, labels


def comparable(d: sg.Decision) -> dict[str, Any]:
    """A decision as a dict without the wall-clock timestamp."""
    out = d.to_dict()
    out.pop("created_at")
    return out


# --------------------------------------------------------------------------- shared runs


@pytest.fixture(scope="module")
def population() -> list[sg.Task]:
    return datasets.arithmetic(N_POPULATION, seed=11)


@pytest.fixture(scope="module")
def eval_run(population: list[sg.Task]) -> list[sg.Decision]:
    return route_all(make_cascade(population), population, run_id="eval", mode="eval")


@pytest.fixture(scope="module")
def serve_run(population: list[sg.Task]) -> list[sg.Decision]:
    return route_all(make_cascade(population), population, run_id="serve", mode="serve")


# --------------------------------------------------------------------------- a. unbiasedness


def test_shadow_audit_is_unbiased_under_stratified_sampling(
    eval_run: list[sg.Decision], serve_run: list[sg.Decision]
) -> None:
    truth_summary = sg.summarize(eval_run)
    served = sg.summarize(serve_run)
    assert truth_summary.disagreement is not None and served.disagreement is not None
    assert served.disagreement_unweighted is not None
    truth = truth_summary.disagreement.value
    weighted = served.disagreement
    unweighted = served.disagreement_unweighted.value
    assert truth is not None and weighted.value is not None and unweighted is not None

    msg = (
        f"truth={truth:.4f} (all {truth_summary.n_skipped} skipped cases, eval mode); "
        f"weighted={weighted.value:.4f} [{weighted.lo:.4f}, {weighted.hi:.4f}] "
        f"n_audited={served.n_audited} n_eff={weighted.n_eff:.1f}; unweighted={unweighted:.4f}"
    )
    print("\n[a]", msg)
    # Same tasks, same deterministic models: serve mode skips exactly the eval-mode skipped set.
    assert served.n_skipped == truth_summary.n_skipped, msg
    assert 0.0 < served.n_audited < served.n_skipped, msg
    assert weighted.lo is not None and weighted.hi is not None
    assert weighted.lo <= truth <= weighted.hi, msg
    # Oversampling the borderline band inflates the naive share; weighting removes that bias.
    assert abs(unweighted - truth) > 2 * abs(weighted.value - truth), msg
    assert unweighted - truth > 0.03, msg

    # Estimated savings vs always using the slow tier, against the eval-mode truth.
    slow_costs = [d.attempts[-1].completion.cost_usd for d in eval_run]  # type: ignore[union-attr]
    true_savings = 1.0 - (served.cost_per_task or 0.0) / (sum(slow_costs) / len(slow_costs))
    est = served.est_savings
    assert est is not None and est.lo is not None and est.hi is not None
    print(f"[a] savings true={true_savings:.4f} est={est.value:.4f} [{est.lo:.4f}, {est.hi:.4f}]")
    assert est.lo <= true_savings <= est.hi


# --------------------------------------------------------------------------- b. overconfidence


def test_audit_detects_overconfident_fast_tier() -> None:
    # Mid-difficulty problems keep the fast tier's confidence away from the 0.99 clip, where
    # overconfidence would be invisible; a uniform 50% audit gives a tight interval.
    tasks = datasets.arithmetic(2000, seed=21, min_steps=3, max_steps=4)
    policy = sg.AuditPolicy(rate=0.5, floor=0.5, seed=1)
    serve_run = route_all(make_cascade(tasks, audit=policy), tasks, run_id="b", mode="serve")
    summary = sg.summarize(serve_run)
    skipped = [
        d for d in serve_run if d.error is None and not d.escalated and d.final_tier == "fast"
    ]
    confs = [d.attempts[0].confidence.score for d in skipped]  # type: ignore[union-attr]
    assert all(c is not None for c in confs)
    implied_error = 1.0 - sum(confs) / len(confs)  # type: ignore[arg-type]
    audited = summary.disagreement
    assert audited is not None and audited.value is not None and audited.lo is not None
    msg = (
        f"audited skipped-case disagreement={audited.value:.4f} "
        f"[{audited.lo:.4f}, {audited.hi:.4f}] vs error implied by confidence "
        f"1-mean(conf)={implied_error:.4f} over {len(skipped)} skipped cases"
    )
    if summary.skipped_error is not None and summary.skipped_error.value is not None:
        msg += f"; graded skipped error={summary.skipped_error.value:.4f}"
    print("\n[b]", msg)
    # The audit's lower confidence bound already exceeds what the confidence signal promises.
    assert audited.lo > implied_error, msg
    assert summary.skipped_error is not None and summary.skipped_error.lo is not None
    assert summary.skipped_error.lo > implied_error, msg


# --------------------------------------------------------------------------- c. sweep


def test_sweep_recommendation_frontier_and_oracle(eval_run: list[sg.Decision]) -> None:
    drop = 0.02
    result = sg.run_sweep(eval_run, objective="max-savings", max_accuracy_drop=drop, seed=0)
    assert result.truth == "reference"
    rec = result.recommendation
    assert rec is not None and rec.holdout is not None, result.notes

    only_fast = result.baselines["only:fast"]
    only_slow = result.baselines["only:slow"]
    oracle = result.baselines["oracle"]
    best_acc = max(only_fast.accuracy.value or 0.0, only_slow.accuracy.value or 0.0)
    hold = rec.holdout
    assert hold.accuracy.value is not None and hold.cost_per_task is not None
    assert only_slow.cost_per_task is not None
    msg = (
        f"thresholds={rec.thresholds} holdout acc={hold.accuracy.value:.4f} "
        f"cost/task={hold.cost_per_task:.6f}; best single tier acc={best_acc:.4f}; "
        f"only:slow cost/task={only_slow.cost_per_task:.6f}; oracle acc="
        f"{oracle.accuracy.value:.4f}; points={len(result.points)} frontier={len(result.frontier)}"
    )
    print("\n[c]", msg)
    # Holdout accuracy within the allowed drop, with sampling tolerance for a 30% split.
    assert hold.accuracy.value >= best_acc - drop - 0.02, msg
    assert hold.cost_per_task < only_slow.cost_per_task, msg

    # Pareto frontier: sorted by cost, and accuracy strictly increases with cost.
    front = [result.points[i] for i in result.frontier]
    assert len(front) >= 2, msg
    for a, b in zip(front, front[1:], strict=False):
        assert a.cost_per_task is not None and b.cost_per_task is not None
        assert a.cost_per_task <= b.cost_per_task
        assert (a.accuracy.value or 0.0) < (b.accuracy.value or 0.0)
    # No point beats the frontier on both axes.
    for p in result.points:
        if p.cost_per_task is None or p.accuracy.value is None:
            continue
        for f in front:
            assert not (
                p.cost_per_task < f.cost_per_task  # type: ignore[operator]
                and p.accuracy.value > (f.accuracy.value or 0.0)
            )
    # The oracle (cheapest correct tier per task) bounds every router's accuracy.
    assert oracle.accuracy.value is not None
    for p in [*result.points, *result.baselines.values()]:
        assert (p.accuracy.value or 0.0) <= oracle.accuracy.value + 1e-12


# --------------------------------------------------------------------------- d. resume


def test_budget_stop_then_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    tasks = datasets.arithmetic(60, seed=5)
    cascade = make_cascade(tasks)

    with sg.Ledger(tmp_path / "full.sqlite") as full_ledger:
        full = sg.run_tasks(cascade, tasks, full_ledger, run_id="r", workers=4)
        assert full.stopped is None and full.completed == len(tasks)
        reference = {d.task.id: comparable(d) for d in full_ledger.decisions("r")}

    with sg.Ledger(tmp_path / "resumed.sqlite") as ledger:
        cap = (full.cost_serving + full.cost_audit) * 0.4
        first = sg.run_tasks(cascade, tasks, ledger, run_id="r", workers=4, max_cost_usd=cap)
        assert first.stopped == "budget", first.summary_line()
        assert 0 < first.completed < len(tasks), first.summary_line()
        assert len(ledger.done_task_ids("r")) == first.completed

        second = sg.run_tasks(cascade, tasks, ledger, run_id="r", workers=4)
        assert second.stopped is None
        assert second.skipped_existing == first.completed
        assert second.completed == len(tasks) - first.completed
        print(f"\n[d] first={first.summary_line()} | second={second.summary_line()}")

        records = list(ledger.decisions("r"))
        ids = [d.task.id for d in records]
        assert sorted(ids) == sorted(t.id for t in tasks)
        assert len(ids) == len(set(ids))
        resumed = {d.task.id: comparable(d) for d in records}
    assert resumed == reference


# --------------------------------------------------------------------------- e. deferred audits


def test_deferred_audits_match_inline_audits(tmp_path: Path) -> None:
    tasks = datasets.arithmetic(120, seed=6)
    inline_policy = dataclasses.replace(STRATIFIED, rate=0.3, floor=0.3, mode="inline")
    deferred_policy = dataclasses.replace(inline_policy, mode="deferred")

    with sg.Ledger(tmp_path / "inline.sqlite") as led:
        sg.run_tasks(make_cascade(tasks, audit=inline_policy), tasks, led, run_id="r")
        inline = sg.summarize(led.decisions("r"))
        inline_records = {d.task.id: comparable(d) for d in led.decisions("r")}

    with sg.Ledger(tmp_path / "deferred.sqlite") as led:
        cascade = make_cascade(tasks, audit=deferred_policy)
        sg.run_tasks(cascade, tasks, led, run_id="r")
        before = sg.summarize(led.decisions("r"))
        n_pending = len(list(led.pending_audits("r")))
        assert n_pending == before.n_pending == inline.n_audited > 0
        assert before.n_audited == 0

        stats_ = sg.run_pending_audits(cascade, led, run_id="r")
        assert stats_.completed == n_pending and stats_.failed == 0
        assert list(led.pending_audits("r")) == []
        after = sg.summarize(led.decisions("r"))
        deferred_records = {d.task.id: comparable(d) for d in led.decisions("r")}

    print(
        f"\n[e] pending={n_pending}; inline disagreement={inline.disagreement}; "
        f"deferred disagreement={after.disagreement}"
    )
    assert after.n_pending == 0
    assert after.n_audited == inline.n_audited
    assert after.n_skipped == inline.n_skipped
    assert after.disagreement == inline.disagreement
    assert after.disagreement_unweighted == inline.disagreement_unweighted
    assert after.cost_audit == pytest.approx(inline.cost_audit)
    assert after.status == inline.status
    # Record for record, a completed deferred audit is indistinguishable from an inline one.
    assert deferred_records == inline_records


# --------------------------------------------------------------------------- f. config path


def test_config_driven_run(tmp_path: Path) -> None:
    config = sg.load_config(EXAMPLES / "simulated.toml")
    tasks = sg.load_tasks(EXAMPLES / "tasks" / "arithmetic-50.jsonl")
    assert len(tasks) == 50
    cascade, settings = config.build(tasks=tasks)
    assert [t.name for t in cascade.tiers] == ["fast", "slow"]

    with sg.Ledger(tmp_path / "ledger.sqlite") as led:
        run_stats = sg.run_tasks(
            cascade,
            tasks,
            led,
            run_id="cfg",
            workers=settings.workers,
            max_cost_usd=settings.max_cost_usd,
            config_snapshot=config.snapshot(),
        )
        summary = sg.summarize(led.decisions("cfg"), tolerance=settings.tolerance)
        runs = led.runs()

    print(f"\n[f] {run_stats.summary_line()}; status={summary.status}")
    assert run_stats.stopped is None and run_stats.failed == 0
    assert summary.n_decisions == 50 and summary.n_errors == 0
    assert summary.n_skipped > 0 and summary.n_audited > 0
    assert summary.served_accuracy is not None
    assert summary.status in {"ok", "breach", "inconclusive"}
    assert [r.run_id for r in runs] == ["cfg"]


# --------------------------------------------------------------------------- g. calibration


def test_isotonic_calibration_reduces_ece(eval_run: list[sg.Decision]) -> None:
    fit_scores, fit_labels = fast_scores_and_labels(eval_run)
    points = sg.fit_isotonic(fit_scores, fit_labels)

    fresh = datasets.arithmetic(1500, seed=12)
    raw_run = route_all(make_cascade(fresh, audit=None), fresh, run_id="raw", mode="serve")
    cal_est = Calibrated(Verbal(), points)
    cal_run = route_all(
        make_cascade(fresh, audit=None, estimator=cal_est), fresh, run_id="cal", mode="serve"
    )
    raw_s, raw_y = fast_scores_and_labels(raw_run)
    cal_s, cal_y = fast_scores_and_labels(cal_run)
    assert raw_y == cal_y  # same fast-tier answers; only the scores differ
    raw_ece, cal_ece = stats.ece(raw_s, raw_y), stats.ece(cal_s, cal_y)
    print(f"\n[g] fresh-task ECE raw={raw_ece:.4f} calibrated={cal_ece:.4f} knots={len(points)}")
    assert cal_ece < raw_ece / 2, (raw_ece, cal_ece)


# --------------------------------------------------------------------------- combined signals


def test_combined_confidence_ranks_errors_better_than_verbal_alone() -> None:
    tasks = datasets.arithmetic(300, seed=13, min_steps=3, max_steps=4)
    verbal_run = route_all(make_cascade(tasks, audit=None), tasks, run_id="v", mode="serve")
    combo = Combine(
        [Verbal(), SelfConsistency(comparator=Numeric(), extractor=FinalLine(), samples=4)],
        method="mean",
    )
    combo_run = route_all(
        make_cascade(tasks, audit=None, estimator=combo), tasks, run_id="c", mode="serve"
    )
    v_auc = stats.auroc(*fast_scores_and_labels(verbal_run))
    c_auc = stats.auroc(*fast_scores_and_labels(combo_run))
    assert v_auc is not None and c_auc is not None
    # Resamples are charged to serving cost.
    v_cost = sum(d.attempts[0].completion.cost_usd for d in verbal_run)  # type: ignore[misc,union-attr]
    c_conf_cost = sum(
        c.cost_usd or 0.0
        for d in combo_run
        for c in (d.attempts[0].confidence.calls if d.attempts[0].confidence else ())
    )
    print(f"\n[combine] AUROC verbal={v_auc:.4f} verbal+self-consistency={c_auc:.4f}")
    assert c_auc > v_auc
    assert c_conf_cost == pytest.approx(4 * v_cost, rel=0.05)
