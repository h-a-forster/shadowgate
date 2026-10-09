"""Tests for shadowgate.audit (decisions built directly from types, plus a cascade end-to-end)."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from shadowgate.audit import AuditSummary, infer_tier_order, summarize
from shadowgate.cascade import AuditPolicy, Cascade, Tier
from shadowgate.stats import clopper_pearson, weighted_proportion, wilson
from shadowgate.types import (
    Attempt,
    Completion,
    ConfidenceResult,
    Decision,
    Judgement,
    Request,
    ShadowResult,
    Task,
)

# --------------------------------------------------------------------------- factories

_UNSET = object()


def judge(eq: bool | None) -> Judgement:
    return Judgement(equivalent=eq, comparator="fake")


def att(
    tier: str,
    *,
    score: float | None = None,
    threshold: float | None = None,
    accepted: bool = True,
    answer: str = "a",
    error: str | None = None,
    correct: bool | None | object = _UNSET,
    agreement: bool | None | object = _UNSET,
    cost: float | None = 0.01,
) -> Attempt:
    completion = None if error else Completion(text=answer, model=tier, cost_usd=cost)
    conf = None if threshold is None else ConfidenceResult("fake", score)
    return Attempt(
        tier=tier,
        backend=f"fake:{tier}",
        completion=completion,
        answer="" if error else answer,
        confidence=conf,
        threshold=threshold,
        accepted=accepted,
        error=error,
        correct=None if correct is _UNSET else judge(correct),  # type: ignore[arg-type]
        agreement=None if agreement is _UNSET else judge(agreement),  # type: ignore[arg-type]
    )


def task(i: int | str, reference: str | None = None) -> Task:
    return Task(id=str(i), prompt=f"q{i}", reference=reference)


def skipped(
    i: int | str,
    *,
    pi: float | None = 0.5,
    status: str = "done",
    equivalent: bool | None = True,
    score: float | None = 0.9,
    correct: bool | None | object = _UNSET,
    audit_correct: bool | None | object = _UNSET,
    run_id: str = "r",
    cost: float | None = 0.01,
    audit_cost: float | None = 0.1,
    shadow: bool = True,
) -> Decision:
    """Serve-mode decision accepted at tier 'fast' (non-final) with an optional shadow."""
    fast = att("fast", score=score, threshold=0.5, accepted=True, correct=correct)
    sh = None
    if shadow and pi is not None:
        if status == "done":
            sh = ShadowResult(
                "slow",
                pi,
                "done",
                attempt=att("slow", cost=audit_cost, correct=audit_correct),
                agreement=judge(equivalent),
            )
        elif status == "error":
            sh = ShadowResult("slow", pi, "error", attempt=att("slow", error="boom"))
        else:
            sh = ShadowResult("slow", pi, status)
    graded = fast.correct
    return Decision(
        run_id=run_id,
        task=task(i),
        answer="a",
        final_tier="fast",
        escalated=False,
        attempts=(fast,),
        cost_usd=cost,
        audit_cost_usd=audit_cost if sh is not None and status == "done" else 0.0,
        latency_s=0.1,
        shadow=sh,
        correct=graded,
    )


def escalated(
    i: int | str,
    *,
    correct: bool | None | object = _UNSET,
    run_id: str = "r",
    cost: float | None = 0.11,
    slow_cost: float | None = 0.1,
    error: bool = False,
) -> Decision:
    fast = att("fast", score=0.2, threshold=0.5, accepted=False, cost=0.01)
    slow = (
        att("slow", error="down")
        if error
        else att("slow", correct=correct, cost=slow_cost, accepted=True)
    )
    return Decision(
        run_id=run_id,
        task=task(i),
        answer="" if error else "a",
        final_tier="slow",
        escalated=True,
        attempts=(fast, slow),
        cost_usd=cost,
        audit_cost_usd=0.0,
        latency_s=0.2,
        correct=None if error else slow.correct,
        error="final tier 'slow' failed: down" if error else None,
    )


def eval_decision(
    i: int | str,
    *,
    score: float = 0.9,
    agree: bool | None = True,
    fast_correct: bool | None | object = _UNSET,
    slow_correct: bool | None | object = _UNSET,
    slow_error: bool = False,
    run_id: str = "e",
    slow_cost: float | None = 0.1,
) -> Decision:
    accepted = score >= 0.5
    fast = att(
        "fast",
        score=score,
        threshold=0.5,
        accepted=accepted,
        correct=fast_correct,
        agreement=_UNSET if slow_error else agree,
    )
    slow = (
        att("slow", error="down")
        if slow_error
        else att("slow", correct=slow_correct, cost=slow_cost)
    )
    served = fast if accepted else slow
    return Decision(
        run_id=run_id,
        task=task(i),
        answer=served.answer,
        final_tier=served.tier,
        escalated=not accepted,
        attempts=(fast, slow),
        cost_usd=0.01 if accepted else 0.11,
        audit_cost_usd=0.0 if slow_error else (slow_cost if accepted else 0.0),
        latency_s=0.1,
        mode="eval",
        correct=served.correct,
        error="final tier 'slow' failed: down" if slow_error and not accepted else None,
    )


def mixed_serve(n_agree: int, n_disagree: int, pi: float = 0.5, start: int = 0) -> list[Decision]:
    out = [skipped(start + k, pi=pi, equivalent=True) for k in range(n_agree)]
    out += [skipped(start + n_agree + k, pi=pi, equivalent=False) for k in range(n_disagree)]
    return out


# --------------------------------------------------------------------------- basics


def test_empty_input_does_not_crash() -> None:
    s = summarize([])
    assert isinstance(s, AuditSummary)
    assert s.n_decisions == 0 and s.n_skipped == 0 and s.run_id == "" and s.mode == "n/a"
    assert s.status == "no-data" and s.disagreement is None
    assert s.escalation_rate.value is None
    assert len(s.bins) == 6 and all(b.n_accepted == 0 for b in s.bins)
    assert s.cost_serving is None and s.cost_per_task is None and s.est_savings is None
    assert s.expected_wrong_skipped is None


def test_empty_with_tolerance_is_no_data() -> None:
    assert summarize([], tolerance=0.05).status == "no-data"


def test_mixed_run_ids_raise() -> None:
    with pytest.raises(ValueError, match="runs"):
        summarize([skipped(1, run_id="a"), skipped(2, run_id="b")])


def test_invalid_bins_and_tolerance_raise() -> None:
    with pytest.raises(ValueError):
        summarize([], bins=(0.5,))
    with pytest.raises(ValueError):
        summarize([], bins=(0.0, 0.5, 0.5, 1.0))
    with pytest.raises(ValueError):
        summarize([], tolerance=1.5)
    with pytest.raises(ValueError):
        summarize([], tolerance=-0.1)


def test_generator_input_accepted() -> None:
    s = summarize(d for d in mixed_serve(3, 1))
    assert s.n_decisions == 4 and s.n_audited == 4


# --------------------------------------------------------------------------- tier order


def test_infer_tier_order_from_partial_serve_sequences() -> None:
    ds = [skipped(1), escalated(2)]
    tiers, final = infer_tier_order(ds)
    assert tiers == ("fast", "slow") and final == "slow"


def test_infer_tier_order_three_tiers_prefixes() -> None:
    a = att("a", score=0.9, threshold=0.5)
    b_rej = att("a", score=0.1, threshold=0.5, accepted=False)
    b = att("b", score=0.9, threshold=0.5)
    c = att("c")
    mk = lambda i, attempts, final: Decision(  # noqa: E731
        "r", task(i), "x", final, len(attempts) > 1, attempts, 0.0, 0.0, 0.0
    )
    ds = [
        mk(1, (a,), "a"),
        mk(2, (b_rej, b), "b"),
        mk(3, (b_rej, att("b", score=0.1, threshold=0.5, accepted=False), c), "c"),
    ]
    tiers, final = infer_tier_order(ds)
    assert tiers == ("a", "b", "c") and final == "c"
    s = summarize(ds)
    assert s.n_skipped == 2 and s.tier_share == {"a": 1, "b": 1, "c": 1}


def test_final_tier_inferred_from_audit_tier_when_never_served() -> None:
    ds = mixed_serve(4, 1)
    tiers, final = infer_tier_order(ds)
    assert tiers == ("fast", "slow") and final == "slow"
    s = summarize(ds)
    assert s.n_skipped == 5 and s.final_tier == "slow" and s.reference_tier == "slow"


def test_final_tier_unknown_treats_all_as_skipped() -> None:
    ds = [skipped(i, shadow=False) for i in range(3)]
    s = summarize(ds)
    assert s.final_tier is None and s.n_skipped == 3
    assert any("never observed" in n for n in s.notes)


# --------------------------------------------------------------------------- serve mode


def test_serve_uniform_pi_matches_clopper_pearson() -> None:
    ds = mixed_serve(17, 3, pi=0.2)
    s = summarize(ds)
    ref = clopper_pearson(3, 20)
    assert s.mode == "serve" and s.n_skipped == 20 and s.n_audited == 20
    assert s.disagreement is not None
    assert s.disagreement.value == pytest.approx(0.15)
    assert s.disagreement.lo == pytest.approx(ref.lo) and s.disagreement.hi == pytest.approx(ref.hi)
    assert s.disagreement_unweighted is not None
    assert s.disagreement_unweighted.value == pytest.approx(0.15)


def test_serve_weighted_disagreement_by_hand() -> None:
    # Stratum A: pi=0.8, 8 audited, 4 disagree. Stratum B: pi=0.1, 10 audited, 1 disagrees.
    ds = [skipped(f"a{k}", pi=0.8, equivalent=k >= 4) for k in range(8)]
    ds += [skipped(f"b{k}", pi=0.1, equivalent=k >= 1) for k in range(10)]
    s = summarize(ds)
    expected = (4 * 1.25 + 1 * 10) / (8 * 1.25 + 10 * 10)
    assert s.disagreement is not None
    assert s.disagreement.value == pytest.approx(expected)
    ys = [k < 4 for k in range(8)] + [k < 1 for k in range(10)]
    ref = weighted_proportion(ys, [1.25] * 8 + [10.0] * 10)
    assert s.disagreement == ref
    assert s.disagreement_unweighted is not None
    assert s.disagreement_unweighted.value == pytest.approx(5 / 18)


def test_not_selected_cases_count_but_are_not_audited() -> None:
    ds = mixed_serve(5, 1) + [skipped(100 + k, status="skipped") for k in range(10)]
    s = summarize(ds)
    assert s.n_skipped == 16 and s.n_audited == 6 and s.n_not_selected == 10
    assert s.expected_wrong_skipped is not None
    assert s.expected_wrong_skipped[0] == pytest.approx(16 / 6)


def test_pending_errors_undecided_counted_and_excluded() -> None:
    ds = mixed_serve(6, 2)
    ds += [skipped(100, status="pending"), skipped(101, status="pending")]
    ds += [skipped(102, status="error")]
    ds += [skipped(103, equivalent=None)]
    s = summarize(ds)
    assert (s.n_pending, s.n_audit_errors, s.n_undecided, s.n_audited) == (2, 1, 1, 8)
    assert s.disagreement is not None and s.disagreement.value == pytest.approx(0.25)
    text = " ".join(s.notes)
    assert "pending" in text and "errored" in text and "undecided" in text


def test_all_pending_gives_no_data() -> None:
    ds = [skipped(k, status="pending") for k in range(5)]
    s = summarize(ds, tolerance=0.1)
    assert s.n_pending == 5 and s.n_audited == 0
    assert s.disagreement is None and s.status == "no-data" and s.audits_to_resolve is None
    assert any("No completed audits" in n for n in s.notes)


def test_no_audit_records_audit_off() -> None:
    ds = [skipped(k, shadow=False) for k in range(4)] + [escalated(10)]
    s = summarize(ds, tolerance=0.1)
    assert s.n_skipped == 4 and s.n_no_audit_record == 4
    assert s.disagreement is None and s.status == "no-data"
    assert any("no audit record" in n for n in s.notes)
    assert s.est_all_slow_cost_per_task is None


def test_all_escalated_has_no_skipped_cases() -> None:
    ds = [escalated(k) for k in range(5)]
    s = summarize(ds, tolerance=0.05)
    assert s.n_skipped == 0 and s.status == "no-data" and s.disagreement is None
    assert s.escalation_rate.value == 1.0
    assert s.tier_share == {"slow": 5}
    assert any("No skipped cases" in n for n in s.notes)
    # all-slow cost comes from the final-tier attempts themselves
    assert s.est_all_slow_cost_per_task is not None
    assert s.est_all_slow_cost_per_task.value == pytest.approx(0.1)


def test_decisions_with_errors() -> None:
    ds = mixed_serve(3, 1) + [escalated(50, error=True), escalated(51)]
    s = summarize(ds)
    assert s.n_decisions == 6 and s.n_errors == 1
    assert s.tier_share == {"fast": 4, "slow": 1}
    assert s.n_skipped == 4
    assert s.escalation_rate.value == pytest.approx(2 / 6)
    assert any("failed" in n for n in s.notes)


def test_escalation_rate_and_tier_share() -> None:
    ds = mixed_serve(6, 0) + [escalated(100 + k) for k in range(4)]
    s = summarize(ds)
    assert s.escalation_rate == wilson(4, 10)
    assert s.tier_share == {"fast": 6, "slow": 4}


def test_invalid_inclusion_prob_excluded() -> None:
    ds = mixed_serve(3, 1) + [skipped(99, pi=0.0)]
    s = summarize(ds)
    assert s.n_audited == 4
    assert any("invalid inclusion probability" in n for n in s.notes)


# --------------------------------------------------------------------------- eval mode


def test_eval_mode_pi_one() -> None:
    ds = [eval_decision(k, agree=k >= 3) for k in range(12)]
    ds += [eval_decision(100 + k, score=0.2) for k in range(4)]
    s = summarize(ds)
    assert s.mode == "eval" and s.n_skipped == 12 and s.n_audited == 12
    assert s.disagreement == weighted_proportion([k < 3 for k in range(12)], [1.0] * 12)
    assert s.disagreement is not None and s.disagreement.value == pytest.approx(0.25)
    assert s.reference_tier == "slow"
    assert s.tier_share == {"fast": 12, "slow": 4}


def test_eval_mode_failed_reference_counts_as_audit_error() -> None:
    ds = [eval_decision(k) for k in range(5)] + [eval_decision(9, slow_error=True)]
    s = summarize(ds)
    assert s.n_errors == 0 and s.n_skipped == 6
    assert s.n_audit_errors == 1 and s.n_audited == 5


def test_eval_mode_undecided() -> None:
    ds = [eval_decision(k) for k in range(5)] + [eval_decision(9, agree=None)]
    s = summarize(ds)
    assert s.n_undecided == 1 and s.n_audited == 5


def test_eval_all_slow_cost_and_savings() -> None:
    ds = [eval_decision(k, slow_cost=0.1) for k in range(6)]
    ds += [eval_decision(10 + k, score=0.1, slow_cost=0.1) for k in range(4)]
    s = summarize(ds)
    assert s.est_all_slow_cost_per_task is not None
    assert s.est_all_slow_cost_per_task.value == pytest.approx(0.1)
    assert s.cost_per_task == pytest.approx((6 * 0.01 + 4 * 0.11) / 10)
    assert s.est_savings is not None
    assert s.est_savings.value == pytest.approx(1 - 0.05 / 0.1)


# --------------------------------------------------------------------------- references


def test_skipped_error_drives_status_when_all_graded() -> None:
    ds = [skipped(k, correct=k >= 2, equivalent=True) for k in range(20)]
    s = summarize(ds, tolerance=0.5)
    assert s.skipped_error == wilson(2, 20)
    assert s.n_graded_skipped == 20
    assert s.status_metric == "skipped_error" and s.status == "ok"
    assert s.expected_wrong_source == "skipped_error"
    assert s.expected_wrong_skipped is not None
    assert s.expected_wrong_skipped[0] == pytest.approx(2.0)
    assert not any("proxy" in n for n in s.notes)


def test_audit_only_status_when_references_drive_status() -> None:
    # 20 graded skipped cases, none wrong (status ok on references); 4 of 20 audits disagree
    # with the audit tier, so the audit alone gives a different status.
    ds = [skipped(k, correct=True, equivalent=k >= 4) for k in range(20)]
    s = summarize(ds, tolerance=0.2)
    assert s.status_metric == "skipped_error" and s.status == "ok"
    assert s.audit_only_status == "inconclusive"
    assert summarize(ds).audit_only_status is None  # no tolerance


def test_audit_only_status_unset_when_disagreement_drives_status() -> None:
    s = summarize(mixed_serve(17, 3), tolerance=0.1)
    assert s.status_metric == "disagreement" and s.audit_only_status is None


def test_partial_references_fall_back_to_disagreement() -> None:
    ds = [skipped(k, correct=True) for k in range(5)]
    ds += [skipped(10 + k, equivalent=k > 0) for k in range(5)]
    s = summarize(ds, tolerance=0.5)
    assert s.n_graded_skipped == 5
    assert s.status_metric == "disagreement"
    assert any("have references" in n for n in s.notes)


def test_served_accuracy_and_audit_tier_error() -> None:
    ds = [skipped(k, correct=k > 0, audit_correct=k > 1, pi=0.5) for k in range(10)]
    ds += [escalated(100 + k, correct=k > 0) for k in range(5)]
    s = summarize(ds)
    assert s.served_accuracy == wilson(9 + 4, 15)
    assert s.audit_tier_error is not None
    assert s.audit_tier_error.value == pytest.approx(0.2)


def test_proxy_note_without_references() -> None:
    s = summarize(mixed_serve(5, 1))
    assert s.skipped_error is None and s.served_accuracy is None
    assert any("proxy" in n for n in s.notes)
    assert s.expected_wrong_source == "disagreement"


# --------------------------------------------------------------------------- status


def test_status_breach_ok_inconclusive_na() -> None:
    breach = summarize(mixed_serve(50, 50), tolerance=0.1)
    assert breach.status == "breach" and breach.audits_to_resolve is None
    ok = summarize(mixed_serve(200, 0), tolerance=0.05)
    assert ok.status == "ok"
    inc = summarize(mixed_serve(17, 3), tolerance=0.1)
    assert inc.status == "inconclusive"
    assert inc.audits_to_resolve is not None and inc.audits_to_resolve > 0
    na = summarize(mixed_serve(18, 2))
    assert na.status == "n/a" and na.audits_to_resolve is None
    assert na.tolerance is None


def test_audits_to_resolve_is_roughly_enough() -> None:
    base = summarize(mixed_serve(16, 4), tolerance=0.1)  # p = 0.2
    assert base.status == "inconclusive"
    extra = base.audits_to_resolve
    assert extra is not None
    n_total = 20 + extra
    k = round(0.2 * n_total)
    more = summarize(mixed_serve(n_total - k, k), tolerance=0.1)
    assert more.status == "breach"
    fewer_total = max(21, (20 + n_total) // 2)
    kf = round(0.2 * fewer_total)
    assert summarize(mixed_serve(fewer_total - kf, kf), tolerance=0.1).status == "inconclusive"


def test_audits_to_resolve_zero_disagreement() -> None:
    s = summarize(mixed_serve(10, 0), tolerance=0.1)
    assert s.status == "inconclusive"
    extra = s.audits_to_resolve
    assert extra is not None and extra > 0
    assert summarize(mixed_serve(10 + extra, 0), tolerance=0.1).status == "ok"


def test_audits_to_resolve_accounts_for_design_effect() -> None:
    eq = summarize(mixed_serve(16, 4, pi=0.5), tolerance=0.1)  # p = 0.2, n_eff = 20
    ds = [skipped(f"a{k}", pi=0.9, equivalent=k >= 3) for k in range(10)]
    ds += [skipped(f"b{k}", pi=0.05, equivalent=k >= 2) for k in range(10)]
    uneq = summarize(ds, tolerance=0.1)  # p ~ 0.21, n_eff ~ 11
    assert eq.status == uneq.status == "inconclusive"
    assert uneq.disagreement is not None and uneq.disagreement.n_eff is not None
    assert uneq.disagreement.n_eff < 20
    assert eq.audits_to_resolve is not None and uneq.audits_to_resolve is not None
    assert uneq.audits_to_resolve > eq.audits_to_resolve


# --------------------------------------------------------------------------- bins


def test_bins_assignment_and_labels() -> None:
    scores = [0.0, 0.49, 0.5, 0.69, 0.7, 0.95, 1.0, 1.2, -0.1]
    ds = [skipped(k, score=sc, equivalent=k % 2 == 0) for k, sc in enumerate(scores)]
    s = summarize(ds)
    counts = [b.n_accepted for b in s.bins]
    # [0,.5): 0.0, 0.49, -0.1 | [.5,.7): 0.5, 0.69 | [.7,.8): 0.7 | ... | [.95,1]: 0.95, 1.0, 1.2
    assert counts == [3, 2, 1, 0, 0, 3]
    assert s.bins[0].label == "[0.00, 0.50)" and s.bins[-1].label == "[0.95, 1.00]"
    assert s.bins[3].disagreement.value is None and s.bins[3].error is None
    assert sum(b.n_audited for b in s.bins) == 9
    assert s.no_score_bin is None


def test_bins_weighted_and_error() -> None:
    ds = [skipped(k, score=0.6, pi=0.5, equivalent=k > 0, correct=k > 1) for k in range(4)]
    s = summarize(ds, bins=(0.0, 0.5, 1.0))
    assert len(s.bins) == 2
    b = s.bins[1]
    assert b.n_accepted == 4 and b.n_audited == 4 and b.n_graded == 4
    assert b.disagreement.value == pytest.approx(0.25)
    assert b.error == wilson(2, 4)


def test_no_score_bin() -> None:
    ds = mixed_serve(3, 0) + [skipped(50, score=None, equivalent=False)]
    s = summarize(ds)
    assert s.no_score_bin is not None
    assert s.no_score_bin.n_accepted == 1 and s.no_score_bin.label == "no score"
    assert math.isnan(s.no_score_bin.lo)
    assert sum(b.n_accepted for b in s.bins) == 3


# --------------------------------------------------------------------------- costs


def test_costs_sums_and_overhead() -> None:
    ds = mixed_serve(2, 0) + [escalated(10, cost=0.2)]
    s = summarize(ds)
    assert s.cost_serving == pytest.approx(0.01 + 0.01 + 0.2)
    assert s.cost_audit == pytest.approx(0.1 + 0.1 + 0.0)
    assert s.cost_per_task == pytest.approx(0.22 / 3)
    assert s.audit_overhead == pytest.approx(0.2 / 0.22)
    assert s.n_unknown_cost == 0 and s.n_unknown_audit_cost == 0


def test_unknown_costs_counted() -> None:
    ds = [skipped(1, cost=None), skipped(2, cost=0.02), skipped(3, audit_cost=None)]
    s = summarize(ds)
    assert s.n_unknown_cost == 1
    assert s.n_unknown_audit_cost == 1
    assert s.cost_serving == pytest.approx(0.03)
    assert s.cost_per_task == pytest.approx(0.015)
    assert any("Unknown cost" in n for n in s.notes)
    assert any("unknown cost" in n for n in s.notes)  # audit attempt with unknown price


def test_all_costs_unknown() -> None:
    ds = [skipped(k, cost=None, audit_cost=None) for k in range(3)]
    s = summarize(ds)
    assert s.cost_serving is None and s.cost_per_task is None and s.audit_overhead is None
    assert s.est_all_slow_cost_per_task is None and s.est_savings is None


def test_serve_all_slow_cost_is_weighted() -> None:
    # Skipped cases audited at pi=0.25 with slow cost 0.04; escalated cases cost 0.2 on slow.
    ds = [skipped(k, pi=0.25, audit_cost=0.04) for k in range(4)]
    ds += [escalated(100 + k, slow_cost=0.2, cost=0.21) for k in range(4)]
    s = summarize(ds)
    est = s.est_all_slow_cost_per_task
    assert est is not None and est.method == "hajek-normal"
    expected = (4 * 4 * 0.04 + 4 * 0.2) / (4 * 4 + 4)
    assert est.value == pytest.approx(expected)
    assert est.lo is not None and est.hi is not None and est.lo < est.value < est.hi
    assert s.est_savings is not None
    assert s.est_savings.value == pytest.approx(1 - s.cost_per_task / expected)
    assert s.est_savings.lo is None or s.est_savings.lo <= s.est_savings.value


# --------------------------------------------------------------------------- notes & misc


def test_low_neff_and_tiny_pi_notes() -> None:
    ds = [skipped(f"a{k}", pi=0.9) for k in range(5)]
    ds += [skipped(f"b{k}", pi=0.01, status="skipped") for k in range(30)]
    ds += [skipped("b-audited", pi=0.01, equivalent=False)]
    s = summarize(ds)
    text = " ".join(s.notes)
    assert "Low effective sample size" in text
    assert "pi = 0.01" in text
    assert "single audit carries" in text


def test_mixed_modes_note() -> None:
    ds = mixed_serve(3, 0) + [eval_decision(50, run_id="r")]
    s = summarize(ds)
    assert s.mode == "mixed" and s.n_skipped == 4
    assert any("mixes modes" in n for n in s.notes)


def test_json_roundtrip_gives_same_summary() -> None:
    ds = mixed_serve(5, 2) + [escalated(20, correct=True), skipped(30, status="pending")]
    back = [Decision.from_dict(d.to_dict()) for d in ds]
    assert summarize(back, tolerance=0.2) == summarize(ds, tolerance=0.2)


def test_level_propagates() -> None:
    s = summarize(mixed_serve(15, 5), level=0.9)
    assert s.disagreement is not None and s.disagreement.level == 0.9
    assert s.level == 0.9
    wide = summarize(mixed_serve(15, 5), level=0.99).disagreement
    assert wide is not None and wide.lo is not None and s.disagreement.lo is not None
    assert wide.lo < s.disagreement.lo


# --------------------------------------------------------------------------- end to end


class _Backend:
    def __init__(self, name: str, answers: dict[str, str], cost: float) -> None:
        self.name = name
        self.answers = answers
        self.cost = cost

    def complete(self, request: Request) -> Completion:
        ans = self.answers[request.tags["task_id"]]
        return Completion(text=f"ANSWER: {ans}", model=self.name, cost_usd=self.cost)


class _Extractor:
    name = "fake"

    def extract(self, text: str) -> str:
        return text.split(":", 1)[1].strip()


class _Estimator:
    name = "fake"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def prepare(self, request: Request) -> Request:
        return request

    def estimate(self, task, request, completion, answer, backend):  # type: ignore[no-untyped-def]
        return ConfidenceResult(self.name, self.scores[task.id])


class _Exact:
    name = "exact"

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        return Judgement(candidate == target, self.name)


def _build(n: int, seed: int) -> tuple[list[Task], Cascade, Cascade]:
    rng = random.Random(seed)
    # (score, P(fast answer differs from slow)) - low confidence cases disagree far more often
    levels: Sequence[tuple[float, float]] = ((0.3, 0.6), (0.6, 0.4), (0.8, 0.1), (0.95, 0.02))
    tasks, fast, slow, scores = [], {}, {}, {}
    for i in range(n):
        tid = f"t{i}"
        score, p_dis = levels[rng.randrange(len(levels))]
        truth = str(i)
        slow[tid] = truth
        fast[tid] = truth + "x" if rng.random() < p_dis else truth
        scores[tid] = score
        tasks.append(Task(id=tid, prompt=f"q{i}", reference=truth))
    tiers = [
        Tier("fast", _Backend("fast", fast, 0.001), threshold=0.5, estimator=_Estimator(scores)),
        Tier("slow", _Backend("slow", slow, 0.01)),
    ]
    policy = AuditPolicy(rate=0.1, strata=((0.0, 0.7, 0.6), (0.7, 1.0, 0.05)), floor=0.01)
    serve = Cascade(tiers, extractor=_Extractor(), audit=policy, judge=_Exact())
    ev = Cascade(tiers, extractor=_Extractor(), judge=_Exact())
    return tasks, serve, ev


def test_end_to_end_weighted_estimate_is_unbiased_and_naive_is_not() -> None:
    tasks, serve, ev = _build(4000, seed=7)
    truth = summarize([ev.route(t, run_id="eval", mode="eval") for t in tasks])
    assert truth.mode == "eval" and truth.n_audited == truth.n_skipped > 0
    assert truth.disagreement is not None and truth.disagreement.value is not None
    true_rate = truth.disagreement.value

    s = summarize([serve.route(t, run_id="serve") for t in tasks], tolerance=0.5)
    assert s.n_skipped == truth.n_skipped  # same routing in both modes
    assert s.n_audited + s.n_not_selected == s.n_skipped
    assert 0 < s.n_audited < s.n_skipped
    est = s.disagreement
    naive = s.disagreement_unweighted
    assert est is not None and naive is not None
    assert est.value is not None and naive.value is not None
    assert est.lo is not None and est.hi is not None
    assert abs(est.value - true_rate) < 0.03
    assert est.lo <= true_rate <= est.hi
    # Low-confidence cases are oversampled and disagree more: the naive mean is biased upward.
    assert naive.value - true_rate > 0.08
    assert est.n_eff is not None and est.n_eff < s.n_audited
    # No comparator -> no grading; the audit tier is only a proxy.
    assert s.skipped_error is None and any("proxy" in n for n in s.notes)
    # All-slow cost: every slow call costs 0.01.
    assert s.est_all_slow_cost_per_task is not None
    assert s.est_all_slow_cost_per_task.value == pytest.approx(0.01)
    assert s.est_savings is not None and s.est_savings.value is not None
    assert 0.0 < s.est_savings.value < 1.0
    # Per-bin weighted estimates track the eval-mode bins.
    for b_est, b_true in zip(s.bins, truth.bins, strict=True):
        assert b_est.n_accepted == b_true.n_accepted
        if b_true.n_accepted and b_est.n_audited >= 30:
            assert b_est.disagreement.value is not None and b_true.disagreement.value is not None
            assert abs(b_est.disagreement.value - b_true.disagreement.value) < 0.1


def test_end_to_end_with_references_uses_skipped_error() -> None:
    tasks, serve, _ = _build(600, seed=3)
    graded = Cascade(serve.tiers, extractor=_Extractor(), comparator=_Exact(), audit=serve.audit)
    s = summarize([graded.route(t, run_id="g") for t in tasks], tolerance=0.2)
    assert s.n_graded_skipped == s.n_skipped > 0
    assert s.status_metric == "skipped_error"
    assert s.audit_tier_error is not None and s.audit_tier_error.value == 0.0
    # fast tier disagrees exactly when it is wrong here, so both rates estimate the same thing
    assert s.skipped_error is not None and s.disagreement is not None
    assert s.skipped_error.lo is not None and s.skipped_error.hi is not None
    assert s.served_accuracy is not None


# --------------------------------------------------------------------------- nonresponse & strata


def test_nonresponse_reweighted_within_pi_stratum_by_hand() -> None:
    # Stratum A: pi = .5, 50 selected, all done, 1 disagrees, 50 not selected.
    # Stratum B: pi = .02, 10 selected: 6 pending, 4 done (2 disagree); 490 not selected.
    ds = [skipped(f"a{k}", pi=0.5, equivalent=k >= 1) for k in range(50)]
    ds += [skipped(f"an{k}", pi=0.5, status="skipped") for k in range(50)]
    ds += [skipped(f"bp{k}", pi=0.02, status="pending") for k in range(6)]
    ds += [skipped(f"b{k}", pi=0.02, equivalent=k >= 2) for k in range(4)]
    ds += [skipped(f"bn{k}", pi=0.02, status="skipped") for k in range(490)]
    s = summarize(ds)
    # r_A = 1 -> w = 2; r_B = 0.4 -> w = 1 / (0.02 * 0.4) = 125.
    assert s.disagreement is not None
    assert s.disagreement.value == pytest.approx((1 * 2 + 2 * 125) / (50 * 2 + 4 * 125))
    ref = weighted_proportion(
        [k < 1 for k in range(50)] + [k < 2 for k in range(4)], [2.0] * 50 + [125.0] * 4
    )
    assert s.disagreement == ref
    assert s.n_unrepresented == 0
    assert any("missing at random within their stratum" in n for n in s.notes)


def test_stratum_dependent_nonresponse_is_unbiased_monte_carlo() -> None:
    # Reviewer's case, scaled down: 60% of audits pending only in the low-pi stratum (MCAR
    # within strata). Unadjusted 1/pi weights are biased low (~0.13 vs truth ~0.21 here; the
    # reviewer measured 0.130 vs 0.208 with pi = .02); the adjusted estimate is unbiased.
    rng = random.Random(5)
    strata = [(200, 0.5, 0.02), (300, 0.2, 0.02), (500, 0.1, 0.40)]
    # Not-selected cases carry no outcome, so they are built once and reused.
    pools = {
        pi: [skipped(f"n{pi}-{j}", pi=pi, status="skipped") for j in range(m)]
        for m, pi, _ in strata
    }
    reps = 30
    ests: list[float] = []
    truths: list[float] = []
    naive: list[float] = []
    for _ in range(reps):
        ds: list[Decision] = []
        k = 0
        for m, pi, e in strata:
            n_not = 0
            for _ in range(m):
                y = rng.random() < e
                k += y
                if rng.random() >= pi:
                    n_not += 1
                    continue
                st = "pending" if pi == 0.1 and rng.random() < 0.6 else "done"
                ds.append(skipped(len(ds), pi=pi, status=st, equivalent=not y))
            ds += pools[pi][:n_not]
        s = summarize(ds)
        assert s.disagreement is not None and s.disagreement.value is not None
        ests.append(s.disagreement.value)
        truths.append(k / len(ds))
        done = [d.shadow for d in ds if d.shadow is not None and d.shadow.status == "done"]
        ys = [sh.agreement is not None and sh.agreement.equivalent is False for sh in done]
        est = weighted_proportion(ys, [1 / sh.inclusion_prob for sh in done]).value
        assert est is not None
        naive.append(est)
    truth = sum(truths) / reps
    assert abs(sum(ests) / reps - truth) < 0.035
    assert truth - sum(naive) / reps > 0.05  # the unadjusted estimator is clearly biased low


def test_unrepresented_stratum_is_excluded_and_reported() -> None:
    ds = mixed_serve(18, 2, pi=0.5)  # 20 audits, 10% disagree
    ds += [skipped(f"p{k}", pi=0.05, status="pending") for k in range(2)]
    ds += [skipped(f"n{k}", pi=0.05, status="skipped") for k in range(38)]
    s = summarize(ds, tolerance=0.5)
    assert s.n_unrepresented == 40
    assert s.disagreement is not None and s.disagreement.value == pytest.approx(0.1)
    assert any("cannot be represented" in n and "pi = 0.05" in n for n in s.notes)
    # The unrepresented stratum holds 2/3 of the skipped cases: "ok" is blocked.
    assert s.sparse_strata == 1 and s.status == "inconclusive"


def _reviewer_design(n: int, disagree: bool) -> list[Decision]:
    """Deterministic version of the reviewer's design: shares .2/.3/.5 at pi .5/.1/.02, with the
    expected number of audits in each stratum."""
    ds: list[Decision] = []
    for share, pi in [(0.2, 0.5), (0.3, 0.1), (0.5, 0.02)]:
        m = round(share * n)
        n_sel = round(m * pi)
        for k in range(m):
            sel = k < n_sel
            ds.append(
                skipped(
                    len(ds),
                    pi=pi,
                    status="done" if sel else "skipped",
                    equivalent=not (disagree and sel and k == 0),
                )
            )
    return ds


def test_sparse_stratum_blocks_ok_regardless_of_outcomes() -> None:
    # N = 500: the pi = .02 stratum holds 50% of skipped cases but only 5 audits.
    for disagree in (False, True):
        s = summarize(_reviewer_design(500, disagree), tolerance=0.9)
        assert s.sparse_strata == 1
        assert s.disagreement is not None and s.disagreement.hi is not None
        assert s.disagreement.hi <= 0.9  # the interval alone would say "ok"
        assert s.status == "inconclusive"
        assert any("pi = 0.02" in n and "never 'ok'" in n for n in s.notes)
    # With enough audits in every large stratum the same design can be "ok".
    s = summarize(_reviewer_design(1000, False), tolerance=0.9)
    assert s.sparse_strata == 0 and s.status == "ok"


def test_single_stratum_few_audits_not_flagged_sparse() -> None:
    s = summarize(mixed_serve(5, 0, pi=0.5), tolerance=0.9)
    assert s.sparse_strata == 0 and s.status == "ok"


def test_continuous_pi_grouped_into_quantile_strata() -> None:
    # 60 distinct pi values -> 5 quantile groups of 12. With every selected audit done the
    # weights are plain 1/pi.
    def pi_of(k: int) -> float:
        return 0.05 + 0.01 * k

    ds = [skipped(k, pi=pi_of(k), equivalent=k % 4 != 0) for k in range(60)]
    s = summarize(ds)
    ys = [k % 4 == 0 for k in range(60)]
    assert s.disagreement == weighted_proportion(ys, [1 / pi_of(k) for k in range(60)])
    # Half the audits of the lowest-pi group pending: survivors there get weight 2/pi.
    pending = {1, 3, 5, 7, 9, 11}
    ds2 = [
        skipped(k, pi=pi_of(k), equivalent=k % 4 != 0, status="pending" if k in pending else "done")
        for k in range(60)
    ]
    s2 = summarize(ds2)
    kept = [k for k in range(60) if k not in pending]
    ws = [(2.0 if k < 12 else 1.0) / pi_of(k) for k in kept]
    assert s2.disagreement == weighted_proportion([k % 4 == 0 for k in kept], ws)


# --------------------------------------------------------------------------- resolving


def test_census_inconclusive_needs_tasks_not_audits() -> None:
    # Every skipped case graded and audited (pi = 1): more audits cannot help.
    ds = [skipped(i, pi=1.0, equivalent=i >= 6, correct=i >= 6) for i in range(60)]
    s = summarize(ds, tolerance=0.08)
    assert s.status == "inconclusive" and s.status_metric == "skipped_error"
    assert s.audits_to_resolve is None
    assert s.tasks_to_resolve is not None and s.tasks_to_resolve > 0
    # Census through disagreement only (no references): also tasks, not audits.
    ds = [skipped(i, pi=1.0, equivalent=i >= 6) for i in range(60)]
    s = summarize(ds, tolerance=0.08)
    assert s.status_metric == "disagreement" and s.status == "inconclusive"
    assert s.audits_to_resolve is None and s.tasks_to_resolve is not None
    n_total = 60 + s.tasks_to_resolve
    more = [skipped(i, pi=1.0, equivalent=i % 10 != 0) for i in range(n_total)]
    assert summarize(more, tolerance=0.08).status == "breach"


def test_eval_mode_inconclusive_has_no_audits_to_resolve() -> None:
    ds = [eval_decision(i, agree=i >= 4) for i in range(30)]
    s = summarize(ds, tolerance=0.1)
    assert s.status == "inconclusive"
    assert s.audits_to_resolve is None and s.tasks_to_resolve is not None


def test_sampled_inconclusive_reports_audits_and_tasks() -> None:
    ds = mixed_serve(17, 3, pi=0.2)
    ds += [skipped(100 + k, pi=0.2, status="skipped") for k in range(80)]
    s = summarize(ds, tolerance=0.1)
    assert s.status == "inconclusive"
    assert s.audits_to_resolve is not None and s.tasks_to_resolve is not None
    # 20 audits out of 100 skipped cases: about 5 skipped cases per extra audit.
    assert s.tasks_to_resolve == pytest.approx(5 * s.audits_to_resolve, abs=5)


def test_docstring_documents_optional_stopping() -> None:
    import shadowgate.audit as audit_mod

    doc = audit_mod.__doc__ or ""
    assert "optional stopping" in doc and "level" in doc
