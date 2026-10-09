"""Tests for shadowgate.cascade using self-contained fakes (no other shadowgate modules)."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from shadowgate.cascade import DEFAULT_TEMPLATE, AuditPolicy, Cascade, Tier, render_template
from shadowgate.errors import BackendError, ConfigError
from shadowgate.types import (
    Backend,
    Completion,
    ConfidenceResult,
    Decision,
    Judgement,
    Request,
    Task,
)

# --------------------------------------------------------------------------- fakes


class FakeBackend:
    """Scripted backend: per-task responses (text or exception), thread-safe call log."""

    def __init__(
        self,
        name: str,
        responses: Mapping[str, str | Exception] | None = None,
        *,
        default: str | Exception = "ANSWER: fallback",
        cost: float | None = 0.01,
        latency: float = 0.5,
    ) -> None:
        self.name = name
        self.responses = dict(responses or {})
        self.default = default
        self.cost = cost
        self.latency = latency
        self.calls: list[Request] = []
        self._lock = threading.Lock()

    def complete(self, request: Request) -> Completion:
        with self._lock:
            self.calls.append(request)
        r = self.responses.get(request.tags.get("task_id", ""), self.default)
        if isinstance(r, Exception):
            raise r
        return Completion(text=r, model=self.name, cost_usd=self.cost, latency_s=self.latency)

    def roles(self) -> list[str]:
        return [c.tags["role"] for c in self.calls]


class FakeExtractor:
    name = "fake"

    def extract(self, text: str) -> str:
        for line in reversed(text.splitlines()):
            if line.upper().startswith("ANSWER:"):
                return line.split(":", 1)[1].strip()
        return text.strip()


class FakeEstimator:
    """Fixed score, per-task scores, or a raising estimator; optional costed calls."""

    def __init__(
        self,
        score: float | None = 0.9,
        *,
        per_task: Mapping[str, float | None] | None = None,
        raises: Exception | None = None,
        call_cost: float | None = None,
        call_latency: float = 0.25,
    ) -> None:
        self.name = "fake-conf"
        self.score = score
        self.per_task = dict(per_task or {})
        self.raises = raises
        self.call_cost = call_cost
        self.call_latency = call_latency

    def prepare(self, request: Request) -> Request:
        return Request(**{**request.__dict__, "prompt": request.prompt + "\n[CONF]"})

    def estimate(
        self, task: Task, request: Request, completion: Completion, answer: str, backend: Backend
    ) -> ConfidenceResult:
        if self.raises is not None:
            raise self.raises
        score = self.per_task.get(task.id, self.score)
        calls: tuple[Completion, ...] = ()
        if self.call_cost is not None:
            calls = (
                Completion(
                    text="", model="monitor", cost_usd=self.call_cost, latency_s=self.call_latency
                ),
            )
        return ConfidenceResult(estimator=self.name, score=score, calls=calls)


class FakeComparator:
    def __init__(
        self,
        *,
        call_cost: float | None = None,
        raises: Exception | None = None,
        name: str = "fake-cmp",
    ) -> None:
        self.name = name
        self.call_cost = call_cost
        self.raises = raises
        self.calls = 0
        self._lock = threading.Lock()

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        with self._lock:
            self.calls += 1
        if self.raises is not None:
            raise self.raises
        calls: tuple[Completion, ...] = ()
        if self.call_cost is not None:
            calls = (Completion(text="VERDICT", model="judge", cost_usd=self.call_cost),)
        return Judgement(
            equivalent=candidate.strip().casefold() == target.strip().casefold(),
            comparator=self.name,
            calls=calls,
        )


def two_tier(
    fast: FakeBackend,
    slow: FakeBackend,
    *,
    estimator: FakeEstimator | None = None,
    threshold: float = 0.8,
    audit: AuditPolicy | None = None,
    comparator: FakeComparator | None = None,
    judge: FakeComparator | None = None,
) -> Cascade:
    return Cascade(
        [
            Tier("fast", fast, threshold=threshold, estimator=estimator or FakeEstimator()),
            Tier("slow", slow),
        ],
        extractor=FakeExtractor(),
        comparator=comparator or FakeComparator(),
        audit=audit,
        judge=judge,
    )


def t(task_id: str = "t1", prompt: str = "What is 2+2?", reference: str | None = None) -> Task:
    return Task(id=task_id, prompt=prompt, reference=reference)


# --------------------------------------------------------------------------- basic routing


def test_accept_at_tier0() -> None:
    fast, slow = FakeBackend("f", default="ANSWER: 4"), FakeBackend("s", default="ANSWER: 4")
    d = two_tier(fast, slow).route(t(), run_id="r")
    assert d.answer == "4"
    assert d.final_tier == "fast"
    assert not d.escalated
    assert len(d.attempts) == 1
    assert d.attempts[0].accepted and d.attempts[0].threshold == 0.8
    assert slow.calls == []
    assert d.error is None and d.mode == "serve" and d.run_id == "r"
    assert d.created_at.endswith("+00:00")


def test_escalate_on_low_confidence() -> None:
    fast, slow = FakeBackend("f", default="ANSWER: 5"), FakeBackend("s", default="ANSWER: 4")
    d = two_tier(fast, slow, estimator=FakeEstimator(0.5)).route(t())
    assert d.answer == "4" and d.final_tier == "slow" and d.escalated
    assert [a.accepted for a in d.attempts] == [False, True]
    assert d.attempts[1].threshold is None and d.attempts[1].confidence is None


def test_threshold_boundary_is_inclusive() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s")
    d = two_tier(fast, slow, estimator=FakeEstimator(0.8), threshold=0.8).route(t())
    assert d.final_tier == "fast"


def test_threshold_above_one_always_escalates() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s")
    d = two_tier(fast, slow, estimator=FakeEstimator(1.0), threshold=1.0001).route(t())
    assert d.final_tier == "slow"


def test_escalate_on_none_confidence() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s", default="ANSWER: 4")
    d = two_tier(fast, slow, estimator=FakeEstimator(None), threshold=0.0).route(t())
    assert d.final_tier == "slow"
    assert d.attempts[0].confidence is not None and d.attempts[0].confidence.score is None


def test_estimator_exception_recorded_and_escalates() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s", default="ANSWER: 4")
    est = FakeEstimator(raises=RuntimeError("boom"))
    d = two_tier(fast, slow, estimator=est).route(t())
    conf = d.attempts[0].confidence
    assert conf is not None and conf.score is None and "boom" in conf.detail["error"]
    assert conf.estimator == "fake-conf"
    assert d.final_tier == "slow" and d.error is None


def test_prepare_applied_and_tags() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s")
    two_tier(fast, slow, estimator=FakeEstimator(0.1)).route(t("abc"))
    assert fast.calls[0].prompt.endswith("[CONF]")
    assert not slow.calls[0].prompt.endswith("[CONF]")  # final tier estimator ignored
    assert dict(fast.calls[0].tags) == {"task_id": "abc", "tier": "fast", "role": "serve"}
    assert dict(slow.calls[0].tags) == {"task_id": "abc", "tier": "slow", "role": "serve"}


def test_three_tier_cascade() -> None:
    a, b, c = FakeBackend("a"), FakeBackend("b", default="ANSWER: B"), FakeBackend("c")
    cas = Cascade(
        [
            Tier("a", a, threshold=0.9, estimator=FakeEstimator(0.5)),
            Tier("b", b, threshold=0.6, estimator=FakeEstimator(0.7)),
            Tier("c", c),
        ],
        extractor=FakeExtractor(),
    )
    d = cas.route(t())
    assert d.final_tier == "b" and d.answer == "B" and d.escalated
    assert len(d.attempts) == 2 and c.calls == []


def test_request_fields_from_tier() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s")
    cas = Cascade(
        [
            Tier(
                "fast",
                fast,
                threshold=0.5,
                estimator=FakeEstimator(),
                system="SYS",
                max_tokens=77,
                temperature=0.3,
                effort="low",
                template="Q: {prompt}",
            ),
            Tier("slow", slow),
        ],
        extractor=FakeExtractor(),
    )
    cas.route(t(prompt="hi"))
    r = fast.calls[0]
    assert (r.system, r.max_tokens, r.temperature, r.effort) == ("SYS", 77, 0.3, "low")
    assert r.prompt == "Q: hi\n[CONF]"


def test_default_template_and_braces_in_prompt() -> None:
    assert "{prompt}" in DEFAULT_TEMPLATE and "ANSWER:" in DEFAULT_TEMPLATE
    fast, slow = FakeBackend("f"), FakeBackend("s")
    prompt = "Format {x} and {0} and {prompt} literally: {{}}"
    cas = Cascade(
        [
            Tier("fast", fast, threshold=0.5, estimator=FakeEstimator(), template="<{prompt}> {y}"),
            Tier("slow", slow),
        ],
        extractor=FakeExtractor(),
    )
    d = cas.route(t(prompt=prompt))
    assert fast.calls[0].prompt == f"<{prompt}> {{y}}\n[CONF]"
    assert d.error is None
    assert render_template(DEFAULT_TEMPLATE, prompt).startswith(prompt)


def test_tier_extractor_overrides_default() -> None:
    class Upper:
        name = "upper"

        def extract(self, text: str) -> str:
            return text.upper()

    fast, slow = FakeBackend("f", default="raw"), FakeBackend("s")
    cas = Cascade(
        [
            Tier("fast", fast, threshold=0.5, estimator=FakeEstimator(), extractor=Upper()),
            Tier("slow", slow),
        ],
        extractor=FakeExtractor(),
    )
    assert cas.route(t()).answer == "RAW"


def test_single_tier_cascade() -> None:
    only = FakeBackend("o", default="ANSWER: 1")
    d = Cascade([Tier("only", only)], extractor=FakeExtractor()).route(t())
    assert d.answer == "1" and d.final_tier == "only" and not d.escalated


# --------------------------------------------------------------------------- failures


def test_backend_failure_escalates() -> None:
    fast = FakeBackend("f", default=BackendError("rate limited", retryable=True))
    slow = FakeBackend("s", default="ANSWER: 4")
    d = two_tier(fast, slow).route(t())
    assert d.final_tier == "slow" and d.answer == "4" and d.error is None
    a0 = d.attempts[0]
    assert a0.completion is None and not a0.accepted and "rate limited" in (a0.error or "")
    assert a0.backend == "f"


def test_unexpected_backend_exception_is_recorded() -> None:
    fast = FakeBackend("f", default=ValueError("bad"))
    slow = FakeBackend("s", default="ANSWER: 4")
    d = two_tier(fast, slow).route(t())
    assert d.final_tier == "slow" and "ValueError" in (d.attempts[0].error or "")


def test_final_tier_failure_sets_error_and_serves_last_answer() -> None:
    fast = FakeBackend("f", default="ANSWER: 5")
    slow = FakeBackend("s", default=BackendError("down"))
    d = two_tier(fast, slow, estimator=FakeEstimator(0.1)).route(t(reference="5"))
    assert d.error is not None and "slow" in d.error and "down" in d.error
    assert d.answer == "5" and d.final_tier == "fast" and d.escalated
    assert d.correct is not None and d.correct.equivalent is True
    assert d.shadow is None


def test_all_tiers_fail() -> None:
    fast = FakeBackend("f", default=BackendError("x"))
    slow = FakeBackend("s", default=BackendError("y"))
    d = two_tier(fast, slow).route(t(reference="4"))
    assert d.error is not None and d.answer == "" and d.final_tier == "slow"
    assert d.correct is None
    assert d.cost_usd == 0.0


def test_comparator_exception_recorded() -> None:
    fast, slow = FakeBackend("f", default="ANSWER: 4"), FakeBackend("s")
    cmp_ = FakeComparator(raises=RuntimeError("cmp broke"))
    d = two_tier(fast, slow, comparator=cmp_).route(t(reference="4"))
    assert d.correct is not None and d.correct.equivalent is None
    assert "cmp broke" in d.correct.detail["error"]


def test_invalid_route_mode() -> None:
    with pytest.raises(ValueError):
        two_tier(FakeBackend("f"), FakeBackend("s")).route(t(), mode="bogus")


# --------------------------------------------------------------------------- grading


def test_grading_with_reference() -> None:
    fast = FakeBackend("f", default="ANSWER: 5")
    slow = FakeBackend("s", default="ANSWER: 4")
    d = two_tier(fast, slow, estimator=FakeEstimator(0.1)).route(t(reference="4"))
    assert d.attempts[0].correct is not None and d.attempts[0].correct.equivalent is False
    assert d.attempts[1].correct is not None and d.attempts[1].correct.equivalent is True
    assert d.correct is not None and d.correct.equivalent is True


def test_no_grading_without_reference_or_comparator() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s")
    d = two_tier(fast, slow).route(t())
    assert d.correct is None and d.attempts[0].correct is None
    cas = Cascade(
        [Tier("only", FakeBackend("o"))], extractor=FakeExtractor(), judge=FakeComparator()
    )
    d2 = cas.route(t(reference="x"))
    assert d2.correct is None


# --------------------------------------------------------------------------- eval mode


def test_eval_mode_runs_all_tiers_and_reports_router_choice() -> None:
    fast = FakeBackend("f", default="ANSWER: 5", cost=0.01, latency=1.0)
    slow = FakeBackend("s", default="ANSWER: 4", cost=0.10, latency=3.0)
    est = FakeEstimator(0.9, call_cost=0.001, call_latency=0.5)
    d = two_tier(fast, slow, estimator=est).route(t(reference="4"), run_id="e", mode="eval")
    assert len(d.attempts) == 2 and len(slow.calls) == 1
    assert d.mode == "eval" and d.shadow is None
    assert d.final_tier == "fast" and d.answer == "5" and not d.escalated
    ag = d.attempts[0].agreement
    assert ag is not None and ag.equivalent is False
    assert d.attempts[1].agreement is None
    assert d.correct is not None and d.correct.equivalent is False
    assert d.cost_usd == pytest.approx(0.011)
    assert d.audit_cost_usd == pytest.approx(0.10)
    assert d.latency_s == pytest.approx(1.5)
    assert fast.roles() == ["eval"] and slow.roles() == ["eval"]


def test_eval_mode_escalated_choice_and_confidence_on_all() -> None:
    a, b, c = FakeBackend("a"), FakeBackend("b", default="ANSWER: B"), FakeBackend("c")
    cas = Cascade(
        [
            Tier("a", a, threshold=0.9, estimator=FakeEstimator(0.5)),
            Tier("b", b, threshold=0.6, estimator=FakeEstimator(0.3)),
            Tier("c", c),
        ],
        extractor=FakeExtractor(),
        judge=FakeComparator(call_cost=0.002),
    )
    d = cas.route(t(), mode="eval")
    assert d.final_tier == "c" and d.escalated
    assert [x.confidence.score if x.confidence else None for x in d.attempts] == [0.5, 0.3, None]
    assert all(x.agreement is not None for x in d.attempts[:2])
    # all three tiers are serving; only judge calls count as audit cost
    assert d.cost_usd == pytest.approx(0.03)
    assert d.audit_cost_usd == pytest.approx(0.004)


def test_eval_mode_last_tier_failure_no_agreement() -> None:
    fast = FakeBackend("f", default="ANSWER: 5")
    slow = FakeBackend("s", default=BackendError("down"))
    d = two_tier(fast, slow).route(t(), mode="eval")
    assert d.final_tier == "fast" and d.error is None
    assert d.attempts[0].agreement is None and d.attempts[1].error is not None


# --------------------------------------------------------------------------- shadow audits


def test_shadow_inline_selected() -> None:
    fast = FakeBackend("f", default="ANSWER: 5", cost=0.01)
    slow = FakeBackend("s", default="ANSWER: 4", cost=0.10)
    pol = AuditPolicy(rate=1.0)
    d = two_tier(fast, slow, audit=pol).route(t(reference="4"), run_id="r")
    assert d.final_tier == "fast" and d.answer == "5"
    sh = d.shadow
    assert sh is not None and sh.status == "done" and sh.inclusion_prob == 1.0
    assert sh.audit_tier == "slow"
    assert sh.attempt is not None and sh.attempt.answer == "4"
    assert sh.attempt.correct is not None and sh.attempt.correct.equivalent is True
    assert sh.agreement is not None and sh.agreement.equivalent is False
    assert d.attempts[0].agreement == sh.agreement
    assert slow.roles() == ["audit"]
    assert d.cost_usd == pytest.approx(0.01)
    assert d.audit_cost_usd == pytest.approx(0.10)


def test_shadow_not_selected_records_prob() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s")
    pol = AuditPolicy(rate=0.01, floor=0.01, seed=3)
    cas = two_tier(fast, slow, audit=pol)
    ds = [cas.route(t(f"t{i}"), run_id="r") for i in range(200)]
    statuses = [d.shadow.status for d in ds if d.shadow]
    assert len(statuses) == 200
    assert all(d.shadow and d.shadow.inclusion_prob == 0.01 for d in ds)
    n_done = statuses.count("done")
    assert statuses.count("skipped") + n_done == 200
    assert n_done < 15
    assert len(slow.calls) == n_done
    skipped = next(d for d in ds if d.shadow and d.shadow.status == "skipped")
    assert skipped.shadow is not None and skipped.shadow.attempt is None
    assert skipped.audit_cost_usd == 0.0


def test_shadow_selection_is_deterministic() -> None:
    pol = AuditPolicy(rate=0.3, seed=7)
    sel1 = [pol.selected("run", f"t{i}", 0.3) for i in range(1000)]
    sel2 = [pol.selected("run", f"t{i}", 0.3) for i in range(1000)]
    assert sel1 == sel2
    assert 230 < sum(sel1) < 370
    other = [AuditPolicy(seed=8).selected("run", f"t{i}", 0.3) for i in range(1000)]
    assert other != sel1
    assert all(pol.selected("x", f"t{i}", 1.0) for i in range(100))


def test_no_shadow_when_escalated_or_off_or_no_policy() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s")
    d = two_tier(fast, slow, estimator=FakeEstimator(0.1), audit=AuditPolicy(rate=1.0)).route(t())
    assert d.shadow is None
    d = two_tier(fast, slow, audit=AuditPolicy(rate=1.0, mode="off")).route(t())
    assert d.shadow is None
    d = two_tier(fast, slow).route(t())
    assert d.shadow is None


def test_shadow_deferred_then_complete_audit() -> None:
    fast = FakeBackend("f", default="ANSWER: 4", cost=0.01)
    slow = FakeBackend("s", default="ANSWER: 4", cost=0.10)
    judge = FakeComparator(call_cost=0.003)
    cas = two_tier(fast, slow, audit=AuditPolicy(rate=1.0, mode="deferred"), judge=judge)
    d = cas.route(t(), run_id="r")
    assert d.shadow is not None and d.shadow.status == "pending"
    assert slow.calls == [] and d.audit_cost_usd == 0.0
    done = cas.complete_audit(d)
    assert done.shadow is not None and done.shadow.status == "done"
    assert done.shadow.agreement is not None and done.shadow.agreement.equivalent is True
    assert done.shadow.inclusion_prob == 1.0
    assert done.attempts[0].agreement == done.shadow.agreement
    assert done.audit_cost_usd == pytest.approx(0.103)
    assert done.cost_usd == d.cost_usd and done.answer == d.answer
    assert len(fast.calls) == 1 and slow.roles() == ["audit"]
    # idempotent on non-pending
    assert cas.complete_audit(done) is done
    assert cas.complete_audit(cas.route(t("x"), mode="eval")).shadow is None


def test_complete_audit_after_json_round_trip() -> None:
    fast, slow = FakeBackend("f", default="ANSWER: 4"), FakeBackend("s", default="ANSWER: 4")
    cas = two_tier(fast, slow, audit=AuditPolicy(rate=1.0, mode="deferred"))
    d = Decision.from_dict(json.loads(json.dumps(cas.route(t()).to_dict())))
    done = cas.complete_audit(d)
    assert done.shadow is not None and done.shadow.status == "done"


def test_audit_failure_status_error() -> None:
    fast = FakeBackend("f")
    slow = FakeBackend("s", default=BackendError("audit down"))
    d = two_tier(fast, slow, audit=AuditPolicy(rate=1.0)).route(t())
    assert d.error is None and d.final_tier == "fast"
    sh = d.shadow
    assert sh is not None and sh.status == "error"
    assert sh.attempt is not None and "audit down" in (sh.attempt.error or "")
    assert sh.agreement is None and d.attempts[0].agreement is None


def test_custom_audit_tier() -> None:
    a, b, c = FakeBackend("a"), FakeBackend("b"), FakeBackend("c")
    cas = Cascade(
        [
            Tier("a", a, threshold=0.5, estimator=FakeEstimator(0.9)),
            Tier("b", b, threshold=0.5, estimator=FakeEstimator(0.9)),
            Tier("c", c),
        ],
        extractor=FakeExtractor(),
        audit=AuditPolicy(rate=1.0, audit_tier="b"),
        judge=FakeComparator(),
    )
    d = cas.route(t())
    assert d.shadow is not None and d.shadow.audit_tier == "b" and d.shadow.status == "done"
    assert b.roles() == ["audit"] and c.calls == []


def test_judge_defaults_to_comparator_and_separate_judge() -> None:
    fast, slow = FakeBackend("f", default="ANSWER: 4"), FakeBackend("s", default="ANSWER: 4")
    cmp_, judge = FakeComparator(name="cmp"), FakeComparator(name="judge")
    d = two_tier(fast, slow, audit=AuditPolicy(rate=1.0), comparator=cmp_).route(t())
    assert d.shadow is not None and d.shadow.agreement is not None
    assert d.shadow.agreement.comparator == "cmp"
    d = two_tier(fast, slow, audit=AuditPolicy(rate=1.0), comparator=cmp_, judge=judge).route(t())
    assert d.shadow is not None and d.shadow.agreement is not None
    assert d.shadow.agreement.comparator == "judge"


# --------------------------------------------------------------------------- audit policy


def test_strata_inclusion_probs() -> None:
    pol = AuditPolicy(rate=0.1, strata=((0.0, 0.8, 0.5), (0.8, 0.95, 0.2), (0.95, 1.0, 0.05)))
    assert pol.inclusion_prob(0.5) == 0.5
    assert pol.inclusion_prob(0.8) == 0.2
    assert pol.inclusion_prob(0.95) == 0.05
    assert pol.inclusion_prob(1.0) == 0.05  # last stratum's hi is inclusive
    assert pol.inclusion_prob(None) == 0.1
    assert pol.inclusion_prob(1.5) == 0.1  # no match -> rate


def test_strata_first_match_wins_and_inner_hi_exclusive() -> None:
    pol = AuditPolicy(rate=0.1, strata=((0.0, 0.5, 0.4), (0.0, 1.0, 0.3), (0.5, 0.6, 0.9)))
    assert pol.inclusion_prob(0.2) == 0.4
    assert pol.inclusion_prob(0.5) == 0.3
    pol2 = AuditPolicy(strata=((0.0, 0.5, 0.4), (0.5, 0.9, 0.3)))
    assert pol2.inclusion_prob(0.9) == 0.3  # last stratum inclusive
    pol3 = AuditPolicy(rate=0.1, strata=((0.0, 0.5, 0.4), (0.5, 0.9, 0.3), (0.9, 0.95, 0.2)))
    assert pol3.inclusion_prob(0.9) == 0.2


def test_floor_applies() -> None:
    pol = AuditPolicy(rate=0.001, floor=0.05, strata=((0.9, 1.0, 0.01),))
    assert pol.inclusion_prob(0.95) == 0.05
    assert pol.inclusion_prob(None) == 0.05


def test_strata_used_in_route() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s")
    pol = AuditPolicy(rate=0.1, strata=((0.8, 0.9, 1.0),))
    est = FakeEstimator(per_task={"lo": 0.85, "hi": 0.97})
    cas = two_tier(fast, slow, estimator=est, audit=pol)
    d_lo = cas.route(t("lo"))
    assert d_lo.shadow is not None and d_lo.shadow.inclusion_prob == 1.0
    assert d_lo.shadow.status == "done"
    d_hi = cas.route(t("hi"))
    assert d_hi.shadow is not None and d_hi.shadow.inclusion_prob == 0.1


def test_strata_lists_normalised_to_tuples() -> None:
    pol = AuditPolicy(strata=[[0, 0.5, 1]])  # type: ignore[arg-type]
    assert pol.strata == ((0.0, 0.5, 1.0),)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate": 0.0},
        {"rate": 1.5},
        {"floor": 0.0},
        {"floor": 2.0},
        {"mode": "sometimes"},
        {"strata": ((0.5, 0.2, 0.1),)},
        {"strata": ((0.0, 0.5, 0.0),)},
        {"strata": ((0.0, 0.5),)},
        {"rate": float("nan")},
    ],
)
def test_audit_policy_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        AuditPolicy(**kwargs)


# --------------------------------------------------------------------------- tier validation


def _mk(tiers: list[Tier], **kw: Any) -> Cascade:
    return Cascade(tiers, extractor=FakeExtractor(), judge=FakeComparator(), **kw)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _mk([]),
        lambda: _mk([Tier("a", FakeBackend("a")), Tier("a", FakeBackend("b"))]),
        lambda: _mk([Tier("a", FakeBackend("a"), threshold=0.5), Tier("b", FakeBackend("b"))]),
        lambda: _mk(
            [Tier("a", FakeBackend("a"), estimator=FakeEstimator()), Tier("b", FakeBackend("b"))]
        ),
        lambda: Tier("a", FakeBackend("a"), threshold=-0.1),
        lambda: Tier("a", FakeBackend("a"), threshold=float("nan")),
        lambda: Tier("a", FakeBackend("a"), template="no token"),
        lambda: Tier("", FakeBackend("a")),
        lambda: Tier("a", FakeBackend("a"), max_tokens=0),
        lambda: _mk([Tier("a", FakeBackend("a"))], audit=AuditPolicy(audit_tier="zzz")),
    ],
)
def test_construction_errors(build: Callable[[], object]) -> None:
    with pytest.raises(ConfigError):
        build()


def test_final_tier_threshold_and_estimator_ignored() -> None:
    est = FakeEstimator(0.0)
    b = FakeBackend("b", default="ANSWER: 7")
    cas = _mk([Tier("only", b, threshold=0.99, estimator=est)])
    d = cas.route(t())
    assert d.answer == "7" and d.attempts[0].accepted
    assert d.attempts[0].confidence is None and d.attempts[0].threshold is None
    assert not b.calls[0].prompt.endswith("[CONF]")


def test_threshold_zero_and_above_one_allowed() -> None:
    _mk(
        [
            Tier("a", FakeBackend("a"), threshold=0.0, estimator=FakeEstimator()),
            Tier("b", FakeBackend("b"), threshold=1.5, estimator=FakeEstimator()),
            Tier("c", FakeBackend("c")),
        ]
    )


# --------------------------------------------------------------------------- costs


def test_costs_and_latency_serve_mode() -> None:
    fast = FakeBackend("f", default="ANSWER: 5", cost=0.01, latency=1.0)
    slow = FakeBackend("s", default="ANSWER: 4", cost=0.10, latency=4.0)
    est = FakeEstimator(0.1, call_cost=0.002, call_latency=0.5)
    judge = FakeComparator(call_cost=0.005)
    d = two_tier(fast, slow, estimator=est, comparator=judge).route(t(reference="4"))
    assert d.cost_usd == pytest.approx(0.01 + 0.002 + 0.10)
    assert d.latency_s == pytest.approx(1.0 + 0.5 + 4.0)
    # grading judge calls on both attempts
    assert d.audit_cost_usd == pytest.approx(0.01)


def test_latency_excludes_audit() -> None:
    fast = FakeBackend("f", latency=1.0)
    slow = FakeBackend("s", latency=9.0)
    d = two_tier(fast, slow, audit=AuditPolicy(rate=1.0)).route(t())
    assert d.latency_s == pytest.approx(1.0)


def test_cost_none_propagation_serving() -> None:
    fast = FakeBackend("f", cost=None)
    slow = FakeBackend("s", cost=0.1)
    d = two_tier(fast, slow, estimator=FakeEstimator(0.1)).route(t())
    assert d.cost_usd is None
    assert d.audit_cost_usd == 0.0


def test_cost_none_propagation_confidence_calls() -> None:
    fast, slow = FakeBackend("f"), FakeBackend("s")

    class UnknownCallEstimator(FakeEstimator):
        def estimate(self, *args: Any) -> ConfidenceResult:
            return ConfidenceResult("u", 0.9, calls=(Completion(text="", model="m"),))

    d = two_tier(fast, slow, estimator=UnknownCallEstimator()).route(t())
    assert d.cost_usd is None


def test_cost_none_propagation_audit() -> None:
    fast = FakeBackend("f", cost=0.01)
    slow = FakeBackend("s", cost=None)
    d = two_tier(fast, slow, audit=AuditPolicy(rate=1.0)).route(t())
    assert d.cost_usd == pytest.approx(0.01)
    assert d.audit_cost_usd is None
    pending = two_tier(fast, slow, audit=AuditPolicy(rate=1.0, mode="deferred")).route(t())
    cas = two_tier(fast, slow, audit=AuditPolicy(rate=1.0, mode="deferred"))
    assert cas.complete_audit(pending).audit_cost_usd is None


def test_cost_none_eval_extra_tier() -> None:
    fast = FakeBackend("f", cost=0.01)
    slow = FakeBackend("s", cost=None)
    d = two_tier(fast, slow).route(t(), mode="eval")
    assert d.cost_usd == pytest.approx(0.01)
    assert d.audit_cost_usd is None


def test_failed_attempt_contributes_zero_cost() -> None:
    fast = FakeBackend("f", default=BackendError("x"))
    slow = FakeBackend("s", cost=0.1)
    d = two_tier(fast, slow).route(t())
    assert d.cost_usd == pytest.approx(0.1)


# --------------------------------------------------------------------------- serialisation, threads


def test_decision_round_trip() -> None:
    fast = FakeBackend("f", default="ANSWER: 5")
    slow = FakeBackend("s", default="ANSWER: 4")
    est = FakeEstimator(0.9, call_cost=0.001)
    cas = two_tier(
        fast,
        slow,
        estimator=est,
        audit=AuditPolicy(rate=1.0),
        judge=FakeComparator(call_cost=0.002),
    )
    for mode in ("serve", "eval"):
        d = cas.route(t(reference="4", prompt="p {x}"), run_id="rt", mode=mode)
        again = Decision.from_dict(json.loads(json.dumps(d.to_dict())))
        assert again == d


def test_round_trip_with_error() -> None:
    fast = FakeBackend("f", default=BackendError("x"))
    slow = FakeBackend("s", default=BackendError("y"))
    d = two_tier(fast, slow).route(t())
    assert Decision.from_dict(json.loads(json.dumps(d.to_dict()))) == d


def test_concurrent_routing() -> None:
    responses = {f"t{i}": f"ANSWER: {i}" for i in range(300)}
    fast = FakeBackend("f", responses)
    slow = FakeBackend("s", responses)
    est = FakeEstimator(per_task={f"t{i}": (0.9 if i % 2 else 0.1) for i in range(300)})
    cas = two_tier(fast, slow, estimator=est, audit=AuditPolicy(rate=0.5, seed=1))
    tasks = [t(f"t{i}", reference=str(i)) for i in range(300)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        ds = list(pool.map(lambda task: cas.route(task, run_id="c"), tasks))
    serial = [cas.route(task, run_id="c") for task in tasks]
    for i, d in enumerate(ds):
        assert d.task.id == f"t{i}" and d.answer == str(i)
        assert d.final_tier == ("fast" if i % 2 else "slow")
        assert d.correct is not None and d.correct.equivalent is True
        assert (d.shadow is None) == (i % 2 == 0)
        assert d.shadow == serial[i].shadow
    assert len(fast.calls) == 600
