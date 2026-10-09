"""Confidence-gated LLM cascade with shadow audits of the skipped cases.

A :class:`Cascade` walks its tiers in order. Each non-final tier answers, its confidence
estimator scores the answer, and the answer is served when the score clears the tier's
threshold; otherwise the task escalates. The final tier always serves what it produces.

Answers served by non-final tiers ("skipped cases") are sampled for a shadow audit under an
:class:`AuditPolicy`: the audit tier, which is always the final tier, re-answers the
task and a judge compares the two answers. Every skipped case records its inclusion probability
so downstream estimates can weight by ``1 / inclusion_prob`` and stay unbiased.

Notes on configuration:

* ``Tier.template`` is rendered by replacing every ``{prompt}`` token with the task prompt
  (plain string replacement, not ``str.format``), so braces in prompts or templates are kept
  verbatim. ``{{`` is therefore *not* an escape sequence.
* A non-final tier needs an ``estimator`` and a ``threshold >= 0``. A threshold above 1 means
  "always escalate" (scores live in [0, 1]); 0 means "accept whenever a score exists".
* The final tier's ``threshold`` and ``estimator`` are ignored: whatever it answers is served.
* ``audit=None`` disables shadow audits entirely (no inclusion probabilities are recorded).
* ``AuditPolicy.audit_tier`` may only name the final tier (or be None, meaning the final tier):
  every non-final tier can accept, so any other audit tier would audit some skipped cases with
  themselves or a weaker tier and the audit would be meaningless.
* A non-final tier never accepts a completion whose ``stop_reason`` is "refusal", "error" or
  "max_tokens", or whose extracted answer is empty: the estimator is not run, the attempt
  escalates, and ``confidence`` records ``score=None`` with ``detail["rejected"]`` giving why.
* If a non-final tier's estimator raises, money its monitor/resample calls already spent is not
  visible, so the decision's ``cost_usd`` becomes unknown (None) and the confidence detail carries
  ``{"error": ..., "cost_unknown": True}`` (unless the exception carries the ``calls`` it made).
* Audit strata: first match wins on ``lo <= conf < hi``; a band's ``hi`` is also inclusive when it
  equals the largest ``hi`` among the bands (or is >= 1), whatever order the bands are listed in.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from shadowgate.errors import ConfigError
from shadowgate.types import (
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
)

if TYPE_CHECKING:
    from shadowgate.extract import Extractor

__all__ = [
    "DEFAULT_TEMPLATE",
    "AUDIT_MODES",
    "ROUTE_MODES",
    "Tier",
    "AuditPolicy",
    "Cascade",
    "render_template",
]

log = logging.getLogger("shadowgate.cascade")

DEFAULT_TEMPLATE = (
    "{prompt}\n\n"
    "Solve the task above. Reason briefly if it helps, then finish with a final line of the "
    "form:\n"
    "ANSWER: <answer>"
)

PROMPT_TOKEN = "{prompt}"
AUDIT_MODES = ("inline", "deferred", "off")
ROUTE_MODES = ("serve", "eval")


def render_template(template: str, prompt: str) -> str:
    """Insert ``prompt`` into ``template`` at every ``{prompt}`` token (no ``str.format``)."""
    return template.replace(PROMPT_TOKEN, prompt)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def _add(a: float | None, b: float | None) -> float | None:
    """Sum two costs; unknown (None) is contagious."""
    if a is None or b is None:
        return None
    return a + b


def _sum_costs(costs: Sequence[float | None]) -> float | None:
    total: float | None = 0.0
    for c in costs:
        total = _add(total, c)
    return total


UNUSABLE_STOP_REASONS = frozenset({"refusal", "error", "max_tokens"})


def _unusable(completion: Completion, answer: str) -> str | None:
    """Why a non-final tier must not accept this completion, or None if it may."""
    if completion.stop_reason in UNUSABLE_STOP_REASONS:
        return f"stop_reason={completion.stop_reason}"
    if not answer.strip():
        return "empty answer"
    return None


def _exc_calls(exc: BaseException) -> tuple[Completion, ...] | None:
    """Completions an estimator exception says it made (``exc.calls``), if it carries them."""
    calls = getattr(exc, "calls", None)
    if calls is None:
        return None
    try:
        calls = tuple(calls)
    except TypeError:
        return None
    return calls if all(isinstance(c, Completion) for c in calls) else None


def _err(exc: BaseException) -> str:
    msg = str(exc)
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class Tier:
    """One rung of the cascade: a backend, a prompt template and an acceptance rule.

    ``threshold`` and ``estimator`` are required for every tier except the last, whose values
    are ignored. ``extractor`` None means "use the cascade's default extractor".
    """

    name: str
    backend: Backend
    threshold: float | None = None
    estimator: ConfidenceEstimator | None = None
    system: str | None = None
    template: str = DEFAULT_TEMPLATE
    max_tokens: int = 2048
    temperature: float | None = None
    effort: str | None = None
    extractor: Extractor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("tier name must be a non-empty string")
        if PROMPT_TOKEN not in self.template:
            raise ConfigError(f"tier {self.name!r}: template must contain {PROMPT_TOKEN!r}")
        if self.threshold is not None and (not _is_number(self.threshold) or self.threshold < 0):
            raise ConfigError(
                f"tier {self.name!r}: threshold must be a number >= 0, got {self.threshold!r}"
            )
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or (self.max_tokens <= 0)
        ):
            raise ConfigError(f"tier {self.name!r}: max_tokens must be a positive integer")

    def render(self, task: Task) -> str:
        return render_template(self.template, task.prompt)


@dataclass(frozen=True)
class AuditPolicy:
    """How skipped cases are sampled for shadow audits.

    ``strata`` are ``(lo, hi, rate)`` bands on the served confidence; the first band with
    ``lo <= conf < hi`` wins; a band whose ``hi`` is the largest ``hi`` (or >= 1) also includes
    ``conf == hi``, regardless of listing order. Unmatched or missing
    confidence uses ``rate``. The result is never below ``floor`` (> 0), which keeps every
    skipped case auditable and inverse-probability estimates unbiased.
    """

    rate: float = 0.1
    strata: tuple[tuple[float, float, float], ...] = ()
    floor: float = 0.01
    mode: str = "inline"
    audit_tier: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not _is_number(self.rate) or not 0 < self.rate <= 1:
            raise ConfigError(f"audit rate must be in (0, 1], got {self.rate!r}")
        if not _is_number(self.floor) or not 0 < self.floor <= 1:
            raise ConfigError(f"audit floor must be in (0, 1], got {self.floor!r}")
        if self.mode not in AUDIT_MODES:
            raise ConfigError(f"audit mode must be one of {AUDIT_MODES}, got {self.mode!r}")
        strata: list[tuple[float, float, float]] = []
        for i, s in enumerate(self.strata):
            try:
                lo, hi, r = s
            except (TypeError, ValueError):
                raise ConfigError(f"audit strata[{i}] must be (lo, hi, rate)") from None
            if not (_is_number(lo) and _is_number(hi) and lo < hi):
                raise ConfigError(f"audit strata[{i}]: need numbers with lo < hi, got {s!r}")
            if not _is_number(r) or not 0 < r <= 1:
                raise ConfigError(f"audit strata[{i}]: rate must be in (0, 1], got {r!r}")
            strata.append((float(lo), float(hi), float(r)))
        object.__setattr__(self, "strata", tuple(strata))

    def inclusion_prob(self, confidence: float | None) -> float:
        value = self.rate
        if confidence is not None and not math.isnan(confidence):
            top = max(hi for _, hi, _ in self.strata) if self.strata else None
            for lo, hi, r in self.strata:
                closed = hi == top or hi >= 1.0
                if lo <= confidence < hi or (closed and confidence == hi):
                    value = r
                    break
        return max(self.floor, value)

    def selected(self, run_id: str, task_id: str, prob: float) -> bool:
        """Reproducible Bernoulli(prob) draw from sha256 of (seed, run_id, task_id)."""
        digest = hashlib.sha256(f"{self.seed}|{run_id}|{task_id}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2.0**64
        return u < prob


# --------------------------------------------------------------------------- cascade


class Cascade:
    """Routes tasks through tiers; see the module docstring and ``docs/design.md``.

    ``comparator`` grades answers against ``task.reference`` (no grading when None). ``judge``
    decides agreement between two model answers (shadow audits, eval mode); it defaults to
    ``comparator`` and, when both are None, to ``shadowgate.compare.from_spec(None)``
    (imported only when an agreement is actually needed).
    Instances hold no per-call mutable state and are safe to share across threads.
    """

    def __init__(
        self,
        tiers: Sequence[Tier],
        *,
        extractor: Extractor | None = None,
        comparator: Comparator | None = None,
        audit: AuditPolicy | None = None,
        judge: Comparator | None = None,
    ) -> None:
        tiers = tuple(tiers)
        if not tiers:
            raise ConfigError("a cascade needs at least one tier")
        seen: set[str] = set()
        for i, t in enumerate(tiers):
            if not isinstance(t, Tier):
                raise ConfigError(f"tiers[{i}] is not a Tier")
            if t.name in seen:
                raise ConfigError(f"duplicate tier name {t.name!r}")
            seen.add(t.name)
            if i < len(tiers) - 1:
                if t.estimator is None:
                    raise ConfigError(f"tier {t.name!r} is not final and needs an estimator")
                if t.threshold is None:
                    raise ConfigError(f"tier {t.name!r} is not final and needs a threshold")
        if audit is not None and audit.audit_tier is not None:
            if audit.audit_tier not in seen:
                raise ConfigError(f"audit tier {audit.audit_tier!r} is not one of {sorted(seen)}")
            if audit.audit_tier != tiers[-1].name:
                raise ConfigError(
                    f"audit tier {audit.audit_tier!r} must be the final tier {tiers[-1].name!r}: "
                    "a non-final audit tier would audit skipped cases with themselves or a "
                    "weaker tier"
                )

        if extractor is None and any(t.extractor is None for t in tiers):
            from shadowgate import extract

            extractor = extract.from_spec(None)
        if judge is None:
            judge = comparator

        self._tiers = tiers
        self._index = {t.name: i for i, t in enumerate(tiers)}
        self._extractor = extractor
        self._comparator = comparator
        self._judge = judge
        self._audit = audit

    # ----------------------------------------------------------------- accessors

    @property
    def tiers(self) -> tuple[Tier, ...]:
        return self._tiers

    @property
    def extractor(self) -> Extractor | None:
        return self._extractor

    @property
    def comparator(self) -> Comparator | None:
        return self._comparator

    @property
    def judge(self) -> Comparator | None:
        """The configured agreement judge (None -> resolved lazily to the default comparator)."""
        return self._judge

    def _agreement_judge(self) -> Comparator:
        if self._judge is not None:
            return self._judge
        from shadowgate import compare

        return compare.from_spec(None)

    @property
    def audit(self) -> AuditPolicy | None:
        return self._audit

    @property
    def audit_tier(self) -> Tier:
        name = self._audit.audit_tier if self._audit is not None else None
        return self._tiers[-1] if name is None else self._tiers[self._index[name]]

    # ----------------------------------------------------------------- routing

    def route(self, task: Task, *, run_id: str = "", mode: str = "serve") -> Decision:
        """Route one task. Model, estimator and comparator failures are recorded, not raised."""
        if mode not in ROUTE_MODES:
            raise ValueError(f"mode must be one of {ROUTE_MODES}, got {mode!r}")
        is_eval = mode == "eval"
        role = "eval" if is_eval else "serve"
        last = len(self._tiers) - 1

        attempts: list[Attempt] = []
        latencies: list[float] = []
        stop: int | None = None  # index of the tier the router serves from
        for i, tier in enumerate(self._tiers):
            att, lat = self._attempt(tier, task, role, final=i == last)
            attempts.append(att)
            latencies.append(lat)
            if att.accepted and stop is None:
                stop = i
                if not is_eval:
                    break

        n_serving = len(self._tiers) if stop is None else stop + 1
        error: str | None = None
        if stop is None:
            # Only possible when the final tier failed: serve the last available answer.
            fail = attempts[-1]
            error = f"final tier {fail.tier!r} failed: {fail.error}"
            served = next(
                (j for j in range(n_serving - 1, -1, -1) if attempts[j].error is None), None
            )
        else:
            served = stop

        # Eval mode: agreement of every non-final attempt with the final tier's answer.
        if is_eval and attempts[-1].error is None:
            ref = attempts[-1].answer
            judge = self._agreement_judge() if last > 0 else None
            for j in range(last):
                if judge is not None and attempts[j].error is None:
                    agreement = self._compare(judge, task, attempts[j].answer, ref)
                    attempts[j] = dataclasses.replace(attempts[j], agreement=agreement)

        attempts = [self._grade(task, a) for a in attempts]

        serving_cost = _sum_costs([self._attempt_cost(a) for a in attempts[:n_serving]])
        audit_costs: list[float | None] = [self._judge_cost(a) for a in attempts]
        audit_costs += [self._attempt_cost(a) for a in attempts[n_serving:]]

        shadow: ShadowResult | None = None
        if (
            not is_eval
            and stop is not None
            and stop < last
            and self._audit is not None
            and self._audit.mode != "off"
        ):
            conf = attempts[stop].confidence
            prob = self._audit.inclusion_prob(conf.score if conf is not None else None)
            audit_name = self.audit_tier.name
            if not self._audit.selected(run_id, task.id, prob):
                shadow = ShadowResult(audit_name, prob, "skipped")
            elif self._audit.mode == "deferred":
                shadow = ShadowResult(audit_name, prob, "pending")
            else:
                shadow = self._shadow(task, attempts[stop].answer, self.audit_tier, prob)
                audit_costs.append(self._shadow_cost(shadow))
                if shadow.agreement is not None:
                    attempts[stop] = dataclasses.replace(attempts[stop], agreement=shadow.agreement)

        if served is None:
            answer, final_tier, correct = "", self._tiers[-1].name, None
        else:
            answer = attempts[served].answer
            final_tier = attempts[served].tier
            correct = attempts[served].correct

        return Decision(
            run_id=run_id,
            task=task,
            answer=answer,
            final_tier=final_tier,
            escalated=n_serving > 1,
            attempts=tuple(attempts),
            cost_usd=serving_cost,
            audit_cost_usd=_sum_costs(audit_costs),
            latency_s=sum(latencies[:n_serving]),
            mode=mode,
            shadow=shadow,
            correct=correct,
            created_at=datetime.now(UTC).isoformat(),
            error=error,
        )

    def complete_audit(self, decision: Decision) -> Decision:
        """Run a deferred shadow audit. Non-pending decisions are returned unchanged.

        Only the audit tier is called (the serving tiers are never re-run); the returned
        Decision has the shadow filled in, the served attempt's ``agreement`` set and
        ``audit_cost_usd`` increased by the audit's cost.
        """
        shadow = decision.shadow
        if shadow is None or shadow.status != "pending":
            return decision
        if shadow.audit_tier not in self._index:
            raise ConfigError(f"audit tier {shadow.audit_tier!r} is not in this cascade")
        tier = self._tiers[self._index[shadow.audit_tier]]
        new_shadow = self._shadow(decision.task, decision.answer, tier, shadow.inclusion_prob)

        attempts = list(decision.attempts)
        if new_shadow.agreement is not None:
            for j in range(len(attempts) - 1, -1, -1):
                if attempts[j].tier == decision.final_tier and attempts[j].accepted:
                    attempts[j] = dataclasses.replace(attempts[j], agreement=new_shadow.agreement)
                    break
        return dataclasses.replace(
            decision,
            attempts=tuple(attempts),
            shadow=new_shadow,
            audit_cost_usd=_add(decision.audit_cost_usd, self._shadow_cost(new_shadow)),
        )

    # ----------------------------------------------------------------- internals

    def _attempt(self, tier: Tier, task: Task, role: str, *, final: bool) -> tuple[Attempt, float]:
        """Run one tier; returns the attempt and its serving latency in seconds."""
        estimator = None if final else tier.estimator
        threshold = None if final else tier.threshold
        request = Request(
            prompt=tier.render(task),
            system=tier.system,
            max_tokens=tier.max_tokens,
            temperature=tier.temperature,
            effort=tier.effort,
            tags={"task_id": task.id, "tier": tier.name, "role": role},
        )

        def failed(msg: str, elapsed: float) -> tuple[Attempt, float]:
            log.debug("tier %s failed on task %s: %s", tier.name, task.id, msg)
            att = Attempt(tier.name, tier.backend.name, None, "", None, threshold, False, error=msg)
            return att, elapsed

        t0 = time.perf_counter()
        if estimator is not None:
            try:
                request = estimator.prepare(request)
            except Exception as exc:
                return failed(f"estimator prepare failed: {_err(exc)}", time.perf_counter() - t0)
        try:
            completion = tier.backend.complete(request)
        except Exception as exc:  # BackendError, or a misbehaving custom backend
            return failed(_err(exc), time.perf_counter() - t0)

        extractor = tier.extractor or self._extractor
        assert extractor is not None  # guaranteed by __init__
        try:
            answer = extractor.extract(completion.text)
        except Exception as exc:
            log.warning("extractor %r raised on task %s: %s", extractor, task.id, exc)
            answer = ""

        confidence: ConfidenceResult | None = None
        if estimator is not None:
            est_name = getattr(estimator, "name", type(estimator).__name__)
            rejected = _unusable(completion, answer)
            if rejected is not None:
                confidence = ConfidenceResult(est_name, None, detail={"rejected": rejected})
            else:
                try:
                    confidence = estimator.estimate(task, request, completion, answer, tier.backend)
                except Exception as exc:
                    calls = _exc_calls(exc)
                    detail: dict[str, Any] = {"error": _err(exc)}
                    if calls is None:
                        # Calls made before the exception are lost: the cost is unknowable.
                        detail["cost_unknown"] = True
                    confidence = ConfidenceResult(est_name, None, calls or (), detail=detail)
            score = confidence.score
            accepted = (
                score is not None
                and threshold is not None
                and not math.isnan(score)
                and score >= threshold
            )
        else:
            accepted = True

        latency = completion.latency_s
        if confidence is not None:
            latency += sum(c.latency_s for c in confidence.calls)
        att = Attempt(
            tier=tier.name,
            backend=tier.backend.name,
            completion=completion,
            answer=answer,
            confidence=confidence,
            threshold=threshold,
            accepted=accepted,
        )
        return att, latency

    def _shadow(self, task: Task, served_answer: str, tier: Tier, prob: float) -> ShadowResult:
        att, _ = self._attempt(tier, task, "audit", final=True)
        if att.error is not None:
            return ShadowResult(tier.name, prob, "error", attempt=att)
        att = self._grade(task, att)
        agreement = self._compare(self._agreement_judge(), task, served_answer, att.answer)
        return ShadowResult(tier.name, prob, "done", attempt=att, agreement=agreement)

    def _grade(self, task: Task, att: Attempt) -> Attempt:
        if task.reference is None or self._comparator is None or att.error is not None:
            return att
        judgement = self._compare(self._comparator, task, att.answer, task.reference)
        return dataclasses.replace(att, correct=judgement)

    @staticmethod
    def _compare(comp: Comparator, task: Task, candidate: str, target: str) -> Judgement:
        try:
            return comp.compare(task, candidate, target)
        except Exception as exc:
            return Judgement(
                equivalent=None,
                comparator=getattr(comp, "name", type(comp).__name__),
                detail={"error": _err(exc)},
            )

    @staticmethod
    def _calls_cost(calls: Sequence[Completion]) -> float | None:
        return _sum_costs([c.cost_usd for c in calls])

    def _attempt_cost(self, att: Attempt) -> float | None:
        """Model call plus confidence calls. A failed call (no completion) counts as 0.

        None (unknown) when the estimator raised and its already-made calls were lost.
        """
        cost = 0.0 if att.completion is None else att.completion.cost_usd
        if att.confidence is not None:
            if att.confidence.detail.get("cost_unknown"):
                return None
            cost = _add(cost, self._calls_cost(att.confidence.calls))
        return cost

    def _judge_cost(self, att: Attempt) -> float | None:
        cost: float | None = 0.0
        for j in (att.correct, att.agreement):
            if j is not None:
                cost = _add(cost, self._calls_cost(j.calls))
        return cost

    def _shadow_cost(self, shadow: ShadowResult) -> float | None:
        cost: float | None = 0.0
        if shadow.attempt is not None:
            cost = _add(self._attempt_cost(shadow.attempt), self._judge_cost(shadow.attempt))
        if shadow.agreement is not None:
            cost = _add(cost, self._calls_cost(shadow.agreement.calls))
        return cost
