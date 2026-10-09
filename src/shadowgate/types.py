"""Core data types shared by every shadowgate module.

All records are frozen dataclasses so they can be logged, hashed and compared safely.
Every record has ``to_dict`` / ``from_dict`` for JSON round-trips (the ledger stores them).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Task",
    "Request",
    "Usage",
    "Completion",
    "Backend",
    "ConfidenceResult",
    "ConfidenceEstimator",
    "Judgement",
    "Comparator",
    "Attempt",
    "ShadowResult",
    "Decision",
]


def _clean(obj: Any) -> Any:
    """Recursively convert dataclasses/tuples into JSON-friendly structures."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _clean(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- tasks & requests


@dataclass(frozen=True)
class Task:
    """One unit of work to route.

    ``reference`` is the ground-truth answer when known (offline evaluation). It is never shown
    to a model. ``meta`` carries free-form fields (dataset name, difficulty, grader hints).
    """

    id: str
    prompt: str
    reference: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Task:
        ref = d.get("reference")
        return cls(
            id=str(d["id"]),
            prompt=str(d["prompt"]),
            reference=None if ref is None else str(ref),
            meta=dict(d.get("meta") or {}),
        )


@dataclass(frozen=True)
class Request:
    """A provider-neutral completion request.

    ``temperature`` None means "provider default" and must not be sent on the wire (several
    current models reject explicit sampling parameters). ``effort`` is passed through to
    providers that support it (Anthropic ``output_config.effort``) and ignored elsewhere.
    ``n_sample`` distinguishes otherwise-identical requests so caches do not collapse repeated
    samples (self-consistency): backends must include it in cache keys but never send it.
    ``extra`` holds provider-specific body parameters that ARE sent on the wire (merged into
    the request body by backends that support passthrough). ``tags`` is bookkeeping that is
    never sent and never part of a cache key (the cascade sets ``task_id``, ``tier``, ``role``).
    """

    prompt: str
    system: str | None = None
    max_tokens: int = 2048
    temperature: float | None = None
    effort: str | None = None
    want_logprobs: bool = False
    stop: tuple[str, ...] = ()
    n_sample: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Request:
        return cls(
            prompt=d["prompt"],
            system=d.get("system"),
            max_tokens=int(d.get("max_tokens", 2048)),
            temperature=d.get("temperature"),
            effort=d.get("effort"),
            want_logprobs=bool(d.get("want_logprobs", False)),
            stop=tuple(d.get("stop") or ()),
            n_sample=int(d.get("n_sample", 0)),
            extra=dict(d.get("extra") or {}),
            tags={str(k): str(v) for k, v in (d.get("tags") or {}).items()},
        )


# --------------------------------------------------------------------------- completions


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> Usage:
        d = d or {}
        return cls(
            int(d.get("input_tokens", 0)),
            int(d.get("output_tokens", 0)),
            int(d.get("cache_read_tokens", 0)),
            int(d.get("cache_write_tokens", 0)),
        )


@dataclass(frozen=True)
class Completion:
    """A model response plus its accounting.

    ``cost_usd`` is None when pricing for the model is unknown; aggregations treat None as
    "unknown" and report it, never as zero. ``logprobs`` holds per-output-token log
    probabilities when the backend returned them. ``cached`` is True when served from the
    local response cache (cost and latency still report the original call's values).
    ``stop_reason`` is the provider's normalized stop reason ("end", "max_tokens", "refusal",
    "stop_sequence", "error" or a provider-specific string).
    """

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    latency_s: float = 0.0
    stop_reason: str = "end"
    logprobs: tuple[float, ...] | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Completion:
        lp = d.get("logprobs")
        cost = d.get("cost_usd")
        return cls(
            text=d.get("text", ""),
            model=d.get("model", ""),
            usage=Usage.from_dict(d.get("usage")),
            cost_usd=None if cost is None else float(cost),
            latency_s=float(d.get("latency_s", 0.0)),
            stop_reason=d.get("stop_reason", "end"),
            logprobs=None if lp is None else tuple(float(x) for x in lp),
            cached=bool(d.get("cached", False)),
        )


@runtime_checkable
class Backend(Protocol):
    """Anything that turns a Request into a Completion.

    ``name`` is a stable identifier such as ``"anthropic:claude-haiku-5-5"``; it is part of
    cache keys and ledger records. Implementations must be thread-safe: the runner calls
    ``complete`` from a thread pool. Transient failures are retried inside the backend; a
    final failure raises ``shadowgate.errors.BackendError``.
    """

    name: str

    def complete(self, request: Request) -> Completion: ...


# --------------------------------------------------------------------------- confidence


@dataclass(frozen=True)
class ConfidenceResult:
    """A confidence score in [0, 1] (higher = more confident the answer is correct).

    ``calls`` lists any extra model calls the estimator made (resamples, a monitor model);
    their cost is charged to the routing decision. ``detail`` holds estimator-specific
    diagnostics (parsed verbal score, vote counts, monitor rationale ...). ``score`` may be
    None when the signal could not be computed (e.g. no parseable confidence); the cascade
    then treats the attempt as not confident (it escalates) and records why.
    """

    estimator: str
    score: float | None
    calls: tuple[Completion, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ConfidenceResult:
        s = d.get("score")
        return cls(
            estimator=d["estimator"],
            score=None if s is None else float(s),
            calls=tuple(Completion.from_dict(c) for c in d.get("calls") or ()),
            detail=dict(d.get("detail") or {}),
        )


@runtime_checkable
class ConfidenceEstimator(Protocol):
    """Scores how likely a tier's answer is to be correct.

    ``prepare`` may rewrite the tier's request before it is sent (e.g. ask the model to append
    a verbal confidence line). ``estimate`` runs after the completion arrives. ``backend`` is
    the tier's own backend (for resampling); estimators that need a separate model (a monitor)
    hold it themselves.
    """

    name: str

    def prepare(self, request: Request) -> Request: ...

    def estimate(
        self,
        task: Task,
        request: Request,
        completion: Completion,
        answer: str,
        backend: Backend,
    ) -> ConfidenceResult: ...


# --------------------------------------------------------------------------- comparison


@dataclass(frozen=True)
class Judgement:
    """Whether two answers (or an answer and a reference) are equivalent.

    ``equivalent`` is None when the comparator could not decide (e.g. an unparseable judge
    reply); such cases are counted separately, never silently as agree or disagree.
    """

    equivalent: bool | None
    comparator: str
    calls: tuple[Completion, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Judgement:
        return cls(
            equivalent=d.get("equivalent"),
            comparator=d["comparator"],
            calls=tuple(Completion.from_dict(c) for c in d.get("calls") or ()),
            detail=dict(d.get("detail") or {}),
        )


@runtime_checkable
class Comparator(Protocol):
    name: str

    def compare(self, task: Task, candidate: str, target: str) -> Judgement: ...


# --------------------------------------------------------------------------- routing records


@dataclass(frozen=True)
class Attempt:
    """One tier's try at a task inside a cascade.

    ``accepted`` is what the router decided at ``threshold`` (always True for a final tier that
    produced an answer). ``correct`` grades ``answer`` against ``task.reference`` when one
    exists. ``agreement`` compares ``answer`` with the final tier's answer;
    it is filled in eval mode for every non-final attempt, and for shadow-audited decisions.
    """

    tier: str
    backend: str
    completion: Completion | None
    answer: str
    confidence: ConfidenceResult | None
    threshold: float | None
    accepted: bool
    error: str | None = None
    correct: Judgement | None = None
    agreement: Judgement | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Attempt:
        c = d.get("completion")
        conf = d.get("confidence")
        thr = d.get("threshold")
        return cls(
            tier=d["tier"],
            backend=d["backend"],
            completion=None if c is None else Completion.from_dict(c),
            answer=d.get("answer", ""),
            confidence=None if conf is None else ConfidenceResult.from_dict(conf),
            threshold=None if thr is None else float(thr),
            accepted=bool(d["accepted"]),
            error=d.get("error"),
            correct=None if d.get("correct") is None else Judgement.from_dict(d["correct"]),
            agreement=None if d.get("agreement") is None else Judgement.from_dict(d["agreement"]),
        )


@dataclass(frozen=True)
class ShadowResult:
    """The shadow audit of a skipped case (an answer served by a non-final tier).

    The audit tier re-answers the task out of band. ``inclusion_prob`` is the probability with
    which this decision was selected for audit; estimators weight by 1/inclusion_prob
    (Horvitz-Thompson / Hajek) so confidence-stratified sampling stays unbiased.
    ``status`` is "done", "pending" (deferred mode, not yet run), "error", or "skipped" (not
    selected for audit; recorded so every skipped case's inclusion probability is known).
    """

    audit_tier: str
    inclusion_prob: float
    status: str
    attempt: Attempt | None = None
    agreement: Judgement | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ShadowResult:
        a = d.get("attempt")
        g = d.get("agreement")
        return cls(
            audit_tier=d["audit_tier"],
            inclusion_prob=float(d["inclusion_prob"]),
            status=d["status"],
            attempt=None if a is None else Attempt.from_dict(a),
            agreement=None if g is None else Judgement.from_dict(g),
        )


@dataclass(frozen=True)
class Decision:
    """The full record of routing one task. This is what the ledger stores.

    ``answer`` comes from the accepted tier (or the last tier that produced output).
    ``cost_usd`` is the serving cost: all attempts plus confidence-estimation calls, excluding
    shadow audit calls, which are reported separately as ``audit_cost_usd``. Costs are None if
    any contributing call had unknown pricing.
    ``correct`` is filled only when the task has a reference and a comparator was supplied.
    ``mode`` is "serve" (normal routing, sampled audits) or "eval" (every tier is run on every
    task so thresholds can be swept offline).
    """

    run_id: str
    task: Task
    answer: str
    final_tier: str
    escalated: bool
    attempts: tuple[Attempt, ...]
    cost_usd: float | None
    audit_cost_usd: float | None
    latency_s: float
    mode: str = "serve"
    shadow: ShadowResult | None = None
    correct: Judgement | None = None
    created_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Decision:
        sh = d.get("shadow")
        cr = d.get("correct")
        cost = d.get("cost_usd")
        acost = d.get("audit_cost_usd")
        return cls(
            run_id=d["run_id"],
            task=Task.from_dict(d["task"]),
            answer=d.get("answer", ""),
            final_tier=d["final_tier"],
            escalated=bool(d["escalated"]),
            attempts=tuple(Attempt.from_dict(a) for a in d.get("attempts") or ()),
            cost_usd=None if cost is None else float(cost),
            audit_cost_usd=None if acost is None else float(acost),
            latency_s=float(d.get("latency_s", 0.0)),
            mode=d.get("mode", "serve"),
            shadow=None if sh is None else ShadowResult.from_dict(sh),
            correct=None if cr is None else Judgement.from_dict(cr),
            created_at=d.get("created_at", ""),
            error=d.get("error"),
        )
