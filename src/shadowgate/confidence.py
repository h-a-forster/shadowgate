"""Confidence estimators: score how likely a tier's answer is to be correct.

Every estimator implements :class:`shadowgate.types.ConfidenceEstimator`. Estimators hold only
immutable configuration, so a single instance can be shared by many worker threads.

Families:

* :class:`Verbal` - the model states its own confidence on a ``CONFIDENCE:`` line.
* :class:`Logprob` - aggregate of output-token probabilities.
* :class:`SelfConsistency` - agreement of extra samples with the primary answer.
* :class:`Monitor` - a separate model judges the proposed answer.
* :class:`CallableEstimator` - wraps a Python function (API only).
* :class:`Combine` - mean / min / max / weighted mean of several estimators.
* :class:`Calibrated` - a monotone piecewise-linear map applied to another estimator.

Verbal parsing rules (shared by :class:`Verbal` and :class:`Monitor`): the **last** line of the
form ``CONFIDENCE: <value>`` (or ``P(correct): <value>`` for the monitor; case-insensitive,
markdown emphasis tolerated) is used. ``<value>`` may be ``0.85``, ``.85``, ``85%``, ``85/100``,
``8.5/10`` or ``8 out of 10``. A bare *integer* in ``(1, 100]`` is read as a percentage
(``85`` -> 0.85); a bare non-integer in ``(1, 100]`` (``1.5``, ``8.5``) is ambiguous between a
percentage and another scale and gives ``None``. ``0,85`` (a single ``0,<digits>`` decimal
comma) is read as 0.85; any other comma (``1,5``) gives ``None``. Anything larger, negative or
unparseable gives ``None``. The value may be followed only by closing punctuation and/or one
parenthetical remark (``0.7 (fairly sure)``); any other trailing text (``0.8 or 0.9``) gives
``None``. Ratios and percentages are clamped to [0, 1]. The words ``high`` / ``medium`` /
``low`` map to :data:`WORD_SCORES` only when ``allow_words=True``.

Every estimator returns ``score=None`` (``detail["reason"] == "empty answer"``) without
making any call when the primary answer is empty or whitespace.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from .errors import BackendError, ConfigError
from .types import (
    Backend,
    Comparator,
    Completion,
    ConfidenceEstimator,
    ConfidenceResult,
    Judgement,
    Request,
    Task,
)

__all__ = [
    "WORD_SCORES",
    "DEFAULT_VERBAL_INSTRUCTION",
    "DEFAULT_MONITOR_PROMPT",
    "parse_confidence",
    "parse_confidence_value",
    "Verbal",
    "Logprob",
    "SelfConsistency",
    "Monitor",
    "CallableEstimator",
    "Combine",
    "Calibrated",
    "from_spec",
]

log = logging.getLogger("shadowgate.confidence")

#: Scores used for verbal confidence words when ``allow_words=True``.
WORD_SCORES: Mapping[str, float] = {"high": 0.9, "medium": 0.6, "low": 0.3}

DEFAULT_VERBAL_INSTRUCTION = (
    "After your answer, add a final line stating how confident you are that the answer is "
    "correct, in exactly this form:\nCONFIDENCE: <number between 0 and 1>"
)

DEFAULT_MONITOR_PROMPT = """\
You are reviewing another model's answer. Estimate the probability that the proposed answer is \
correct.

Question:
{question}

Proposed answer:
{answer}

Check the answer briefly. Then give a calibrated probability: close to 1 only if you are sure \
it is correct, around 0.5 if you cannot tell, close to 0 if it is probably wrong. End your reply \
with a final line in exactly this form:
P(correct): <number between 0 and 1>"""


class _Extractor(Protocol):
    name: str

    def extract(self, text: str) -> str: ...


# --------------------------------------------------------------------------- parsing

_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)"
_PCT_RE = re.compile(rf"^({_NUM})\s*%")
_RATIO_RE = re.compile(rf"^({_NUM})\s*(?:/|\s+out\s+of\s+)\s*({_NUM})", re.IGNORECASE)
_COMMA_RE = re.compile(r"^0,(\d+)(?![\d.,])")
_PLAIN_RE = re.compile(rf"^({_NUM})(?!\d)")
_WORD_RE = re.compile(r"^(high|medium|low)\b", re.IGNORECASE)
# what may follow a value: closing punctuation and at most one parenthetical remark
_TAIL_RE = re.compile(r"[\s.!;,)\]]*(?:\([^()]*\)[\s.!;,]*)?")
_DECORATION = "*_`\"' \t"

_LABELS = {
    "confidence": r"confidence",
    "p_correct": r"p\s*\(\s*correct\s*\)",
}


def _label_regex(labels: Sequence[str]) -> re.Pattern[str]:
    alt = "|".join(_LABELS[label] for label in labels)
    return re.compile(
        rf"^[\s>#*_`-]*(?:{alt})[\s*_`]*"
        # ":" or "=", or a dash followed by a number or a word level ("Confidence - 0.85")
        r"(?:[:=]|[-–—](?=[\s*_`]*(?:[\d.]|(?:very\s+)?(?:high|medium|low)\b)))"
        r"\s*(.*?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )


_VERBAL_LINE = _label_regex(["confidence"])
_MONITOR_LINE = _label_regex(["p_correct", "confidence"])


def parse_confidence_value(raw: str, *, allow_words: bool = False) -> float | None:
    """Parse the value part of a confidence line. Returns a score in [0, 1] or None.

    See the module docstring for the accepted forms and the ambiguity rules.
    """
    s = raw.strip().strip(_DECORATION).strip()
    if not s:
        return None

    def tail_ok(m: re.Match[str]) -> bool:
        return _TAIL_RE.fullmatch(s[m.end() :]) is not None

    m = _PCT_RE.match(s)
    if m:
        return _clamp(float(m.group(1)) / 100.0) if tail_ok(m) else None
    m = _RATIO_RE.match(s)
    if m:
        den = float(m.group(2))
        if den <= 0 or not tail_ok(m):
            return None
        return _clamp(float(m.group(1)) / den)
    m = _COMMA_RE.match(s)
    if m:
        return float("0." + m.group(1)) if tail_ok(m) else None
    m = _PLAIN_RE.match(s)
    if m:
        if not tail_ok(m):
            return None
        x = float(m.group(1))
        if x <= 1.0:
            return x
        if x <= 100.0 and "." not in m.group(1):
            return x / 100.0
        return None  # > 100, or a non-integer in (1, 100]: ambiguous scale
    if allow_words:
        m = _WORD_RE.match(s)
        if m and tail_ok(m):
            return WORD_SCORES[m.group(1).lower()]
    return None


def parse_confidence(
    text: str, *, allow_words: bool = False, monitor: bool = False
) -> tuple[float | None, str | None]:
    """Find the last confidence line in ``text`` and parse it.

    Returns ``(score, raw)`` where ``raw`` is the text after the label (None when no line was
    found). ``monitor=True`` also accepts ``P(correct):`` lines.
    """
    pattern = _MONITOR_LINE if monitor else _VERBAL_LINE
    matches = pattern.findall(text or "")
    if not matches:
        return None, None
    raw = matches[-1]
    return parse_confidence_value(raw, allow_words=allow_words), raw


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Substitute ``{name}`` placeholders in one pass, without interpreting other braces.

    Substituted values are never rescanned, so e.g. ``{answer}`` inside the task prompt stays
    literal.
    """
    if not values:
        return template
    pattern = re.compile(r"\{(" + "|".join(re.escape(k) for k in values) + r")\}")
    return pattern.sub(lambda m: values[m.group(1)], template)


def _is_empty(answer: str | None) -> bool:
    return not (answer or "").strip()


def _empty_result(name: str) -> ConfidenceResult:
    return ConfidenceResult(name, None, detail={"reason": "empty answer"})


def _confidence_tags(request: Request) -> dict[str, str]:
    tags = dict(request.tags)
    tags["role"] = "confidence"
    return tags


# --------------------------------------------------------------------------- estimators


class Verbal:
    """Ask the model to end with ``CONFIDENCE: <p>`` and parse it."""

    def __init__(self, *, instruction: str | None = None, allow_words: bool = False) -> None:
        if instruction is not None and "confidence:" not in instruction.lower():
            raise ConfigError("verbal: 'instruction' must ask for a 'CONFIDENCE:' line")
        self.instruction = instruction or DEFAULT_VERBAL_INSTRUCTION
        self.allow_words = allow_words
        self.name = "verbal"

    def prepare(self, request: Request) -> Request:
        if self.instruction in request.prompt:
            return request
        prompt = request.prompt.rstrip("\n") + "\n\n" + self.instruction
        return dataclasses.replace(request, prompt=prompt)

    def estimate(
        self, task: Task, request: Request, completion: Completion, answer: str,
        backend: Backend,
    ) -> ConfidenceResult:
        if _is_empty(answer):
            return _empty_result(self.name)
        score, raw = parse_confidence(completion.text, allow_words=self.allow_words)
        detail: dict[str, Any] = {"raw": raw}
        if raw is None:
            detail["reason"] = "no CONFIDENCE line"
        elif score is None:
            detail["reason"] = "unparseable CONFIDENCE value"
        return ConfidenceResult(self.name, score, detail=detail)


_AGGREGATES = ("mean", "min", "geo_mean")


class Logprob:
    """Aggregate output-token probabilities (``mean``, ``min`` or ``geo_mean``)."""

    def __init__(self, *, aggregate: str = "mean") -> None:
        if aggregate not in _AGGREGATES:
            raise ConfigError(
                f"logprob: unknown aggregate {aggregate!r}; expected one of {list(_AGGREGATES)}"
            )
        self.aggregate = aggregate
        self.name = f"logprob({aggregate})"

    def prepare(self, request: Request) -> Request:
        if request.want_logprobs:
            return request
        return dataclasses.replace(request, want_logprobs=True)

    def estimate(
        self, task: Task, request: Request, completion: Completion, answer: str,
        backend: Backend,
    ) -> ConfidenceResult:
        if _is_empty(answer):
            return _empty_result(self.name)
        lps = completion.logprobs
        if not lps:
            return ConfidenceResult(
                self.name, None, detail={"reason": "completion has no logprobs", "n_tokens": 0}
            )
        vals = [float(x) for x in lps]
        if any(math.isnan(x) for x in vals):
            return ConfidenceResult(
                self.name, None, detail={"reason": "NaN logprob", "n_tokens": len(vals)}
            )
        if self.aggregate == "mean":
            score = sum(math.exp(x) for x in vals) / len(vals)
        elif self.aggregate == "min":
            score = math.exp(min(vals))
        else:
            score = math.exp(sum(vals) / len(vals))
        return ConfidenceResult(
            self.name, _clamp(score), detail={"n_tokens": len(vals), "aggregate": self.aggregate}
        )


class SelfConsistency:
    """Resample the tier backend ``samples`` times; score = agreement with the primary answer.

    ``score = (1 + agree) / (1 + successful_samples)``. Undecided comparisons
    (``equivalent=None``) count as disagreement and are reported in ``detail["undecided"]``.
    A sample whose extracted answer is empty counts as disagreeing without calling the
    comparator (``detail["empty"]``). Samples whose backend call raises (any exception) are
    skipped and counted in ``detail["failed"]``; calls from the other samples are kept. If
    every sample fails the score is None. A comparator that raises counts as undecided.
    """

    def __init__(
        self,
        *,
        comparator: Comparator,
        extractor: _Extractor,
        samples: int = 5,
        temperature: float | None = None,
        concurrency: int = 1,
    ) -> None:
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ConfigError("self_consistency: 'samples' must be an integer >= 1")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ConfigError("self_consistency: 'concurrency' must be an integer >= 1")
        self.comparator = comparator
        self.extractor = extractor
        self.samples = samples
        self.temperature = temperature
        self.concurrency = concurrency
        self.name = f"self_consistency(k={samples})"

    def prepare(self, request: Request) -> Request:
        return request

    def _sample(self, backend: Backend, request: Request, i: int) -> Completion | Exception:
        temp = self.temperature if self.temperature is not None else request.temperature
        req = dataclasses.replace(
            request, n_sample=i, temperature=temp, tags=_confidence_tags(request)
        )
        try:
            return backend.complete(req)
        except BackendError as exc:
            return exc
        except Exception as exc:  # a failed sample must not discard the other samples' calls
            log.warning("self_consistency sample %d failed: %r", i, exc)
            return exc

    def estimate(
        self, task: Task, request: Request, completion: Completion, answer: str,
        backend: Backend,
    ) -> ConfidenceResult:
        if _is_empty(answer):
            return _empty_result(self.name)
        idx = range(1, self.samples + 1)
        if self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, self.samples)) as pool:
                results = list(pool.map(lambda i: self._sample(backend, request, i), idx))
        else:
            results = [self._sample(backend, request, i) for i in idx]

        calls: list[Completion] = []
        answers: list[str] = []
        votes: list[bool | None] = []
        errors: list[str] = []
        agree = undecided = empty = 0
        for res in results:
            if isinstance(res, BackendError):
                errors.append(str(res))
                continue
            if isinstance(res, Exception):
                errors.append(f"{type(res).__name__}: {res}")
                continue
            calls.append(res)
            sample_answer = self.extractor.extract(res.text)
            answers.append(sample_answer)
            if _is_empty(sample_answer):
                empty += 1
                votes.append(False)
                continue
            try:
                judgement: Judgement = self.comparator.compare(task, sample_answer, answer)
            except Exception as exc:
                errors.append(f"comparator {type(exc).__name__}: {exc}")
                undecided += 1
                votes.append(None)
                continue
            calls.extend(judgement.calls)
            votes.append(judgement.equivalent)
            if judgement.equivalent is True:
                agree += 1
            elif judgement.equivalent is None:
                undecided += 1

        ok = len(answers)
        detail: dict[str, Any] = {
            "samples": self.samples,
            "successful": ok,
            "failed": self.samples - ok,
            "agree": agree,
            "undecided": undecided,
            "empty": empty,
            "answers": answers,
            "votes": votes,
            "comparator": self.comparator.name,
        }
        if errors:
            detail["errors"] = errors
        if ok == 0:
            detail["reason"] = "all samples failed"
            return ConfidenceResult(self.name, None, calls=tuple(calls), detail=detail)
        return ConfidenceResult(
            self.name, (1 + agree) / (1 + ok), calls=tuple(calls), detail=detail
        )


class Monitor:
    """A separate model reads the question and the proposed answer and states ``P(correct)``.

    The prompt template may use ``{question}`` (the task prompt), ``{answer}`` (the extracted
    answer) and ``{response}`` (the full tier response text).
    """

    def __init__(
        self,
        backend: Backend,
        *,
        prompt: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        effort: str | None = None,
        allow_words: bool = False,
    ) -> None:
        if prompt is not None and "{answer}" not in prompt and "{response}" not in prompt:
            raise ConfigError("monitor: 'prompt' must contain {answer} or {response}")
        self.backend = backend
        self.prompt = prompt or DEFAULT_MONITOR_PROMPT
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.effort = effort
        self.allow_words = allow_words
        self.name = f"monitor({backend.name})"

    def prepare(self, request: Request) -> Request:
        return request

    def estimate(
        self, task: Task, request: Request, completion: Completion, answer: str,
        backend: Backend,
    ) -> ConfidenceResult:
        if _is_empty(answer):
            return _empty_result(self.name)
        text = _fill(
            self.prompt,
            {"question": task.prompt, "answer": answer, "response": completion.text},
        )
        req = Request(
            prompt=text,
            system=self.system,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            effort=self.effort,
            tags=_confidence_tags(request),
        )
        try:
            reply = self.backend.complete(req)
        except BackendError as exc:
            return ConfidenceResult(
                self.name, None, detail={"error": str(exc), "backend": self.backend.name}
            )
        except Exception as exc:  # unexpected backend failure: no reply, so no call to keep
            log.warning("monitor backend %s failed: %r", self.backend.name, exc)
            return ConfidenceResult(
                self.name,
                None,
                detail={"error": f"{type(exc).__name__}: {exc}", "backend": self.backend.name},
            )
        try:
            score, raw = parse_confidence(reply.text, allow_words=self.allow_words, monitor=True)
        except Exception as exc:  # e.g. a malformed reply object; keep the paid call
            return ConfidenceResult(
                self.name,
                None,
                calls=(reply,),
                detail={"error": f"{type(exc).__name__}: {exc}", "backend": self.backend.name},
            )
        detail: dict[str, Any] = {"raw": raw, "backend": self.backend.name}
        if raw is None:
            detail["reason"] = "no P(correct) line"
        elif score is None:
            detail["reason"] = "unparseable P(correct) value"
        return ConfidenceResult(self.name, score, calls=(reply,), detail=detail)


class CallableEstimator:
    """Wrap ``fn(task, completion, answer) -> float | None``.

    Results are clamped to [0, 1]; NaN or a non-numeric return gives None. Exceptions raised
    by ``fn`` propagate (they indicate a bug in the wrapped function).
    """

    def __init__(
        self,
        fn: Callable[[Task, Completion, str], float | None],
        *,
        name: str | None = None,
    ) -> None:
        if not callable(fn):
            raise ConfigError("callable: 'fn' must be callable")
        self.fn = fn
        self.name = name or f"callable({getattr(fn, '__name__', 'fn')})"

    def prepare(self, request: Request) -> Request:
        return request

    def estimate(
        self, task: Task, request: Request, completion: Completion, answer: str,
        backend: Backend,
    ) -> ConfidenceResult:
        if _is_empty(answer):
            return _empty_result(self.name)
        value = self.fn(task, completion, answer)
        if value is None:
            return ConfidenceResult(self.name, None, detail={"reason": "function returned None"})
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return ConfidenceResult(
                self.name, None, detail={"reason": f"non-numeric result {type(value).__name__}"}
            )
        x = float(value)
        if math.isnan(x):
            return ConfidenceResult(self.name, None, detail={"reason": "function returned NaN"})
        detail: dict[str, Any] = {}
        if x != _clamp(x):
            detail["raw"] = x
        return ConfidenceResult(self.name, _clamp(x), detail=detail)


_METHODS = ("mean", "min", "max", "weighted")


class Combine:
    """Combine member estimators with ``mean`` / ``min`` / ``max`` / ``weighted``.

    Any member None -> combined None, unless ``ignore_missing=True`` in which case missing
    members are dropped (weights renormalised over the rest); all missing -> None.

    A member that raises counts as a missing score (its error is in that member's
    ``detail["members"][i]["error"]``); calls made by the other members are kept.
    """

    def __init__(
        self,
        members: Sequence[ConfidenceEstimator],
        *,
        method: str = "mean",
        weights: Sequence[float] | None = None,
        ignore_missing: bool = False,
    ) -> None:
        members = tuple(members)
        if not members:
            raise ConfigError("combine: 'members' must not be empty")
        if method not in _METHODS:
            raise ConfigError(f"combine: unknown method {method!r}; expected one of {_METHODS}")
        norm: tuple[float, ...] | None = None
        if method == "weighted":
            if weights is None:
                raise ConfigError("combine: method 'weighted' requires 'weights'")
            if len(weights) != len(members):
                raise ConfigError(
                    f"combine: {len(weights)} weights for {len(members)} members"
                )
            ws = [float(w) for w in weights]
            if any(w < 0 or math.isnan(w) for w in ws) or sum(ws) <= 0:
                raise ConfigError("combine: weights must be non-negative with a positive sum")
            total = sum(ws)
            norm = tuple(w / total for w in ws)
        elif weights is not None:
            raise ConfigError("combine: 'weights' is only valid with method 'weighted'")
        self.members = members
        self.method = method
        self.weights = norm
        self.ignore_missing = ignore_missing
        self.name = f"combine({method}: {', '.join(m.name for m in members)})"

    def prepare(self, request: Request) -> Request:
        for m in self.members:
            request = m.prepare(request)
        return request

    def estimate(
        self, task: Task, request: Request, completion: Completion, answer: str,
        backend: Backend,
    ) -> ConfidenceResult:
        if _is_empty(answer):
            return _empty_result(self.name)
        results: list[ConfidenceResult] = []
        member_detail: list[dict[str, Any]] = []
        for m in self.members:
            try:
                r = m.estimate(task, request, completion, answer, backend)
            except Exception as exc:  # keep the calls already made by the other members
                log.warning("combine member %s failed: %r", m.name, exc)
                r = ConfidenceResult(m.name, None)
                error = f"{type(exc).__name__}: {exc}"
                member_detail.append({"estimator": m.name, "score": None, "error": error})
            else:
                member_detail.append({"estimator": r.estimator, "score": r.score})
            results.append(r)
        calls = tuple(c for r in results for c in r.calls)
        detail: dict[str, Any] = {"method": self.method, "members": member_detail}
        if self.weights is not None:
            detail["weights"] = list(self.weights)
        present = [
            (r.score, self.weights[i] if self.weights else 1.0)
            for i, r in enumerate(results)
            if r.score is not None
        ]
        missing = len(results) - len(present)
        if missing and not self.ignore_missing:
            detail["reason"] = f"{missing} member(s) returned no score"
            return ConfidenceResult(self.name, None, calls=calls, detail=detail)
        if not present:
            detail["reason"] = "no member returned a score"
            return ConfidenceResult(self.name, None, calls=calls, detail=detail)
        scores = [s for s, _ in present]
        if self.method == "min":
            score = min(scores)
        elif self.method == "max":
            score = max(scores)
        elif self.method == "mean":
            score = sum(scores) / len(scores)
        else:
            wsum = sum(w for _, w in present)
            if wsum <= 0:
                detail["reason"] = "present members have zero total weight"
                return ConfidenceResult(self.name, None, calls=calls, detail=detail)
            score = sum(s * w for s, w in present) / wsum
        return ConfidenceResult(self.name, _clamp(score), calls=calls, detail=detail)


class Calibrated:
    """Apply a monotone piecewise-linear map (``points=[(x, y), ...]``) to another estimator.

    ``x`` must be strictly increasing, ``y`` non-decreasing and within [0, 1]. Scores outside
    the ``x`` range are clamped to the first / last ``y``.
    """

    def __init__(
        self, base: ConfidenceEstimator, points: Sequence[Sequence[float]]
    ) -> None:
        pts: list[tuple[float, float]] = []
        for p in points:
            if len(p) != 2:
                raise ConfigError("calibrated: each point must be [x, y]")
            x, y = float(p[0]), float(p[1])
            if math.isnan(x) or math.isnan(y):
                raise ConfigError("calibrated: points must not contain NaN")
            if not 0.0 <= y <= 1.0:
                raise ConfigError(f"calibrated: y={y} is outside [0, 1]")
            pts.append((x, y))
        if not pts:
            raise ConfigError("calibrated: 'points' must not be empty")
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            if x1 <= x0:
                raise ConfigError("calibrated: point x values must be strictly increasing")
            if y1 < y0:
                raise ConfigError("calibrated: point y values must be non-decreasing")
        self.base = base
        self.points = tuple(pts)
        self.name = f"calibrated({base.name})"

    def apply(self, x: float) -> float:
        pts = self.points
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]  # pragma: no cover - unreachable for sorted points

    def prepare(self, request: Request) -> Request:
        return self.base.prepare(request)

    def estimate(
        self, task: Task, request: Request, completion: Completion, answer: str,
        backend: Backend,
    ) -> ConfidenceResult:
        if _is_empty(answer):
            return _empty_result(self.name)
        inner = self.base.estimate(task, request, completion, answer, backend)
        detail: dict[str, Any] = {"base": inner.estimator, "raw_score": inner.score,
                                  "base_detail": dict(inner.detail)}
        score = None if inner.score is None else self.apply(inner.score)
        return ConfidenceResult(self.name, score, calls=inner.calls, detail=detail)


# --------------------------------------------------------------------------- factory

_ALLOWED_KEYS: Mapping[str, frozenset[str]] = {
    "verbal": frozenset({"instruction", "allow_words"}),
    "logprob": frozenset({"aggregate"}),
    "self_consistency": frozenset({"samples", "temperature", "concurrency"}),
    "monitor": frozenset(
        {"backend", "prompt", "system", "max_tokens", "temperature", "effort", "allow_words"}
    ),
    "callable": frozenset({"fn", "name"}),
    "combine": frozenset({"members", "method", "weights", "ignore_missing"}),
    "calibrated": frozenset({"base", "points"}),
}


def from_spec(
    spec: Mapping[str, Any],
    *,
    backends: Mapping[str, Backend],
    comparator: Comparator,
    extractor: _Extractor,
) -> ConfidenceEstimator:
    """Build an estimator from a config table keyed by ``spec["type"]``.

    ``callable`` is API-only: its ``fn`` must be a Python callable. ``calibrated`` takes the
    wrapped estimator's spec as ``base``; ``combine`` takes member specs as ``members``.
    """
    if not isinstance(spec, Mapping):
        raise ConfigError(f"confidence spec must be a table, got {type(spec).__name__}")
    kind = spec.get("type")
    if not isinstance(kind, str) or kind not in _ALLOWED_KEYS:
        raise ConfigError(
            f"unknown confidence type {kind!r}; expected one of {sorted(_ALLOWED_KEYS)}"
        )
    unknown = sorted(set(spec) - {"type"} - _ALLOWED_KEYS[kind])
    if unknown:
        raise ConfigError(f"confidence type {kind!r}: unknown key(s) {unknown}")
    opts = {k: v for k, v in spec.items() if k != "type"}

    def sub(s: Any) -> ConfidenceEstimator:
        return from_spec(s, backends=backends, comparator=comparator, extractor=extractor)

    if kind == "verbal":
        return Verbal(**opts)
    if kind == "logprob":
        return Logprob(**opts)
    if kind == "self_consistency":
        return SelfConsistency(comparator=comparator, extractor=extractor, **opts)
    if kind == "monitor":
        name = opts.pop("backend", None)
        if name is None:
            raise ConfigError("confidence type 'monitor': missing key 'backend'")
        if name not in backends:
            raise ConfigError(
                f"confidence type 'monitor': unknown backend {name!r}; "
                f"available: {sorted(backends)}"
            )
        return Monitor(backends[name], **opts)
    if kind == "callable":
        if "fn" not in opts:
            raise ConfigError("confidence type 'callable': missing key 'fn' (API only)")
        return CallableEstimator(opts["fn"], name=opts.get("name"))
    if kind == "combine":
        members = opts.pop("members", None)
        if not isinstance(members, Sequence) or isinstance(members, str) or not members:
            raise ConfigError("confidence type 'combine': 'members' must be a non-empty list")
        return Combine([sub(m) for m in members], **opts)
    # calibrated
    if "base" not in opts or "points" not in opts:
        raise ConfigError("confidence type 'calibrated': requires 'base' and 'points'")
    return Calibrated(sub(opts["base"]), opts["points"])
