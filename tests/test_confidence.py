from __future__ import annotations

import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

from shadowgate import confidence as conf
from shadowgate.errors import BackendError, ConfigError
from shadowgate.types import (
    Completion,
    ConfidenceEstimator,
    ConfidenceResult,
    Judgement,
    Request,
    Task,
)

# --------------------------------------------------------------------------- fakes


class FakeBackend:
    """Thread-safe fake: ``fn(request) -> str`` (or raises); records every request."""

    def __init__(self, fn: Callable[[Request], str] | str, name: str = "fake:model") -> None:
        self.name = name
        self._fn = fn if callable(fn) else (lambda _r, _t=fn: _t)
        self._lock = threading.Lock()
        self.requests: list[Request] = []

    def complete(self, request: Request) -> Completion:
        with self._lock:
            self.requests.append(request)
        text = self._fn(request)
        return Completion(text=text, model=self.name, cost_usd=0.001)


class FakeExtractor:
    name = "fake_extract"

    def extract(self, text: str) -> str:
        for line in reversed(text.splitlines()):
            if line.startswith("ANSWER:"):
                return line[len("ANSWER:"):].strip()
        return ""


class FakeComparator:
    """Equal strings agree; candidate "?" is undecided; optional judge call attached."""

    name = "fake_compare"

    def __init__(self, with_call: bool = False) -> None:
        self.with_call = with_call

    def compare(self, task: Task, candidate: str, target: str) -> Judgement:
        calls = (Completion(text="judge", model="judge"),) if self.with_call else ()
        if candidate == "?":
            return Judgement(None, self.name, calls=calls)
        return Judgement(candidate == target, self.name, calls=calls)


class Fixed:
    """Estimator returning a fixed score, optionally rewriting prompts."""

    def __init__(self, score: float | None, name: str = "fixed", suffix: str = "") -> None:
        self.score = score
        self.name = name
        self.suffix = suffix

    def prepare(self, request: Request) -> Request:
        return Request(prompt=request.prompt + self.suffix)

    def estimate(self, task, request, completion, answer, backend) -> ConfidenceResult:
        calls = (Completion(text=self.name, model="m"),)
        return ConfidenceResult(self.name, self.score, calls=calls)


TASK = Task(id="t1", prompt="What is 2+2?", reference="4")
REQ = Request(prompt="What is 2+2?", temperature=0.2, tags={"task_id": "t1", "tier": "fast"})
NOBACKEND = FakeBackend("unused")


def comp(text: str, logprobs: tuple[float, ...] | None = None) -> Completion:
    return Completion(text=text, model="m", logprobs=logprobs)


def build(spec, backends=None) -> ConfidenceEstimator:
    return conf.from_spec(
        spec, backends=backends or {}, comparator=FakeComparator(), extractor=FakeExtractor()
    )


# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.85", 0.85),
        (".85", 0.85),
        ("85%", 0.85),
        ("85 %", 0.85),
        ("85/100", 0.85),
        ("8.5/10", 0.85),
        ("1", 1.0),
        ("0", 0.0),
        ("85", 0.85),
        ("100", 1.0),
        ("**0.7**", 0.7),
        ("`0.7`", 0.7),
        ("0.7.", 0.7),
        ("0.7 (fairly sure)", 0.7),
        ("150%", 1.0),
        ("12/10", 1.0),
        ("150", None),
        ("-0.2", None),
        ("5/0", None),
        ("", None),
        ("unsure", None),
        ("high", None),
        # regressions: scales and trailing text
        ("8 out of 10", 0.8),
        ("8.5 out of 10.", 0.85),
        ("3 Out Of 4 (decent)", 0.75),
        ("0,85", 0.85),  # a lone 0,<digits> decimal comma
        ("1,5", None),
        ("0,8,5", None),
        ("1.5", None),  # non-integer in (1, 100]: ambiguous scale
        ("8.5", None),
        ("85", 0.85),
        ("2", 0.02),  # bare integer in (1, 100] stays a percentage
        ("0.8 or 0.9", None),
        ("0.8 maybe", None),
        ("85% but unsure", None),
        ("8/10 points", None),
        ("0.7, (fairly sure).", 0.7),
    ],
)
def test_parse_value(raw: str, expected: float | None) -> None:
    got = conf.parse_confidence_value(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_parse_words_opt_in() -> None:
    assert conf.parse_confidence_value("high") is None
    assert conf.parse_confidence_value("High", allow_words=True) == conf.WORD_SCORES["high"]
    assert conf.parse_confidence_value("medium", allow_words=True) == conf.WORD_SCORES["medium"]
    assert conf.parse_confidence_value("low.", allow_words=True) == conf.WORD_SCORES["low"]


def test_parse_last_line_wins_and_markdown() -> None:
    text = "CONFIDENCE: 0.2\nreasoning\n**Confidence:** 0.9"
    assert conf.parse_confidence(text) == (0.9, "** 0.9")
    assert conf.parse_confidence("no line here") == (None, None)
    assert conf.parse_confidence("confidence = 70%")[0] == pytest.approx(0.7)


def test_parse_monitor_labels() -> None:
    assert conf.parse_confidence("P(correct): 0.3", monitor=True)[0] == pytest.approx(0.3)
    assert conf.parse_confidence("p( correct ) = 30%", monitor=True)[0] == pytest.approx(0.3)
    assert conf.parse_confidence("CONFIDENCE: 0.4", monitor=True)[0] == pytest.approx(0.4)
    assert conf.parse_confidence("P(correct): 0.3")[0] is None  # verbal ignores P(correct)


# --------------------------------------------------------------------------- verbal


def test_verbal_prepare_appends_and_is_idempotent() -> None:
    est = conf.Verbal()
    once = est.prepare(REQ)
    assert once.prompt.startswith(REQ.prompt)
    assert "CONFIDENCE:" in once.prompt
    assert once.tags == REQ.tags and once.temperature == REQ.temperature
    assert est.prepare(once) == once
    assert REQ.prompt == "What is 2+2?"


def test_verbal_custom_instruction() -> None:
    est = conf.Verbal(instruction="Finish with CONFIDENCE: p")
    assert est.prepare(REQ).prompt.endswith("Finish with CONFIDENCE: p")
    with pytest.raises(ConfigError):
        conf.Verbal(instruction="say how sure you are")


def test_verbal_estimate() -> None:
    est = conf.Verbal()
    assert est.name == "verbal"
    r = est.estimate(TASK, REQ, comp("ANSWER: 4\nCONFIDENCE: 85%"), "4", NOBACKEND)
    assert r.score == pytest.approx(0.85)
    assert r.detail["raw"] == "85%"
    assert r.calls == ()
    missing = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", NOBACKEND)
    assert missing.score is None and missing.detail["raw"] is None
    bad = est.estimate(TASK, REQ, comp("CONFIDENCE: very"), "4", NOBACKEND)
    assert bad.score is None and bad.detail["raw"] == "very"
    word = est.estimate(TASK, REQ, comp("CONFIDENCE: high"), "4", NOBACKEND)
    assert word.score is None
    worded = conf.Verbal(allow_words=True)
    assert worded.estimate(TASK, REQ, comp("CONFIDENCE: high"), "4", NOBACKEND).score == 0.9


# --------------------------------------------------------------------------- logprob


def test_logprob_aggregates() -> None:
    lps = (math.log(0.9), math.log(0.5), math.log(0.8))
    c = comp("x", lps)
    mean = conf.Logprob().estimate(TASK, REQ, c, "x", NOBACKEND)
    assert mean.score == pytest.approx((0.9 + 0.5 + 0.8) / 3)
    assert mean.detail["n_tokens"] == 3
    assert conf.Logprob(aggregate="min").estimate(TASK, REQ, c, "x", NOBACKEND).score == (
        pytest.approx(0.5)
    )
    geo = conf.Logprob(aggregate="geo_mean").estimate(TASK, REQ, c, "x", NOBACKEND)
    assert geo.score == pytest.approx((0.9 * 0.5 * 0.8) ** (1 / 3))
    assert conf.Logprob(aggregate="min").name == "logprob(min)"


def test_logprob_missing_and_prepare() -> None:
    est = conf.Logprob()
    assert est.estimate(TASK, REQ, comp("x"), "x", NOBACKEND).score is None
    assert est.estimate(TASK, REQ, comp("x", ()), "x", NOBACKEND).score is None
    prepared = est.prepare(REQ)
    assert prepared.want_logprobs and prepared.prompt == REQ.prompt
    assert est.prepare(prepared) is prepared
    with pytest.raises(ConfigError):
        conf.Logprob(aggregate="median")


# --------------------------------------------------------------------------- self-consistency


def test_self_consistency_scores_and_requests() -> None:
    answers = {1: "4", 2: "5", 3: "4", 4: "?"}
    backend = FakeBackend(lambda r: f"ANSWER: {answers[r.n_sample]}")
    est = conf.SelfConsistency(
        comparator=FakeComparator(), extractor=FakeExtractor(), samples=4
    )
    assert est.name == "self_consistency(k=4)"
    assert est.prepare(REQ) is REQ
    r = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", backend)
    assert r.score == pytest.approx((1 + 2) / (1 + 4))
    assert r.detail["agree"] == 2 and r.detail["undecided"] == 1
    assert r.detail["successful"] == 4 and r.detail["failed"] == 0
    assert r.detail["answers"] == ["4", "5", "4", "?"]
    assert len(r.calls) == 4
    assert [q.n_sample for q in backend.requests] == [1, 2, 3, 4]
    for q in backend.requests:
        assert q.temperature == 0.2  # inherits request temperature
        assert q.tags["role"] == "confidence" and q.tags["task_id"] == "t1"
        assert q.prompt == REQ.prompt
    assert REQ.tags.get("role") is None


def test_self_consistency_temperature_override_and_judge_calls() -> None:
    backend = FakeBackend("ANSWER: 4")
    est = conf.SelfConsistency(
        comparator=FakeComparator(with_call=True), extractor=FakeExtractor(), samples=2,
        temperature=1.0,
    )
    r = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", backend)
    assert r.score == 1.0
    assert all(q.temperature == 1.0 for q in backend.requests)
    assert len(r.calls) == 4  # 2 samples + 2 judge calls


def test_self_consistency_tolerates_failures() -> None:
    def fn(r: Request) -> str:
        if r.n_sample == 2:
            raise BackendError("boom", backend="fake")
        return "ANSWER: 4"

    est = conf.SelfConsistency(comparator=FakeComparator(), extractor=FakeExtractor(), samples=3)
    r = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", FakeBackend(fn))
    assert r.score == pytest.approx(3 / 3)
    assert r.detail["failed"] == 1 and r.detail["successful"] == 2
    assert r.detail["errors"] == ["boom"]


def test_self_consistency_all_fail() -> None:
    def fn(r: Request) -> str:
        raise BackendError("down")

    est = conf.SelfConsistency(comparator=FakeComparator(), extractor=FakeExtractor(), samples=3)
    r = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", FakeBackend(fn))
    assert r.score is None and r.detail["failed"] == 3 and r.calls == ()


def test_self_consistency_concurrent_matches_sequential() -> None:
    backend = FakeBackend(lambda r: f"ANSWER: {'4' if r.n_sample % 2 else '3'}")
    kw = {"comparator": FakeComparator(), "extractor": FakeExtractor(), "samples": 6}
    seq = conf.SelfConsistency(**kw).estimate(TASK, REQ, comp(""), "4", backend)
    par = conf.SelfConsistency(concurrency=4, **kw).estimate(TASK, REQ, comp(""), "4", backend)
    assert seq.score == par.score == pytest.approx(4 / 7)
    assert seq.detail["answers"] == par.detail["answers"]


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_self_consistency_validates_samples(bad) -> None:
    with pytest.raises(ConfigError):
        conf.SelfConsistency(comparator=FakeComparator(), extractor=FakeExtractor(), samples=bad)


# --------------------------------------------------------------------------- monitor


def test_monitor_default_prompt_and_parse() -> None:
    backend = FakeBackend("Looks right.\nP(correct): 0.92", name="anthropic:claude-haiku-5-5")
    est = conf.Monitor(backend)
    assert est.name == "monitor(anthropic:claude-haiku-5-5)"
    assert est.prepare(REQ) is REQ
    r = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", NOBACKEND)
    assert r.score == pytest.approx(0.92)
    assert len(r.calls) == 1 and r.calls[0].text.endswith("0.92")
    sent = backend.requests[0]
    assert "What is 2+2?" in sent.prompt and "\n4\n" in sent.prompt
    assert "P(correct):" in sent.prompt
    assert sent.tags["role"] == "confidence" and sent.tags["task_id"] == "t1"
    assert not NOBACKEND.requests


def test_monitor_custom_prompt_placeholders_and_braces() -> None:
    backend = FakeBackend("CONFIDENCE: 40%")
    est = conf.Monitor(backend, prompt='Q={question} A={answer} R={response} json {"k": 1}')
    r = est.estimate(TASK, REQ, comp("full text"), "4", NOBACKEND)
    assert r.score == pytest.approx(0.4)
    assert backend.requests[0].prompt == 'Q=What is 2+2? A=4 R=full text json {"k": 1}'
    with pytest.raises(ConfigError):
        conf.Monitor(backend, prompt="no placeholders")


def test_monitor_unparseable_and_error() -> None:
    r = conf.Monitor(FakeBackend("I think so")).estimate(TASK, REQ, comp("x"), "4", NOBACKEND)
    assert r.score is None and len(r.calls) == 1

    def fail(_r: Request) -> str:
        raise BackendError("rate limited", backend="m", retryable=True)

    r = conf.Monitor(FakeBackend(fail)).estimate(TASK, REQ, comp("x"), "4", NOBACKEND)
    assert r.score is None and r.detail["error"] == "rate limited" and r.calls == ()


# --------------------------------------------------------------------------- callable


def test_callable_estimator() -> None:
    def length_score(task, completion, answer):
        return len(answer) / 10

    est = conf.CallableEstimator(length_score)
    assert est.name == "callable(length_score)"
    assert est.estimate(TASK, REQ, comp(""), "abc", NOBACKEND).score == pytest.approx(0.3)
    assert est.estimate(TASK, REQ, comp(""), "x" * 20, NOBACKEND).score == 1.0
    none = conf.CallableEstimator(lambda *a: None, name="n")
    assert none.name == "n" and none.estimate(TASK, REQ, comp(""), "4", NOBACKEND).score is None
    nan = conf.CallableEstimator(lambda *a: float("nan"))
    assert nan.estimate(TASK, REQ, comp(""), "4", NOBACKEND).score is None
    with pytest.raises(ConfigError):
        conf.CallableEstimator("not callable")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- combine


@pytest.mark.parametrize(
    ("method", "weights", "expected"),
    [("mean", None, 0.5), ("min", None, 0.2), ("max", None, 0.8), ("weighted", [3, 1], 0.65)],
)
def test_combine_methods(method, weights, expected) -> None:
    est = conf.Combine([Fixed(0.8, "a"), Fixed(0.2, "b")], method=method, weights=weights)
    r = est.estimate(TASK, REQ, comp(""), "4", NOBACKEND)
    assert r.score == pytest.approx(expected)
    assert r.detail["members"] == [
        {"estimator": "a", "score": 0.8},
        {"estimator": "b", "score": 0.2},
    ]
    assert [c.text for c in r.calls] == ["a", "b"]
    assert est.name == f"combine({method}: a, b)"


def test_combine_missing() -> None:
    members = [Fixed(0.8, "a"), Fixed(None, "b"), Fixed(0.4, "c")]
    strict = conf.Combine(members)
    assert strict.estimate(TASK, REQ, comp(""), "4", NOBACKEND).score is None
    lenient = conf.Combine(members, ignore_missing=True)
    assert lenient.estimate(TASK, REQ, comp(""), "4", NOBACKEND).score == pytest.approx(0.6)
    weighted = conf.Combine(members, method="weighted", weights=[1, 5, 3], ignore_missing=True)
    assert weighted.estimate(TASK, REQ, comp(""), "4", NOBACKEND).score == pytest.approx(
        (0.8 * 1 + 0.4 * 3) / 4
    )
    all_missing = conf.Combine([Fixed(None)], ignore_missing=True)
    assert all_missing.estimate(TASK, REQ, comp(""), "4", NOBACKEND).score is None


def test_combine_validation_and_prepare() -> None:
    with pytest.raises(ConfigError):
        conf.Combine([Fixed(0.1), Fixed(0.2)], method="weighted", weights=[1])
    with pytest.raises(ConfigError):
        conf.Combine([Fixed(0.1)], method="weighted")
    with pytest.raises(ConfigError):
        conf.Combine([Fixed(0.1)], method="weighted", weights=[0])
    with pytest.raises(ConfigError):
        conf.Combine([Fixed(0.1)], method="mean", weights=[1])
    with pytest.raises(ConfigError):
        conf.Combine([], method="mean")
    with pytest.raises(ConfigError):
        conf.Combine([Fixed(0.1)], method="median")
    est = conf.Combine([Fixed(0.1, suffix="|a"), Fixed(0.2, suffix="|b")])
    assert est.prepare(Request(prompt="p")).prompt == "p|a|b"
    w = conf.Combine([Fixed(0.1), Fixed(0.2)], method="weighted", weights=[2, 2])
    assert w.weights == (0.5, 0.5)


def test_combine_verbal_and_logprob_prepare() -> None:
    est = conf.Combine([conf.Verbal(), conf.Logprob()])
    p = est.prepare(REQ)
    assert p.want_logprobs and "CONFIDENCE:" in p.prompt


# --------------------------------------------------------------------------- calibrated


def test_calibrated_interpolation_and_clamp() -> None:
    pts = [[0.2, 0.1], [0.6, 0.5], [1.0, 0.9]]
    est = conf.Calibrated(Fixed(0.4, "base"), pts)
    assert est.name == "calibrated(base)"
    r = est.estimate(TASK, REQ, comp(""), "4", NOBACKEND)
    assert r.score == pytest.approx(0.3)
    assert r.detail["raw_score"] == 0.4 and len(r.calls) == 1
    assert est.apply(0.0) == 0.1 and est.apply(1.5) == 0.9
    assert est.apply(0.6) == pytest.approx(0.5) and est.apply(0.8) == pytest.approx(0.7)
    none = conf.Calibrated(Fixed(None), pts).estimate(TASK, REQ, comp(""), "4", NOBACKEND)
    assert none.score is None
    assert conf.Calibrated(Fixed(0.3), [[0.5, 0.7]]).apply(0.1) == 0.7


@pytest.mark.parametrize(
    "points",
    [[], [[0.5, 0.1], [0.4, 0.2]], [[0.1, 0.1], [0.1, 0.2]], [[0.1, 0.5], [0.2, 0.4]],
     [[0.1, 1.2]], [[0.1, -0.1]], [[0.1, 0.2, 0.3]], [[float("nan"), 0.1]]],
)
def test_calibrated_validation(points) -> None:
    with pytest.raises(ConfigError):
        conf.Calibrated(Fixed(0.3), points)


# --------------------------------------------------------------------------- from_spec


def test_from_spec_builds_each_type() -> None:
    monitor_backend = FakeBackend("P(correct): 0.5", name="anthropic:claude-haiku-5-5")
    backends = {"judge": monitor_backend}
    assert isinstance(build({"type": "verbal", "allow_words": True}), conf.Verbal)
    assert build({"type": "logprob", "aggregate": "geo_mean"}).name == "logprob(geo_mean)"
    sc = build({"type": "self_consistency", "samples": 3, "temperature": 0.7})
    assert isinstance(sc, conf.SelfConsistency) and sc.temperature == 0.7
    mon = build({"type": "monitor", "backend": "judge", "max_tokens": 64}, backends)
    assert mon.name == "monitor(anthropic:claude-haiku-5-5)"
    cb = build({"type": "callable", "fn": lambda *a: 0.5, "name": "mine"})
    assert cb.name == "mine"
    combo = build(
        {
            "type": "combine",
            "method": "weighted",
            "weights": [1, 1],
            "members": [{"type": "verbal"}, {"type": "monitor", "backend": "judge"}],
        },
        backends,
    )
    assert combo.name == "combine(weighted: verbal, monitor(anthropic:claude-haiku-5-5))"
    cal = build({"type": "calibrated", "base": {"type": "verbal"}, "points": [[0, 0], [1, 1]]})
    assert cal.name == "calibrated(verbal)"
    for est in (sc, mon, cb, combo, cal):
        assert isinstance(est, ConfidenceEstimator)


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ({"type": "nope"}, "unknown confidence type"),
        ({}, "unknown confidence type"),
        ({"type": "verbal", "temprature": 1}, "temprature"),
        ({"type": "monitor"}, "missing key 'backend'"),
        ({"type": "monitor", "backend": "missing"}, "available: ['judge']"),
        ({"type": "callable"}, "fn"),
        ({"type": "combine", "members": []}, "members"),
        ({"type": "combine", "members": [{"type": "bad"}]}, "unknown confidence type"),
        ({"type": "calibrated", "base": {"type": "verbal"}}, "points"),
        ({"type": "logprob", "aggregate": "max"}, "aggregate"),
    ],
)
def test_from_spec_errors(spec, fragment) -> None:
    with pytest.raises(ConfigError) as info:
        build(spec, {"judge": FakeBackend("x")})
    assert fragment in str(info.value)


def test_from_spec_rejects_non_mapping() -> None:
    with pytest.raises(ConfigError):
        build(["verbal"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- thread safety


def test_estimators_are_thread_safe() -> None:
    """One shared instance per estimator, many threads, per-task deterministic results."""

    def sample_text(r: Request) -> str:
        tid = int(r.tags["task_id"])
        return f"ANSWER: {tid if r.n_sample <= tid % 4 else -1}"

    def monitor_text(r: Request) -> str:
        tid = int(r.tags["task_id"])
        return f"P(correct): {tid % 10 / 10}"

    tier = FakeBackend(sample_text)
    sc = conf.SelfConsistency(
        comparator=FakeComparator(), extractor=FakeExtractor(), samples=3, concurrency=2
    )
    mon = conf.Monitor(FakeBackend(monitor_text))
    verbal = conf.Verbal()
    combo = conf.Combine([verbal, mon, sc], method="mean")

    def run(i: int) -> tuple[float | None, ...]:
        task = Task(id=str(i), prompt=f"q{i}")
        req = combo.prepare(Request(prompt=f"q{i}", tags={"task_id": str(i)}))
        c = comp(f"ANSWER: {i}\nCONFIDENCE: {i % 7 / 7}")
        return (
            verbal.estimate(task, req, c, str(i), tier).score,
            mon.estimate(task, req, c, str(i), tier).score,
            sc.estimate(task, req, c, str(i), tier).score,
            combo.estimate(task, req, c, str(i), tier).score,
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(run, range(200)))
    for i, (v, m, s, c) in enumerate(results):
        assert v == pytest.approx(i % 7 / 7)
        assert m == pytest.approx(i % 10 / 10)
        assert s == pytest.approx((1 + min(i % 4, 3)) / 4)
        assert c == pytest.approx((v + m + s) / 3)


# --------------------------------------------------------------------------- regressions


def test_empty_answer_gives_none_for_every_estimator_without_calls() -> None:
    backend = FakeBackend("ANSWER: \nCONFIDENCE: 0.9")
    mon_backend = FakeBackend("P(correct): 0.9")
    sc = conf.SelfConsistency(comparator=FakeComparator(), extractor=FakeExtractor(), samples=3)
    estimators = [
        conf.Verbal(),
        conf.Logprob(),
        sc,
        conf.Monitor(mon_backend),
        conf.CallableEstimator(lambda *a: 1.0),
        conf.Combine([Fixed(1.0, "a"), Fixed(1.0, "b")]),
        conf.Calibrated(Fixed(1.0), [[0.0, 0.0], [1.0, 1.0]]),
    ]
    c = comp("ANSWER: \nCONFIDENCE: 0.9", (0.0, 0.0))
    for est in estimators:
        for answer in ("", "   \n"):
            r = est.estimate(TASK, REQ, c, answer, backend)
            assert r.score is None, est.name
            assert r.detail["reason"] == "empty answer"
            assert r.calls == ()
    assert backend.requests == [] and mon_backend.requests == []


def test_self_consistency_empty_samples_disagree() -> None:
    # every sample is empty (e.g. refusals): the comparator must not see them as agreeing
    calls: list[tuple[str, str]] = []

    class Recording(FakeComparator):
        def compare(self, task, candidate, target):
            calls.append((candidate, target))
            return super().compare(task, candidate, target)

    answers = {1: "", 2: "4", 3: "  "}
    backend = FakeBackend(lambda r: f"ANSWER: {answers[r.n_sample]}")
    est = conf.SelfConsistency(comparator=Recording(), extractor=FakeExtractor(), samples=3)
    r = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", backend)
    assert r.score == pytest.approx((1 + 1) / (1 + 3))
    assert r.detail["empty"] == 2 and r.detail["votes"] == [False, True, False]
    assert calls == [("4", "4")]
    assert len(r.calls) == 3


@pytest.mark.parametrize("concurrency", [1, 3])
def test_self_consistency_non_backend_error_keeps_other_samples(concurrency: int) -> None:
    def fn(r: Request) -> str:
        if r.n_sample == 2:
            raise RuntimeError("bug in backend")
        return "ANSWER: 4"

    est = conf.SelfConsistency(
        comparator=FakeComparator(),
        extractor=FakeExtractor(),
        samples=3,
        concurrency=concurrency,
    )
    r = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", FakeBackend(fn))
    assert r.score == 1.0
    assert len(r.calls) == 2
    assert r.detail["failed"] == 1 and r.detail["successful"] == 2
    assert r.detail["errors"] == ["RuntimeError: bug in backend"]


def test_self_consistency_comparator_error_is_undecided() -> None:
    class Broken(FakeComparator):
        def compare(self, task, candidate, target):
            raise ValueError("bad")

    est = conf.SelfConsistency(comparator=Broken(), extractor=FakeExtractor(), samples=2)
    r = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", FakeBackend("ANSWER: 4"))
    assert r.score == pytest.approx(1 / 3)
    assert r.detail["undecided"] == 2 and len(r.calls) == 2


def test_monitor_non_backend_error_gives_none() -> None:
    def fail(r: Request) -> str:
        raise KeyError("oops")

    r = conf.Monitor(FakeBackend(fail)).estimate(TASK, REQ, comp("x"), "4", NOBACKEND)
    assert r.score is None and r.calls == ()
    assert r.detail["error"].startswith("KeyError")


def test_combine_member_exception_keeps_other_calls() -> None:
    def bad(task, completion, answer):
        raise RuntimeError("bug")

    mon = conf.Monitor(FakeBackend("P(correct): 0.9"))
    est = conf.Combine([mon, conf.CallableEstimator(bad, name="bad")])
    r = est.estimate(TASK, REQ, comp("ANSWER: 4"), "4", NOBACKEND)
    assert r.score is None
    assert len(r.calls) == 1 and r.calls[0].text == "P(correct): 0.9"
    assert r.detail["members"][1] == {
        "estimator": "bad",
        "score": None,
        "error": "RuntimeError: bug",
    }
    lenient = conf.Combine([mon, conf.CallableEstimator(bad)], ignore_missing=True)
    r2 = lenient.estimate(TASK, REQ, comp("ANSWER: 4"), "4", NOBACKEND)
    assert r2.score == pytest.approx(0.9) and len(r2.calls) == 1


def test_monitor_does_not_substitute_placeholders_inside_values() -> None:
    backend = FakeBackend("P(correct): 0.5")
    est = conf.Monitor(backend, prompt="Q: {question}\nA: {answer}\nR: {response}")
    task = Task(id="t1", prompt="Explain the {answer} and {response} fields")
    est.estimate(task, REQ, comp("full {question}"), "42", NOBACKEND)
    assert backend.requests[0].prompt == (
        "Q: Explain the {answer} and {response} fields\nA: 42\nR: full {question}"
    )
