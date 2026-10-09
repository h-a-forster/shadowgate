from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence

import pytest

from shadowgate.backends.simulated import SimulatedBackend, _wrong_options, sigmoid
from shadowgate.errors import BackendError, ConfigError
from shadowgate.pricing import Pricing
from shadowgate.types import Request, Task

ANSWER_RE = re.compile(r"^ANSWER: (.*)$", re.M)
CONF_RE = re.compile(r"^CONFIDENCE: ([0-9.]+)$", re.M)


def make_tasks(n: int = 2000, seed: int = 7) -> list[Task]:
    rng = random.Random(seed)
    return [
        Task(
            id=f"t{i:05d}",
            prompt=f"Compute item {i:05d}: what is {rng.randint(1, 99)} plus something?",
            reference=str(rng.randint(-50, 5000)),
            meta={"difficulty": rng.uniform(-3.0, 3.0)},
        )
        for i in range(n)
    ]


TASKS = make_tasks()


def ask(task: Task, *, n_sample: int = 0, confidence: bool = True, logprobs: bool = False,
        tag: bool = True) -> Request:
    prompt = task.prompt + ("\nEnd with CONFIDENCE: <0-1>." if confidence else "")
    return Request(prompt=prompt, n_sample=n_sample, want_logprobs=logprobs,
                   tags={"task_id": task.id} if tag else {})


def answer_of(text: str) -> str:
    m = ANSWER_RE.search(text)
    assert m, text
    return m.group(1)


def conf_of(text: str) -> float:
    m = CONF_RE.search(text)
    assert m, text
    return float(m.group(1))


def run(backend: SimulatedBackend, tasks: Sequence[Task] = TASKS) -> tuple[list[bool], list[float]]:
    correct, conf = [], []
    for t in tasks:
        c = backend.complete(ask(t))
        correct.append(answer_of(c.text) == t.reference)
        conf.append(conf_of(c.text))
    return correct, conf


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def test_accuracy_matches_sigmoid_mean() -> None:
    skill = 0.5
    b = SimulatedBackend.from_tasks(TASKS, name="m", skill=skill, seed=1)
    correct, _ = run(b)
    expected = sum(sigmoid(skill - t.meta["difficulty"]) for t in TASKS) / len(TASKS)
    assert abs(sum(correct) / len(correct) - expected) < 0.035


def test_higher_skill_higher_accuracy() -> None:
    accs = []
    for skill in (-1.0, 0.0, 1.5):
        correct, _ = run(SimulatedBackend.from_tasks(TASKS, name="m", skill=skill, seed=3))
        accs.append(sum(correct) / len(correct))
    assert accs[0] + 0.1 < accs[1] < accs[2] - 0.1


def test_confidence_is_informative() -> None:
    b = SimulatedBackend.from_tasks(TASKS, name="m", skill=0.0, seed=2)
    correct, conf = run(b)
    assert auroc(conf, correct) > 0.6
    mean_ok = sum(c for c, y in zip(conf, correct, strict=True) if y) / sum(correct)
    mean_bad = sum(c for c, y in zip(conf, correct, strict=True) if not y) / (
        len(correct) - sum(correct)
    )
    assert mean_ok > mean_bad
    assert all(0.01 <= c <= 0.99 for c in conf)


def test_discrimination_raises_auroc() -> None:
    low = SimulatedBackend.from_tasks(TASKS, name="m", skill=0.0, seed=2, discrimination=0.0)
    high = SimulatedBackend.from_tasks(TASKS, name="m", skill=0.0, seed=2, discrimination=0.4)
    c1, s1 = run(low)
    c2, s2 = run(high)
    assert c1 == c2  # correctness draws do not depend on confidence parameters
    assert auroc(s2, c2) > auroc(s1, c1) + 0.03


def test_overconfidence_shifts_mean_confidence() -> None:
    subset = TASKS[:600]
    _, base = run(SimulatedBackend.from_tasks(subset, name="m", skill=0.0, seed=4), subset)
    _, over = run(
        SimulatedBackend.from_tasks(subset, name="m", skill=0.0, seed=4, overconfidence=0.2),
        subset,
    )
    shift = sum(over) / len(over) - sum(base) / len(base)
    assert 0.12 < shift < 0.21


def test_deterministic_and_seed_sensitive() -> None:
    subset = TASKS[:300]
    a = SimulatedBackend.from_tasks(subset, name="m", skill=0.3, seed=5)
    b = SimulatedBackend.from_tasks(subset, name="m", skill=0.3, seed=5)
    c = SimulatedBackend.from_tasks(subset, name="m", skill=0.3, seed=6)
    ra = [a.complete(ask(t, logprobs=True)) for t in subset]
    rb = [b.complete(ask(t, logprobs=True)) for t in subset]
    rc = [c.complete(ask(t, logprobs=True)) for t in subset]
    assert ra == rb
    assert [x.text for x in ra] != [x.text for x in rc]


def test_wrong_answers_never_equal_reference_and_disagree_more() -> None:
    b = SimulatedBackend.from_tasks(TASKS[:400], name="m", skill=-0.5, seed=8)
    wrong_pairs = wrong_agree = 0
    for t in TASKS[:400]:
        samples = []
        for k in range(5):
            info = b.describe(ask(t, n_sample=k))
            ans = answer_of(b.complete(ask(t, n_sample=k)).text)
            assert ans == info["answer"]
            assert (ans == t.reference) == info["correct"]
            samples.append(ans)
        wrong = [s for s in samples if s != t.reference]
        for i in range(len(wrong)):
            for j in range(i + 1, len(wrong)):
                wrong_pairs += 1
                wrong_agree += wrong[i] == wrong[j]
    # Correct samples always agree with each other; wrong ones agree only sometimes.
    assert wrong_pairs > 100
    assert 0.05 < wrong_agree / wrong_pairs < 0.7


@pytest.mark.parametrize(
    "ref", ["0", "7", "-12", "100", "1,250", "3.14", "0.5", "B", "d", "yes", "False", "Paris"]
)
def test_wrong_options_exclude_reference(ref: str) -> None:
    opts = _wrong_options(ref)
    assert opts
    assert ref not in opts
    assert len(set(opts)) == len(opts)


def test_wrong_numeric_is_plausible() -> None:
    assert "1260" in _wrong_options("1,250") or "1251" in _wrong_options("1,250")
    assert all(abs(float(o) - 3.14) <= 1.0 + 1e-9 for o in _wrong_options("3.14"))
    assert set(_wrong_options("B")) == {"A", "C", "D"}
    assert _wrong_options("Yes") == ["No"]


def test_output_format_and_confidence_line() -> None:
    t = TASKS[0]
    b = SimulatedBackend.from_tasks([t], name="m", skill=1.0)
    with_conf = b.complete(ask(t)).text.splitlines()
    assert len(with_conf) == 3
    assert with_conf[1].startswith("ANSWER: ")
    assert re.fullmatch(r"CONFIDENCE: \d\.\d\d", with_conf[2])
    plain = b.complete(ask(t, confidence=False)).text.splitlines()
    assert len(plain) == 2
    via_system = b.complete(
        Request(prompt=t.prompt, system="Reply with CONFIDENCE: x", tags={"task_id": t.id})
    )
    assert "CONFIDENCE:" in via_system.text
    silent = SimulatedBackend.from_tasks([t], name="m", skill=1.0, emit_confidence=False)
    assert "CONFIDENCE" not in silent.complete(ask(t)).text


def test_logprobs_track_confidence() -> None:
    b = SimulatedBackend.from_tasks(TASKS[:200], name="m", skill=0.0, seed=9)
    for t in TASKS[:200]:
        c = b.complete(ask(t, logprobs=True))
        assert c.logprobs is not None and len(c.logprobs) >= 4
        mean_p = sum(math.exp(x) for x in c.logprobs) / len(c.logprobs)
        assert abs(mean_p - conf_of(c.text)) < 0.006  # text rounds to 2 decimals
    assert b.complete(ask(TASKS[0])).logprobs is None
    off = SimulatedBackend.from_tasks(TASKS[:1], name="m", skill=0.0, emit_logprobs=False)
    assert off.complete(ask(TASKS[0], logprobs=True)).logprobs is None


def test_usage_cost_latency() -> None:
    t = TASKS[1]
    pricing = Pricing(input_per_mtok=2.0, output_per_mtok=10.0)
    b = SimulatedBackend.from_tasks(
        [t], name="m", skill=0.0, pricing=pricing, latency_s=0.5, latency_per_token_s=0.02
    )
    req = Request(prompt=t.prompt, system="sys", tags={"task_id": t.id})
    c = b.complete(req)
    assert c.usage.input_tokens == math.ceil((len("sys") + len(t.prompt)) / 4)
    assert c.usage.output_tokens == math.ceil(len(c.text) / 4)
    assert c.cost_usd == pytest.approx(pricing.cost(c.usage))
    assert c.latency_s == pytest.approx(0.5 + 0.02 * c.usage.output_tokens)
    assert c.model == "sim:m" and b.name == "sim:m"
    assert c.stop_reason == "end"
    short = b.complete(Request(prompt=t.prompt, max_tokens=3, tags={"task_id": t.id}))
    assert short.stop_reason == "max_tokens" and short.usage.output_tokens <= 3


def test_default_pricing_has_cost() -> None:
    b = SimulatedBackend.from_tasks(TASKS[:1], name="m", skill=0.0)
    assert b.complete(ask(TASKS[0])).cost_usd > 0


def test_task_lookup() -> None:
    tasks = [Task("a", "What is 2+2?", "4"), Task("b", "What is 2+2? Explain.", "4")]
    b = SimulatedBackend.from_tasks(tasks, name="m", skill=5.0)
    assert b.describe(Request(prompt="Q: What is 2+2? Explain. Be brief."))["task_id"] == "b"
    assert b.describe(Request(prompt="Q: What is 2+2?"))["task_id"] == "a"
    assert b.describe(Request(prompt="anything", tags={"task_id": "a"}))["task_id"] == "a"
    with pytest.raises(BackendError):
        b.complete(Request(prompt="unrelated"))
    with pytest.raises(BackendError):
        b.complete(Request(prompt="What is 2+2?", tags={"task_id": "zzz"}))


def test_independent_names_and_no_reference() -> None:
    t = Task("x", "open question", None)
    b = SimulatedBackend.from_tasks([t], name="m", skill=10.0)
    assert answer_of(b.complete(ask(t)).text).startswith("answer-")
    subset = TASKS[:300]
    a1 = SimulatedBackend.from_tasks(subset, name="one", skill=0.0)
    a2 = SimulatedBackend.from_tasks(subset, name="two", skill=0.0)
    assert [a1.describe(ask(t))["correct"] for t in subset] != [
        a2.describe(ask(t))["correct"] for t in subset
    ]


def test_invalid_config() -> None:
    with pytest.raises(ConfigError):
        SimulatedBackend.from_tasks([Task("a", "p"), Task("a", "q")], name="m")
    with pytest.raises(ConfigError):
        SimulatedBackend.from_tasks([Task("a", "p", meta={"difficulty": "hard"})], name="m")
    with pytest.raises(ConfigError):
        SimulatedBackend.from_tasks([], name="")


def test_cache_fingerprint_covers_all_params() -> None:
    tasks = [Task("a", "p", "1")]
    base = SimulatedBackend("m", skill=1.0, tasks=tasks).cache_fingerprint()
    for kw in ({"skill": 2.0}, {"seed": 1}, {"overconfidence": 0.1}, {"confidence_noise": 0.2},
               {"discrimination": 0.3}, {"systematic_error": 0.5},
               {"pricing": Pricing(100.0, 500.0)}, {"latency_s": 1.0},
               {"latency_per_token_s": 0.5}, {"emit_confidence": False},
               {"emit_logprobs": False}):
        merged = {"skill": 1.0, **kw}
        assert SimulatedBackend("m", tasks=tasks, **merged).cache_fingerprint() != base, kw
    assert SimulatedBackend("m", skill=1.0, tasks=tasks).cache_fingerprint() == base


def test_cached_simulated_bypasses_cache_identical_prompts() -> None:
    from shadowgate.backends.cache import CachedBackend, CacheStore

    tk = [Task("a", "What is the capital?", "Paris"), Task("b", "What is the capital?", "Rome")]
    store = CacheStore(":memory:")
    cb = CachedBackend(SimulatedBackend("m", skill=20, tasks=tk), store)
    for t in tk:
        out = cb.complete(Request(prompt=t.prompt, tags={"task_id": t.id}))
        assert out.text.endswith(f"ANSWER: {t.reference}") and out.cached is False
    assert len(store) == 0
    # Re-configured backend sharing the store answers with its own parameters and pricing.
    cheap = CachedBackend(SimulatedBackend("m", skill=20, tasks=tk), store)
    dear = CachedBackend(
        SimulatedBackend("m", skill=20, tasks=tk, pricing=Pricing(100.0, 500.0)), store
    )
    req = Request(prompt=tk[0].prompt, tags={"task_id": "a"})
    assert dear.complete(req).cost_usd > cheap.complete(req).cost_usd
