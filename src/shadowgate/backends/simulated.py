"""SimulatedBackend: a deterministic offline model with a known skill level.

It powers the offline demo and most tests. Every random quantity is a pure function of
``(seed, name, task_id, n_sample, purpose)`` hashed with sha256, so results are reproducible
across processes, platforms and thread schedules.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..errors import BackendError, ConfigError
from ..types import Completion, Request, Task, Usage

if TYPE_CHECKING:
    from ..pricing import Pricing

__all__ = ["SimulatedBackend", "sigmoid"]

_REASONING = (
    "Working through the problem step by step.",
    "Breaking the question into parts and combining the results.",
    "Checking the key quantities before answering.",
    "Reasoning from the stated facts to the result.",
    "Considering the question carefully and computing the answer.",
)
_INT_RE = re.compile(r"^\s*(-?)(\d{1,3}(?:,\d{3})+|\d+)\s*$")
_FLOAT_RE = re.compile(r"^\s*-?\d+\.(\d+)\s*$")
_LETTER_RE = re.compile(r"^\s*([A-Ja-j])\s*$")
_FLIPS = {"yes": "no", "no": "yes", "true": "false", "false": "true"}
_TEXT_DISTRACTORS = ("unknown", "none of the above")
_N_LOGPROB_TOKENS = 6


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _default_pricing() -> Pricing:
    from ..pricing import Pricing

    return Pricing(input_per_mtok=1.0, output_per_mtok=5.0)


class SimulatedBackend:
    """Deterministic simulated model for tests, demos and threshold experiments.

    Generative model, for a task with difficulty ``d`` (``task.meta["difficulty"]``, default 0)
    and sample index ``k`` (``request.n_sample``):

    * ``p = sigmoid(skill - d)`` is the probability the sample is correct.
    * ``correct = U(correct, k) < p``, where ``U(purpose, k)`` is a uniform draw from
      ``sha256(seed|name|task_id|k|purpose)``. Samples are independent given the task.
    * A correct sample answers ``task.reference``. A wrong sample answers a plausible
      perturbation that never equals the reference: with probability ``systematic_error`` the
      task's "favourite" mistake (shared by all samples of that task, as real models repeat
      their mistakes), otherwise a mistake drawn per sample. Numeric references get small
      offsets or a x10 / /10 slip; choice letters get another letter; yes/no and true/false
      flip; other text gets ``"not <ref>"`` or a distractor. Wrong samples therefore agree with
      each other less often than correct samples do, which gives self-consistency signal.
    * Verbal confidence ``c = clip(p + overconfidence + discrimination * (correct - p)
      + confidence_noise * z, 0.01, 0.99)`` with ``z ~ N(0, 1)`` (Box-Muller from two hash
      uniforms). Averaged over samples ``E[c] ~= p + overconfidence``; the ``discrimination``
      term makes ``c`` informative about this particular sample (correct samples score
      ``discrimination`` higher on average than wrong ones), and the noise keeps it imperfect.
    * Output text is one reasoning sentence, then ``ANSWER: <answer>``, then
      ``CONFIDENCE: <c:.2f>`` when ``emit_confidence`` and the prompt or system prompt
      contains ``"CONFIDENCE:"``.
    * With ``request.want_logprobs`` and ``emit_logprobs``, a short tuple of per-token
      logprobs whose mean ``exp`` is approximately ``c``.
    * Usage: ``ceil(chars / 4)`` for input (system + prompt) and output; cost via ``pricing``;
      ``latency_s = latency_s + latency_per_token_s * output_tokens`` (reported, never slept).
      Output longer than ``request.max_tokens`` is truncated with stop reason ``"max_tokens"``.

    The task is located by ``request.tags["task_id"]``; otherwise the task whose prompt is
    contained in ``request.prompt`` (longest match wins). An unknown task raises BackendError.

    ``cacheable`` is False: :class:`~shadowgate.backends.cache.CachedBackend` passes requests
    straight through. The backend is deterministic and free, so caching gains nothing, and the
    cache key excludes ``tags`` (the task id), so two tasks with identical prompts would collide.
    """

    cacheable = False

    def __init__(
        self,
        name: str,
        *,
        skill: float,
        tasks: Iterable[Task],
        seed: int = 0,
        overconfidence: float = 0.0,
        confidence_noise: float = 0.1,
        discrimination: float = 0.15,
        systematic_error: float = 0.35,
        pricing: Pricing | None = None,
        latency_s: float = 0.4,
        latency_per_token_s: float = 0.01,
        emit_confidence: bool = True,
        emit_logprobs: bool = True,
    ) -> None:
        if not name:
            raise ConfigError("simulated backend needs a non-empty name")
        if confidence_noise < 0:
            raise ConfigError("simulated backend: confidence_noise must be >= 0")
        if not 0.0 <= systematic_error <= 1.0:
            raise ConfigError("simulated backend: systematic_error must be in [0, 1]")
        self.raw_name = name
        self.name = f"sim:{name}"
        self.skill = float(skill)
        self.seed = int(seed)
        self.overconfidence = float(overconfidence)
        self.confidence_noise = float(confidence_noise)
        self.discrimination = float(discrimination)
        self.systematic_error = float(systematic_error)
        self.pricing = pricing if pricing is not None else _default_pricing()
        self.latency_s = float(latency_s)
        self.latency_per_token_s = float(latency_per_token_s)
        self.emit_confidence = bool(emit_confidence)
        self.emit_logprobs = bool(emit_logprobs)

        self._tasks: dict[str, Task] = {}
        self._difficulty: dict[str, float] = {}
        for task in tasks:
            if task.id in self._tasks:
                raise ConfigError(f"simulated backend: duplicate task id {task.id!r}")
            raw = task.meta.get("difficulty", 0.0) if task.meta else 0.0
            try:
                diff = float(raw)
            except (TypeError, ValueError):
                raise ConfigError(
                    f"simulated backend: task {task.id!r} has non-numeric difficulty {raw!r}"
                ) from None
            self._tasks[task.id] = task
            self._difficulty[task.id] = diff
        # Longest prompts first so the most specific containment match wins.
        self._by_prompt = sorted(self._tasks.values(), key=lambda t: len(t.prompt), reverse=True)

    @classmethod
    def from_tasks(
        cls, tasks: Iterable[Task], *, name: str = "model", skill: float = 0.0, **kwargs: Any
    ) -> SimulatedBackend:
        """Convenience constructor: ``SimulatedBackend.from_tasks(tasks, name=..., skill=...)``."""
        return cls(name, skill=skill, tasks=tasks, **kwargs)

    def __repr__(self) -> str:
        return f"SimulatedBackend({self.raw_name!r}, skill={self.skill}, seed={self.seed})"

    def cache_fingerprint(self) -> str:
        """Every parameter that affects completions (generative params, seed, pricing)."""
        return repr((
            "simulated", self.raw_name, self.skill, self.seed, self.overconfidence,
            self.confidence_noise, self.discrimination, self.systematic_error,
            repr(self.pricing), self.latency_s, self.latency_per_token_s,
            self.emit_confidence, self.emit_logprobs,
        ))

    # ------------------------------------------------------------------ draws

    def _uniform(self, task_id: str, n_sample: int | str, purpose: str) -> float:
        blob = f"{self.seed}|{self.raw_name}|{task_id}|{n_sample}|{purpose}".encode()
        x = int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")
        return (x + 0.5) / 2.0**64  # strictly inside (0, 1)

    def _normal(self, task_id: str, n_sample: int, purpose: str) -> float:
        u1 = self._uniform(task_id, n_sample, purpose + ":1")
        u2 = self._uniform(task_id, n_sample, purpose + ":2")
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ------------------------------------------------------------------ model

    def find_task(self, request: Request) -> Task:
        """Locate the task a request is about (tag first, then prompt containment)."""
        tid = request.tags.get("task_id") if request.tags else None
        if tid is not None:
            task = self._tasks.get(tid)
            if task is None:
                raise BackendError(f"simulated backend: unknown task id {tid!r}", backend=self.name)
            return task
        for task in self._by_prompt:
            if task.prompt and task.prompt in request.prompt:
                return task
        raise BackendError(
            "simulated backend: request matches no known task (set tags['task_id'])",
            backend=self.name,
        )

    def p_correct(self, task: Task) -> float:
        return sigmoid(self.skill - self._difficulty[task.id])

    def _correct_answer(self, task: Task) -> str:
        if task.reference is not None:
            return task.reference
        h = hashlib.sha256(f"noref|{task.id}".encode()).hexdigest()[:8]
        return f"answer-{h}"

    def _wrong_answer(self, task: Task, n_sample: int) -> str:
        ref = self._correct_answer(task)
        if self._uniform(task.id, n_sample, "systematic") < self.systematic_error:
            u = self._uniform(task.id, "task", "wrong")  # shared by all samples of this task
        else:
            u = self._uniform(task.id, n_sample, "wrong")
        options = _wrong_options(ref)
        return options[min(int(u * len(options)), len(options) - 1)]

    def _sample(self, task: Task, n_sample: int) -> tuple[str, bool, float, float]:
        p = self.p_correct(task)
        correct = self._uniform(task.id, n_sample, "correct") < p
        answer = self._correct_answer(task) if correct else self._wrong_answer(task, n_sample)
        z = self._normal(task.id, n_sample, "confidence")
        c = (
            p
            + self.overconfidence
            + self.discrimination * ((1.0 if correct else 0.0) - p)
            + self.confidence_noise * z
        )
        return answer, correct, p, _clip(c, 0.01, 0.99)

    def describe(self, request: Request) -> dict[str, Any]:
        """Ground truth behind a request's completion (for tests and diagnostics)."""
        task = self.find_task(request)
        answer, correct, p, c = self._sample(task, request.n_sample)
        return {"task_id": task.id, "answer": answer, "correct": correct, "p_correct": p,
                "confidence": c}

    def _logprobs(self, task: Task, n_sample: int, c: float) -> tuple[float, ...]:
        out: list[float] = []
        for i in range(_N_LOGPROB_TOKENS // 2):
            d = 0.05 * self._uniform(task.id, n_sample, f"logprob:{i}")
            d = min(d, c - 0.001, 0.999 - c)  # keep the symmetric pair inside (0, 1)
            out.append(math.log(c + d))
            out.append(math.log(c - d))
        return tuple(out)

    # ------------------------------------------------------------------ Backend

    def complete(self, request: Request) -> Completion:
        task = self.find_task(request)
        answer, _correct, _p, c = self._sample(task, request.n_sample)
        reasoning = _REASONING[int(self._uniform(task.id, request.n_sample, "reason") * 5) % 5]
        text = f"{reasoning}\nANSWER: {answer}"
        asks = "CONFIDENCE:" in request.prompt or "CONFIDENCE:" in (request.system or "")
        if self.emit_confidence and asks:
            text += f"\nCONFIDENCE: {c:.2f}"
        stop_reason = "end"
        out_tokens = math.ceil(len(text) / 4)
        if out_tokens > request.max_tokens:
            text = text[: max(request.max_tokens, 0) * 4]
            out_tokens = math.ceil(len(text) / 4)
            stop_reason = "max_tokens"
        usage = Usage(
            input_tokens=math.ceil((len(request.system or "") + len(request.prompt)) / 4),
            output_tokens=out_tokens,
        )
        logprobs = None
        if request.want_logprobs and self.emit_logprobs:
            logprobs = self._logprobs(task, request.n_sample, c)
        return Completion(
            text=text,
            model=self.name,
            usage=usage,
            cost_usd=self.pricing.cost(usage),
            latency_s=self.latency_s + self.latency_per_token_s * out_tokens,
            stop_reason=stop_reason,
            logprobs=logprobs,
        )


def _wrong_options(ref: str) -> list[str]:
    """Plausible wrong answers for ``ref``; never contains ``ref`` itself."""
    m = _INT_RE.match(ref)
    if m:
        value = int(m.group(2).replace(",", ""))
        if m.group(1):
            value = -value
        cands = [value + d for d in (1, -1, 2, -2, 3, -3, 5, -5, 10, -10)]
        if value != 0:
            cands.append(value * 10)
            if value % 10 == 0:
                cands.append(value // 10)
        return [str(v) for v in dict.fromkeys(cands) if v != value]
    m = _FLOAT_RE.match(ref)
    if m:
        decimals = len(m.group(1))
        fvalue = float(ref)
        step = 10.0**-decimals
        deltas = (step, -step, 2 * step, -2 * step, 5 * step, -5 * step, 1.0, -1.0)
        outs = [f"{fvalue + d:.{decimals}f}" for d in deltas]
        return [s for s in dict.fromkeys(outs) if float(s) != fvalue]
    m = _LETTER_RE.match(ref)
    if m:
        letter = m.group(1)
        upper = letter.upper()
        top = max(ord("D"), ord(upper))
        letters = [chr(o) for o in range(ord("A"), top + 1) if chr(o) != upper]
        return letters if letter.isupper() else [x.lower() for x in letters]
    flipped = _FLIPS.get(ref.strip().lower())
    if flipped is not None:
        return [flipped.capitalize() if ref.strip()[:1].isupper() else flipped]
    return [f"not {ref}", *(d for d in _TEXT_DISTRACTORS if d != ref.strip().lower())]

