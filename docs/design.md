# shadowgate design

shadowgate routes each task through a cascade of model tiers. A cheap, fast tier answers first;
a confidence estimator scores that answer; if the score clears the tier's threshold the answer is
served, otherwise the task escalates to the next (slower, stronger) tier. This is the
"System 1.5" pattern (Oh & Gobet, *System 1.5: Designing Metacognition in Artificial
Intelligence*, NeurIPS 2024 workshop): a metacognitive monitor decides when fast processing is
good enough.

Routers like this are usually shipped with a claimed cost saving and no measurement of how often
the fast path is wrong on the cases it keeps. shadowgate's core feature is the **shadow audit**: it
samples accepted (non-escalated) decisions, re-answers them with the reference tier out of band,
and estimates the fast path's disagreement and error rate on skipped cases with honest
confidence intervals. Combined with **eval mode** (every tier on every task) it sweeps thresholds
and draws the accuracy/cost Pareto curve, so a threshold is chosen from data rather than vibes.

## Vocabulary

| Term | Meaning |
|---|---|
| tier | one model + prompt + (optional) confidence estimator + threshold |
| accepted / skipped case | a decision answered by a non-final tier without escalation |
| reference tier | the tier used for shadow audits; is the last tier |
| disagreement rate | share of skipped cases where the reference tier's answer differs |
| error rate | share of skipped cases whose answer is wrong vs ground truth (needs references) |
| inclusion probability π | chance a skipped case is selected for audit; estimates weight by 1/π |
| serve mode | normal routing; audits are sampled |
| eval mode | every tier runs on every task; enables offline threshold sweeps |

## Package layout and ownership

```
src/shadowgate/
  types.py          core records (Task, Request, Completion, Attempt, Decision, ...)   [fixed]
  errors.py         exception hierarchy                                               [fixed]
  pricing.py        Pricing, price table, cost_of(model, usage)
  backends/
    base.py         RetryPolicy, call_with_retries, TransientError, request_key       [fixed]
    __init__.py     make_backend(spec) factory + re-exports
    anthropic.py    AnthropicBackend (official `anthropic` SDK, optional extra)
    openai_compat.py OpenAICompatBackend (stdlib HTTP; OpenAI, Ollama, vLLM, OpenRouter...)
    claude_code.py  ClaudeCodeBackend (`claude -p` CLI, uses the CLI's own auth)
    command.py      CommandBackend (any shell command: prompt on stdin, text on stdout)
    simulated.py    SimulatedBackend (deterministic offline model with known skill)
    replay.py       ReplayBackend (serve recorded completions from JSONL)
    function.py     FunctionBackend (wrap a Python callable)
    cache.py        CachedBackend (SQLite response cache wrapper)
  extract.py        answer extractors
  compare.py        comparators (exact, normalized, numeric, choice, contains, regex, judge)
  confidence.py     confidence estimators
  cascade.py        Tier, AuditPolicy, Cascade
  ledger.py         SQLite decision ledger
  stats.py          interval estimators, calibration metrics, bootstrap
  audit.py          AuditSummary from decisions
  sweep.py          threshold sweep, Pareto frontier, threshold selection
  svg.py            dependency-free SVG charts
  report.py         Markdown + self-contained HTML reports
  datasets.py       task file loading + built-in generators
  config.py         TOML config -> objects
  runner.py         concurrent batch runner (resume, budget cap, interrupts)
  cli.py            `shadowgate` command
```

Runtime dependencies: **none** (stdlib only, Python >= 3.11). The Anthropic backend imports the
official `anthropic` SDK lazily and raises `ConfigError` with an install hint
(`pip install "shadowgate[anthropic]"`) if it is missing. No numpy.

## Conventions every module follows

* `from __future__ import annotations`; full type hints; ruff (line length 100) clean.
* Factories: each pluggable family exposes `from_spec(spec: Mapping[str, Any], **deps)` keyed by
  `spec["type"]`. Unknown types or unknown keys raise `ConfigError` naming the offending key.
* Never silently coerce unknowns: unknown cost stays `None`; undecidable comparisons are
  `Judgement(equivalent=None)`; unparseable confidence is `ConfidenceResult(score=None)`.
* Thread safety: backends, estimators, comparators and the ledger are called from worker threads.
* No printing from library code; the CLI owns output. Use `logging.getLogger("shadowgate.<mod>")`.
* Secrets: API keys come from environment variables named in config (`api_key_env`), never from
  config values, and are never logged, cached, or written to the ledger.
* Determinism: anything random takes a seed and uses its own `random.Random`; hashing for
  sampling uses sha256 of stable strings, not Python's `hash()`.

## pricing.py

```python
@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None   # None -> same as input
    cache_write_per_mtok: float | None = None  # None -> same as input
    def cost(self, usage: Usage) -> float

PRICES: dict[str, Pricing]          # built-in table, keyed by model id; PRICES_AS_OF = "YYYY-MM-DD"
def lookup(model: str) -> Pricing | None          # exact id, then provider-prefix-stripped id
def cost_of(model: str, usage: Usage, override: Pricing | None = None) -> float | None
def pricing_from_spec(spec: Mapping | None) -> Pricing | None   # {"input": .., "output": .., "cache_read": .., "cache_write": ..} per MTok
```

## Backends

All implement `types.Backend`: `name: str` and `complete(request) -> Completion`, thread-safe.
Each backend's `complete` uses `call_with_retries` for transient failures, measures latency with
`Timer`, computes `cost_usd` via `pricing.cost_of` (or the provider-reported cost when the provider
returns one, e.g. the Claude Code CLI), and normalises `stop_reason` to
`"end" | "max_tokens" | "stop_sequence" | "refusal" | "error" | <raw>`.

`make_backend(spec: Mapping, *, cache: CacheStore | None = None) -> Backend` builds one from a
config table: `{"type": "anthropic"|"openai"|"claude-code"|"command"|"simulated"|"replay", ...}`.
Common keys: `model`, `pricing` (override table), `timeout_s`, `max_attempts`, `name` (override).
When `cache` is given the result is wrapped in `CachedBackend`.

* **AnthropicBackend(model, *, api_key_env="ANTHROPIC_API_KEY", base_url=None, timeout_s=600,
  retry=RetryPolicy(), pricing=None, client=None)** - official SDK `client.messages.create`.
  Sends `temperature` only when not None; `effort` via `output_config={"effort": ...}`;
  `system`; `stop_sequences`; `extra` merged into kwargs. Disable SDK-internal retries
  (`max_retries=0`) so ours are the only ones; map `RateLimitError`, `APIStatusError` with
  retryable status (see `RETRYABLE_STATUS`), `APITimeoutError`, `APIConnectionError` to
  `TransientError` (with `retry-after` header when present), everything else to `BackendError`.
  Text = concatenation of `text` blocks only (ignore `thinking` blocks). Usage maps
  `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`.
  Handles `stop_reason == "refusal"` (returns completion with stop_reason "refusal" and whatever
  text exists). No logprobs (Claude does not expose them). `client` injectable for tests.
  Name: `anthropic:<model>`.
* **OpenAICompatBackend(model, *, base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
  timeout_s=600, retry, pricing, headers=None)** - stdlib `urllib` POST to
  `{base_url}/chat/completions`; api key optional (local servers). `want_logprobs` -> `logprobs:
  true` and parse `choices[0].logprobs.content[*].logprob`. Retry on 408/409/425/429/5xx,
  `URLError`, timeouts; honour `Retry-After`. Name: `openai:<model>` (or `name` override).
* **ClaudeCodeBackend(model, *, executable="claude", timeout_s=600, retry, extra_args=())** - runs
  `claude -p --output-format json --model <model>` with the prompt on stdin, in a fresh empty
  temporary working directory so no project instructions leak in, with tools disabled where the
  CLI supports it. Parses the JSON result: text, `usage`, `total_cost_usd` (provider-reported
  cost), `is_error`. Name: `claude-code:<model>`.
* **CommandBackend(command: list[str] | str, *, name, timeout_s, retry, pricing=None)** - prompt
  on stdin (system prompt, if any, prepended with a blank line), stdout is the text; non-zero exit
  is a BackendError (retryable if exit code in a configurable set). Usage estimated as
  `ceil(len(chars)/4)` and flagged `detail`-free; cost via pricing if given else None.
* **SimulatedBackend(name, *, skill, tasks, seed=0, overconfidence=0.0, confidence_noise=0.1,
  pricing=Pricing(...), latency_s=..., emit_confidence=True, emit_logprobs=True)** - offline model
  for tests and the demo. It finds the task via `request.tags["task_id"]` (fallback: the task
  whose prompt is contained in `request.prompt`). P(correct) = sigmoid(skill - difficulty) with
  difficulty from `task.meta["difficulty"]` (default 0). Correctness is drawn deterministically
  from sha256(seed, name, task_id, n_sample). Wrong answers are plausible perturbations of the
  reference (numeric: off-by-small-amount; text: a distractor). Output format:
  `"<one-line reasoning>\nANSWER: <answer>"` plus `"\nCONFIDENCE: <c>"` when the request asks for
  verbal confidence (prompt contains "CONFIDENCE:"); `c` = clip(p_correct + overconfidence +
  N(0, confidence_noise)). When `want_logprobs`, emits logprobs whose mean exp tracks `c`.
  Usage from character counts; cost via its pricing. Reports `latency_s` without sleeping.
  `SimulatedBackend.from_tasks(tasks, ...)` convenience. Name: `sim:<name>`.
* **ReplayBackend(path)** - JSONL lines `{"key": request_key, "completion": {...}}` or
  `{"task_id", "tier", "completion"}`; raises BackendError on miss.
* **FunctionBackend(fn: Callable[[Request], str | Completion], *, name)**.
* **CachedBackend(inner, store: CacheStore)** and **CacheStore(path)** - SQLite table
  `(key TEXT PRIMARY KEY, backend TEXT, completion_json TEXT, created_at TEXT)`, WAL mode,
  thread-safe. Key = `request_key(inner.name, request)`. Cache hits return the stored completion
  with `cached=True`. Errors are never cached. `name` passes through unchanged.

## extract.py

```python
class Extractor(Protocol):
    name: str
    def extract(self, text: str) -> str
def from_spec(spec: Mapping | None) -> Extractor     # None -> FinalLine()
```
Types: `final_line` (`prefix="ANSWER:"`, last occurrence, case-insensitive, strips markdown
bold/backticks/trailing period; fallback: last non-empty line that isn't a CONFIDENCE line),
`last_number` (last int/decimal/fraction/scientific in text, commas removed),
`choice` (single letter A-J, from "ANSWER: (C)" or last standalone letter), `regex` (`pattern`,
`group`), `json_field` (`field`, parses first JSON object in text), `identity` (stripped text).
Extraction never raises; returns `""` when nothing found.

## compare.py

```python
def from_spec(spec: Mapping | None, *, backends: Mapping[str, Backend] = {}) -> Comparator  # None -> Normalized()
```
Types: `exact`; `normalized` (casefold, strip punctuation/whitespace/articles, unicode NFKC);
`numeric` (`rel_tol=1e-6`, `abs_tol=1e-9`; parses ints, decimals, fractions, percentages,
currency, thousands separators; falls back to normalized text when either side is non-numeric
and `fallback_text=True`, else `equivalent=None`); `choice` (letters); `contains` (target in
candidate after normalisation); `regex` (target is a pattern); `judge` (`backend` name from
`backends`, `prompt` template with `{question}`, `{candidate}`, `{target}`; parses
`VERDICT: EQUIVALENT|DIFFERENT`; unparseable -> None; the judge call goes in `Judgement.calls`).
Empty candidate is never equivalent to a non-empty target (equivalent=False).

## confidence.py

Estimators implement `types.ConfidenceEstimator`.

```python
def from_spec(spec: Mapping, *, backends: Mapping[str, Backend], comparator: Comparator,
              extractor: Extractor) -> ConfidenceEstimator
```
* `verbal` - `prepare` appends an instruction to end with `CONFIDENCE: <number between 0 and 1>`
  (customisable `instruction`); `estimate` parses the last such line (accepts `0.85`, `85%`,
  `85/100`); clamps to [0, 1]; missing -> score None.
* `logprob` - requires `want_logprobs`; `prepare` sets it; `aggregate="mean"|"min"|"geo_mean"` of
  token probabilities (optionally only over the answer line's tokens is out of scope); None if the
  completion has no logprobs.
* `self_consistency` - draws `samples=k` extra completions from the tier backend with
  `temperature` (default None = provider default) and `n_sample=1..k`, extracts answers, score =
  share of all k+1 answers equal (via comparator) to the primary answer. Extra calls in `calls`.
* `monitor` - a separate `backend` reads the question and the proposed answer and replies
  `P(correct): <number>`; default prompt asks for a calibrated probability; parse like verbal.
  The monitor call goes in `calls`.
* `callable` - wraps a Python `fn(task, completion, answer) -> float | None` (API only).
* `combine` - `members=[spec, ...]`, `method="mean"|"min"|"max"|"weighted"` with `weights`; any
  member None -> combined None unless `ignore_missing=True`. Calls concatenate.
* `calibrated` - wraps another estimator with a fitted monotone map (`points=[[x, y], ...]`,
  piecewise-linear interpolation). `sweep.fit_isotonic` produces the points.

## cascade.py

```python
@dataclass(frozen=True)
class Tier:
    name: str
    backend: Backend
    threshold: float | None = None          # required for non-final tiers
    estimator: ConfidenceEstimator | None = None   # required for non-final tiers
    system: str | None = None
    template: str = DEFAULT_TEMPLATE        # must contain "{prompt}"
    max_tokens: int = 2048
    temperature: float | None = None
    effort: str | None = None
    extractor: Extractor | None = None      # None -> cascade default

@dataclass(frozen=True)
class AuditPolicy:
    rate: float = 0.1                       # uniform inclusion probability
    strata: tuple[tuple[float, float, float], ...] = ()   # (lo, hi, rate) on confidence; first match wins; the band(s) with the largest hi (or hi >= 1) include conf == hi
    floor: float = 0.01                     # min inclusion prob; must be > 0 to keep estimates unbiased
    mode: str = "inline"                    # "inline" | "deferred" | "off"
    audit_tier: str | None = None           # None or the last tier's name; any other tier is a ConfigError
    seed: int = 0
    def inclusion_prob(self, confidence: float | None) -> float
    def selected(self, run_id: str, task_id: str, prob: float) -> bool   # sha256-based, reproducible

class Cascade:
    def __init__(self, tiers: Sequence[Tier], *, extractor: Extractor | None = None,
                 comparator: Comparator | None = None,       # answer vs answer / reference
                 audit: AuditPolicy | None = None, judge: Comparator | None = None)  # judge: audit agreement, defaults to comparator
    def route(self, task: Task, *, run_id: str = "", mode: str = "serve") -> Decision
    def complete_audit(self, decision: Decision) -> Decision   # runs a pending shadow audit
```
`DEFAULT_TEMPLATE` asks the model to solve the task and finish with a line `ANSWER: <answer>`.

`route` algorithm:

1. For tier i: request = `Request(prompt=template.format(prompt=task.prompt), system, max_tokens,
   temperature, effort, tags={"task_id", "tier", "role": "serve"})`, passed through
   `estimator.prepare` when present. Call the backend.
2. `BackendError` -> Attempt(error=str, accepted=False, completion=None); continue to the next tier
   (escalation on failure). If the last tier fails, Decision.error is set, answer is the best
   accepted-or-last available answer or "".
3. Extract the answer. Non-final tier: if the completion's stop_reason is "refusal", "error" or
   "max_tokens", or the answer is empty, the estimator is skipped and the attempt escalates
   (`confidence.score = None`, `detail["rejected"]` says why); otherwise estimate confidence;
   `accepted = score is not None and score >= threshold`. If the estimator raises without
   carrying its `calls`, `detail = {"error", "cost_unknown": True}` and `cost_usd` is None.
   Final tier: accepted = True.
4. Serve mode stops at the first accepted tier. Eval mode runs every tier regardless; the
   decision's `answer`/`final_tier`/`escalated` still reflect what the router would have done at
   the configured thresholds; non-final attempts get `agreement` vs the last tier's answer.
5. Grading: when `task.reference` is set and a comparator exists, every attempt gets `correct`;
   `Decision.correct` = correctness of the served answer.
6. Shadow (serve mode only, accepted at a non-final tier, policy mode != "off"): π =
   `inclusion_prob(score)`; if `selected(run_id, task.id, π)`: inline -> run the audit tier (role
   "audit"), compare served answer vs audit answer with `judge` -> ShadowResult(status="done",
   attempt, agreement); deferred -> ShadowResult(status="pending"). Not selected -> `shadow=None`
   but π is still recorded in `Decision` via `ShadowResult(status="skipped", inclusion_prob=π)`
   so estimators know every accepted case's π. Audit tier failure -> status "error".
7. Costs: `cost_usd` = sum of serving attempts (up to and including the accepted one) +
   their confidence `calls`; `audit_cost_usd` = shadow attempt + judge calls + grading judge calls
   + (eval mode) attempts beyond the accepted one. If any contributing cost is None the sum is None.
   `latency_s` = wall time of the serving path (sequential sum of serving attempts and confidence
   calls), excluding audits.
8. `created_at` = UTC ISO-8601.

## ledger.py

```python
class Ledger:
    SCHEMA_VERSION = 1
    def __init__(self, path: str | Path)                # creates parent dirs + schema; WAL; thread-safe
    def start_run(self, run_id: str, *, config: Mapping, mode: str, note: str = "") -> None  # idempotent
    def runs(self) -> list[RunInfo]
    def record(self, decision: Decision) -> None        # upsert on (run_id, task.id)
    def has(self, run_id: str, task_id: str) -> bool
    def done_task_ids(self, run_id: str) -> set[str]
    def decisions(self, run_id: str | None = None) -> Iterator[Decision]   # None -> latest run
    def pending_audits(self, run_id: str | None = None) -> Iterator[Decision]
    def latest_run_id(self) -> str | None
    def export_jsonl(self, path, run_id=None) -> int
    def import_jsonl(self, path) -> int
    def close(self)
```
Tables: `meta(key, value)`, `runs(run_id PK, created_at, mode, config_json, note)`,
`decisions(run_id, task_id, created_at, mode, final_tier, escalated, cost_usd, audit_status,
decision_json, PRIMARY KEY(run_id, task_id))`. Opening a ledger with a newer schema version raises
`LedgerError`. `RunInfo(run_id, created_at, mode, n_decisions, note, config)`.

## stats.py

Pure-Python (math + random). All functions validate input and raise `InsufficientData` or
`ValueError` with a clear message.

```python
@dataclass(frozen=True)
class Estimate:
    value: float | None; lo: float | None; hi: float | None; n: int; method: str
    level: float = 0.95; n_eff: float | None = None

def wilson(k: int, n: int, level=0.95) -> Estimate
def clopper_pearson(k: int, n: int, level=0.95) -> Estimate         # exact; implement beta quantile via bisection on regularized incomplete beta
def weighted_proportion(ys: Sequence[bool], weights: Sequence[float], level=0.95) -> Estimate
    # Hajek ratio estimator; linearised variance; Kish effective n; CI = Wilson interval at n_eff
    # (keeps CI in [0,1]); equals wilson() when weights are equal
def mean_ci(xs, level=0.95) -> Estimate                               # t-free normal approx + n
def bootstrap_ci(stat: Callable[[Sequence[int]], float], n: int, *, reps=2000, level=0.95, seed=0) -> tuple[float, float]
    # resamples indices; stat receives index list; percentile interval
def paired_diff(a: Sequence[float], b: Sequence[float], level=0.95, reps=2000, seed=0) -> Estimate   # bootstrap on paired differences
def mcnemar_p(b: int, c: int) -> float                                # exact binomial two-sided
def z_for(level: float) -> float                                      # inverse normal CDF (Acklam or bisection on erf)
def required_n(p: float, half_width: float, level=0.95) -> int        # audits needed for a CI half-width

# calibration / selective prediction (scores in [0,1], labels bool)
def brier(scores, labels) -> float
def ece(scores, labels, bins=10) -> float                              # equal-width bins, weighted |acc-conf|
def reliability(scores, labels, bins=10) -> list[Bin]                  # Bin(lo, hi, n, mean_conf, accuracy, ci_lo, ci_hi)
def auroc(scores, labels) -> float | None                              # Mann-Whitney with ties; None if one class missing
def risk_coverage(scores, labels) -> list[tuple[float, float, float]]  # (threshold, coverage, risk), sorted by coverage
def aurc(scores, labels) -> float
```

## audit.py

```python
@dataclass(frozen=True)
class BinSummary:
    lo: float; hi: float; n_accepted: int; n_audited: int; disagreement: Estimate; error: Estimate | None

@dataclass(frozen=True)
class AuditSummary:
    run_id: str; mode: str
    n_decisions: int; n_errors: int
    n_skipped: int                       # accepted at a non-final tier
    escalation_rate: Estimate            # share of decisions escalated past tier 0
    tier_share: dict[str, int]           # decisions served per tier
    n_audited: int; n_pending: int; n_audit_errors: int; n_undecided: int
    disagreement: Estimate | None        # weighted (Hajek) skipped-case disagreement vs audit tier
    skipped_error: Estimate | None       # vs references, over all graded skipped cases
    audit_tier_error: Estimate | None    # audit tier's own error on audited cases (if references)
    served_accuracy: Estimate | None     # overall accuracy of served answers (if references)
    expected_wrong_skipped: tuple[float, float, float] | None   # (est, lo, hi) count of wrong-but-kept answers
    bins: list[BinSummary]
    cost_serving: float | None; cost_audit: float | None; cost_per_task: float | None
    audit_overhead: float | None         # cost_audit / cost_serving
    est_all_slow_cost_per_task: Estimate | None   # from audit-tier attempts' mean cost
    est_savings: Estimate | None                  # 1 - serving / all-slow
    tolerance: float | None
    status: str                          # "ok" | "breach" | "inconclusive" | "no-data"
    audits_to_resolve: int | None        # extra audits for CI to clear tolerance, if inconclusive
    notes: list[str]                     # human-readable caveats (e.g. "audit tier is not ground truth")

def summarize(decisions: Iterable[Decision], *, tolerance: float | None = None,
              bins: Sequence[float] = (0, .5, .7, .8, .9, .95, 1.0), level=0.95) -> AuditSummary
```
Status uses `disagreement` (or `skipped_error` when references exist for every skipped case):
`breach` if lo > tolerance, `ok` if hi <= tolerance, else `inconclusive`.
Works for eval-mode runs too: there every non-final attempt already has `agreement`, so every
skipped case is "audited" with π = 1.

## sweep.py

Input: eval-mode decisions (every tier answered, confidence recorded for non-final tiers).

```python
@dataclass(frozen=True)
class OperatingPoint:
    thresholds: tuple[float, ...]     # one per non-final tier
    accuracy: Estimate                # vs truth
    cost_per_task: float | None
    latency_per_task: float
    escalation_rate: float
    tier_share: tuple[float, ...]
    skipped_error: Estimate | None

@dataclass(frozen=True)
class SweepResult:
    truth: str                        # "reference" | "audit-tier"
    n: int
    tiers: tuple[str, ...]
    points: list[OperatingPoint]
    frontier: list[int]               # indices into points, Pareto-optimal (max accuracy, min cost)
    baselines: dict[str, OperatingPoint]   # "only:<tier>" for each tier, "oracle"
    calibration: dict[str, CalibrationReport]  # per non-final tier: ece, brier, auroc, aurc, reliability bins, risk_coverage
    recommendation: Recommendation | None

@dataclass(frozen=True)
class Recommendation:
    objective: str; thresholds: tuple[float, ...]; point: OperatingPoint
    holdout: OperatingPoint | None    # same thresholds evaluated on the held-out split
    note: str

def sweep(decisions, *, truth="auto", grid: Sequence[float] | None = None,
          objective: str = "max-savings", max_accuracy_drop: float = 0.01,
          min_accuracy: float | None = None, budget_per_task: float | None = None,
          holdout: float = 0.3, seed: int = 0, level=0.95) -> SweepResult
def fit_isotonic(scores, labels) -> list[tuple[float, float]]     # pool-adjacent-violators; for `calibrated` estimator
```
Simulation of a threshold vector over eval-mode records: walk tiers in order; serve at the
first tier whose confidence >= its threshold (missing confidence = not accepted); cost = sum of
attempt + confidence-call costs for tiers walked; correctness from `Attempt.correct` (truth
"reference") or `Attempt.agreement` / equality with the last tier (truth "audit-tier"; last tier
counts as correct). `grid` default = unique observed scores thinned to <= 101 quantiles plus 0
and 1.0001 (never escalate / always escalate). For > 1 non-final tier, a joint grid capped at
20,000 combinations (thinned per tier). Objectives:
`max-savings` (min cost s.t. accuracy >= best single-tier accuracy - max_accuracy_drop),
`max-accuracy` (s.t. budget_per_task), `min-accuracy` (min cost s.t. accuracy >= min_accuracy).
Selection runs on a seeded split; the chosen thresholds are re-evaluated on the held-out part
(`holdout=0` disables). Accuracy CIs are Wilson; cost is a mean.

## svg.py / report.py

`svg.py`: `line_chart(series, *, x_label, y_label, title, width=640, height=360, x_range=None,
y_range=None, points=None, annotations=None) -> str`, `scatter(...)`, `bar_with_ci(...)`.
Pure string building, `currentColor` and CSS variables so charts work in light and dark themes,
escaped text, deterministic output, accessible `<title>`/`aria-label`.

`report.py`:
```python
def render_markdown(summary: AuditSummary, sweep: SweepResult | None = None, *, title: str = ...) -> str
def render_html(summary: AuditSummary, sweep: SweepResult | None = None, *, title: str = ...) -> str
def write_report(path, summary, sweep=None, *, fmt: str | None = None) -> Path   # fmt from suffix
```
Sections: headline (status badge, skipped-case disagreement/error with CI, escalation rate,
cost/task, savings), audit by confidence bin (table + CI chart), tier usage, (eval) Pareto chart
with frontier, baselines and the recommended point, reliability diagram, risk-coverage curve,
caveats. Self-contained HTML (inline CSS, no external requests), light/dark via
`prefers-color-scheme`, responsive, readable at phone width.

## datasets.py

```python
def load_tasks(path: str | Path, *, limit: int | None = None) -> list[Task]
    # .jsonl/.json/.csv; fields id (or generated "t0001"), prompt (aliases: question, input),
    # reference (aliases: answer, target, label); other fields -> meta. Duplicate ids -> DatasetError
def save_tasks(tasks, path) -> None
def arithmetic(n: int = 200, *, seed: int = 0, min_steps: int = 1, max_steps: int = 6) -> list[Task]
    # multi-step word problems with integer answers; meta.difficulty grows with steps
def gsm8k(path) -> list[Task]       # local GSM8K-format JSONL ({"question", "answer"} with "#### <n>")
GENERATORS = {"arithmetic": arithmetic}
```

## config.py

TOML (stdlib `tomllib`). `load_config(path) -> Config`, `Config.build() -> (Cascade, RunSettings)`.
Sections `[run]` (name, workers, max_cost_usd, cache path, ledger path, seed), `[backends.<id>]`,
`[[tiers]]` (name, backend, threshold, confidence = {...}, system, template, max_tokens,
temperature, effort, extractor), `[answer]` (extractor, comparator), `[audit]` (rate, strata,
floor, mode, tier, seed, tolerance, judge). Validation errors name the TOML path
(`tiers[0].threshold`). A literal `api_key` field is rejected (use `api_key_env`). See
`examples/*.toml`.

## runner.py

```python
@dataclass
class RunStats:
    run_id: str; submitted: int; completed: int; skipped_existing: int; failed: int
    cost_serving: float; cost_audit: float; unknown_cost: int; elapsed_s: float; stopped: str | None
def run(cascade, tasks, ledger, *, run_id, mode="serve", workers=4, max_cost_usd=None,
        resume=True, config_snapshot=None, progress: Callable[[RunStats, Decision], None] | None = None) -> RunStats
def run_pending_audits(cascade, ledger, *, run_id=None, workers=4, max_cost_usd=None, progress=None) -> RunStats
```
Thread pool with at most `workers` in flight; each finished Decision is written to the ledger
immediately (crash-safe, resumable: tasks already in the ledger for this run are skipped). Budget:
stop submitting once spent >= cap and report `stopped="budget"`; in-flight tasks finish and are
recorded. Ctrl-C: stop submitting, wait for in-flight, record, `stopped="interrupted"`, re-raise
nothing (CLI exits 130). A task whose route raised unexpectedly is recorded as a Decision with
`error` set and counted in `failed`.

## cli.py

```
shadowgate init [DIR]                         example config + tasks
shadowgate demo [--out DIR] [--n N] [--seed S]  fully offline simulated end-to-end; writes report
shadowgate run -c CONFIG -t TASKS [--ledger PATH] [--run-id ID] [--mode serve|eval]
               [--limit N] [--workers N] [--max-cost USD] [--no-resume]
shadowgate audit [--ledger PATH] [--run-id ID] [--tolerance T] [--json] [--fail-on-breach]
shadowgate audit --run-pending -c CONFIG [--ledger PATH] [--run-id ID]
shadowgate sweep [--ledger PATH] [--run-id ID] [--objective O] [--max-drop X] [--json]
shadowgate report [--ledger PATH] [--run-id ID] -o OUT(.html|.md) [--tolerance T]
shadowgate runs [--ledger PATH]
shadowgate export [--ledger PATH] [--run-id ID] -o OUT.jsonl
shadowgate datasets make arithmetic -n N [--seed S] -o OUT.jsonl
```
Default ledger: `.shadowgate/ledger.sqlite`. Exit codes: 0 ok, 1 runtime error, 2 usage/config
error, 3 audit breach with `--fail-on-breach`, 130 interrupted. Errors print one line to stderr
(`shadowgate: error: ...`), no tracebacks unless `SHADOWGATE_DEBUG=1`.
