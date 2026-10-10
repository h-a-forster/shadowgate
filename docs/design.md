# shadowgate design

shadowgate routes each task through a cascade of model tiers. A cheap, fast tier answers first;
a confidence estimator scores that answer; if the score clears the tier's threshold the answer is
served, otherwise the task escalates to the next, more capable tier. This is the "System 1.5"
pattern (Oh & Gobet, *System 1.5: Designing Metacognition in Artificial Intelligence*, NeurIPS
2024 workshop): a metacognitive monitor decides when fast processing is good enough.

Routers like this usually ship with a claimed cost saving and no measurement of how often the
fast path is wrong on the cases it answers alone. shadowgate's core feature is the **shadow
audit**: it samples skipped cases, re-answers them with the final tier out of band, and estimates
the fast path's disagreement and error rate on skipped cases with confidence intervals. With
**eval mode** (every tier on every task) it sweeps thresholds and draws the accuracy/cost Pareto
curve, so a threshold is chosen from measured data.

## Vocabulary

| Term | Meaning |
|---|---|
| tier | one model + prompt + (optional) confidence estimator + threshold |
| final tier | the last tier; it always serves and, in its audit tier role, re-answers sampled skipped cases |
| skipped case | a decision answered by a non-final tier without escalating |
| disagreement rate | share of skipped cases where the final tier's answer differs |
| error rate | share of skipped cases whose answer is wrong vs ground truth (needs references) |
| inclusion probability π | chance a skipped case is selected for audit; estimates weight by 1/π |
| serve mode | normal routing; audits are sampled |
| eval mode | every tier runs on every task; enables offline threshold sweeps |

## Package layout

```
src/shadowgate/
  __init__.py       public API re-exports, __version__
  __main__.py       `python -m shadowgate`
  types.py          core records (Task, Request, Completion, Attempt, Decision, ...)
  errors.py         exception hierarchy
  pricing.py        Pricing, price table, cost_of(model, usage)
  backends/
    base.py         RetryPolicy, call_with_retries, TransientError, request_key
    __init__.py     make_backend(spec) factory + re-exports
    _proc.py        subprocess runner with process-tree kill on timeout
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
  report.py         Markdown, HTML and plain-text reports
  datasets.py       task file loading + built-in generators
  config.py         TOML config -> objects
  runner.py         concurrent batch runner (resume, budget cap, interrupts)
  cli.py            `shadowgate` command
```

Runtime dependencies: **none** (stdlib only, Python >= 3.11). The Anthropic backend imports the
official `anthropic` SDK lazily and raises `ConfigError` with an install hint
(`pip install anthropic`) if it is missing.
No numpy.

## Conventions

* `from __future__ import annotations`; full type hints; ruff (line length 100) clean.
* Factories: each pluggable family exposes `from_spec(spec, **deps)` keyed by `spec["type"]`.
  Unknown types or unknown keys raise `ConfigError` naming the offending key.
* Unknowns stay unknown: unknown cost is `None`; undecidable comparisons are
  `Judgement(equivalent=None)`; unparseable confidence is `ConfidenceResult(score=None)`.
* Thread safety: backends, estimators, comparators and the ledger are called from worker threads.
* Library code does not print; the CLI owns output. Modules log to
  `logging.getLogger("shadowgate.<mod>")`.
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
    long_context: Pricing | None = None        # rate card used above the threshold
    long_context_threshold: int = 100_000      # prompt tokens (input + cache read + cache write)
    def card_for(self, usage: Usage) -> Pricing
    def cost(self, usage: Usage) -> float

PRICES: dict[str, Pricing]          # built-in table, keyed by model id; PRICES_AS_OF = "YYYY-MM-DD"
def lookup(model: str) -> Pricing | None          # exact id, then provider-prefix-stripped id
def cost_of(model: str, usage: Usage, override: Pricing | None = None) -> float | None
def cost_with_cache_ttl(model: str, usage: Usage, *, write_1h_tokens: int = 0,
                        override: Pricing | None = None) -> float | None
def pricing_from_spec(spec: Mapping[str, Any] | None) -> Pricing | None
    # {"input", "output", "cache_read", "cache_write", "long_context", "long_context_threshold"},
    # USD per MTok; long_context is a nested table of the same shape
```

## Backends

All implement `types.Backend`: `name: str` and `complete(request) -> Completion`, thread-safe.
Each backend's `complete` uses `call_with_retries` for transient failures, reports the latency
of the final successful attempt (backoff sleeps and failed attempts excluded), computes `cost_usd`
via `pricing.cost_of`, and normalizes `stop_reason` to
`"end" | "max_tokens" | "stop_sequence" | "refusal" | "error" | <raw>`.

`make_backend(spec, *, cache: CacheStore | None = None, tasks: Iterable[Task] | None = None) ->
Backend` builds one from a config table: `{"type": "anthropic"|"openai"|"claude-code"|"command"|
"simulated"|"replay", ...}`. Common keys: `model`, `pricing` (override table), `timeout_s`,
`max_attempts`, `name` (override); `replay` takes only `path`, `name` and `key_backend`. The
`simulated` type needs `tasks`. When `cache` is given the result is wrapped in `CachedBackend`.

* **AnthropicBackend(model, \*, api_key_env="ANTHROPIC_API_KEY", base_url=None, timeout_s=600,
  retry=None, pricing=None, client=None, name=None)** - official SDK `client.messages.create`.
  Sends `temperature` only when not None; `effort` via `output_config={"effort": ...}`;
  `system`; `stop_sequences`; `extra` merged into kwargs. SDK-internal retries are disabled
  (`max_retries=0`) so shadowgate's `RetryPolicy` is the only retry layer. `RateLimitError`,
  `APIStatusError` with a retryable status (`RETRYABLE_STATUS`), `APITimeoutError` and
  `APIConnectionError` become `TransientError` (carrying the `retry-after` header when present);
  everything else becomes `BackendError`. Text is the concatenation of `text` blocks only
  (`thinking` blocks are ignored). Usage maps `input_tokens`, `output_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`. A `"refusal"` stop reason returns a
  completion with stop_reason "refusal" and whatever text exists. No logprobs (Claude does not
  expose them). `client` is injectable for tests. Name: `anthropic:<model>`.
* **OpenAICompatBackend(model, \*, base_url="https://api.openai.com/v1",
  api_key_env="OPENAI_API_KEY", timeout_s=600, retry=None, pricing=None, headers=None,
  name=None)** - stdlib `urllib` POST to `{base_url}/chat/completions`; the API key is optional
  (local servers). `want_logprobs` sends `logprobs: true` and parses
  `choices[0].logprobs.content[*].logprob`. Retries on 408/409/425/429/5xx, `URLError` and
  timeouts, and honors `Retry-After`. Name: `openai:<model>` (or the `name` override).
* **ClaudeCodeBackend(model, \*, executable="claude", timeout_s=600, retry=None, extra_args=(),
  isolation_args=DEFAULT_ISOLATION_ARGS, default_system=DEFAULT_SYSTEM, name=None, env=None,
  pricing=None)** - runs `claude -p --output-format json --model <model>` with the prompt on
  stdin, in a fresh empty temporary working directory so no project instructions leak in, with
  tools, settings, MCP servers and session persistence disabled. With Windows `.cmd` shims no
  free text goes on the command line (the system prompt goes through `--system-prompt-file`).
  Timeouts kill the whole process tree. Cost order: `pricing` override, then the price table
  (1-hour cache writes at 2x input), then the CLI's `total_cost_usd` only when the CLI recognized
  the model; otherwise None. Name: `claude-code:<model>`.
* **CommandBackend(command, \*, name=None, timeout_s=600, retry=None, pricing=None,
  retryable_exit_codes=(), cwd=None, env=None)** - prompt on stdin (system prompt, if any,
  prepended with a blank line); stdout is the text. A non-zero exit is a `BackendError`,
  retryable when the exit code is in `retryable_exit_codes`. Usage is estimated as
  `ceil(chars / 4)`; cost comes from `pricing` when given, else None. Default name:
  `command:<stem of argv[0]>`.
* **SimulatedBackend(name, \*, skill, tasks, seed=0, overconfidence=0.0, confidence_noise=0.1,
  discrimination=0.15, systematic_error=0.35, pricing=None, latency_s=0.4,
  latency_per_token_s=0.01, emit_confidence=True, emit_logprobs=True)** - offline model for tests
  and the demo. It finds the task via `request.tags["task_id"]` (fallback: the task whose prompt
  is contained in `request.prompt`). P(correct) = sigmoid(skill - difficulty), with difficulty
  from `task.meta["difficulty"]` (default 0). Correctness is drawn deterministically from
  sha256(seed, name, task_id, n_sample). Wrong answers are plausible perturbations of the
  reference (numeric: off by a small amount; text: a distractor). Output format:
  `"<one-line reasoning>\nANSWER: <answer>"` plus `"\nCONFIDENCE: <c>"` when the request asks for
  verbal confidence (prompt contains "CONFIDENCE:"), with
  `c = clip(p + overconfidence + discrimination·(correct - p) + confidence_noise·z, 0.01, 0.99)`.
  When `want_logprobs` is set, it emits logprobs whose mean exp tracks `c`. Usage comes from
  character counts and cost from its pricing (`pricing=None` uses a built-in default). Reports
  `latency_s` without sleeping. `SimulatedBackend.from_tasks(tasks, *, name="model", skill=0.0,
  **kwargs)` is a shortcut. Name: `sim:<name>`.
* **ReplayBackend(path, \*, name=None, key_backend=None)** - JSONL lines
  `{"key": request_key, "completion": {...}}` or `{"task_id", "tier" (optional), "completion"}`;
  raises `BackendError` on a miss.
* **FunctionBackend(fn: Callable[[Request], str | Completion], \*, name="function", pricing=None,
  retry=None)** - wraps a Python callable; by default it does not retry.
* **CachedBackend(inner, store: CacheStore)** and **CacheStore(path)** - SQLite table
  `completions(key TEXT PRIMARY KEY, backend TEXT, completion_json TEXT, created_at TEXT)`, WAL
  mode, thread-safe. Key = sha256 of `request_key(inner.name, request)` plus a backend
  fingerprint (`cache_fingerprint()` when defined, else class name + model, base_url, pricing
  and similar non-secret settings), so reconfiguring a backend never serves stale answers.
  Backends with `cacheable = False` (the simulated backend) bypass the cache. Cache hits return
  the stored completion with `cached=True`. Errors are never cached. `name` passes through
  unchanged.

## extract.py

```python
class Extractor(Protocol):
    name: str
    def extract(self, text: str) -> str
def from_spec(spec: Mapping[str, Any] | str | None) -> Extractor   # None -> FinalLine(); a string is {"type": <string>}
```
Types: `final_line` (`prefix="ANSWER:"`, `alt_prefixes=("Final answer:",)`, last occurrence,
case-insensitive, strips markdown bold/backticks/trailing period; fallback: last non-empty line
that isn't a CONFIDENCE line), `last_number` (last int/decimal/fraction/scientific in text,
commas removed), `choice` (single letter from `letters`, default A-J, from "ANSWER: (C)" or the
last standalone letter), `regex` (`pattern`, `group`, `flags`, `which="last"|"first"`),
`json_field` (`field`, parses the first JSON object in text), `identity` (stripped text).
Extraction never raises; it returns `""` when nothing is found.

## compare.py

```python
def from_spec(spec: Mapping[str, Any] | str | None, *,
              backends: Mapping[str, Backend] | None = None) -> Comparator   # None -> Normalized()
```
Types: `exact`; `normalized` (casefold, strip punctuation/whitespace/articles, Unicode NFKC);
`numeric` (`rel_tol=1e-6`, `abs_tol=1e-9`, `percent="either"|"ratio"|"strict"`; parses ints,
decimals, fractions, percentages, currency, thousands separators; falls back to normalized text
when either side is non-numeric and `fallback_text=True`, else `equivalent=None`); `choice`
(letters); `contains` (target in candidate after normalization); `regex` (target is a pattern;
`mode`, `ignore_case`); `judge` (`backend` name from `backends`, `prompt` template with
`{question}`, `{candidate}`, `{target}`; parses `VERDICT: EQUIVALENT|DIFFERENT`; unparseable ->
None; the judge call goes in `Judgement.calls`). An empty candidate is never equivalent to a
non-empty target (equivalent=False); two empty answers are undecidable (equivalent=None).
`numeric` compares two integers exactly (`1000000` != `1000001`); tolerances apply only when
either side is a decimal, fraction or scientific value.

## confidence.py

Estimators implement `types.ConfidenceEstimator`.

```python
def from_spec(spec: Mapping[str, Any], *, backends: Mapping[str, Backend], comparator: Comparator,
              extractor: Extractor) -> ConfidenceEstimator
```
* `verbal` - `prepare` appends an instruction to end with `CONFIDENCE: <number between 0 and 1>`
  (customizable `instruction`); `estimate` parses the last such line (accepts `0.85`, `85%`,
  `85/100`, `8 out of 10`, `0,85`; a bare integer in (1, 100] is a percentage, a bare
  non-integer there such as `1.5` is ambiguous -> None; trailing text other than punctuation or
  one parenthetical -> None); clamps to [0, 1]; missing -> score None.
* `logprob` - requires `want_logprobs`, which `prepare` sets; `aggregate="mean"|"min"|"geo_mean"`
  of token probabilities over the whole reply; None if the completion has no logprobs.
* `self_consistency` - draws `samples=k` (default 5) extra completions from the tier backend with
  `temperature` (default None = provider default) and `n_sample=1..k`, extracts answers, and
  scores the share of all k+1 answers equal (via comparator) to the primary answer. Extra calls go
  in `calls`.
* `monitor` - a separate `backend` reads the question and the proposed answer and replies
  `P(correct): <number>`; the default prompt asks for a calibrated probability; parsed like
  verbal. The monitor call goes in `calls`.
* `callable` - wraps a Python `fn(task, completion, answer) -> float | None` (API only).
* Every estimator returns score None (`detail["reason"] == "empty answer"`), with no calls, when
  the primary answer is empty. Self-consistency counts empty sample answers as disagreeing and a
  sample whose backend raises (any exception) as failed, keeping the other samples' calls.
* `combine` - `members=[spec, ...]`, `method="mean"|"min"|"max"|"weighted"` with `weights`; any
  member None -> combined None unless `ignore_missing=True`. Calls concatenate. A member that
  raises counts as None (error in `detail["members"]`); the other members' calls are kept.
* `calibrated` - wraps the estimator given as `base` with a fitted monotone map
  (`points=[[x, y], ...]`, piecewise-linear interpolation). `sweep.fit_isotonic` produces the
  points.

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
    audit_tier: str | None = None           # None or the final tier's name; any other tier is a ConfigError
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

1. For tier i: request = `Request(prompt=<template with {prompt} replaced>, system, max_tokens,
   temperature, effort, tags={"task_id", "tier", "role"})`, with role "serve" in serve mode and
   "eval" in eval mode, passed through `estimator.prepare` when present. Call the backend.
2. `BackendError` -> Attempt(error=str, accepted=False, completion=None); continue to the next tier
   (escalation on failure). If the final tier fails, Decision.error is set and the answer is the
   answer of the first accepted attempt, else the last available answer, else "".
3. Extract the answer. Non-final tier: if the completion's stop_reason is "refusal", "error" or
   "max_tokens", or the answer is empty, the estimator is skipped and the attempt escalates
   (`confidence.score = None`, `detail["rejected"]` says why); otherwise estimate confidence;
   `accepted = score is not None and score >= threshold`. If the estimator raises without
   carrying its `calls`, `detail = {"error", "cost_unknown": True}` and `cost_usd` is None.
   Final tier: accepted = True.
4. Serve mode stops at the first accepted tier. Eval mode runs every tier regardless; the
   decision's `answer`/`final_tier`/`escalated` still reflect what the router would have done at
   the configured thresholds; non-final attempts get `agreement` vs the final tier's answer.
5. Grading: when `task.reference` is set and a comparator exists, every attempt gets `correct`;
   `Decision.correct` = correctness of the served answer.
6. Shadow audit (serve mode only, a skipped case, policy mode != "off"): π =
   `inclusion_prob(score)`. If `selected(run_id, task.id, π)`: inline -> run the audit tier (role
   "audit") and compare the served answer with the audit answer using `judge` ->
   `ShadowResult(status="done", attempt, agreement)`; deferred -> `ShadowResult(status="pending")`.
   Not selected -> `ShadowResult(status="skipped")`. Every skipped case records its π in
   `ShadowResult.inclusion_prob`, so estimators know the π of the whole population. Audit tier
   failure -> status "error". `ShadowResult` fields: `(audit_tier, inclusion_prob, status,
   attempt=None, agreement=None)`.
7. Costs: `cost_usd` = sum of serving attempts (up to and including the accepted one) +
   their confidence `calls`; `audit_cost_usd` = shadow attempt + judge calls + grading judge calls
   + (eval mode) attempts beyond the accepted one. If any contributing cost is None the sum is None.
   `latency_s` = wall time of the serving path (sequential sum of serving attempts and confidence
   calls), excluding audits.
8. `created_at` = UTC ISO-8601.

## ledger.py

```python
class Ledger:
    SCHEMA_VERSION = 2
    def __init__(self, path: str | Path)                # creates parent dirs + schema; WAL; thread-safe
    def start_run(self, run_id: str, *, config: Mapping, mode: str, note: str = "") -> None  # idempotent
    def runs(self) -> list[RunInfo]
    def record(self, decision: Decision) -> None        # upsert on (run_id, task.id)
    def has(self, run_id: str, task_id: str) -> bool
    def done_task_ids(self, run_id: str, *, include_errors: bool = False) -> set[str]
    def decisions(self, run_id: str | None = None) -> Iterator[Decision]   # None -> latest run
    def pending_audits(self, run_id: str | None = None) -> Iterator[Decision]
    def latest_run_id(self) -> str | None
    def export_jsonl(self, path, run_id=None) -> int    # {"shadowgate_run": {run_id, created_at, mode, note, config}} line, then decisions
    def import_jsonl(self, path) -> int                 # header creates the run (or fills an empty config); mode mismatch -> LedgerError
    def close(self)
```
Tables: `meta(key, value)`, `runs(run_id PK, created_at, mode, config_json, note)`,
`decisions(run_id, task_id, created_at, mode, final_tier, escalated, cost_usd, audit_status,
error, decision_json, PRIMARY KEY(run_id, task_id))`. A version 1 ledger is migrated in place
(version 2 added `decisions.error`). Opening a ledger with a newer schema version raises
`LedgerError`. `RunInfo(run_id, created_at, mode, n_decisions, note="", config={})`.

## stats.py

Pure Python (math + random). All functions validate input and raise `InsufficientData` or
`ValueError` with a clear message.

```python
@dataclass(frozen=True)
class Estimate:
    value: float | None; lo: float | None; hi: float | None; n: int; method: str
    level: float = 0.95; n_eff: float | None = None

def wilson(k: int, n: int, level=0.95) -> Estimate
def clopper_pearson(k: int, n: int, level=0.95) -> Estimate         # exact; beta quantiles by bisection on the regularized incomplete beta
def weighted_proportion(ys: Sequence[bool], weights: Sequence[float], level=0.95,
                        method="korn-graubard") -> Estimate
    # Hajek ratio estimator; linearized variance; n_eff = min(Kish, variance-matched);
    # CI = Korn-Graubard (Clopper-Pearson with x = p*n_eff of n_eff, non-integer, via the
    # regularized incomplete beta; equals clopper_pearson() when weights are equal) or
    # method="wilson" (Wilson at n_eff; equals wilson() when weights are equal)
def mean_ci(xs, level=0.95) -> Estimate                               # normal approximation (no t)
def bootstrap_ci(stat: Callable[[Sequence[int]], float], n: int, *, reps=2000, level=0.95, seed=0) -> tuple[float, float]
    # resamples indices; stat receives index list; percentile interval
def paired_diff(a: Sequence[float], b: Sequence[float], level=0.95, reps=2000, seed=0) -> Estimate   # bootstrap on paired differences
def mcnemar_p(b: int, c: int) -> float                                # exact binomial two-sided
def z_for(level: float) -> float                                      # inverse normal CDF: Acklam's approximation plus one Halley step
def required_n(p: float, half_width: float, level=0.95) -> int        # audits needed for a CI half-width

# calibration / selective prediction (scores in [0,1], labels bool)
def brier(scores, labels) -> float
def ece(scores, labels, bins=10) -> float                              # equal-width bins, weighted |acc-conf|
def reliability(scores, labels, bins=10, level=0.95) -> list[Bin]      # Bin(lo, hi, n, mean_conf, accuracy, ci_lo, ci_hi)
def auroc(scores, labels) -> float | None                              # Mann-Whitney with ties; None if one class missing
def risk_coverage(scores, labels) -> list[tuple[float, float, float]]  # (threshold, coverage, risk), sorted by coverage
def aurc(scores, labels) -> float
```

## audit.py

```python
@dataclass(frozen=True)
class BinSummary:
    lo: float; hi: float; n_accepted: int; n_audited: int; disagreement: Estimate; error: Estimate | None
    label: str = ""; n_graded: int = 0

@dataclass(frozen=True)
class AuditSummary:
    run_id: str
    mode: str                            # "serve" | "eval" | "mixed" | "n/a" (empty input)
    n_decisions: int; n_errors: int
    n_skipped: int                       # skipped cases (decisions without error)
    escalation_rate: Estimate            # share of decisions escalated past tier 0
    tier_share: dict[str, int]           # decisions served per tier
    n_audited: int; n_pending: int; n_audit_errors: int; n_undecided: int
    disagreement: Estimate | None        # weighted (Hajek) skipped-case disagreement vs audit tier
    skipped_error: Estimate | None       # vs references, over all graded skipped cases
    audit_tier_error: Estimate | None    # audit tier's own error on audited cases (if references)
    served_accuracy: Estimate | None     # overall accuracy of served answers (if references)
    expected_wrong_skipped: tuple[float, float, float] | None   # (est, lo, hi) count of wrong skipped answers
    bins: list[BinSummary]
    cost_serving: float | None; cost_audit: float | None; cost_per_task: float | None
    audit_overhead: float | None         # cost_audit / cost_serving
    est_all_slow_cost_per_task: Estimate | None   # from audit-tier attempts' mean cost
    est_savings: Estimate | None                  # 1 - serving / all-final-tier
    tolerance: float | None
    status: str                          # "ok" | "breach" | "inconclusive" | "no-data" | "n/a"
    audits_to_resolve: int | None        # extra audits for CI to clear tolerance, if inconclusive;
                                         # None when audits cannot help (status uses skipped_error,
                                         # or every skipped case is audited, π = 1)
    notes: list[str]                     # human-readable caveats (e.g. "audit tier is not ground truth")
    # further fields with defaults: tiers, final_tier, reference_tier, status_metric,
    # no_score_bin, level, counts of unknown costs and unaudited cases, and:
    n_unrepresented: int = 0             # skipped cases in π strata with no completed audit (excluded)
    sparse_strata: int = 0               # π strata with >= 10% of skipped cases and < 10 audits
    tasks_to_resolve: int | None = None  # extra skipped cases for CI to clear tolerance, if inconclusive
    audit_only_status: str | None = None  # status from disagreement alone, when references drive status

def summarize(decisions: Iterable[Decision], *, tolerance: float | None = None,
              bins: Sequence[float] = (0, .5, .7, .8, .9, .95, 1.0), level=0.95) -> AuditSummary
```
Status uses `disagreement` (or `skipped_error` when references exist for every skipped case):
`breach` if lo > tolerance, `ok` if hi <= tolerance, else `inconclusive`; `no-data` when the
metric has no observations; `n/a` when no tolerance is given. Weights are 1/(π·r_h), where r_h =
completed-and-decided audits / selected audits in the case's π stratum (strata = distinct π
values, or 5 quantile groups of π when there are more than 10 distinct values): pending, failed
and undecided audits are nonresponse, assumed missing at random *within* their stratum. A
stratum with no completed audit cannot be represented: it is excluded from the estimate's target
and counted in `n_unrepresented`. If the skipped cases span several strata and one holding >= 10%
of them has < 10 completed audits, `ok` is downgraded to `inconclusive` (a rule that depends only
on the design and the audit counts, never on outcomes). The status uses a fixed-sample interval:
checking repeatedly and stopping at the first `ok` inflates false `ok` (simulated 2.4% -> 12.2%
when checking every 50 audits), so fix the number of audits in advance or use a stricter `level`
(e.g. 0.99) for repeated checks. It works for eval-mode runs too: there every non-final attempt
already has `agreement`, so every skipped case is "audited" with π = 1.

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
    # also: n_skipped_records, n_undecided, notes, grids

@dataclass(frozen=True)
class Recommendation:
    objective: str; thresholds: tuple[float, ...]; point: OperatingPoint
    holdout: OperatingPoint | None    # same thresholds evaluated on the held-out split
    note: str
    # also: delta_vs_best, best_single_tier, n_selection, n_holdout, point_index

def sweep(decisions, *, truth="auto", grid: Sequence[float] | None = None,
          objective: str = "max-savings", max_accuracy_drop: float = 0.01,
          min_accuracy: float | None = None, budget_per_task: float | None = None,
          holdout: float = 0.3, seed: int = 0, level=0.95) -> SweepResult
def simulate(decisions, thresholds, *, truth="auto", level=0.95) -> OperatingPoint
def pareto_frontier(points: Sequence[OperatingPoint]) -> list[int]
def fit_isotonic(scores, labels) -> list[tuple[float, float]]     # pool-adjacent-violators; for `calibrated` estimator
```
Simulation of a threshold vector over eval-mode records: walk tiers in order; serve at the
first tier whose confidence >= its threshold (missing confidence = not accepted); cost = sum of
attempt + confidence-call costs for tiers walked; correctness from `Attempt.correct` (truth
"reference") or `Attempt.agreement` / equality with the final tier (truth "audit-tier"; the
final tier counts as correct). `grid` default = unique observed scores thinned to <= 101
quantiles plus 0 and 1.0001 (never escalate / always escalate). For > 1 non-final tier, a joint
grid capped at 20,000 combinations (`MAX_COMBINATIONS`, thinned per tier). Objectives
(`OBJECTIVES`): `max-savings` (min cost s.t. accuracy >= best single-tier accuracy -
max_accuracy_drop), `max-accuracy` (s.t. budget_per_task), `min-accuracy` (min cost s.t.
accuracy >= min_accuracy). Selection runs on a seeded split; the chosen thresholds are
re-evaluated on the held-out part (`holdout=0` disables). Accuracy CIs are Wilson; cost is a mean.

## svg.py / report.py

`svg.py`: `xy_chart(series=(), *, title, x_label="", y_label="", width=640, height=360,
x_range=None, y_range=None, x_log=False, markers=(), error_bars=(), ref_lines=(), diagonal=None,
...) -> str`, with `line_chart(series, **kwargs)` and `scatter(series, **kwargs)` as wrappers,
and `bar_with_ci(...)`. Pure string building, `currentColor` and CSS variables so charts work in
light and dark themes, escaped text, deterministic output, accessible `<title>`/`aria-label`.

`report.py`:
```python
def render_markdown(summary: AuditSummary, sweep: SweepResult | None = None, *,
                    title: str | None = None, now: datetime | None = None) -> str
def render_html(summary: AuditSummary, sweep: SweepResult | None = None, *,
                title: str | None = None, now: datetime | None = None) -> str
def render_text(summary: AuditSummary, sweep: SweepResult | None = None) -> str
def write_report(path, summary, sweep=None, *, fmt: str | None = None,
                 title: str | None = None, now: datetime | None = None) -> Path   # fmt from suffix; html, md or text
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
def gsm8k(path: str | Path, *, limit: int | None = None) -> list[Task]
    # local GSM8K-format JSONL ({"question", "answer"} with "#### <n>")
GENERATORS = {"arithmetic": arithmetic}
def make(name: str, **kwargs) -> list[Task]
```

## config.py

TOML (stdlib `tomllib`). `load_config(path) -> Config`;
`Config.build(*, tasks: Sequence[Task] | None = None) -> tuple[Cascade, RunSettings]` (`tasks`
is required when a simulated backend is used). `RunSettings` holds `name, workers, max_cost_usd,
cache_path, ledger_path, seed, tolerance`. Sections `[run]` (name, workers, max_cost_usd, cache
path, ledger path, seed), `[backends.<id>]`, `[[tiers]]` (name, backend, threshold,
confidence = {...}, system, template, max_tokens, temperature, effort, extractor), `[answer]`
(extractor, comparator), `[audit]` (rate, strata, floor, mode, tier, seed, tolerance, judge).
Validation errors name the TOML path (`tiers[0].threshold`). A literal `api_key` field is
rejected (use `api_key_env`). See [configuration.md](configuration.md) and `examples/*.toml`.

## runner.py

```python
@dataclass
class RunStats:
    run_id: str; submitted: int = 0; completed: int = 0; skipped_existing: int = 0
    retried: int = 0; failed: int = 0; cost_serving: float = 0.0; cost_audit: float = 0.0
    unknown_cost: int = 0; elapsed_s: float = 0.0; stopped: str | None = None
    total: int = 0; escalated: int = 0; audited: int = 0
def run(cascade, tasks, ledger, *, run_id, mode="serve", workers=4, max_cost_usd=None,
        resume=True, config_snapshot=None, progress: Callable[[RunStats, Decision], None] | None = None,
        stop_event: threading.Event | None = None, max_consecutive_failures: int | None = 20) -> RunStats
def run_pending_audits(cascade, ledger, *, run_id=None, workers=4, max_cost_usd=None, progress=None,
                       stop_event=None, max_consecutive_failures=20) -> RunStats
```
A thread pool keeps at most `workers` tasks in flight; each finished Decision is written to the
ledger immediately (crash-safe and resumable: tasks already in the ledger for this run are
skipped, and tasks recorded with an error are routed again). Budget: submission stops once spend
reaches the cap and `stopped="budget"`; in-flight tasks finish and are recorded. After
`max_consecutive_failures` failures in a row, submission stops with `stopped="failures"`.
Ctrl-C (or `stop_event`): submission stops, in-flight tasks finish and are recorded, and
`stopped="interrupted"`; the interrupt is not re-raised (the CLI exits 130). A task whose route
raised unexpectedly is recorded as a Decision with `error` set and counted in `failed`.

## cli.py

```
shadowgate [-v] [--version] COMMAND ...
shadowgate init [--force] [DIR]               example config + tasks
shadowgate demo [--out DIR] [--n N] [--seed S] [--open] [-q]   offline simulated end-to-end; writes report
shadowgate run -c CONFIG -t TASKS [--ledger PATH] [--run-id ID] [--mode serve|eval]
               [--limit N] [--workers N] [--max-cost USD] [--no-resume] [-q]
shadowgate audit [--ledger PATH] [--run-id ID] [--tolerance T] [--level L] [--json] [--fail-on-breach]
shadowgate audit --run-pending -c CONFIG [--ledger PATH] [--run-id ID] [--workers N] [--max-cost USD]
shadowgate sweep [--ledger PATH] [--run-id ID] [--objective O] [--max-drop X] [--min-accuracy X]
                 [--budget X] [--holdout F] [--seed S] [--level L] [--json]
shadowgate calibrate [--ledger PATH] [--run-id ID] [--tier NAME] [--truth auto|reference|audit-tier]
                     [--max-knots K] [--json]
shadowgate report [--ledger PATH] [--run-id ID] [--sweep-run-id ID] -o OUT(.html|.md)
                  [--tolerance T] [--level L]
shadowgate runs [--ledger PATH] [--json]
shadowgate export [--ledger PATH] [--run-id ID] -o OUT.jsonl      # {"shadowgate_run": {...}} header, then decisions
shadowgate import [--ledger PATH] FILE.jsonl [FILE.jsonl ...]     # header restores mode, note, redacted config
shadowgate datasets make arithmetic -n N [--seed S] [--min-steps K] [--max-steps K] -o OUT.jsonl
shadowgate datasets show TASKS [--limit N]
```
Default ledger: `.shadowgate/ledger.sqlite`. `--level` is in (0.5, 1), default 0.95. Exit codes:
0 ok, 1 runtime error or a run stopped after repeated failures, 2 usage/config error, 3 audit
breach with `--fail-on-breach`, 4 spending cap reached, 130 interrupted. Errors print one line to
stderr (`shadowgate: error: ...`), no tracebacks unless `SHADOWGATE_DEBUG=1`. Full reference:
[cli.md](cli.md).
