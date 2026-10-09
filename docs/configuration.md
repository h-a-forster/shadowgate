# Configuration reference

A shadowgate config is a TOML file with five top-level sections. Any other top-level key is an
error.

| Section | Required | Purpose |
|---|---|---|
| [`[run]`](#run) | no | run name, concurrency, spending cap, cache, ledger path, seed |
| [`[backends.<id>]`](#backends) | yes (at least the ones tiers use) | model backends, referenced by id |
| [`[[tiers]]`](#tiers) | yes, at least one | cascade tiers, cheapest first; the last tier always serves |
| [`[answer]`](#answer) | no | answer extractor and comparator |
| [`[audit]`](#audit) | no | shadow-audit sampling, tolerance and audit judge |

Complete, runnable configs are in [`examples/`](../examples/). The smallest one that works
offline is [`examples/simulated.toml`](../examples/simulated.toml).

## Validation rules

- Unknown keys are rejected with the TOML path and a "did you mean" hint:
  `tiers[1].backend: unknown backend "slwo" (defined: fast, slow)`.
- Credentials are rejected anywhere in the file. Keys named `api_key`, `apikey`, `api-key`,
  `token`, `access_token`, `auth_token`, `secret`, `secret_key`, `client_secret` or `password`
  (case-insensitive) are errors, as are `Authorization`, `Proxy-Authorization`, `X-Api-Key`,
  `Api-Key` and `X-Auth-Token` headers. Name an environment variable with `api_key_env` instead.
- There is no `${VAR}` expansion. `api_key_env` is the only environment lookup.
- Relative paths in `run.ledger`, `run.cache`, a `replay` backend's `path` and a `command`
  backend's `cwd` resolve against the config file's directory. The default ledger path is
  relative to the working directory.
- Only backends referenced by a tier, a `monitor` estimator or a `judge` comparator are built.
  Each is built once and shared.
- Several tables accept a string shorthand for `{ type = "..." }`:
  `extractor = "final_line"` is the same as `extractor = { type = "final_line" }`. This applies to
  `[answer].extractor`, `[answer].comparator`, `tiers[].extractor`, `tiers[].confidence` and
  `[audit].judge`.

`shadowgate run` validates the whole file before the first model call. A config error exits
with code 2 and one line naming the offending path.

## `[run]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | config file stem | Prefix of generated run ids (`<name>-<mode>-<YYYYmmdd-HHMMSS>`). |
| `workers` | integer >= 1 | `4` | Tasks in flight. `--workers` overrides. |
| `max_cost_usd` | number > 0 | unset (no cap) | Stop submitting new tasks once serving + audit spend reaches this. In-flight tasks finish and are recorded. `--max-cost` overrides. |
| `cache` | boolean or path | `false` | `true` puts `cache.sqlite` next to the ledger; a string is the cache file path. Identical requests to an identically configured backend are served from the cache. Errors are never cached. Simulated backends bypass it. |
| `ledger` | path | `.shadowgate/ledger.sqlite` | SQLite ledger for decisions. `--ledger` overrides. |
| `seed` | integer | `0` | Default for `[audit].seed` and for every `simulated` backend's `seed`. |

Only `run` and `audit --run-pending` read the config. `audit`, `sweep`, `calibrate`, `report`,
`runs`, `export` and `import` use `.shadowgate/ledger.sqlite` unless you pass `--ledger`. If you
set `run.ledger`, pass the same path to those commands.

## `[backends]`

Each `[backends.<id>]` table defines one backend. `<id>` is how tiers, monitors and judges refer
to it. `type` is required and selects the keys below.

Keys shared by several types:

| Key | Types | Default | Meaning |
|---|---|---|---|
| `type` | all | required | `anthropic`, `openai`, `claude-code`, `command`, `simulated` or `replay`. |
| `model` | anthropic, openai, claude-code (required); simulated (optional) | | Provider model id. |
| `name` | all | see per type | Backend name recorded in the ledger and used in cache keys. |
| `timeout_s` | anthropic, openai, claude-code, command | `600` | Per-call timeout in seconds (> 0). |
| `max_attempts` | anthropic, openai, claude-code, command | `6` | Attempts per call, including the first. Retries use exponential backoff with full jitter (1 s base, 60 s cap) and honor `Retry-After` up to 300 s. |
| `pricing` | all except replay | built-in table | Price override; see [`pricing`](#pricing). |

### `type = "anthropic"`

Official Anthropic SDK. Install with `pip install "shadowgate[anthropic]"`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model` | string | required | e.g. `"claude-haiku-5-5"`. |
| `api_key_env` | string | `"ANTHROPIC_API_KEY"` | Name of the environment variable holding the key. |
| `base_url` | string | SDK default | API base URL. |
| `name` | string | `anthropic:<model>` | |

SDK-internal retries are disabled; `max_attempts` is the only retry loop. A tier's `effort` is
sent as `output_config.effort`. Claude does not return logprobs, so the `logprob` estimator
returns no score on this backend.

### `type = "openai"`

Any OpenAI-compatible `/chat/completions` endpoint over stdlib HTTP: OpenAI, Ollama, vLLM,
llama.cpp server, LM Studio, OpenRouter.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model` | string | required | |
| `base_url` | string | `"https://api.openai.com/v1"` | `/chat/completions` is appended. |
| `api_key_env` | string | `"OPENAI_API_KEY"` | If the variable is unset, no `Authorization` header is sent. For a local server, name a variable you do not set so an OpenAI key is not forwarded to it. |
| `headers` | table of strings | none | Extra HTTP headers. Credential headers are rejected. |
| `name` | string | `openai:<model>` | |

Redirects are refused. A tier's `effort` is ignored by this backend. Retries cover HTTP 408, 409,
425, 429, 5xx, connection errors and timeouts.

### `type = "claude-code"`

Runs `claude -p --output-format json --model <model>` with the prompt on stdin, using the CLI's
own login. Each call runs in a fresh empty directory with tools, settings sources, session
persistence, slash commands and non-explicit MCP servers disabled.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model` | string | required | Use a full model id; short aliases may resolve differently across CLI versions. |
| `executable` | string | `"claude"` | Program name (resolved on `PATH`) or full path. |
| `extra_args` | array of strings | `[]` | Appended after the built-in flags. |
| `default_system` | string | `"You are a helpful assistant."` | System prompt used when a tier sets none; replaces the CLI's agentic default. |
| `name` | string | `claude-code:<model>` | |

A tier's `effort` is passed as `--effort`. Cost is resolved in this order: `pricing` override,
the built-in price table for the model the CLI reports, the CLI's `total_cost_usd` (only when
every model's cost basis is known), otherwise unknown. On a subscription plan the dollar figure
is notional: it is what the same tokens would cost at API prices.

### `type = "command"`

Runs any program once per request. The prompt goes to stdin (the system prompt, if any, first,
followed by a blank line); stdout is the completion text. No shell is involved.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `command` | string or array of strings | required | A string is split with POSIX shell rules on POSIX and passed to `CreateProcess` unchanged on Windows. |
| `cwd` | path | working directory | Relative to the config file. |
| `retryable_exit_codes` | array of integers | `[]` | Non-zero exit codes that are retried. Any other non-zero exit fails the call. Timeouts are always retried. |
| `name` | string | `command:<program stem>` | |

Token usage is estimated as `ceil(chars / 4)`. Cost is computed only when `pricing` is set;
otherwise it is unknown (not zero).

### `type = "simulated"`

A deterministic offline model for demos, tests and threshold experiments. It needs the task list
(the CLI passes `-t`) because it answers from task references. For task difficulty `d`
(`meta.difficulty`, default 0) it is correct with probability `p = sigmoid(skill - d)`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `skill` | number | required | Higher is stronger. The built-in arithmetic generator produces difficulties of about 2 to 17 at its default 1-6 steps, and up to about 31 at 12 steps. |
| `seed` | integer | `run.seed` | Seed for all draws. |
| `overconfidence` | number | `0.0` | Added to the stated confidence on average. |
| `confidence_noise` | number >= 0 | `0.1` | Standard deviation of Gaussian noise on the stated confidence. |
| `discrimination` | number | `0.15` | How much higher a correct sample's confidence is than a wrong one's, on average. |
| `systematic_error` | number in [0, 1] | `0.35` | Probability that a wrong answer is the task's shared "favorite" mistake rather than a fresh one. |
| `pricing` | table | `{ input = 1.0, output = 5.0 }` | USD per million tokens. |
| `latency_s` | number | `0.4` | Reported base latency (never slept). |
| `latency_per_token_s` | number | `0.01` | Reported latency per output token. |
| `emit_confidence` | boolean | `true` | Emit a `CONFIDENCE:` line when the prompt asks for one. |
| `emit_logprobs` | boolean | `true` | Emit token logprobs when requested. |
| `model`, `name` | string | backend id | Backend name is `sim:<name>`, else `sim:<model>`, else `sim:<id>`. |

### `type = "replay"`

Serves recorded completions from a JSONL file and fails on a miss.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `path` | path | required | JSONL lines `{"key": ..., "completion": {...}}` or `{"task_id": ..., "tier": ..., "completion": {...}}`. Relative to the config file. |
| `key_backend` | string | the backend's `name` | Backend name used to compute request keys. |
| `name` | string | `replay:<file stem>` | |

### `pricing`

USD per million tokens. A `pricing` table on a backend always wins over the built-in table
(`shadowgate.pricing.PRICES`, dated in `PRICES_AS_OF`). A model missing from both gets an unknown
cost, never zero.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `input` | number >= 0 | required | |
| `output` | number >= 0 | required | |
| `cache_read` | number >= 0 | `input` | Prompt-cache read rate. |
| `cache_write` | number >= 0 | `input` | Prompt-cache write rate. |
| `long_context` | table | none | A second rate card (same keys, not nested further) used when the prompt exceeds the threshold. |
| `long_context_threshold` | integer >= 0 | `100000` | Prompt tokens (input + cache read + cache write) above which `long_context` applies. |

```toml
pricing = { input = 0.10, output = 0.50 }
```

## `[[tiers]]`

Tiers are listed cheapest first. Every tier except the last needs `threshold` and `confidence`;
the last tier must have neither, because it always serves.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `backend` | string | required | A `[backends]` id. |
| `name` | string | the backend id | Tier name; must be unique. |
| `threshold` | number in [0, 1] | required on non-final tiers | Serve when confidence >= threshold, otherwise escalate. `0` accepts whenever a score exists. |
| `confidence` | table or string | required on non-final tiers | Confidence estimator; see [Confidence estimators](#confidence-estimators). |
| `system` | string | none | System prompt. May be empty. |
| `template` | string | see below | Prompt template; must contain `{prompt}`. Every `{prompt}` is replaced by plain string substitution; other braces are kept as written. |
| `max_tokens` | integer >= 1 | `2048` | Output token limit. |
| `temperature` | number >= 0 | provider default | Not sent when unset. |
| `effort` | string | none | Sent by `anthropic` (`output_config.effort`) and `claude-code` (`--effort`); ignored by other backends. |
| `extractor` | table or string | `[answer].extractor` | Per-tier answer extractor. |

Default template:

```text
{prompt}

Solve the task above. Reason briefly if it helps, then finish with a final line of the form:
ANSWER: <answer>
```

A non-final tier escalates without running its estimator when the completion stopped for
`refusal`, `error` or `max_tokens`, or when the extracted answer is empty. A backend error on any
tier escalates to the next tier.

### Confidence estimators

| `type` | Keys (default) | Notes |
|---|---|---|
| `verbal` | `instruction` (asks for `CONFIDENCE: <number between 0 and 1>`), `allow_words` (`false`) | Appends the instruction to the prompt and parses the last `CONFIDENCE:` line. Accepts `0.85`, `85%`, `85/100`, `8 out of 10`; a bare integer in (1, 100] is a percentage. Unparseable means no score, which escalates. With `allow_words = true`, `high`/`medium`/`low` map to 0.9/0.6/0.3. |
| `logprob` | `aggregate` (`"mean"`; or `"min"`, `"geo_mean"`) | Requests token logprobs and aggregates token probabilities. No score when the backend returns none (Anthropic, Claude Code, command). |
| `self_consistency` | `samples` (`5`), `temperature` (provider default), `concurrency` (`1`) | Draws `samples` extra completions from the same backend. Score = `(1 + agreeing samples) / (1 + successful samples)`, using the `[answer]` comparator. |
| `monitor` | `backend` (required), `prompt`, `system`, `max_tokens` (`1024`), `temperature`, `effort`, `allow_words` (`false`) | A second model reads the question and proposed answer and replies `P(correct): <number>`. A custom `prompt` may use `{question}`, `{answer}` and `{response}` and must contain `{answer}` or `{response}`. |
| `combine` | `members` (required, array of estimator tables), `method` (`"mean"`; or `"min"`, `"max"`, `"weighted"`), `weights` (required for `weighted`), `ignore_missing` (`false`) | Any missing member score makes the combined score missing unless `ignore_missing = true`. |
| `calibrated` | `base` (required, estimator table), `points` (required, array of `[x, y]` in [0, 1]) | Applies a monotone piecewise-linear map to another estimator's score. [`shadowgate calibrate`](cli.md#calibrate) fits the points from an eval-mode run and prints this line. |

The `callable` estimator wraps a Python function and is available only through the
[Python API](guide.md#python-api).

```toml
confidence = { type = "combine", method = "min", members = [
  { type = "verbal" },
  { type = "monitor", backend = "fast" },
] }
```

## `[answer]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `extractor` | table or string | `final_line` | Pulls the answer out of a completion. Tiers can override. |
| `comparator` | table or string | `normalized` | Grades answers against task references and decides agreement between tiers. Also the default audit judge and the comparator `self_consistency` uses. |

### Extractors

| `type` | Keys (default) | Behavior |
|---|---|---|
| `final_line` | `prefix` (`"ANSWER:"`), `alt_prefixes` (`["Final answer:"]`) | Text after the last prefix, case-insensitive, with markdown bold, backticks and a trailing period removed. Falls back to the last non-empty line that is not a `CONFIDENCE:` line. |
| `last_number` | | Last integer, decimal, fraction or scientific number; thousands separators removed. |
| `choice` | `letters` (`"ABCDEFGHIJ"`) | A single option letter, from `ANSWER: (C)` or the last standalone letter. |
| `regex` | `pattern` (required), `group` (1 if the pattern has a group, else 0), `flags` (`""`; any of `i`, `m`, `s`, `x`), `which` (`"last"` or `"first"`) | Stripped match group. |
| `json_field` | `field` (required, dotted path; integer parts index arrays) | Field of the first JSON object in the text. |
| `identity` | | The stripped completion text. |

Extraction never raises; it returns an empty string when nothing is found.

### Comparators

| `type` | Keys (default) | Behavior |
|---|---|---|
| `exact` | | String equality. |
| `normalized` | | Equality after Unicode NFKC, casefolding, and stripping punctuation, whitespace and articles. |
| `numeric` | `rel_tol` (`1e-6`), `abs_tol` (`1e-9`), `fallback_text` (`true`), `percent` (`"either"`; or `"ratio"`, `"strict"`) | Parses integers, decimals, fractions, percentages, currency and thousands separators. Two integers compare exactly; tolerances apply only when either side is a decimal, fraction or scientific value. With `fallback_text = false`, a non-numeric side gives "undecided" instead of a text comparison. |
| `choice` | `letters` (`"ABCDEFGHIJ"`) | Compares extracted option letters. |
| `contains` | | Target appears in the candidate after normalization. |
| `regex` | `mode` (`"fullmatch"` or `"search"`), `ignore_case` (`false`) | The target is a regular expression. |
| `judge` | `backend` (required), `prompt`, `system`, `max_tokens` (`512`), `temperature`, `effort`, `shortcut` (`true`) | A model replies `VERDICT: EQUIVALENT` or `VERDICT: DIFFERENT`. A custom `prompt` must contain `{candidate}` and `{target}` and may contain `{question}`. With `shortcut = true`, answers equal after normalization skip the model call. A reply without a verdict is "undecided". |

An empty candidate never matches a non-empty target. Two empty answers are undecided.

## `[audit]`

Controls shadow audits of accepted (skipped) cases in serve mode. If the section is absent,
audits run inline at the defaults below.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mode` | string | `"inline"` | `inline`: audit during the run. `deferred`: record selected cases as pending; run them with `shadowgate audit --run-pending -c CONFIG`. `off`: no audits and no inclusion probabilities. |
| `rate` | number in (0, 1] | `0.1` | Inclusion probability for confidences outside every stratum. |
| `strata` | array of `[lo, hi, rate]` | `[]` | Bands on the served confidence with their own inclusion probability. First match on `lo <= conf < hi` wins; the band with the largest `hi` (or any `hi >= 1`) also includes `conf == hi`. `lo < hi`, both in [0, 1]; `rate` in (0, 1]. |
| `floor` | number in (0, 1] | `0.01` | Minimum inclusion probability. Must be positive so every accepted case can be audited and the weighted estimate stays unbiased. |
| `tier` | string | the last tier | Audit (reference) tier. Only the last tier's name is accepted. |
| `seed` | integer | `run.seed` | Seed for the reproducible, sha256-based selection of audited cases. |
| `tolerance` | number in (0, 1) | unset | Acceptable skipped-case error or disagreement rate. Sets the audit status and the default for `audit`, `report` and `--fail-on-breach`. Stored with the run, so it survives `export` and `import`. |
| `judge` | table or string | `[answer].comparator` | Comparator for served answer vs audit answer. Use a `judge` comparator for free-text answers. |

The inclusion probability for a case with confidence `c` is `max(floor, stratum_rate(c) or rate)`.
See [the guide](guide.md#audit-sampling-and-weighting) for why the weights matter.

```toml
[audit]
mode = "inline"
rate = 0.1
floor = 0.02
strata = [[0.8, 0.9, 0.3], [0.9, 1.0, 0.1]]
tolerance = 0.05
```

## Task files

`-t/--tasks` takes `.jsonl`, `.json` or `.csv`.

| Field | Aliases | Required | Meaning |
|---|---|---|---|
| `prompt` | `question`, `input` | yes | Task text, substituted into the tier template. |
| `id` | | no | Unique id; generated as `t0001`, `t0002`, ... when missing. Duplicates are an error. |
| `reference` | `answer`, `target`, `label` | no | Ground truth. Enables error rates and `truth = reference` sweeps. |
| anything else | | no | Stored in `meta`. `meta.difficulty` is read by the simulated backend. |

```json
{"id": "q1", "prompt": "What is 17 * 23? Give the answer as a single integer.", "reference": "391"}
```
