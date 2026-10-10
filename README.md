# shadowgate

Confidence-gated LLM cascades that measure how often the cheap tier is wrong on the cases it
answers alone.

[Guide](https://github.com/h-a-forster/shadowgate/blob/main/docs/guide.md) | [Results](https://github.com/h-a-forster/shadowgate/blob/main/docs/results.md) | [CLI](https://github.com/h-a-forster/shadowgate/blob/main/docs/cli.md) | [Configuration](https://github.com/h-a-forster/shadowgate/blob/main/docs/configuration.md)

## Why

A cascade sends each task to a cheap model and escalates to a stronger one when confidence is
low (the "System 1.5" pattern, Oh & Gobet 2024). Such routers usually ship with a cost-saving
figure and no measurement of the error rate on the cases the cheap model answers alone, because
the final tier never sees them. shadowgate samples those skipped cases, re-answers them with the
final tier, and reports the disagreement and error rate with confidence intervals.

## Install

Python 3.11 or newer. No runtime dependencies.

```sh
pip install git+https://github.com/h-a-forster/shadowgate
uv tool install git+https://github.com/h-a-forster/shadowgate   # CLI only, with uv
pip install anthropic   # only for the `anthropic` backend
```

The import name and the CLI command are `shadowgate`. Not published on PyPI.

## Quickstart

Offline demo with two simulated models (400 tasks, no network, no keys; progress lines and
next-step hints trimmed):

```sh
shadowgate demo --out shadowgate-demo
```

```text
shadowgate demo: 400 simulated arithmetic tasks (seed 0); fast tier skill 8.5 (overconfident), final tier skill 24
  eval  run demo-n400-s0-eval: 400/400 done | 140 escalated | 0 audited | $0.1517 serving + $0.1946 audit | 0.8s
  serve run demo-n400-s0-serve: 400/400 done | 140 escalated | 89 audited | $0.1517 serving + $0.0677 audit | 0.7s

Serve mode: confidence threshold 0.55, stratified shadow audits
  escalation rate         35.0% [30.5%, 39.8%]
  answered by fast tier   260 decisions, 89 shadow-audited by 'slow'
  disagreement on skipped 10.8% [5.1%, 19.4%] (weighted by 1/pi)
  error on skipped        12.3% [8.9%, 16.9%] (vs references)
  served accuracy         92.0% [88.9%, 94.3%]
  cost per task           $0.000379
  est. savings vs slow    54.8% [53.6%, 55.9%]
  audit status            breach (tolerance 5.0%)
  audit-only status       breach (disagreement, no references)

Eval mode: every tier on every task, threshold sweep
  only:fast               accuracy 57.0% [52.1%, 61.8%]       cost/task $0.000027
  only:slow               accuracy 100.0% [99.0%, 100.0%]     cost/task $0.000839
  oracle                  accuracy 100.0% [99.0%, 100.0%]     cost/task $0.000433
  recommended threshold   0.89 -> accuracy 99.2% [95.4%, 99.9%], cost/task $0.000487 (held-out n=120)
  cost vs slow-only       -41.9%

The fast tier reports 95% mean confidence on the answers it keeps; the audit estimates 11% of them disagree with the slow tier (95% CI 5-19%), more than the 5% its confidence implies.
```

The demo config sets the threshold too low on purpose. The fast tier answers 65% of tasks without
escalating and is overconfident on them: the audit puts disagreement on skipped cases at 10.8%,
and grading against references puts the error at 12.3%. The lower bound of the error interval
(8.9%) is above the 5% tolerance, so the status is `breach`; the audit alone, without
references, also gives `breach`. The eval-mode sweep recommends 0.89
instead. The demo writes `shadowgate-demo/report.html` and prints the `audit`, `sweep` and
`calibrate` commands to explore the two runs.

A real two-tier config (excerpt of [`examples/anthropic.toml`](https://github.com/h-a-forster/shadowgate/blob/main/examples/anthropic.toml)):

```toml
[run]
max_cost_usd = 5.00
cache = true

[backends.fast]
type = "anthropic"
model = "claude-haiku-5-5"
api_key_env = "ANTHROPIC_API_KEY"

[backends.slow]
type = "anthropic"
model = "claude-opus-5-5"
api_key_env = "ANTHROPIC_API_KEY"

[[tiers]]
name = "haiku"
backend = "fast"
threshold = 0.8
confidence = { type = "verbal" }   # model ends with "CONFIDENCE: <0-1>"

[[tiers]]
name = "opus"
backend = "slow"                   # final tier: always serves

[answer]
comparator = { type = "numeric" }

[audit]
rate = 0.1
floor = 0.02
strata = [[0.8, 0.9, 0.3], [0.9, 1.0, 0.1]]   # audit more near the threshold
tolerance = 0.05
```

Choose a threshold in eval mode, then run and audit in serve mode:

```sh
export ANTHROPIC_API_KEY=...
shadowgate run -c examples/anthropic.toml -t examples/tasks/arithmetic-50.jsonl --mode eval --run-id calib
shadowgate sweep --run-id calib
shadowgate run -c examples/anthropic.toml -t examples/tasks/arithmetic-50.jsonl --run-id prod
shadowgate audit --run-id prod --fail-on-breach
shadowgate report --run-id prod --sweep-run-id calib -o report.html
```

`init`, `calibrate`, `export` and `import` are covered in the
[guide](https://github.com/h-a-forster/shadowgate/blob/main/docs/guide.md#serve-mode-and-eval-mode) and the [CLI reference](https://github.com/h-a-forster/shadowgate/blob/main/docs/cli.md).

## How it works

```text
task -> cheap tier -> confidence >= threshold? --yes--> serve, record pi
                              | no                        |
                              v                           v  selected with probability pi
                         final tier -> serve         final tier re-answers (shadow audit)
                                                          |
                                                          v
                                          disagreement estimate weighted by 1/pi
```

- Each tier is a backend, a prompt template, an answer extractor and, except for the final tier,
  a confidence estimator and threshold. The final tier always serves.
- Every skipped case gets an inclusion probability π from `[audit]` (a base rate, optional
  confidence bands, and a floor > 0). Selection is a reproducible sha256 draw.
- Audited cases are re-answered by the final tier and compared with the served answer. The
  disagreement rate is a Hájek estimate weighted by 1/π, with a Clopper-Pearson interval at the
  effective sample size.
- When tasks have references, every answer is also graded, and the error rate on skipped cases is
  reported alongside.
- Eval mode runs every tier on every task. `sweep` replays the routing rule over a threshold grid
  without new calls, draws the accuracy/cost Pareto frontier, and picks a threshold on 70% of the
  data and validates it on the other 30%.
- Every decision goes to a SQLite ledger as it finishes. Runs resume, respect a spending cap and
  can use a response cache.
- `audit --fail-on-breach` exits 3 when the lower bound of the interval exceeds the tolerance.
  `--level` sets the interval's confidence level (default 0.95).

Backends: Anthropic (official SDK), OpenAI-compatible HTTP (OpenAI, Ollama, vLLM, OpenRouter),
the Claude Code CLI, any local command, a deterministic simulator, and replay of recorded
completions. Confidence signals: `verbal`, `logprob`, `self_consistency`, `monitor`, `combine`
and `calibrated`; the [guide](https://github.com/h-a-forster/shadowgate/blob/main/docs/guide.md#confidence-signals) lists their costs and
trade-offs. Everything the CLI does is also available from the
[Python API](https://github.com/h-a-forster/shadowgate/blob/main/docs/guide.md#python-api).

## Results

Main run (2026-10-10): Claude Haiku 5.5 (effort `low`, verbal confidence, threshold 0.80) in front
of Claude Opus 5.5 or Claude Sonnet 5.5 on 1680 MMLU-Pro questions (120 per subject) and 520
BIG-Bench Hard examples. Both have gold answers graded by exact match. Every tier answered every
task, so the cheap tier's true error on the cases it served alone is known. Run through the
Claude Code CLI for $58 of list-price calls.

| MMLU-Pro | Haiku -> Opus | Haiku -> Sonnet |
|---|---|---|
| Accuracy: Haiku only / final tier only / cascade | 83.0% / 91.2% / 88.1% | 83.0% / 88.3% / 86.2% |
| Served by Haiku alone | 80.0% | 80.0% |
| Haiku error on those, vs gold | 9.3% (7.9-11.0) | 9.3% (7.9-11.0) |
| Haiku disagreement with the final tier on those | 6.1% (4.9-7.5) | 4.5% (3.5-5.7) |
| Audit-only estimate (~465 weighted audits) | 5.1% (3.1-7.8) | 4.1% (2.3-6.6) |
| Haiku errors the final tier shared (audit cannot see them) | 44% | 61% |
| Saving vs final tier alone: serving / with audit | 72% / 45% | 69% / 41% |

The audit estimates disagreement accurately: re-drawn 1000 times offline, its interval covered
the true disagreement rate 99.7-99.8% of the time. It covered the gold-graded error only 41.6%
(Opus) and 0.7% (Sonnet) of the time, because the final tier repeats many of Haiku's mistakes. A
weaker final tier repeats more of them and makes the cheap tier look better. On BBH, the Haiku ->
Sonnet audit reported `ok` against a 5% tolerance (0.8%, upper bound 4.8%) while the gold-graded
error was 5.4% (3.7-7.8).

Part of that gap is label noise, not missed errors. In 13 hand-checked MMLU-Pro cases where both
models gave the same "wrong" answer, 5 gold labels were wrong, 6 were ambiguous, and 2 were real
errors. Read the audit as disagreement with a final tier that is itself wrong 9-12% of the time
on MMLU-Pro. Do not read it as error.

A drift check (offline): a threshold tuned on STEM questions (0.88, 3.3% error) gave 12.3% error
on the humanities questions Haiku served alone. At that volume (~41 audits) the audit never said
`ok`, but it confirmed the breach in only 8-11% of re-draws.

An earlier run on generated arithmetic, where Opus was 100% correct, is also in the results
document. Setup, BBH numbers, coverage, drift and limitations:
[docs/results.md](https://github.com/h-a-forster/shadowgate/blob/main/docs/results.md).

## Documentation

- [Guide](https://github.com/h-a-forster/shadowgate/blob/main/docs/guide.md): concepts, confidence signals, audit weighting, reading the numbers,
  choosing a tolerance and audit rate, Python API.
- [Configuration](https://github.com/h-a-forster/shadowgate/blob/main/docs/configuration.md): every TOML key.
- [CLI](https://github.com/h-a-forster/shadowgate/blob/main/docs/cli.md): every command, option and exit code.
- [Examples](https://github.com/h-a-forster/shadowgate/blob/main/examples/README.md): configs for each backend.
- [Results](https://github.com/h-a-forster/shadowgate/blob/main/docs/results.md): the MMLU-Pro, BIG-Bench Hard and arithmetic experiments and how to reproduce them.
- [Design](https://github.com/h-a-forster/shadowgate/blob/main/docs/design.md): architecture and module contracts.

## Limitations

- Disagreement with the final tier is not error unless tasks carry references; a mistake both
  tiers make is invisible to the audit. On MMLU-Pro, 44-61% of the cheap tier's errors against
  gold labels were shared with the final tier (some of them label noise).
- Costs are estimates from a built-in price table or your `pricing` override. Claude Code CLI
  costs on a subscription are notional.
- A threshold chosen on an eval set holds for tasks like that set. The serve-mode audit is how
  drift shows up.

More in the [guide](https://github.com/h-a-forster/shadowgate/blob/main/docs/guide.md#limitations).

## Related work

Cascades with learned or confidence-based deferral are well studied (FrugalGPT, AutoMix, RouteLLM
and others); shadowgate adds continuous, weighted measurement of the cases the cheap tier answers
alone. The closest existing tool is [safeswap](https://github.com/shubh-tiwari/safeswap), which
shadow samples cheap-model requests and reports quality loss with intervals. See
[docs/related-work.md](https://github.com/h-a-forster/shadowgate/blob/main/docs/related-work.md).

## Contributing and license

See [CONTRIBUTING.md](https://github.com/h-a-forster/shadowgate/blob/main/CONTRIBUTING.md) and [SECURITY.md](https://github.com/h-a-forster/shadowgate/blob/main/SECURITY.md). MIT licensed; see
[LICENSE](https://github.com/h-a-forster/shadowgate/blob/main/LICENSE).
