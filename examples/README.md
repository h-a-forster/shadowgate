# Examples

Every config here is a complete `shadowgate` TOML file. Run one against the bundled task file,
then inspect the result:

```sh
shadowgate run -c examples/<config>.toml -t examples/tasks/arithmetic-50.jsonl
shadowgate audit                 # skipped-case disagreement/error with confidence intervals
shadowgate report -o report.html
```

Decisions go to `.shadowgate/ledger.sqlite` in the working directory unless `--ledger` (or
`[run] ledger`) says otherwise. Add `--mode eval` to `run` to answer every task with every tier,
then `shadowgate sweep` to choose thresholds from the accuracy/cost curve.

| File | Needs | What it shows |
|---|---|---|
| [`simulated.toml`](simulated.toml) | nothing (offline) | Two simulated models: a cheap, overconfident fast tier and a near-perfect `slow` tier. Verbal confidence, numeric answer comparison, inline audits with confidence strata and a 5% tolerance. The threshold (0.55) is deliberately too low, so on enough tasks the audit reports a breach. Deterministic; used by `shadowgate demo` and the test suite. |
| [`basic.toml`](basic.toml) | `ANTHROPIC_API_KEY`, `pip install anthropic` | The smallest useful config: two Anthropic tiers, a verbal-confidence threshold, default audit settings. |
| [`anthropic.toml`](anthropic.toml) | `ANTHROPIC_API_KEY`, `pip install anthropic` | A fuller Anthropic setup: per-tier effort and token limits, retries, a spending cap, response cache, stratified inline audits, an explicit audit tier and tolerance. Comments show a monitor-model confidence alternative and a model-graded judge. |
| [`claude-code.toml`](claude-code.toml) | the `claude` CLI, logged in | The same cascade through `claude -p`, using the CLI's own login, with cost from the shadowgate price table. Each call runs in an empty directory with tools disabled. |
| [`openai-compatible.toml`](openai-compatible.toml) | a local OpenAI-compatible server (Ollama, vLLM, ...) | Two local models with token-logprob confidence on the small one, zero pricing, and deferred audits (`shadowgate audit --run-pending -c examples/openai-compatible.toml` runs them later). |
| [`python_api.py`](python_api.py) | nothing (offline) | The same ideas without a config file: builds a two-tier simulated cascade in Python, routes 300 tasks into a temporary ledger and prints the audit summary. Comments show where a real backend goes. |
| [`tasks/arithmetic-50.jsonl`](tasks/arithmetic-50.jsonl) | | 50 generated multi-step word problems with integer answers and a `difficulty` per task (`shadowgate datasets make arithmetic -n 50 --seed 0 -o tasks.jsonl` makes more). |

Run the Python example with:

```sh
uv run python examples/python_api.py
```

API keys are only ever read from the environment variable named by `api_key_env`; a literal
`api_key` in a config file is rejected.
