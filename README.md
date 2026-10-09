# shadowgate

Confidence-gated LLM model cascades that measure how often the cheap tier is wrong on the cases
it keeps.

## Why

A cascade sends each task to a cheap model and escalates to a stronger one when confidence is
low (the "System 1.5" pattern, Oh & Gobet 2024). Such routers usually ship with a cost-saving
figure and no measurement of the error rate on the cases the cheap model keeps, because the
strong model never sees them. shadowgate samples those kept cases, re-answers them with the
strong model, and reports the disagreement and error rate with confidence intervals.

## Install

Python 3.11 or newer. No runtime dependencies.

```sh
pip install shadowgate
pip install "shadowgate[anthropic]"   # adds the official Anthropic SDK
uv tool install shadowgate            # CLI only, with uv
uv add "shadowgate[anthropic]"        # as a project dependency, with uv
```

## Quickstart

Offline demo with two simulated models (400 tasks, no network, no keys; output trimmed):

```sh
shadowgate demo --out shadowgate-demo
```

```text
Serve mode: confidence threshold 0.80, stratified shadow audits
  escalation rate         52.8% [47.9%, 57.6%]
  kept by fast tier       189 decisions, 51 shadow-audited by 'slow'
  disagreement on kept    0.0% [0.0%, 7.4%] (weighted by 1/pi)
  error on kept           2.1% [0.8%, 5.3%] (vs references)
  served accuracy         96.0% [93.6%, 97.5%]
  cost per task           $0.000529
  est. savings vs slow    36.2% [34.4%, 37.9%]
  audit status            inconclusive (tolerance 5.0%; ~31 more audits to resolve)

Eval mode: every tier on every task, threshold sweep
  only:fast               accuracy 54.2% [49.4%, 59.1%]       cost/task $0.000027
  only:slow               accuracy 97.0% [94.8%, 98.3%]       cost/task $0.000839
  oracle                  accuracy 97.0% [94.8%, 98.3%]       cost/task $0.000455
  recommended threshold   0.74 -> accuracy 94.2% [88.4%, 97.1%], cost/task $0.000565 (held-out n=120)
```

It writes `shadowgate-demo/report.html`. Disagreement is 0% while error is 2.1%: the strong tier
sometimes makes the same mistake, so disagreement alone understates error.

A real two-tier config (excerpt of [`examples/anthropic.toml`](examples/anthropic.toml)):

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
strata = [[0.8, 0.9, 0.3], [0.9, 1.0, 0.1]]   # audit borderline acceptances more
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

`shadowgate init` writes an offline starter config and 20 tasks to try the same commands without
a key.

## How it works

```text
task -> cheap tier -> confidence >= threshold? --yes--> serve, record pi
                              | no                        |
                              v                           v  selected with probability pi
                         strong tier -> serve        strong tier re-answers (shadow audit)
                                                          |
                                                          v
                                          disagreement estimate weighted by 1/pi
```

- Each tier is a backend, a prompt template, an answer extractor and, except for the last tier, a
  confidence estimator and threshold. The last tier always serves.
- Every kept case gets an inclusion probability π from `[audit]` (a base rate, optional
  confidence bands, and a floor > 0). Selection is a reproducible sha256 draw.
- Audited cases are re-answered by the last tier and compared with the served answer. The
  disagreement rate is a Hájek estimate weighted by 1/π, with a Clopper-Pearson interval at the
  effective sample size.
- When tasks have references, every answer is also graded, and the error rate on kept cases is
  reported alongside.
- Eval mode runs every tier on every task. `sweep` replays the routing rule over a threshold grid
  without new calls, draws the accuracy/cost Pareto frontier, and picks a threshold on 70% of the
  data and validates it on the other 30%.
- Every decision goes to a SQLite ledger as it finishes. Runs resume, respect a spending cap and
  can use a response cache.
- `audit --fail-on-breach` exits 3 when the lower bound of the interval exceeds the tolerance.

Backends: Anthropic (official SDK), OpenAI-compatible HTTP (OpenAI, Ollama, vLLM, OpenRouter),
the Claude Code CLI, any shell command, a deterministic simulator, and replay of recorded
completions.

## Results

<!-- RESULTS -->

## Python API

```python
import shadowgate as sg
from shadowgate.backends import AnthropicBackend
from shadowgate.compare import Numeric
from shadowgate.confidence import Verbal

cascade = sg.Cascade(
    [
        sg.Tier("haiku", AnthropicBackend("claude-haiku-5-5"), threshold=0.8, estimator=Verbal()),
        sg.Tier("opus", AnthropicBackend("claude-opus-5-5")),
    ],
    comparator=Numeric(),
    audit=sg.AuditPolicy(rate=0.1, strata=((0.8, 0.9, 0.3), (0.9, 1.0, 0.1)), floor=0.02),
)
tasks = sg.load_tasks("examples/tasks/arithmetic-50.jsonl")
with sg.Ledger(".shadowgate/ledger.sqlite") as ledger:
    sg.run_tasks(cascade, tasks, ledger, run_id="api")
    summary = sg.summarize(ledger.decisions("api"), tolerance=0.05)
print(summary.status, summary.disagreement)
```

An offline version with simulated models is in [`examples/python_api.py`](examples/python_api.py).

## Confidence signals

| Signal | Extra cost | Needs | Notes |
|---|---|---|---|
| `verbal` | none | model follows a `CONFIDENCE:` format | Cheapest; often overconfident. Unparseable replies escalate. |
| `logprob` | none | backend returns token logprobs | OpenAI-compatible servers only; not Anthropic or Claude Code. |
| `self_consistency` | `samples` extra calls (default 5) | nonzero temperature | Strong on short checkable answers; repeated mistakes still agree. |
| `monitor` | 1 call on a second backend | a monitor model | Separates answering from grading. |
| `combine` | sum of members | members | `min`, `max`, `mean` or `weighted`. |
| `calibrated` | as its base | eval-mode data | Monotone remap so thresholds read as probabilities. |

`shadowgate sweep` reports AUROC, AURC, ECE and Brier score per tier to compare signals on your tasks.

## Documentation

- [Guide](docs/guide.md): concepts, audit weighting, reading the numbers, choosing a tolerance
  and audit rate.
- [Configuration](docs/configuration.md): every TOML key.
- [CLI](docs/cli.md): every command, option and exit code.
- [Examples](examples/README.md): configs for each backend.
- [Design](docs/design.md): architecture and module contracts.

## Limitations

- The audit tier is a proxy. Disagreement with the strong model is not error unless tasks carry
  references, and identical mistakes in both tiers are invisible to it.
- Agreement is decided by a comparator or a model judge. Their mistakes bias the rate.
- Every kept case needs π > 0 (`[audit] floor`). Small π values give heavy weights and wide,
  unstable intervals; the output flags them.
- Costs are estimates from a built-in price table (dated in `shadowgate.pricing.PRICES_AS_OF`)
  or your `pricing` override. Unknown prices stay unknown, never zero.
- Claude Code CLI costs are notional on subscription plans: they price the tokens at API rates.
- A threshold chosen on an eval set holds for tasks like that set. The serve-mode audit is how
  drift shows up.

## Related work

Cascades with learned or confidence-based deferral are well studied (FrugalGPT, AutoMix, RouteLLM
and others); shadowgate adds continuous, weighted measurement of the cases the cheap tier keeps.
The closest existing tool is [safeswap](https://github.com/shubh-tiwari/safeswap), which shadow
samples cheap-model requests and reports quality loss with intervals. See
[docs/related-work.md](docs/related-work.md).

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). MIT licensed; see
[LICENSE](LICENSE).
