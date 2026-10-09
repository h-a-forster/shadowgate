# Guide

This page explains what shadowgate measures and how to read the numbers. For every config key see
[configuration.md](configuration.md); for every command see [cli.md](cli.md).

## The problem

A confidence-gated cascade sends each task to a cheap model first. If the cheap model's
confidence clears a threshold, its answer is served. Otherwise the task escalates to a stronger,
more expensive model. Oh & Gobet (2024) call this kind of metacognitive gate "System 1.5".

The cost saving is easy to measure: count how many tasks stopped at the cheap tier. The quality
cost is not. The cases the cheap tier keeps are, by construction, never seen by the strong tier,
so nobody observes how often they are wrong. shadowgate estimates that number with a confidence
interval, in production and offline.

## Tiers

A tier is one backend plus a prompt template, an answer extractor, and, for every tier except
the last, a confidence estimator and a threshold.

```text
task -> tier 0 (cheap) --conf >= t0--> serve
            | conf < t0, no score, refusal, max_tokens, empty answer, or backend error
            v
        tier 1 --conf >= t1--> serve
            |
            v
        last tier (strong) -----------> serve (always)
```

- Tiers are ordered cheapest first. The last tier has no threshold and always serves.
- A missing confidence score never accepts. An unparseable `CONFIDENCE:` line escalates rather
  than being read as 0 or 1.
- A backend error escalates to the next tier. If the last tier fails, the decision is recorded
  with an error.
- Vocabulary: a **skipped case** (or kept, or accepted case) is a task served by a non-final tier.
  The **reference tier** (or audit tier) is the last tier.

More than two tiers work. The audit always compares against the last tier.

## Confidence signals

| Signal | Extra calls per task at that tier | Needs | Trade-offs |
|---|---|---|---|
| `verbal` | 0 | a model that follows the `CONFIDENCE:` format | Cheapest. Often overconfident (Xiong et al., 2024). A reply without a parseable `CONFIDENCE:` line escalates. |
| `logprob` | 0 | a backend that returns token logprobs (OpenAI-compatible servers; not Anthropic or the Claude Code CLI) | No prompt change. Mean token probability over the whole reply mixes reasoning tokens with answer tokens; `min` is stricter, `geo_mean` sits between. |
| `self_consistency` | `samples` (default 5) | sampling at nonzero temperature to be informative | Strong signal on tasks with a short checkable answer. Multiplies the cheap tier's cost by about `1 + samples`. Models that repeat the same mistake agree with themselves. |
| `monitor` | 1 on the monitor backend | a second model (often the cheap one again) | Separates answering from grading. Costs one extra call; the monitor's own calibration is unknown until you sweep it. |
| `combine` | sum of members | members | `min` is conservative: escalate if any signal is low. `ignore_missing` controls whether one missing member blocks acceptance. |
| `calibrated` | same as `base` | eval-mode data to fit the map | Remaps scores so a threshold of 0.9 means about 90% correct. The map is monotone, so it keeps the ranking (up to ties) and the accuracy/cost frontier; it changes which threshold value lands where. |

What matters for routing is ranking: a higher score should mean a more likely correct answer.
`shadowgate sweep` reports AUROC and AURC (ranking quality) and ECE and Brier score (calibration)
for each non-final tier, so you can compare signals on your own tasks before choosing one.

## Serve mode and eval mode

| | Serve mode (`--mode serve`, default) | Eval mode (`--mode eval`) |
|---|---|---|
| Tiers called | until the first accepted tier | every tier, every task |
| Cost | what production costs, plus sampled audits | sum of all tiers |
| Audit | a random sample of skipped cases, weighted by 1/π | every skipped case, π = 1 |
| Output | `audit`: skipped-case error estimate for the configured threshold | `sweep`: any threshold, simulated after the fact |
| Use for | monitoring a deployed threshold | choosing a threshold |

Eval mode records what the router would have done at the configured thresholds, and also every
tier's answer and confidence. `sweep` then replays the routing rule for every threshold on a grid
without new model calls. A typical sequence:

1. Run eval mode on a few hundred representative tasks, ideally with references.
2. `shadowgate sweep` to pick a threshold. Put it in the config.
3. Run serve mode on live or new traffic with audits on.
4. `shadowgate audit --fail-on-breach` in CI or a scheduled job.

## Threshold selection

`sweep` evaluates each threshold vector on all usable decisions and marks the Pareto frontier:
points where no other point is both cheaper and more accurate. It also reports three baselines:
`only:<tier>` for each tier, and `oracle`, which serves from the cheapest tier that happens to be
correct. The oracle's cost is a lower bound no real router reaches.

The recommendation is chosen on a random selection split (70% by default) and re-evaluated on the
held-out split (30%). Picking the best of 80+ thresholds on the same data you report on is
optimistic; the held-out row is the number to trust. When the held-out accuracy falls below the
selection target, the output says so. With `--holdout 0` the output states that its estimates are
in-sample.

Objectives:

- `max-savings`: cheapest point within `--max-drop` (default 1 point) of the best single tier.
- `min-accuracy`: cheapest point with accuracy >= `--min-accuracy`.
- `max-accuracy`: most accurate point with cost per task <= `--budget`.

When tasks have references, accuracy means correctness. Without references it means agreement
with the last tier, and the last tier counts as correct by definition (`truth: audit-tier`).

## Audit sampling and weighting

In serve mode, each skipped case is selected for a shadow audit with a known inclusion
probability π. Selection is a reproducible draw from sha256 of the audit seed, run id and task
id. Selected cases are re-answered by the last tier and the two answers are compared with the
audit judge (by default the `[answer]` comparator).

π depends on the served confidence through `[audit] strata`:

```toml
[audit]
rate = 0.1                                  # π outside the bands
strata = [[0.8, 0.9, 0.3], [0.9, 1.0, 0.1]] # π = 0.3 in [0.8, 0.9), 0.1 in [0.9, 1.0]
floor = 0.02                                # π never goes below this
```

Borderline acceptances are where errors concentrate, so they get audited more. That makes the
audited sample unrepresentative: a plain average over audited cases would over-count the
borderline band and overstate the error rate. shadowgate weights each audited case by 1/π and
reports the Hájek ratio estimate

```text
p̂ = Σ (yᵢ / πᵢ) / Σ (1 / πᵢ)      over audited cases, yᵢ = 1 if the answers disagree
```

which is consistent for the disagreement rate over all skipped cases (Horvitz & Thompson, 1952;
Hájek, 1971). The interval is a Clopper-Pearson interval evaluated at an effective sample size
`n_eff` (Korn & Graubard), where `n_eff` is the smaller of Kish's effective size and a
variance-matched size. Unequal weights shrink `n_eff` below the number of audits, so the
interval is wider than an unweighted one would be.

Why the floor must be positive: a case with π = 0 can never be audited, so no weighting can
recover its error rate. If the floor is very small, the few audits in that band carry large
weights (1/0.01 = 100) and the estimate leans on them. The output flags both situations.

Every skipped case records its π, including the ones not selected, so the estimator knows the
full population it is weighting back to.

### Audit modes

- `inline`: the audit runs in the same worker right after serving. Audit latency does not count
  toward serving latency, but it does use worker slots and spend.
- `deferred`: selected cases are marked pending; `shadowgate audit --run-pending -c CONFIG`
  runs them later, for example off-peak.
- `off`: no audits, no inclusion probabilities. `audit` can still report escalation, cost and,
  if tasks have references, error rates.

## Reading the numbers

Output of `shadowgate audit` on the demo serve run (400 simulated tasks, threshold 0.80):

```text
Status: INCONCLUSIVE - The interval for skipped-case error vs references straddles tolerance 5%.
  About 31 more audits would resolve it if the rate holds.

The fast tier (fast) answered 47% of 400 tasks without escalating. On those, it disagrees with slow
  on an estimated 0% (95% CI 0.0-7.4%, 51 audits, n_eff 48). Against reference answers, the kept
  answers are wrong on 2.1% (95% CI 0.8-5.3%, 189 graded). That is about 4.0 wrong answers among the
  189 kept (1.6-10.0).
```

| Field | Meaning |
|---|---|
| disagreement on kept | Weighted share of skipped cases where the last tier gives a different answer. Needs no references. |
| error on kept | Share of skipped cases whose served answer is wrong against the task reference. Every skipped case with a reference is graded, so this is an unweighted Wilson interval over all of them, not only the audited ones. |
| expected wrong-but-kept | The error (or disagreement) rate times the number of skipped cases, with its interval. |
| escalation rate | Share of tasks that went past the first tier. |
| served accuracy | Accuracy of what was actually served, all tiers together (needs references). |
| est. savings vs \<last tier\> | 1 - serving cost / estimated cost of sending every task to the last tier. The all-last-tier cost is estimated from last-tier attempts on escalated and audited cases, weighted by 1/π. Excludes audit cost. |
| audit overhead | Audit spend as a share of serving spend. |
| by confidence bin | Disagreement and error per confidence band. Shows whether errors cluster near the threshold. |
| status | `ok`, `breach`, `inconclusive`, `no-data` or `n/a`; see [cli.md](cli.md#audit). |

In the example above, disagreement is 0% while error is 2.1%. The simulated slow tier sometimes
makes the same mistake as the fast tier, so the two agree on a wrong answer. Disagreement is a
lower bound on error in that case. This is the main reason to keep references for at least a
calibration set.

The status uses the error rate when every skipped case has a reference, otherwise the
disagreement rate. `inconclusive` means the interval still contains the tolerance; the output
estimates how many more audits would move it to `ok` or `breach` if the observed rate holds.

## Choosing a tolerance and an audit rate

The tolerance is a product decision: the share of kept answers you accept being wrong (or, without
references, different from the strong model). Set it before looking at results.

The audit rate follows from how tight the interval must be. With a 95% interval, the number of
effective audits needed for a half-width `h` around a rate `p` is about
`1.96² · p(1 - p) / h²` (`shadowgate.stats.required_n`):

| Expected rate p | Half-width h | Effective audits |
|---|---|---|
| 2% | ±1 point | 753 |
| 2% | ±2 points | 189 |
| 5% | ±2 points | 457 |
| 5% | ±5 points | 73 |
| 10% | ±5 points | 139 |

Divide by the number of skipped cases per period to get a rate. With 5,000 skipped cases per day
and a target of 457 effective audits per week, a uniform rate of about 1.3% is enough. Stratified
sampling needs more raw audits for the same `n_eff`, because unequal weights reduce it; it pays
off when errors concentrate in the oversampled band.

Audit cost is roughly `audit rate × skipped cases × last-tier cost per call`. At a 10% rate on a
cascade that keeps half its traffic, audits add about 5% of an all-strong-model bill.

To read a breach early, oversample the band nearest the threshold. To keep cost down on a
tier with very high confidence, use a low rate in the top band with a floor of at least 0.01.

## CI gate

```sh
shadowgate run -c cascade.toml -t regression.jsonl --run-id "ci-$GIT_SHA"
shadowgate audit --run-id "ci-$GIT_SHA" --tolerance 0.05 --fail-on-breach
```

Exit code 3 fails the job only on `breach`: the lower bound of the interval is above the
tolerance. `inconclusive` exits 0. If you want to fail on `inconclusive` too, read `status` from
`--json` output.

## Python API

The CLI is a thin layer over the library. The same cascade in Python, offline
(from [`examples/python_api.py`](../examples/python_api.py)):

```python
import shadowgate as sg
from shadowgate.backends import SimulatedBackend
from shadowgate.compare import Numeric
from shadowgate.confidence import Verbal
from shadowgate.datasets import arithmetic
from shadowgate.extract import FinalLine
from shadowgate.pricing import Pricing

tasks = arithmetic(300, seed=0)
fast = SimulatedBackend("fast", skill=8.0, tasks=tasks, overconfidence=0.1, pricing=Pricing(0.5, 2.5))
strong = SimulatedBackend("strong", skill=16.0, tasks=tasks, pricing=Pricing(5.0, 25.0))

cascade = sg.Cascade(
    [sg.Tier("fast", fast, threshold=0.8, estimator=Verbal()), sg.Tier("strong", strong)],
    extractor=FinalLine(),
    comparator=Numeric(),
    audit=sg.AuditPolicy(rate=0.2, strata=((0.8, 0.9, 0.5), (0.9, 1.0, 0.2)), floor=0.05),
)

with sg.Ledger("tmp/api-ledger.sqlite") as ledger:
    sg.run_tasks(cascade, tasks, ledger, run_id="api", workers=4)
    summary = sg.summarize(ledger.decisions("api"), tolerance=0.05)
print(summary.status, summary.disagreement)
```

Replace the simulated backends with `sg.backends.AnthropicBackend("claude-haiku-5-5")` or
`sg.backends.OpenAICompatBackend(...)` for real models. `sg.load_config(path).build(tasks=...)`
returns the cascade and run settings from a TOML file. The `callable` confidence estimator
(`shadowgate.confidence.CallableEstimator`) wraps any Python function
`fn(task, completion, answer) -> float | None` and is only available here.

## Limitations

- **The audit tier is a proxy.** Disagreement with the last tier is not error unless tasks carry
  references. When both tiers make the same mistake, disagreement understates error, as the demo
  shows.
- **Comparator and judge errors.** Agreement is decided by a comparator. A `numeric` comparator
  on free text, or a model `judge` that misreads equivalent answers, biases the rate in either
  direction. Undecided comparisons are excluded and counted; if they are not random, the estimate
  is biased.
- **π floor.** Unbiasedness needs every skipped case to have π > 0. Very small π values give
  heavy weights and unstable estimates; the output warns when a few audits carry much of the
  weight or when `n_eff` is below 30.
- **Non-random missing audits.** Audits that fail, stay pending or are undecided are dropped. The
  estimate assumes they are missing at random given π.
- **Costs are estimates.** Costs come from a built-in price table (dated in
  `shadowgate.pricing.PRICES_AS_OF`) or your `pricing` override. Unknown prices stay unknown and
  are excluded from totals, with a note. Prices change.
- **Claude Code CLI cost is notional on subscriptions.** It reports what the tokens would cost at
  API rates, not what a subscription charges.
- **Distribution shift.** A threshold chosen in eval mode holds for tasks like the eval set. The
  serve-mode audit is how you detect when it stops holding.
- **Sequential checks.** Running `audit` repeatedly on a growing run and stopping at the first
  `ok` inflates the false-acceptance rate. Fix the run size, or treat repeated checks as
  monitoring rather than a test.
