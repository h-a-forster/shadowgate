# Results: Haiku 5.5 -> Opus 5.5 on multi-step arithmetic

One real run, 2026-10-09. Full report: [`results/arithmetic-haiku-opus.html`](results/arithmetic-haiku-opus.html)
(also on [GitHub Pages](https://h-a-forster.github.io/shadowgate/results/arithmetic-haiku-opus.html)).
Raw decisions: [`experiments/arithmetic-haiku-opus/`](../experiments/arithmetic-haiku-opus/).

## Setup

| | |
|---|---|
| Tasks | 200 generated word problems, 6-12 arithmetic steps, integer answers (`datasets make arithmetic -n 200 --seed 2026 --min-steps 6 --max-steps 12`) |
| Fast tier | `claude-haiku-5-5`, effort `low` |
| Final tier | `claude-opus-5-5`, default effort |
| Confidence | verbal: the fast tier ends with `CONFIDENCE: <0-1>` |
| Threshold | 0.80 |
| Grading | numeric comparison against the generator's reference answer |
| Audit (serve run) | inline; inclusion probability 0.5 for confidence in [0.80, 0.95), 0.2 for [0.95, 1.00] |
| Backend | Claude Code CLI 2.1.287; cost from the shadowgate price table (USD list prices) |

Two runs share one response cache. The eval run calls both tiers on every task. The serve run
routes normally and audits a sample of the skipped cases; every call it makes is a cache hit, so the
two runs see identical model outputs. In particular, each serve-run audit replays the Opus answer
from the eval run, so the audit does not capture run-to-run variation in the final tier's answers;
a fresh Opus call could disagree with its own earlier answer.

## Results

| Policy | Accuracy (95% CI) | Cost per task | Escalated |
|---|---|---|---|
| Haiku only | 95.0% (91.0-97.3) | $0.00047 | - |
| Opus only | 100.0% (98.1-100.0) | $0.0115 | - |
| Cascade, threshold 0.80, serving only | 98.5% (95.7-99.5) | $0.0016 | 9.5% |
| Cascade, threshold 0.80, serving + audit (serve run) | 98.5% (95.7-99.5) | $0.0046 | 9.5% |
| Cascade, threshold 0.85 (sweep pick) | 99.0% (96.4-99.7) | $0.0024 | 16.5% |
| Oracle router (cost lower bound) | 100.0% | $0.0010 | 5.0% |

At threshold 0.80 the cascade's serving cost is 86.5% less than Opus alone (95% CI 85.6-87.3%)
and it loses 1.5 points of accuracy. That figure excludes the audit. The serve run spent $0.3136 on
serving and $0.6030 on audits ($0.9166 in all, $0.0046 per task, audit overhead 192%); all-in, the
saving against Opus alone ($0.0115 per task) is 60%. The other rows exclude audit spend. The sweep's `max-savings` objective (accuracy within 1 point of Opus) picked
0.85 on a 140-task selection split. On the 60 held-out tasks it scored 98.3%, just under the target;
the report flags this. With 60 held-out tasks the interval is wide (91.1-99.7%).

Verbal confidence ranked answers well: AUROC 0.91 for "Haiku is correct". It was underconfident:
mean confidence 0.93 on skipped cases implies about 7% errors; the measured error was 1.7% (ECE
0.063).

## What the audit found

Haiku answered 181 of 200 tasks without escalating (the skipped cases). The serve run audited 51
of them with Opus.

| Confidence bin | Skipped | Audited | Disagreement with Opus (95% CI) | Error vs reference (95% CI) |
|---|---|---|---|---|
| [0.80, 0.90) | 44 | 15 | 13.3% (1.7-40.5) | 4.5% (1.3-15.1) |
| [0.90, 0.95) | 30 | 16 | 0% (0.0-20.6) | 3.3% (0.6-16.7) |
| [0.95, 1.00] | 107 | 20 | 0% (0.0-16.8) | 0% (0.0-3.5) |
| All skipped | 181 | 51 | 2.5% (0.1-12.7), weighted | 1.7% (0.6-4.8) |
| Status vs 5% tolerance | | | inconclusive (audit only) | ok (all 181 graded) |

- Three skipped cases were wrong, at confidence 0.80, 0.85 and 0.93. All three were off by a factor
  of about 3 to 4 (for example 816 where the reference is 3432).
- Seven of Haiku's ten errors had confidence 0.60 or below and escalated.
- The weighted disagreement estimate (2.5%) agrees with the full error count (1.7%) but its interval
  is much wider, because it rests on 51 audits rather than 181 graded answers. On production traffic
  without reference answers, the audit estimate is what you get.
- Two statuses, against a 5% tolerance. With references, `shadowgate audit` grades all 181
  skipped cases and reports `ok` (upper bound 4.8%); this is the status the published report
  shows, and it does not depend on the audit sample. From the 51 audits alone the status is
  `inconclusive` (upper bound 12.7%); that is the status a run without references would get. The
  report and `shadowgate audit` now print both when references are present.
- Audit spend was 1.9x the serving spend. Audits run on Opus, which costs about 25x Haiku per task;
  the audit rates here were set high so a 200-task run yields a usable sample. At a flat 5% audit
  rate the overhead would be about 0.3x.

## Coverage check (offline)

Because the eval run called Opus on every task, the audit can be re-drawn offline.
[`simulate_audit.py`](../experiments/arithmetic-haiku-opus/simulate_audit.py) draws the serve-mode
audit sample 1000 times (seeds 0-999, same strata: inclusion probability 0.5 below confidence 0.95,
0.2 above), strips the references so only the audit counts, and checks the weighted 95% interval
against the true skipped-case rate (3 of 181, 1.66%, graded against references). No model calls:

```sh
uv run python experiments/arithmetic-haiku-opus/simulate_audit.py
```

| | |
|---|---|
| Audits per draw | mean 58.3 (40-80) |
| Mean point estimate | 1.69% (true 1.66%) |
| Interval covers the true rate | 1000 of 1000 draws (100.0%) |
| Draws with no disagreement observed | 11.0% |
| Upper bound | median 10.3%, 5th-95th percentile 7.7-13.5% |
| Status vs 5% tolerance | inconclusive in 1000 of 1000 draws |

The weighted estimate is unbiased here and the Korn-Graubard interval is conservative (it never
missed, against 95% nominal). The cost is width: at this audit size the upper bound never fell
below 5%, so the audit alone could not have confirmed the tolerance on any draw. This checks one
population with three errors; it is not a general coverage result.

## Reproduce

```sh
sh experiments/arithmetic-haiku-opus/run.sh
```

The script regenerates the task file, runs eval and serve modes, sweeps and writes the report. It
needs a logged-in Claude Code CLI and makes about 400 model calls (about $2.40 at list prices).
To rebuild the report from the committed decisions without model calls:

```sh
shadowgate import experiments/arithmetic-haiku-opus/decisions-eval.jsonl --ledger tmp/ledger.sqlite
shadowgate import experiments/arithmetic-haiku-opus/decisions-serve.jsonl --ledger tmp/ledger.sqlite
shadowgate report --ledger tmp/ledger.sqlite --run-id serve --sweep-run-id eval -o report.html
```

## Limitations

- **Opus was 100% correct on all 200 tasks, so disagreement with Opus equals error by
  construction.** This is the main limitation of the result: it does not exercise a final tier
  that makes its own mistakes, where disagreement and error diverge and errors both tiers share go
  unseen.
- One task family, 200 tasks, one run. Read the confidence intervals, not the point estimates.
- Arithmetic has exact reference answers. Most production tasks do not, and there the audit reports
  disagreement with the final tier, not error.
- Costs are list-price estimates from the shadowgate price table. Through the Claude Code CLI on a
  subscription, no per-call charge applies.
- The measured costs are dominated by Claude Code CLI overhead. Each Opus call writes about 1000
  prompt-cache tokens at the 1-hour rate (about $0.008) against about 176 output tokens (about
  $0.0035), for $0.0115 per call. Sent directly to the API with only the task prompt (about 250
  input tokens), an Opus call would cost about $0.0045. Haiku calls carry the same overhead
  ($0.00047 measured, about $0.0003 direct). The cost ratios above are therefore specific to the
  CLI backend.
- Serve-run audits were cache hits from the eval run, so they reuse one Opus answer per task and
  do not capture the final tier's run-to-run variance.
- Latency is not reported: each call starts a CLI process, which dominates the timing.
- The held-out split is 60 tasks, too small to confirm a 1-point accuracy target.
