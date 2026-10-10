# Results

Two experiments with real models:

1. **MMLU-Pro and BIG-Bench Hard, 2026-10-10** (below): Haiku -> Opus and Haiku -> Sonnet on
   ground-truth datasets where the final tier is often wrong. This is the main result.
2. **Multi-step arithmetic, 2026-10-09** ([further down](#earlier-run-haiku-55---opus-55-on-multi-step-arithmetic)):
   Haiku -> Opus on 200 generated problems. Opus got all 200 right, so disagreement equalled error
   by construction. That run cannot show the gap between them.

## MMLU-Pro and BIG-Bench Hard: disagreement is not error

Raw decisions, configs and scripts: [`experiments/mmlu-pro-cascade/`](../experiments/mmlu-pro-cascade/)
and [`experiments/bbh-cascade/`](../experiments/bbh-cascade/). Analysis:
[`experiments/analyze_cascade.py`](../experiments/analyze_cascade.py) (outputs in each folder's
`analysis.txt` and `analysis.json`). Reports: [MMLU-Pro Haiku -> Opus](results/mmlu-pro-haiku-opus.html),
[MMLU-Pro Haiku -> Sonnet](results/mmlu-pro-haiku-sonnet.html), [BBH Haiku -> Opus](results/bbh-haiku-opus.html),
[BBH Haiku -> Sonnet](results/bbh-haiku-sonnet.html).

### Setup

| | |
|---|---|
| MMLU-Pro | 1680 questions from the 12,032-question test split, 120 per subject (14 subjects), 10 options, graded by exact match on the option letter. Drawn with seed 2026 after a 20-question pilot (seed 1). [`make_tasks.py`](../experiments/mmlu-pro-cascade/make_tasks.py) |
| BIG-Bench Hard | 520 examples, 20 from each of 26 tasks, graded by exact match after normalization (case, punctuation, whitespace). `dyck_languages` is excluded (bracket targets do not survive normalization), and so are 3 examples whose target is option text instead of a label. [`make_tasks.py`](../experiments/bbh-cascade/make_tasks.py) |
| Cheap tier | `haiku` (Claude Haiku 5.5), effort `low`, verbal confidence (`CONFIDENCE: <0-1>`), threshold 0.80 |
| Final tiers | `opus` (Claude Opus 5.5) or `sonnet` (Claude Sonnet 5.5), default effort |
| Prompt | Question and options, then "Think step by step, keeping the reasoning brief", then a final `ANSWER:` line |
| Eval run | Every tier answers every task, so the cheap tier's error and its disagreement with the final tier are known for every skipped case |
| Serve run | Normal routing plus inline audits: inclusion probability 0.5 for confidence in [0.80, 0.95), 0.2 for [0.95, 1.00]; tolerance 5%. Every call is a response-cache hit from the eval run, so it costs nothing extra and sees the same answers |
| Haiku reuse | Haiku runs once per task. Both pair configs share one ledger and response cache, so the Haiku -> Sonnet runs reuse the Haiku answers from the Haiku -> Opus runs (every Haiku answer is identical across the two pairs; all Sonnet-run Haiku calls are cache hits). The Opus and Sonnet columns are therefore paired comparisons on the same Haiku answers, not independent replications. |
| Backend | Claude Code CLI 2.1.296 (`claude -p`), model aliases `haiku`, `sonnet`, `opus` |
| Spend | $58.06 in CLI-reported `total_cost_usd` over 6,656 calls, including the pilot (budget $90; [`claude_budget.py`](../experiments/mmlu-pro-cascade/claude_budget.py) logged every call and would have refused calls past $88) |

Hugging Face is blocked in the environment that ran this. The MMLU-Pro questions come from the
authors' own copy of the test split in [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)
(`eval_results/model_outputs_gpt-4o-2024-08-06_5shots.zip`: same 12,032 question ids, questions,
options and gold letters; the stored model outputs are ignored). One question (id 3983) is
dropped because its `answer` and `answer_index` fields disagree.

### Accuracy and cost

| Policy | MMLU-Pro accuracy | Cost per task | BBH accuracy | Cost per task |
|---|---|---|---|---|
| Haiku only | 83.0% (81.2-84.8) | $0.00058 | 91.2% (88.4-93.3) | $0.00048 |
| Sonnet only | 88.3% (86.6-89.7) | $0.0088 | 94.2% (91.9-95.9) | $0.0073 |
| Opus only | 91.2% (89.8-92.5) | $0.0178 | 95.0% (92.8-96.6) | $0.0142 |
| Haiku -> Sonnet, threshold 0.80 | 86.2% (84.5-87.8) | $0.0027 + $0.0025 audit | 92.7% (90.1-94.6) | $0.0013 + $0.0017 audit |
| Haiku -> Opus, threshold 0.80 | 88.1% (86.5-89.6) | $0.0049 + $0.0048 audit | 93.5% (91.0-95.3) | $0.0020 + $0.0035 audit |

Accuracy is against gold labels, with 95% Wilson intervals. Haiku served 80.0% of MMLU-Pro tasks
and 89.6% of BBH tasks alone. Costs per task are recorded costs: Opus and Sonnet from the CLI's
`total_cost_usd`, Haiku from shadowgate's price table (see Limitations).

| Saving against the final tier alone | Serving only | All-in (serving + audit) |
|---|---|---|
| MMLU-Pro, Haiku -> Opus | 72.2% | 45.5% |
| MMLU-Pro, Haiku -> Sonnet | 69.3% | 41.4% |
| BBH, Haiku -> Opus | 85.6% | 61.3% |
| BBH, Haiku -> Sonnet | 82.5% | 59.1% |

The audit rates here (0.5 and 0.2) are high so that one run gives a large audit sample. Audit
spend was 0.9-1.7x serving spend. The all-in saving is the honest figure at these rates.

### Error against disagreement on the skipped cases

The eval run gives the full truth for all skipped cases (1344 on MMLU-Pro, 466 on BBH). "Shared"
errors are skipped cases where Haiku is wrong and the final tier gave the same wrong answer. The
audit cannot see them. "False alarms" are cases where Haiku is right and the final tier disagrees.

| | MMLU-Pro, Opus | MMLU-Pro, Sonnet | BBH, Opus | BBH, Sonnet |
|---|---|---|---|---|
| Final tier's own error (all tasks) | 8.8% (7.5-10.2) | 11.7% (10.3-13.4) | 5.0% (3.4-7.2) | 5.8% (4.1-8.1) |
| Haiku error on skipped cases, vs gold | 9.3% (7.9-11.0) | 9.3% (7.9-11.0) | 5.4% (3.7-7.8) | 5.4% (3.7-7.8) |
| Haiku disagreement with final tier, all skipped | 6.1% (4.9-7.5) | 4.5% (3.5-5.7) | 3.2% (2.0-5.2) | 2.8% (1.6-4.7) |
| Haiku errors shared with the final tier | 55 of 125 (44%) | 76 of 125 (61%) | 13 of 25 | 14 of 25 |
| False alarms | 12 | 11 | 3 | 2 |
| Serve-run audit: weighted disagreement | 5.1% (3.1-7.8), 462 audits | 4.1% (2.3-6.6), 471 audits | 2.7% (0.5-7.6), 127 audits | 0.8% (0.0-4.8), 122 audits |
| Status vs 5%, audit only | inconclusive | inconclusive | inconclusive | **ok** |
| Status vs 5%, graded against gold | breach | breach | inconclusive | inconclusive |
| AUROC of Haiku's confidence (Haiku correct) | 0.819 | 0.819 | 0.836 | 0.836 |

What this shows:

- **The audit estimates disagreement well and error poorly.** Against gold labels, the audit
  measures the wrong quantity. On MMLU-Pro the weighted audit estimate (5.1% and 4.1%) tracks the
  true disagreement (6.1% and 4.5%). Haiku's gold-graded error is 9.3%. A weaker final tier
  shares more of the cheap tier's errors (Sonnet 61%, Opus 44%), so it reports lower disagreement
  and looks better.
- **One audit said `ok` when the gold-graded error was above tolerance.** On BBH, the Haiku ->
  Sonnet audit put disagreement at 0.8% (upper bound 4.8%), status `ok` against 5%. The
  gold-graded error was 5.4% (3.7-7.8). The gold interval contains 5%, so this is not a proven
  breach. The `ok` itself was mostly a lucky draw: re-drawn 5000 times offline, the audit-only
  status was `ok` in only 1.4% of draws (Monte Carlo SE 0.2 points). The lasting point is the gap
  between disagreement (2.8%) and gold-graded error (5.4%). The audit is blind to errors the
  final tier shares (14 of 25 Haiku errors here), so even a well-sized audit would centre on
  disagreement, not error.
- **Some of the gap is probably label noise.** Thirteen of the 55 MMLU-Pro shared errors
  (Haiku -> Opus) were drawn at random and read by hand
  ([review](../experiments/mmlu-pro-cascade/shared-errors-review.md)): 5 had wrong gold labels,
  6 were ambiguous or had two defensible options, and 2 looked like real errors by both models.
  On BBH the shared errors cluster in tasks with known label problems (`geometric_shapes` 4,
  `date_understanding` 3, `salient_translation_error_detection` 3, `causal_judgement` 2,
  `ruin_names` 1). Two `date_understanding` items spot-checked by hand looked like wrong gold labels (not
  written up, so treat this as anecdotal). The
  gold-graded error is not shown to be an upper bound on true error. Noisy labels cut both ways:
  a wrong gold label can also mark a wrong Haiku answer as right, and the hand check only looked
  at cases graded wrong. The 13 cases come from two ad hoc random seeds (7 and 8) and one
  reviewer, so they show that label noise exists, not how large it is. True error could lie
  anywhere from below the disagreement rate to above the gold-graded rate. This run cannot pin it
  down.
- **The reference is itself wrong.** Against gold labels, Opus was wrong on 8.8% of MMLU-Pro
  tasks and Sonnet on 11.7%. An audit against such a tier measures agreement with a model that is
  wrong about one time in ten.

### Coverage check (offline, 5000 seeds)

As in the earlier run, the serve-mode audit is re-drawn from the eval decisions with references
stripped (same strata, seeds 0-4999). The weighted 95% interval is then scored against both the
true disagreement rate and the gold-graded error rate:

| | MMLU-Pro, Opus | MMLU-Pro, Sonnet | BBH, Opus | BBH, Sonnet |
|---|---|---|---|---|
| Audits per draw (mean) | 455 | 455 | 122 | 122 |
| Mean estimate | 6.10% (true disagreement 6.10%) | 4.45% (4.46%) | 3.24% (3.22%) | 2.80% (2.79%) |
| Interval covers true disagreement | 99.8% (±0.07) | 99.9% (±0.05) | 99.9% (±0.03) | 100.0% (no misses) |
| Interval covers gold-graded error | **38.6%** (±0.7) | **0.8%** (±0.1) | 98.4% (±0.2) | 97.1% (±0.2) |
| Status vs 5% | inconclusive 94.5%, breach 5.5% (±0.3) | inconclusive 99.3%, ok 0.7% (±0.1) | inconclusive 99.4%, ok 0.6% (±0.1) | inconclusive 98.6%, ok 1.4% (±0.2) |

"±" is the Monte Carlo standard error over the 5000 draws; the 1000-seed figures in earlier drafts
differed by up to 3 points (for example 41.6% for the first error-coverage cell) through seed noise
alone.

The weighted (Hajek) estimator is only approximately unbiased for disagreement: it is a ratio
estimator, and its bias is small at these sample sizes (mean estimates within 0.02 points of the
truth above). Its Korn-Graubard interval over-covers (99.8-100% against 95% nominal). Part of
that over-coverage comes from the interval ignoring the finite-population correction: the audit
samples about 34% of the 1344 skipped MMLU-Pro cases (26% of the 466 on BBH) without replacement,
so the with-replacement variance is too large. The interval is conservative here, not accurate.
As an estimate of error it is biased downwards by the shared errors. With ~455 audits on MMLU-Pro
the interval is narrow enough to exclude the gold-graded error most of the time. With ~122 audits
on BBH it is wide enough to include it. In all four populations the gold-graded error is above
5%, yet the audit-only status was `ok` in 0.6-1.4% of draws in three of them.

### Drift check (offline): threshold set on STEM, served on humanities

From the MMLU-Pro eval decisions: the threshold is the lowest observed confidence at which Haiku's
gold-graded error on STEM skipped cases (math, physics, chemistry, engineering, computer science,
biology; 720 tasks) is at most 5%. That is 0.88. STEM at 0.88: Haiku serves 71.2% alone with
3.3% (2.1-5.2) error. The same threshold on humanities (law, history, philosophy; 360 tasks):

| Humanities at threshold 0.88 | Haiku -> Opus | Haiku -> Sonnet |
|---|---|---|
| Served by Haiku alone | 122 of 360 (33.9%) | 122 of 360 (33.9%) |
| Haiku error vs gold | 12.3% (7.6-19.3) | 12.3% (7.6-19.3) |
| Disagreement with final tier | 9.8% (5.7-16.4) | 9.0% (5.1-15.4) |
| Audit re-drawn 5000 times (~41 audits each) | mean 9.9%; breach 10.8% (±0.4), inconclusive 89.2% | mean 9.1%; breach 8.1% (±0.4), inconclusive 91.9% |

Haiku's confidence drops on humanities, so it serves fewer of them, but the ones it keeps are wrong
nearly four times as often as on STEM. The audit never reported `ok` on this shifted traffic. At
this volume (~41 audits) it confirmed the breach in only 8-11% of draws; the median upper bound
was about 25%. Catching drift of this size needs a few hundred audits on the shifted slice.

### Limitations

- **Gold labels are noisy.** Both benchmarks have wrong and ambiguous labels (see above). Error
  "against gold" may overstate or understate true error. The 13 hand-checked cases (two ad hoc
  seeds, one reviewer) do not estimate the size or sign of the bias.
- **The two pairs are not independent.** Haiku-side quantities (Haiku error on skipped cases,
  AUROC, skip rate) are identical across pairs by construction, so they are not two confirmations.
- **One run, one prompt, one threshold.** Haiku runs at effort `low`; the final tiers at default
  effort. Different prompts or efforts change every number.
- **Serve-run audits are cache hits.** Each audit replays the eval run's final-tier answer, so
  the final tier's run-to-run variance is not measured. The serve run's costs are the
  recorded costs of those cached answers.
- **Costs.** Opus and Sonnet costs are the CLI's own `total_cost_usd`: the CLI reported more than
  one model in `modelUsage` for those calls, so shadowgate kept the alias as the model name and
  used the CLI's figure. Haiku costs come from shadowgate's price table: $1.23 for all
  Haiku calls (MMLU-Pro, BBH and pilot), against the $1.66 the CLI reported for them. At the
  CLI's Haiku price the savings above are 1.1-2.3 points lower. All figures include the Claude
  Code CLI's prompt overhead (1,200 or more cache-write tokens per call) and are notional on a
  subscription.
- **Drift check.** The drift result uses 360 humanities tasks and 122 skipped cases, so its
  intervals are wide.
- **Latency is not reported.** Each call starts a CLI process, which dominates the timing.

### Reproduce

```sh
git clone --depth 1 https://github.com/TIGER-AI-Lab/MMLU-Pro /tmp/MMLU-Pro
git clone --depth 1 https://github.com/suzgunmirac/BIG-Bench-Hard /tmp/BIG-Bench-Hard
sh experiments/mmlu-pro-cascade/run.sh   # ~5,000 calls, ~$47 CLI-reported
sh experiments/bbh-cascade/run.sh        # ~1,560 calls, ~$12 CLI-reported
```

The offline analysis (no model calls) runs from the committed decisions:

```sh
uv run python experiments/analyze_cascade.py experiments/mmlu-pro-cascade
uv run python experiments/analyze_cascade.py experiments/bbh-cascade
```

The analysis defaults to 5000 seeds and takes a few minutes per experiment. Pass `--seeds` to
change it.

## Earlier run: Haiku 5.5 -> Opus 5.5 on multi-step arithmetic

One real run, 2026-10-09. Full report: [`results/arithmetic-haiku-opus.html`](results/arithmetic-haiku-opus.html)
(also on [GitHub Pages](https://h-a-forster.github.io/shadowgate/results/arithmetic-haiku-opus.html)).
Raw decisions: [`experiments/arithmetic-haiku-opus/`](../experiments/arithmetic-haiku-opus/).

### Setup

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

### Results

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

### What the audit found

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

### Coverage check (offline)

Because the eval run called Opus on every task, the audit can be re-drawn offline.
[`simulate_audit.py`](../experiments/arithmetic-haiku-opus/simulate_audit.py) draws the serve-mode
audit sample 5000 times (seeds 0-4999, same strata: inclusion probability 0.5 below confidence 0.95,
0.2 above), strips the references so only the audit counts, and checks the weighted 95% interval
against the true skipped-case rate (3 of 181, 1.66%, graded against references). No model calls:

```sh
uv run python experiments/arithmetic-haiku-opus/simulate_audit.py
```

| | |
|---|---|
| Audits per draw | mean 58.4 (39-82) |
| Mean point estimate | 1.67% (true 1.66%) |
| Interval covers the true rate | 5000 of 5000 draws (100.0%) |
| Draws with no disagreement observed | 12.7% (Monte Carlo SE 0.5); exact value 12.5% |
| Upper bound | median 10.3%, 5th-95th percentile 7.4-13.6% |
| Status vs 5% tolerance | inconclusive in 5000 of 5000 draws |

The exact no-disagreement share follows from the design: all three disagreements sit in the 0.5
stratum (confidence 0.80, 0.85, 0.93), so no audit sees any of them with probability
0.5^3 = 12.5%. An earlier 1000-seed run gave 11.0%, 1.5 points off through seed noise.

The weighted estimate is close to the truth here and the Korn-Graubard interval is conservative
(it never missed in 5000 draws, against 95% nominal), partly because it ignores the
finite-population correction (about 32% of the 181 skipped cases are audited). The cost is
width: at this audit size the upper bound never fell below 5%, so the audit alone could not have
confirmed the tolerance on any draw. This checks one population with three errors; it is not a
general coverage result.

### Reproduce

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

### Limitations

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
