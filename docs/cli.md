# CLI reference

```text
shadowgate [-h] [--version] [-v] COMMAND ...
```

| Command | Purpose |
|---|---|
| [`init`](#init) | Write a starter config and 20 example tasks. |
| [`demo`](#demo) | Offline end-to-end demo with simulated models; writes a report. |
| [`run`](#run) | Route tasks through the cascade and record every decision in the ledger. |
| [`audit`](#audit) | Summarize the shadow audit of a run; optionally run deferred audits; CI gate. |
| [`sweep`](#sweep) | Sweep thresholds over an eval-mode run and recommend one. |
| [`calibrate`](#calibrate) | Fit an isotonic calibration map for a tier's confidence and print a config line. |
| [`report`](#report) | Write an HTML or Markdown report. |
| [`runs`](#runs) | List the runs in a ledger. |
| [`export`](#export) | Export a run as JSONL: a run header line, then one decision per line. |
| [`import`](#import) | Import exported JSONL into a ledger, creating it if needed. |
| [`datasets`](#datasets) | Generate or inspect task files. |

Global options:

| Option | Meaning |
|---|---|
| `--version` | Print the version and exit. |
| `-v`, `--verbose` | More logging; `-vv` for debug. |

Set `SHADOWGATE_DEBUG=1` to print tracebacks. Otherwise errors are one line on stderr, prefixed
`shadowgate: error:`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | Runtime error (including a malformed import file), or a run stopped after repeated failures. |
| 2 | Usage or configuration error: bad arguments, config, task file, or missing ledger. |
| 3 | Audit breach with `--fail-on-breach`. |
| 4 | A run stopped because it reached its spending cap. |
| 130 | Interrupted (Ctrl-C). Completed work is in the ledger. |

## Ledger and run selection

Every command except `init`, `demo` and `datasets` takes `--ledger PATH` (default
`.shadowgate/ledger.sqlite` in the working directory). Only `run` and `audit --run-pending` read
the config, so only they pick up `[run] ledger`. If your config sets a ledger path, pass it to the
other commands too.

`--run-id ID` selects a run. Without it, `audit`, `report` and `export` use the latest run, and
`sweep` and `calibrate` use the latest eval-mode run.

## Interval level

`audit`, `sweep` and `report` take `--level L`: the confidence level of every interval they print
(default `0.95`, must be in (0.5, 1)). The audit status compares a fixed-sample interval with the
tolerance. If you check the same growing run repeatedly, use a stricter level such as `0.99`; see
[the guide](guide.md#limitations).

## `init`

```text
shadowgate init [--force] [DIR]
```

Writes `DIR/shadowgate.toml` (a two-tier simulated cascade with comments showing real backends)
and `DIR/tasks.jsonl` (20 arithmetic tasks). `DIR` defaults to `.`.

| Option | Meaning |
|---|---|
| `--force` | Overwrite existing files. |

## `demo`

```text
shadowgate demo [--out DIR] [--n N] [--seed S] [--open] [-q]
```

Generates arithmetic tasks and runs a simulated two-tier cascade once in eval mode and once in
serve mode. Prints both summaries and the commands to explore the runs, and writes `report.html`
and `report.md`. No network, no keys. Re-running with the same options resumes the existing runs.

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | `shadowgate-demo` | Output directory: `shadowgate.toml`, `tasks.jsonl`, `ledger.sqlite`, `report.html`, `report.md`. |
| `--n N` | `400` | Number of tasks. |
| `--seed S` | `0` | Dataset and model seed. |
| `--open` | off | Open the HTML report in a browser. |
| `-q`, `--quiet` | off | No progress output. |

## `run`

```text
shadowgate run -c CONFIG -t TASKS [--ledger PATH] [--run-id ID] [--mode {serve,eval}]
               [--limit N] [--workers N] [--max-cost USD] [--no-resume] [-q]
```

| Option | Default | Meaning |
|---|---|---|
| `-c`, `--config` | required | TOML config. |
| `-t`, `--tasks` | required | `.jsonl`, `.json` or `.csv` task file. |
| `--ledger PATH` | `[run] ledger`, else `.shadowgate/ledger.sqlite` | |
| `--run-id ID` | `<run.name>-<mode>-<timestamp>` | Reuse an id to resume. |
| `--mode` | `serve` | `serve`: stop at the first accepted tier and sample audits. `eval`: run every tier on every task. |
| `--limit N` | all | First N tasks only. |
| `--workers N` | `[run] workers` (4) | Tasks in flight. |
| `--max-cost USD` | `[run] max_cost_usd` | Spending cap (serving + audit). Exit 4 when reached. |
| `--no-resume` | off | Re-route tasks already recorded under this run id. |
| `-q`, `--quiet` | off | Print only the final summary. |

Each decision is written to the ledger as soon as it finishes. Re-running with the same
`--run-id` (and the same `--mode`) skips tasks already recorded. Without `--run-id`, every
invocation starts a new run. On a budget stop, interrupt or failure streak, the command prints
the exact resume command.

## `audit`

```text
shadowgate audit [--ledger PATH] [--run-id ID] [--tolerance T] [--level L] [--json] [--fail-on-breach]
shadowgate audit --run-pending -c CONFIG [--ledger PATH] [--run-id ID] [--workers N] [--max-cost USD]
```

Prints the skipped-case disagreement rate (weighted by 1/π), the error rate against references
when tasks have them, escalation rate, served accuracy, cost per task, estimated savings against
always using the final tier, a per-confidence-bin table and caveats.

| Option | Default | Meaning |
|---|---|---|
| `--run-id ID` | latest run | |
| `--tolerance T` | `[audit] tolerance` stored with the run | Acceptable skipped-case error or disagreement rate. |
| `--level L` | `0.95` | Interval confidence level. |
| `--json` | off | Machine-readable output. |
| `--fail-on-breach` | off | Exit 3 when the status is `breach`. |
| `--run-pending` | off | First run deferred audits (`[audit] mode = "deferred"`), then summarize. |
| `-c`, `--config` | | Config used to build the audit tier. Required with `--run-pending`, rejected without it. |
| `--workers N` | `[run] workers` | With `--run-pending`. |
| `--max-cost USD` | `[run] max_cost_usd` | With `--run-pending`. |

Status values:

| Status | Condition |
|---|---|
| `ok` | Upper bound of the interval <= tolerance. |
| `breach` | Lower bound > tolerance. |
| `inconclusive` | The interval contains the tolerance. The output estimates how many more audits (or tasks) would resolve it. |
| `no-data` | No audited or graded skipped cases. |
| `n/a` | No tolerance given. |

The status uses the error rate against references when every skipped case has a reference,
otherwise the audit disagreement rate. Only `breach` triggers exit 3; `inconclusive` exits 0.

## `sweep`

```text
shadowgate sweep [--ledger PATH] [--run-id ID] [--objective {max-savings,max-accuracy,min-accuracy}]
                 [--max-drop X] [--min-accuracy X] [--budget X] [--holdout F] [--seed S]
                 [--level L] [--json]
```

Simulates every threshold on a grid over an eval-mode run, with no new model calls. Prints
single-tier and oracle baselines, the accuracy/cost Pareto frontier, a recommended threshold
chosen on one split and re-evaluated on the held-out split, and calibration metrics (ECE, Brier,
AUROC, AURC) for each non-final tier.

| Option | Default | Meaning |
|---|---|---|
| `--run-id ID` | latest eval-mode run | |
| `--objective` | `max-savings` | See below. |
| `--max-drop X` | `0.01` | For `max-savings`: allowed accuracy drop below the best single tier. |
| `--min-accuracy X` | | Required for `min-accuracy`. |
| `--budget X` | | USD per task, for `max-accuracy`. |
| `--holdout F` | `0.3` | Held-out share. `0` disables validation. |
| `--seed S` | `0` | Split seed. |
| `--level L` | `0.95` | Interval confidence level. |
| `--json` | off | Machine-readable output. |

| Objective | Picks |
|---|---|
| `max-savings` | Cheapest point within `--max-drop` of the best single tier's accuracy. |
| `max-accuracy` | Most accurate point with cost per task <= `--budget`. |
| `min-accuracy` | Cheapest point with accuracy >= `--min-accuracy`. |

Accuracy is measured against task references when every task has one. Otherwise it is
agreement with the final tier, and the output says `truth: audit-tier`.

## `calibrate`

```text
shadowgate calibrate [--ledger PATH] [--run-id ID] [--tier NAME]
                     [--truth {auto,reference,audit-tier}] [--max-knots K] [--json]
```

Fits a monotone (isotonic) map from one non-final tier's confidence to its observed rate of
correct answers, using an eval-mode run. Prints ECE and Brier score before calibration, after it
in-sample, and after it with 2-fold cross-fitting (the estimate to trust for new data), then a
line to paste into the tier's config:

```toml
confidence = { type = "calibrated", base = { type = "verbal" }, points = [[0.3, 0.0], [0.76, 0.0], ...] }
```

Thresholds then apply to the calibrated scale. Re-run eval mode and `sweep` to pick a new one.

| Option | Default | Meaning |
|---|---|---|
| `--run-id ID` | latest eval-mode run | A serve-mode run is accepted with a note: its labels come from sampled audits, not a uniform sample. |
| `--tier NAME` | first tier | Must be a non-final tier. |
| `--truth` | `auto` | `reference`: correctness against task references. `audit-tier`: agreement with the final tier. `auto`: `reference` when every scored attempt is graded, else `audit-tier`. |
| `--max-knots K` | `20` | Maximum number of map points (>= 2). |
| `--json` | off | Machine-readable output. |

It needs at least 10 labelled (confidence, outcome) pairs and exits 2 otherwise.

## `report`

```text
shadowgate report [--ledger PATH] [--run-id ID] [--sweep-run-id ID] -o OUT [--tolerance T] [--level L]
```

Writes a self-contained report. The format follows the suffix of `-o`: `.html` (inline CSS and
SVG, no external requests, light and dark themes) or `.md`.

| Option | Meaning |
|---|---|
| `-o`, `--output` | Required. `.html` or `.md`. |
| `--run-id ID` | Run to report (default: latest). |
| `--sweep-run-id ID` | Add the threshold sweep of an eval-mode run to the report of a serve-mode run. An eval-mode run includes its own sweep. |
| `--tolerance T` | Override the stored tolerance. |
| `--level L` | Interval confidence level (default `0.95`). |

## `runs`

```text
shadowgate runs [--ledger PATH] [--json]
```

Lists run id, mode, number of decisions and creation time (UTC). `--json` also includes each
run's note and redacted config.

## `export`

```text
shadowgate export [--ledger PATH] [--run-id ID] -o OUT.jsonl
```

The first line is a run header, `{"shadowgate_run": {...}}`, with the run id, creation time,
mode, note and redacted config. Each following line is one decision: the task, every attempt with
its completion, confidence, grading and cost, and the shadow audit result. `--run-id` defaults to
the latest run.

## `import`

```text
shadowgate import [--ledger PATH] FILE.jsonl [FILE.jsonl ...]
```

Loads files written by `export` into a ledger, creating it if needed. Decisions are upserted by
(run id, task id), so importing a file twice changes nothing. The run header restores the run's
mode, note and redacted config, so `audit` and `report` pick up its `[audit] tolerance`. A header
whose mode differs from an existing run with the same id is an error. Files without a header still
import; their runs then have no stored config. A malformed line fails the file before any of its
decisions are written.

```sh
shadowgate import a.jsonl b.jsonl --ledger merged.sqlite
```

## `datasets`

```text
shadowgate datasets make NAME -n N [--seed S] [--min-steps K] [--max-steps K] -o OUT.jsonl
shadowgate datasets show TASKS [--limit N]
```

`make` generates a built-in synthetic dataset. The only generator is `arithmetic`: multi-step
word problems with integer answers, where `meta.difficulty` grows with the number of steps.
`show` prints the task count, how many have references, the level mix and the first `--limit`
tasks (default 3).

| Option | Default | Meaning |
|---|---|---|
| `-n N` | required | Number of tasks. |
| `--seed S` | `0` | |
| `--min-steps K`, `--max-steps K` | `1`, `6` | Range of operations per problem, at most 12. |
| `-o`, `--output` | required | Output `.jsonl`. |

## Example session

Offline, using the starter config:

```sh
shadowgate init sg-example
shadowgate run -c sg-example/shadowgate.toml -t sg-example/tasks.jsonl --run-id first
shadowgate audit --run-id first --tolerance 0.05
shadowgate run -c sg-example/shadowgate.toml -t sg-example/tasks.jsonl --mode eval --run-id first-eval
shadowgate sweep --run-id first-eval --holdout 0
shadowgate calibrate --run-id first-eval
shadowgate report --run-id first --sweep-run-id first-eval -o report.html
shadowgate runs
shadowgate export --run-id first -o first.jsonl
shadowgate import first.jsonl --ledger copy.sqlite
```

With 20 tasks the intervals are wide. Use `shadowgate datasets make arithmetic -n 500 -o tasks.jsonl`
for a larger file.
