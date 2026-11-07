# shadowgate 0.1.1 release notes

Full list of changes: [CHANGELOG.md](../CHANGELOG.md).

## Results

- New experiments on ground-truth benchmarks: Claude Haiku 5.5 in front of Opus 5.5 or
  Sonnet 5.5 on 1680 MMLU-Pro questions and 520 BIG-Bench Hard examples, with raw decisions,
  configs and offline analysis under `experiments/`. On MMLU-Pro the cheap tier's error on the
  cases it served alone was 9.3% against gold labels, while the audit measured 5.1% (Opus) and
  4.1% (Sonnet) disagreement: 44-61% of its errors were shared by the final tier and are
  invisible to an audit. See [docs/results.md](results.md).

## Audit interval

- Sampled audit intervals now account for sampling without replacement from the run's skipped
  cases. On the published runs the old interval covered the true disagreement 99.8-100% of the time
  at nominal 95%; the new one covers about 97% on MMLU-Pro and 99.0-99.5% on BBH, where few cases
  disagree. `inconclusive` fell from 94.5-99.4% of re-drawn audits to 74.4-95.7%, and
  `audits_to_resolve` from 2124 to 897 on the MMLU-Pro Sonnet run. Intervals, audit-only statuses
  in simulation and planning counts change; point estimates do not. The interval describes this
  run's skipped cases, not future traffic, and needs about five expected audits in every stratum. A census (eval mode) keeps the ordinary interval.
- `weighted_proportion(..., finite=True)` and `stats.FiniteSample`; the audit uses them. Audit
  estimates report method `hajek-korn-graubard-fpc`, and `n_eff` can now exceed the audit count.

## Library

- `AuditSummary.audit_only_status`: the status the audit alone would give when references
  drive the reported status; shown by `shadowgate audit`, the report and the demo.
- `Completion.meta`: backend provenance; the Claude Code backend records the prompt hash and
  CLI version.
- `CachedBackend` makes one inner call for concurrent identical requests.

## CLI and validation

- `run --max-cost` and `audit --run-pending --max-cost` reject NaN, infinite and negative values.
  A NaN cap used to be accepted and never stopped the run.
- `audit --run-pending` validates `--workers`; `--workers 0` used to fall back to the config.
- `audit --workers` and `audit --max-cost` without `--run-pending` are rejected instead of being
  ignored, as `-c` already was.
- `sweep` rejects a NaN `--max-drop`, a NaN or negative `--budget` and a `--min-accuracy` outside
  [0, 1]; these used to end in "no recommendation".
- `run` exits 1 when every task failed (it used to exit 0 on runs shorter than the failure-streak
  limit), and its warning names the first error instead of pointing at the ledger.
- Resuming a run with changed `[backends]`, `[[tiers]]`, `[answer]` or `[audit]` settings prints
  a warning. Such a run mixes two configurations, and only the newer one stays in the ledger.

## Release engineering

- Workflow actions are pinned to commit SHAs. CI installs with `uv sync --locked`, and the release
  workflow also runs `ruff format --check`.

## Compatibility

No breaking changes to the library, ledgers or exports. Ledgers and exports from 0.1.0 load
unchanged; decisions recorded before this release have no `meta`. The stricter CLI checks above
turn some invalid invocations into exit code 2.
