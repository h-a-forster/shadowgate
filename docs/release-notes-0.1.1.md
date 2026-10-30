# shadowgate 0.1.1 release notes

Released 2026-10-30. Full list of changes: [CHANGELOG.md](../CHANGELOG.md#011---2026-10-30).

## Results

- New experiments on ground-truth benchmarks: Claude Haiku 5.5 in front of Opus 5.5 or
  Sonnet 5.5 on 1680 MMLU-Pro questions and 520 BIG-Bench Hard examples, with raw decisions,
  configs and offline analysis under `experiments/`. On MMLU-Pro the cheap tier's error on the
  cases it served alone was 9.3% against gold labels, while the audit measured 5.1% (Opus) and
  4.1% (Sonnet) disagreement: 44-61% of its errors were shared by the final tier and are
  invisible to an audit. See [docs/results.md](results.md).

## Library

- `AuditSummary.audit_only_status`: the status the audit alone would give when references
  drive the reported status; shown by `shadowgate audit`, the report and the demo.
- `Completion.meta`: backend provenance; the Claude Code backend records the prompt hash and
  CLI version.
- `CachedBackend` makes one inner call for concurrent identical requests.

## Compatibility

No breaking changes. Ledgers and exports from 0.1.0 load unchanged; decisions recorded before
this release have no `meta`.
