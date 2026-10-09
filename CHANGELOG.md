# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `AuditSummary.audit_only_status`: when references drive the status, the status the audit
  disagreement alone would give. The report and `shadowgate audit` show it next to the
  reference-based status; the demo prints it as `audit-only status`.
- `experiments/arithmetic-haiku-opus/simulate_audit.py`: offline coverage check of the audit
  interval on the recorded run.

### Changed

- Results docs report both statuses, all-in cost including audit spend, and the limitations of the
  published run (final tier 100% correct, CLI cost overhead, cached serve-run audits).

## [0.1.0] - 2026-10-10

Initial release.

### Added

- Cascade routing: tasks run through tiers, cheapest first; a non-final tier serves when its
  confidence meets its threshold and escalates otherwise. Missing or unparseable confidence
  escalates.
- Confidence estimators: `verbal`, `logprob`, `self_consistency`, `monitor`, `combine`,
  `calibrated`, and `callable` (Python API only).
- Answer extractors and comparators, including a model `judge` comparator.
- Shadow audit of skipped cases: reproducible sampling with known inclusion probabilities
  (base rate, confidence strata, floor), inline or deferred, with a Hájek estimate of
  disagreement weighted by 1/π, error rates against references, and a tolerance status.
- Eval mode (every tier on every task), threshold sweep with an accuracy/cost Pareto frontier,
  held-out threshold selection, and calibration metrics (ECE, Brier, AUROC, AURC).
- Isotonic confidence calibration that prints a ready `calibrated` config line.
- Reports in HTML and Markdown with inline SVG charts.
- Backends: Anthropic (optional `anthropic` extra), OpenAI-compatible HTTP, Claude Code CLI, any
  command, a deterministic simulator, and replay of recorded completions.
- SQLite ledger (resumable runs) and SQLite response cache; JSONL export with a run header line,
  and import that restores the run's mode, note and redacted config.
- Concurrent runner with a spending cap and clean interrupt handling.
- TOML config with path-named validation errors; literal credentials are rejected.
- `shadowgate` CLI: `init`, `demo`, `run`, `audit`, `sweep`, `calibrate`, `report`, `runs`,
  `export`, `import`, `datasets`. `--level` sets the interval confidence level for `audit`,
  `sweep` and `report`.
- Results of a Claude Haiku 5.5 -> Opus 5.5 run on multi-step arithmetic, with committed decisions.
- CI: lint, type-check, tests on Linux, Windows and macOS (Python 3.11 to 3.13), and a wheel
  smoke test. Tagged releases attach the wheel and sdist to a GitHub release.
- Stdlib-only runtime; Python 3.11+. Installed from GitHub; not published on PyPI.

[Unreleased]: https://github.com/h-a-forster/shadowgate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/h-a-forster/shadowgate/releases/tag/v0.1.0
