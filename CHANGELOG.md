# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Completion.meta`: backend-specific provenance, serialized with the completion. The
  claude-code backend records `prompt_sha256` (system prompt and stdin as sent) and
  `cli_version` (from `--version`, probed once per backend). Decisions recorded before this
  change have no `meta`; the published experiment files are unchanged.
- `AuditSummary.audit_only_status`: when references drive the status, the status the audit
  disagreement alone would give. The report and `shadowgate audit` show it next to the
  reference-based status; the demo prints it as `audit-only status`.
- `experiments/arithmetic-haiku-opus/simulate_audit.py`: offline coverage check of the audit
  interval on the recorded run.
- MMLU-Pro (1680 questions) and BIG-Bench Hard (520 examples) experiments with Haiku -> Opus and
  Haiku -> Sonnet through the Claude Code CLI, in `experiments/mmlu-pro-cascade/` and
  `experiments/bbh-cascade/`: raw decisions, configs, task builders, a budget guard around the
  CLI, and `experiments/analyze_cascade.py` (error vs disagreement, coverage over 5000 seeds
  with Monte Carlo standard errors, drift check).

### Changed

- `experiments/mmlu-pro-cascade/claude_budget.py` reserves `SHADOWGATE_RESERVE_USD` for every
  call in flight (tracked in `costs.jsonl.inflight.json`, with a TTL for reservations left by
  crashed processes) instead of one reserve for all workers, so parallel workers can no longer
  overshoot the cap by several whole calls. Locking now works on Windows (`msvcrt`) as well as
  POSIX (`fcntl`), torn log lines are skipped, and `--version` passes straight through. The
  published runs used the earlier guard; their spend ($58.06) stayed well below the $88 cap, so
  no published result is affected.
- Results docs report both statuses, all-in cost including audit spend, and the limitations of the
  published run (final tier 100% correct, CLI cost overhead, cached serve-run audits).
- docs/results.md and the README lead with the MMLU-Pro and BBH results, where the final tier
  is often wrong and the audit's disagreement estimate falls below the gold-graded error.
- Results wording: coverage figures re-run with 5000 seeds and reported with Monte Carlo standard
  errors; the audit interval is described as conservative (finite-population correction ignored)
  and the Hajek estimator as approximately unbiased; the label-noise argument is softened; Haiku
  answers shared across pairs are disclosed; cost sources are stated consistently. Also:
  `make_tasks.py` docstring and `run.sh` now match the committed pilot files.

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
