# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-09

Initial release.

### Added

- Cascade routing: tasks run through model tiers, each with a confidence estimator and threshold;
  low-confidence answers escalate to the next tier.
- Confidence estimators, with unparseable or unavailable confidence reported as unknown rather
  than coerced.
- Shadow audit: samples accepted (non-escalated) decisions, re-answers them with a reference tier
  out of band, and estimates disagreement and error rates with confidence intervals using
  inverse-probability weighting.
- Eval mode (every tier on every task), threshold sweeps, and an accuracy/cost Pareto frontier
  with held-out threshold selection.
- Reports in HTML and Markdown with inline SVG charts.
- Backends: Anthropic (optional `anthropic` extra), OpenAI-compatible HTTP APIs, Claude Code CLI,
  arbitrary shell command, deterministic simulated model, and replay of recorded completions.
- SQLite ledger (crash-safe, resumable runs) and SQLite response cache.
- Concurrent batch runner with budget cap and clean interrupt handling.
- `shadowgate` CLI: `init`, `demo`, `run`, `audit`, `sweep`, `report`, `runs`, `export`,
  `datasets`.
- Stdlib-only runtime; Python 3.11+.

[Unreleased]: ../../compare/v0.1.0...HEAD
[0.1.0]: ../../releases/tag/v0.1.0
