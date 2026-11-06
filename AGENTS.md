# AGENTS.md

Guide for coding agents (and humans) working on shadowgate. [CONTRIBUTING.md](CONTRIBUTING.md)
holds the full principles and [docs/design.md](docs/design.md) the module contracts; read the
design document before changing routing, auditing or statistics.

## What this is

A stdlib-only Python package and CLI for confidence-gated LLM cascades: tasks go to a cheap
tier first and escalate when confidence is below a threshold, and a shadow audit re-answers a
reproducible, inverse-probability-weighted sample of the cases the cheap tier served alone.
`experiments/` holds the published runs (MMLU-Pro, BIG-Bench Hard, arithmetic) with raw
decisions and offline analysis; `docs/results.md` reports them.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11, no runtime dependencies (provider SDKs are
  optional extras imported lazily). `ruff check` + `ruff format` (line length 100), mypy,
  pytest.
- Unknowns stay unknown: missing cost is `None`, an undecidable comparison is
  `equivalent=None`, unparseable confidence escalates. Never report missing data as zero.
- Statistics changes need tests against reference values (hand-computed or published), not
  the code's own output.
- Determinism: every random draw takes a seed and its own `random.Random`; sampling hashes
  use sha256 of stable strings.
- No network in tests. Real-provider tests are marked `live` and run only with
  `SHADOWGATE_LIVE=1`; they never run in CI.
- Secrets come only from environment variables named by `api_key_env`; never log, cache or
  write them to the ledger. Library code logs; only the CLI prints.
- Results: every number in README.md and docs/results.md must trace to an experiment's
  `analysis.txt` / `analysis.json` or its committed decisions. Re-running the experiments
  costs real model spend (about $60 CLI-reported); the offline analysis does not.
- Workflow actions are pinned to commit SHAs with a version comment; Dependabot updates both.
- Scratch files go in `tmp/` (gitignored). The repository is public: change it through pull
  requests with CI. No personal names, local user paths or machine details in committed files.

## How to test

```sh
uv sync --all-groups
uv run pytest -q              # offline; several minutes on Windows (process-spawning tests)
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python experiments/analyze_cascade.py experiments/bbh-cascade   # offline, a few minutes
```

## Releasing

Bump `__version__` in `src/shadowgate/__init__.py` and `version` / `date-released` in
`CITATION.cff`, date the CHANGELOG section, add `docs/release-notes-<version>.md`, then push a
matching `v<version>` tag; the release workflow checks the tag against the package version.

## Status

0.1.1 is prepared on branch `polish-2026-10-30`, not tagged or pushed. Open work lives in
[backlog.md](backlog.md).
