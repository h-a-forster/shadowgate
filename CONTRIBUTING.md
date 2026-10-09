# Contributing to shadowgate

Thanks for your interest in improving shadowgate. Bug reports, fixes, new backends, estimators,
comparators and documentation improvements are all welcome.

## Development setup

shadowgate uses [uv](https://docs.astral.sh/uv/) for development. Python 3.11 or newer is required.

```sh
uv sync --all-groups      # create .venv with the package (editable) and dev tools
uv run pytest             # run the test suite
uv run ruff check .       # lint
uv run mypy               # type-check (config in pyproject.toml)
```

Optional extras (for example the Anthropic SDK) can be added with `uv sync --all-groups --extra anthropic`.

## Project layout

The architecture, module responsibilities and conventions are described in
[docs/design.md](docs/design.md). Read it before making non-trivial changes. In short:

- `src/shadowgate/` holds the library and the `shadowgate` CLI (`cli.py`).
- `src/shadowgate/backends/` holds model backends.
- `tests/` holds the test suite; `tests/fixtures/` holds static test data.
- `examples/` holds example configs and task files.

## Principles

- **Stdlib-only runtime.** The package has no runtime dependencies. Provider SDKs are optional
  extras imported lazily, with a clear install hint if they are missing. Do not add numpy or
  similar libraries.
- **Never silently coerce unknowns.** Unknown cost stays `None`; an undecidable comparison is
  `Judgement(equivalent=None)`; unparseable confidence is `ConfidenceResult(score=None)`. Reports
  and estimates must show missing data as missing, not as zero.
- **Statistics need reference values.** Any change to estimators, confidence intervals,
  inverse-probability weighting, sweeps or Pareto selection must come with tests that check
  results against known reference values (hand-computed, published, or from an established
  library), not only against the code's own output.
- **No network in tests.** Use the `simulated`, `replay` or function backends. Tests that call a
  real provider must be marked `@pytest.mark.live`; they are skipped unless `SHADOWGATE_LIVE=1`
  is set, and they never run in CI.
- **Determinism.** Anything random takes a seed and its own `random.Random`; hashing for sampling
  uses sha256 of stable strings, never Python's `hash()`.
- **Secrets.** API keys come only from environment variables named in config (`api_key_env`).
  Never log, cache or write them to the ledger.
- **Library code does not print.** The CLI owns output; library modules use
  `logging.getLogger("shadowgate.<module>")`.

## Adding a backend, estimator or comparator

Each pluggable family is built by a factory function `from_spec(spec, **deps)` that dispatches on
`spec["type"]` (see [docs/design.md](docs/design.md) for the exact signatures). To add one:

1. Implement the class in the appropriate module (a new file under `src/shadowgate/backends/` for
   a backend). It must be thread-safe: instances are called from worker threads.
2. Register a new `type` key in the family's `from_spec`. Validate the spec strictly: unknown
   keys must raise `ConfigError` naming the offending key. Secrets go through `api_key_env`,
   never a literal value.
3. Add tests covering construction from a spec, rejection of unknown keys, normal behavior, and
   failure/unknown cases (for example returning `None` rather than guessing). Backends that talk
   to a service need offline tests (stub the transport) plus optional `live` tests.
4. Document the new `type` and its options in `docs/design.md` or `README.md`, and add a
   `CHANGELOG.md` entry.

## Commits and pull requests

- Keep pull requests focused on one change; open an issue first for larger features.
- Write commit messages in the imperative mood with a short summary line
  (for example "Add regex comparator").
- Before opening a pull request, make sure `uv run pytest`, `uv run ruff check .` and `uv run mypy` pass.
- Update `CHANGELOG.md` under "Unreleased" for user-visible changes, and update the docs when
  behavior or configuration changes.
- CI runs lint, the test matrix (Linux, Windows, macOS; Python 3.11 to 3.13) and a build smoke
  test; all must pass before merge.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
