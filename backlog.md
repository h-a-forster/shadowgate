# Backlog

## In progress

## Waiting on me

- 0.1.1 is released: `polish-2026-10-30` merged through a PR, the release date set in
  `CHANGELOG.md`, `CITATION.cff` and `docs/release-notes-0.1.1.md`, and tag `v0.1.1` pushed.
- Dependabot PR #4 is closed after the merge: it edits the tag pins that SHA pins replace, and
  Dependabot then proposes the setup-uv bump against the SHA.
- Merged remote branches are deleted: `claude/fix-review-2026-10-10`,
  `claude/hygiene-2026-10-15`, `claude/research-notes-2026-10-10`, `upgrades-2026-10-23`.

## Decided

- 2026-10-10: releases ship from GitHub only; PyPI waits until the project has users.

## Proposed

- The audit interval is closer to 95% at low disagreement counts: it still covers 98.9-100% on
  BBH and arithmetic, where 3-15 skipped cases disagree.
- `svg.py` (1,505 lines of hand-rolled charts) stops being a maintenance liability, by tests on
  its output or a smaller scope.
- Repeated audit checks have a valid status (an anytime-valid confidence sequence), so
  monitoring a growing run no longer needs a stricter fixed `--level`.
- The `choice` extractor returns no answer for hedged lines such as `ANSWER: A, C, D`, so a
  multi-letter reply escalates instead of being read as one letter.
- The shared-error rate can be estimated without gold labels (needs new model runs).
- A diverse reference panel is tested against shared errors (needs new model runs).
- Label noise is sized by a larger, blinded relabelling of the MMLU-Pro and BBH shared errors.
- Audit effort follows drift, tested with subject or confidence strata on the humanities shift.
