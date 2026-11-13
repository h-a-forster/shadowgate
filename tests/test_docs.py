"""Every number in README.md and CITATION.cff is copied from generated output.

Experiment figures come from `experiments/numbers.json`, which `experiments/doc_numbers.py`
builds from the committed analyses; CI reruns `analyze_cascade.py`, `simulate_designs.py` and
`doc_numbers.py` and fails on any diff. The demo figures come from running `shadowgate demo`.
A claim below is a fragment of the README with `{key}` placeholders; a number that is not a
result (a version, a config value, a figure from superseded code) is listed as fixed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from docnumbers import check

from shadowgate.cli import main

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "experiments" / "numbers.json"
TIMING = re.compile(r" \| \d+\.\ds$")


def ci(key: str) -> str:
    return f"{{{key}}}% ({{{key}.lo}}-{{{key}.hi}})"


def pair_row(label: str, field: str, fmt: str = "{v}%") -> str:
    cells = [fmt.format(v=f"{{mmlu.{p}.{field}}}") for p in ("opus", "sonnet")]
    return f"| {label} | {cells[0]} | {cells[1]} |"


README = [
    "on {mmlu.opus.tasks} MMLU-Pro\nquestions at threshold 0.80",
    "Haiku's error on the cases it served alone was {mmlu.opus.cheap_error_skipped}% against "
    "gold labels, but its disagreement with Opus on those cases was "
    "{mmlu.opus.disagreement_skipped}%, and the audit-only estimate was "
    + ci("mmlu.opus.audit_only"),
    "{mmlu.opus.shared_share}% of Haiku's errors were shared by Opus "
    "({mmlu.sonnet.shared_share}% for Sonnet)",
    "Against Sonnet, disagreement was {mmlu.sonnet.disagreement_skipped}% and audit-only "
    + ci("mmlu.sonnet.audit_only")
    + " for the same {mmlu.sonnet.cheap_error_skipped}% gold-graded error",
    "Re-drawn {mmlu.opus.coverage.seeds} times, the audit's interval covered the gold-graded error "
    "{mmlu.opus.coverage.error}% of the time with Opus and {mmlu.sonnet.coverage.error_2}% with "
    "Sonnet, but covered the true disagreement about {mmlu.opus.coverage.disagreement_0}%",
    "Haiku -> Opus scored {mmlu.opus.cascade_accuracy}% against {mmlu.opus.final_accuracy}% for "
    "Opus alone and {mmlu.opus.cheap_accuracy}% for Haiku alone, saving "
    "{mmlu.opus.saving_serving}% serving cost and {mmlu.opus.saving_all_in}% once the audit is "
    "paid for; Haiku -> Sonnet saved {mmlu.sonnet.saving_serving}% and "
    "{mmlu.sonnet.saving_all_in}%",
    "In {review.cases} hand-checked MMLU-Pro cases",
    'same "wrong" answer, {review.gold_wrong} gold labels were wrong, {review.ambiguous} were '
    "ambiguous and {review.real_errors} were real errors",
    "A threshold tuned on STEM ({mmlu.opus.drift.threshold}, {mmlu.opus.drift.stem_error}% error) "
    "gave {mmlu.opus.drift.humanities_error}% error on the humanities questions",
    "with about {mmlu.drift.audits} audits the audit never said `ok` but confirmed the breach in "
    "only {mmlu.drift.breach_min}-{mmlu.drift.breach_max}% of draws",
    "on {mmlu.opus.tasks} MMLU-Pro questions ({mmlu.per_subject} per subject) and "
    "{bbh.opus.tasks} BIG-Bench Hard examples",
    "Claude Code CLI for ${mmlu.cli_cost} of CLI-reported cost",
    "| Accuracy: Haiku only / final tier only / cascade | {mmlu.opus.cheap_accuracy}% / "
    "{mmlu.opus.final_accuracy}% / {mmlu.opus.cascade_accuracy}% | "
    "{mmlu.sonnet.cheap_accuracy}% / {mmlu.sonnet.final_accuracy}% / "
    "{mmlu.sonnet.cascade_accuracy}% |",
    pair_row("Served by Haiku alone", "served_alone"),
    "| Haiku error on those, vs gold | "
    + ci("mmlu.opus.cheap_error_skipped")
    + " | "
    + ci("mmlu.sonnet.cheap_error_skipped")
    + " |",
    "| Haiku disagreement with the final tier on those | "
    + ci("mmlu.opus.disagreement_skipped")
    + " | "
    + ci("mmlu.sonnet.disagreement_skipped")
    + " |",
    "| Audit-only estimate (weighted; {mmlu.opus.audits} and {mmlu.sonnet.audits} audits) | "
    + ci("mmlu.opus.audit_only")
    + " | "
    + ci("mmlu.sonnet.audit_only")
    + " |",
    pair_row("Haiku errors the final tier shared (audit cannot see them)", "shared_share"),
    "| Saving vs final tier alone: serving / with audit | {mmlu.opus.saving_serving}% / "
    "{mmlu.opus.saving_all_in}% | {mmlu.sonnet.saving_serving}% / {mmlu.sonnet.saving_all_in}% |",
    "re-drawn {mmlu.opus.coverage.seeds} times offline, its 95% interval covered the true "
    "disagreement rate of the run's skipped cases about {mmlu.opus.coverage.disagreement_0}% of "
    "the time on MMLU-Pro ({bbh.coverage.disagreement_min}-{bbh.coverage.disagreement_max}% on "
    "BBH",
    "It covered the gold-graded error only {mmlu.opus.coverage.error}% (Opus) and "
    "{mmlu.sonnet.coverage.error_2}% (Sonnet) of the time",
    "reported `ok` against a {bbh.sonnet.tolerance}% tolerance ({bbh.sonnet.audit_only}%, upper "
    "bound {bbh.sonnet.audit_only.hi}%) while the gold-graded error was "
    + ci("bbh.sonnet.ref_error")
    + ". An `ok` came up in {bbh.sonnet.coverage.ok}% of re-draws",
    "In {review.cases} hand-checked MMLU-Pro cases (drawn",
    "{review.gold_wrong} gold labels were wrong, {review.ambiguous} were ambiguous, and "
    "{review.real_errors} were real errors",
    "itself wrong {mmlu.final_error_min}-{mmlu.final_error_max}% of the time",
    "a threshold tuned on STEM questions ({mmlu.opus.drift.threshold}, "
    "{mmlu.opus.drift.stem_error}% error) gave {mmlu.opus.drift.humanities_error}% error",
    "(~{mmlu.drift.audits} audits)",
    "in only {mmlu.drift.breach_min}-{mmlu.drift.breach_max}% of re-draws",
    "where Opus was {arithmetic.final_accuracy}% correct",
    "{mmlu.shared_share_min}-{mmlu.shared_share_max}% of Haiku's errors were",
    "({mmlu.opus.disagreement_skipped}% vs {mmlu.sonnet.disagreement_skipped}%)",
    "About {mmlu.drift.audits} audits caught the humanities breach in "
    "{mmlu.drift.breach_min}-{mmlu.drift.breach_max}%",
    "The {review.cases}-case hand check",
    "{mmlu.shared_share_min}-{mmlu.shared_share_max}% of the cheap tier's errors",
]

FIXED = [
    "Claude Haiku 5.5",
    "Claude Opus 5.5",
    "Sonnet 5.5",
    "Python 3.11",
    "Oh & Gobet 2024",
    "threshold 0.80",
    "70% of the data",
    "the other 30%",
    "`audit --fail-on-breach` exits 3",
    "(default 0.95)",
    "its 95% interval",
    "about five expected audits",
    "covered 99.8-100%",  # the interval before the fix; superseded code, said so in the text
    "two ad hoc seeds",
    '"System 1.5"',
    "a floor > 0",
    "weighted by 1/π",
]

DEMO_CLAIMS = [
    "answers {answered}% of tasks without escalating",
    "disagreement on skipped cases at {disagreement}%",
    "puts the error at {error}%",
    "({error_lo}%) is above the {tolerance}% tolerance",
    "recommends {recommended}",
    "Offline demo with two simulated models ({n} tasks",
]


def demo_values(text: str) -> dict[str, str]:
    def grab(pattern: str) -> str:
        found = re.search(pattern, text)
        assert found, pattern
        return found.group(1)

    escalated = float(grab(r"escalation rate\s+([\d.]+)%"))
    return {
        "answered": f"{100 - escalated:.0f}",
        "disagreement": grab(r"disagreement on skipped\s+([\d.]+)%"),
        "error": grab(r"error on skipped\s+([\d.]+)%"),
        "error_lo": grab(r"error on skipped\s+[\d.]+% \[([\d.]+)%"),
        "tolerance": grab(r"tolerance ([\d.]+?)(?:\.0)?%"),
        "recommended": grab(r"recommended threshold\s+([\d.]+)"),
        "n": grab(r"demo: (\d+) simulated"),
    }


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> str:
    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main(["demo", "--out", str(tmp_path_factory.mktemp("demo") / "d")]) == 0
    return out.getvalue()


def test_readme_numbers_are_copied_from_generated_output(demo: str) -> None:
    if not NUMBERS.exists():
        pytest.skip("experiments/ is not part of the sdist")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    values = json.loads(NUMBERS.read_text(encoding="utf-8"))
    values.update(demo_values(demo))
    result = check(text, README + DEMO_CLAIMS, FIXED, values)
    problems = [f"claim not found: {c}" for c in result.missing_claims]
    problems += [f"fixed text not found: {c}" for c in result.missing_fixed]
    problems += [f"number not from generated output: {c}" for c in result.unchecked]
    assert not problems, "\n".join(problems)


def test_readme_demo_output_is_what_the_demo_prints(demo: str) -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = text.split("shadowgate demo --out shadowgate-demo\n```\n\n```text\n", 1)[1]
    block = block.split("\n```", 1)[0]
    printed = {TIMING.sub("", line.rstrip()) for line in demo.splitlines()}
    missing = [line for line in block.splitlines() if TIMING.sub("", line) not in printed]
    assert not missing, missing


def test_citation_has_no_unchecked_number() -> None:
    path = ROOT / "CITATION.cff"
    if not path.exists():
        pytest.skip("CITATION.cff is not part of the sdist")
    result = check(path.read_text(encoding="utf-8"), [], ["cff-version: 1.2.0"], {})
    assert not result.unchecked, result.unchecked


def test_the_check_catches_a_changed_or_unlisted_number() -> None:
    text = "Error 9.3% (7.9-11.0) on 1680 tasks. See `claude-opus-5-5`, 2026-10-10, v0.1.1."
    claims = ["Error {a}% ({b}-{c}) on {n} tasks"]
    values = {"a": "9.3", "b": "7.9", "c": "11.0", "n": "1680"}
    clean = check(text, claims, [], values)
    assert not (clean.missing_claims or clean.unchecked)
    assert check(text, claims, [], {**values, "a": "9.4"}).missing_claims
    assert check(text + " Also 12 more.", claims, [], values).unchecked
