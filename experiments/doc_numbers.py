"""Write experiments/numbers.json: every figure README.md quotes, from committed experiment files.

Reads each experiment's `analysis.json` (written by `analyze_cascade.py` from the committed
decisions), the MMLU-Pro call log `costs.jsonl`, the hand review `shared-errors-review.md` and the
arithmetic run's eval decisions. No model calls and no simulation, so it runs in a second; CI
reruns the analyses, then this, and diffs `experiments/`. `tests/test_docs.py` checks the README
against the output.

    uv run python experiments/doc_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_cascade as ac  # noqa: E402

PAIRS = {"opus": "Opus", "sonnet": "Sonnet"}


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def interval(n: dict[str, str], key: str, e: dict) -> None:
    n[key], n[f"{key}.lo"], n[f"{key}.hi"] = pct(e["value"]), pct(e["lo"]), pct(e["hi"])


def cascade(n: dict[str, str], name: str, analysis: dict) -> None:
    for pair, block in analysis.items():
        key = f"{name}.{pair}"
        t = block["truth"]
        n[f"{key}.tasks"] = str(t["n_tasks"])
        n[f"{key}.served_alone"] = pct(t["skip_rate"])
        interval(n, f"{key}.cheap_error_skipped", t["cheap_error_skipped"])
        interval(n, f"{key}.disagreement_skipped", t["disagreement_skipped"])
        n[f"{key}.shared_share"] = pct(t["shared_errors"] / t["cheap_wrong"], 0)
        n[f"{key}.cheap_accuracy"] = pct(1 - t["cheap_error_all"]["value"])
        n[f"{key}.final_accuracy"] = pct(1 - t["final_error_all"]["value"])
        n[f"{key}.final_error_0"] = pct(t["final_error_all"]["value"], 0)
        # Served wrong: the cheap tier's errors on skipped cases plus the final tier's errors on
        # escalated ones (final errors on all tasks minus those on skipped tasks).
        final_wrong = t["final_error_all"]["value"] * t["n_tasks"]
        final_wrong_skipped = t["final_error_skipped"]["value"] * t["n_skipped"]
        served_wrong = t["cheap_wrong"] + round(final_wrong - final_wrong_skipped)
        n[f"{key}.cascade_accuracy"] = pct(1 - served_wrong / t["n_tasks"])
        s = block["serve_audit"]
        n[f"{key}.audits"] = str(s["n_audited"])
        n[f"{key}.audit_only"] = pct(s["disagreement_weighted"]["value"])
        n[f"{key}.audit_only.lo"] = pct(s["disagreement_weighted"]["lo"])
        n[f"{key}.audit_only.hi"] = pct(s["disagreement_weighted"]["hi"])
        n[f"{key}.audit_only_status"] = s["audit_only_status"]
        interval(n, f"{key}.ref_error", s["skipped_error_ref"])
        c = block["cost"]
        n[f"{key}.saving_serving"] = pct(c["saving_serving_only"], 0)
        n[f"{key}.saving_all_in"] = pct(c["saving_all_in"], 0)
        cov = block["coverage"]
        n[f"{key}.coverage.seeds"] = str(cov["seeds"])
        n[f"{key}.coverage.disagreement"] = pct(cov["coverage_of_disagreement"])
        n[f"{key}.coverage.disagreement_0"] = pct(cov["coverage_of_disagreement"], 0)
        n[f"{key}.coverage.error"] = pct(cov["coverage_of_error"])
        n[f"{key}.coverage.error_2"] = pct(cov["coverage_of_error"], 2)
        n[f"{key}.coverage.ok"] = pct(cov["status"].get("ok", 0.0))
        n[f"{key}.tolerance"] = pct(cov["tolerance"], 0)
        if "drift" in block:
            d = block["drift"]
            n[f"{key}.drift.threshold"] = f"{d['threshold']:.2f}"
            n[f"{key}.drift.stem_error"] = pct(d["stem"]["cheap_error_skipped"]["value"])
            n[f"{key}.drift.humanities_error"] = pct(
                d["humanities"]["cheap_error_skipped"]["value"]
            )
            n[f"{key}.drift.audits"] = f"{d['humanities_audit']['audits_mean']:.0f}"
            n[f"{key}.drift.breach"] = pct(d["humanities_audit"]["status"]["breach"], 0)
            n[f"{key}.drift.ok"] = pct(d["humanities_audit"]["status"].get("ok", 0.0), 0)


def numbers() -> dict[str, str]:
    n: dict[str, str] = {}
    for name, folder in (("mmlu", "mmlu-pro-cascade"), ("bbh", "bbh-cascade")):
        analysis = json.loads((HERE / folder / "analysis.json").read_text(encoding="utf-8"))
        cascade(n, name, analysis)
    for name in ("mmlu", "bbh"):
        cov = [float(n[f"{name}.{p}.coverage.disagreement"]) for p in PAIRS]
        n[f"{name}.coverage.disagreement_min"] = f"{min(cov):.1f}"
        n[f"{name}.coverage.disagreement_max"] = f"{max(cov):.1f}"
        shared = [int(n[f"{name}.{p}.shared_share"]) for p in PAIRS]
        n[f"{name}.shared_share_min"], n[f"{name}.shared_share_max"] = (
            str(min(shared)),
            str(max(shared)),
        )
    breach = sorted(int(n[f"mmlu.{p}.drift.breach"]) for p in PAIRS)
    n["mmlu.drift.breach_min"], n["mmlu.drift.breach_max"] = str(breach[0]), str(breach[-1])
    audits = {n[f"mmlu.{p}.drift.audits"] for p in PAIRS}
    if len(audits) != 1:
        raise ValueError(f"drift audits differ between pairs: {audits}")
    n["mmlu.drift.audits"] = audits.pop()
    errors = sorted(int(n[f"mmlu.{p}.final_error_0"]) for p in PAIRS)
    n["mmlu.final_error_min"], n["mmlu.final_error_max"] = str(errors[0]), str(errors[-1])

    tasks = [
        json.loads(line)
        for line in (HERE / "mmlu-pro-cascade/tasks.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    per_subject = {t["meta"]["subject"] for t in tasks}
    counts = {sum(t["meta"]["subject"] == s for t in tasks) for s in per_subject}
    if len(counts) != 1:
        raise ValueError(f"subjects have unequal task counts: {counts}")
    n["mmlu.per_subject"] = str(counts.pop())

    calls = [
        json.loads(line)
        for line in (HERE / "mmlu-pro-cascade/costs.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    n["mmlu.cli_cost"] = f"{sum(c['total_cost_usd'] or 0 for c in calls):.0f}"

    review = (HERE / "mmlu-pro-cascade/shared-errors-review.md").read_text("utf-8")
    rows = re.findall(r"^\| \d+ \|.*\| ([^|]+) \|$", review, re.M)  # question rows only
    kinds = {"gold wrong": 0, "ambiguous": 0, "likely a real model error": 0}
    for verdict in rows:
        kinds[next(k for k in kinds if verdict.startswith(k))] += 1
    n["review.cases"] = str(len(rows))
    n["review.gold_wrong"] = str(kinds["gold wrong"])
    n["review.ambiguous"] = str(kinds["ambiguous"])
    n["review.real_errors"] = str(kinds["likely a real model error"])

    arithmetic = ac.load(HERE / "arithmetic-haiku-opus/decisions-eval.jsonl")
    right = sum(ac.eq(d.attempts[1].correct) is True for d in arithmetic)
    n["arithmetic.final_accuracy"] = pct(right / len(arithmetic), 0)
    return dict(sorted(n.items()))


def main() -> None:
    text = json.dumps(numbers(), indent=1) + "\n"
    (HERE / "numbers.json").write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {HERE / 'numbers.json'}")


if __name__ == "__main__":
    main()
