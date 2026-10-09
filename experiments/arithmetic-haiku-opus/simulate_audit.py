"""Offline check of the audit interval's coverage on the recorded Haiku -> Opus run.

The eval run called Opus on every task, so every skipped case's disagreement with Opus is known.
This script re-draws the serve-mode audit sample many times with the same design (inclusion
probability 0.5 for confidence in [0.80, 0.95), 0.2 for [0.95, 1.00]), strips the references so
the status is driven by the audit alone, and checks how often the weighted interval covers the
true skipped-case rate (the reference-graded error over all skipped cases). No model calls.

    uv run python experiments/arithmetic-haiku-opus/simulate_audit.py [--seeds 1000]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from shadowgate.audit import summarize
from shadowgate.types import Decision, ShadowResult

HERE = Path(__file__).resolve().parent
STRATA = [(0.80, 0.95, 0.5), (0.95, 1.0, 0.2)]  # [audit] strata in shadowgate.toml
TOLERANCE = 0.05


def inclusion_prob(score: float) -> float:
    for lo, hi, pi in STRATA:
        if lo <= score < hi or (hi == 1.0 and score == 1.0):
            return pi
    raise ValueError(f"no stratum for confidence {score}")


def load_eval() -> list[Decision]:
    with open(HERE / "decisions-eval.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [Decision.from_dict(r) for r in rows if "shadowgate_run" not in r]


def strip_reference(d: Decision) -> Decision:
    """Drop the reference and every reference grade, keeping agreement with Opus."""
    attempts = tuple(replace(a, correct=None) for a in d.attempts)
    return replace(d, task=replace(d.task, reference=None), attempts=attempts, correct=None)


def draw(decisions: list[Decision], rng: random.Random) -> list[Decision]:
    """One serve-mode audit sample: skipped cases audited by Opus with probability pi."""
    out = []
    for d in decisions:
        if d.escalated:
            out.append(replace(d, mode="serve", audit_cost_usd=0.0))
            continue
        fast, final = d.attempts
        assert fast.confidence is not None and fast.confidence.score is not None
        pi = inclusion_prob(fast.confidence.score)
        audit_cost = 0.0
        if rng.random() < pi:
            shadow = ShadowResult("opus", pi, "done", attempt=final, agreement=fast.agreement)
            audit_cost = final.completion.cost_usd if final.completion else None
        else:
            shadow = ShadowResult("opus", pi, "skipped")
        fast_only = (replace(fast, agreement=None),)
        out.append(
            replace(d, mode="serve", attempts=fast_only, shadow=shadow, audit_cost_usd=audit_cost)
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=1000)
    args = ap.parse_args()

    decisions = load_eval()
    graded = summarize(decisions, tolerance=TOLERANCE)
    assert graded.skipped_error is not None and graded.skipped_error.value is not None
    true_rate = graded.skipped_error.value
    blind = [strip_reference(d) for d in decisions]

    covered = below = above = 0
    statuses: Counter[str] = Counter()
    n_audits: list[int] = []
    values: list[float] = []
    uppers: list[float] = []
    for seed in range(args.seeds):
        s = summarize(draw(blind, random.Random(seed)), tolerance=TOLERANCE)
        est = s.disagreement
        assert est is not None and est.lo is not None and est.hi is not None
        if est.lo <= true_rate <= est.hi:
            covered += 1
        elif true_rate < est.lo:
            below += 1
        else:
            above += 1
        statuses[s.status] += 1
        values.append(est.value if est.value is not None else 0.0)
        uppers.append(est.hi)
        n_audits.append(s.n_audited)

    n = args.seeds
    print(f"skipped cases: {graded.n_skipped}, true rate (vs reference): {true_rate:.4f}")
    print(
        f"seeds: {n}, audits per draw: mean {sum(n_audits) / n:.1f}, "
        f"min {min(n_audits)}, max {max(n_audits)}"
    )
    print(
        f"coverage of the 95% interval: {covered / n:.3f} "
        f"(true rate below interval {below / n:.3f}, above {above / n:.3f})"
    )
    uppers.sort()
    print(
        f"point estimate: mean {sum(values) / n:.4f}, zero disagreements in "
        f"{sum(1 for v in values if v == 0) / n:.3f} of draws"
    )
    print(
        f"upper bound: median {uppers[n // 2]:.4f}, 5th-95th percentile "
        f"{uppers[n // 20]:.4f}-{uppers[n - 1 - n // 20]:.4f}"
    )
    print(
        f"status vs {TOLERANCE:.0%} tolerance: "
        + ", ".join(f"{k} {v / n:.3f}" for k, v in sorted(statuses.items()))
    )


if __name__ == "__main__":
    main()
