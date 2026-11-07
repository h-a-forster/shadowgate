"""Offline check of the audit interval on known populations, stressed and published.

Each design is a list of strata (cases, inclusion probability, disagreements). Every draw audits
each case independently with its inclusion probability, as the serve-mode audit does, and scores
``weighted_proportion(..., finite=True)`` against the true rate. "before" is the same interval
with a mean-weight pseudo-observation, the choice this check rejected. The second table sweeps a
light stratum plus a heavy one to find where the interval stops holding. No model calls.

    uv run python experiments/audit-interval/simulate_designs.py \
        --json experiments/audit-interval/analysis.json > experiments/audit-interval/analysis.txt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from shadowgate.stats import FiniteSample, weighted_proportion

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import analyze_cascade as ac  # noqa: E402

LEVEL = 0.95
Strata = list[tuple[int, float, int]]

STRESS: dict[str, Strata] = {
    "150@0.5 (0) + 300@0.2 (0) + 1000@0.05 (40)": [(150, 0.5, 0), (300, 0.2, 0), (1000, 0.05, 40)],
    "200@0.9 (0) + 800@0.02 (50)": [(200, 0.9, 0), (800, 0.02, 50)],
    "400@0.5 (0) + 1000@0.2 (60)": [(400, 0.5, 0), (1000, 0.2, 60)],
    "400@0.5 (0) + 1000@0.05 (10)": [(400, 0.5, 0), (1000, 0.05, 10)],
    "300@0.5 (30) + 1000@0.2 (30)": [(300, 0.5, 30), (1000, 0.2, 30)],
    "300@0.5 (150) + 1000@0.2 (300)": [(300, 0.5, 150), (1000, 0.2, 300)],
}
GRID_PI = (0.01, 0.02, 0.05, 0.1, 0.2)
GRID_AUDITS = (2, 3, 5, 10, 20, 50)  # expected audits in the heavy stratum, N pi
GRID_HITS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)  # expected disagreements audited there, K pi
GRID_LIGHT = (400, 0.5, 0)


def published(directory: str, pair: str) -> Strata:
    """Strata of a published run's skipped cases, from its committed eval decisions."""
    cases: Counter[float] = Counter()
    hits: Counter[float] = Counter()
    for d in ac.load(ROOT / directory / f"decisions-{pair}-eval.jsonl"):
        if ac.served_alone(d, ac.THRESHOLD):
            score = ac.score(d)
            assert score is not None
            pi = ac.inclusion_prob(score)
            cases[pi] += 1
            hits[pi] += not ac.eq(d.attempts[0].agreement)
    return [(cases[pi], pi, hits[pi]) for pi in sorted(cases, reverse=True)]


def tails(design: Strata, seeds: int) -> dict[str, Any]:
    units = [(pi, i < k) for n, pi, k in design for i in range(n)]
    true = sum(y for _, y in units) / len(units)
    tally = {name: {"lo_above": 0, "hi_below": 0, "width": 0.0} for name in ("before", "after")}
    for seed in range(seeds):
        rng = random.Random(seed)
        ys: list[bool] = []
        weights: list[float] = []
        for pi, y in units:
            if rng.random() < pi:
                ys.append(y)
                weights.append(1.0 / pi)
        est = weighted_proportion(ys, weights, LEVEL, finite=True)
        assert est.lo is not None and est.hi is not None
        sample = FiniteSample.of(ys, weights)
        mean = replace(sample, max_weight=sample.weight / sample.n)
        for name, (lo, hi) in (("before", mean.bounds(LEVEL)), ("after", (est.lo, est.hi))):
            tally[name]["lo_above"] += true < lo
            tally[name]["hi_below"] += hi < true
            tally[name]["width"] += hi - lo
    out: dict[str, Any] = {"cases": len(units), "true": true}
    for name, t in tally.items():
        low, high = t["lo_above"] / seeds, t["hi_below"] / seeds
        out[name] = {
            "coverage": 1.0 - low - high,
            "lo_above_truth": low,
            "hi_below_truth": high,
            "mean_width": t["width"] / seeds,
        }
    return out


def grid_cell(cell: tuple[float, int, float, int]) -> dict[str, Any]:
    pi, audits, hits, seeds = cell
    n = round(audits / pi)
    k = min(n, max(1, round(hits / pi)))
    res = tails([GRID_LIGHT, (n, pi, k)], seeds)["after"]
    return {"pi": pi, "expected_audits": audits, "cases": n, "disagreements": k, **res}


def row(label: str, r: dict[str, Any]) -> str:
    b, a = r["before"], r["after"]
    return (
        f"{label:46} {r['true']:6.2%}  {b['coverage']:.3f} {b['lo_above_truth']:.3f} "
        f"{b['hi_below_truth']:.3f} {b['mean_width']:.3f}   {a['coverage']:.3f} "
        f"{a['lo_above_truth']:.3f} {a['hi_below_truth']:.3f} {a['mean_width']:.3f}"
    )


def tails_of(args: tuple[Strata, int]) -> dict[str, Any]:
    return tails(*args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=4000)
    ap.add_argument("--grid-seeds", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    designs = dict(STRESS)
    designs["arithmetic 74@0.5 (3) + 107@0.2 (0)"] = [(74, 0.5, 3), (107, 0.2, 0)]
    for label, directory in (("MMLU-Pro", "mmlu-pro-cascade"), ("BBH", "bbh-cascade")):
        for pair in ac.PAIRS:
            strata = published(directory, pair)
            text = " + ".join(f"{n}@{pi:g} ({k})" for n, pi, k in strata)
            designs[f"{label} {pair} {text}"] = strata

    cells = [(p, m, h, args.grid_seeds) for p in GRID_PI for m in GRID_AUDITS for h in GRID_HITS]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(tails_of, [(d, args.seeds) for d in designs.values()]))
        grid = list(pool.map(grid_cell, cells))

    print(f"Designs ({args.seeds} draws each, 95% interval): N@pi (disagreements)")
    print(f"{'':46} {'true':>6}  before: cov lo>t hi<t width   after: cov lo>t hi<t width")
    for label, r in zip(designs, results, strict=True):
        print(row(label, r))
    print()
    print(
        f"Heavy stratum beside {GRID_LIGHT[0]}@{GRID_LIGHT[1]} (0), {args.grid_seeds} draws per "
        f"cell, {len(GRID_PI)} pi x {len(GRID_HITS)} disagreement counts: worst tail by N pi"
    )
    print("N pi   worst hi<t   worst lo>t")
    worst = []
    for audits in GRID_AUDITS:
        rows = [g for g in grid if g["expected_audits"] == audits]
        hi = max(g["hi_below_truth"] for g in rows)
        lo = max(g["lo_above_truth"] for g in rows)
        worst.append({"expected_audits": audits, "worst_hi_below": hi, "worst_lo_above": lo})
        print(f"{audits:4}   {hi:10.3f}   {lo:10.3f}")

    if args.json:
        payload = {
            "seeds": args.seeds,
            "grid_seeds": args.grid_seeds,
            "level": LEVEL,
            "designs": {
                k: {"strata": designs[k], **r} for k, r in zip(designs, results, strict=True)
            },
            "grid_worst": worst,
            "grid": grid,
        }
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
