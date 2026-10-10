"""Analyze a two-pair cascade experiment (Haiku -> Opus, Haiku -> Sonnet) from exported decisions.

Reads ``decisions-<pair>-eval.jsonl`` and ``decisions-<pair>-serve.jsonl`` from an experiment
directory (written by its ``run.sh``) and prints, per pair:

* ground truth from the eval run (every tier answered every task): the cheap tier's true error
  on the cases it would serve alone (skipped cases) against its disagreement with the final tier,
  split into errors the audit sees and errors both tiers share (invisible to the audit);
* the final tier's own error;
* the serve run's audit-only estimate (weighted disagreement, references ignored) with its
  interval and status, next to the reference-graded rate;
* serving, audit and all-in cost against the final tier alone;
* AUROC of the cheap tier's verbal confidence;
* an offline coverage check: the serve-mode audit re-drawn over many seeds (default 5000) with
  references stripped, scoring the interval against the true disagreement rate and the true
  error rate, with Monte Carlo standard errors (``mc_se``) for each simulated proportion;
* a drift check: the threshold is picked on STEM subjects and applied to humanities.

No model calls. ``--json`` writes every number to a file.

    uv run python experiments/analyze_cascade.py experiments/mmlu-pro-cascade --json results.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from shadowgate.audit import summarize
from shadowgate.stats import auroc, wilson
from shadowgate.types import Decision, ShadowResult

PAIRS = ("opus", "sonnet")
THRESHOLD = 0.8  # [[tiers]] threshold in the pair configs
STRATA = [(0.80, 0.95, 0.5), (0.95, 1.0, 0.2)]  # [audit] strata in the pair configs
RATE = 0.2  # [audit] rate outside the strata
TOLERANCE = 0.05
STEM = {"math", "physics", "chemistry", "engineering", "computer science", "biology"}
HUMANITIES = {"law", "history", "philosophy"}


# --------------------------------------------------------------------------- loading


def load(path: Path) -> list[Decision]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [Decision.from_dict(r) for r in rows if "shadowgate_run" not in r]


def score(d: Decision) -> float | None:
    c = d.attempts[0].confidence
    return None if c is None else c.score


def eq(j: Any) -> bool | None:
    return None if j is None else j.equivalent


def served_alone(d: Decision, t: float) -> bool:
    """Would the cheap tier serve this task alone at threshold ``t``?"""
    fast = d.attempts[0]
    s = score(d)
    return fast.error is None and bool(fast.answer) and s is not None and s >= t


# --------------------------------------------------------------------------- estimates


def pct(e: Any) -> dict[str, float | int | None]:
    return {"value": e.value, "lo": e.lo, "hi": e.hi, "n": e.n}


def truth(decisions: list[Decision], t: float) -> dict[str, Any]:
    """Ground truth on the skipped cases at threshold ``t`` (eval run, every tier answered)."""
    skipped = [d for d in decisions if served_alone(d, t)]
    n = len(skipped)
    cheap_wrong = disagree = shared = caught = false_alarm = both_wrong_differ = 0
    for d in skipped:
        fast, final = d.attempts[0], d.attempts[1]
        fw, gw, ag = eq(fast.correct) is False, eq(final.correct) is False, eq(fast.agreement)
        cheap_wrong += fw
        disagree += ag is False
        shared += fw and ag is True  # both wrong with the same answer: the audit cannot see it
        caught += fw and ag is False and not gw  # audit flags a real error, final tier right
        both_wrong_differ += fw and ag is False and gw
        false_alarm += (not fw) and ag is False  # audit flags a correct answer
    final_wrong = sum(eq(d.attempts[1].correct) is False for d in decisions)
    final_wrong_skipped = sum(eq(d.attempts[1].correct) is False for d in skipped)
    return {
        "threshold": t,
        "n_tasks": len(decisions),
        "n_skipped": n,
        "skip_rate": n / len(decisions) if decisions else None,
        "cheap_error_skipped": pct(wilson(cheap_wrong, n)),
        "disagreement_skipped": pct(wilson(disagree, n)),
        "cheap_wrong": cheap_wrong,
        "disagree": disagree,
        "shared_errors": shared,
        "caught_errors": caught,
        "both_wrong_differ": both_wrong_differ,
        "false_alarms": false_alarm,
        "final_error_all": pct(wilson(final_wrong, len(decisions))),
        "final_error_skipped": pct(wilson(final_wrong_skipped, n)),
        "cheap_error_all": pct(
            wilson(sum(eq(d.attempts[0].correct) is False for d in decisions), len(decisions))
        ),
    }


def aurocs(decisions: list[Decision]) -> dict[str, float | None]:
    scored = [d for d in decisions if score(d) is not None and d.attempts[0].answer]
    s = [float(score(d) or 0.0) for d in scored]
    return {
        "n": len(scored),
        "cheap_correct": auroc(s, [eq(d.attempts[0].correct) is True for d in scored]),
        "agrees_with_final": auroc(s, [eq(d.attempts[0].agreement) is True for d in scored]),
    }


def attempt_cost(d: Decision, i: int) -> float:
    c = d.attempts[i].completion
    return 0.0 if c is None or c.cost_usd is None else c.cost_usd


def costs(eval_ds: list[Decision], serve_ds: list[Decision]) -> dict[str, float | None]:
    """Recorded costs: final tier alone (eval run) vs the serve run's serving and audit.

    Each attempt's ``cost_usd`` as recorded: the price table where shadowgate resolved the model
    (Haiku here), otherwise the CLI's ``total_cost_usd`` (Opus and Sonnet here).
    """
    n = len(serve_ds)
    final_only = sum(attempt_cost(d, 1) for d in eval_ds) / len(eval_ds)
    cheap_only = sum(attempt_cost(d, 0) for d in eval_ds) / len(eval_ds)
    serving = sum(d.cost_usd or 0.0 for d in serve_ds) / n
    audit = sum(d.audit_cost_usd or 0.0 for d in serve_ds) / n
    return {
        "cheap_only_per_task": cheap_only,
        "final_only_per_task": final_only,
        "serving_per_task": serving,
        "audit_per_task": audit,
        "all_in_per_task": serving + audit,
        "saving_serving_only": 1 - serving / final_only,
        "saving_all_in": 1 - (serving + audit) / final_only,
        "audit_overhead": audit / serving if serving else None,
    }


def serve_audit(serve_ds: list[Decision]) -> dict[str, Any]:
    s = summarize(serve_ds, tolerance=TOLERANCE)
    d = s.disagreement
    return {
        "n_skipped": s.n_skipped,
        "n_audited": s.n_audited,
        "disagreement_weighted": None if d is None else {**pct(d), "n_eff": d.n_eff},
        "audit_only_status": s.audit_only_status or s.status,
        "skipped_error_ref": None if s.skipped_error is None else pct(s.skipped_error),
        "reference_status": s.status if s.status_metric == "skipped_error" else None,
        "final_tier_error_on_audits": None
        if s.audit_tier_error is None
        else pct(s.audit_tier_error),
    }


# --------------------------------------------------------------------------- audit simulation


def inclusion_prob(conf: float) -> float:
    for lo, hi, pi in STRATA:
        if lo <= conf < hi or (hi >= 1.0 and conf == hi):
            return pi
    return RATE


def strip_reference(d: Decision) -> Decision:
    attempts = tuple(replace(a, correct=None) for a in d.attempts)
    return replace(d, task=replace(d.task, reference=None), attempts=attempts, correct=None)


def draw(decisions: list[Decision], t: float, rng: random.Random) -> list[Decision]:
    """One serve-mode audit sample at threshold ``t``, built from eval-mode decisions."""
    out = []
    for d in decisions:
        if not served_alone(d, t):
            out.append(replace(d, mode="serve", escalated=True, audit_cost_usd=0.0))
            continue
        fast, final = d.attempts[0], d.attempts[1]
        s = score(d)
        assert s is not None
        pi = inclusion_prob(s)
        if rng.random() < pi:
            shadow = ShadowResult(final.tier, pi, "done", attempt=final, agreement=fast.agreement)
        else:
            shadow = ShadowResult(final.tier, pi, "skipped")
        out.append(
            replace(
                d,
                mode="serve",
                escalated=False,
                final_tier=fast.tier,
                answer=fast.answer,
                attempts=(replace(fast, agreement=None, accepted=True),),
                shadow=shadow,
                audit_cost_usd=0.0,
            )
        )
    return out


def mc_se(p: float | None, k: int) -> float | None:
    """Monte Carlo standard error of a proportion ``p`` estimated from ``k`` simulated draws."""
    return None if p is None or k == 0 else (p * (1 - p) / k) ** 0.5


def coverage(decisions: list[Decision], t: float, seeds: int, tol: float) -> dict[str, Any]:
    tr = truth(decisions, t)
    true_err = tr["cheap_error_skipped"]["value"]
    true_dis = tr["disagreement_skipped"]["value"]
    blind = [strip_reference(d) for d in decisions]
    cover_dis = cover_err = 0
    statuses: Counter[str] = Counter()
    values: list[float] = []
    uppers: list[float] = []
    n_aud: list[int] = []
    for seed in range(seeds):
        s = summarize(draw(blind, t, random.Random(seed)), tolerance=tol)
        e = s.disagreement
        if e is None or e.lo is None or e.hi is None or e.value is None:
            statuses["no-data"] += 1
            continue
        cover_dis += e.lo <= true_dis <= e.hi
        cover_err += e.lo <= true_err <= e.hi
        statuses[s.status] += 1
        values.append(e.value)
        uppers.append(e.hi)
        n_aud.append(s.n_audited)
    k = len(values)
    uppers.sort()
    cov_dis = cover_dis / k if k else None
    cov_err = cover_err / k if k else None
    return {
        "threshold": t,
        "seeds": seeds,
        "tolerance": tol,
        "true_error": true_err,
        "true_disagreement": true_dis,
        "audits_mean": sum(n_aud) / k if k else None,
        "estimate_mean": sum(values) / k if k else None,
        "coverage_of_disagreement": cov_dis,
        "coverage_of_disagreement_mc_se": mc_se(cov_dis, k),
        "coverage_of_error": cov_err,
        "coverage_of_error_mc_se": mc_se(cov_err, k),
        "upper_median": uppers[k // 2] if k else None,
        "status": {key: v / seeds for key, v in sorted(statuses.items())},
        "status_mc_se": {key: mc_se(v / seeds, seeds) for key, v in sorted(statuses.items())},
    }


# --------------------------------------------------------------------------- drift


def pick_threshold(decisions: list[Decision], target: float) -> float | None:
    """Lowest threshold whose skipped-case error (vs references) is at most ``target``."""
    grid = sorted({s for d in decisions if (s := score(d)) is not None})
    for t in grid:
        tr = truth(decisions, t)
        e = tr["cheap_error_skipped"]["value"]
        if tr["n_skipped"] >= 20 and e is not None and e <= target:
            return t
    return None


def drift(decisions: list[Decision], target: float, seeds: int) -> dict[str, Any]:
    def subj(d: Decision) -> str:
        return str(d.task.meta.get("subject", ""))

    stem = [d for d in decisions if subj(d) in STEM]
    hum = [d for d in decisions if subj(d) in HUMANITIES]
    t = pick_threshold(stem, target)
    if t is None:
        return {"target": target, "threshold": None, "note": "no STEM threshold meets target"}
    return {
        "target": target,
        "threshold": t,
        "stem": truth(stem, t),
        "humanities": truth(hum, t),
        "humanities_audit": coverage(hum, t, seeds, target),
    }


# --------------------------------------------------------------------------- report


def f(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def ci(e: dict[str, Any] | None) -> str:
    if e is None or e.get("value") is None:
        return "-"
    return f"{f(e['value'])} ({f(e['lo'])}-{f(e['hi'])})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--seeds", type=int, default=5000)
    ap.add_argument("--drift-target", type=float, default=TOLERANCE)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    out: dict[str, Any] = {}
    for pair in PAIRS:
        ev = args.dir / f"decisions-{pair}-eval.jsonl"
        if not ev.exists():
            continue
        eval_ds = load(ev)
        sv = args.dir / f"decisions-{pair}-serve.jsonl"
        serve_ds = load(sv) if sv.exists() else []
        res: dict[str, Any] = {
            "truth": truth(eval_ds, THRESHOLD),
            "auroc": aurocs(eval_ds),
            "coverage": coverage(eval_ds, THRESHOLD, args.seeds, TOLERANCE),
        }
        if serve_ds:
            res["serve_audit"] = serve_audit(serve_ds)
            res["cost"] = costs(eval_ds, serve_ds)
        if any(d.task.meta.get("subject") in STEM for d in eval_ds):
            res["drift"] = drift(eval_ds, args.drift_target, args.seeds)
        out[pair] = res

        tr = res["truth"]
        print(f"== haiku -> {pair}  ({tr['n_tasks']} tasks, threshold {THRESHOLD})")
        print(f"  skipped cases            {tr['n_skipped']} ({f(tr['skip_rate'])})")
        print(f"  cheap error on skipped   {ci(tr['cheap_error_skipped'])}  [truth]")
        print(f"  disagreement on skipped  {ci(tr['disagreement_skipped'])}  [census]")
        print(
            f"  of {tr['cheap_wrong']} cheap errors: {tr['caught_errors']} flagged with final "
            f"right, {tr['both_wrong_differ']} flagged with final also wrong, "
            f"{tr['shared_errors']} shared (invisible); {tr['false_alarms']} false alarms"
        )
        print(f"  final-tier error, all    {ci(tr['final_error_all'])}")
        print(f"  cheap-tier error, all    {ci(tr['cheap_error_all'])}")
        a = res["auroc"]
        print(
            f"  AUROC cheap correct      {a['cheap_correct']:.3f}  (agrees with final "
            f"{a['agrees_with_final']:.3f}, n {a['n']})"
        )
        if "serve_audit" in res:
            sa = res["serve_audit"]
            print(
                f"  serve audit              {sa['n_audited']} of {sa['n_skipped']} audited; "
                f"weighted disagreement {ci(sa['disagreement_weighted'])}, "
                f"audit-only status {sa['audit_only_status']}"
            )
            print(
                f"  reference-graded         {ci(sa['skipped_error_ref'])}, "
                f"status {sa['reference_status']}"
            )
            c = res["cost"]
            print(
                f"  cost/task (recorded)      final only ${c['final_only_per_task']:.5f}; "
                f"serving ${c['serving_per_task']:.5f} + audit ${c['audit_per_task']:.5f}"
                f" = ${c['all_in_per_task']:.5f}"
            )
            print(
                f"  saving vs final only     serving {f(c['saving_serving_only'])}, "
                f"all-in {f(c['saving_all_in'])}"
            )
        cv = res["coverage"]
        print(
            f"  coverage ({cv['seeds']} seeds)    mean est {f(cv['estimate_mean'], 2)}, "
            f"audits {cv['audits_mean']:.1f}; covers disagreement "
            f"{f(cv['coverage_of_disagreement'])} (MC SE "
            f"{f(cv['coverage_of_disagreement_mc_se'], 2)}), covers error "
            f"{f(cv['coverage_of_error'])} (MC SE {f(cv['coverage_of_error_mc_se'], 2)}); "
            f"status {cv['status']}"
        )
        if "drift" in res:
            dr = res["drift"]
            if dr["threshold"] is None:
                print(f"  drift                    {dr['note']}")
            else:
                st, hu, ha = dr["stem"], dr["humanities"], dr["humanities_audit"]
                print(
                    f"  drift: threshold {dr['threshold']} picked on STEM "
                    f"(error {ci(st['cheap_error_skipped'])}, skip {f(st['skip_rate'])})"
                )
                print(
                    f"    humanities: error {ci(hu['cheap_error_skipped'])}, disagreement "
                    f"{ci(hu['disagreement_skipped'])}, skip {f(hu['skip_rate'])}"
                )
                print(
                    f"    humanities audit sim: mean est {f(ha['estimate_mean'], 2)}, "
                    f"covers error {f(ha['coverage_of_error'])}, status {ha['status']} "
                    f"(MC SE {ha['status_mc_se']})"
                )
        print()

    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
