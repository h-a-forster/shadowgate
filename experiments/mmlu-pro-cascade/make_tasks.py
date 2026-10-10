"""Build a subject-stratified MMLU-Pro task file for shadowgate.

Source: the MMLU-Pro test split (12,032 questions, TIGER-Lab/MMLU-Pro on Hugging Face). This
environment cannot reach huggingface.co, so the script reads the authors' own copy of the test
split that ships inside ``eval_results/model_outputs_gpt-4o-2024-08-06_5shots.zip`` in
https://github.com/TIGER-AI-Lab/MMLU-Pro (same question ids, questions, options and gold
answers; the gpt-4o predictions in that file are ignored).
Question 3983 is dropped: its ``answer`` (C) and ``answer_index`` (1) disagree.

    git clone --depth 1 https://github.com/TIGER-AI-Lab/MMLU-Pro /tmp/MMLU-Pro
    python make_tasks.py /tmp/MMLU-Pro -n 20 --seed 1 -o pilot-tasks.jsonl
    python make_tasks.py /tmp/MMLU-Pro -n 1680 --seed 2026 --skip pilot-tasks.jsonl -o tasks.jsonl

Sampling: equal counts per subject (14 subjects; when n is not a multiple of 14, a seeded
choice of subjects gets one extra), drawn with ``random.Random(seed)`` from the
questions of that subject sorted by id. Task ids are ``mmlupro-<question_id>``. The ``--skip``
option excludes tasks already used (for example a pilot file), so later runs draw fresh tasks.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

ZIP = "eval_results/model_outputs_gpt-4o-2024-08-06_5shots.zip"
LETTERS = "ABCDEFGHIJ"
STEM = {"math", "physics", "chemistry", "engineering", "computer science", "biology"}
HUMANITIES = {"law", "history", "philosophy"}


def load(repo: Path) -> list[dict]:
    with zipfile.ZipFile(repo / ZIP) as z:
        (name,) = [n for n in z.namelist() if n.endswith(".json")]
        rows = json.loads(z.read(name))
    out = []
    for r in rows:
        options = r["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        idx = int(r["answer_index"])
        if LETTERS[idx] != r["answer"]:
            continue  # inconsistent gold fields (question 3983: answer C, answer_index 1)
        out.append({**r, "options": list(options), "answer_index": idx})
    assert len(rows) == 12032 and len(out) == 12031, (len(rows), len(out))
    return out


def prompt(r: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
    return (
        f"The following is a multiple-choice question about {r['category']}.\n\n"
        f"Question: {r['question'].strip()}\n\nOptions:\n{opts}"
    )


def group(category: str) -> str:
    if category in STEM:
        return "stem"
    if category in HUMANITIES:
        return "humanities"
    return "social/other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo", type=Path, help="clone of github.com/TIGER-AI-Lab/MMLU-Pro")
    ap.add_argument("-n", type=int, required=True, help="total tasks")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip", type=Path, action="append", default=[], help="task file to exclude")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    skip: set[str] = set()
    for p in args.skip:
        skip |= {json.loads(line)["id"] for line in p.read_text().splitlines() if line.strip()}
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in load(args.repo):
        if f"mmlupro-{r['question_id']}" not in skip:
            by_cat[r["category"]].append(r)
    cats = sorted(by_cat)
    rng = random.Random(args.seed)
    base, extra = divmod(args.n, len(cats))
    bonus = set(rng.sample(cats, extra))  # subjects that get one more task when n % 14 != 0
    tasks = []
    for c in cats:
        pool = sorted(by_cat[c], key=lambda r: int(r["question_id"]))
        for r in rng.sample(pool, base + (c in bonus)):
            tasks.append(
                {
                    "id": f"mmlupro-{r['question_id']}",
                    "prompt": prompt(r),
                    "reference": r["answer"],
                    "meta": {
                        "subject": c,
                        "group": group(c),
                        "src": r["src"],
                        "n_options": len(r["options"]),
                    },
                }
            )
    rng.shuffle(tasks)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks ({base}-{base + bool(extra)} per subject) to {args.out}")


if __name__ == "__main__":
    main()
