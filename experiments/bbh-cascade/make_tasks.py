"""Build a task-stratified BIG-Bench Hard (BBH) task file for shadowgate.

Source: https://github.com/suzgunmirac/BIG-Bench-Hard (MIT), ``bbh/*.json``. Each example has an
``input`` and an exact-match ``target``.

    git clone --depth 1 https://github.com/suzgunmirac/BIG-Bench-Hard /tmp/BIG-Bench-Hard
    python make_tasks.py /tmp/BIG-Bench-Hard -n 20 --seed 2026 -o tasks.jsonl

``dyck_languages`` is excluded: its targets are bracket sequences, which the ``normalized``
comparator (it strips punctuation) cannot grade. Three multiple-choice examples whose target is
option text rather than a label are dropped (see ``MALFORMED``). The other 26 tasks contribute
``-n`` examples each, drawn with ``random.Random(seed)``; task ids keep the source index.
Each prompt states the expected answer format.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

EXCLUDE = {"dyck_languages"}
# Multiple-choice tasks with a few targets that are option text instead of a label
# (movie_recommendation: 1, ruin_names: 2); those examples are dropped.
MALFORMED = {"movie_recommendation", "ruin_names"}
MC = re.compile(r"^\([A-R]\)$")


def answer_format(name: str, targets: list[str]) -> str:
    if all(MC.match(t) for t in targets):
        return "the option label in parentheses, for example (A)"
    if name == "word_sorting":
        return "the sorted words separated by single spaces"
    if all(re.fullmatch(r"-?\d+", t) for t in targets):
        return "a single integer"
    labels = sorted(set(targets))
    if len(labels) <= 3:
        return "exactly one of: " + ", ".join(labels)
    raise ValueError(f"{name}: no answer format for targets {labels[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo", type=Path, help="clone of github.com/suzgunmirac/BIG-Bench-Hard")
    ap.add_argument("-n", type=int, required=True, help="examples per BBH task")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = []
    for path in sorted((args.repo / "bbh").glob("*.json")):
        name = path.stem
        if name in EXCLUDE:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))["examples"]
        # keep the index into the source file for the task id
        examples = [
            (i, e) for i, e in enumerate(raw) if name not in MALFORMED or MC.match(e["target"])
        ]
        fmt = answer_format(name, [e["target"] for _, e in examples])
        for i, e in rng.sample(examples, args.n):
            tasks.append(
                {
                    "id": f"bbh-{name}-{i}",
                    "prompt": f"{e['input'].strip()}\n\nGive the answer as {fmt}.",
                    "reference": e["target"],
                    "meta": {"subject": name, "index": i},
                }
            )
    rng.shuffle(tasks)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks ({args.n} per BBH task) to {args.out}")


if __name__ == "__main__":
    main()
