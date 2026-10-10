#!/usr/bin/env python3
"""Budget guard around the Claude Code CLI, used as the claude-code backend's ``executable``.

Runs ``claude`` with the given arguments and stdin, logs the ``total_cost_usd`` the CLI reports
to ``costs.jsonl`` (one line per real call), and refuses to start a call once the logged total
plus a per-call reserve would exceed ``SHADOWGATE_BUDGET_USD`` (default 88). A refused call
returns a CLI-style error result, which the backend records as an error.

This is the hard cap: it counts the CLI's own reported cost, including retries, not the
shadowgate price-table estimate that ``[run].max_cost_usd`` uses.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path(os.environ.get("SHADOWGATE_COST_LOG", HERE / "costs.jsonl"))
BUDGET = float(os.environ.get("SHADOWGATE_BUDGET_USD", "88"))
# Headroom for one call, not per call: with parallel workers the cap can be overshot by a few calls.
RESERVE = float(os.environ.get("SHADOWGATE_RESERVE_USD", "0.25"))


def spent() -> float:
    if not LOG.exists():
        return 0.0
    total = 0.0
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += float(json.loads(line).get("total_cost_usd") or 0.0)
    return total


def main() -> int:
    args = sys.argv[1:]
    model = args[args.index("--model") + 1] if "--model" in args else "?"
    with open(str(LOG) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        total = spent()
        fcntl.flock(lock, fcntl.LOCK_UN)
    if total + RESERVE > BUDGET:
        msg = f"budget guard: ${total:.4f} spent, cap ${BUDGET:.2f}; call refused"
        print(json.dumps({"type": "result", "is_error": True, "result": msg}))
        print(msg, file=sys.stderr)
        return 1
    claude = shutil.which("claude")
    if claude is None:
        print("claude not found on PATH", file=sys.stderr)
        return 127
    stdin = sys.stdin.buffer.read()
    t0 = time.time()
    proc = subprocess.run([claude, *args], input=stdin, capture_output=True)
    cost = 0.0
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace").strip().splitlines()[-1])
        cost = float(data.get("total_cost_usd") or 0.0)
    except (ValueError, IndexError, AttributeError):
        data = None
    entry = {
        "ts": round(t0, 3),
        "model": model,
        "total_cost_usd": cost,
        "exit": proc.returncode,
        "parsed": data is not None,
        "seconds": round(time.time() - t0, 2),
    }
    with open(str(LOG) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)
    sys.stdout.buffer.write(proc.stdout)
    sys.stderr.buffer.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
