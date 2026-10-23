#!/usr/bin/env python3
"""Budget guard around the Claude Code CLI, used as the claude-code backend's ``executable``.

Runs ``claude`` with the given arguments and stdin, logs the ``total_cost_usd`` the CLI reports
to ``costs.jsonl`` (one line per real call), and refuses to start a call once the logged total,
plus the reserves held by calls still in flight, plus this call's own reserve would exceed
``SHADOWGATE_BUDGET_USD`` (default 88). A refused call returns a CLI-style error result, which
the backend records as an error.

Every admitted call holds a reserve of ``SHADOWGATE_RESERVE_USD`` (default 0.25) in a
reservations file until it finishes, so N parallel workers can overshoot the cap only by the
amount by which N calls exceed their reserves, not by N whole calls. A reservation left behind
by a crashed process expires after ``SHADOWGATE_RESERVATION_TTL_S`` seconds (default 3600).
The check-and-reserve step and every log write happen under an exclusive file lock
(``fcntl`` on POSIX, ``msvcrt`` on Windows).

This is the hard cap: it counts the CLI's own reported cost, including retries, not the
shadowgate price-table estimate that ``[run].max_cost_usd`` uses. ``--version`` (and ``-v``) is
passed straight through to the CLI without a budget check or a log entry.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

HERE = Path(__file__).resolve().parent
PASSTHROUGH = (["--version"], ["-v"])


class Config:
    """Paths and limits, read from the environment when the guard starts."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        env = dict(os.environ) if env is None else env
        self.log = Path(env.get("SHADOWGATE_COST_LOG") or HERE / "costs.jsonl")
        self.budget = float(env.get("SHADOWGATE_BUDGET_USD") or "88")
        self.reserve = float(env.get("SHADOWGATE_RESERVE_USD") or "0.25")
        self.ttl_s = float(env.get("SHADOWGATE_RESERVATION_TTL_S") or "3600")
        if self.reserve < 0 or self.ttl_s <= 0:
            raise ValueError(
                "SHADOWGATE_RESERVE_USD must be >= 0 and SHADOWGATE_RESERVATION_TTL_S > 0"
            )

    @property
    def lock_path(self) -> Path:
        return Path(str(self.log) + ".lock")

    @property
    def reservations_path(self) -> Path:
        return Path(str(self.log) + ".inflight.json")


# --------------------------------------------------------------------------- locking


def _lock_file(f: IO[Any]) -> None:
    if os.name == "nt":
        import msvcrt

        f.seek(0)
        while True:
            try:
                # LK_LOCK would sleep a whole second between attempts; poll faster instead.
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.01)
    else:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock_file(f: IO[Any]) -> None:
    if os.name == "nt":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold an exclusive, cross-process lock on ``path`` (created if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        _lock_file(f)
        try:
            yield
        finally:
            _unlock_file(f)


# --------------------------------------------------------------------------- ledger state


def spent(log: Path) -> float:
    """Total CLI-reported cost in the log. Unparseable (e.g. torn) lines are skipped."""
    if not log.exists():
        return 0.0
    total = 0.0
    with open(log, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                total += float(json.loads(line).get("total_cost_usd") or 0.0)
            except (ValueError, AttributeError, TypeError):
                continue
    return total


def _read_reservations(path: Path, now: float, ttl_s: float) -> dict[str, dict[str, float]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    live: dict[str, dict[str, float]] = {}
    for key, entry in data.items():
        try:
            amount, ts = float(entry["usd"]), float(entry["ts"])
        except (TypeError, KeyError, ValueError):
            continue
        if now - ts < ttl_s:
            live[str(key)] = {"usd": amount, "ts": ts}
    return live


def _write_reservations(path: Path, data: dict[str, dict[str, float]]) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def reserve(cfg: Config, now: float | None = None) -> tuple[str | None, float, float]:
    """Try to reserve budget for one call.

    Returns (reservation id or None if refused, logged spend, reserves held by others).
    """
    now = time.time() if now is None else now
    with locked(cfg.lock_path):
        total = spent(cfg.log)
        held = _read_reservations(cfg.reservations_path, now, cfg.ttl_s)
        others = sum(e["usd"] for e in held.values())
        if total + others + cfg.reserve > cfg.budget:
            return None, total, others
        rid = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        held[rid] = {"usd": cfg.reserve, "ts": now}
        _write_reservations(cfg.reservations_path, held)
    return rid, total, others


def settle(cfg: Config, rid: str, entry: dict[str, Any] | None) -> None:
    """Append ``entry`` to the cost log (if given) and release reservation ``rid``."""
    with locked(cfg.lock_path):
        if entry is not None:
            with open(cfg.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        held = _read_reservations(cfg.reservations_path, time.time(), cfg.ttl_s)
        held.pop(rid, None)
        _write_reservations(cfg.reservations_path, held)


# --------------------------------------------------------------------------- main


def _parse_cost(stdout: bytes) -> tuple[dict[str, Any] | None, float]:
    try:
        data = json.loads(stdout.decode("utf-8", errors="replace").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None, 0.0
    if not isinstance(data, dict):
        return None, 0.0
    try:
        cost = float(data.get("total_cost_usd") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return data, cost


def main(argv: list[str] | None = None, cfg: Config | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    claude = os.environ.get("SHADOWGATE_CLAUDE_BIN") or shutil.which("claude")
    if claude is None:
        print("claude not found on PATH", file=sys.stderr)
        return 127
    if args in PASSTHROUGH:
        return subprocess.run([claude, *args]).returncode
    cfg = cfg or Config()
    model = args[args.index("--model") + 1] if "--model" in args[:-1] else "?"
    rid, total, others = reserve(cfg)
    if rid is None:
        msg = (
            f"budget guard: ${total:.4f} spent, ${others:.2f} reserved in flight, "
            f"cap ${cfg.budget:.2f}; call refused"
        )
        print(json.dumps({"type": "result", "is_error": True, "result": msg}))
        print(msg, file=sys.stderr)
        return 1
    entry: dict[str, Any] | None = None
    try:
        stdin = sys.stdin.buffer.read()
        t0 = time.time()
        proc = subprocess.run([claude, *args], input=stdin, capture_output=True)
        data, cost = _parse_cost(proc.stdout)
        entry = {
            "ts": round(t0, 3),
            "model": model,
            "total_cost_usd": cost,
            "exit": proc.returncode,
            "parsed": data is not None,
            "seconds": round(time.time() - t0, 2),
        }
    finally:
        # On an exception before the call ran, the reservation is released with no log entry.
        settle(cfg, rid, entry)
    sys.stdout.buffer.write(proc.stdout)
    sys.stderr.buffer.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
