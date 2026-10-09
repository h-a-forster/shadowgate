"""SQLite response cache: CacheStore and the CachedBackend wrapper."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from ..types import Backend, Completion, Request
from .base import request_key

__all__ = ["CacheStore", "CachedBackend"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS completions (
    key TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    completion_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class CacheStore:
    """Thread-safe SQLite key/value store for completions (WAL mode).

    One connection shared across threads, serialised by a lock. ``path`` may be ``":memory:"``.
    Parent directories are created. Stored completions are returned with ``cached=True``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def get(self, key: str) -> Completion | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT completion_json FROM completions WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["cached"] = True
        return Completion.from_dict(data)

    def put(self, key: str, backend: str, completion: Completion) -> None:
        data = completion.to_dict()
        data["cached"] = False
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO completions (key, backend, completion_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, backend, blob, now),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> CacheStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CachedBackend:
    """Wrap a backend with a response cache keyed by ``request_key(inner.name, request)``.

    Hits return the stored completion with ``cached=True`` (cost and latency are the original
    call's). Exceptions propagate and completions with ``stop_reason == "error"`` are not
    stored. ``name`` is the inner backend's name, so ledgers and keys are unaffected.
    """

    def __init__(self, inner: Backend, store: CacheStore) -> None:
        self.inner = inner
        self.store = store
        self.name = inner.name

    def complete(self, request: Request) -> Completion:
        key = request_key(self.inner.name, request)
        hit = self.store.get(key)
        if hit is not None:
            return hit
        completion = self.inner.complete(request)
        if completion.stop_reason != "error":
            self.store.put(key, self.inner.name, completion)
        return completion
