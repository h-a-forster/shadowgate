"""SQLite decision ledger.

Every routed :class:`~shadowgate.types.Decision` is written here as soon as it finishes, so a run
can be resumed after a crash and audited, swept or reported later (possibly from another process
while the run is still going).

Storage notes:

* One SQLite file, WAL journal, ``synchronous=NORMAL`` and a generous busy timeout. Each
  :meth:`Ledger.record` call is its own transaction, so a crash loses at most the decision that
  was being written. Other processes can open the same file and read concurrently.
* A single connection is shared by all threads of one :class:`Ledger` and guarded by a lock.
* ``decision_json`` holds ``Decision.to_dict()`` serialised as strict JSON. Non-finite floats
  (NaN, +/-inf) are not valid JSON; they are stored as the strings ``"NaN"``, ``"Infinity"`` and
  ``"-Infinity"``, which ``float()`` parses back, so numeric fields round-trip through
  ``Decision.from_dict``. The indexed ``cost_usd`` column stores NULL for a non-finite cost.
* Run configs are stored with secret-looking keys (``api_key``, ``token``, ``secret``,
  ``password`` ...) redacted recursively, as defence in depth: secrets are supposed to come from
  environment variables and never appear in config values at all. Keys ending in ``_env`` name
  an environment variable rather than hold a secret and are kept.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from .errors import LedgerError
from .types import Decision

__all__ = ["Ledger", "RunInfo", "redact_config"]

log = logging.getLogger("shadowgate.ledger")

_META_VERSION_KEY = "shadowgate_schema_version"
_BUSY_TIMEOUT_S = 30.0
REDACTED = "***REDACTED***"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    mode        TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    note        TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS decisions (
    run_id        TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    mode          TEXT NOT NULL,
    final_tier    TEXT NOT NULL,
    escalated     INTEGER NOT NULL,
    cost_usd      REAL,
    audit_status  TEXT,
    error         TEXT,
    decision_json TEXT NOT NULL,
    PRIMARY KEY (run_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_decisions_audit ON decisions (run_id, audit_status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_at);
"""


@dataclass(frozen=True)
class RunInfo:
    """Summary of one run stored in the ledger (config is the redacted snapshot)."""

    run_id: str
    created_at: str
    mode: str
    n_decisions: int
    note: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)


_RUN_HEADER = "shadowgate_run"


# --------------------------------------------------------------------------- helpers


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


_SECRET_WORDS = frozenset(
    {"token", "secret", "secrets", "password", "passwd", "passphrase", "apikey", "credential",
     "credentials"}
)
_SECRET_PAIRS = frozenset({("api", "key"), ("private", "key"), ("access", "key"),
                           ("secret", "key"), ("auth", "key")})


def _is_secret_key(key: str) -> bool:
    # split camelCase, then on separators: "apiKey" -> "api_key", "max_tokens" -> [max, tokens]
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower()
    words = [w for w in re.split(r"[^a-z0-9]+", snake) if w]
    if not words or words[-1] == "env":
        return False
    if any(w in _SECRET_WORDS for w in words):
        return True
    return any(pair in _SECRET_PAIRS for pair in zip(words, words[1:], strict=False))


def redact_config(obj: Any) -> Any:
    """Return a copy of ``obj`` with values under secret-looking keys replaced, recursively."""
    if isinstance(obj, Mapping):
        return {
            str(k): (REDACTED if _is_secret_key(str(k)) and v is not None else redact_config(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact_config(v) for v in obj]
    return obj


def _finite_json(obj: Any) -> Any:
    """Replace non-finite floats with float()-parseable strings so strict JSON can hold them."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_json(v) for v in obj]
    return obj


def _dumps(obj: Any) -> str:
    return json.dumps(_finite_json(obj), ensure_ascii=False, allow_nan=False, default=str)


def _finite_or_none(x: float | None) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return float(x)


# --------------------------------------------------------------------------- ledger


class Ledger:
    """Thread-safe SQLite store of routing decisions, keyed by ``(run_id, task_id)``."""

    #: Version history: 1 = initial; 2 = ``decisions.error`` column (migrated in place from 1,
    #: backfilled from ``decision_json``).
    SCHEMA_VERSION = 2

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        self._lock = threading.RLock()
        self._closed = False
        target = str(path)
        if target != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(
                target,
                timeout=_BUSY_TIMEOUT_S,
                isolation_level=None,  # autocommit; transactions are explicit
                check_same_thread=False,
            )
        except sqlite3.Error as exc:
            raise LedgerError(f"cannot open ledger {target}: {exc}") from exc
        try:
            self._init_schema()
        except BaseException:
            self._conn.close()
            self._closed = True
            raise

    # ------------------------------------------------------------------ setup

    def _init_schema(self) -> None:
        conn = self._conn
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_S * 1000)}")
            # Reading sqlite_master is the first real read: a non-database file fails here.
            tables = {
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        except sqlite3.DatabaseError as exc:
            raise LedgerError(f"{self.path} is not a shadowgate ledger (not SQLite): {exc}") \
                from exc

        stored_version: int | None = None
        if tables:
            version = None
            if "meta" in tables:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key = ?", (_META_VERSION_KEY,)
                ).fetchone()
                version = None if row is None else row[0]
            if version is None:
                raise LedgerError(
                    f"{self.path} is an SQLite database but not a shadowgate ledger "
                    f"(no {_META_VERSION_KEY!r} in meta; tables: {sorted(tables)})"
                )
            try:
                v = int(version)
            except ValueError as exc:
                raise LedgerError(f"{self.path}: invalid schema version {version!r}") from exc
            if v > self.SCHEMA_VERSION:
                raise LedgerError(
                    f"{self.path} uses ledger schema version {v}, newer than this shadowgate "
                    f"supports ({self.SCHEMA_VERSION}); upgrade shadowgate to read it"
                )
            stored_version = v

        try:
            mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()
            if mode and str(mode[0]).lower() not in ("wal", "memory"):
                log.warning("ledger %s: WAL unavailable, journal_mode=%s", self.path, mode[0])
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("BEGIN IMMEDIATE")
            for stmt in _SCHEMA.split(";"):
                if stmt.strip():
                    conn.execute(stmt)
            if stored_version is not None and stored_version < self.SCHEMA_VERSION:
                self._migrate(stored_version)
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (_META_VERSION_KEY, str(self.SCHEMA_VERSION)),
            )
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise LedgerError(f"cannot initialise ledger {self.path}: {exc}") from exc

    def _migrate(self, from_version: int) -> None:
        """Upgrade an older schema in place. Caller holds the write transaction."""
        conn = self._conn
        if from_version < 2:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
            if "error" not in cols:
                conn.execute("ALTER TABLE decisions ADD COLUMN error TEXT")
            rows = conn.execute("SELECT rowid, decision_json FROM decisions").fetchall()
            for rowid, raw in rows:
                try:
                    err = json.loads(raw).get("error")
                except (ValueError, AttributeError):
                    continue  # corrupt rows surface on read, not during migration
                if err is not None:
                    conn.execute(
                        "UPDATE decisions SET error = ? WHERE rowid = ?", (str(err), rowid)
                    )
            log.info("ledger %s: migrated schema %d -> 2", self.path, from_version)

    # ------------------------------------------------------------------ plumbing

    def _check_open(self) -> None:
        if self._closed:
            raise LedgerError(f"ledger {self.path} is closed")

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            self._check_open()
            try:
                return self._conn.execute(sql, params).fetchall()
            except sqlite3.Error as exc:
                raise LedgerError(f"ledger {self.path}: query failed: {exc}") from exc

    def _ensure_run(self, run_id: str, mode: str, created_at: str) -> None:
        """Insert a minimal runs row if missing; raise on a mode conflict. Caller holds a txn."""
        row = self._conn.execute("SELECT mode FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO runs (run_id, created_at, mode, config_json, note) "
                "VALUES (?, ?, ?, '{}', '')",
                (run_id, created_at or _utc_now(), mode),
            )
        elif row[0] != mode:
            raise LedgerError(
                f"run {run_id!r} is a {row[0]!r} run; refusing to record a {mode!r} decision "
                "into it (use a different run_id)"
            )

    def _upsert(self, d: Decision) -> None:
        shadow_status = d.shadow.status if d.shadow is not None else None
        self._conn.execute(
            "INSERT INTO decisions (run_id, task_id, created_at, mode, final_tier, escalated, "
            "cost_usd, audit_status, error, decision_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (run_id, task_id) DO UPDATE SET created_at = excluded.created_at, "
            "mode = excluded.mode, final_tier = excluded.final_tier, "
            "escalated = excluded.escalated, cost_usd = excluded.cost_usd, "
            "audit_status = excluded.audit_status, error = excluded.error, "
            "decision_json = excluded.decision_json",
            (
                d.run_id,
                d.task.id,
                d.created_at or _utc_now(),
                d.mode,
                d.final_tier,
                int(bool(d.escalated)),
                _finite_or_none(d.cost_usd),
                shadow_status,
                d.error,
                _dumps(d.to_dict()),
            ),
        )

    def _write_decisions(self, decisions: list[Decision]) -> None:
        with self._lock:
            self._check_open()
            conn = self._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                for d in decisions:
                    self._ensure_run(d.run_id, d.mode, d.created_at)
                    self._upsert(d)
                conn.execute("COMMIT")
            except BaseException as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                if isinstance(exc, sqlite3.Error):
                    raise LedgerError(f"ledger {self.path}: write failed: {exc}") from exc
                raise

    @staticmethod
    def _decode(run_id: str, task_id: str, raw: str) -> Decision:
        try:
            return Decision.from_dict(json.loads(raw))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise LedgerError(
                f"corrupt decision record for run {run_id!r}, task {task_id!r}: {exc}"
            ) from exc

    def _resolve_run(self, run_id: str | None) -> str | None:
        return self.latest_run_id() if run_id is None else run_id

    # ------------------------------------------------------------------ runs

    def start_run(
        self, run_id: str, *, config: Mapping[str, Any], mode: str, note: str = ""
    ) -> None:
        """Register a run. Idempotent; re-starting with a different ``mode`` raises LedgerError.

        Repeating the call (e.g. when resuming) keeps the original ``created_at`` and refreshes
        the stored config snapshot and note.
        """
        config_json = _dumps(redact_config(dict(config)))
        with self._lock:
            self._check_open()
            conn = self._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT mode FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO runs (run_id, created_at, mode, config_json, note) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (run_id, _utc_now(), mode, config_json, note),
                    )
                elif row[0] != mode:
                    raise LedgerError(
                        f"run {run_id!r} already exists with mode {row[0]!r}; cannot start it "
                        f"again in mode {mode!r} (use a different run_id)"
                    )
                else:
                    conn.execute(
                        "UPDATE runs SET config_json = ?, note = ? WHERE run_id = ?",
                        (config_json, note, run_id),
                    )
                conn.execute("COMMIT")
            except BaseException as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                if isinstance(exc, sqlite3.Error):
                    raise LedgerError(f"ledger {self.path}: start_run failed: {exc}") from exc
                raise

    def runs(self) -> list[RunInfo]:
        """All runs, oldest first."""
        rows = self._query(
            "SELECT r.run_id, r.created_at, r.mode, r.note, r.config_json, "
            "(SELECT COUNT(*) FROM decisions d WHERE d.run_id = r.run_id) "
            "FROM runs r ORDER BY r.created_at, r.rowid"
        )
        out: list[RunInfo] = []
        for run_id, created_at, mode, note, config_json, n in rows:
            try:
                config = json.loads(config_json or "{}")
            except ValueError as exc:
                raise LedgerError(f"corrupt config for run {run_id!r}: {exc}") from exc
            out.append(RunInfo(run_id, created_at, mode, int(n), note or "", config))
        return out

    def latest_run_id(self) -> str | None:
        rows = self._query(
            "SELECT run_id FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
        )
        return rows[0][0] if rows else None

    # ------------------------------------------------------------------ decisions

    def record(self, decision: Decision) -> None:
        """Upsert one decision (committed immediately). Creates a minimal run row if needed."""
        self._write_decisions([decision])

    def has(self, run_id: str, task_id: str) -> bool:
        return bool(
            self._query(
                "SELECT 1 FROM decisions WHERE run_id = ? AND task_id = ?", (run_id, task_id)
            )
        )

    def done_task_ids(self, run_id: str, *, include_errors: bool = False) -> set[str]:
        """Task ids recorded for ``run_id``; failed decisions (``error`` set) only on request."""
        sql = "SELECT task_id FROM decisions WHERE run_id = ?"
        if not include_errors:
            sql += " AND error IS NULL"
        return {r[0] for r in self._query(sql, (run_id,))}

    def decisions(self, run_id: str | None = None) -> Iterator[Decision]:
        """Decisions of ``run_id`` (default: the latest run) in insertion order."""
        rid = self._resolve_run(run_id)
        if rid is None:
            return iter(())
        rows = self._query(
            "SELECT task_id, decision_json FROM decisions WHERE run_id = ? ORDER BY rowid",
            (rid,),
        )
        return (self._decode(rid, t, raw) for t, raw in rows)

    def pending_audits(self, run_id: str | None = None) -> Iterator[Decision]:
        """Decisions whose shadow audit is deferred (``shadow.status == "pending"``)."""
        rid = self._resolve_run(run_id)
        if rid is None:
            return iter(())
        rows = self._query(
            "SELECT task_id, decision_json FROM decisions "
            "WHERE run_id = ? AND audit_status = 'pending' ORDER BY rowid",
            (rid,),
        )
        return (self._decode(rid, t, raw) for t, raw in rows)

    # ------------------------------------------------------------------ JSONL

    def export_jsonl(self, path: str | Path, run_id: str | None = None) -> int:
        """Write the run's metadata, then one ``Decision.to_dict()`` per line; returns the count.

        The first line is ``{"shadowgate_run": {...}}`` with the run's mode, note and redacted
        config snapshot, so an import restores settings such as ``audit.tolerance``.
        """
        rid = self._resolve_run(run_id)
        info = next((r for r in self.runs() if r.run_id == rid), None)
        p = Path(path)
        if p.parent != Path():
            p.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            if info is not None:
                header = {
                    "run_id": info.run_id,
                    "created_at": info.created_at,
                    "mode": info.mode,
                    "note": info.note,
                    "config": dict(info.config),
                }
                fh.write(_dumps({_RUN_HEADER: header}))
                fh.write("\n")
            for d in self.decisions(rid):
                fh.write(_dumps(d.to_dict()))
                fh.write("\n")
                n += 1
        return n

    def import_jsonl(self, path: str | Path) -> int:
        """Upsert decisions from a JSONL file written by ``export_jsonl``; returns the count.

        Run metadata lines (``{"shadowgate_run": ...}``) create the run with its config, or fill
        in the config of an existing run that has none. Files without them still import; the
        runs then have an empty config. The whole file is validated before anything is written,
        then written in one transaction.
        """
        items: list[Decision] = []
        headers: list[dict[str, Any]] = []
        with Path(path).open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and _RUN_HEADER in obj:
                        h = obj[_RUN_HEADER]
                        if (
                            not isinstance(h, dict)
                            or not isinstance(h.get("run_id"), str)
                            or not isinstance(h.get("mode"), str)
                        ):
                            raise ValueError("run header needs string run_id and mode")
                        headers.append(h)
                    else:
                        items.append(Decision.from_dict(obj))
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise LedgerError(
                        f"{path}:{lineno}: invalid decision record: {exc}"
                    ) from exc
        for h in headers:
            self._import_run_header(h)
        self._write_decisions(items)
        return len(items)

    def _import_run_header(self, h: Mapping[str, Any]) -> None:
        config_json = _dumps(redact_config(dict(h.get("config") or {})))
        with self._lock:
            self._check_open()
            conn = self._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT mode, config_json FROM runs WHERE run_id = ?", (h["run_id"],)
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO runs (run_id, created_at, mode, config_json, note) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (h["run_id"], str(h.get("created_at") or _utc_now()), h["mode"],
                         config_json, str(h.get("note") or "")),
                    )
                elif row[0] != h["mode"]:
                    raise LedgerError(
                        f"run {h['run_id']!r} already exists with mode {row[0]!r}; the import "
                        f"has mode {h['mode']!r}"
                    )
                elif row[1] in ("", "{}"):
                    conn.execute(
                        "UPDATE runs SET config_json = ? WHERE run_id = ?",
                        (config_json, h["run_id"]),
                    )
                conn.execute("COMMIT")
            except BaseException as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                if isinstance(exc, sqlite3.Error):
                    raise LedgerError(f"ledger {self.path}: write failed: {exc}") from exc
                raise

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Ledger({str(self.path)!r})"
