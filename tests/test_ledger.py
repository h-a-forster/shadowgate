from __future__ import annotations

import dataclasses
import json
import math
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from shadowgate.errors import LedgerError
from shadowgate.ledger import REDACTED, Ledger, RunInfo, redact_config
from shadowgate.types import (
    Attempt,
    Completion,
    ConfidenceResult,
    Decision,
    Judgement,
    ShadowResult,
    Task,
    Usage,
)


def _completion(text: str, model: str = "m-small", cost: float | None = 0.001) -> Completion:
    return Completion(
        text=text,
        model=model,
        usage=Usage(12, 34, 5, 6),
        cost_usd=cost,
        latency_s=0.25,
        stop_reason="end",
        logprobs=(-0.1, -0.02),
        cached=False,
    )


def make_decision(
    run_id: str = "r1",
    task_id: str = "t1",
    *,
    mode: str = "serve",
    shadow_status: str | None = "done",
    answer: str = "42",
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> Decision:
    task = Task(id=task_id, prompt=f"What is {task_id}? é中", reference="42",
                meta={"dataset": "toy", "difficulty": 3, "tags": ["a", "b"]})
    conf = ConfidenceResult(
        estimator="self_consistency",
        score=0.4,
        calls=(_completion("41"), _completion("42")),
        detail={"votes": {"42": 2, "41": 1}},
    )
    small = Attempt(
        tier="small",
        backend="sim:small",
        completion=_completion("41"),
        answer="41",
        confidence=conf,
        threshold=0.7,
        accepted=False,
        correct=Judgement(False, "exact"),
        agreement=Judgement(False, "normalized", detail={"a": "41", "b": "42"}),
    )
    large = Attempt(
        tier="large",
        backend="sim:large",
        completion=_completion("42", model="m-large", cost=None),
        answer=answer,
        confidence=None,
        threshold=None,
        accepted=True,
        correct=Judgement(True, "exact"),
    )
    failed = Attempt(
        tier="mid", backend="sim:mid", completion=None, answer="", confidence=None,
        threshold=0.5, accepted=False, error="BackendError: boom",
    )
    shadow = None
    if shadow_status is not None:
        shadow = ShadowResult(
            audit_tier="large",
            inclusion_prob=0.25,
            status=shadow_status,
            attempt=large if shadow_status == "done" else None,
            agreement=Judgement(
                None, "judge", calls=(_completion("UNSURE", model="judge"),),
                detail={"raw": "?"},
            ) if shadow_status == "done" else None,
        )
    return Decision(
        run_id=run_id,
        task=task,
        answer=answer,
        final_tier="large",
        escalated=True,
        attempts=(small, failed, large),
        cost_usd=None,
        audit_cost_usd=0.003,
        latency_s=1.5,
        mode=mode,
        shadow=shadow,
        correct=Judgement(True, "exact"),
        created_at=created_at,
        error=None,
    )


@pytest.fixture
def ledger(tmp_path: Path):
    lg = Ledger(tmp_path / "sub" / "ledger.sqlite")
    yield lg
    lg.close()


def test_round_trip_complex_decision(ledger: Ledger) -> None:
    d = make_decision()
    ledger.record(d)
    got = list(ledger.decisions("r1"))
    assert got == [d]
    assert ledger.has("r1", "t1")
    assert not ledger.has("r1", "nope")
    assert ledger.done_task_ids("r1") == {"t1"}


def test_upsert_keeps_position_and_replaces(ledger: Ledger) -> None:
    ledger.record(make_decision(task_id="a"))
    ledger.record(make_decision(task_id="b"))
    newer = make_decision(task_id="a", answer="43", shadow_status=None)
    ledger.record(newer)
    got = list(ledger.decisions("r1"))
    assert [d.task.id for d in got] == ["a", "b"]
    assert got[0] == newer
    assert ledger.runs()[0].n_decisions == 2


def test_latest_run_selection(ledger: Ledger) -> None:
    assert ledger.latest_run_id() is None
    assert list(ledger.decisions()) == []
    ledger.start_run("old", config={}, mode="serve")
    ledger.start_run("new", config={}, mode="eval")
    ledger.record(make_decision("old", "x"))
    ledger.record(make_decision("new", "y", mode="eval"))
    assert ledger.latest_run_id() == "new"
    assert [d.task.id for d in ledger.decisions()] == ["y"]
    infos = ledger.runs()
    assert [r.run_id for r in infos] == ["old", "new"]
    assert all(isinstance(r, RunInfo) for r in infos)


def test_latest_run_ties_broken_by_insertion(ledger: Ledger) -> None:
    # auto-created runs take created_at from the decision -> identical timestamps
    ledger.record(make_decision("first", "x"))
    ledger.record(make_decision("second", "x"))
    assert ledger.latest_run_id() == "second"


def test_pending_audits(ledger: Ledger) -> None:
    ledger.record(make_decision(task_id="p1", shadow_status="pending"))
    ledger.record(make_decision(task_id="d1", shadow_status="done"))
    ledger.record(make_decision(task_id="n1", shadow_status=None))
    ledger.record(make_decision(task_id="p2", shadow_status="pending"))
    assert [d.task.id for d in ledger.pending_audits()] == ["p1", "p2"]
    # completing the audit removes it from pending
    ledger.record(make_decision(task_id="p1", shadow_status="done"))
    assert [d.task.id for d in ledger.pending_audits("r1")] == ["p2"]


def test_concurrent_records(ledger: Ledger) -> None:
    ledger.start_run("conc", config={"workers": 8}, mode="serve")

    def work(worker: int) -> None:
        for i in range(200):
            ledger.record(make_decision("conc", f"w{worker}-{i}"))
            if i % 50 == 0:
                ledger.done_task_ids("conc")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(8)))
    assert len(ledger.done_task_ids("conc")) == 1600
    assert ledger.runs()[0].n_decisions == 1600


def test_reopen_persistence_and_second_reader(tmp_path: Path) -> None:
    path = tmp_path / "l.sqlite"
    d = make_decision()
    with Ledger(path) as lg:
        lg.start_run("r1", config={"a": 1}, mode="serve", note="hello")
        lg.record(d)
        # a second connection (as another process would) can read while the first is open
        with Ledger(path) as reader:
            assert list(reader.decisions()) == [d]
    with Ledger(path) as lg:
        assert list(lg.decisions("r1")) == [d]
        (info,) = lg.runs()
        assert info.note == "hello" and info.config == {"a": 1} and info.mode == "serve"
        assert info.n_decisions == 1


def test_close_idempotent_and_use_after_close(tmp_path: Path) -> None:
    lg = Ledger(tmp_path / "l.sqlite")
    lg.close()
    lg.close()
    with pytest.raises(LedgerError):
        lg.record(make_decision())


def test_newer_schema_version_rejected(tmp_path: Path) -> None:
    path = tmp_path / "l.sqlite"
    Ledger(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value = ? WHERE key = 'shadowgate_schema_version'",
                 (str(Ledger.SCHEMA_VERSION + 1),))
    conn.commit()
    conn.close()
    with pytest.raises(LedgerError, match="newer"):
        Ledger(path)


def test_non_sqlite_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("this is not a database\n" * 100, encoding="utf-8")
    with pytest.raises(LedgerError, match="not a shadowgate ledger"):
        Ledger(path)


def test_foreign_sqlite_db_rejected(tmp_path: Path) -> None:
    path = tmp_path / "other.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(LedgerError, match="not a shadowgate ledger"):
        Ledger(path)


def test_empty_file_becomes_ledger(tmp_path: Path) -> None:
    path = tmp_path / "empty.sqlite"
    path.write_bytes(b"")
    with Ledger(path) as lg:
        assert lg.runs() == []


def test_export_import_round_trip(tmp_path: Path) -> None:
    ds = [make_decision("r1", f"t{i}", shadow_status=("pending" if i % 2 else "done"))
          for i in range(5)]
    out = tmp_path / "out" / "r1.jsonl"
    with Ledger(tmp_path / "a.sqlite") as a:
        for d in ds:
            a.record(d)
        assert a.export_jsonl(out) == 5
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5 and json.loads(lines[0])["task"]["id"] == "t0"
    with Ledger(tmp_path / "b.sqlite") as b:
        assert b.import_jsonl(out) == 5
        assert list(b.decisions("r1")) == ds
        assert [r.run_id for r in b.runs()] == ["r1"]
        assert len(list(b.pending_audits())) == 2
        # import is an upsert
        assert b.import_jsonl(out) == 5
        assert b.runs()[0].n_decisions == 5


def test_import_bad_line_names_line(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(make_decision().to_dict()) + "\n{not json\n", encoding="utf-8")
    with Ledger(tmp_path / "l.sqlite") as lg:
        with pytest.raises(LedgerError, match=":2:"):
            lg.import_jsonl(p)
        assert lg.runs() == []  # nothing written


def test_redaction(ledger: Ledger) -> None:
    config = {
        "backends": {
            "a": {"api_key": "sk-123", "api_key_env": "ANTHROPIC_API_KEY", "max_tokens": 10},
            "b": {"apiKey": "x", "nested": [{"password": "p", "access_token": "t"}]},
        },
        "Secret": "s",
        "token": None,
        "run": {"name": "x"},
    }
    ledger.start_run("r", config=config, mode="serve")
    stored = ledger.runs()[0].config
    assert stored["backends"]["a"] == {
        "api_key": REDACTED, "api_key_env": "ANTHROPIC_API_KEY", "max_tokens": 10,
    }
    assert stored["backends"]["b"]["apiKey"] == REDACTED
    assert stored["backends"]["b"]["nested"] == [{"password": REDACTED, "access_token": REDACTED}]
    assert stored["Secret"] == REDACTED
    assert stored["token"] is None
    assert stored["run"] == {"name": "x"}
    assert redact_config({"private_key": "k"}) == {"private_key": REDACTED}
    assert "sk-123" not in json.dumps(stored)


def test_start_run_idempotent_and_mode_conflict(ledger: Ledger) -> None:
    ledger.start_run("r", config={"v": 1}, mode="serve")
    created = ledger.runs()[0].created_at
    ledger.start_run("r", config={"v": 2}, mode="serve", note="resumed")
    (info,) = ledger.runs()
    assert info.created_at == created and info.config == {"v": 2} and info.note == "resumed"
    with pytest.raises(LedgerError, match="mode"):
        ledger.start_run("r", config={}, mode="eval")
    with pytest.raises(LedgerError, match="eval"):
        ledger.record(make_decision("r", "t", mode="eval"))
    assert not ledger.has("r", "t")


def test_record_auto_creates_run(ledger: Ledger) -> None:
    ledger.record(make_decision("auto", "t", mode="eval"))
    (info,) = ledger.runs()
    assert info.run_id == "auto" and info.mode == "eval" and info.config == {}
    assert info.n_decisions == 1


def test_corrupt_json_row_names_run_and_task(tmp_path: Path) -> None:
    path = tmp_path / "l.sqlite"
    with Ledger(path) as lg:
        lg.record(make_decision("rx", "tx"))
    conn = sqlite3.connect(path)
    conn.execute("UPDATE decisions SET decision_json = '{broken'")
    conn.commit()
    conn.close()
    with Ledger(path) as lg, pytest.raises(LedgerError, match="'rx'.*'tx'"):
        list(lg.decisions("rx"))


def test_non_finite_floats_are_stored_as_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "l.sqlite"
    base = make_decision()
    d = dataclasses.replace(base, cost_usd=math.inf, latency_s=math.nan)
    with Ledger(path) as lg:
        lg.record(d)
        (got,) = list(lg.decisions())
    assert got.cost_usd == math.inf and math.isnan(got.latency_s)
    conn = sqlite3.connect(path)
    raw, cost = conn.execute("SELECT decision_json, cost_usd FROM decisions").fetchone()
    conn.close()
    assert cost is None
    json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))


def test_threaded_readers_and_writer_on_separate_connections(tmp_path: Path) -> None:
    path = tmp_path / "l.sqlite"
    writer = Ledger(path)
    reader = Ledger(path)
    stop = threading.Event()
    errors: list[BaseException] = []

    def read_loop() -> None:
        try:
            while not stop.is_set():
                list(reader.decisions("r1"))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    t = threading.Thread(target=read_loop)
    t.start()
    try:
        for i in range(100):
            writer.record(make_decision("r1", f"t{i}"))
    finally:
        stop.set()
        t.join()
        reader.close()
        writer.close()
    assert not errors


# --------------------------------------------------------------------------- errors & migration


def test_done_task_ids_excludes_errors_by_default(ledger: Ledger) -> None:
    ledger.record(make_decision(task_id="ok"))
    ledger.record(dataclasses.replace(make_decision(task_id="bad"), error="BackendError: 401"))
    assert ledger.done_task_ids("r1") == {"ok"}
    assert ledger.done_task_ids("r1", include_errors=True) == {"ok", "bad"}
    # A successful re-route replaces the failed record and clears the error.
    ledger.record(make_decision(task_id="bad"))
    assert ledger.done_task_ids("r1") == {"ok", "bad"}


_V1_DDL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, mode TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}', note TEXT NOT NULL DEFAULT '');
CREATE TABLE decisions (run_id TEXT NOT NULL, task_id TEXT NOT NULL, created_at TEXT NOT NULL,
    mode TEXT NOT NULL, final_tier TEXT NOT NULL, escalated INTEGER NOT NULL, cost_usd REAL,
    audit_status TEXT, decision_json TEXT NOT NULL, PRIMARY KEY (run_id, task_id));
CREATE INDEX idx_decisions_audit ON decisions (run_id, audit_status);
CREATE INDEX idx_runs_created ON runs (created_at);
"""


def test_migrates_v1_ledger_in_place(tmp_path: Path) -> None:
    path = tmp_path / "v1.sqlite"
    ok = make_decision(task_id="ok")
    bad = dataclasses.replace(make_decision(task_id="bad"), error="BackendError: 401")
    conn = sqlite3.connect(path)
    conn.executescript(_V1_DDL)
    conn.execute("INSERT INTO meta VALUES ('shadowgate_schema_version', '1')")
    conn.execute("INSERT INTO runs VALUES ('r1', '2026-01-01', 'serve', '{}', '')")
    for d in (ok, bad):
        conn.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", d.task.id, d.created_at, "serve", d.final_tier, 1, None, "done",
             json.dumps(d.to_dict())),
        )
    conn.commit()
    conn.close()

    with Ledger(path) as lg:
        assert lg.done_task_ids("r1") == {"ok"}
        assert lg.done_task_ids("r1", include_errors=True) == {"ok", "bad"}
        assert [d.task.id for d in lg.decisions("r1")] == ["ok", "bad"]
        lg.record(make_decision(task_id="bad"))  # upsert writes the new column
        assert lg.done_task_ids("r1") == {"ok", "bad"}
    conn = sqlite3.connect(path)
    version = conn.execute(
        "SELECT value FROM meta WHERE key = 'shadowgate_schema_version'"
    ).fetchone()[0]
    cols = [r[1] for r in conn.execute("PRAGMA table_info(decisions)")]
    conn.close()
    assert version == str(Ledger.SCHEMA_VERSION) == "2"
    assert "error" in cols
    Ledger(path).close()  # reopening a migrated file is a no-op
