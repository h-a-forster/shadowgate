from __future__ import annotations

import dataclasses
import logging
import threading
import time
from pathlib import Path

import pytest

from shadowgate.errors import DatasetError, LedgerError
from shadowgate.ledger import Ledger
from shadowgate.runner import RunStats, run, run_pending_audits
from shadowgate.types import Attempt, Decision, ShadowResult, Task


def tasks(n: int, prefix: str = "t") -> list[Task]:
    return [Task(id=f"{prefix}{i}", prompt=f"q{i}", reference=str(i)) for i in range(n)]


def make_decision(
    task: Task,
    run_id: str,
    mode: str = "serve",
    *,
    cost: float | None = 0.01,
    audit_cost: float | None = 0.0,
    escalated: bool = False,
    shadow: ShadowResult | None = None,
    error: str | None = None,
) -> Decision:
    att = Attempt(
        tier="small",
        backend="fake:small",
        completion=None,
        answer=task.reference or "",
        confidence=None,
        threshold=0.5,
        accepted=True,
    )
    return Decision(
        run_id=run_id,
        task=task,
        answer=task.reference or "",
        final_tier="small",
        escalated=escalated,
        attempts=(att,),
        cost_usd=cost,
        audit_cost_usd=audit_cost,
        latency_s=0.0,
        mode=mode,
        shadow=shadow,
        created_at="2026-01-01T00:00:00+00:00",
        error=error,
    )


class FakeCascade:
    """Configurable fake with route/complete_audit; tracks concurrency."""

    def __init__(
        self,
        *,
        delay: float = 0.0,
        cost: float | None = 0.01,
        audit_cost: float | None = 0.0,
        fail_ids: set[str] | None = None,
        raise_ids: set[str] | None = None,
        always_error: bool = False,
        escalate_ids: set[str] | None = None,
        shadow_status: str | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.delay = delay
        self.cost = cost
        self.audit_cost = audit_cost
        self.fail_ids = fail_ids or set()
        self.raise_ids = raise_ids or set()
        self.always_error = always_error
        self.escalate_ids = escalate_ids or set()
        self.shadow_status = shadow_status
        self.gate = gate
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[str] = []
        self.audit_calls: list[str] = []

    def _enter(self, tid: str) -> None:
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.calls.append(tid)

    def _exit(self) -> None:
        with self.lock:
            self.in_flight -= 1

    def route(self, task: Task, *, run_id: str = "", mode: str = "serve") -> Decision:
        self._enter(task.id)
        try:
            if self.gate is not None:
                assert self.gate.wait(5)
            if self.delay:
                time.sleep(self.delay)
            if task.id in self.raise_ids:
                raise RuntimeError(f"boom {task.id}")
            error = (
                "final tier 'big' failed: BackendError: 401"
                if (self.always_error or task.id in self.fail_ids)
                else None
            )
            shadow = (
                None if self.shadow_status is None else ShadowResult("big", 0.5, self.shadow_status)
            )
            return make_decision(
                task,
                run_id,
                mode,
                cost=self.cost,
                audit_cost=self.audit_cost,
                escalated=task.id in self.escalate_ids,
                shadow=shadow,
                error=error,
            )
        finally:
            self._exit()

    def complete_audit(self, decision: Decision) -> Decision:
        self._enter(decision.task.id)
        try:
            with self.lock:
                self.audit_calls.append(decision.task.id)
            if self.delay:
                time.sleep(self.delay)
            if decision.task.id in self.raise_ids:
                raise RuntimeError("audit boom")
            status = "error" if decision.task.id in self.fail_ids else "done"
            assert decision.shadow is not None
            new_shadow = dataclasses.replace(decision.shadow, status=status)
            new_cost = (
                None
                if self.audit_cost is None or decision.audit_cost_usd is None
                else decision.audit_cost_usd + self.audit_cost
            )
            return dataclasses.replace(decision, shadow=new_shadow, audit_cost_usd=new_cost)
        finally:
            self._exit()


@pytest.fixture
def ledger(tmp_path: Path):
    with Ledger(tmp_path / "ledger.sqlite") as lg:
        yield lg


# --------------------------------------------------------------------------- run: basics


def test_all_tasks_recorded(ledger: Ledger) -> None:
    c = FakeCascade()
    stats = run(c, tasks(25), ledger, run_id="r1", workers=4)
    assert stats.completed == 25 and stats.submitted == 25 and stats.total == 25
    assert stats.stopped is None and stats.failed == 0
    assert ledger.done_task_ids("r1") == {f"t{i}" for i in range(25)}
    assert stats.cost_serving == pytest.approx(0.25)
    assert stats.elapsed_s > 0


def test_start_run_registers_config_and_mode(ledger: Ledger) -> None:
    run(
        FakeCascade(),
        tasks(2),
        ledger,
        run_id="r1",
        mode="eval",
        config_snapshot={"tiers": ["a", "b"]},
    )
    info = {r.run_id: r for r in ledger.runs()}["r1"]
    assert info.mode == "eval" and info.config == {"tiers": ["a", "b"]}
    assert all(d.mode == "eval" for d in ledger.decisions("r1"))


def test_mode_conflict_raises_ledger_error(ledger: Ledger) -> None:
    run(FakeCascade(), tasks(1), ledger, run_id="r1", mode="serve")
    with pytest.raises(LedgerError):
        run(FakeCascade(), tasks(1), ledger, run_id="r1", mode="eval")


def test_concurrency_bound_respected(ledger: Ledger) -> None:
    c = FakeCascade(delay=0.02)
    stats = run(c, tasks(30), ledger, run_id="r1", workers=3)
    assert stats.completed == 30
    assert c.max_in_flight <= 3
    assert c.max_in_flight >= 2  # actually concurrent


def test_single_worker_is_sequential(ledger: Ledger) -> None:
    c = FakeCascade(delay=0.005)
    run(c, tasks(10), ledger, run_id="r1", workers=1)
    assert c.max_in_flight == 1
    assert c.calls == [f"t{i}" for i in range(10)]


def test_bounded_submission_not_all_at_once(ledger: Ledger) -> None:
    gate = threading.Event()
    c = FakeCascade(gate=gate)
    seen: list[int] = []

    def progress(stats: RunStats, d: Decision) -> None:
        seen.append(stats.submitted)

    t = threading.Timer(0.3, gate.set)
    t.start()
    stats = run(c, tasks(20), ledger, run_id="r1", workers=2, progress=progress)
    t.join()
    assert stats.completed == 20
    # When the first decision is recorded, at most workers (+1 refill) were ever submitted.
    assert seen[0] <= 3


def test_invalid_workers(ledger: Ledger) -> None:
    with pytest.raises(ValueError):
        run(FakeCascade(), tasks(1), ledger, run_id="r1", workers=0)
    assert ledger.runs() == []


def test_duplicate_ids_raise_before_start(ledger: Ledger) -> None:
    ts = tasks(3) + [Task(id="t1", prompt="again")]
    c = FakeCascade()
    with pytest.raises(DatasetError, match="t1"):
        run(c, ts, ledger, run_id="r1")
    assert c.calls == []
    assert ledger.runs() == []


def test_empty_tasks(ledger: Ledger) -> None:
    stats = run(FakeCascade(), [], ledger, run_id="r1")
    assert stats.completed == 0 and stats.stopped is None
    assert [r.run_id for r in ledger.runs()] == ["r1"]


# --------------------------------------------------------------------------- resume


def test_resume_skips_existing(ledger: Ledger) -> None:
    run(FakeCascade(), tasks(5), ledger, run_id="r1")
    c = FakeCascade()
    stats = run(c, tasks(8), ledger, run_id="r1", resume=True)
    assert stats.skipped_existing == 5
    assert stats.completed == 3
    assert sorted(c.calls) == ["t5", "t6", "t7"]
    assert len(ledger.done_task_ids("r1")) == 8
    assert stats.summary_line().startswith("8/8 done")


def test_no_resume_reruns_and_upserts(ledger: Ledger) -> None:
    run(FakeCascade(cost=0.01), tasks(4), ledger, run_id="r1")
    c = FakeCascade(cost=0.02)
    stats = run(c, tasks(4), ledger, run_id="r1", resume=False)
    assert stats.skipped_existing == 0 and stats.completed == 4
    ds = list(ledger.decisions("r1"))
    assert len(ds) == 4
    assert all(d.cost_usd == pytest.approx(0.02) for d in ds)


def test_resume_is_per_run(ledger: Ledger) -> None:
    run(FakeCascade(), tasks(3), ledger, run_id="r1")
    stats = run(FakeCascade(), tasks(3), ledger, run_id="r2")
    assert stats.skipped_existing == 0 and stats.completed == 3


# --------------------------------------------------------------------------- budget


def test_budget_stop_with_in_flight_completion(ledger: Ledger) -> None:
    c = FakeCascade(cost=0.10, delay=0.02)
    stats = run(c, tasks(50), ledger, run_id="r1", workers=4, max_cost_usd=0.25)
    assert stats.stopped == "budget" and stats.budget_exceeded
    # Every submitted task finished and was recorded, even past the cap.
    assert stats.completed == stats.submitted
    assert len(ledger.done_task_ids("r1")) == stats.completed
    assert 3 <= stats.completed < 50
    assert stats.cost_total >= 0.25
    # Overshoot is bounded by what was in flight when the cap was reached.
    assert stats.completed <= 3 + 4


def test_budget_includes_audit_cost(ledger: Ledger) -> None:
    c = FakeCascade(cost=0.0, audit_cost=0.1)
    stats = run(c, tasks(10), ledger, run_id="r1", workers=1, max_cost_usd=0.3)
    assert stats.stopped == "budget"
    assert stats.completed == 3
    assert stats.cost_audit == pytest.approx(0.3)


def test_zero_budget_submits_nothing(ledger: Ledger) -> None:
    c = FakeCascade()
    stats = run(c, tasks(5), ledger, run_id="r1", max_cost_usd=0.0)
    assert stats.stopped == "budget" and stats.submitted == 0 and c.calls == []


def test_unknown_cost_counted_and_warned_once(
    ledger: Ledger, caplog: pytest.LogCaptureFixture
) -> None:
    c = FakeCascade(cost=None)
    with caplog.at_level(logging.WARNING, logger="shadowgate.runner"):
        stats = run(c, tasks(6), ledger, run_id="r1", max_cost_usd=1.0)
    assert stats.unknown_cost == 6
    assert stats.cost_serving == 0.0
    assert stats.stopped is None  # unknown costs cannot trip the cap
    warnings = [r for r in caplog.records if "unknown" in r.getMessage()]
    assert len(warnings) == 1


def test_unknown_cost_no_warning_without_cap(
    ledger: Ledger, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="shadowgate.runner"):
        stats = run(FakeCascade(audit_cost=None), tasks(3), ledger, run_id="r1")
    assert stats.unknown_cost == 3
    assert not [r for r in caplog.records if "unknown" in r.getMessage()]


# --------------------------------------------------------------------------- failures


def test_route_exception_recorded_as_error_decision(ledger: Ledger) -> None:
    c = FakeCascade(raise_ids={"t2"})
    stats = run(c, tasks(5), ledger, run_id="r1")
    assert stats.completed == 5 and stats.failed == 1 and stats.stopped is None
    d = {x.task.id: x for x in ledger.decisions("r1")}["t2"]
    assert d.error is not None and "RuntimeError" in d.error and "boom t2" in d.error
    assert d.final_tier == "" and d.attempts == () and d.answer == ""
    assert d.cost_usd is None and d.audit_cost_usd is None
    assert stats.unknown_cost == 1


def test_decision_error_counts_as_failed(ledger: Ledger) -> None:
    stats = run(FakeCascade(fail_ids={"t0", "t3"}), tasks(5), ledger, run_id="r1")
    assert stats.failed == 2 and stats.completed == 5


def test_consecutive_failure_stop(ledger: Ledger) -> None:
    c = FakeCascade(always_error=True)
    stats = run(c, tasks(200), ledger, run_id="r1", workers=2, max_consecutive_failures=5)
    assert stats.stopped == "failures"
    assert stats.failed == stats.completed
    assert 5 <= stats.completed <= 5 + 2
    assert len(c.calls) == stats.completed


def test_consecutive_failure_streak_resets_on_success(ledger: Ledger) -> None:
    fail = {f"t{i}" for i in range(30) if i % 3 != 0}  # two failures, then a success
    stats = run(
        FakeCascade(fail_ids=fail),
        tasks(30),
        ledger,
        run_id="r1",
        workers=1,
        max_consecutive_failures=3,
    )
    assert stats.stopped is None and stats.completed == 30 and stats.failed == 20


def test_consecutive_failure_check_disabled(ledger: Ledger) -> None:
    stats = run(
        FakeCascade(raise_ids={f"t{i}" for i in range(30)}),
        tasks(30),
        ledger,
        run_id="r1",
        max_consecutive_failures=None,
    )
    assert stats.stopped is None and stats.failed == 30


# --------------------------------------------------------------------------- interrupts


def test_stop_event_interrupts_and_records_in_flight(ledger: Ledger) -> None:
    ev = threading.Event()
    c = FakeCascade(delay=0.02)

    def progress(stats: RunStats, d: Decision) -> None:
        if stats.completed == 5:
            ev.set()

    stats = run(c, tasks(100), ledger, run_id="r1", workers=3, progress=progress, stop_event=ev)
    assert stats.stopped == "interrupted"
    assert stats.completed == stats.submitted
    assert 5 <= stats.completed <= 5 + 3
    assert len(ledger.done_task_ids("r1")) == stats.completed


def test_stop_event_set_before_start(ledger: Ledger) -> None:
    ev = threading.Event()
    ev.set()
    c = FakeCascade()
    stats = run(c, tasks(5), ledger, run_id="r1", stop_event=ev)
    assert stats.stopped == "interrupted" and stats.submitted == 0 and c.calls == []


def test_keyboard_interrupt_drains_in_flight(ledger: Ledger) -> None:
    c = FakeCascade(delay=0.02)
    raised = []

    def progress(stats: RunStats, d: Decision) -> None:
        if stats.completed == 4 and not raised:
            raised.append(True)
            raise KeyboardInterrupt

    stats = run(c, tasks(100), ledger, run_id="r1", workers=3, progress=progress)
    assert stats.stopped == "interrupted"
    assert stats.completed == stats.submitted
    assert stats.completed <= 4 + 3
    assert len(ledger.done_task_ids("r1")) == stats.completed


def test_second_keyboard_interrupt_abandons_in_flight(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shadowgate.runner as runner_mod

    gate = threading.Event()  # workers stay blocked until the test ends
    c = FakeCascade(gate=gate)
    real_wait = runner_mod.wait
    count = {"n": 0}

    def fake_wait(fs, timeout=None, return_when=None):  # type: ignore[no-untyped-def]
        count["n"] += 1
        if count["n"] <= 2:  # Ctrl-C twice while waiting on blocked workers
            raise KeyboardInterrupt
        return real_wait(fs, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(runner_mod, "wait", fake_wait)
    try:
        stats = run(c, tasks(10), ledger, run_id="r1", workers=2)
    finally:
        gate.set()
    assert stats.stopped == "interrupted"
    assert stats.submitted == 2 and stats.completed == 0
    assert ledger.done_task_ids("r1") == set()


# --------------------------------------------------------------------------- progress & stats


def test_progress_called_on_main_thread_per_decision(ledger: Ledger) -> None:
    threads: set[str] = set()
    seen: list[str] = []

    def progress(stats: RunStats, d: Decision) -> None:
        threads.add(threading.current_thread().name)
        seen.append(d.task.id)

    run(FakeCascade(delay=0.005), tasks(12), ledger, run_id="r1", workers=4, progress=progress)
    assert sorted(seen) == sorted(f"t{i}" for i in range(12))
    assert threads == {threading.main_thread().name}


def test_progress_exceptions_ignored(ledger: Ledger, caplog: pytest.LogCaptureFixture) -> None:
    def progress(stats: RunStats, d: Decision) -> None:
        raise ValueError("bad callback")

    with caplog.at_level(logging.ERROR, logger="shadowgate.runner"):
        stats = run(FakeCascade(), tasks(6), ledger, run_id="r1", progress=progress)
    assert stats.completed == 6 and stats.stopped is None and stats.failed == 0
    assert any("progress callback" in r.getMessage() for r in caplog.records)


def test_escalated_and_audited_counters(ledger: Ledger) -> None:
    c = FakeCascade(escalate_ids={"t1", "t2"}, shadow_status="done")
    stats = run(c, tasks(5), ledger, run_id="r1")
    assert stats.escalated == 2 and stats.audited == 5


def test_summary_line_format() -> None:
    s = RunStats(
        run_id="r",
        completed=120,
        total=120,
        escalated=37,
        audited=12,
        cost_serving=0.4312,
        cost_audit=0.0911,
        elapsed_s=41.23,
    )
    assert s.summary_line() == (
        "120/120 done · 37 escalated · 12 audited · $0.4312 serving + $0.0911 audit · 41.2s"
    )
    s2 = RunStats(run_id="r", completed=3, total=10, failed=1, unknown_cost=2, stopped="budget")
    line = s2.summary_line()
    assert "3/10 done" in line and "1 failed" in line and "2 unknown cost" in line
    assert line.endswith("(stopped: budget)")


# --------------------------------------------------------------------------- pending audits


def _seed_pending(ledger: Ledger, n: int, run_id: str = "r1") -> None:
    run(FakeCascade(shadow_status="pending", audit_cost=0.0), tasks(n), ledger, run_id=run_id)


def test_pending_audits_processed(ledger: Ledger) -> None:
    _seed_pending(ledger, 6)
    c = FakeCascade(audit_cost=0.05, delay=0.01)
    stats = run_pending_audits(c, ledger, run_id="r1", workers=3)
    assert stats.total == 6 and stats.completed == 6 and stats.audited == 6
    assert stats.cost_audit == pytest.approx(0.30) and stats.cost_serving == 0.0
    assert c.max_in_flight <= 3
    assert list(ledger.pending_audits("r1")) == []
    assert all(d.shadow is not None and d.shadow.status == "done" for d in ledger.decisions("r1"))


def test_pending_audits_default_latest_run(ledger: Ledger) -> None:
    _seed_pending(ledger, 2, run_id="old")
    _seed_pending(ledger, 3, run_id="new")
    stats = run_pending_audits(FakeCascade(), ledger)
    assert stats.run_id == "new" and stats.completed == 3
    assert len(list(ledger.pending_audits("old"))) == 2


def test_pending_audits_empty_ledger(ledger: Ledger) -> None:
    stats = run_pending_audits(FakeCascade(), ledger)
    assert stats.total == 0 and stats.completed == 0 and stats.stopped is None


def test_pending_audits_budget(ledger: Ledger) -> None:
    _seed_pending(ledger, 10)
    stats = run_pending_audits(
        FakeCascade(audit_cost=0.1), ledger, run_id="r1", workers=1, max_cost_usd=0.25
    )
    assert stats.stopped == "budget" and stats.completed == 3
    assert len(list(ledger.pending_audits("r1"))) == 7


def test_pending_audits_failures_and_exceptions(ledger: Ledger) -> None:
    _seed_pending(ledger, 5)
    c = FakeCascade(fail_ids={"t1"}, raise_ids={"t3"})
    stats = run_pending_audits(c, ledger, run_id="r1")
    assert stats.completed == 5 and stats.failed == 2 and stats.audited == 3
    by_id = {d.task.id: d for d in ledger.decisions("r1")}
    assert by_id["t1"].shadow is not None and by_id["t1"].shadow.status == "error"
    # An unexpected exception leaves the record pending so it can be retried.
    assert [d.task.id for d in ledger.pending_audits("r1")] == ["t3"]


def test_pending_audits_consecutive_failure_stop(ledger: Ledger) -> None:
    _seed_pending(ledger, 40)
    c = FakeCascade(fail_ids={f"t{i}" for i in range(40)})
    stats = run_pending_audits(c, ledger, run_id="r1", workers=1, max_consecutive_failures=4)
    assert stats.stopped == "failures" and stats.completed == 4


def test_pending_audits_stop_event(ledger: Ledger) -> None:
    _seed_pending(ledger, 30)
    ev = threading.Event()

    def progress(stats: RunStats, d: Decision) -> None:
        if stats.completed == 2:
            ev.set()

    stats = run_pending_audits(
        FakeCascade(delay=0.01), ledger, run_id="r1", workers=2, progress=progress, stop_event=ev
    )
    assert stats.stopped == "interrupted" and stats.completed == stats.submitted
    assert stats.completed <= 2 + 2


def test_pending_audits_unknown_cost(ledger: Ledger) -> None:
    _seed_pending(ledger, 3)
    stats = run_pending_audits(FakeCascade(audit_cost=None), ledger, run_id="r1")
    assert stats.unknown_cost == 3 and stats.cost_audit == 0.0
