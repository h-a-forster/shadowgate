"""Concurrent batch runner: routes tasks through a cascade and records every decision.

Two entry points share one scheduling loop:

* :func:`run` routes a batch of tasks and writes each finished :class:`Decision` to the ledger
  as soon as it completes (crash-safe and resumable).
* :func:`run_pending_audits` completes deferred shadow audits already stored in the ledger.

Scheduling semantics (both entry points):

* At most ``workers`` items are in flight; new work is submitted only as earlier work finishes.
* All ledger writes and ``progress`` callbacks happen on the calling (main) thread.
* Budget: once known spend reaches ``max_cost_usd`` no new work is submitted; in-flight work
  finishes and is recorded, and ``RunStats.stopped == "budget"``. Unknown costs (``None``) cannot
  be counted against the cap; they are tallied in ``RunStats.unknown_cost`` and a warning is
  logged once. :class:`~shadowgate.errors.BudgetExceeded` is never raised here: the caller decides
  how to report a budget stop (see :attr:`RunStats.budget_exceeded`).
* Failures: a decision whose ``error`` is set (e.g. the final tier's backend failed) counts as
  failed, as does an unexpected exception from the cascade. After ``max_consecutive_failures``
  failures in a row (by completion order) submission stops with ``stopped == "failures"``; this
  catches systemic problems such as missing credentials before a whole dataset is consumed.
  ``None`` disables the check.
* Interrupts: setting ``stop_event`` or a ``KeyboardInterrupt`` stops submission; in-flight work
  finishes and is recorded and ``stopped == "interrupted"`` (nothing is re-raised). A second
  ``KeyboardInterrupt`` while draining abandons the in-flight work: worker threads are daemon
  threads, so their results are discarded and they do not block interpreter exit.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

from .errors import DatasetError
from .ledger import Ledger
from .types import Decision, Task

__all__ = ["RunStats", "run", "run_pending_audits", "DEFAULT_MAX_CONSECUTIVE_FAILURES"]

log = logging.getLogger("shadowgate.runner")

DEFAULT_MAX_CONSECUTIVE_FAILURES = 20
_POLL_S = 0.1  # wait() timeout; keeps the main thread responsive to Ctrl-C on every platform

T = TypeVar("T")


class _Router(Protocol):
    def route(self, task: Task, *, run_id: str = ..., mode: str = ...) -> Decision: ...


class _Auditor(Protocol):
    def complete_audit(self, decision: Decision) -> Decision: ...


@dataclass
class RunStats:
    """Counters for one runner invocation (not cumulative across resumed invocations).

    ``total`` is the number of input items considered (tasks, or pending audits).
    ``submitted`` were handed to a worker; ``completed`` were recorded (including failed ones);
    ``skipped_existing`` were already in the ledger (resume); ``retried`` were recorded with
    ``error`` set by an earlier invocation and routed again on resume. ``cost_serving`` and
    ``cost_audit`` sum the known costs of the decisions recorded by this invocation (for
    :func:`run_pending_audits`, ``cost_audit`` is the audit cost added by this invocation).
    ``unknown_cost`` counts recorded decisions with at least one unknown cost. ``stopped`` is
    None when every item was processed, else "budget", "failures" or "interrupted".
    """

    run_id: str
    submitted: int = 0
    completed: int = 0
    skipped_existing: int = 0
    retried: int = 0
    failed: int = 0
    cost_serving: float = 0.0
    cost_audit: float = 0.0
    unknown_cost: int = 0
    elapsed_s: float = 0.0
    stopped: str | None = None
    total: int = 0
    escalated: int = 0
    audited: int = 0

    @property
    def cost_total(self) -> float:
        return self.cost_serving + self.cost_audit

    @property
    def budget_exceeded(self) -> bool:
        return self.stopped == "budget"

    def summary_line(self) -> str:
        done = self.completed + self.skipped_existing
        parts = [f"{done}/{self.total} done"]
        if self.skipped_existing:
            parts.append(f"{self.skipped_existing} resumed")
        if self.retried:
            parts.append(f"{self.retried} retried")
        parts.append(f"{self.escalated} escalated")
        parts.append(f"{self.audited} audited")
        if self.failed:
            parts.append(f"{self.failed} failed")
        parts.append(f"${self.cost_serving:.4f} serving + ${self.cost_audit:.4f} audit")
        if self.unknown_cost:
            parts.append(f"{self.unknown_cost} unknown cost")
        parts.append(f"{self.elapsed_s:.1f}s")
        line = " | ".join(parts)
        if self.stopped:
            line += f" (stopped: {self.stopped})"
        return line


# --------------------------------------------------------------------------- scheduling


def _spawn(fn: Callable[[], T]) -> Future[T]:
    """Run ``fn`` on a fresh daemon thread; the returned Future carries its outcome."""
    fut: Future[T] = Future()
    fut.set_running_or_notify_cancel()

    def target() -> None:
        try:
            result = fn()
        except BaseException as exc:  # surfaced to the main thread via the future
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    threading.Thread(target=target, name="shadowgate-worker", daemon=True).start()
    return fut


def _drive(
    items: Sequence[T],
    work: Callable[[T], Callable[[], Any]],
    handle: Callable[[T, Future[Any]], tuple[bool, Decision | None]],
    after: Callable[[Decision], None],
    *,
    workers: int,
    budget_hit: Callable[[], bool],
    stop_event: threading.Event | None,
    max_consecutive_failures: int | None,
    on_submit: Callable[[], None],
) -> str | None:
    """Bounded-concurrency loop. Returns the stop reason (None when every item was processed).

    ``handle`` records one finished item and returns ``(failed, recorded decision or None)``;
    ``after`` (the progress hook) runs only once the item is no longer pending, so an interrupt
    raised from it cannot cause the item to be handled twice.
    """

    pending: dict[Future[Any], T] = {}
    next_index = 0
    stopped: str | None = None
    streak = 0

    while True:
        try:
            while stopped is None and next_index < len(items) and len(pending) < workers:
                if stop_event is not None and stop_event.is_set():
                    stopped = "interrupted"
                    break
                if budget_hit():
                    stopped = "budget"
                    break
                item = items[next_index]
                next_index += 1
                pending[_spawn(work(item))] = item
                on_submit()
            if not pending:
                break
            done, _ = wait(list(pending), timeout=_POLL_S, return_when=FIRST_COMPLETED)
            for fut in done:
                failed, decision = handle(pending[fut], fut)
                del pending[fut]
                streak = streak + 1 if failed else 0
                if (
                    stopped is None
                    and max_consecutive_failures is not None
                    and streak >= max_consecutive_failures
                ):
                    log.error(
                        "stopping after %d consecutive failures (likely a systemic problem "
                        "such as missing credentials or an unreachable endpoint)",
                        streak,
                    )
                    stopped = "failures"
                if decision is not None:
                    after(decision)
            if stopped is None and stop_event is not None and stop_event.is_set():
                stopped = "interrupted"
        except KeyboardInterrupt:
            if stopped == "interrupted" and pending:
                log.warning(
                    "second interrupt: abandoning %d in-flight item(s) without recording them",
                    len(pending),
                )
                return "interrupted"
            log.warning(
                "interrupted: waiting for %d in-flight item(s); interrupt again to abandon them",
                len(pending),
            )
            stopped = "interrupted"
    return stopped


# --------------------------------------------------------------------------- accounting


class _Accountant:
    """Shared budget / stats bookkeeping (main thread only)."""

    def __init__(self, stats: RunStats, max_cost_usd: float | None) -> None:
        self.stats = stats
        self.max_cost_usd = max_cost_usd
        self._warned_unknown = False

    def budget_hit(self) -> bool:
        return self.max_cost_usd is not None and self.stats.cost_total >= self.max_cost_usd

    def add_costs(self, serving: float | None, audit: float | None, *, count_serving: bool) -> None:
        unknown = False
        if count_serving:
            if serving is None:
                unknown = True
            else:
                self.stats.cost_serving += serving
        if audit is None:
            unknown = True
        else:
            self.stats.cost_audit += audit
        if unknown:
            self.stats.unknown_cost += 1
            if self.max_cost_usd is not None and not self._warned_unknown:
                self._warned_unknown = True
                log.warning(
                    "some costs are unknown (no pricing for a model); the budget cap of $%.4f "
                    "only counts known costs and cannot be enforced for the rest",
                    self.max_cost_usd,
                )


def _check_args(workers: int, max_consecutive_failures: int | None) -> None:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if max_consecutive_failures is not None and max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be >= 1 or None")


def _call_progress(
    progress: Callable[[RunStats, Decision], None] | None, stats: RunStats, d: Decision
) -> None:
    if progress is None:
        return
    try:
        progress(stats, d)
    except Exception:
        log.exception("progress callback raised; ignoring")


def _describe(exc: BaseException) -> str:
    msg = str(exc)
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


# --------------------------------------------------------------------------- run


def run(
    cascade: _Router,
    tasks: Iterable[Task],
    ledger: Ledger,
    *,
    run_id: str,
    mode: str = "serve",
    workers: int = 4,
    max_cost_usd: float | None = None,
    resume: bool = True,
    config_snapshot: dict[str, Any] | None = None,
    progress: Callable[[RunStats, Decision], None] | None = None,
    stop_event: threading.Event | None = None,
    max_consecutive_failures: int | None = DEFAULT_MAX_CONSECUTIVE_FAILURES,
) -> RunStats:
    """Route ``tasks`` through ``cascade`` with bounded concurrency, recording to ``ledger``.

    Duplicate task ids raise :class:`DatasetError` before anything is written. The run is
    registered with ``ledger.start_run(run_id, config=config_snapshot or {}, mode=mode)``.
    With ``resume=True`` tasks already recorded for ``run_id`` are skipped (counted in
    ``skipped_existing``), except failed records (``error`` set), which are routed again and
    replaced (counted in ``retried``); with ``resume=False`` every task is routed again and its
    record is replaced (upsert). An unexpected exception from ``cascade.route`` is recorded as a
    Decision with ``error`` set, empty ``final_tier``/``answer``/``attempts`` and unknown costs.
    See the module docstring for budget, failure and interrupt semantics.
    """
    _check_args(workers, max_consecutive_failures)
    t0 = time.perf_counter()
    task_list = list(tasks)
    seen: set[str] = set()
    dupes: list[str] = []
    for t in task_list:
        if t.id in seen:
            dupes.append(t.id)
        seen.add(t.id)
    if dupes:
        shown = ", ".join(repr(x) for x in sorted(set(dupes))[:5])
        raise DatasetError(f"duplicate task ids in input: {shown}")

    ledger.start_run(run_id, config=config_snapshot or {}, mode=mode)
    stats = RunStats(run_id=run_id, total=len(task_list))

    if resume:
        done_ids = ledger.done_task_ids(run_id)
        failed_ids = ledger.done_task_ids(run_id, include_errors=True) - done_ids
        todo = [t for t in task_list if t.id not in done_ids]
        stats.skipped_existing = len(task_list) - len(todo)
        stats.retried = sum(1 for t in todo if t.id in failed_ids)
    else:
        todo = task_list

    acct = _Accountant(stats, max_cost_usd)

    def work(task: Task) -> Callable[[], Decision]:
        def call() -> Decision:
            start = time.perf_counter()
            try:
                return cascade.route(task, run_id=run_id, mode=mode)
            except Exception as exc:
                # Wrapped so the main thread can record the elapsed time too.
                raise _RouteFailure(exc, time.perf_counter() - start) from exc

        return call

    def handle(task: Task, fut: Future[Decision]) -> tuple[bool, Decision | None]:
        exc = fut.exception()
        if exc is None:
            d = fut.result()
        else:
            cause = exc.exc if isinstance(exc, _RouteFailure) else exc
            latency = exc.elapsed_s if isinstance(exc, _RouteFailure) else 0.0
            log.error("route raised for task %r: %s", task.id, _describe(cause), exc_info=cause)
            d = Decision(
                run_id=run_id,
                task=task,
                answer="",
                final_tier="",
                escalated=False,
                attempts=(),
                cost_usd=None,
                audit_cost_usd=None,
                latency_s=latency,
                mode=mode,
                created_at=datetime.now(UTC).isoformat(),
                error=f"unexpected error: {_describe(cause)}",
            )
        ledger.record(d)
        stats.completed += 1
        failed = d.error is not None
        if failed:
            stats.failed += 1
        if d.escalated:
            stats.escalated += 1
        if d.shadow is not None and d.shadow.status == "done":
            stats.audited += 1
        acct.add_costs(d.cost_usd, d.audit_cost_usd, count_serving=True)
        stats.elapsed_s = time.perf_counter() - t0
        return failed, d

    def on_submit() -> None:
        stats.submitted += 1

    try:
        stats.stopped = _drive(
            todo,
            work,
            handle,
            lambda d: _call_progress(progress, stats, d),
            workers=workers,
            budget_hit=acct.budget_hit,
            stop_event=stop_event,
            max_consecutive_failures=max_consecutive_failures,
            on_submit=on_submit,
        )
    finally:
        stats.elapsed_s = time.perf_counter() - t0
    return stats


class _RouteFailure(Exception):
    def __init__(self, exc: BaseException, elapsed_s: float) -> None:
        super().__init__(str(exc))
        self.exc = exc
        self.elapsed_s = elapsed_s


# --------------------------------------------------------------------------- deferred audits


def run_pending_audits(
    cascade: _Auditor,
    ledger: Ledger,
    *,
    run_id: str | None = None,
    workers: int = 4,
    max_cost_usd: float | None = None,
    progress: Callable[[RunStats, Decision], None] | None = None,
    stop_event: threading.Event | None = None,
    max_consecutive_failures: int | None = DEFAULT_MAX_CONSECUTIVE_FAILURES,
) -> RunStats:
    """Complete deferred shadow audits of ``run_id`` (default: the latest run).

    Each pending decision goes through ``cascade.complete_audit`` and the updated decision is
    upserted. An audit whose resulting ``shadow.status`` is "error" counts as failed (it is
    recorded, so it is not retried automatically). An unexpected exception leaves the record
    untouched (still pending, so a later call retries it) and counts as failed. Only the audit
    cost added by this call is counted (``cost_audit``) and checked against ``max_cost_usd``.
    Budget, failure-streak and interrupt semantics match :func:`run`.
    """
    _check_args(workers, max_consecutive_failures)
    t0 = time.perf_counter()
    rid = run_id if run_id is not None else ledger.latest_run_id()
    items = [] if rid is None else list(ledger.pending_audits(rid))
    stats = RunStats(run_id=rid or "", total=len(items))
    acct = _Accountant(stats, max_cost_usd)

    def work(d: Decision) -> Callable[[], Decision]:
        return lambda: cascade.complete_audit(d)

    def handle(old: Decision, fut: Future[Decision]) -> tuple[bool, Decision | None]:
        exc = fut.exception()
        if exc is not None:
            log.error(
                "complete_audit raised for task %r: %s", old.task.id, _describe(exc), exc_info=exc
            )
            stats.completed += 1
            stats.failed += 1
            stats.elapsed_s = time.perf_counter() - t0
            return True, None
        new = fut.result()
        ledger.record(new)
        stats.completed += 1
        status = new.shadow.status if new.shadow is not None else None
        failed = status == "error"
        if failed:
            stats.failed += 1
        if status == "done":
            stats.audited += 1
        if new.escalated:
            stats.escalated += 1
        if new.audit_cost_usd is None or old.audit_cost_usd is None:
            delta: float | None = None
        else:
            delta = max(0.0, new.audit_cost_usd - old.audit_cost_usd)
        acct.add_costs(None, delta, count_serving=False)
        stats.elapsed_s = time.perf_counter() - t0
        return failed, new

    def on_submit() -> None:
        stats.submitted += 1

    try:
        stats.stopped = _drive(
            items,
            work,
            handle,
            lambda d: _call_progress(progress, stats, d),
            workers=workers,
            budget_hit=acct.budget_hit,
            stop_event=stop_event,
            max_consecutive_failures=max_consecutive_failures,
            on_submit=on_submit,
        )
    finally:
        stats.elapsed_s = time.perf_counter() - t0
    return stats
