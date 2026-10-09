"""Run a child process with a hard overall timeout that also kills its descendants.

``subprocess.run(timeout=...)`` kills only the direct child; on Windows a grandchild that
inherited the output pipes keeps them open and the following ``communicate()`` blocks until the
grandchild exits. Here the child gets its own process group (POSIX: a new session, killed with
``os.killpg``; Windows: ``CREATE_NEW_PROCESS_GROUP``, killed with ``taskkill /T /F``), the pipes are
pumped by daemon threads, and every wait is bounded by the deadline.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import IO

__all__ = ["ProcessTimeout", "ProcResult", "kill_tree", "run_process"]

# After the deadline (and the kill), how long to wait for the pipe readers to finish.
_DRAIN_GRACE_S = 2.0


class ProcessTimeout(Exception):
    """The child (or a descendant holding its pipes) outlived the timeout; the tree was killed."""


@dataclass(frozen=True)
class ProcResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Best-effort kill of ``proc`` and all of its descendants."""
    if os.name == "nt":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
    else:
        with contextlib.suppress(OSError, AttributeError):
            os.killpg(proc.pid, signal.SIGKILL)  # type: ignore[attr-defined]
    with contextlib.suppress(OSError):
        proc.kill()


def _reader(stream: IO[bytes], sink: list[bytes]) -> None:
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            sink.append(chunk)
    except (OSError, ValueError):
        pass


def _writer(stream: IO[bytes], data: bytes) -> None:
    try:
        if data:
            stream.write(data)
    except (OSError, ValueError):
        pass  # child exited or closed stdin early
    finally:
        with contextlib.suppress(OSError):
            stream.close()


def run_process(
    argv: Sequence[str] | str,
    *,
    input: bytes,
    timeout: float,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ProcResult:
    """Run ``argv`` with ``input`` on stdin and return its exit code and captured output.

    Raises ``ProcessTimeout`` when the child has not exited and closed its output pipes within
    ``timeout`` seconds (the whole process tree is killed first), ``OSError`` when it cannot be
    started.
    """
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    deadline = time.monotonic() + timeout
    proc = subprocess.Popen(  # noqa: S603 - argv is built by the caller, no shell
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        **kwargs,  # type: ignore[arg-type]
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    out: list[bytes] = []
    err: list[bytes] = []
    threads = [
        threading.Thread(target=_writer, args=(proc.stdin, input), daemon=True),
        threading.Thread(target=_reader, args=(proc.stdout, out), daemon=True),
        threading.Thread(target=_reader, args=(proc.stderr, err), daemon=True),
    ]
    for t in threads:
        t.start()
    timed_out = False
    try:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        else:
            # The child exited; a descendant may still hold the pipes open.
            for t in threads[1:]:
                t.join(max(0.0, deadline - time.monotonic()))
            timed_out = any(t.is_alive() for t in threads[1:])
        if timed_out:
            kill_tree(proc)
    except BaseException:
        kill_tree(proc)
        raise
    finally:
        if timed_out:
            grace = time.monotonic() + _DRAIN_GRACE_S
            for t in threads:
                t.join(max(0.0, grace - time.monotonic()))
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=max(0.0, grace - time.monotonic()))
    if timed_out:
        raise ProcessTimeout(f"process did not finish within {timeout:g}s")
    for t in threads:
        t.join(0.1)
    return ProcResult(proc.returncode, b"".join(out), b"".join(err))
