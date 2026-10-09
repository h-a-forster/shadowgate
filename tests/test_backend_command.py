from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from shadowgate.backends.base import RetryPolicy
from shadowgate.backends.command import CommandBackend, estimate_tokens
from shadowgate.errors import BackendError, ConfigError
from shadowgate.types import Backend, Request, Usage

FAST_RETRY = RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0)


def _script(tmp_path: Path, body: str, name: str = "model.py") -> list[str]:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(path)]


ECHO = """
import sys
data = sys.stdin.buffer.read().decode("utf-8")
sys.stdout.buffer.write(("ECHO:" + data + "\\n").encode("utf-8"))
"""


class FakePricing:
    def cost(self, usage: Usage) -> float:
        return usage.input_tokens * 1.0 + usage.output_tokens * 2.0


def test_echo_and_usage(tmp_path: Path) -> None:
    b = CommandBackend(_script(tmp_path, ECHO), name="echo", retry=FAST_RETRY)
    assert isinstance(b, Backend)
    comp = b.complete(Request(prompt="hello world"))
    assert comp.text == "ECHO:hello world"
    assert comp.usage.input_tokens == estimate_tokens("hello world") == 3
    assert comp.usage.output_tokens == estimate_tokens("ECHO:hello world") == 4
    assert comp.cost_usd is None
    assert comp.stop_reason == "end"
    assert comp.model == "echo"
    assert comp.latency_s > 0


def test_system_prepended(tmp_path: Path) -> None:
    b = CommandBackend(_script(tmp_path, ECHO), name="echo")
    comp = b.complete(Request(prompt="Q", system="SYS"))
    assert comp.text == "ECHO:SYS\n\nQ"


def test_pricing_used(tmp_path: Path) -> None:
    b = CommandBackend(_script(tmp_path, ECHO), name="echo", pricing=FakePricing())  # type: ignore[arg-type]
    comp = b.complete(Request(prompt="abcd"))
    assert comp.cost_usd == pytest.approx(1 * 1.0 + 3 * 2.0)


def test_utf8_and_invalid_bytes(tmp_path: Path) -> None:
    body = """
    import sys
    data = sys.stdin.buffer.read().decode("utf-8")
    sys.stdout.buffer.write(data.encode("utf-8") + b" \\xff\\xfe")
    """
    b = CommandBackend(_script(tmp_path, body), name="u")
    comp = b.complete(Request(prompt="東京 ✓"))
    assert comp.text.startswith("東京 ✓ ")
    assert "�" in comp.text


def test_default_name(tmp_path: Path) -> None:
    b = CommandBackend(["python3", "x.py"])
    assert b.name == "command:python3"


def test_string_command(tmp_path: Path) -> None:
    script = tmp_path / "echo model.py"
    script.write_text(textwrap.dedent(ECHO), encoding="utf-8")
    cmd = f'"{sys.executable}" "{script}"'
    b = CommandBackend(cmd, name="s")
    assert b.complete(Request(prompt="hi")).text == "ECHO:hi"


def test_empty_command_rejected() -> None:
    with pytest.raises(ConfigError):
        CommandBackend([])
    with pytest.raises(ConfigError):
        CommandBackend("   ")


def test_nonzero_exit_not_retryable(tmp_path: Path) -> None:
    counter = tmp_path / "n.txt"
    body = f"""
    import sys
    p = {str(counter)!r}
    open(p, "a").write("x")
    sys.stderr.write("model exploded")
    sys.exit(3)
    """
    b = CommandBackend(_script(tmp_path, body), name="bad", retry=FAST_RETRY)
    with pytest.raises(BackendError) as ei:
        b.complete(Request(prompt="x"))
    assert ei.value.retryable is False
    assert "code 3" in str(ei.value) and "model exploded" in str(ei.value)
    assert counter.read_text() == "x"


def test_retryable_exit_code_retried_then_succeeds(tmp_path: Path) -> None:
    counter = tmp_path / "n.txt"
    body = f"""
    import os, sys
    p = {str(counter)!r}
    n = len(open(p).read()) if os.path.exists(p) else 0
    open(p, "a").write("x")
    if n < 2:
        sys.exit(75)
    sys.stdout.write("finally")
    """
    b = CommandBackend(
        _script(tmp_path, body), name="flaky", retry=FAST_RETRY, retryable_exit_codes={75}
    )
    assert b.complete(Request(prompt="x")).text == "finally"
    assert counter.read_text() == "xxx"


def test_retryable_exit_code_exhausted(tmp_path: Path) -> None:
    body = """
    import sys
    sys.exit(75)
    """
    b = CommandBackend(
        _script(tmp_path, body), name="flaky", retry=FAST_RETRY, retryable_exit_codes=[75]
    )
    with pytest.raises(BackendError) as ei:
        b.complete(Request(prompt="x"))
    assert ei.value.retryable is True


def test_timeout_is_transient(tmp_path: Path) -> None:
    body = """
    import time
    time.sleep(5)
    """
    b = CommandBackend(
        _script(tmp_path, body), name="slow", timeout_s=0.5, retry=RetryPolicy(max_attempts=1)
    )
    with pytest.raises(BackendError) as ei:
        b.complete(Request(prompt="x"))
    assert ei.value.retryable is True
    assert "timed out" in str(ei.value)


def test_missing_program() -> None:
    b = CommandBackend(["/no/such/program-xyz"], name="missing", retry=FAST_RETRY)
    with pytest.raises(BackendError) as ei:
        b.complete(Request(prompt="x"))
    assert ei.value.retryable is False


def test_env_and_cwd(tmp_path: Path) -> None:
    body = """
    import os, sys
    sys.stdout.write(os.environ["SG_TEST_VAR"] + "|" + os.getcwd())
    """
    b = CommandBackend(
        _script(tmp_path, body), name="env", env={"SG_TEST_VAR": "v1"}, cwd=tmp_path
    )
    text = b.complete(Request(prompt="x")).text
    var, cwd = text.split("|", 1)
    assert var == "v1"
    assert Path(cwd).resolve() == tmp_path.resolve()


GRANDCHILD = """
import subprocess, sys, time
# The grandchild inherits stdout/stderr; it keeps the pipes open after we are killed.
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 30)
print("late")
"""


@pytest.mark.parametrize("child_sleep", ["30", "0"])
def test_timeout_with_grandchild_is_enforced(tmp_path: Path, child_sleep: str) -> None:
    import time

    # child_sleep=30: child and grandchild both hang. 0: child exits at once, but the
    # grandchild holds stdout open, so the output is not complete before the deadline.
    cmd = [*_script(tmp_path, GRANDCHILD, name="gc.py"), child_sleep]
    b = CommandBackend(cmd, name="gc", timeout_s=1.0, retry=RetryPolicy(max_attempts=1))
    t0 = time.monotonic()
    with pytest.raises(BackendError) as ei:
        b.complete(Request(prompt="x"))
    elapsed = time.monotonic() - t0
    assert ei.value.retryable is True and "timed out" in str(ei.value)
    assert elapsed < 1.0 + 4.0, elapsed


def test_latency_is_final_attempt_only(tmp_path: Path) -> None:
    body = """
    import pathlib, sys, time
    marker = pathlib.Path(sys.argv[1])
    if not marker.exists():
        marker.write_text("x")
        time.sleep(0.6)
        sys.exit(75)
    print("ok")
    """
    cmd = [*_script(tmp_path, body, name="flaky.py"), str(tmp_path / "marker")]
    b = CommandBackend(cmd, name="flaky", retry=FAST_RETRY, retryable_exit_codes=[75])
    comp = b.complete(Request(prompt="x"))
    assert comp.text == "ok"
    assert comp.latency_s < 0.6
