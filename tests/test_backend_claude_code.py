from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from shadowgate.backends.base import RetryPolicy
from shadowgate.backends.claude_code import DEFAULT_SYSTEM, ClaudeCodeBackend
from shadowgate.errors import BackendError, ConfigError
from shadowgate.types import Request

FIXTURE = Path(__file__).parent / "fixtures" / "claude_code_result.json"

# A stand-in for the CLI. It logs each invocation (argv, cwd, stdin, cwd contents) and replays
# scripted responses in order: {"stdout": ..., "stderr": ..., "exit": ..., "sleep": ...}.
FAKE_CLI = textwrap.dedent(
    """
    import json, os, sys, time
    cfg_path = sys.argv[1]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    log_path = cfg_path + ".log"
    calls = []
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            calls = json.load(f)
    stdin = sys.stdin.buffer.read().decode("utf-8")
    calls.append({"argv": sys.argv[2:], "cwd": os.getcwd(), "stdin": stdin,
                  "cwd_entries": os.listdir(".")})
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(calls, f)
    resp = cfg["responses"][min(len(calls), len(cfg["responses"])) - 1]
    if resp.get("sleep"):
        time.sleep(resp["sleep"])
    sys.stdout.buffer.write(resp.get("stdout", "").encode("utf-8"))
    sys.stderr.buffer.write(resp.get("stderr", "").encode("utf-8"))
    sys.exit(resp.get("exit", 0))
    """
)

FAST_RETRY = RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0)


def _result(**overrides: Any) -> str:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data.update(overrides)
    return json.dumps(data)


def _error_result(message: str, status: int | None = None) -> str:
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "result": message,
        "api_error_status": status,
        "total_cost_usd": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    })


class FakeCli:
    def __init__(self, tmp_path: Path) -> None:
        self.script = tmp_path / "fake_claude.py"
        self.script.write_text(FAKE_CLI, encoding="utf-8")
        self.cfg = tmp_path / "cfg.json"

    def respond(self, *responses: dict[str, Any]) -> None:
        self.cfg.write_text(json.dumps({"responses": list(responses)}), encoding="utf-8")

    @property
    def executable(self) -> list[str]:
        return [sys.executable, str(self.script), str(self.cfg)]

    @property
    def calls(self) -> list[dict[str, Any]]:
        log = Path(str(self.cfg) + ".log")
        return json.loads(log.read_text(encoding="utf-8")) if log.exists() else []

    def backend(self, **kwargs: Any) -> ClaudeCodeBackend:
        kwargs.setdefault("retry", FAST_RETRY)
        return ClaudeCodeBackend("haiku", executable=self.executable, **kwargs)


@pytest.fixture
def cli(tmp_path: Path) -> FakeCli:
    return FakeCli(tmp_path)


def test_fixture_shape() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for key in ("type", "subtype", "is_error", "result", "usage", "total_cost_usd",
                "stop_reason", "session_id", "modelUsage"):
        assert key in data
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens"):
        assert key in data["usage"]


def test_success_parses_result(cli: FakeCli) -> None:
    cli.respond({"stdout": _result(result="Paris")})
    comp = cli.backend().complete(Request(prompt="Capital of France?"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert comp.text == "Paris"
    assert comp.cost_usd == pytest.approx(fixture["total_cost_usd"])
    assert comp.usage.input_tokens == fixture["usage"]["input_tokens"]
    assert comp.usage.output_tokens == fixture["usage"]["output_tokens"]
    assert comp.usage.cache_read_tokens == fixture["usage"]["cache_read_input_tokens"]
    assert comp.usage.cache_write_tokens == fixture["usage"]["cache_creation_input_tokens"]
    assert comp.stop_reason == "end"
    assert comp.model == next(iter(fixture["modelUsage"]))
    assert comp.latency_s > 0


def test_name_and_protocol() -> None:
    from shadowgate.types import Backend

    b = ClaudeCodeBackend("sonnet", executable=["x"])
    assert b.name == "claude-code:sonnet"
    assert isinstance(b, Backend)
    assert ClaudeCodeBackend("sonnet", executable=["x"], name="cc").name == "cc"


def test_invocation_flags_stdin_and_cwd(cli: FakeCli, tmp_path: Path) -> None:
    cli.respond({"stdout": _result()})
    prompt = "Say OK " + "x" * 50_000  # far beyond any argv-safe size: must go via stdin
    cli.backend(extra_args=("--verbose",)).complete(
        Request(prompt=prompt, system="Be terse.", effort="low")
    )
    (call,) = cli.calls
    argv = call["argv"]
    assert call["stdin"] == prompt
    assert prompt not in argv
    assert argv[:5] == ["-p", "--output-format", "json", "--model", "haiku"]
    assert argv[argv.index("--system-prompt") + 1] == "Be terse."
    assert argv[argv.index("--effort") + 1] == "low"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in argv
    assert argv[-1] == "--verbose"
    # Fresh, empty, temporary working directory that is removed afterwards.
    assert call["cwd_entries"] == []
    assert Path(call["cwd"]).resolve() != Path(os.getcwd()).resolve()
    assert not Path(call["cwd"]).exists()


def test_default_system_and_no_effort(cli: FakeCli) -> None:
    cli.respond({"stdout": _result()})
    cli.backend().complete(Request(prompt="hi"))
    argv = cli.calls[0]["argv"]
    assert argv[argv.index("--system-prompt") + 1] == DEFAULT_SYSTEM
    assert "--effort" not in argv


def test_keep_cli_default_system(cli: FakeCli) -> None:
    cli.respond({"stdout": _result()})
    cli.backend(default_system=None, isolation_args=()).complete(Request(prompt="hi"))
    argv = cli.calls[0]["argv"]
    assert "--system-prompt" not in argv
    assert "--tools" not in argv


def test_long_system_prompt_goes_in_band(cli: FakeCli) -> None:
    cli.respond({"stdout": _result()})
    system = "rule " * 5000
    cli.backend().complete(Request(prompt="question", system=system))
    call = cli.calls[0]
    assert call["stdin"] == f"{system}\n\nquestion"
    assert call["argv"][call["argv"].index("--system-prompt") + 1] == DEFAULT_SYSTEM


def test_utf8_round_trip(cli: FakeCli) -> None:
    cli.respond({"stdout": _result(result="naïve — 東京")})
    comp = cli.backend().complete(Request(prompt="Ünïcödé ✓"))
    assert comp.text == "naïve — 東京"
    assert cli.calls[0]["stdin"] == "Ünïcödé ✓"


def test_stop_reason_mapping(cli: FakeCli) -> None:
    cli.respond({"stdout": _result(stop_reason="max_tokens")})
    assert cli.backend().complete(Request(prompt="x")).stop_reason == "max_tokens"


def test_stdout_noise_tolerated(cli: FakeCli) -> None:
    cli.respond({"stdout": "warning: something\n" + _result(result="ok") + "\n"})
    assert cli.backend().complete(Request(prompt="x")).text == "ok"


def test_rate_limit_is_retried_then_succeeds(cli: FakeCli) -> None:
    cli.respond(
        {"stdout": _error_result("API Error: 429 rate limit exceeded", 429), "exit": 1},
        {"stdout": _result(result="done")},
    )
    comp = cli.backend().complete(Request(prompt="x"))
    assert comp.text == "done"
    assert len(cli.calls) == 2


def test_overload_exhausts_retries(cli: FakeCli) -> None:
    cli.respond({"stdout": _error_result("Overloaded", 529), "exit": 1})
    with pytest.raises(BackendError) as ei:
        cli.backend().complete(Request(prompt="x"))
    assert ei.value.retryable is True
    assert ei.value.status == 529
    assert len(cli.calls) == FAST_RETRY.max_attempts


def test_timeout_is_transient(cli: FakeCli) -> None:
    cli.respond({"stdout": _result(), "sleep": 5})
    with pytest.raises(BackendError) as ei:
        cli.backend(timeout_s=0.5, retry=RetryPolicy(max_attempts=1)).complete(
            Request(prompt="x")
        )
    assert ei.value.retryable is True
    assert "timed out" in str(ei.value)


def test_auth_error_not_retried(cli: FakeCli) -> None:
    cli.respond({"stdout": _error_result("Not logged in · Please run /login"), "exit": 1})
    with pytest.raises(BackendError) as ei:
        cli.backend().complete(Request(prompt="x"))
    assert ei.value.retryable is False
    assert "log in" in str(ei.value)
    assert len(cli.calls) == 1


def test_auth_status_not_retried(cli: FakeCli) -> None:
    cli.respond({"stdout": _error_result("Invalid bearer", 401), "exit": 1})
    with pytest.raises(BackendError) as ei:
        cli.backend().complete(Request(prompt="x"))
    assert ei.value.retryable is False and ei.value.status == 401


def test_other_error_not_retried(cli: FakeCli) -> None:
    cli.respond({"stdout": _error_result("prompt is invalid"), "exit": 1})
    with pytest.raises(BackendError) as ei:
        cli.backend().complete(Request(prompt="x"))
    assert ei.value.retryable is False
    assert len(cli.calls) == 1


def test_no_json_output(cli: FakeCli) -> None:
    cli.respond({"stdout": "", "stderr": "unknown option --frobnicate", "exit": 2})
    with pytest.raises(BackendError) as ei:
        cli.backend().complete(Request(prompt="x"))
    assert "frobnicate" in str(ei.value)
    assert ei.value.retryable is False


def test_missing_executable() -> None:
    b = ClaudeCodeBackend("haiku", executable="definitely-not-a-real-claude-cli-xyz")
    with pytest.raises(BackendError) as ei:
        b.complete(Request(prompt="x"))
    assert "not found" in str(ei.value)
    assert ei.value.retryable is False


def test_missing_executable_in_argv() -> None:
    b = ClaudeCodeBackend("haiku", executable=["/no/such/dir/claude-xyz"], retry=FAST_RETRY)
    with pytest.raises(BackendError) as ei:
        b.complete(Request(prompt="x"))
    assert ei.value.retryable is False


def test_invalid_model_rejected() -> None:
    with pytest.raises(ConfigError):
        ClaudeCodeBackend("haiku & calc", executable=["x"])


def test_invalid_effort_rejected(cli: FakeCli) -> None:
    cli.respond({"stdout": _result()})
    with pytest.raises(BackendError):
        cli.backend().complete(Request(prompt="x", effort="low & calc"))
    assert cli.calls == []
