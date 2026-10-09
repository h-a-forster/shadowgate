from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from shadowgate.backends.base import RetryPolicy, TransientError, request_key
from shadowgate.backends.cache import CachedBackend, CacheStore
from shadowgate.backends.function import FunctionBackend
from shadowgate.backends.replay import ReplayBackend
from shadowgate.errors import BackendError, ConfigError
from shadowgate.pricing import Pricing
from shadowgate.types import Completion, Request, Usage

FULL = Completion(
    text="ANSWER: 42",
    model="m",
    usage=Usage(10, 5, 2, 1),
    cost_usd=0.0012,
    latency_s=1.25,
    stop_reason="max_tokens",
    logprobs=(-0.1, -0.25),
)


# --------------------------------------------------------------------------- replay


def write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_replay_by_key_and_by_task(tmp_path: Path) -> None:
    req = Request(prompt="hello", n_sample=1)
    key = request_key("openai:gpt", req)
    path = write_jsonl(
        tmp_path / "rec.jsonl",
        [
            {"key": key, "completion": FULL.to_dict()},
            {"task_id": "t1", "tier": "fast", "completion": {"text": "fast", "model": "a"}},
            {"task_id": "t1", "completion": {"text": "any", "model": "a"}},
        ],
    )
    rb = ReplayBackend(path, key_backend="openai:gpt")
    assert rb.name == "replay:rec"
    assert len(rb) == 3
    assert rb.complete(Request(prompt="hello", n_sample=1, tags={"x": "y"})) == FULL
    assert rb.complete(Request(prompt="q", tags={"task_id": "t1", "tier": "fast"})).text == "fast"
    assert rb.complete(Request(prompt="q", tags={"task_id": "t1", "tier": "slow"})).text == "any"
    with pytest.raises(BackendError):
        rb.complete(Request(prompt="hello", n_sample=2))
    with pytest.raises(BackendError):
        rb.complete(Request(prompt="q", tags={"task_id": "t2"}))


def test_replay_name_is_default_key_backend(tmp_path: Path) -> None:
    req = Request(prompt="p")
    path = write_jsonl(
        tmp_path / "r.jsonl", [{"key": request_key("rec", req), "completion": FULL.to_dict()}]
    )
    assert ReplayBackend(path, name="rec").complete(req) == FULL


def test_replay_bad_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ReplayBackend(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"key": "k", "completion": {}}\nnot json\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=":2"):
        ReplayBackend(bad)
    write_jsonl(bad, [{"completion": {}}])
    with pytest.raises(ConfigError):
        ReplayBackend(bad)
    write_jsonl(bad, [{"key": "k"}])
    with pytest.raises(ConfigError):
        ReplayBackend(bad)


# --------------------------------------------------------------------------- function


def test_function_backend_string() -> None:
    fb = FunctionBackend(lambda r: r.prompt.upper(), name="upper",
                         pricing=Pricing(input_per_mtok=1.0, output_per_mtok=2.0))
    c = fb.complete(Request(prompt="abcdefgh", system="s"))
    assert c.text == "ABCDEFGH"
    assert c.model == "upper" and fb.name == "upper"
    assert c.usage == Usage(input_tokens=3, output_tokens=2)
    assert c.cost_usd == pytest.approx((3 * 1.0 + 2 * 2.0) / 1e6)
    assert c.latency_s >= 0
    assert FunctionBackend(lambda r: "x").complete(Request(prompt="p")).cost_usd is None


def test_function_backend_completion_passthrough_and_errors() -> None:
    assert FunctionBackend(lambda r: FULL).complete(Request(prompt="p")) == FULL

    def boom(r: Request) -> str:
        raise ValueError("nope")

    with pytest.raises(BackendError, match="nope"):
        FunctionBackend(boom).complete(Request(prompt="p"))
    with pytest.raises(BackendError):
        FunctionBackend(lambda r: 3).complete(Request(prompt="p"))  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError):
        FunctionBackend("not callable")  # type: ignore[arg-type]


def test_function_backend_retries_transient() -> None:
    calls = []

    def flaky(r: Request) -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("busy", retry_after=0)
        return "ok"

    fb = FunctionBackend(flaky, retry=RetryPolicy(max_attempts=3))
    assert fb.complete(Request(prompt="p")).text == "ok"
    assert len(calls) == 3


# --------------------------------------------------------------------------- cache


class Counting:
    def __init__(self, name: str = "inner", stop_reason: str = "end") -> None:
        self.name = name
        self.calls = 0
        self.stop_reason = stop_reason
        self._lock = threading.Lock()

    def complete(self, request: Request) -> Completion:
        with self._lock:
            self.calls += 1
        return Completion(
            text=f"echo {request.prompt} #{request.n_sample}",
            model="m",
            usage=Usage(3, 4),
            cost_usd=0.5,
            latency_s=2.0,
            stop_reason=self.stop_reason,
            logprobs=(-0.5,),
        )


def test_cache_store_roundtrip_exact(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "dir" / "cache.sqlite"
    with CacheStore(path) as store:
        assert store.get("k") is None
        store.put("k", "b", FULL)
        hit = store.get("k")
        assert hit is not None
        assert hit.cached is True
        assert hit == Completion.from_dict({**FULL.to_dict(), "cached": True})
        assert len(store) == 1
    assert path.exists()
    with sqlite3.connect(path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        cols = [r[1] for r in conn.execute("PRAGMA table_info(completions)")]
    assert mode.lower() == "wal"
    assert cols == ["key", "backend", "completion_json", "created_at"]
    with CacheStore(path) as again:  # persists across instances
        assert again.get("k") is not None


def test_cached_backend_hits_and_keys(tmp_path: Path) -> None:
    inner = Counting()
    cb = CachedBackend(inner, CacheStore(tmp_path / "c.sqlite"))
    assert cb.name == "inner"
    first = cb.complete(Request(prompt="a", tags={"task_id": "1"}))
    assert first.cached is False
    second = cb.complete(Request(prompt="a", tags={"task_id": "2"}))  # tags not in key
    assert second.cached is True
    assert second.text == first.text and second.cost_usd == 0.5 and second.latency_s == 2.0
    cb.complete(Request(prompt="a", n_sample=1))  # n_sample is in the key
    assert inner.calls == 2
    other = CachedBackend(Counting(name="other"), cb.store)  # backend name is in the key
    assert other.complete(Request(prompt="a")).cached is False


def test_cached_backend_never_caches_errors() -> None:
    store = CacheStore(":memory:")
    err = Counting(stop_reason="error")
    cb = CachedBackend(err, store)
    cb.complete(Request(prompt="a"))
    cb.complete(Request(prompt="a"))
    assert err.calls == 2 and len(store) == 0

    class Failing:
        name = "f"

        def complete(self, request: Request) -> Completion:
            raise BackendError("down", backend="f")

    with pytest.raises(BackendError):
        CachedBackend(Failing(), store).complete(Request(prompt="a"))
    assert len(store) == 0


def test_cache_concurrent_threads(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "conc.sqlite")
    inner = Counting()
    cb = CachedBackend(inner, store)
    reqs = [Request(prompt=f"p{i % 50}") for i in range(800)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(cb.complete, reqs))
    assert len(store) == 50
    for req, res in zip(reqs, results, strict=True):
        assert res.text == f"echo {req.prompt} #0"
    # Each prompt is computed at least once; races may compute a few twice, never 800 times.
    assert 50 <= inner.calls < 800
    before = inner.calls
    with ThreadPoolExecutor(max_workers=8) as pool:
        again = list(pool.map(cb.complete, reqs))
    assert inner.calls == before
    assert all(r.cached for r in again)
    store.close()


class Configured(Counting):
    """Backend with public config attributes (like the HTTP/CLI backends) plus secrets."""

    def __init__(self, model: str = "m1", base_url: str = "http://a", **kw: object) -> None:
        super().__init__(name="cfg")
        self.model = model
        self.base_url = base_url
        self.pricing = kw.get("pricing")
        self.headers = kw.get("headers", {})
        self.env = kw.get("env")
        self.api_key_env = kw.get("api_key_env", "KEY")


def test_cache_key_includes_backend_config() -> None:
    store = CacheStore(":memory:")
    req = Request(prompt="a")
    assert CachedBackend(Configured(), store).complete(req).cached is False
    assert CachedBackend(Configured(), store).complete(req).cached is True
    # Same name, different model / endpoint / pricing: no stale hits.
    for changed in (
        Configured(model="m2"),
        Configured(base_url="http://b"),
        Configured(pricing=Pricing(1.0, 2.0)),
    ):
        assert CachedBackend(changed, store).complete(req).cached is False
    # Secret-bearing attributes are not part of the fingerprint.
    same = Configured(headers={"Authorization": "Bearer x"}, env={"T": "y"}, api_key_env="OTHER")
    assert CachedBackend(same, store).complete(req).cached is True


def test_cache_uses_custom_fingerprint_and_cacheable_flag() -> None:
    store = CacheStore(":memory:")

    class Fp(Counting):
        fp = "v1"

        def cache_fingerprint(self) -> str:
            return self.fp

    a = Fp()
    CachedBackend(a, store).complete(Request(prompt="a"))
    b = Fp()
    b.fp = "v2"
    assert CachedBackend(b, store).complete(Request(prompt="a")).cached is False

    class NoCache(Counting):
        cacheable = False

    inner = NoCache()
    cb = CachedBackend(inner, store)
    n = len(store)
    cb.complete(Request(prompt="z"))
    assert cb.complete(Request(prompt="z")).cached is False
    assert inner.calls == 2 and len(store) == n
