"""TOML configuration: parse, validate and build a :class:`~shadowgate.cascade.Cascade`.

A config file has five sections::

    [run]                 run settings (name, workers, budget, cache, ledger, seed)
    [backends.<id>]       model backends, referenced by id from tiers, monitors and judges
    [[tiers]]             cascade tiers, cheapest first; the last tier always serves
    [answer]              answer extractor and comparator (grading + agreement)
    [audit]               shadow-audit policy, tolerance and audit judge

Validation happens in :func:`parse_config` (structure, types, unknown keys with "did you mean"
suggestions, cross references) and every error names the TOML path of the offending value,
e.g. ``tiers[1].backend: unknown backend "slwo" (defined: fast, slow)``. Building objects is
deferred to :meth:`Config.build`; only backends that are actually referenced are built, each
exactly once, and shared between tiers, confidence monitors and judges.

Relative paths (``run.ledger``, ``run.cache``, a ``replay`` backend's ``path``, a ``command``
backend's ``cwd``) are resolved against the config file's directory. The defaults
(``.shadowgate/ledger.sqlite``; ``cache = true`` -> ``cache.sqlite`` next to the ledger) are
relative to the working directory, matching the CLI.

Secrets never live in config files: a literal ``api_key`` / ``token`` / ``secret`` /
``password`` key anywhere, or an ``Authorization``-style header, is rejected. Name an
environment variable with ``api_key_env`` instead. That is the only form of environment
lookup: there is no generic ``${VAR}`` expansion, so strings are always used verbatim.
"""

from __future__ import annotations

import copy
import datetime as _dt
import difflib
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import ConfigError

if TYPE_CHECKING:
    from .cascade import Cascade
    from .types import Backend, Task

__all__ = [
    "DEFAULT_LEDGER",
    "DEFAULT_CACHE_NAME",
    "RunSettings",
    "Config",
    "load_config",
    "parse_config",
]

DEFAULT_LEDGER = Path(".shadowgate") / "ledger.sqlite"
DEFAULT_CACHE_NAME = "cache.sqlite"

SECTIONS = ("run", "backends", "tiers", "answer", "audit")
RUN_KEYS = ("name", "workers", "max_cost_usd", "cache", "ledger", "seed")
TIER_KEYS = (
    "name",
    "backend",
    "threshold",
    "confidence",
    "system",
    "template",
    "max_tokens",
    "temperature",
    "effort",
    "extractor",
)
ANSWER_KEYS = ("extractor", "comparator")
AUDIT_KEYS = ("rate", "strata", "floor", "mode", "tier", "seed", "tolerance", "judge")
AUDIT_MODES = ("inline", "deferred", "off")

# Literal credential keys rejected anywhere in a config (compared case-insensitively).
SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "token",
        "access_token",
        "auth_token",
        "secret",
        "secret_key",
        "client_secret",
        "password",
    }
)
# Header names that carry credentials; rejected in a backend's ``headers`` table.
SECRET_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key", "x-auth-token"}
)
_REDACTED = "<redacted>"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROMPT_TOKEN = "{prompt}"


# --------------------------------------------------------------------------- data classes


@dataclass(frozen=True)
class RunSettings:
    """Settings from ``[run]`` (plus ``[audit].tolerance``) that the runner and CLI use."""

    name: str = "shadowgate"
    workers: int = 4
    max_cost_usd: float | None = None
    cache_path: Path | None = None
    ledger_path: Path = DEFAULT_LEDGER
    seed: int = 0
    tolerance: float | None = None


@dataclass(frozen=True)
class Config:
    """A validated configuration. ``raw`` is the parsed TOML; nothing is built until
    :meth:`build`, so loading a config never touches the network or the filesystem."""

    path: Path | None
    raw: Mapping[str, Any]
    settings: RunSettings
    base_dir: Path | None = field(default=None, compare=False)

    def build(self, *, tasks: Sequence[Task] | None = None) -> tuple[Cascade, RunSettings]:
        """Build the cascade. ``tasks`` is required when a referenced backend is ``simulated``
        (its answers derive from the task references)."""
        return _build(self, tasks), self.settings

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe deep copy of ``raw`` for the ledger, with credential-like values redacted."""
        return _snapshot(self.raw)


# --------------------------------------------------------------------------- entry points


def load_config(path: str | Path) -> Config:
    """Read and validate a TOML config file. Errors are :class:`ConfigError`."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {p}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read config file {p}: {exc.strerror or exc}") from None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{p}: config file is not valid UTF-8 ({exc.reason})") from None
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        # The message carries "(at line N, column M)".
        raise ConfigError(f"{p}: invalid TOML: {exc}") from None
    return parse_config(parsed, base_dir=p.resolve().parent, path=p)


def parse_config(
    data: Mapping[str, Any], *, base_dir: Path | None = None, path: str | Path | None = None
) -> Config:
    """Validate a parsed config mapping. Relative paths resolve against ``base_dir`` (kept
    relative to the working directory when None)."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"config must be a table, got {type(data).__name__}")
    p = Path(path) if path is not None else None
    if base_dir is None and p is not None:
        base_dir = p.resolve().parent
    base = Path(base_dir) if base_dir is not None else None
    raw = copy.deepcopy(dict(data))
    _check_secrets(raw, "")
    settings, _ = _validate(raw, base, p)
    return Config(path=p, raw=raw, settings=settings, base_dir=base)


# --------------------------------------------------------------------------- helpers


def _join(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else key


def _q(s: Any) -> str:
    return f'"{s}"'


def _suggest(key: str, allowed: Sequence[str]) -> str:
    match = difflib.get_close_matches(str(key), list(allowed), n=1, cutoff=0.6)
    return f' (did you mean "{match[0]}"?)' if match else ""


def _unknown(path: str, key: str, allowed: Sequence[str], where: str) -> ConfigError:
    return ConfigError(
        f"{_join(path, key)}: unknown key {_q(key)} in {where}{_suggest(key, allowed)}; "
        f"allowed: {', '.join(sorted(allowed))}"
    )


def _check_keys(table: Mapping[str, Any], allowed: Sequence[str], path: str, where: str) -> None:
    for key in table:
        if key not in allowed:
            raise _unknown(path, key, allowed, where)


def _table(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: must be a table, got {_type(value)}")
    return value


def _type(value: Any) -> str:
    names = {bool: "boolean", int: "integer", float: "float", str: "string", list: "array"}
    for t, n in names.items():
        if type(value) is t:
            return f"{n} {value!r}" if t in (bool, int, float, str) else n
    return "table" if isinstance(value, Mapping) else type(value).__name__


def _num(
    value: Any,
    path: str,
    *,
    lo: float | None = None,
    hi: float | None = None,
    lo_open: bool = False,
    hi_open: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        raise ConfigError(f"{path}: must be a number, got {_type(value)}")
    x = float(value)
    bad = (
        (lo is not None and (x <= lo if lo_open else x < lo))
        or (hi is not None and (x >= hi if hi_open else x > hi))
        or math.isinf(x)
    )
    if bad:
        left = "(" if lo_open else "["
        right = ")" if hi_open else "]"
        lo_s = "-inf" if lo is None else f"{lo:g}"
        hi_s = "inf" if hi is None else f"{hi:g}"
        raise ConfigError(f"{path}: must be in {left}{lo_s}, {hi_s}{right}, got {value!r}")
    return x


def _int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: must be an integer, got {_type(value)}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path}: must be an integer >= {minimum}, got {value}")
    return value


def _str(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path}: must be a string, got {_type(value)}")
    if nonempty and not value.strip():
        raise ConfigError(f"{path}: must be a non-empty string")
    return value


def _resolve(value: str, base: Path | None) -> Path:
    p = Path(value).expanduser()
    if base is not None and not p.is_absolute():
        p = base / p
    return p


# --------------------------------------------------------------------------- secrets


def _check_secrets(node: Any, path: str) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            sub = _join(path, str(key))
            if str(key).lower() in SECRET_KEYS:
                raise ConfigError(
                    f"{sub}: literal credentials are not allowed in config files; store the "
                    "value in an environment variable and set 'api_key_env' to its name"
                )
            if key == "headers" and isinstance(value, Mapping):
                for h in value:
                    if str(h).lower() in SECRET_HEADERS:
                        raise ConfigError(
                            f"{_join(sub, str(h))}: credential headers are not allowed in "
                            "config files; set 'api_key_env' to the name of an environment "
                            "variable holding the key"
                        )
            _check_secrets(value, sub)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _check_secrets(item, _join(path, i))


def _snapshot(node: Any) -> Any:
    if isinstance(node, Mapping):
        out: dict[str, Any] = {}
        for key, value in node.items():
            k = str(key)
            if k.lower() in SECRET_KEYS:
                out[k] = _REDACTED
            elif k == "headers" and isinstance(value, Mapping):
                out[k] = {
                    str(h): _REDACTED if str(h).lower() in SECRET_HEADERS else _snapshot(v)
                    for h, v in value.items()
                }
            else:
                out[k] = _snapshot(value)
        return out
    if isinstance(node, (list, tuple)):
        return [_snapshot(v) for v in node]
    if isinstance(node, (_dt.datetime, _dt.date, _dt.time)):
        return node.isoformat()
    if isinstance(node, Path):
        return str(node)
    if isinstance(node, float) and not math.isfinite(node):
        return str(node)
    if node is None or isinstance(node, (str, int, float, bool)):
        return node
    return str(node)


# --------------------------------------------------------------------------- factory key tables


def _backend_keys() -> tuple[Mapping[str, frozenset[str]], Mapping[str, tuple[str, ...]]]:
    from . import backends

    return getattr(backends, "_KEYS", {}), getattr(backends, "_REQUIRED", {})


def _extractor_keys() -> Mapping[str, frozenset[str]]:
    from . import extract

    types_ = getattr(extract, "_TYPES", {})
    return {k: v[1] for k, v in types_.items()}


def _comparator_keys() -> Mapping[str, frozenset[str]]:
    from . import compare

    keys = {k: v[1] for k, v in getattr(compare, "_SIMPLE", {}).items()}
    keys["judge"] = getattr(compare, "_JUDGE_KEYS", frozenset({"backend"}))
    return keys


def _confidence_keys() -> Mapping[str, frozenset[str]]:
    from . import confidence

    return getattr(confidence, "_ALLOWED_KEYS", {})


# --------------------------------------------------------------------------- validation


def _check_type(spec: Mapping[str, Any], path: str, kinds: Sequence[str], what: str) -> str:
    if "type" not in spec:
        raise ConfigError(
            f"{_join(path, 'type')}: missing; {what} type is one of {', '.join(kinds)}"
        )
    kind = spec["type"]
    if not isinstance(kind, str) or kind not in kinds:
        hint = _suggest(kind, kinds) if isinstance(kind, str) else ""
        raise ConfigError(
            f"{_join(path, 'type')}: unknown {what} type {_q(kind)}{hint}; expected one of "
            f"{', '.join(kinds)}"
        )
    return kind


def _check_backend_ref(value: Any, path: str, defined: Sequence[str], refs: list[str]) -> str:
    name = _str(value, path)
    if name not in defined:
        listed = ", ".join(defined) if defined else "none; add a [backends.<id>] table"
        raise ConfigError(
            f"{path}: unknown backend {_q(name)}{_suggest(name, defined)} (defined: {listed})"
        )
    if name not in refs:
        refs.append(name)
    return name


def _check_backend(bid: str, spec: Any, path: str) -> None:
    spec = _table(spec, path)
    keys, required = _backend_keys()
    kinds = tuple(keys) or ("anthropic", "openai", "claude-code", "command", "simulated", "replay")
    kind = _check_type(spec, path, kinds, "backend")
    if kind in keys:
        allowed = ("type", *sorted(keys[kind]))
        _check_keys(spec, allowed, path, f"a {kind!r} backend")
    for key in required.get(kind, ()):
        if spec.get(key) in (None, ""):
            raise ConfigError(f"{_join(path, key)}: required for backend type {kind!r}")
    if "api_key_env" in spec:
        env = _str(spec["api_key_env"], _join(path, "api_key_env"))
        if not _ENV_NAME.match(env):
            raise ConfigError(
                f"{_join(path, 'api_key_env')}: must be the NAME of an environment variable "
                "(letters, digits, underscores), not the key itself"
            )
    for key in ("model", "name", "base_url", "executable", "default_system", "cwd", "path"):
        if key in spec:
            _str(spec[key], _join(path, key))
    if "pricing" in spec:
        _table(spec["pricing"], _join(path, "pricing"))
    if "headers" in spec:
        headers = _table(spec["headers"], _join(path, "headers"))
        for h, v in headers.items():
            _str(v, _join(_join(path, "headers"), h), nonempty=False)


def _check_extractor(spec: Any, path: str) -> None:
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = _table(spec, path)
    keys = _extractor_keys()
    if not keys:
        return
    kind = _check_type(spec, path, tuple(keys), "extractor")
    _check_keys(spec, ("type", *sorted(keys[kind])), path, f"a {kind!r} extractor")


def _check_comparator(spec: Any, path: str, defined: Sequence[str], refs: list[str]) -> None:
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = _table(spec, path)
    keys = _comparator_keys()
    kind = _check_type(spec, path, tuple(keys), "comparator")
    _check_keys(spec, ("type", *sorted(keys[kind])), path, f"a {kind!r} comparator")
    if kind == "judge":
        if "backend" not in spec:
            raise ConfigError(f"{_join(path, 'backend')}: required for a 'judge' comparator")
        _check_backend_ref(spec["backend"], _join(path, "backend"), defined, refs)


def _check_confidence(spec: Any, path: str, defined: Sequence[str], refs: list[str]) -> None:
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = _table(spec, path)
    keys = dict(_confidence_keys())
    keys.pop("callable", None)  # needs a Python callable: API only
    kind = spec.get("type")
    if kind == "callable":
        raise ConfigError(
            f"{_join(path, 'type')}: 'callable' estimators wrap a Python function and can "
            "only be built through the API"
        )
    kind = _check_type(spec, path, tuple(keys), "confidence")
    _check_keys(spec, ("type", *sorted(keys[kind])), path, f"a {kind!r} estimator")
    if kind == "monitor":
        if "backend" not in spec:
            raise ConfigError(f"{_join(path, 'backend')}: required for a 'monitor' estimator")
        _check_backend_ref(spec["backend"], _join(path, "backend"), defined, refs)
    elif kind == "combine":
        members = spec.get("members")
        mpath = _join(path, "members")
        if not isinstance(members, list) or not members:
            raise ConfigError(f"{mpath}: must be a non-empty array of estimator tables")
        for i, member in enumerate(members):
            _check_confidence(member, _join(mpath, i), defined, refs)
    elif kind == "calibrated":
        if "base" not in spec:
            raise ConfigError(f"{_join(path, 'base')}: required (the estimator to calibrate)")
        _check_confidence(spec["base"], _join(path, "base"), defined, refs)
        points = spec.get("points")
        ppath = _join(path, "points")
        if not isinstance(points, list) or not points:
            raise ConfigError(f"{ppath}: must be a non-empty array of [x, y] pairs")
        for i, pt in enumerate(points):
            if not isinstance(pt, list) or len(pt) != 2:
                raise ConfigError(f"{_join(ppath, i)}: must be an [x, y] pair")
            _num(pt[0], _join(ppath, i), lo=0, hi=1)
            _num(pt[1], _join(ppath, i), lo=0, hi=1)


def _validate(
    raw: Mapping[str, Any], base: Path | None, path: Path | None
) -> tuple[RunSettings, list[str]]:
    """Validate everything; return the run settings and referenced backend ids (in order)."""
    _check_keys(raw, SECTIONS, "", "the top level")

    # [run]
    run = _table(raw.get("run", {}), "run")
    _check_keys(run, RUN_KEYS, "run", "[run]")
    name = (
        _str(run["name"], "run.name")
        if "name" in run
        else (path.stem if path is not None else "shadowgate")
    )
    workers = _int(run.get("workers", 4), "run.workers", minimum=1)
    max_cost = (
        _num(run["max_cost_usd"], "run.max_cost_usd", lo=0, lo_open=True)
        if run.get("max_cost_usd") is not None
        else None
    )
    seed = _int(run.get("seed", 0), "run.seed")
    ledger = (
        _resolve(_str(run["ledger"], "run.ledger"), base) if "ledger" in run else DEFAULT_LEDGER
    )
    cache_value = run.get("cache", False)
    if cache_value is True:
        cache: Path | None = ledger.parent / DEFAULT_CACHE_NAME
    elif cache_value is False:
        cache = None
    elif isinstance(cache_value, str):
        cache = _resolve(_str(cache_value, "run.cache"), base)
    else:
        raise ConfigError(
            f"run.cache: must be true, false or a file path, got {_type(cache_value)}"
        )

    # [backends.<id>]
    backends = _table(raw.get("backends", {}), "backends")
    defined = list(backends)
    for bid, spec in backends.items():
        if not str(bid).strip():
            raise ConfigError("backends: backend ids must be non-empty")
        _check_backend(bid, spec, _join("backends", bid))
    refs: list[str] = []

    # [answer]
    answer = _table(raw.get("answer", {}), "answer")
    _check_keys(answer, ANSWER_KEYS, "answer", "[answer]")
    if "extractor" in answer:
        _check_extractor(answer["extractor"], "answer.extractor")
    if "comparator" in answer:
        _check_comparator(answer["comparator"], "answer.comparator", defined, refs)

    # [[tiers]]
    if "tiers" not in raw:
        raise ConfigError("tiers: missing; add at least one [[tiers]] table")
    tiers = raw["tiers"]
    if not isinstance(tiers, list) or not tiers:
        raise ConfigError("tiers: must be a non-empty array of [[tiers]] tables")
    names: dict[str, int] = {}
    last = len(tiers) - 1
    for i, tier in enumerate(tiers):
        tp = _join("tiers", i)
        tier = _table(tier, tp)
        _check_keys(tier, TIER_KEYS, tp, "[[tiers]]")
        if "backend" not in tier:
            raise ConfigError(
                f"{_join(tp, 'backend')}: missing; name one of the [backends] "
                f"ids ({', '.join(defined) or 'none defined'})"
            )
        bname = _check_backend_ref(tier["backend"], _join(tp, "backend"), defined, refs)
        tname = _str(tier["name"], _join(tp, "name")) if "name" in tier else bname
        if tname in names:
            raise ConfigError(
                f"{_join(tp, 'name')}: duplicate tier name {_q(tname)} (also tiers[{names[tname]}]"
                "); give each tier a distinct 'name'"
            )
        names[tname] = i
        if i < last:
            if "threshold" not in tier:
                raise ConfigError(
                    f"{_join(tp, 'threshold')}: missing; every tier except the last needs a "
                    "confidence threshold in [0, 1]"
                )
            if "confidence" not in tier:
                raise ConfigError(
                    f"{_join(tp, 'confidence')}: missing; every tier except the last needs a "
                    'confidence estimator, e.g. confidence = { type = "verbal" }'
                )
        else:
            if "threshold" in tier:
                raise ConfigError(
                    f"{_join(tp, 'threshold')}: final tier never escalates; remove threshold"
                )
            if "confidence" in tier:
                raise ConfigError(
                    f"{_join(tp, 'confidence')}: the final tier never escalates, so its "
                    "confidence is never used; remove it"
                )
        if "threshold" in tier:
            _num(tier["threshold"], _join(tp, "threshold"), lo=0, hi=1)
        if "confidence" in tier:
            _check_confidence(tier["confidence"], _join(tp, "confidence"), defined, refs)
        if "system" in tier:
            _str(tier["system"], _join(tp, "system"), nonempty=False)
        if "template" in tier:
            template = _str(tier["template"], _join(tp, "template"))
            if _PROMPT_TOKEN not in template:
                raise ConfigError(f"{_join(tp, 'template')}: must contain {_PROMPT_TOKEN}")
        if "max_tokens" in tier:
            _int(tier["max_tokens"], _join(tp, "max_tokens"), minimum=1)
        if "temperature" in tier:
            _num(tier["temperature"], _join(tp, "temperature"), lo=0)
        if "effort" in tier:
            _str(tier["effort"], _join(tp, "effort"))
        if "extractor" in tier:
            _check_extractor(tier["extractor"], _join(tp, "extractor"))

    # [audit]
    audit = _table(raw.get("audit", {}), "audit")
    _check_keys(audit, AUDIT_KEYS, "audit", "[audit]")
    if "rate" in audit:
        _num(audit["rate"], "audit.rate", lo=0, hi=1, lo_open=True)
    if "floor" in audit:
        _num(audit["floor"], "audit.floor", lo=0, hi=1, lo_open=True)
    if "mode" in audit:
        mode = _str(audit["mode"], "audit.mode")
        if mode not in AUDIT_MODES:
            raise ConfigError(
                f"audit.mode: must be one of {', '.join(AUDIT_MODES)}, got {_q(mode)}"
                f"{_suggest(mode, AUDIT_MODES)}"
            )
    if "tier" in audit:
        tier_name = _str(audit["tier"], "audit.tier")
        if tier_name not in names:
            raise ConfigError(
                f"audit.tier: unknown tier {_q(tier_name)}{_suggest(tier_name, list(names))} "
                f"(tiers: {', '.join(names)})"
            )
    if "seed" in audit:
        _int(audit["seed"], "audit.seed")
    tolerance = (
        _num(audit["tolerance"], "audit.tolerance", lo=0, hi=1, lo_open=True, hi_open=True)
        if audit.get("tolerance") is not None
        else None
    )
    if "strata" in audit:
        strata = audit["strata"]
        if not isinstance(strata, list):
            raise ConfigError("audit.strata: must be an array of [lo, hi, rate] triples")
        for i, s in enumerate(strata):
            sp = _join("audit.strata", i)
            if not isinstance(s, list) or len(s) != 3:
                raise ConfigError(f"{sp}: must be a [lo, hi, rate] triple, got {_type(s)}")
            lo = _num(s[0], f"{sp}[0] (lo)", lo=0, hi=1)
            hi = _num(s[1], f"{sp}[1] (hi)", lo=0, hi=1)
            if lo >= hi:
                raise ConfigError(f"{sp}: lo must be < hi, got {s!r}")
            _num(s[2], f"{sp}[2] (rate)", lo=0, hi=1, lo_open=True)
    if "judge" in audit:
        _check_comparator(audit["judge"], "audit.judge", defined, refs)

    settings = RunSettings(
        name=name,
        workers=workers,
        max_cost_usd=max_cost,
        cache_path=cache,
        ledger_path=ledger,
        seed=seed,
        tolerance=tolerance,
    )
    return settings, refs


# --------------------------------------------------------------------------- build


def _wrap(path: str, exc: ConfigError) -> ConfigError:
    msg = str(exc)
    return ConfigError(msg if msg.startswith(path) else f"{path}: {msg}")


def _shorthand(spec: Any) -> Any:
    return {"type": spec} if isinstance(spec, str) else spec


def _backend_spec(cfg: Config, bid: str) -> dict[str, Any]:
    spec = dict(cfg.raw["backends"][bid])
    kind = spec["type"]
    if kind == "replay" and "path" in spec:
        spec["path"] = str(_resolve(spec["path"], cfg.base_dir))
    if kind == "command" and "cwd" in spec:
        spec["cwd"] = str(_resolve(spec["cwd"], cfg.base_dir))
    if kind == "simulated":
        if "name" not in spec and "model" not in spec:
            spec["name"] = bid
        spec.setdefault("seed", cfg.settings.seed)
    return spec


def _build(cfg: Config, tasks: Sequence[Task] | None) -> Cascade:
    from . import compare, confidence, extract
    from .backends import make_backend
    from .cascade import AuditPolicy, Cascade, Tier

    raw = cfg.raw
    settings, refs = _validate(raw, cfg.base_dir, cfg.path)
    specs = raw.get("backends", {})
    simulated = [b for b in refs if specs[b].get("type") == "simulated"]
    task_list = list(tasks) if tasks is not None else None
    if simulated and task_list is None:
        raise ConfigError(
            f"backends.{simulated[0]}: simulated backends derive their answers from the task "
            "list; pass it with Config.build(tasks=...) (the CLI does this with -t/--tasks)"
        )

    store = None
    if cfg.settings.cache_path is not None:
        from .backends.cache import CacheStore

        store = CacheStore(cfg.settings.cache_path)

    built: dict[str, Backend] = {}
    for bid in refs:
        bpath = _join("backends", bid)
        try:
            built[bid] = make_backend(_backend_spec(cfg, bid), cache=store, tasks=task_list)
        except ConfigError as exc:
            raise _wrap(bpath, exc) from exc

    answer = raw.get("answer", {})
    try:
        default_extractor = extract.from_spec(_shorthand(answer.get("extractor")))
    except ConfigError as exc:
        raise _wrap("answer.extractor", exc) from exc
    try:
        comparator = compare.from_spec(_shorthand(answer.get("comparator")), backends=built)
    except ConfigError as exc:
        raise _wrap("answer.comparator", exc) from exc

    tiers: list[Tier] = []
    for i, t in enumerate(raw["tiers"]):
        tp = _join("tiers", i)
        tier_extractor = None
        if "extractor" in t:
            try:
                tier_extractor = extract.from_spec(_shorthand(t["extractor"]))
            except ConfigError as exc:
                raise _wrap(_join(tp, "extractor"), exc) from exc
        estimator = None
        if "confidence" in t:
            try:
                estimator = confidence.from_spec(
                    _shorthand(t["confidence"]),
                    backends=built,
                    comparator=comparator,
                    extractor=tier_extractor or default_extractor,
                )
            except (ConfigError, TypeError, ValueError) as exc:
                raise _wrap(_join(tp, "confidence"), ConfigError(str(exc))) from exc
        kwargs: dict[str, Any] = {}
        for key in ("system", "template", "max_tokens", "temperature", "effort"):
            if key in t:
                kwargs[key] = t[key]
        if "threshold" in t:
            kwargs["threshold"] = float(t["threshold"])
        try:
            tiers.append(
                Tier(
                    name=t.get("name", t["backend"]),
                    backend=built[t["backend"]],
                    estimator=estimator,
                    extractor=tier_extractor,
                    **kwargs,
                )
            )
        except ConfigError as exc:
            raise _wrap(tp, exc) from exc

    a = raw.get("audit", {})
    judge = None
    if "judge" in a:
        try:
            judge = compare.from_spec(_shorthand(a["judge"]), backends=built)
        except ConfigError as exc:
            raise _wrap("audit.judge", exc) from exc
    policy_kwargs: dict[str, Any] = {"seed": a.get("seed", settings.seed)}
    for key in ("rate", "floor", "mode"):
        if key in a:
            policy_kwargs[key] = a[key]
    if "tier" in a:
        policy_kwargs["audit_tier"] = a["tier"]
    if "strata" in a:
        policy_kwargs["strata"] = tuple(tuple(float(x) for x in s) for s in a["strata"])
    try:
        policy = AuditPolicy(**policy_kwargs)
    except ConfigError as exc:
        raise _wrap("audit", exc) from exc

    try:
        return Cascade(
            tiers, extractor=default_extractor, comparator=comparator, audit=policy, judge=judge
        )
    except ConfigError as exc:
        raise _wrap("tiers", exc) from exc
