"""ReplayBackend: serve recorded completions from a JSONL file (no model calls)."""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import BackendError, ConfigError
from ..types import Completion, Request
from .base import request_key

__all__ = ["ReplayBackend"]


class ReplayBackend:
    """Serve completions recorded earlier, for reproducible offline reruns.

    Each JSONL line is either ``{"key": <request_key>, "completion": {...}}`` or
    ``{"task_id": ..., "tier": ..., "completion": {...}}`` (``tier`` optional: a line without it
    matches any tier). Lookup order: request key, then ``(tags["task_id"], tags["tier"])``, then
    ``task_id`` alone. Keys are computed as ``request_key(key_backend, request)`` where
    ``key_backend`` is the name of the backend that recorded them (defaults to ``name``).
    A miss raises BackendError. Blank lines and lines starting with ``#`` are ignored.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        key_backend: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.name = name or f"replay:{self.path.stem}"
        self.key_backend = key_backend or self.name
        self._by_key: dict[str, Completion] = {}
        self._by_task: dict[tuple[str, str | None], Completion] = {}
        self._load()

    def _load(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ConfigError(f"replay file {str(self.path)!r} cannot be read: {exc}") from exc
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            where = f"{self.path}:{lineno}"
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{where}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj, dict) or not isinstance(obj.get("completion"), dict):
                raise ConfigError(f"{where}: expected an object with a 'completion' object")
            completion = Completion.from_dict(obj["completion"])
            if "key" in obj:
                self._by_key[str(obj["key"])] = completion
            elif "task_id" in obj:
                tier = obj.get("tier")
                self._by_task[(str(obj["task_id"]), None if tier is None else str(tier))] = (
                    completion
                )
            else:
                raise ConfigError(f"{where}: line needs 'key' or 'task_id'")

    def __len__(self) -> int:
        return len(self._by_key) + len(self._by_task)

    def complete(self, request: Request) -> Completion:
        hit = self._by_key.get(request_key(self.key_backend, request))
        if hit is None and request.tags:
            tid = request.tags.get("task_id")
            if tid is not None:
                hit = self._by_task.get((tid, request.tags.get("tier")))
                if hit is None:
                    hit = self._by_task.get((tid, None))
        if hit is None:
            raise BackendError(
                f"replay miss: no recorded completion for this request in {self.path}",
                backend=self.name,
            )
        return hit
