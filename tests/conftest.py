"""Shared pytest configuration: tests marked ``live`` run only when SHADOWGATE_LIVE=1."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("SHADOWGATE_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test: set SHADOWGATE_LIVE=1 to call real providers")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
