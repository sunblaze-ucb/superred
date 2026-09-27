"""Tests for ``run_all``: the unified multi-Controller sweep coordinator.

``run_all`` only touches a controller through ``.label`` and
``.run(reporter=...)``, so these use a lightweight duck-typed stub rather than
standing up full Controllers; the coordination logic (input-order results,
concurrency cap, reporter injection) is what is under test here.
"""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from anonframework.core import reporting
from anonframework.core.controller import run_all
from anonframework.core.reporting import Dashboard, NullReporter, ThreatModelContext


class _Tracker:
    def __init__(self) -> None:
        self.live = 0
        self.peak = 0


class _StubController:
    def __init__(self, label: str, result: int, tracker: _Tracker, delay: float = 0.0) -> None:
        self._label = label
        self._result = result
        self._tracker = tracker
        self._delay = delay
        self.reporter: Any = "unset"

    @property
    def label(self) -> str:
        return self._label

    @property
    def context(self) -> ThreatModelContext:
        return ThreatModelContext(
            label=self._label, attacker="atk", target="tgt", claim="clm", model="mdl", n_tasks=1
        )

    async def run(self, *, reporter: Any = None) -> int:
        self.reporter = reporter
        self._tracker.live += 1
        self._tracker.peak = max(self._tracker.peak, self._tracker.live)
        try:
            await asyncio.sleep(self._delay)
            return self._result
        finally:
            self._tracker.live -= 1


def test_run_all_returns_results_in_input_order() -> None:
    tracker = _Tracker()
    stubs = [_StubController(f"c{i}", i * 10, tracker) for i in range(5)]
    results = asyncio.run(run_all(stubs, report=False))
    assert results == [0, 10, 20, 30, 40]


def test_run_all_caps_concurrency() -> None:
    tracker = _Tracker()
    stubs = [_StubController(f"c{i}", i, tracker, delay=0.02) for i in range(6)]
    asyncio.run(run_all(stubs, concurrency=2, report=False))
    assert tracker.peak == 2  # never more than 2 threat models at once


def test_run_all_unbounded_runs_all_concurrently() -> None:
    tracker = _Tracker()
    stubs = [_StubController(f"c{i}", i, tracker, delay=0.02) for i in range(4)]
    asyncio.run(run_all(stubs, report=False))
    assert tracker.peak == 4  # default concurrency is "all of them"


def test_run_all_empty_is_noop() -> None:
    assert asyncio.run(run_all([], report=False)) == []


def test_run_all_report_false_injects_null_reporter() -> None:
    tracker = _Tracker()
    stubs = [_StubController("c0", 1, tracker)]
    asyncio.run(run_all(stubs, report=False))
    assert isinstance(stubs[0].reporter, NullReporter)


def test_run_all_off_tty_uses_plain_reporter(monkeypatch: pytest.MonkeyPatch) -> None:
    from anonframework.core.reporting import PlainReporter

    monkeypatch.setattr("anonframework.core.controller.should_use_plain", lambda *a, **k: True)
    tracker = _Tracker()
    stubs = [_StubController("c0", 1, tracker)]
    asyncio.run(run_all(stubs))  # report="auto", but forced off-TTY -> plain lines
    assert isinstance(stubs[0].reporter, PlainReporter)


def test_run_all_shares_one_dashboard_on_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    # On a TTY, run_all creates exactly ONE Dashboard for the whole sweep, tells
    # it to expect all n lanes, hands each controller a lane from it, and closes
    # it once at the end -- the coordination that keeps one clean canvas.
    reporting._reset_for_tests()
    created: list[Dashboard] = []

    def fake_dashboard() -> Dashboard:
        d = Dashboard(
            console=Console(file=StringIO(), force_terminal=True, width=140, color_system=None),
            redirect=False,
        )
        created.append(d)
        return d

    monkeypatch.setattr("anonframework.core.controller.should_use_plain", lambda *a, **k: False)
    monkeypatch.setattr("anonframework.core.controller.Dashboard", fake_dashboard)

    tracker = _Tracker()
    stubs = [_StubController(f"c{i}", i, tracker) for i in range(3)]
    asyncio.run(run_all(stubs, concurrency=2))  # report="auto" default

    assert len(created) == 1  # one dashboard, not one-per-controller
    assert created[0]._expected == 3  # told to keep the canvas alive for all 3
    assert created[0]._stopped is True  # closed once at the end
    assert all(type(s.reporter).__name__ == "_RichLane" for s in stubs)  # lanes, not plain
