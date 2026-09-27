"""Dashboard + reporter-resolution coverage for ``anonframework.core.reporting``.

Complements ``tests/test_reporting.py`` (which drives the observer contract
through a real Controller and a single-lane canvas) by exercising the rich
:class:`Dashboard` internals that a single-lane happy path never reaches:

* a multi-lane canvas driven through every ``stop_reason`` + skip, then torn
  down cleanly (the last live frame is the summary);
* per-threat-model active-task rows (running tasks listed beneath a lane);
* a second Dashboard degrading to :class:`PlainReporter` when the single live
  canvas is already held;
* the context-manager force-stop, the ``_bar`` / ``_fmt_dur`` formatting edges,
  the ``scope_desc`` variants, the ``PlainReporter`` skip/silent branches,
  ``should_use_plain`` env + isatty edges, the ``LoggingBridge`` info level, and
  ``resolve_reporter`` / ``get_default_dashboard`` resolution.

Every Dashboard test calls ``reporting._reset_for_tests()`` first and drives
lanes inside ``asyncio.run`` so the dashboard binds the running loop.
"""

from __future__ import annotations

import asyncio
import logging
from io import StringIO
from typing import Any

from rich.console import Console

from anonframework.core import reporting
from anonframework.core.reporting import (
    _REFRESH_INTERVAL_S,
    Dashboard,
    DiagnosticEvent,
    LoggingBridge,
    NullReporter,
    PlainReporter,
    RunCompleteEvent,
    TaskCompleteEvent,
    TaskSkippedEvent,
    TaskStartEvent,
    ThreatModelContext,
    ThreatModelEndEvent,
    should_use_plain,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _ctx(label: str = "exp", **overrides: Any) -> ThreatModelContext:
    base: dict[str, Any] = {
        "label": label,
        "attacker": "atk",
        "target": "tgt",
        "claim": "clm",
        "model": "test-model",
        "scope": ("external",),
        "n_tasks": 1,
    }
    base.update(overrides)
    return ThreatModelContext(**base)


def _run_complete(
    task_index: int = 1, run_number: int = 1, success: bool = True
) -> RunCompleteEvent:
    return RunCompleteEvent(
        task_index=task_index,
        goal="g",
        run_number=run_number,
        primary_score=1.0 if success else 0.0,
        success=success,
        evaluated=True,
        errored=not success,
        done=success,
        run_cost_delta_usd=0.001,
        cumulative_cost_usd=0.001 * run_number,
    )


def _task_complete(
    task_index: int,
    success: bool,
    stop_reason: str,
    *,
    best_score: float = 1.0,
    error: str | None = None,
) -> TaskCompleteEvent:
    return TaskCompleteEvent(
        task_index=task_index,
        goal=f"g{task_index}",
        success=success,
        stop_reason=stop_reason,
        best_score=best_score,
        n_runs=1,
        cost_usd=0.01,
        calls=2,
        error=error,
    )


def _end(ctx: ThreatModelContext, **overrides: Any) -> ThreatModelEndEvent:
    base: dict[str, Any] = {
        "context": ctx,
        "n_tasks": 1,
        "n_success": 1,
        "n_completed": 1,
        "n_error": 0,
        "n_budget_exhausted": 0,
        "n_skipped": 0,
        "asr": 1.0,
        "max_primary_score": 1.0,
        "mean_primary_score": 1.0,
        "total_calls": 2,
        "total_cost_usd": 0.01,
        "duration_s": 0.1,
    }
    base.update(overrides)
    return ThreatModelEndEvent(**base)


class _FakeTTY:
    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        return len(s)


class _DiagCapture:
    """Minimal reporter that records only diagnostics."""

    def __init__(self) -> None:
        self.diags: list[DiagnosticEvent] = []

    def on_threat_model_start(self, ctx: ThreatModelContext) -> None: ...
    def on_task_start(self, ev: TaskStartEvent) -> None: ...
    def on_run_complete(self, ev: RunCompleteEvent) -> None: ...
    def on_task_complete(self, ev: TaskCompleteEvent) -> None: ...
    def on_task_skipped(self, ev: TaskSkippedEvent) -> None: ...
    def on_diagnostic(self, ev: DiagnosticEvent) -> None:
        self.diags.append(ev)

    def on_threat_model_end(self, ev: ThreatModelEndEvent) -> None: ...


# ---------------------------------------------------------------------------
# (a) Multi-lane canvas through every stop_reason + skip -> final table
# ---------------------------------------------------------------------------


def test_dashboard_two_lanes_all_stop_reasons() -> None:
    reporting._reset_for_tests()
    sink = StringIO()
    console = Console(file=sink, force_terminal=True, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)

    async def drive() -> None:
        lane1 = dashboard.reporter_for("lane-1")
        lane2 = dashboard.reporter_for("lane-2")
        # First start acquires the canvas; the second re-enters _ensure_started
        # (already started -> early return) and just adds a second lane.
        lane1.on_threat_model_start(_ctx("lane-1", n_tasks=5))
        lane2.on_threat_model_start(_ctx("lane-2", n_tasks=1))
        assert dashboard._canvas_ok is True
        assert len(dashboard._lanes) == 2  # two threat-model blocks

        # lane-1: one task per stop_reason (done/max_runs/budget/error) + a skip.
        for i, (stop_reason, ok) in enumerate(
            [("done", True), ("max_runs", False), ("budget_exhausted", False), ("error", False)],
            start=1,
        ):
            lane1.on_task_start(TaskStartEvent(task_index=i, goal=f"g{i}"))
            lane1.on_run_complete(_run_complete(task_index=i, success=ok))
            lane1.on_task_complete(
                _task_complete(i, ok, stop_reason, error="boom" if stop_reason == "error" else None)
            )
        lane1.on_task_skipped(TaskSkippedEvent(task_index=5, goal="skip me"))
        # on_diagnostic is a safe no-op on the canvas now (there is no pane).
        lane1.on_diagnostic(
            DiagnosticEvent(label="lane-1", task_index=1, level="warning", message="heads up")
        )

        # lane-2: a single success.
        lane2.on_task_start(TaskStartEvent(task_index=1, goal="solo"))
        lane2.on_run_complete(_run_complete())
        lane2.on_task_complete(_task_complete(1, True, "done"))

        # Let a trailing-edge refresh flush actually paint (body/side/header).
        await asyncio.sleep(_REFRESH_INTERVAL_S * 2.5)

        # End lane-1 while lane-2 is still active (active-lanes>0 -> refresh),
        # then end lane-2 (last lane -> shutdown + final summary print).
        lane1.on_threat_model_end(
            _end(
                _ctx("lane-1", n_tasks=5),
                n_tasks=4,
                n_success=1,
                n_completed=3,
                n_error=1,
                n_budget_exhausted=1,
                n_skipped=1,
                asr=1 / 3,
                max_primary_score=1.0,
                mean_primary_score=0.25,
            )
        )
        assert dashboard._stopped is False  # lane-2 keeps it open
        lane2.on_threat_model_end(_end(_ctx("lane-2", n_tasks=1)))

    asyncio.run(drive())

    assert dashboard._stopped is True
    assert reporting._ACTIVE_LIVE is None
    assert "anonframework" in sink.getvalue()
    # The last frame (both threat models ✓) is itself the summary; there is no
    # separate final-results table. Render it at a controlled width to assert.
    frame = _render_to_str(dashboard)
    assert "atk" in frame and "tgt" in frame  # per-threat-model identity is shown
    assert "clm" in frame  # the security claim renders on the identity line
    assert "ASR" in frame and "attacker $" in frame  # the metrics columns
    reporting._reset_for_tests()


# ---------------------------------------------------------------------------
# (b) Active tasks render indented beneath their threat model
# ---------------------------------------------------------------------------


def _render_to_str(dashboard: Dashboard) -> str:
    cap = Console(file=StringIO(), width=140, color_system=None)
    cap.print(dashboard._render())
    return cap.file.getvalue()


def test_dashboard_identity_shows_budget_and_target_model_dot() -> None:
    reporting._reset_for_tests()
    console = Console(file=StringIO(), force_terminal=True, width=170, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)
    captured: dict[str, str] = {}

    async def drive() -> None:
        lane = dashboard.reporter_for("tm")
        lane.on_threat_model_start(_ctx(model="mdl", task_cost_cap_usd=0.66))
        captured["frame"] = _render_to_str(dashboard)

    asyncio.run(drive())
    frame = captured["frame"]
    assert "tgt · clm · mdl" in frame  # claim comes before model, dot-separated
    assert "$0.66/task" in frame  # the per-task budget shows on the identity line


def test_dashboard_expect_keeps_canvas_alive_across_zero_active() -> None:
    # expect(n) makes a sweep of n lanes share ONE canvas: it stays alive across
    # the moments no lane is active (which a capped sweep hits between waves),
    # instead of stopping and forcing later lanes to a degraded plain reporter.
    reporting._reset_for_tests()
    console = Console(file=StringIO(), force_terminal=True, width=140, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)
    dashboard.expect(2)

    async def drive() -> None:
        first = dashboard.reporter_for("a")
        first.on_threat_model_start(_ctx("a"))
        first.on_threat_model_end(_end(_ctx("a")))
        # 1 of 2 expected done: active is 0 but the canvas must NOT tear down.
        assert dashboard._stopped is False
        assert dashboard._canvas_ok is True
        second = dashboard.reporter_for("b")
        second.on_threat_model_start(_ctx("b"))
        assert dashboard._canvas_ok is True  # still the live canvas, not degraded
        second.on_threat_model_end(_end(_ctx("b")))
        assert dashboard._stopped is True  # both expected done -> final frame + stop

    asyncio.run(drive())


def test_dashboard_preregister_shows_queued_then_running() -> None:
    # A pre-registered lane shows up dimmed ("queued") before its Controller
    # starts; when it starts, the SAME row swaps to the running one in place.
    reporting._reset_for_tests()
    console = Console(file=StringIO(), force_terminal=True, width=160, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)

    async def drive() -> None:
        dashboard.preregister("a", _ctx("a", attacker="atk-x", n_tasks=5))
        frame = _render_to_str(dashboard)
        assert "queued" in frame and "atk-x" in frame  # visible before it runs
        assert dashboard._lanes["a"].pending is True
        assert dashboard._active_lanes == 0  # queued does not count as running

        lane = dashboard.reporter_for("a")
        lane.on_threat_model_start(_ctx("a", attacker="atk-x", n_tasks=5))
        assert dashboard._lanes["a"].pending is False  # same lane, now running
        assert dashboard._active_lanes == 1
        running_frame = _render_to_str(dashboard)
        assert "queued" not in running_frame
        assert "0/5" in running_frame  # the running row shows real progress

    asyncio.run(drive())


def test_dashboard_shows_active_tasks_under_each_lane() -> None:
    reporting._reset_for_tests()
    console = Console(file=StringIO(), force_terminal=True, width=140, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)
    captured: dict[str, str] = {}

    async def drive() -> None:
        lane = dashboard.reporter_for("lane")
        lane.on_threat_model_start(_ctx("lane", n_tasks=4, max_runs_per_task=8))
        # Two tasks running (started, not completed) + one already done.
        lane.on_task_start(TaskStartEvent(task_index=1, goal="Leak the secret [x]"))
        lane.on_run_complete(_run_complete(task_index=1, run_number=3, success=False))
        lane.on_task_start(TaskStartEvent(task_index=2, goal="Exfiltrate data"))
        # Task 3 starts then completes (every completed task was started first).
        lane.on_task_start(TaskStartEvent(task_index=3, goal="already breached"))
        lane.on_task_complete(_task_complete(3, True, "done"))
        await asyncio.sleep(_REFRESH_INTERVAL_S * 2.5)
        captured["body"] = _render_to_str(dashboard)
        lane.on_threat_model_end(_end(_ctx("lane", n_tasks=4)))

    asyncio.run(drive())
    body = captured["body"]
    assert "Leak the secret [x]" in body  # untrusted markup rendered literally
    assert "Exfiltrate data" in body  # both running tasks are listed
    assert "run 3/8" in body  # live per-task status
    assert "running 2" in body  # two tasks in-flight
    reporting._reset_for_tests()


# ---------------------------------------------------------------------------
# (c) Second Dashboard degrades to PlainReporter while the canvas is held
# ---------------------------------------------------------------------------


def test_second_dashboard_degrades_to_plain() -> None:
    reporting._reset_for_tests()
    sink1 = StringIO()
    sink2 = StringIO()
    console1 = Console(file=sink1, force_terminal=True, color_system=None)
    console2 = Console(file=sink2, force_terminal=True, color_system=None)
    d1 = Dashboard(console=console1, redirect=False)
    d2 = Dashboard(console=console2, redirect=False)

    async def drive() -> None:
        lane1 = d1.reporter_for("first")
        lane1.on_threat_model_start(_ctx("first"))
        assert d1._canvas_ok is True

        # d2 cannot get the single live canvas -> its lane delegates to plain.
        lane2 = d2.reporter_for("second")
        lane2.on_threat_model_start(_ctx("second", n_tasks=2))
        assert d2._canvas_ok is False
        lane2.on_task_start(TaskStartEvent(task_index=1, goal="g"))
        lane2.on_run_complete(_run_complete())
        lane2.on_task_complete(_task_complete(1, True, "done"))
        lane2.on_task_skipped(TaskSkippedEvent(task_index=2, goal="skip via plain"))
        lane2.on_diagnostic(
            DiagnosticEvent(label="second", task_index=1, level="error", message="plain diag")
        )
        lane2.on_threat_model_end(_end(_ctx("second", n_tasks=2)))

        lane1.on_threat_model_end(_end(_ctx("first")))

    asyncio.run(drive())

    # d2 produced plain, line-oriented output on its own console.
    plain = sink2.getvalue()
    assert "Threat model:" in plain  # plain banner
    assert "[SKIP] skip via plain" in plain  # plain skip line
    assert "Overall:" in plain  # plain end summary
    assert d1._stopped is True
    assert reporting._ACTIVE_LIVE is None
    reporting._reset_for_tests()


# ---------------------------------------------------------------------------
# (d) Context-manager force-stop
# ---------------------------------------------------------------------------


def test_dashboard_context_manager_force_stops() -> None:
    reporting._reset_for_tests()
    sink = StringIO()
    console = Console(file=sink, force_terminal=True, color_system=None)
    captured: dict[str, Dashboard] = {}

    async def drive() -> None:
        with Dashboard(console=console, redirect=False) as dashboard:
            captured["d"] = dashboard
            lane = dashboard.reporter_for("cm")
            lane.on_threat_model_start(_ctx("cm"))
            lane.on_task_start(TaskStartEvent(task_index=1, goal="g"))
            assert dashboard._canvas_ok is True
            # Leave the block WITHOUT ending the lane -> __exit__ -> _force_stop.

    asyncio.run(drive())
    # __exit__ -> _force_stop stops the Live (releases the canvas) without
    # marking a full shutdown (_stopped): the two are deliberately distinct so a
    # render error in _shutdown cannot disarm this fallback.
    assert captured["d"]._live_stopped is True
    assert reporting._ACTIVE_LIVE is None
    reporting._reset_for_tests()


# ---------------------------------------------------------------------------
# (a') Defensive lane-lookup branches (unknown label is a no-op)
# ---------------------------------------------------------------------------


def test_dashboard_unknown_label_calls_are_noops() -> None:
    reporting._reset_for_tests()
    sink = StringIO()
    console = Console(file=sink, force_terminal=True, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)

    async def drive() -> None:
        real = dashboard.reporter_for("real")
        real.on_threat_model_start(_ctx("real"))
        # Internal updates for a label with no lane must not raise.
        dashboard._task_start("ghost", TaskStartEvent(task_index=1, goal="g"))
        dashboard._run_complete("ghost", _run_complete())
        dashboard._task_complete("ghost", _task_complete(1, True, "done"))
        dashboard._task_skipped("ghost", TaskSkippedEvent(task_index=1, goal="g"))
        # _lane_end for an unknown label: no lane to stamp, but active-lane
        # bookkeeping still drives the last-lane shutdown.
        dashboard._lane_end("ghost", _end(_ctx("ghost")))
        assert dashboard._stopped is True
        # A second shutdown (via the real lane) is a no-op.
        real.on_threat_model_end(_end(_ctx("real")))

    asyncio.run(drive())
    assert dashboard._stopped is True
    reporting._reset_for_tests()


# ---------------------------------------------------------------------------
# No running loop -> plain fallback + idempotent force-stop
# ---------------------------------------------------------------------------


def test_dashboard_without_running_loop_degrades_to_plain() -> None:
    reporting._reset_for_tests()
    sink = StringIO()
    console = Console(file=sink, force_terminal=True, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)
    lane = dashboard.reporter_for("no-loop")

    # Called outside any event loop: _ensure_started cannot bind one, so the
    # canvas is never acquired and the lane degrades to plain output.
    lane.on_threat_model_start(_ctx("no-loop"))
    assert dashboard._loop is None
    assert dashboard._canvas_ok is False
    lane.on_task_complete(_task_complete(1, True, "done"))
    lane.on_threat_model_end(_end(_ctx("no-loop")))

    out = sink.getvalue()
    assert "Threat model:" in out
    assert "Overall:" in out

    # Force-stop with no live canvas is a safe, idempotent no-op (there is no
    # Live to release, so the shared canvas stays unheld).
    dashboard._force_stop()
    dashboard._force_stop()
    assert reporting._ACTIVE_LIVE is None
    reporting._reset_for_tests()


# ---------------------------------------------------------------------------
# (e) _bar / _fmt_dur formatting edges
# ---------------------------------------------------------------------------


def test_bar_clamps_and_fills() -> None:
    assert reporting._bar(0.0) == "░" * 12
    assert reporting._bar(1.0) == "█" * 12
    assert reporting._bar(1.5) == "█" * 12  # clamped above 1
    assert reporting._bar(-0.5) == "░" * 12  # clamped below 0
    half = reporting._bar(0.5)
    assert len(half) == 12
    assert half.count("█") == 6


def test_fmt_dur_seconds_minutes_hours() -> None:
    assert reporting._fmt_dur(5) == "5s"
    assert reporting._fmt_dur(59) == "59s"
    assert reporting._fmt_dur(90) == "1m30s"
    assert reporting._fmt_dur(3700) == "1h01m"


# ---------------------------------------------------------------------------
# scope_desc variants
# ---------------------------------------------------------------------------


def test_scope_desc_variants() -> None:
    assert _ctx("l", scope_label="dyn").scope_desc == "dyn (per-task)"
    assert (
        _ctx("l", scope=("external",), read_only=("internal",)).scope_desc
        == "[external] read_only=[internal]"
    )
    assert _ctx("l", scope=("external",)).scope_desc == "[external]"
    assert _ctx("l", scope=()).scope_desc == "[(none)]"


# ---------------------------------------------------------------------------
# PlainReporter skip / silent branches
# ---------------------------------------------------------------------------


def test_plain_reporter_edge_branches() -> None:
    out = StringIO()
    err = StringIO()
    reporter = PlainReporter(file=out, err=err)

    # on_task_start is a deliberate no-op.
    reporter.on_task_start(TaskStartEvent(task_index=1, goal="g"))
    assert out.getvalue() == ""

    # A non-success completion renders [FAIL], not [OK].
    reporter.on_task_complete(
        _task_complete(1, success=False, stop_reason="max_runs", best_score=0.0)
    )
    assert "[FAIL]" in out.getvalue()

    # A non-error diagnostic is silent on both streams.
    reporter.on_diagnostic(
        DiagnosticEvent(label="l", task_index=1, level="info", message="quiet please")
    )
    assert "quiet please" not in out.getvalue()
    assert "quiet please" not in err.getvalue()

    # An end summary with no errors/budget/skips omits the secondary counts line.
    out2 = StringIO()
    PlainReporter(file=out2).on_threat_model_end(
        _end(
            _ctx("l"),
            n_error=0,
            n_budget_exhausted=0,
            n_skipped=0,
            asr=None,
            max_primary_score=None,
        )
    )
    text = out2.getvalue()
    assert "Overall:" in text
    assert "Errors:" not in text
    assert "ASR n/a" in text  # asr None branch
    assert "Highest score: n/a" in text  # max None branch


# ---------------------------------------------------------------------------
# should_use_plain env + isatty edges
# ---------------------------------------------------------------------------


def test_should_use_plain_no_color_plus_dumb_terminal() -> None:
    tty = _FakeTTY()
    # NO_COLOR alone keeps the canvas; combined with a dumb terminal forces plain.
    assert should_use_plain(tty, env={"NO_COLOR": "1"}) is False
    assert should_use_plain(tty, env={"NO_COLOR": "1", "TERM": "dumb"}) is True


def test_should_use_plain_swallows_isatty_error() -> None:
    class _BadStream:
        def isatty(self) -> bool:
            raise ValueError("isatty exploded")

    # A stream whose isatty() raises is treated as non-TTY -> plain.
    assert should_use_plain(_BadStream(), env={}) is True


# ---------------------------------------------------------------------------
# LoggingBridge info level + sink
# ---------------------------------------------------------------------------


def _record(msg: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(
        name="anonframework.test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_logging_bridge_info_level_and_sink_and_filter() -> None:
    reporter = _DiagCapture()
    sink_calls: list[DiagnosticEvent] = []
    bridge = LoggingBridge("lbl", reporter, sink=sink_calls.append)

    token = reporting.current_task.set(("lbl", 3))
    try:
        bridge.emit(_record("info level", level=logging.INFO))
    finally:
        reporting.current_task.reset(token)

    assert len(reporter.diags) == 1
    assert reporter.diags[0].level == "info"  # below WARNING -> "info"
    assert reporter.diags[0].task_index == 3
    assert sink_calls and sink_calls[0].message == "info level"

    # A record attributed to a different controller is dropped.
    other = reporting.current_task.set(("someone-else", 1))
    try:
        bridge.emit(_record("not mine", level=logging.INFO))
    finally:
        reporting.current_task.reset(other)
    assert len(reporter.diags) == 1


# ---------------------------------------------------------------------------
# resolve_reporter / get_default_dashboard
# ---------------------------------------------------------------------------


def test_get_default_dashboard_is_singleton() -> None:
    reporting._reset_for_tests()
    d1 = reporting.get_default_dashboard()
    d2 = reporting.get_default_dashboard()
    assert d1 is d2
    assert isinstance(d1, Dashboard)
    reporting._reset_for_tests()


def test_resolve_reporter_all_branches(monkeypatch: Any) -> None:
    reporting._reset_for_tests()

    # report=False disables reporting.
    assert isinstance(reporting.resolve_reporter("l", report=False), NullReporter)

    # Plain environment -> PlainReporter.
    monkeypatch.setattr(reporting, "should_use_plain", lambda stream=None: True)
    assert isinstance(reporting.resolve_reporter("l"), PlainReporter)

    # Interactive terminal -> a Dashboard lane.
    monkeypatch.setattr(reporting, "should_use_plain", lambda stream=None: False)
    lane = reporting.resolve_reporter("l")
    assert isinstance(lane, reporting._RichLane)

    reporting._reset_for_tests()


# ---------------------------------------------------------------------------
# (k) A render bug never crashes the run and never strands the terminal
# ---------------------------------------------------------------------------


def test_dashboard_render_exception_never_crashes_the_run() -> None:
    # Both the steady-state repaint (_flush_refresh, inside a loop callback) and
    # the final paint (_shutdown) render best-effort: a raising _render must drop
    # the frame / still stop the canvas, never escape into the asyncio loop or
    # leave the terminal in alt-screen state.
    reporting._reset_for_tests()
    console = Console(file=StringIO(), force_terminal=True, width=140, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)

    async def drive() -> None:
        lane = dashboard.reporter_for("lane-1")
        lane.on_threat_model_start(_ctx("lane-1", n_tasks=1))
        assert dashboard._canvas_ok is True

        # Break rendering only after the canvas is up.
        def boom() -> object:
            raise RuntimeError("render blew up")

        dashboard._render = boom  # type: ignore[method-assign]

        # Steady-state repaint swallows the render error and keeps the canvas.
        dashboard._flush_refresh()
        assert dashboard._stopped is False
        assert dashboard._canvas_ok is True

        # Final paint (last lane ends -> _shutdown) still stops cleanly.
        lane.on_task_complete(_task_complete(1, True, "done"))
        lane.on_threat_model_end(_end(_ctx("lane-1", n_tasks=1)))

    asyncio.run(drive())

    assert dashboard._stopped is True
    assert dashboard._live_stopped is True
    assert reporting._ACTIVE_LIVE is None  # canvas released for the next run
    reporting._reset_for_tests()


def test_dashboard_lane_asr_stays_within_100pct_on_success_error() -> None:
    # A task can be success=True yet stop_reason="error" (goal met, then
    # reset_ephemeral_state failed). That success must NOT count toward the lane
    # ASR numerator, or n_success/n_completed would exceed 100%.
    reporting._reset_for_tests()
    console = Console(file=StringIO(), force_terminal=True, width=140, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)

    async def drive() -> None:
        lane_r = dashboard.reporter_for("lane-1")
        lane_r.on_threat_model_start(_ctx("lane-1", n_tasks=2))
        lane_r.on_task_complete(_task_complete(1, True, "done"))  # completed success
        lane_r.on_task_complete(  # success=True but errored -> excluded from ASR
            _task_complete(2, True, "error", error="reset failed")
        )
        lane = dashboard._lanes["lane-1"]
        assert lane.n_completed == 1  # only the 'done' task is a completed denominator
        assert lane.n_success == 1  # the errored success is not counted
        assert lane.n_error == 1
        assert lane.n_success <= lane.n_completed  # ASR <= 100%
        lane_r.on_threat_model_end(_end(_ctx("lane-1", n_tasks=2)))

    asyncio.run(drive())
    reporting._reset_for_tests()
