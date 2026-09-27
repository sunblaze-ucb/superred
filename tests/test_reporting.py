"""Tests for the live progress reporting seam (``anonframework.core.reporting``).

Covers the observer contract end-to-end:

* a ``RecordingReporter`` fake driven through a real ``Controller`` asserts the
  ordered lifecycle (start / per-task start / per-run / complete / skipped /
  single end), per-run cost delta vs cumulative, the ``evaluated`` flag, and the
  ``stop_reason`` for the done (``CountingOptimizer``) and error
  (``FailingOnEventOptimizer``) paths;
* ``PlainReporter`` rendering into an in-memory stream (banner + per-task line +
  end summary; errors routed to the err stream);
* ``should_use_plain`` gating + the ``ProgressReporter`` runtime Protocol;
* the shared ``Dashboard`` rich canvas lifecycle inside ``asyncio.run``;
* the ``LoggingBridge`` attribution + cross-label filtering.
"""

from __future__ import annotations

import asyncio
import logging
from io import StringIO
from typing import Any

from rich.console import Console

from anonframework.core import reporting
from anonframework.core.controller import Controller, TargetFactory
from anonframework.core.interfaces.security_claim import SecurityClaim
from anonframework.core.reporting import (
    Dashboard,
    DiagnosticEvent,
    LoggingBridge,
    NullReporter,
    PlainReporter,
    ProgressReporter,
    RunCompleteEvent,
    TaskCompleteEvent,
    TaskSkippedEvent,
    TaskStartEvent,
    ThreatModelContext,
    ThreatModelEndEvent,
    should_use_plain,
)

from .conftest import (
    EXTERNAL_TAG,
    CountingOptimizer,
    FailingOnEventOptimizer,
    NotApplicableTask,
    StubTarget,
    StubTask,
)

EXTERNAL_SCOPE = frozenset({EXTERNAL_TAG})


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class RecordingReporter:
    """A ``ProgressReporter`` that appends every call as ``(method, event)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def on_threat_model_start(self, ctx: ThreatModelContext) -> None:
        self.calls.append(("start", ctx))

    def on_task_start(self, ev: TaskStartEvent) -> None:
        self.calls.append(("task_start", ev))

    def on_run_complete(self, ev: RunCompleteEvent) -> None:
        self.calls.append(("run_complete", ev))

    def on_task_complete(self, ev: TaskCompleteEvent) -> None:
        self.calls.append(("task_complete", ev))

    def on_task_skipped(self, ev: TaskSkippedEvent) -> None:
        self.calls.append(("task_skipped", ev))

    def on_diagnostic(self, ev: DiagnosticEvent) -> None:
        self.calls.append(("diagnostic", ev))

    def on_threat_model_end(self, ev: ThreatModelEndEvent) -> None:
        self.calls.append(("end", ev))

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def events(self, method: str) -> list[Any]:
        return [ev for name, ev in self.calls if name == method]


def _ctx(**overrides: Any) -> ThreatModelContext:
    base: dict[str, Any] = {
        "label": "exp",
        "attacker": "atk",
        "target": "tgt",
        "claim": "clm",
        "model": "test-model",
        "scope": ("external",),
        "n_tasks": 2,
    }
    base.update(overrides)
    return ThreatModelContext(**base)


# ---------------------------------------------------------------------------
# (1) RecordingReporter driven through a real Controller
# ---------------------------------------------------------------------------


async def test_lifecycle_sequence_success_and_skip() -> None:
    """A multi-run task (2 runs, done) + a NotApplicable task (skipped)
    produces the full ordered lifecycle and matching end aggregates.

    The task deliberately does not win: a won task ends at run 1, and this
    exercises the MULTI-run sequence.
    """
    reporter = RecordingReporter()
    controller = Controller(
        optimizer_factory=lambda: CountingOptimizer(stop_after=2),
        target_factory=TargetFactory(create=StubTarget),
        security_claim=SecurityClaim.from_tasks(
            [StubTask(goal_text="alpha", success=False), NotApplicableTask()]
        ),
        scope=EXTERNAL_SCOPE,
    )
    await controller.run(reporter=reporter)

    names = reporter.names()
    # Exactly one start (first) and one end (last).
    assert names[0] == "start"
    assert names[-1] == "end"
    assert names.count("start") == 1
    assert names.count("end") == 1

    # Both tasks (including the skipped one) get an on_task_start.
    starts = reporter.events("task_start")
    assert {ev.task_index for ev in starts} == {1, 2}

    # Task 1 (StubTask) ran to a verdict; task 2 (NotApplicable) was skipped.
    completes = reporter.events("task_complete")
    assert [ev.task_index for ev in completes] == [1]
    assert completes[0].stop_reason == "done"
    assert completes[0].success is False

    skips = reporter.events("task_skipped")
    assert [ev.task_index for ev in skips] == [2]

    # Two runs were reported for task 1 (CountingOptimizer stop_after=2), both
    # evaluated by the evaluator, and cumulative cost equals the running sum of
    # per-run deltas.
    runs = reporter.events("run_complete")
    assert [ev.run_number for ev in runs] == [1, 2]
    assert all(ev.evaluated is True for ev in runs)
    assert all(ev.errored is False for ev in runs)
    running = 0.0
    for ev in runs:
        running += ev.run_cost_delta_usd
        assert ev.cumulative_cost_usd == running

    # End aggregate: one evaluated task, one completed success, one skipped.
    end = reporter.events("end")[0]
    assert end.n_tasks == 1
    assert end.n_completed == 1
    assert end.n_success == 0
    assert end.n_skipped == 1
    assert end.asr == 0.0


async def test_lifecycle_ordering_within_task() -> None:
    """Within a single task the events arrive start -> run(s) -> complete."""
    reporter = RecordingReporter()
    controller = Controller(
        optimizer_factory=lambda: CountingOptimizer(stop_after=1),
        target_factory=TargetFactory.singleton(StubTarget()),
        security_claim=SecurityClaim.from_tasks([StubTask(goal_text="solo")]),
        scope=EXTERNAL_SCOPE,
    )
    await controller.run(reporter=reporter)

    names = reporter.names()
    assert names == [
        "start",
        "task_start",
        "run_complete",
        "task_complete",
        "end",
    ]


async def test_lifecycle_error_stop_reason() -> None:
    """An optimizer that raises mid-run yields errored run + error task."""
    reporter = RecordingReporter()
    controller = Controller(
        optimizer_factory=FailingOnEventOptimizer,
        target_factory=TargetFactory.singleton(StubTarget()),
        security_claim=SecurityClaim.from_tasks([StubTask(goal_text="boom")]),
        scope=EXTERNAL_SCOPE,
        max_runs_per_task=1,
    )
    await controller.run(reporter=reporter)

    run = reporter.events("run_complete")[0]
    assert run.errored is True
    assert run.evaluated is False
    assert run.primary_score == 0.0

    complete = reporter.events("task_complete")[0]
    assert complete.stop_reason == "error"
    assert complete.success is False
    assert complete.error is not None

    end = reporter.events("end")[0]
    assert end.n_error == 1
    assert end.n_completed == 0
    assert end.asr is None


# ---------------------------------------------------------------------------
# (2) PlainReporter rendering
# ---------------------------------------------------------------------------


def test_plain_reporter_banner_line_and_summary() -> None:
    out = StringIO()
    err = StringIO()
    reporter = PlainReporter(file=out, err=err)

    reporter.on_threat_model_start(_ctx(n_tasks=3))
    reporter.on_task_complete(
        TaskCompleteEvent(
            task_index=1,
            goal="steal the secret",
            success=True,
            stop_reason="done",
            best_score=0.875,
            n_runs=2,
            cost_usd=0.0,
        )
    )
    reporter.on_task_skipped(TaskSkippedEvent(task_index=2, goal="n/a goal"))
    reporter.on_diagnostic(
        DiagnosticEvent(label="exp", task_index=1, level="error", message="kaboom")
    )
    reporter.on_threat_model_end(
        ThreatModelEndEvent(
            context=_ctx(n_tasks=3),
            n_tasks=2,
            n_success=1,
            n_completed=2,
            n_error=0,
            n_budget_exhausted=0,
            n_skipped=1,
            asr=0.5,
            max_primary_score=0.875,
            mean_primary_score=0.4,
            total_calls=7,
            total_cost_usd=0.0,
            duration_s=3.2,
        )
    )

    text = out.getvalue()
    # Start banner: mirrors the old summary header.
    assert "Threat model:" in text
    assert "model=test-model" in text
    assert "attacker=atk" in text
    assert "tasks=3" in text
    # Per-task completion line (success -> OK) and the skip line.
    assert "[OK] steal the secret" in text
    assert "score=0.8750" in text
    assert "[SKIP] n/a goal" in text
    # End summary content.
    assert "Overall: 1/2 completed tasks succeeded (ASR 50.0%)" in text
    assert "Highest score: 0.8750" in text
    assert "Attacker LLM: 7 calls" in text
    assert "(3.2s)" in text

    # Error diagnostic routed to the err stream, not the progress stream.
    assert "kaboom" in err.getvalue()
    assert "kaboom" not in text


def test_plain_reporter_runs_silent_unless_show_runs() -> None:
    out = StringIO()
    silent = PlainReporter(file=out)
    ev = RunCompleteEvent(
        task_index=1,
        goal="g",
        run_number=1,
        primary_score=0.5,
        success=False,
        evaluated=True,
        errored=False,
        done=False,
        run_cost_delta_usd=0.0,
        cumulative_cost_usd=0.0,
    )
    silent.on_run_complete(ev)
    assert out.getvalue() == ""

    out2 = StringIO()
    loud = PlainReporter(file=out2, show_runs=True)
    loud.on_run_complete(ev)
    assert "task 1 run 1" in out2.getvalue()


# ---------------------------------------------------------------------------
# (3) should_use_plain gating + Protocol conformance
# ---------------------------------------------------------------------------


class _FakeTTY:
    """A stream that claims to be a TTY (so env is the only plain trigger)."""

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        return len(s)


def test_should_use_plain_non_tty_stream() -> None:
    # A StringIO is not a TTY -> plain, regardless of env.
    assert should_use_plain(StringIO(), env={}) is True


def test_should_use_plain_env_forces() -> None:
    tty = _FakeTTY()
    # A real TTY with a neutral env keeps the canvas (negative control).
    assert should_use_plain(tty, env={}) is False
    # Each forcing env variable flips it to plain.
    assert should_use_plain(tty, env={"CI": "1"}) is True
    assert should_use_plain(tty, env={"ANONFRAMEWORK_NO_DASHBOARD": "1"}) is True
    assert should_use_plain(tty, env={"TERM": "dumb"}) is True


def test_reporters_satisfy_protocol() -> None:
    assert isinstance(NullReporter(), ProgressReporter)
    assert isinstance(PlainReporter(), ProgressReporter)
    assert isinstance(RecordingReporter(), ProgressReporter)


# ---------------------------------------------------------------------------
# (4) Dashboard rich canvas lifecycle
# ---------------------------------------------------------------------------


def test_dashboard_canvas_lifecycle() -> None:
    reporting._reset_for_tests()
    sink = StringIO()
    console = Console(file=sink, force_terminal=True, color_system=None)
    dashboard = Dashboard(console=console, redirect=False)

    async def drive() -> None:
        lane = dashboard.reporter_for("lane-1")
        lane.on_threat_model_start(_ctx(label="lane-1", n_tasks=1))
        # Canvas acquired on a real running loop.
        assert dashboard._canvas_ok is True
        assert reporting._ACTIVE_LIVE is not None
        lane.on_task_start(TaskStartEvent(task_index=1, goal="g"))
        lane.on_run_complete(
            RunCompleteEvent(
                task_index=1,
                goal="g",
                run_number=1,
                primary_score=1.0,
                success=True,
                evaluated=True,
                errored=False,
                done=True,
                run_cost_delta_usd=0.0,
                cumulative_cost_usd=0.0,
            )
        )
        lane.on_task_complete(
            TaskCompleteEvent(
                task_index=1,
                goal="g",
                success=True,
                stop_reason="done",
                best_score=1.0,
                n_runs=1,
                cost_usd=0.0,
            )
        )
        lane.on_threat_model_end(
            ThreatModelEndEvent(
                context=_ctx(label="lane-1", n_tasks=1),
                n_tasks=1,
                n_success=1,
                n_completed=1,
                n_error=0,
                n_budget_exhausted=0,
                n_skipped=1,
                asr=1.0,
                max_primary_score=1.0,
                mean_primary_score=1.0,
                total_calls=0,
                total_cost_usd=0.0,
                duration_s=0.1,
            )
        )

    asyncio.run(drive())

    # After the last lane ended the canvas is released.
    assert dashboard._stopped is True
    assert reporting._ACTIVE_LIVE is None
    # The last frame is itself the summary (there is no separate table): the
    # live console captured the brand; render at a controlled width to assert
    # the per-threat-model identity is shown.
    assert "anonframework" in sink.getvalue()
    cap = Console(file=StringIO(), width=140, color_system=None)
    cap.print(dashboard._render())
    assert "atk" in cap.file.getvalue()
    assert "clm" in cap.file.getvalue()  # the security claim renders on the identity line

    reporting._reset_for_tests()


# ---------------------------------------------------------------------------
# (5) LoggingBridge attribution + cross-label filtering
# ---------------------------------------------------------------------------


def _record(msg: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name="anonframework.test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_logging_bridge_attributes_and_filters() -> None:
    reporter = RecordingReporter()
    bridge = LoggingBridge("my-label", reporter)

    # A record emitted under this controller's context is attributed to its task.
    token = reporting.current_task.set(("my-label", 1))
    try:
        bridge.emit(_record("hello world", level=logging.WARNING))
    finally:
        reporting.current_task.reset(token)

    diags = reporter.events("diagnostic")
    assert len(diags) == 1
    assert diags[0].task_index == 1
    assert diags[0].label == "my-label"
    assert diags[0].level == "warning"
    assert diags[0].message == "hello world"
    assert diags[0].logger_name == "anonframework.test"

    # A record belonging to a DIFFERENT controller's context is ignored.
    token2 = reporting.current_task.set(("other-label", 5))
    try:
        bridge.emit(_record("not mine"))
    finally:
        reporting.current_task.reset(token2)
    assert len(reporter.events("diagnostic")) == 1


def test_logging_bridge_unattributed_is_experiment_scope() -> None:
    reporter = RecordingReporter()
    sink_calls: list[DiagnosticEvent] = []
    bridge = LoggingBridge("my-label", reporter, sink=sink_calls.append)

    # No current_task context -> experiment-level record (task_index None).
    bridge.emit(_record("startup", level=logging.ERROR))

    diag = reporter.events("diagnostic")[0]
    assert diag.task_index is None
    assert diag.level == "error"
    # The optional sink also received it.
    assert sink_calls and sink_calls[0].message == "startup"
