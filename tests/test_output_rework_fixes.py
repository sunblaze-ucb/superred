"""Regression tests for defects found in adversarial review of the output rework.

Each test pins one confirmed fix so it cannot silently regress:
- the default dashboard re-arms across sequential sweeps (was single-use);
- ``run()`` always ends the reporter lane and releases the lock, even on an
  unexpected abort, and one bad task's orchestration cannot abort the sweep;
- ASR is clamped to [0, 1] (a task can be success=True yet stop_reason=error);
- the snapshot copies the appended-in-place experiment log (was hardlinked);
- per-run cost is a delta distinct from the cumulative snapshot.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from anonframework.core import persistence, reporting
from anonframework.core.controller import (
    Controller,
    RunResult,
    TargetFactory,
    TaskResult,
    ThreatModelResult,
    _threat_model_end_event,
)
from anonframework.core.interfaces.security_claim import SecurityClaim
from anonframework.core.reporting import (
    NullReporter,
    TaskStartEvent,
    ThreatModelContext,
    ThreatModelEndEvent,
)
from anonframework.core.types.evaluation import EvaluationResult, Score
from anonframework.core.types.llm import LLMUsage
from anonframework.core.types.trajectory import Trajectory
from tests.conftest import (
    EXTERNAL_TAG,
    STUB_LLM_CONFIG,
    CountingOptimizer,
    NotApplicableTask,
    StubTarget,
    StubTask,
)


class RecordingReporter(NullReporter):
    """Captures the lifecycle calls it receives."""

    def __init__(self) -> None:
        self.starts = 0
        self.ends: list[ThreatModelEndEvent] = []

    def on_threat_model_start(self, ctx: ThreatModelContext) -> None:
        self.starts += 1

    def on_threat_model_end(self, ev: ThreatModelEndEvent) -> None:
        self.ends.append(ev)


def _ctx() -> ThreatModelContext:
    return ThreatModelContext(label="x", attacker="a", target="t", claim="c", model=None)


# ---------------------------------------------------------------------------
# FIX 1: the default dashboard re-arms after a completed sweep
# ---------------------------------------------------------------------------


def test_get_default_dashboard_rearms_after_shutdown() -> None:
    reporting._reset_for_tests()
    d1 = reporting.get_default_dashboard()
    assert reporting.get_default_dashboard() is d1  # same until spent
    d1._stopped = True  # simulate the first sweep finishing (shutdown done)
    d2 = reporting.get_default_dashboard()
    assert d2 is not d1, "a spent dashboard must be replaced, not reused"
    assert d2._stopped is False
    reporting._reset_for_tests()


# ---------------------------------------------------------------------------
# FIX 2: run() always ends the reporter + releases the lock on abort
# ---------------------------------------------------------------------------


def _controller(tmp_path: object, **kw: object) -> Controller:
    return Controller(
        optimizer_factory=lambda: CountingOptimizer(stop_after=1),
        target_factory=TargetFactory(create=lambda: StubTarget()),
        security_claim=SecurityClaim.from_tasks([StubTask(goal_text="Leak")]),
        scope=frozenset({EXTERNAL_TAG}),
        llm_config=STUB_LLM_CONFIG,
        max_runs_per_task=1,
        persist=True,
        results_dir=str(tmp_path),
        attacker_label="pair",
        target_label="ad",
        claim_label="hb",
        **kw,  # type: ignore[arg-type]
    )


async def test_run_aborts_cleanly_and_releases_lock(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting._reset_for_tests()
    rec = RecordingReporter()

    async def boom(*a: object, **k: object) -> None:
        raise RuntimeError("iterate boom")

    monkeypatch.setattr(Controller, "_iterate_tasks", boom)
    with pytest.raises(RuntimeError, match="iterate boom"):
        await _controller(tmp_path).run(reporter=rec)

    # The reporter lane was started AND ended (frees a shared live canvas).
    assert rec.starts == 1
    assert len(rec.ends) == 1

    # The .lock was released on the abort path: a second run of the same
    # experiment identity succeeds instead of raising "locked".
    monkeypatch.undo()
    result = await _controller(tmp_path).run(reporter=NullReporter())
    assert len(result.task_results) == 1


class _RaisingStartReporter(NullReporter):
    def on_task_start(self, ev: TaskStartEvent) -> None:
        raise RuntimeError("start boom")


async def test_run_one_contains_orchestration_error(tmp_path: object) -> None:
    reporting._reset_for_tests()
    controller = Controller(
        optimizer_factory=lambda: CountingOptimizer(stop_after=1),
        target_factory=TargetFactory(create=lambda: StubTarget()),
        security_claim=SecurityClaim.from_tasks([StubTask(goal_text="a"), StubTask(goal_text="b")]),
        scope=frozenset({EXTERNAL_TAG}),
        llm_config=STUB_LLM_CONFIG,
        max_runs_per_task=1,
        persist=False,
    )
    # A reporter callback raising must NOT abort the whole threat model.
    result = await controller.run(reporter=_RaisingStartReporter())
    assert len(result.task_results) == 2
    assert all(tr.stop_reason == "error" for tr in result.task_results)


# ---------------------------------------------------------------------------
# FIX 3: ASR is clamped to [0, 1] at every metric site
# ---------------------------------------------------------------------------


def _task_result(goal: str, success: bool, stop_reason: str, score: float) -> TaskResult:
    s = Score(value=score)
    return TaskResult(
        task=StubTask(goal_text=goal),
        runs=[],
        best_score=s,
        best_evaluation=EvaluationResult(success=success, primary_score=s),
        success=success,
        llm_usage=LLMUsage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )


def _task_view(success: bool, stop_reason: str) -> persistence.TaskView:
    return persistence.TaskView(
        index=1,
        goal="g",
        goal_hash="",
        dir="",
        status=stop_reason,
        success=success,
        stop_reason=stop_reason,
        best_score=1.0,
        n_runs=1,
        calls=0,
        cost=0.0,
        error=None,
        started_at=None,
        ended_at=None,
    )


def test_asr_clamped_when_success_and_error_controller_site() -> None:
    # One clean success + one success that then errored on reset_ephemeral_state.
    ok = _task_result("a", success=True, stop_reason="done", score=1.0)
    weird = _task_result("b", success=True, stop_reason="error", score=1.0)
    tmr = ThreatModelResult(
        scope=frozenset({EXTERNAL_TAG}),
        read_only=frozenset(),
        llm_config=None,
        task_results=[ok, weird],
    )
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ev = _threat_model_end_event(_ctx(), tmr, t0, t0)
    assert ev.n_completed == 1
    assert ev.n_success == 1, "the success+error task must not inflate the numerator"
    assert ev.asr == 1.0


def test_asr_clamped_when_success_and_error_persistence_site() -> None:
    views = [_task_view(True, "done"), _task_view(True, "error")]
    summary = persistence._compute_summary(views, n_skipped=0)
    assert summary["n_completed"] == 1
    assert summary["n_success"] == 1
    assert summary["asr"] == 1.0
    assert summary["n_failed"] >= 0


# ---------------------------------------------------------------------------
# FIX 4: the snapshot copies the appended-in-place experiment log
# ---------------------------------------------------------------------------


def test_snapshot_copies_experiment_logs_not_hardlink(tmp_path: object) -> None:
    from pathlib import Path

    exp = Path(str(tmp_path)) / "exp"
    (exp / "logs").mkdir(parents=True)
    (exp / "tasks").mkdir()
    (exp / "logs" / "diagnostics.log").write_text("OLD\n", encoding="utf-8")
    persistence._atomic_write_json(exp / "result.json", {"marker": "v1"})

    snap = persistence.snapshot_current(exp)
    assert snap is not None
    # Simulate the resume appending to the live experiment log in place.
    with open(exp / "logs" / "diagnostics.log", "a", encoding="utf-8") as fh:
        fh.write("NEW\n")

    snap_log = (snap / "logs" / "diagnostics.log").read_text(encoding="utf-8")
    assert "OLD" in snap_log
    assert "NEW" not in snap_log, "the snapshot log must be immutable (copied, not hardlinked)"


# ---------------------------------------------------------------------------
# FIX 8: per-run cost delta is distinct from the cumulative snapshot
# ---------------------------------------------------------------------------


def test_iterations_delta_distinct_from_cumulative() -> None:
    def run(cum_calls: int, cum_cost: float, d_calls: int, d_cost: float, done: bool) -> RunResult:
        return RunResult(
            trajectory=Trajectory(filtered_scope=frozenset({EXTERNAL_TAG})),
            evaluation=EvaluationResult(success=done, primary_score=Score(value=cum_cost)),
            llm_usage=LLMUsage(calls=cum_calls, cost=cum_cost),
            run_usage_delta=LLMUsage(calls=d_calls, cost=d_cost),
            evaluated=True,
            done=done,
        )

    r1 = run(2, 0.5, 2, 0.5, False)
    r2 = run(5, 1.2, 3, 0.7, True)
    tr = TaskResult(
        task=StubTask(goal_text="g"),
        runs=[r1, r2],
        best_score=Score(value=1.2),
        best_evaluation=r2.evaluation,
        success=True,
        llm_usage=LLMUsage(calls=5, cost=1.2),
        stop_reason="done",
    )
    it = persistence._build_iterations_json(tr, 1)
    runs = it["runs"]
    assert runs[0]["usage_delta"]["cost"] == 0.5
    assert runs[0]["usage_cumulative"]["cost"] == 0.5
    assert runs[1]["usage_delta"]["cost"] == 0.7
    assert runs[1]["usage_cumulative"]["cost"] == 1.2
    # The delta-vs-cumulative distinction is the whole point: summing deltas
    # gives the task total; summing cumulatives would double-count.
    assert abs(sum(r["usage_delta"]["cost"] for r in runs) - 1.2) < 1e-9


# ---------------------------------------------------------------------------
# A skipped task leaves no stray staging dir
# ---------------------------------------------------------------------------


async def test_skipped_task_leaves_no_stray_staging(tmp_path: object) -> None:
    reporting._reset_for_tests()
    controller = Controller(
        optimizer_factory=lambda: CountingOptimizer(stop_after=1),
        target_factory=TargetFactory(create=lambda: StubTarget()),
        security_claim=SecurityClaim.from_tasks([StubTask(goal_text="a"), NotApplicableTask()]),
        scope=frozenset({EXTERNAL_TAG}),
        llm_config=STUB_LLM_CONFIG,
        max_runs_per_task=1,
        persist=True,
        results_dir=str(tmp_path),
        report=False,
    )
    result = await controller.run()
    assert len(result.skipped_tasks) == 1
    exp = next(d for d in Path(str(tmp_path)).iterdir() if d.is_dir())
    stray = list((exp / "tasks").glob("*.wip")) + list((exp / "tasks").glob("*.old"))
    assert stray == [], f"skipped task left stray staging dirs: {stray}"
