"""Controller error-containment + persistence-error paths.

The controller must survive every per-task failure and every persistence-side
failure without abandoning the whole threat model.  These tests drive a real
``Controller`` and assert:

* a target factory whose ``create()`` raises yields a synthesized
  ``stop_reason="error"`` task result (not a crashed run);
* a task whose ``evaluate()`` raises mid-run is contained as
  ``stop_reason="error"`` with the traceback captured on ``TaskResult.error``;
* a ``NotApplicable`` task surfaces as a skipped task;
* with persistence on, a failing ``publish_task`` / ``finalize`` is swallowed
  (the run still returns its result), and a diagnostic emitted during a task is
  written to the per-task JSONL sink.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from anonframework.core.controller import Controller, TargetFactory
from anonframework.core.interfaces.security_claim import SecurityClaim
from anonframework.core.persistence import ExperimentSession
from anonframework.core.types.evaluation import EvaluationResult
from anonframework.core.types.security_domain import Scope
from anonframework.core.types.trajectory import Trajectory

from .conftest import (
    EXTERNAL_TAG,
    STUB_LLM_CONFIG,
    CountingOptimizer,
    NotApplicableTask,
    StubTarget,
    StubTask,
)

EXTERNAL_SCOPE: Scope = frozenset({EXTERNAL_TAG})


class EvaluateRaisesTask(StubTask):
    """A task whose ``evaluate()`` blows up mid-run."""

    async def evaluate(self, trajectory: Trajectory, target: object) -> EvaluationResult:
        raise RuntimeError("evaluate exploded")


class LoggingConfigureTask(StubTask):
    """A task that emits a log record while configuring the target."""

    async def configure_target(self, target: object) -> None:
        logging.getLogger("anonframework.test.controller_errors").warning(
            "configuring for %s", self.goal.description
        )


def _controller(tasks: list[StubTask], **overrides: object) -> Controller:
    kwargs: dict[str, object] = dict(
        optimizer_factory=lambda: CountingOptimizer(stop_after=1),
        target_factory=TargetFactory(create=StubTarget, concurrency=1),
        security_claim=SecurityClaim.from_tasks(tasks),
        scope=EXTERNAL_SCOPE,
        llm_config=STUB_LLM_CONFIG,
        max_runs_per_task=1,
    )
    kwargs.update(overrides)
    return Controller(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Per-task error containment (returned result shape)
# ---------------------------------------------------------------------------


async def test_target_factory_create_raises_is_synthesized_error() -> None:
    def _raising_create() -> StubTarget:
        raise RuntimeError("cannot build target")

    controller = _controller(
        [StubTask(goal_text="needs a target")],
        target_factory=TargetFactory(create=_raising_create),
    )
    result = await controller.run()

    assert len(result.task_results) == 1
    tr = result.task_results[0]
    assert tr.stop_reason == "error"
    assert tr.success is False
    assert tr.error is not None
    assert tr.runs == []  # no run ever started


async def test_task_evaluate_raises_is_error_stop_reason() -> None:
    controller = _controller([EvaluateRaisesTask(goal_text="explodes on eval")])
    result = await controller.run()

    tr = result.task_results[0]
    assert tr.stop_reason == "error"
    assert tr.success is False
    assert tr.error is not None
    assert "evaluate exploded" in tr.error


async def test_not_applicable_task_is_skipped() -> None:
    controller = _controller(
        [StubTask(goal_text="applies"), NotApplicableTask()]  # type: ignore[list-item]
    )
    result = await controller.run()

    assert len(result.task_results) == 1
    assert result.task_results[0].success is True
    assert len(result.skipped_tasks) == 1


# ---------------------------------------------------------------------------
# Persistence-side failures are swallowed (run still returns)
# ---------------------------------------------------------------------------


async def test_publish_task_failure_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(self: ExperimentSession, index: int, tr: object) -> None:
        raise RuntimeError("disk full during publish")

    monkeypatch.setattr(ExperimentSession, "publish_task", _boom)

    controller = _controller(
        [StubTask(goal_text="alpha")],
        persist=True,
        report=False,
        results_dir=tmp_path,
    )
    result = await controller.run()  # must not raise

    # The in-memory result is preserved even though persistence failed.
    assert len(result.task_results) == 1
    assert result.task_results[0].stop_reason == "success"


async def test_finalize_failure_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(self: ExperimentSession, started: object, ended: object) -> None:
        raise RuntimeError("cannot write result.json")

    monkeypatch.setattr(ExperimentSession, "finalize", _boom)

    controller = _controller(
        [StubTask(goal_text="beta")],
        persist=True,
        report=False,
        results_dir=tmp_path,
    )
    result = await controller.run()  # must not raise

    assert len(result.task_results) == 1
    assert result.task_results[0].success is True


async def test_diagnostic_sink_writes_per_task_jsonl(tmp_path: Path) -> None:
    """A log record emitted during a task lands in that task's diagnostics log."""
    controller = _controller(
        [LoggingConfigureTask(goal_text="logs a warning")],
        persist=True,
        report=False,
        results_dir=tmp_path,
    )
    result = await controller.run()

    assert len(result.task_results) == 1
    # The record must land in THIS task's own log (attributed via the contextvar),
    # not merely somewhere in the tree — assert the specific published task dir.
    from anonframework.core.persistence import iter_task_dirs

    exp_dir = next(d for d in tmp_path.iterdir() if d.is_dir())
    task_dirs = iter_task_dirs(exp_dir)
    assert len(task_dirs) == 1
    task_log = (task_dirs[0] / "logs" / "diagnostics.log").read_text(encoding="utf-8")
    assert "configuring for logs a warning" in task_log
