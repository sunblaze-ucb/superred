"""Per-task wall-clock bound: ``Controller(task_time_cap_s=...)``.

The existing per-task bounds (``task_cost_cap_usd``, ``max_runs_per_task``)
bound the WORK a task may do. Neither can stop a task whose provider call
blocks forever: no spend accrues and no run completes, so the task hangs until
something outside the framework kills the whole process. These tests pin the
wall-clock bound that does stop it.

They also pin the disposition of what the cap cancels, which is two different
things wearing one name:

- the cap cut the task short AFTER it had completed and judged runs -- a real
  measurement, truncated. It keeps those runs, and a resume keeps the task.
- the cap cancelled the task with nothing judged -- no measurement at all. A
  resume recomputes it, and it is never counted as a verdict on the attack.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import IO, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anonframework.core.controller import Controller, TargetFactory
from anonframework.core.interfaces.security_claim import SecurityClaim
from anonframework.core.persistence import (
    _KEPT_STATUSES,
    TaskView,
    _compute_summary,
    _status_of,
    is_kept,
    iter_task_dirs,
    load_manifest,
    load_task,
    plan_resume,
)
from anonframework.core.reporting import PlainReporter, TaskCompleteEvent
from anonframework.core.types.event import Event, EventHandler, EventResponse, EventResponseHandler
from anonframework.core.types.security_domain import Scope

from .conftest import EXTERNAL_TAG, STUB_LLM_CONFIG, StubOptimizer, StubTarget, StubTask

EXTERNAL_SCOPE: Scope = frozenset({EXTERNAL_TAG})


class HangingTarget(StubTarget):
    """Blocks forever without spending anything.

    This is precisely the shape the work bounds cannot catch: no LLM cost
    accrues, so ``task_cost_cap_usd`` never trips, and no run ever completes,
    so ``max_runs_per_task`` never trips either.
    """

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        await asyncio.Event().wait()


class TruncatedTarget(StubTarget):
    """Completes ``runs_before_hang`` full runs, then blocks forever.

    The realistic shape of a wall-clock timeout: not a task that did nothing,
    but a task that was measuring fine and got cut off.
    """

    def __init__(self, runs_before_hang: int = 2) -> None:
        super().__init__()
        self._remaining = runs_before_hang

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        if self._remaining <= 0:
            await asyncio.Event().wait()
        self._remaining -= 1
        await super().run(emit, send_event)


class HangingOptimizer(StubOptimizer):
    """Blocks inside ``on_event``, where closing the channel cannot reach it.

    The target-side hangs above are interrupted by the cap itself, because the
    task is parked in the cancelled coroutine when the deadline lands.  This
    one hangs the optimizer, which runs as a separate task the controller only
    joins during cleanup -- after the cancellation has already been delivered,
    so nothing is left to interrupt the join.
    """

    async def on_event(self, event: Event) -> EventResponse:
        await asyncio.Event().wait()


class WarmupTimeoutOptimizer(StubOptimizer):
    """Raises a TimeoutError of its own from ``initialize``.

    The shape of a provider client read timeout during a warmup call.  It is
    the same exception type the wall-clock cap raises, and it reaches the same
    handler.
    """

    async def initialize(
        self,
        goal: Any,
        controllables: Any,
        observables: Any,
        llm_client: Any,
    ) -> None:
        raise TimeoutError("HTTPReadTimeout: warmup call to the provider timed out")


class HangingTeardownOptimizer(StubOptimizer):
    """Blocks forever in ``teardown``, which runs after the cap has fired."""

    async def teardown(self) -> None:
        await asyncio.Event().wait()


class HangingTeardownTarget(HangingTarget):
    """Blocks forever in ``run`` and again in ``teardown``."""

    async def teardown(self) -> None:
        await asyncio.Event().wait()


def _controller(
    *,
    target_factory: TargetFactory,
    tasks: list[StubTask],
    cap: float | None = 0.5,
    **overrides: Any,
) -> Controller:
    kwargs: dict[str, Any] = dict(
        scope=EXTERNAL_SCOPE,
        optimizer_factory=lambda: StubOptimizer(),
        target_factory=target_factory,
        security_claim=SecurityClaim.from_tasks(tasks),
        llm_config=STUB_LLM_CONFIG,
        task_time_cap_s=cap,
        report=False,
        attacker_label="atk",
        target_label="tgt",
        claim_label="clm",
    )
    kwargs.update(overrides)
    return Controller(**kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_a_non_positive_cap() -> None:
    for bad in (0, -1.5):
        with pytest.raises(ValueError, match="task_time_cap_s must be positive"):
            Controller(
                scope=EXTERNAL_SCOPE,
                optimizer_factory=lambda: StubOptimizer(),
                target_factory=TargetFactory.singleton(StubTarget()),
                security_claim=SecurityClaim.from_tasks([StubTask()]),
                llm_config=STUB_LLM_CONFIG,
                task_time_cap_s=bad,
            )


# ---------------------------------------------------------------------------
# Classification: what the cap left behind decides what the record says
# ---------------------------------------------------------------------------


def test_a_truncated_task_and_an_empty_one_get_different_statuses() -> None:
    """One judged run makes a timeout a measurement; none makes it nothing."""
    assert _status_of(False, "timeout", 2) == "timeout"
    assert _status_of(False, "timeout", 0) == "timeout_empty"
    # Truncation is reported even when a run succeeded before the cap: the
    # record must never hide that the task was cut short.
    assert _status_of(True, "timeout", 3) == "timeout"


def test_only_the_truncated_timeout_is_kept_on_resume() -> None:
    assert "timeout" in _KEPT_STATUSES
    assert "timeout_empty" not in _KEPT_STATUSES
    assert is_kept("timeout", n_runs=1) is True
    assert is_kept("timeout_empty", n_runs=0) is False
    # Guard for records written before a timeout retained its runs: those are
    # always zero-run, and a zero-run record holds no measurement.
    assert is_kept("timeout", n_runs=0) is False
    assert is_kept("success", n_runs=1) is True
    assert is_kept("error", n_runs=3) is False


def test_a_truncated_timeout_counts_as_a_failure_but_an_empty_one_does_not() -> None:
    """The time cap is part of the threat model, so exceeding it is a failure.

    An attacker that has not succeeded within its time budget has failed under
    the threat model being measured, exactly as one that exhausted its cost
    budget has, so a truncated task enters the ASR denominator as a
    non-success. "timeout_empty" stays out: nothing was judged in a whole time
    budget, which is far more likely a hung provider call than an attacker
    working to the wire, and an outage counted as attacker failure is the one
    error this must not make.
    """

    class View:
        def __init__(self, stop_reason: str, success: bool, status: str) -> None:
            self.stop_reason = stop_reason
            self.success = success
            self.status = status
            self.best_score = 0.0
            self.calls = 0
            self.cost = 0.0

    views = cast(
        "list[TaskView]",
        [
            View("done", True, "success"),
            View("done", False, "failed"),
            View("timeout", False, "timeout"),
            View("timeout", False, "timeout_empty"),
        ],
    )
    summary = _compute_summary(views, n_skipped=0)
    assert summary["n_timeout"] == 1
    assert summary["n_timeout_empty"] == 1
    # done+done+timeout = 3 completed; the empty one is excluded.
    assert summary["n_completed"] == 3
    assert summary["n_failed"] == 2
    assert summary["asr"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# The real controller: what survives the cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hanging_task_is_cancelled_and_recorded_as_timeout() -> None:
    controller = _controller(
        target_factory=TargetFactory(create=HangingTarget),
        tasks=[StubTask()],
        cap=0.25,
    )
    result = await asyncio.wait_for(controller.run(), timeout=20)

    assert len(result.task_results) == 1
    tr = result.task_results[0]
    assert tr.stop_reason == "timeout"
    assert tr.success is False
    assert tr.runs == []
    assert "task_time_cap_s" in (tr.best_evaluation.rationale or "")


@pytest.mark.asyncio
async def test_the_cap_also_bounds_a_task_whose_optimizer_hangs() -> None:
    """The cap must bound the task even when the hang is in cleanup.

    The controller joins the optimizer task in the ``finally`` of the very
    coroutine the cap cancels.  An unbounded join there defeated the bound that
    cancelled it: the cancellation had already been delivered, so a fresh
    suspend in the ``finally`` had nothing left to interrupt it and the task ran
    forever.  ``asyncio.wait_for`` below is the test's own backstop -- if the
    cap does not hold, it raises rather than hanging the suite.
    """
    controller = _controller(
        target_factory=TargetFactory(create=StubTarget),
        tasks=[StubTask()],
        cap=0.25,
        optimizer_factory=lambda: HangingOptimizer(),
    )
    result = await asyncio.wait_for(controller.run(), timeout=30)

    tr = result.task_results[0]
    assert tr.stop_reason == "timeout"


@pytest.mark.asyncio
async def test_a_timeout_the_task_raised_itself_is_not_blamed_on_the_cap() -> None:
    """A client read timeout is an error, not a cap expiry.

    Both arrive as ``TimeoutError``.  Catching the type alone recorded a task
    that ran for milliseconds under a 60s cap as ``stop_reason="timeout"``,
    with a rationale asserting the cap "was exceeded", and dropped the real
    exception entirely.
    """
    controller = _controller(
        target_factory=TargetFactory(create=StubTarget),
        tasks=[StubTask()],
        cap=60.0,
        optimizer_factory=lambda: WarmupTimeoutOptimizer(),
    )
    result = await asyncio.wait_for(controller.run(), timeout=30)

    tr = result.task_results[0]
    assert tr.stop_reason == "error"
    assert "task_time_cap_s" not in (tr.best_evaluation.rationale or "")
    assert "HTTPReadTimeout" in (tr.error or "")


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("optimizer teardown", {"optimizer_factory": lambda: HangingTeardownOptimizer()}),
        ("target teardown", {"target_factory": TargetFactory(create=HangingTeardownTarget)}),
    ],
)
@pytest.mark.asyncio
async def test_cleanup_that_blocks_cannot_outlive_the_cap(
    label: str, overrides: dict[str, Any]
) -> None:
    """Every await in the cleanup path needs its own bound.

    The cap cancels the task body, so the ``finally`` runs while a cancellation
    is already unwinding -- and the cap fires only once, so a fresh suspend
    down there has nothing left to interrupt it.  A teardown that blocks
    therefore pinned the task forever despite the cap.  Both the optimizer's
    teardown and the target's are on that path.
    """
    kwargs: dict[str, Any] = dict(
        target_factory=TargetFactory(create=HangingTarget),
        tasks=[StubTask()],
        cap=0.25,
    )
    kwargs.update(overrides)
    controller = _controller(**kwargs)

    with patch("anonframework.core.controller._CLEANUP_GRACE_S", 0.2):
        started = asyncio.get_running_loop().time()
        result = await asyncio.wait_for(controller.run(), timeout=30)
        elapsed = asyncio.get_running_loop().time() - started

    assert result.task_results[0].stop_reason == "timeout"
    # cap + at most two cleanup budgets, plus slack for a loaded machine.
    assert elapsed < 0.25 + 2 * 0.2 + 5.0, f"{label}: cleanup overran its budget ({elapsed:.2f}s)"


@pytest.mark.asyncio
async def test_the_cap_does_not_leave_the_optimizer_running() -> None:
    """An abandoned optimizer must still be cancelled, not merely unjoined.

    The controller stops waiting for an optimizer that will not return, but it
    must not walk away leaving one executing -- with its LLM client and its
    spend -- after the task result has been recorded.
    """
    started: list[asyncio.Task[Any]] = []

    class TrackedHangingOptimizer(HangingOptimizer):
        async def run(self, channel: Any) -> None:
            current = asyncio.current_task()
            assert current is not None
            started.append(current)
            await super().run(channel)

    controller = _controller(
        target_factory=TargetFactory(create=StubTarget),
        tasks=[StubTask()],
        cap=0.25,
        optimizer_factory=lambda: TrackedHangingOptimizer(),
    )
    with patch("anonframework.core.controller._CLEANUP_GRACE_S", 0.2):
        await asyncio.wait_for(controller.run(), timeout=30)

    assert started, "the optimizer never started; the test proves nothing"
    await asyncio.sleep(0.05)
    assert all(t.cancelled() or t.done() for t in started), (
        "an optimizer task was left running after the run finished"
    )


@pytest.mark.asyncio
async def test_a_healthy_task_under_a_cap_gets_an_unbounded_teardown() -> None:
    """The cleanup budget is for a task the cap cut short, not for every task.

    Keying it on the cap merely being SET cancelled the teardown of a task that
    finished normally, well inside a generous cap -- so a target that stops
    containers or closes a proxy in teardown leaked exactly what it was about to
    release. A leak, introduced by the fix meant to prevent one.
    """
    finished: dict[str, bool] = {}

    class SlowTeardownTarget(StubTarget):
        async def teardown(self) -> None:
            await asyncio.sleep(0.3)
            finished["target"] = True

    class SlowTeardownOptimizer(StubOptimizer):
        async def teardown(self) -> None:
            await asyncio.sleep(0.3)
            finished["optimizer"] = True

    controller = _controller(
        target_factory=TargetFactory(create=SlowTeardownTarget),
        tasks=[StubTask()],
        cap=30.0,  # generous; never expires
        optimizer_factory=lambda: SlowTeardownOptimizer(),
    )
    with patch("anonframework.core.controller._CLEANUP_GRACE_S", 0.05):
        result = await asyncio.wait_for(controller.run(), timeout=30)

    assert result.task_results[0].stop_reason != "timeout"
    assert finished == {"target": True, "optimizer": True}, (
        f"a teardown was cut short on a task the cap never touched: {finished}"
    )


@pytest.mark.asyncio
async def test_without_a_cap_cleanup_is_not_bounded() -> None:
    """``task_time_cap_s=None`` means unbounded, cleanup included.

    A slow but finite teardown must run to completion rather than being
    cancelled by a budget the caller never asked for.
    """
    finished = False

    class SlowTeardownOptimizer(StubOptimizer):
        async def teardown(self) -> None:
            nonlocal finished
            await asyncio.sleep(0.3)
            finished = True

    controller = _controller(
        target_factory=TargetFactory(create=StubTarget),
        tasks=[StubTask()],
        cap=None,
        optimizer_factory=lambda: SlowTeardownOptimizer(),
    )
    with patch("anonframework.core.controller._CLEANUP_GRACE_S", 0.05):
        await asyncio.wait_for(controller.run(), timeout=30)

    assert finished, "teardown was cut short even though no cap was set"


@pytest.mark.asyncio
async def test_a_truncated_task_keeps_the_runs_it_completed() -> None:
    """The runs a task finished before the cap are measurements, not debris.

    Previously ``asyncio.wait_for`` cancelled the coroutine that owned them, so
    every completed, judged run died with the frame and the record claimed the
    task had measured nothing.
    """
    controller = _controller(
        target_factory=TargetFactory(create=lambda: TruncatedTarget(runs_before_hang=2)),
        tasks=[StubTask(score=0.6, success=False)],
        cap=0.5,
    )
    result = await asyncio.wait_for(controller.run(), timeout=20)

    tr = result.task_results[0]
    assert tr.stop_reason == "timeout"
    assert len(tr.runs) == 2
    assert all(r.evaluated and not r.errored for r in tr.runs)
    # The judge's own verdict, not a synthesized zero.
    assert tr.best_score.value == pytest.approx(0.6)
    assert all(len(r.trajectory.snapshot()) > 0 for r in tr.runs)


@pytest.mark.asyncio
async def test_the_cap_bounds_each_task_independently() -> None:
    """Three hanging tasks each time out; one slow task cannot consume another's budget."""
    controller = _controller(
        target_factory=TargetFactory(create=HangingTarget, concurrency=3),
        tasks=[StubTask(goal_text="a"), StubTask(goal_text="b"), StubTask(goal_text="c")],
        cap=0.25,
    )
    result = await asyncio.wait_for(controller.run(), timeout=20)
    assert [tr.stop_reason for tr in result.task_results] == ["timeout"] * 3


@pytest.mark.asyncio
async def test_attacker_spend_survives_the_cancellation() -> None:
    """Money spent before the cap is real money and must be recorded.

    The LLM client lives inside the cancelled coroutine, so without publishing
    its usage a timed-out task would report ``$0.00`` for a task that may have
    spent up to its whole cost cap.
    """

    class SpendingOptimizer(StubOptimizer):
        """Bills one LLM call per event it handles."""

        async def on_event(self, event: Event) -> EventResponse:
            await self.llm.complete([{"role": "user", "content": "x"}])
            return await super().on_event(event)

    response = MagicMock()
    response.usage = MagicMock()
    with (
        patch("anonframework.core.llm.completion_cost", return_value=0.25),
        patch("anonframework.core.llm.acompletion", new=AsyncMock(return_value=response)),
    ):
        controller = _controller(
            target_factory=TargetFactory(create=lambda: TruncatedTarget(runs_before_hang=1)),
            tasks=[StubTask(success=False)],
            cap=0.5,
            optimizer_factory=SpendingOptimizer,
        )
        result = await asyncio.wait_for(controller.run(), timeout=20)

    tr = result.task_results[0]
    assert tr.stop_reason == "timeout"
    assert tr.llm_usage.calls >= 1
    assert tr.llm_usage.cost > 0.0


@pytest.mark.asyncio
async def test_default_is_unbounded_and_changes_nothing() -> None:
    controller = _controller(
        target_factory=TargetFactory.singleton(StubTarget()),
        tasks=[StubTask()],
        cap=None,
        optimizer_factory=lambda: StubOptimizer(done=True),
    )
    result = await controller.run()
    assert result.task_results[0].stop_reason != "timeout"


@pytest.mark.asyncio
async def test_a_task_that_finishes_inside_the_cap_is_untouched() -> None:
    controller = _controller(
        target_factory=TargetFactory.singleton(StubTarget()),
        tasks=[StubTask()],
        cap=30,
        optimizer_factory=lambda: StubOptimizer(done=True),
    )
    result = await controller.run()
    # StubTask succeeds, so the claim's verdict ends the task before the
    # optimizer's own done is consulted. The point stands: not "timeout".
    assert result.task_results[0].stop_reason == "success"


# ---------------------------------------------------------------------------
# The stored record, and what a resume does with it
# ---------------------------------------------------------------------------


def _experiment_dir(results_root: Path) -> Path:
    dirs = [p for p in results_root.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def _only_task(results_root: Path) -> dict[str, Any]:
    task_dirs = list(iter_task_dirs(_experiment_dir(results_root)))
    assert len(task_dirs) == 1
    return load_task(task_dirs[0])


@pytest.mark.asyncio
async def test_a_truncated_task_is_stored_with_its_runs_and_its_cap(tmp_path: Path) -> None:
    controller = _controller(
        target_factory=TargetFactory(create=lambda: TruncatedTarget(runs_before_hang=2)),
        tasks=[StubTask(score=0.6, success=False)],
        cap=0.5,
        persist=True,
        results_dir=tmp_path,
    )
    await asyncio.wait_for(controller.run(), timeout=20)

    task = _only_task(tmp_path)
    assert task["status"] == "timeout"
    assert task["stop_reason"] == "timeout"
    assert task["n_runs"] == 2
    assert task["n_measured_runs"] == 2
    assert task["best_score"]["value"] == pytest.approx(0.6)
    # A kept timeout depends on a HOST parameter, so the record has to say
    # which cap truncated it -- identity_hash deliberately does not.
    assert task["task_time_cap_s"] == pytest.approx(0.5)

    exp_dir = _experiment_dir(tmp_path)
    manifest = load_manifest(exp_dir)
    assert manifest["experiment"]["task_time_cap_s"] == pytest.approx(0.5)
    assert manifest["summary"]["n_timeout"] == 1
    assert manifest["summary"]["n_timeout_empty"] == 0
    # The cap is a threat-model parameter, so running out of time without
    # succeeding is a failure, and it is counted as one.
    assert manifest["summary"]["n_completed"] == 1
    assert manifest["summary"]["n_failed"] == 1
    assert manifest["summary"]["asr"] == pytest.approx(0.0)
    # The trajectories of the completed runs are on disk, not discarded.
    traj_dir = exp_dir / manifest["tasks"][0]["dir"] / "trajectories"
    assert sorted(p.name for p in traj_dir.iterdir()) == ["run_00001.json", "run_00002.json"]
    assert json.loads((traj_dir / "run_00001.json").read_text())["trajectory"]


@pytest.mark.asyncio
async def test_a_task_cancelled_with_nothing_judged_is_stored_as_empty(tmp_path: Path) -> None:
    controller = _controller(
        target_factory=TargetFactory(create=HangingTarget),
        tasks=[StubTask()],
        cap=0.25,
        persist=True,
        results_dir=tmp_path,
    )
    await asyncio.wait_for(controller.run(), timeout=20)

    task = _only_task(tmp_path)
    assert task["status"] == "timeout_empty"
    assert task["stop_reason"] == "timeout"
    assert task["n_runs"] == 0
    assert task["n_measured_runs"] == 0
    assert load_manifest(_experiment_dir(tmp_path))["summary"]["n_timeout_empty"] == 1


@pytest.mark.asyncio
async def test_a_truncated_task_is_not_recomputed_by_a_resume(tmp_path: Path) -> None:
    """The convergence property: a reliably-slow task is measured once.

    Without it the documented "run it again until it exits clean" loop never
    terminates, because the task recomputes -- and re-bills -- on every pass.
    """

    def build() -> Controller:
        return _controller(
            target_factory=TargetFactory(create=lambda: TruncatedTarget(runs_before_hang=2)),
            tasks=[StubTask(score=0.6, success=False)],
            cap=0.5,
            persist=True,
            results_dir=tmp_path,
        )

    await asyncio.wait_for(build().run(), timeout=20)
    exp_dir = _experiment_dir(tmp_path)
    assert plan_resume(exp_dir, ["Test goal"], overwrite=False).keep == frozenset({1})

    # Second pass: the task is kept, so nothing is recomputed or re-billed.
    result = await asyncio.wait_for(build().run(), timeout=20)
    tr = result.task_results[0]
    assert tr.stop_reason == "timeout"
    assert tr.n_runs == 2
    assert _only_task(tmp_path)["status"] == "timeout"


@pytest.mark.asyncio
async def test_a_task_that_measured_nothing_is_recomputed_by_a_resume(tmp_path: Path) -> None:
    """The other half: an empty record is never kept as though it were a measurement."""
    controller = _controller(
        target_factory=TargetFactory(create=HangingTarget),
        tasks=[StubTask()],
        cap=0.25,
        persist=True,
        results_dir=tmp_path,
    )
    await asyncio.wait_for(controller.run(), timeout=20)

    plan = plan_resume(_experiment_dir(tmp_path), ["Test goal"], overwrite=False)
    assert plan.keep == frozenset()
    assert plan.rerun == frozenset({1})


def test_a_truncated_task_does_not_print_as_a_failed_attack() -> None:
    """The console must not show a task the cap cut short as a verdict."""

    class Sink:
        def __init__(self) -> None:
            self.text = ""

        def write(self, s: str) -> int:
            self.text += s
            return len(s)

        def flush(self) -> None:
            return None

    out = Sink()
    reporter = PlainReporter(file=cast("IO[str]", out), err=cast("IO[str]", Sink()))
    reporter.on_task_complete(
        TaskCompleteEvent(
            task_index=1,
            goal="g",
            success=True,
            best_score=1.0,
            n_runs=2,
            stop_reason="timeout",
            cost_usd=0.0,
        )
    )
    assert "TIMEOUT" in out.text


@pytest.mark.asyncio
async def test_an_interrupt_is_not_pinned_by_a_hanging_teardown() -> None:
    """Ctrl-C must not be swallowed by cleanup that will not return.

    The cleanup budget keys on a cancellation being in flight rather than on
    the cap having fired, so an outer cancellation of the run bounds cleanup
    the same way the cap does.  Keyed on the cap alone, a target whose teardown
    blocks pinned the interrupt forever.
    """
    controller = _controller(
        target_factory=TargetFactory(create=HangingTeardownTarget),
        tasks=[StubTask()],
        cap=60.0,  # generous; the interrupt, not the cap, is what bounds this
    )
    with patch("anonframework.core.controller._CLEANUP_GRACE_S", 0.2):
        run = asyncio.ensure_future(controller.run())
        await asyncio.sleep(0.2)  # let the task park inside target.run()
        run.cancel()
        done, _pending = await asyncio.wait({run}, timeout=10)

    assert done, "the interrupt was pinned by a teardown that never returns"
