"""Controller: orchestrates one red-teaming threat model.

One ``Controller`` instance evaluates one security claim against one
``(scope, llm_config)`` threat model.  Each task in the claim runs
against a fresh :class:`Target` from the configured
:class:`TargetFactory`; tasks run concurrently up to
``target_factory.concurrency`` at a time, with each task's events,
responses, and trajectory entries recorded on its own trajectory — the
single source of truth per run.

To sweep multiple scopes or attacker models, instantiate one
``Controller`` per threat model at the experiment level and run them
sequentially or via ``asyncio.gather``.

Usage::

    target_factory = TargetFactory(
        create=lambda: MyTarget(...),
        concurrency=8,
    )
    controller = Controller(
        optimizer_factory=lambda: MyOptimizer(),
        target_factory=target_factory,
        security_claim=claim,
        scope=my_scope,
        llm_config=my_llm_config,  # or omit for non-LLM optimizers
    )
    result = await controller.run()  # -> ThreatModelResult
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import traceback
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal, cast

from anonframework.core.channel import EventChannel
from anonframework.core.interfaces.optimizer import Optimizer
from anonframework.core.interfaces.security_claim import SecurityClaim
from anonframework.core.interfaces.target import Target
from anonframework.core.interfaces.task import NotApplicable, Task
from anonframework.core.llm import LLMClient
from anonframework.core.middleware import compose, security_domain_filter, trajectory_recorder
from anonframework.core.persistence import (
    ExperimentMeta,
    ExperimentSession,
    reconstruct_kept_task_result,
    resolve_results_root,
)
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
    current_task,
    resolve_reporter,
    should_use_plain,
)
from anonframework.core.types.controllable import Controllable
from anonframework.core.types.evaluation import EvaluationResult, Score
from anonframework.core.types.events import RunEndEvent, RunEndResponse, RunStartEvent
from anonframework.core.types.llm import BudgetExhaustedError, LLMConfig, LLMUsage
from anonframework.core.types.observable import Observable, ObservableValue
from anonframework.core.types.security_domain import Scope, scope_includes
from anonframework.core.types.trajectory import Trajectory

logger = logging.getLogger(__name__)

# Type alias for optimizer factories.
OptimizerFactory = Callable[[], Optimizer]

# Type alias for a per-task scope resolver: given the Task about to run,
# return the read & write Scope to enforce for it.  Pass one as ``scope`` to
# the Controller (in place of a fixed ``Scope``) for dynamic per-task scoping;
# resolvers must return the target's exported tag singletons (scope matching is
# by identity).
ScopeResolver = Callable[[Task[Target]], Scope]

# Reason a task's run loop ended.
StopReason = Literal["success", "done", "max_runs", "budget_exhausted", "error", "timeout"]


@dataclass(frozen=True)
class _TaskScope:
    """The security scope enforced for one task (resolved once per task).

    ``write`` is the read & write (injectable) scope, ``read_only`` the
    visible-but-not-injectable tags, and ``visibility = write | read_only`` is
    everything the optimizer can see.  In static-scope mode every task shares
    the same value; with a per-task resolver ``write`` is resolved per task.
    """

    write: Scope
    read_only: Scope
    visibility: Scope


# ---------------------------------------------------------------------------
# Target factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetFactory:
    """Produces fresh :class:`Target` instances and declares parallel capacity.

    The controller calls :attr:`create` once per task and runs up to
    :attr:`concurrency` tasks in parallel within a threat model.  Each
    task owns its target's full lifecycle: ``configure_target`` →
    ``run``/``reset_ephemeral_state`` loop → ``teardown``.

    Attributes:
        create: Zero-arg callable that returns a new :class:`Target`.
            The controller calls this once per task; the instance is
            discarded after ``teardown``.
        concurrency: Maximum tasks that may run in parallel against
            independent target instances from this factory.  Defaults to
            ``1`` (sequential).  Target authors choose this based on
            external rate limits or resource cost — a cheap, stateless
            target (a chatbot wrapping an API) can comfortably use ``8``
            or more; a target that boots a sandbox should usually stay
            at ``1`` unless it pools internally.
    """

    create: Callable[[], Target]
    concurrency: int = 1

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("TargetFactory.concurrency must be at least 1")

    @classmethod
    def singleton(cls, target: Target) -> TargetFactory:
        """Wrap a single target instance so the factory always returns it.

        Concurrency is locked to ``1`` because a shared instance cannot
        safely serve parallel tasks — mutable target state would race.

        Intended for tests and small migrations.  Note that the controller
        still calls ``target.teardown()`` once per task in the security
        claim, so a singleton-wrapped target needs an idempotent
        ``teardown()`` (or a no-op one) if the claim has more than one
        task.  For real targets that hold expensive resources, prefer a
        non-singleton factory whose ``create()`` returns a fresh instance
        each call.
        """
        return cls(create=lambda: target, concurrency=1)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Result of a single optimizer run (one target execution + evaluation).

    Attributes:
        trajectory: The run trajectory.
        evaluation: The evaluation result for this run.
        llm_usage: Cumulative optimizer LLM usage after this run (a running
            total across the task's runs, not this run's delta).
        run_usage_delta: This run's own usage (``llm_usage`` minus the
            previous run's cumulative snapshot).  Summing deltas across a
            task's runs equals the task total; summing ``llm_usage`` does
            not (it would multiply-count the cumulative snapshots).
        started_at: Wall-clock UTC when this run started, or ``None``.
        ended_at: Wall-clock UTC when this run finished, or ``None``.
        evaluated: ``True`` if the score came from the evaluator; ``False``
            for the synthetic zero appended on an error/budget path.
        errored: ``True`` if the run raised mid-execution.
        done: Whether the optimizer signalled it wanted to stop after this run.
    """

    trajectory: Trajectory
    evaluation: EvaluationResult
    llm_usage: LLMUsage
    run_usage_delta: LLMUsage = LLMUsage()
    started_at: datetime | None = None
    ended_at: datetime | None = None
    evaluated: bool = True
    errored: bool = False
    done: bool = False


@dataclass(frozen=True)
class TaskResult:
    """Result of evaluating a single task across all optimizer runs.

    Attributes:
        task: The task that was evaluated.
        runs: All run results, in order.  When the task ended with
            ``stop_reason="error"`` due to a failure inside a run, the
            run-in-progress is appended with the partial trajectory it
            had accumulated and a zero-score :class:`EvaluationResult`.
        best_score: Highest primary score achieved across all runs.
        best_evaluation: The EvaluationResult that produced the best score.
        success: Whether any run achieved the adversarial goal.
        llm_usage: Total optimizer LLM usage across all runs.
        stop_reason: Why the run loop ended.  ``"success"`` means the
            security claim judged a run successful and the controller ended
            the task; the optimizer's own ``done`` is not consulted in that
            case, so this is distinguishable from an attacker that stopped
            itself after winning.  ``"done"`` means the optimizer
            returned ``RunEndResponse(done=True)``.  ``"max_runs"`` means
            the safety cap ``max_runs_per_task`` was reached.
            ``"budget_exhausted"`` means a :class:`BudgetExhaustedError`
            was raised by the LLM client.  ``"error"`` means an unexpected
            exception escaped the optimizer, target, or evaluator and the
            task was abandoned.  ``"timeout"`` means the per-task wall-clock
            cap ``task_time_cap_s`` expired and the task was cancelled;
            ``runs`` then holds every run that had completed before the cap
            (possibly none), and ``best_evaluation`` is the judge's own
            verdict on the best of them.
        scope: The read & write scope actually enforced for this task.  In
            static-scope mode it equals the controller's ``scope``; with a
            per-task resolver it is the scope resolved for this task (the
            source of truth, since ``ThreatModelResult.scope`` is empty then).
        read_only: The visible-but-not-injectable tags enforced for this task.
        error: Formatted exception (type + message + traceback) when the
            task ended with ``stop_reason="error"``; ``None`` otherwise.
            Lands in the persisted JSON for offline debugging.
    """

    task: Task[Target]
    runs: list[RunResult]
    best_score: Score
    best_evaluation: EvaluationResult
    success: bool
    llm_usage: LLMUsage
    stop_reason: StopReason
    scope: Scope = frozenset()
    read_only: Scope = frozenset()
    error: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    # Explicit run count for a resumed (lightweight) result whose ``runs`` list
    # is empty; ``None`` means use ``len(runs)`` (the normal fresh-run case).
    n_runs: int | None = None


@dataclass(frozen=True)
class ThreatModelResult:
    """Results for a single threat model (scope + LLM config combination).

    Attributes:
        scope: The read & write scope tested (visible and injectable).  In
            dynamic (per-task resolver) mode there is no single run scope, so
            this is empty and ``scope_label`` carries the run identity; the
            scope actually enforced per task lives on ``TaskResult.scope``.
        read_only: Extra visible-but-not-injectable tags (empty for an
            all-read & write run, and empty in dynamic mode).
        llm_config: The LLM configuration used, or ``None`` when no LLM
            configs were provided.
        task_cost_cap_usd: The attacker's per-task cost cap in USD (from the
            Controller), or ``None`` for unlimited.
        task_results: Results for each evaluated task.
        skipped_tasks: Tasks that raised NotApplicable during configure.
        scope_label: Run identity when scope is resolved per task (the
            ``scope_label`` passed to the Controller); ``None`` in static mode.
    """

    scope: Scope
    read_only: Scope
    llm_config: LLMConfig | None
    task_results: list[TaskResult]
    task_cost_cap_usd: float | None = None
    skipped_tasks: list[Task[Target]] = field(default_factory=list)
    scope_label: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


def _format_exception(exc: BaseException) -> str:
    """Format an exception with type, message, and traceback for the JSON log."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _controllable_as_observable(controllable: Controllable) -> ObservableValue:
    """Re-present a read-only controllable as an observable for the optimizer.

    When a controllable's tag is visible but not in the write scope, the
    optimizer can read the surface but never inject into it — so for this
    threat model it is an observable, not a controllable.  Surfacing it in
    the ``observables`` list (rather than ``controllables``) keeps the
    optimizer's view honest: ``controllables`` means "injectable".

    ``content`` is ``None`` because the value is not known statically; it is
    revealed at runtime on the trajectory via the controllable's (declined)
    events.  The name and security domain are preserved so the optimizer can
    still identify the surface.
    """
    return ObservableValue(
        observable=Observable(
            name=controllable.name,
            security_domain=controllable.security_domain,
            description=controllable.description,
            observable_type=controllable.value_type,
        ),
        content=None,
    )


def _synthesize_empty_task_result(
    task: Task[Target],
    *,
    stop_reason: StopReason,
    rationale: str,
    usage: LLMUsage = LLMUsage(),
    exc: BaseException | None = None,
    scope: Scope = frozenset(),
    read_only: Scope = frozenset(),
) -> TaskResult:
    """Build a zero-score TaskResult for a task whose run loop never produced a run.

    Used by the error and budget-exhausted paths that abandon a task before
    any run completes. When *exc* is provided its formatted traceback is
    stored on ``TaskResult.error`` so the failure is recoverable from the
    persisted detail file.  *scope*/*read_only* record the scope that was (or
    would have been) enforced for the task; they are empty when the failure
    happened before the per-task scope could be resolved.
    """
    zero = Score(value=0.0, name="primary")
    return TaskResult(
        task=task,
        runs=[],
        best_score=zero,
        best_evaluation=EvaluationResult(
            success=False,
            primary_score=zero,
            sub_scores={},
            rationale=rationale,
        ),
        success=False,
        llm_usage=usage,
        stop_reason=stop_reason,
        scope=scope,
        read_only=read_only,
        error=_format_exception(exc) if exc is not None else None,
    )


@dataclass
class _TaskProgress:
    """Runs a task has completed so far, readable from outside its coroutine.

    ``_run_task`` accumulates into this instead of into locals, so a
    cancellation (the wall-clock cap) does not destroy work that was already
    finished and judged.  Without it, everything a timed-out task achieved dies
    with the frame and the record cannot distinguish a task that measured
    nothing from one that measured plenty and was cut off at the end.
    """

    runs: list[RunResult] = field(default_factory=list)
    llm_client: LLMClient | None = None

    @property
    def usage(self) -> LLMUsage:
        """Attacker spend so far -- real money, even when the task is cancelled."""
        return self.llm_client.usage if self.llm_client is not None else LLMUsage()

    # Derived, not tracked: the runs already determine both, and maintaining
    # them alongside would be a second copy to keep in step. Only JUDGED runs
    # count -- an errored run carries a synthesized evaluation, never a verdict.

    @property
    def best(self) -> tuple[Score, EvaluationResult] | None:
        """Score and verdict of the highest-scoring judged run, if any."""
        judged = [r for r in self.runs if r.evaluated]
        if not judged:
            return None
        b = max(judged, key=lambda r: r.evaluation.primary_score.value)
        return b.evaluation.primary_score, b.evaluation

    @property
    def success(self) -> bool:
        """Whether any judged run met the goal."""
        return any(r.evaluation.success for r in self.runs if r.evaluated)


def _truncated_task_result(
    task: Task[Target],
    progress: _TaskProgress,
    *,
    rationale: str,
    scope: Scope = frozenset(),
    read_only: Scope = frozenset(),
) -> TaskResult:
    """Build the TaskResult for a task cancelled at the wall-clock cap.

    Carries every run the task had completed, with their judged evaluations and
    trajectories, and the attacker spend they cost.  ``best_evaluation`` stays
    the judge's own verdict on the best run -- the cap is reported through
    ``stop_reason``, never by overwriting a real evaluation.  When nothing
    completed, the result is a zero-run one whose *rationale* names the cap, and
    persistence records it under a status a resume will recompute.
    """
    zero = Score(value=0.0, name="primary")
    best = progress.best
    best_evaluation = (
        best[1]
        if best is not None
        else EvaluationResult(success=False, primary_score=zero, sub_scores={}, rationale=rationale)
    )
    return TaskResult(
        task=task,
        runs=list(progress.runs),
        best_score=best[0] if best is not None else zero,
        best_evaluation=best_evaluation,
        success=progress.success,
        llm_usage=progress.usage,
        stop_reason="timeout",
        scope=scope,
        read_only=read_only,
    )


#: Budget for the cleanup that follows a task, when ``task_time_cap_s`` is set.
#: Cleanup runs in the ``finally`` of the coroutine the cap cancels, so it
#: cannot rely on a second cancellation to interrupt it -- the cap fires once.
#: Every await down there therefore needs its own bound, or the cap does not
#: hold. ``_run_task`` spends one budget across the optimizer join and the
#: optimizer teardown; ``_execute_task`` spends another on the target teardown.
_CLEANUP_GRACE_S = 5.0

#: Extra time a cancelled cleanup gets to finish unwinding before it is left
#: to a done-callback. Short: this is only about letting a well-behaved
#: cancellation land before the caller touches the same object again.
_ABANDON_SETTLE_S = 0.5


def _cancellation_in_flight() -> bool:
    """True when the running task is unwinding a cancellation.

    ``asyncio.timeout`` cancels the task it guards, so inside the ``finally``
    this reads 1 for a task the wall-clock cap cut short and 0 for one that
    finished on its own. The counter is only decremented by ``uncancel()`` in
    the timeout's ``__aexit__``, which runs after this frame is gone.
    """
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _log_abandoned(description: str, fut: asyncio.Future[None]) -> None:
    """Consume an abandoned cleanup's outcome so asyncio never reports it bare."""
    if fut.cancelled():
        return
    exc = fut.exception()
    if exc is not None:
        logger.error("%s failed after being abandoned", description, exc_info=exc)


async def _finish_or_abandon(
    work: Awaitable[None] | asyncio.Task[None],
    description: str,
    deadline: float,
) -> BaseException | None:
    """Await *work* until *deadline*; past it, cancel *work* and stop waiting.

    Cleanup that runs in the ``finally`` of a coroutine the wall-clock cap has
    already cancelled has nothing left to interrupt it: the cap fires once, so
    a fresh suspend down there is unbounded no matter what cancelled the body.
    That is how an optimizer -- or its teardown -- blocked on something the
    channel close cannot reach pins a task past every configured bound.

    Abandoning rather than awaiting the cancellation is deliberate.  Awaiting a
    cancel the work is slow to honour reintroduces the unbounded wait this
    exists to remove; an abandoned task is cancelled and simply not joined.

    The caller's own cancellation is never swallowed: *work* is cancelled so it
    is not left running behind us, and the exception propagates.

    Args:
        work: Coroutine or task to finish.
        description: Used in the log line if *work* has to be abandoned.
        deadline: Loop-clock time to stop waiting, or ``math.inf`` for no bound.

    Returns:
        The exception *work* completed with, or ``None`` if it completed
        cleanly, was cancelled, or was abandoned.
    """
    loop = asyncio.get_running_loop()
    fut = asyncio.ensure_future(work)
    timeout = None if deadline == math.inf else max(0.0, deadline - loop.time())
    try:
        done, _pending = await asyncio.wait({fut}, timeout=timeout)
    except BaseException:
        fut.cancel()
        raise
    if not done:
        logger.warning("%s did not finish within the cleanup budget, abandoning it", description)
        fut.cancel()
        # Give the cancellation a brief, bounded chance to land. Work that
        # honours it promptly -- the normal case -- has finished unwinding by
        # the time we return, so a caller that goes on to touch the same object
        # is not racing it. Work that does not is left to the callback below.
        await asyncio.wait({fut}, timeout=_ABANDON_SETTLE_S)
        if not fut.done():
            # Nothing will retrieve this future's outcome. Without a callback,
            # a coroutine that turns the cancellation into some other exception
            # surfaces at garbage-collection time as an unattributed asyncio
            # "Task exception was never retrieved", with no clue which cleanup
            # it came from.
            fut.add_done_callback(partial(_log_abandoned, description))
        elif not fut.cancelled() and fut.exception() is not None:
            logger.error("%s failed while being abandoned", description, exc_info=fut.exception())
        return None
    if fut.cancelled():
        return None
    return fut.exception()


async def _swallow(coro: Awaitable[None], description: str) -> None:
    """Await *coro*, logging and swallowing any Exception.

    Used in error paths and ``finally`` blocks where a failing cleanup
    must not mask an in-flight exception or block the next task.
    """
    try:
        await coro
    except Exception:
        logger.exception("%s failed", description)


# ---------------------------------------------------------------------------
# Reporter-event builders (result objects -> progress payloads)
# ---------------------------------------------------------------------------


def _usage_delta(prev: LLMUsage, cur: LLMUsage) -> LLMUsage:
    """This run's own usage: the cumulative snapshot minus the previous one."""
    return LLMUsage(calls=cur.calls - prev.calls, cost=cur.cost - prev.cost)


def _run_complete_event(index: int, goal: str, run_number: int, run: RunResult) -> RunCompleteEvent:
    return RunCompleteEvent(
        task_index=index,
        goal=goal,
        run_number=run_number,
        primary_score=run.evaluation.primary_score.value,
        success=run.evaluation.success,
        evaluated=run.evaluated,
        errored=run.errored,
        done=run.done,
        run_cost_delta_usd=run.run_usage_delta.cost,
        cumulative_cost_usd=run.llm_usage.cost,
    )


def _task_complete_event(index: int, tr: TaskResult) -> TaskCompleteEvent:
    return TaskCompleteEvent(
        task_index=index,
        goal=tr.task.goal.description,
        success=tr.success,
        stop_reason=tr.stop_reason,
        best_score=tr.best_score.value,
        # n_runs override: a kept (resumed) task carries runs=[] in memory but
        # its real count on disk, so the reported count matches a fresh run.
        n_runs=tr.n_runs if tr.n_runs is not None else len(tr.runs),
        cost_usd=tr.llm_usage.cost,
        calls=tr.llm_usage.calls,
        error=tr.error,
    )


def _threat_model_end_event(
    ctx: ThreatModelContext,
    result: ThreatModelResult,
    started_at: datetime,
    ended_at: datetime,
) -> ThreatModelEndEvent:
    trs = result.task_results
    completed_reasons = ("success", "done", "max_runs", "budget_exhausted")
    # Count a success only among completed tasks: a task can be success=True yet
    # stop_reason="error" (goal met, then reset_ephemeral_state failed), so an
    # unguarded numerator would push ASR above 100%.  "success" must be in this
    # tuple: it is the reason a won task ends, so omitting it would
    # drop every win from BOTH the numerator and the denominator.
    n_success = sum(1 for t in trs if t.success and t.stop_reason in completed_reasons)
    n_completed = sum(1 for t in trs if t.stop_reason in completed_reasons)
    n_error = sum(1 for t in trs if t.stop_reason == "error")
    n_budget = sum(1 for t in trs if t.stop_reason == "budget_exhausted")
    n_timeout = sum(1 for t in trs if t.stop_reason == "timeout")
    scores = [t.best_score.value for t in trs]
    return ThreatModelEndEvent(
        context=ctx,
        n_tasks=len(trs),
        n_success=n_success,
        n_completed=n_completed,
        n_error=n_error,
        n_budget_exhausted=n_budget,
        n_timeout=n_timeout,
        n_skipped=len(result.skipped_tasks),
        asr=(n_success / n_completed) if n_completed else None,
        max_primary_score=max(scores) if scores else None,
        mean_primary_score=(sum(scores) / len(scores)) if scores else None,
        total_calls=sum(t.llm_usage.calls for t in trs),
        total_cost_usd=sum(t.llm_usage.cost for t in trs),
        duration_s=(ended_at - started_at).total_seconds(),
    )


def _aborted_end_event(
    ctx: ThreatModelContext, started_at: datetime, ended_at: datetime
) -> ThreatModelEndEvent:
    """A zeroed end event for a run that aborted before producing a result.

    Emitted from ``run()``'s ``finally`` so the reporter lane always ends (and
    the shared live canvas is freed) even when the run raised mid-way.
    """
    return ThreatModelEndEvent(
        context=ctx,
        n_tasks=0,
        n_success=0,
        n_completed=0,
        n_error=0,
        n_budget_exhausted=0,
        n_skipped=0,
        asr=None,
        max_primary_score=None,
        mean_primary_score=None,
        total_calls=0,
        total_cost_usd=0.0,
        duration_s=(ended_at - started_at).total_seconds(),
    )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class Controller:
    """Orchestrates one threat model: a single security claim evaluated
    against a single ``(scope, llm_config)`` combination.

    One ``Controller`` instance is one threat model.  To sweep multiple
    scopes or attacker models, construct one ``Controller`` per
    combination at the experiment level and ``asyncio.gather`` them (or
    iterate sequentially).

    Within the threat model, tasks run concurrently up to
    ``target_factory.concurrency`` at a time.  Each task owns its
    target's full lifecycle so concurrent tasks never share mutable
    target state.

    Args:
        optimizer_factory: A callable that returns a new :class:`Optimizer`
            instance.  A fresh optimizer is created for each task.
        target_factory: Produces fresh :class:`Target` instances and
            declares how many tasks may run in parallel against
            independent instances (see :class:`TargetFactory`).
        security_claim: The collection of tasks to evaluate.
        scope: Either a fixed read & write security domain scope (a
            ``frozenset[SecurityDomainTag]`` the attacker can both *see* and
            *inject* into) applied to every task, OR a ``ScopeResolver``
            (``Callable[[Task], Scope]``) resolved once per task to vary the
            scope per task (e.g. grant only the task-relevant tool).  A
            resolver MUST return the target's exported tag singletons (scope
            matching is by identity) and requires ``scope_label`` (below).  It
            may raise ``NotApplicable`` (equivalent to returning an empty set),
            contributing no write tags; the task is skipped only when this
            leaves the total visibility (``scope | read_only``) empty.  Controllables,
            observables, trajectory entries, and feedback sub-scores under the
            (resolved) scope tags and their descendants are exposed to the
            optimizer, and its controllable events are offered for injection.
        read_only: Extra tags the attacker can *see* but not inject into.
            Empty (default) means the whole ``scope`` is read & write
            (the classic behavior).  Tags here (and their descendants)
            are visible on every surface — trajectory, observables,
            feedback — but their controllable events are answered with
            ``ControllableNoInjection`` without consulting the optimizer.
            Put a read & write tag in ``scope`` and a visible-only
            ancestor in ``read_only`` to make just that subtree
            injectable (e.g. ``scope={system_prompt}, read_only={system}``).
            A ``read_only`` tag already covered by ``scope`` has no effect:
            read & write overrules (only ``scope`` drives the injection
            decision), so the tag stays injectable.  Two fixed scopes cannot
            both be empty (a construction error).  Like ``scope``, ``read_only``
            may instead be a ``ScopeResolver`` resolved once per task
            (independently of ``scope``); it then requires ``scope_label`` and
            its ``NotApplicable`` (or an empty return) contributes no read-only
            tags.  A task is skipped only when the resolved visibility
            (``scope | read_only``) is empty; any tag in either dimension runs it.
        llm_config: LLM access configuration for the optimizer, or
            ``None`` for non-LLM optimizers (in which case the optimizer
            receives a noop client that raises on any call).
        task_time_cap_s: Optional per-task WALL-CLOCK cap in seconds. ``None``
            (default) means unbounded. Where ``task_cost_cap_usd`` and
            ``max_runs_per_task`` bound the WORK a task may do, this bounds the
            TIME it may take, which is the only thing that stops a task whose
            provider call blocks without ever returning or timing out. On
            expiry the task is cancelled and recorded with
            ``stop_reason="timeout"``, KEEPING every run it had already
            completed and had judged, together with their trajectories and the
            attacker spend they cost. Truncation is not data loss.

            The cap bounds the task's own work; the cleanup that follows is
            bounded separately, because it runs in the ``finally`` of the
            coroutine the cap cancels and so cannot be interrupted by the cap
            again. Setting a cap therefore also bounds cleanup: the optimizer
            join and the optimizer teardown share one ``_CLEANUP_GRACE_S``
            budget, and the target teardown gets its own, so a cancelled task
            can overrun its cap by at most twice that. Work that does not
            finish inside its budget is cancelled and abandoned rather than
            joined. The budget is keyed on a cancellation actually being in
            flight -- normally the cap's, though an outer cancellation of the
            run (Ctrl-C) bounds cleanup the same way. A task that ends
            normally always gets an unbounded teardown, and with no cap set
            the controller never cancels, which is what ``None`` means.

            A TimeoutError the task raises itself -- a provider client read
            timeout, say -- is recorded as an error, not as cap expiry.

            The record then splits by what survived. With at least one judged
            run the persisted status is ``"timeout"``: a real measurement,
            truncated, kept by a resume exactly as ``"budget_exhausted"`` is.
            It counts in the ASR denominator as a non-success: an attacker
            that has not succeeded inside its time budget has failed under the
            threat model being measured, exactly as one that exhausted its cost
            budget has.

            With nothing judged the status is ``"timeout_empty"``: no
            measurement at all, so a resume recomputes it and it is excluded
            from the ASR entirely. After a full time budget with nothing judged,
            a hung provider call is far likelier than an attacker working to the
            wire, and counting an outage as attacker failure is the one error
            this must not make.

            Deliberately NOT part of the experiment identity (it does not
            appear in ``ExperimentMeta.identity_hash``), unlike the cost and run
            caps. Those are machine-independent properties of the measurement;
            a wall-clock cap is a property of the HOST, so folding it into the
            identity would re-key every result whenever the experiment moved to
            a faster or slower machine. Since a truncated task IS kept, the cap
            is instead recorded in the manifest, in ``result.json`` and in every
            task record, so a reader can always tell which cap truncated what.
        task_cost_cap_usd: Per-task cost cap in USD for the attacker's
            optimizer LLM. A fresh client is built per task, so this bounds
            the attacker's cumulative spend *per task* and resets each task
            (a full run costs up to about ``num_tasks * task_cost_cap_usd``).
            ``None`` (default) means unlimited. It applies only to the
            attacker: the judge and target are never bounded by it.
        max_runs_per_task: Safety limit on runs per task. ``None`` (default)
            uses the built-in cap of 100; pass an explicit positive int to
            override.
        results_dir: Optional directory for persisted artifacts. When set,
            the completed threat model is written atomically to
            ``{results_dir}/{scope}__{model}.json`` (plus a sibling
            subfolder with per-task detail files) when ``run()`` finishes.
            In dynamic-scope mode the ``{scope}`` stem is the ``scope_label``
            and each detail file records that task's own resolved scope.
            ``LLMConfig.api_key`` and ``api_base`` are excluded;
            trajectory contents are not scrubbed for secrets.
        scope_label: Required (and only valid) when ``scope`` or ``read_only``
            is a resolver: a short path-safe name identifying the run, used for
            the persisted filename stem and ``ThreatModelResult.scope_label``
            (since no single concrete scope names the run).  Must be ``None``
            when both are fixed scopes.
    """

    DEFAULT_MAX_RUNS_PER_TASK = 100

    def __init__(
        self,
        optimizer_factory: OptimizerFactory,
        target_factory: TargetFactory,
        security_claim: SecurityClaim[Target],
        scope: Scope | ScopeResolver,
        read_only: Scope | ScopeResolver = frozenset(),
        llm_config: LLMConfig | None = None,
        task_cost_cap_usd: float | None = None,
        task_time_cap_s: float | None = None,
        max_runs_per_task: int | None = None,
        include_feedback: bool = True,
        results_dir: str | Path | None = None,
        scope_label: str | None = None,
        persist: bool = True,
        overwrite: bool = False,
        report: bool | Literal["auto"] = "auto",
        attacker_label: str | None = None,
        target_label: str | None = None,
        claim_label: str | None = None,
    ) -> None:
        # ``scope`` (read & write: visible AND injectable; the classic
        # behavior) and ``read_only`` (visible-but-not-injectable tags) may
        # EACH be a fixed Scope or a Callable[[Task], Scope] resolved once per
        # task (dynamic per-task scoping), resolved independently.  The run is
        # "dynamic" when EITHER is a resolver; ``scope_label`` then names the
        # persisted artifacts (no single concrete scope exists to name the
        # run): required then, forbidden when both are fixed.
        scope_is_resolver = callable(scope)
        read_only_is_resolver = callable(read_only)
        self._dynamic_scope: bool = scope_is_resolver or read_only_is_resolver
        self._scope_label: str | None = scope_label
        # Capture the fixed value for whichever of scope / read_only is not a
        # resolver; the resolver path ignores these.
        static_scope: Scope = frozenset() if scope_is_resolver else cast("Scope", scope)
        static_read_only: Scope = frozenset() if read_only_is_resolver else cast("Scope", read_only)
        self._resolve_write_scope: ScopeResolver = (
            cast("ScopeResolver", scope) if scope_is_resolver else (lambda _task: static_scope)
        )
        self._resolve_read_only: ScopeResolver = (
            cast("ScopeResolver", read_only)
            if read_only_is_resolver
            else (lambda _task: static_read_only)
        )
        if self._dynamic_scope:
            if not (scope_label and scope_label.strip()):
                raise ValueError(
                    "scope_label is required (non-empty) when scope or read_only "
                    "is a callable resolver"
                )
            # No single run-level scope; each TaskResult records the resolved
            # scope and the naming scope is empty.
            self._naming_scope: Scope = frozenset()
            self._naming_read_only: Scope = frozenset()
        else:
            if not (static_scope | static_read_only):
                raise ValueError("scope and read_only cannot both be empty")
            if scope_label is not None:
                raise ValueError(
                    "scope_label is only valid when scope or read_only is a callable resolver"
                )
            self._naming_scope = static_scope
            self._naming_read_only = static_read_only
        resolved_max_runs = (
            self.DEFAULT_MAX_RUNS_PER_TASK if max_runs_per_task is None else max_runs_per_task
        )
        if resolved_max_runs < 1:
            raise ValueError("max_runs_per_task must be at least 1")
        if task_cost_cap_usd is not None and task_cost_cap_usd < 0:
            raise ValueError("task_cost_cap_usd must be non-negative")
        if task_time_cap_s is not None and task_time_cap_s <= 0:
            raise ValueError("task_time_cap_s must be positive")
        self._optimizer_factory = optimizer_factory
        self._target_factory = target_factory
        self._security_claim = security_claim
        self._llm_config: LLMConfig | None = llm_config
        self._task_cost_cap_usd: float | None = task_cost_cap_usd
        self._task_time_cap_s: float | None = task_time_cap_s
        self._max_runs_per_task = resolved_max_runs
        self._include_feedback = include_feedback
        self._results_dir: Path | None = Path(results_dir) if results_dir is not None else None
        self._persist = persist
        self._overwrite = overwrite
        self._report: bool | Literal["auto"] = report
        self._attacker_label = attacker_label
        self._target_label = target_label
        self._claim_label = claim_label

    def _task_scope_for(self, task: Task[Target]) -> _TaskScope:
        """Resolve the scope enforced for *task* (once per task).

        In static mode this returns the fixed scope for every task; per-task
        resolvers for ``scope`` and/or ``read_only`` are called here.  Either
        resolver raising ``NotApplicable`` contributes an empty set for its
        dimension, exactly like returning ``frozenset()``.  The task is SKIPPED
        (``NotApplicable`` propagates to ``run_one``) when the resolved
        visibility (``write | read_only``) is empty: no tag is granted in either
        dimension, so the attacker has no surface at all.  Any tag in either
        dimension (read or write, fixed or resolved) means the task runs.  A
        resolver raising any other exception is not caught here and surfaces as
        a per-task error in ``run_one``.
        """
        try:
            write = self._resolve_write_scope(task)
        except NotApplicable:
            write = frozenset()
        try:
            read_only = self._resolve_read_only(task)
        except NotApplicable:
            read_only = frozenset()
        visibility = write | read_only
        if not visibility:
            raise NotApplicable(
                f"no security domain tag resolved for task {task.goal.description!r}; skipping"
            )
        return _TaskScope(write=write, read_only=read_only, visibility=visibility)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        """This threat model's stable key: its results directory name
        ``{attacker}__{target}__{claim}__{model}-{hash8}``.  Used as the lane
        key in a shared dashboard and as the persistence directory stem."""
        return self._build_meta(n_tasks=0).dirname()

    @property
    def context(self) -> ThreatModelContext:
        """This threat model's display identity, used to pre-register a lane on a
        shared dashboard before the run starts (see :func:`run_all`).  Counts the
        security claim for ``n_tasks``; the claim is re-iterable, so ``run``
        re-counts it."""
        n_tasks = sum(1 for _ in self._security_claim)
        return self._build_context(self._build_meta(n_tasks=n_tasks))

    async def run(self, *, reporter: ProgressReporter | None = None) -> ThreatModelResult:
        """Evaluate the security claim under this controller's threat model.

        Streams live progress to a reporter (a shared rich dashboard on a TTY,
        plain lines otherwise) and, unless disabled with ``persist=False``,
        writes a self-describing result tree under the results root — resuming a
        prior run of the same experiment so only errored tasks recompute.

        Returns:
            A :class:`ThreatModelResult` for the configured ``(scope,
            llm_config)`` (kept + reran tasks merged in claim order).
        """
        tasks = list(self._security_claim)
        meta = self._build_meta(n_tasks=len(tasks))
        label = meta.dirname()
        ctx = self._build_context(meta)
        if reporter is None:
            reporter = resolve_reporter(label, report=self._report)
        started_at = datetime.now(UTC)
        reporter.on_threat_model_start(ctx)

        session: ExperimentSession | None = None
        result: ThreatModelResult | None = None
        try:
            if self._persist:
                root = resolve_results_root(self._results_dir)
                session = ExperimentSession.open(
                    root, meta, [t.goal.description for t in tasks], overwrite=self._overwrite
                )
                logger.info("anonframework: writing results to %s", session.experiment_dir)

            rerun = (
                set(session.plan.rerun) if session is not None else set(range(1, len(tasks) + 1))
            )
            keep = set(session.plan.keep) if session is not None else set()

            # Kept tasks are already on disk: pre-fill them into the live view
            # and the merged result without re-running or re-reading trajectories.
            kept_results: dict[int, TaskResult] = {}
            for i in sorted(keep):
                assert session is not None
                kept = reconstruct_kept_task_result(session.experiment_dir, i, tasks[i - 1])
                kept_results[i] = kept
                reporter.on_task_complete(_task_complete_event(i, kept))

            # Bridge Python logging -> reporter (side pane) + per-task JSONL sink
            # for the run; removed afterwards so the framework never leaves a
            # handler on the root logger.
            bridge = LoggingBridge(label, reporter, self._diagnostic_sink(session))
            root_logger = logging.getLogger()
            root_logger.addHandler(bridge)
            try:
                reran, skipped = await self._iterate_tasks(tasks, rerun, session, reporter, label)
            finally:
                root_logger.removeHandler(bridge)

            task_results = [
                reran[i] if i in reran else kept_results[i]
                for i in range(1, len(tasks) + 1)
                if i in reran or i in kept_results
            ]
            result = ThreatModelResult(
                scope=self._naming_scope,
                read_only=self._naming_read_only,
                llm_config=self._llm_config,
                task_cost_cap_usd=self._task_cost_cap_usd,
                task_results=task_results,
                skipped_tasks=[t for _, t in skipped],
                scope_label=self._scope_label,
                started_at=started_at,
                ended_at=datetime.now(UTC),
            )
            if session is not None:
                try:
                    session.finalize(started_at, result.ended_at)
                    session = None  # lock released by finalize; nothing to abort
                except Exception:
                    logger.exception("anonframework: failed to finalize results (continuing)")
            return result
        finally:
            # Always end the reporter lane (frees the shared live canvas) and
            # release the experiment lock, even if the run aborted before a
            # result was built (e.g. an unexpected I/O error opening the dir).
            ended_at = (
                result.ended_at
                if result is not None and result.ended_at is not None
                else datetime.now(UTC)
            )
            end_ev = (
                _threat_model_end_event(ctx, result, started_at, ended_at)
                if result is not None
                else _aborted_end_event(ctx, started_at, ended_at)
            )
            try:
                reporter.on_threat_model_end(end_ev)
            except Exception:
                logger.exception("anonframework: reporter on_threat_model_end failed")
            if session is not None:
                session.abort()

    # ------------------------------------------------------------------
    # Experiment identity + reporting helpers
    # ------------------------------------------------------------------

    def _attacker_name(self) -> str:
        # Never construct an optimizer just to name it (fresh-per-task is an
        # invariant, and construction may have side effects).  Use the label,
        # else a class/function factory's name, else a generic default.
        if self._attacker_label:
            return self._attacker_label
        name = getattr(self._optimizer_factory, "__name__", "")
        return name if name and name != "<lambda>" else "optimizer"

    def _build_meta(self, n_tasks: int) -> ExperimentMeta:
        return ExperimentMeta(
            attacker=self._attacker_name(),
            target=self._target_label or "target",
            claim=self._claim_label or type(self._security_claim).__name__,
            model=self._llm_config.model if self._llm_config else None,
            scope=tuple(sorted(t.name for t in self._naming_scope)),
            read_only=tuple(sorted(t.name for t in self._naming_read_only)),
            scope_label=self._scope_label,
            task_cost_cap_usd=self._task_cost_cap_usd,
            task_time_cap_s=self._task_time_cap_s,
            max_runs_per_task=self._max_runs_per_task,
            include_feedback=self._include_feedback,
            concurrency=self._target_factory.concurrency,
            n_tasks=n_tasks,
        )

    def _build_context(self, meta: ExperimentMeta) -> ThreatModelContext:
        return ThreatModelContext(
            label=meta.dirname(),
            attacker=meta.attacker,
            target=meta.target,
            claim=meta.claim,
            model=meta.model,
            scope=meta.scope,
            read_only=meta.read_only,
            scope_label=meta.scope_label,
            task_cost_cap_usd=meta.task_cost_cap_usd,
            max_runs_per_task=meta.max_runs_per_task,
            include_feedback=meta.include_feedback,
            concurrency=meta.concurrency,
            n_tasks=meta.n_tasks,
        )

    def _diagnostic_sink(
        self, session: ExperimentSession | None
    ) -> Callable[[DiagnosticEvent], None] | None:
        """Per-task JSONL log sink for the logging bridge (``None`` if no persist)."""
        if session is None:
            return None

        def sink(ev: DiagnosticEvent) -> None:
            try:
                line = json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "level": ev.level,
                        "logger_name": ev.logger_name,
                        "task_index": ev.task_index,
                        "message": ev.message,
                    }
                )
                with open(session.task_log_path(ev.task_index), "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:  # pragma: no cover - per-task log I/O must never break a run
                pass

        return sink

    # ------------------------------------------------------------------
    # Task iteration (one threat model)
    # ------------------------------------------------------------------

    async def _iterate_tasks(
        self,
        tasks: list[Task[Target]],
        rerun: set[int],
        session: ExperimentSession | None,
        reporter: ProgressReporter,
        label: str,
    ) -> tuple[dict[int, TaskResult], list[tuple[int, Task[Target]]]]:
        """Run the *rerun* set concurrently, streaming progress + persisting.

        Only tasks in *rerun* execute; kept tasks are pre-loaded by the caller.
        Each runs under a fresh target, with per-task diagnostics attributed via
        the ``current_task`` contextvar and, when a session is given, its own
        staging dir published atomically on completion.  Returns
        ``(reran_results_by_index, skipped)`` where *skipped* pairs the 1-based
        index with the ``Task`` that raised ``NotApplicable``.
        """
        sem = asyncio.Semaphore(self._target_factory.concurrency)

        async def run_one(
            index: int, task: Task[Target]
        ) -> tuple[str, int, TaskResult | Task[Target]]:
            async with sem:
                token = current_task.set((label, index))
                try:
                    reporter.on_task_start(
                        TaskStartEvent(task_index=index, goal=task.goal.description)
                    )
                    if session is not None:
                        session.begin_task(index, task.goal.description)
                    started = datetime.now(UTC)
                    outcome = await self._execute_task(index, task, reporter)
                    ended = datetime.now(UTC)
                    if isinstance(outcome, TaskResult):
                        outcome = replace(outcome, started_at=started, ended_at=ended)
                        if session is not None:
                            try:
                                session.publish_task(index, outcome)
                            except Exception:
                                logger.exception(
                                    "Task %r: failed to persist (continuing)",
                                    task.goal.description,
                                )
                        reporter.on_task_complete(_task_complete_event(index, outcome))
                        return "result", index, outcome
                    reporter.on_task_skipped(
                        TaskSkippedEvent(task_index=index, goal=task.goal.description)
                    )
                    if session is not None:
                        session.mark_skipped(index, task.goal.description)
                    return "skip", index, task
                except Exception as exc:
                    # An unexpected orchestration failure (a reporter callback,
                    # begin_task I/O, ...) must not abort the whole threat model
                    # via gather's first-exception behaviour: contain it as an
                    # error task so the other tasks still complete.
                    logger.exception(
                        "Task %r: orchestration error, recording as error",
                        task.goal.description,
                    )
                    tr = _synthesize_empty_task_result(
                        task,
                        stop_reason="error",
                        rationale="Unexpected orchestration error.",
                        exc=exc,
                    )
                    try:
                        reporter.on_task_complete(_task_complete_event(index, tr))
                    except Exception:  # pragma: no cover - reporter already failing
                        pass
                    return "result", index, tr
                finally:
                    current_task.reset(token)

        outcomes = await asyncio.gather(*(run_one(i, tasks[i - 1]) for i in sorted(rerun)))

        reran: dict[int, TaskResult] = {}
        skipped: list[tuple[int, Task[Target]]] = []
        for kind, index, obj in outcomes:
            if kind == "result":
                reran[index] = cast("TaskResult", obj)
            else:
                skipped.append((index, cast("Task[Target]", obj)))
        return reran, skipped

    async def _execute_task(
        self, index: int, task: Task[Target], reporter: ProgressReporter
    ) -> TaskResult | Task[Target]:
        """Resolve scope, create the target, run the loop, tear down.

        Contains every per-task error so one bad task cannot abort the threat
        model.  Returns a :class:`TaskResult` (success/error/budget) or the
        ``Task`` itself when it raised ``NotApplicable`` (skipped).
        """
        try:
            task_scope = self._task_scope_for(task)
        except NotApplicable:
            logger.info("Task %r not applicable (scope resolver), skipping", task.goal.description)
            return task
        except Exception as exc:
            logger.exception(
                "Task %r: scope resolver failed, recording as error", task.goal.description
            )
            return _synthesize_empty_task_result(
                task,
                stop_reason="error",
                rationale="Scope resolver failed before run loop started.",
                exc=exc,
            )
        try:
            target = self._target_factory.create()
        except Exception as exc:
            logger.exception(
                "Task %r: target_factory.create() failed, recording as error",
                task.goal.description,
            )
            return _synthesize_empty_task_result(
                task,
                stop_reason="error",
                rationale="Unexpected error before run loop started.",
                exc=exc,
                scope=task_scope.write,
                read_only=task_scope.read_only,
            )
        # Set when the wall-clock cap cut this task short; the `finally` below
        # uses it to decide whether the target teardown needs a bound.
        cap_expired = False
        try:
            try:
                progress = _TaskProgress()
                if self._task_time_cap_s is None:
                    return await self._run_task(task, target, task_scope, reporter, index, progress)
                # Wall-clock bound. The work bounds (task_cost_cap_usd,
                # max_runs_per_task) cannot stop a task whose provider call
                # blocks forever: no spend accrues and no run completes, so the
                # task hangs indefinitely. wait_for cancels the whole task
                # coroutine; the caller's finally still tears the target down.
                # The runs it had already completed survive in ``progress``,
                # which the loop writes to as it goes.
                # ``asyncio.timeout`` rather than ``wait_for`` so the cap can
                # be told apart from a TimeoutError the task raised itself --
                # a provider client read timeout, say, which reaches here from
                # optimizer.initialize(). ``wait_for`` raises the same type for
                # both, so a client timeout was recorded as stop_reason
                # "timeout" with a rationale asserting a cap that had not
                # expired, and the real exception was dropped.
                try:
                    async with asyncio.timeout(self._task_time_cap_s) as cap:
                        return await self._run_task(
                            task, target, task_scope, reporter, index, progress
                        )
                except TimeoutError:
                    if not cap.expired():
                        raise  # the task's own timeout; the handler below records it
                    cap_expired = True
                    logger.warning(
                        "Task %r: exceeded task_time_cap_s=%s, cancelled after %d run(s)",
                        task.goal.description,
                        self._task_time_cap_s,
                        len(progress.runs),
                    )
                    return _truncated_task_result(
                        task,
                        progress,
                        rationale=(
                            f"Task exceeded task_time_cap_s="
                            f"{self._task_time_cap_s}s and was cancelled."
                        ),
                        scope=task_scope.write,
                        read_only=task_scope.read_only,
                    )
            except NotApplicable:
                logger.info("Task %r not applicable, skipping", task.goal.description)
                return task
            except Exception as exc:
                logger.exception(
                    "Task %r: unexpected error before run loop, recording as error",
                    task.goal.description,
                )
                return _synthesize_empty_task_result(
                    task,
                    stop_reason="error",
                    rationale="Unexpected error before run loop started.",
                    exc=exc,
                    scope=task_scope.write,
                    read_only=task_scope.read_only,
                )
        finally:
            # Bounded whenever a cancellation put us here. The cap's own is
            # already spent by now -- the timeout context has exited and
            # uncancelled it -- so it takes a flag; an outer cancellation
            # (Ctrl-C) is still in flight and reads directly, exactly as in
            # _run_task. A task that ended normally gets an unbounded teardown,
            # because cutting one short leaks whatever it was releasing.
            target_deadline = (
                asyncio.get_running_loop().time() + _CLEANUP_GRACE_S
                if cap_expired or _cancellation_in_flight()
                else math.inf
            )
            target_exc = await _finish_or_abandon(
                target.teardown(), "target.teardown post-task", target_deadline
            )
            if target_exc is not None and not isinstance(target_exc, Exception):
                raise target_exc
            if target_exc is not None:
                logger.error("target.teardown post-task failed", exc_info=target_exc)

    # ------------------------------------------------------------------
    # Per-task run
    # ------------------------------------------------------------------

    async def _run_task(
        self,
        task: Task[Target],
        target: Target,
        task_scope: _TaskScope,
        reporter: ProgressReporter,
        index: int,
        progress: _TaskProgress,
    ) -> TaskResult:
        """Run the optimizer loop for a single task.

        A fresh optimizer and :class:`LLMClient` are created for each call;
        the caller (``_iterate_tasks``) supplies a fresh target and owns its
        teardown after this returns.  *task_scope* is the scope enforced for
        this task (resolved once by the caller); LLM config is read from
        ``self`` (constant for the threat model).

        Completed runs, the best score so far and the attacker's spend are
        accumulated into *progress*, which the caller owns.  That is what lets
        the wall-clock cap cancel this coroutine without destroying the runs it
        had already finished and had judged.
        """
        # Configure target (NotApplicable propagates to caller)
        await task.configure_target(target)

        # Create a fresh LLM client if we have a config.  The per-task cost
        # cap is applied here: a fresh client per task makes it a per-task budget.
        llm_client: LLMClient | None = (
            LLMClient(self._llm_config, cost_cap_usd=self._task_cost_cap_usd)
            if self._llm_config
            else None
        )
        # Published immediately: spend made before the first run completes is
        # still real money, and must survive a cancellation.
        progress.llm_client = llm_client

        # Fresh optimizer for this task
        optimizer = self._optimizer_factory()

        # Split the target's controllables by access level.  The optimizer
        # only gets the injectable ones in ``controllables``; a controllable
        # that is visible but not writable (read-only this run) is surfaced
        # in ``observables`` instead, so the list it can inject into is honest.
        all_controllables = target.get_controllables()
        controllables = [
            c for c in all_controllables if scope_includes(task_scope.write, c.security_domain)
        ]
        # Readable surfaces: in-visibility observables, plus read-only
        # controllables re-presented as observables.
        observables = [
            o
            for o in target.get_observables()
            if scope_includes(task_scope.visibility, o.observable.security_domain)
        ]
        observables += [
            _controllable_as_observable(c)
            for c in all_controllables
            if scope_includes(task_scope.visibility, c.security_domain)
            and not scope_includes(task_scope.write, c.security_domain)
        ]
        try:
            await optimizer.initialize(
                task.goal,
                controllables,
                observables,
                llm_client if llm_client is not None else LLMClient._make_noop(),
            )
        except BudgetExhaustedError:
            # Optimizer exhausted its LLM budget inside initialize (e.g. a
            # warmup call). Return a budget_exhausted result directly so it
            # isn't misclassified as a generic error by _iterate_tasks.
            await _swallow(optimizer.teardown(), "optimizer.teardown after init-budget-exhausted")
            logger.info(
                "Task %r: LLM budget exhausted during optimizer.initialize, stopping task",
                task.goal.description,
            )
            return _synthesize_empty_task_result(
                task,
                stop_reason="budget_exhausted",
                rationale="LLM budget exhausted before first run completed.",
                usage=llm_client.usage if llm_client else LLMUsage(),
                scope=task_scope.write,
                read_only=task_scope.read_only,
            )
        except Exception:
            # Tear down before re-raising so the optimizer doesn't leak;
            # _iterate_tasks catches and records the error.
            await _swallow(optimizer.teardown(), "optimizer.teardown after init-error")
            raise

        # Create channel and launch optimizer as concurrent task.
        channel = EventChannel()

        async def _optimizer_with_error_propagation() -> None:
            try:
                await optimizer.run(channel)
            except Exception as exc:
                channel.set_error(exc)
                raise

        optimizer_task = asyncio.create_task(_optimizer_with_error_propagation())

        # Aliased onto the caller-owned holder: appends are visible to the
        # caller even if this coroutine is cancelled and never returns.
        runs: list[RunResult] = progress.runs
        # Default reason: if the for-loop exits without an explicit break,
        # the safety cap was reached.
        stop_reason: StopReason = "max_runs"
        error_text: str | None = None
        prev_usage = LLMUsage()

        try:
            for run_number in range(1, self._max_runs_per_task + 1):
                # Trajectory is owned by _run_task (not _run_single) so the
                # partial trajectory survives any exception inside the run.
                trajectory = Trajectory(filtered_scope=task_scope.visibility)
                run_started = datetime.now(UTC)
                try:
                    evaluation, done = await self._run_single(
                        task,
                        target,
                        channel,
                        run_number,
                        trajectory,
                        task_scope,
                    )
                except BudgetExhaustedError:
                    logger.info(
                        "Task %r: LLM budget exhausted during run %d, stopping task",
                        task.goal.description,
                        run_number,
                    )
                    stop_reason = "budget_exhausted"
                    break
                except Exception as exc:
                    # _run_single failed mid-run — preserve the partial
                    # trajectory it accumulated, attach a zero-score
                    # evaluation, and store the formatted exception on
                    # ``TaskResult.error`` so the failure is recoverable
                    # from the persisted JSON.
                    logger.exception(
                        "Task %r: unexpected error during run %d, stopping task",
                        task.goal.description,
                        run_number,
                    )
                    error_text = _format_exception(exc)
                    error_eval = EvaluationResult(
                        success=False,
                        primary_score=Score(value=0.0, name="primary"),
                        sub_scores={},
                        rationale=f"Run {run_number} failed: {type(exc).__name__}: {exc}",
                    )
                    run_usage = llm_client.usage if llm_client else LLMUsage()
                    error_run = RunResult(
                        trajectory=trajectory,
                        evaluation=error_eval,
                        llm_usage=run_usage,
                        run_usage_delta=_usage_delta(prev_usage, run_usage),
                        started_at=run_started,
                        ended_at=datetime.now(UTC),
                        evaluated=False,
                        errored=True,
                        done=False,
                    )
                    prev_usage = run_usage
                    runs.append(error_run)
                    reporter.on_run_complete(
                        _run_complete_event(index, task.goal.description, run_number, error_run)
                    )
                    stop_reason = "error"
                    break

                # _run_single succeeded — record the run.
                run_usage = llm_client.usage if llm_client else LLMUsage()
                run_result = RunResult(
                    trajectory=trajectory,
                    evaluation=evaluation,
                    llm_usage=run_usage,
                    run_usage_delta=_usage_delta(prev_usage, run_usage),
                    started_at=run_started,
                    ended_at=datetime.now(UTC),
                    evaluated=True,
                    errored=False,
                    done=done,
                )
                prev_usage = run_usage
                runs.append(run_result)
                reporter.on_run_complete(
                    _run_complete_event(index, task.goal.description, run_number, run_result)
                )

                # The claim, not the attacker, decides success, so the loop
                # ends here without asking the optimizer.  Under a blind threat
                # model (``include_feedback=False``) the optimizer is not told
                # it has won and would otherwise spend the rest of its budget
                # against an already-broken target.  Checked BEFORE
                # reset_ephemeral_state: there is no next run to reset for, and
                # targets already document that reset is not called after the
                # final run.
                if evaluation.success:
                    stop_reason = "success"
                    break

                # Reset ephemeral target state for next run within this task.  If
                # reset_ephemeral_state raises, the successful run we just appended stays
                # — only the task is abandoned, with the reset exception
                # captured on ``error``.
                try:
                    await target.reset_ephemeral_state()
                except Exception as exc:
                    logger.exception(
                        "Task %r: target.reset_ephemeral_state() failed after run %d, "
                        "stopping task",
                        task.goal.description,
                        run_number,
                    )
                    error_text = _format_exception(exc)
                    stop_reason = "error"
                    break

                if done:
                    stop_reason = "done"
                    break

        finally:
            channel.close()
            # Closing the channel is the cooperative stop signal, and a
            # well-behaved optimizer returns on it. One blocked elsewhere -- a
            # provider call that never returns -- does not, and neither the
            # join nor the teardown below can be interrupted by the cap, which
            # has already fired. Both get a shared budget; see
            # ``_finish_or_abandon``. Bound cleanup only while a cancellation
            # is actually in flight -- the cap's, or an outer one such as
            # Ctrl-C, which must not be pinned by a hanging teardown either.
            # A task that ends normally keeps its original unbounded wait.
            # That is the case the budget exists for: the cap fires once, so a
            # fresh suspend in this `finally` has nothing left to interrupt it.
            # On a healthy task the enclosing `asyncio.timeout` is still armed
            # and bounds these awaits already, so imposing a second, much
            # shorter budget would cut short a teardown that is simply slow --
            # stopping containers, closing a proxy -- and leak what it was
            # about to release.
            deadline = (
                asyncio.get_running_loop().time() + _CLEANUP_GRACE_S
                if _cancellation_in_flight()
                else math.inf
            )
            optimizer_exc = await _finish_or_abandon(
                optimizer_task,
                f"Task {task.goal.description!r}: optimizer",
                deadline,
            )
            if optimizer_exc is not None and not isinstance(optimizer_exc, Exception):
                # BaseException that is not an Exception (SystemExit,
                # KeyboardInterrupt): not ours to turn into a diagnostic.
                raise optimizer_exc
            if optimizer_exc is not None:
                # Optimizer raised outside any in-flight channel.send
                # (background work, or its own teardown after the run loop
                # exited). Captured as a diagnostic without changing
                # stop_reason -- the run loop's classification is authoritative.
                logger.error(
                    "Task %r: optimizer task raised during teardown",
                    task.goal.description,
                    exc_info=optimizer_exc,
                )
                if error_text is None:
                    error_text = _format_exception(optimizer_exc)
            teardown_exc = await _finish_or_abandon(
                optimizer.teardown(), "optimizer.teardown post-run", deadline
            )
            if teardown_exc is not None and not isinstance(teardown_exc, Exception):
                # Same reasoning as the optimizer join above: a BaseException
                # that is not an Exception is not ours to turn into a log line.
                raise teardown_exc
            if teardown_exc is not None:
                logger.error("optimizer.teardown post-run failed", exc_info=teardown_exc)
            # No post-task reset_ephemeral_state() here: the caller
            # (_execute_task) tears the target down immediately after this
            # method returns, and the target is never reused across tasks
            # (TargetFactory.create() hands out a fresh instance per task).
            # Resetting ephemeral state right before discarding the whole
            # instance is pure waste.

        # If the loop ended before any run completed (budget exhausted or
        # error on run 1), synthesize a zero-score result so the task still
        # appears in results.
        best = progress.best
        if best is None:
            best_score = Score(value=0.0, name="primary")
            if stop_reason == "error":
                rationale = "Unexpected error before first run completed."
            else:
                rationale = "LLM budget exhausted before first run completed."
            best_evaluation = EvaluationResult(
                success=False,
                primary_score=best_score,
                sub_scores={},
                rationale=rationale,
            )
        else:
            best_score, best_evaluation = best

        task_usage = llm_client.usage if llm_client else LLMUsage()
        return TaskResult(
            task=task,
            runs=runs,
            best_score=best_score,
            best_evaluation=best_evaluation,
            success=progress.success,
            llm_usage=task_usage,
            stop_reason=stop_reason,
            scope=task_scope.write,
            read_only=task_scope.read_only,
            error=error_text,
        )

    async def _run_single(
        self,
        task: Task[Target],
        target: Target,
        channel: EventChannel,
        run_number: int,
        trajectory: Trajectory,
        task_scope: _TaskScope,
    ) -> tuple[EvaluationResult, bool]:
        """Execute a single optimizer iteration (one target run + evaluation).

        Order: target.run → evaluate → RunEndEvent (persisted) → close.
        The optimizer reads the evaluation from RunEndEvent.evaluation
        (when ``include_feedback=True``) or from the trajectory.

        The trajectory is constructed by the caller (``_run_task``) so that
        a failure mid-run still leaves a partial trajectory accessible for
        persistence and debugging.

        Returns:
            A tuple of (evaluation, done) where done is True if the
            optimizer wants to stop.
        """
        assert task_scope.visibility, "scope must contain at least one tag"

        # Signal run start — optimizer gets filtered view
        await channel.send(RunStartEvent(trajectory=trajectory.filtered))

        # Build the event pipeline: record → filter → send to optimizer.
        # The filter gets the read & write scope: controllable events under
        # read-only tags (outside it) are declined with
        # ControllableNoInjection without consulting the optimizer, but the
        # recorder (outermost) still records them — being in the full
        # visibility scope, they stay visible through the optimizer's
        # filtered trajectory view.
        send_event = compose(
            trajectory_recorder(trajectory),
            security_domain_filter(task_scope.write),
        )(channel.send)

        # Target runs — events go through the pipeline
        await target.run(trajectory.emit, send_event)

        # -- Evaluate --
        evaluation = await task.evaluate(trajectory, target)

        # Filter sub_scores to only include in-scope scores.
        # primary_score, success, and rationale are always included
        # (the optimizer needs the main signal).
        filtered_sub = {
            k: v
            for k, v in evaluation.sub_scores.items()
            if v.security_domain is None or scope_includes(task_scope.visibility, v.security_domain)
        }
        filtered_eval = EvaluationResult(
            success=evaluation.success,
            primary_score=evaluation.primary_score,
            sub_scores=filtered_sub,
            rationale=evaluation.rationale,
        )

        # RunEndEvent is persisted to the trajectory.
        # include_feedback controls whether evaluation data is attached.
        run_end_eval = filtered_eval if self._include_feedback else None
        run_end = RunEndEvent(
            evaluation=run_end_eval,
            security_domain=next(iter(task_scope.visibility)),
        )
        trajectory.emit(run_end)
        end_response = await channel.send(run_end)

        trajectory.close()

        done = isinstance(end_response, RunEndResponse) and end_response.done

        logger.info(
            "Task %r run %d: score=%.4f success=%s done=%s",
            task.goal.description,
            run_number,
            evaluation.primary_score.value,
            evaluation.success,
            done,
        )

        return evaluation, done


async def run_all(
    controllers: Sequence[Controller],
    *,
    concurrency: int | None = None,
    report: bool | Literal["auto"] = "auto",
) -> list[ThreatModelResult]:
    """Run several threat models (Controllers) as one coordinated sweep.

    One ``Controller`` is one threat model; this is the unified entry point for
    running several of them together.  It owns a SINGLE live dashboard for the
    whole sweep, so the threat models share one terminal canvas (one block
    each) instead of each ``Controller.run()`` independently grabbing the
    process-wide dashboard.  Under a concurrency cap the latter stops and
    re-arms the canvas between waves, leaving a garbled mix of stacked canvases
    and plain-reporter fallbacks; running through ``run_all`` keeps one clean
    canvas from start to finish.  Every threat model is shown from the outset,
    the ones still queued behind the concurrency cap dimmed until a slot frees.

    Args:
        controllers: The threat models to run, one ``Controller`` each.  They
            should have distinct identities (attacker/target/claim/model/scope),
            which they already need for distinct result directories.
        concurrency: Max threat models running at once (default: all of them).
            Lower it to bound resource use; each admitted controller still runs
            its own tasks up to its ``TargetFactory.concurrency``.
        report: ``"auto"``/``True`` render a shared live dashboard on a TTY
            (plain lines off a TTY); ``False`` is silent.  This one decision
            applies to the whole sweep, overriding each controller's ``report``.

    Returns:
        Each controller's :class:`ThreatModelResult`, in input order.
    """
    controllers = list(controllers)
    if not controllers:
        return []
    limit = len(controllers) if concurrency is None else max(1, concurrency)
    semaphore = asyncio.Semaphore(limit)

    dashboard: Dashboard | None = None
    if report is not False and not should_use_plain():
        dashboard = Dashboard()
        dashboard.expect(len(controllers))
        # Show every threat model from the start (queued ones dimmed) rather than
        # popping a lane in only when the semaphore admits it.
        for controller in controllers:
            dashboard.preregister(controller.label, controller.context)

    def reporter_for(controller: Controller) -> ProgressReporter:
        if dashboard is not None:
            return dashboard.reporter_for(controller.label)
        if report is False:
            return NullReporter()
        return PlainReporter()

    async def run_one(controller: Controller) -> ThreatModelResult:
        async with semaphore:
            return await controller.run(reporter=reporter_for(controller))

    try:
        return list(await asyncio.gather(*(run_one(c) for c in controllers)))
    finally:
        if dashboard is not None:
            dashboard.close()
