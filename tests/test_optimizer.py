"""Unit tests for the Optimizer base class: _dispatch lifecycle, trajectory tracking."""

from __future__ import annotations

import asyncio

import pytest

from anonframework.core.channel import EventChannel
from anonframework.core.interfaces.optimizer import Optimizer
from anonframework.core.llm import LLMClient
from anonframework.core.types.controllable import Controllable
from anonframework.core.types.event import Event, EventResponse
from anonframework.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from anonframework.core.types.goal import Goal
from anonframework.core.types.observable import ObservableValue
from anonframework.core.types.security_domain import SecurityDomainTag
from anonframework.core.types.trajectory import Trajectory


class MinimalOptimizer(Optimizer):
    """Concrete optimizer for testing base class behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.events_seen: list[Event] = []

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)

    async def on_event(self, event: Event) -> EventResponse:
        self.events_seen.append(event)
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=False)
        if isinstance(event, ControllablePreCallEvent):
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value="test",
            )
        return EventResponse(event=event)


class TestOptimizerTrajectoryTracking:
    """Verifies _dispatch correctly manages trajectory state."""

    async def test_run_start_sets_current_trajectory(self) -> None:
        opt = MinimalOptimizer()
        channel = EventChannel()
        task = asyncio.create_task(opt.run(channel))
        trajectory = Trajectory()
        await channel.send(RunStartEvent(trajectory=trajectory))
        assert opt.current_trajectory is trajectory
        assert opt.past_trajectories == []
        channel.close()
        await task

    async def test_run_end_archives_trajectory(self) -> None:
        opt = MinimalOptimizer()
        channel = EventChannel()
        task = asyncio.create_task(opt.run(channel))
        trajectory = Trajectory()
        await channel.send(RunStartEvent(trajectory=trajectory))
        await channel.send(RunEndEvent())
        assert opt.current_trajectory is None
        assert len(opt.past_trajectories) == 1
        assert opt.past_trajectories[0] is trajectory
        channel.close()
        await task

    async def test_multiple_runs_accumulate_trajectories(self) -> None:
        opt = MinimalOptimizer()
        channel = EventChannel()
        task = asyncio.create_task(opt.run(channel))
        t1, t2 = Trajectory(), Trajectory()
        await channel.send(RunStartEvent(trajectory=t1))
        await channel.send(RunEndEvent())
        await channel.send(RunStartEvent(trajectory=t2))
        await channel.send(RunEndEvent())
        assert len(opt.past_trajectories) == 2
        assert opt.past_trajectories[0] is t1
        assert opt.past_trajectories[1] is t2
        channel.close()
        await task

    async def test_run_end_without_start_does_not_crash(self) -> None:
        """RunEndEvent when _current_trajectory is None should not error."""
        opt = MinimalOptimizer()
        channel = EventChannel()
        task = asyncio.create_task(opt.run(channel))
        await channel.send(RunEndEvent())
        assert opt.past_trajectories == []
        assert opt.current_trajectory is None
        channel.close()
        await task

    async def test_past_trajectories_returns_copy(self) -> None:
        opt = MinimalOptimizer()
        channel = EventChannel()
        task = asyncio.create_task(opt.run(channel))
        t = Trajectory()
        await channel.send(RunStartEvent(trajectory=t))
        await channel.send(RunEndEvent())
        past = opt.past_trajectories
        past.clear()
        assert len(opt.past_trajectories) == 1
        channel.close()
        await task


class TestOptimizerDefaultRun:
    """Verifies the default run() processes events sequentially."""

    async def test_sequential_processing(self) -> None:
        opt = MinimalOptimizer()
        channel = EventChannel()
        task = asyncio.create_task(opt.run(channel))
        tag = SecurityDomainTag("ext")
        ctrl = Controllable(name="input", security_domain=tag)
        t = Trajectory()
        await channel.send(RunStartEvent(trajectory=t))
        resp = await channel.send(ControllablePreCallEvent(controllable=ctrl, request="hi"))
        assert isinstance(resp, ControllableInjection)
        assert resp.value == "test"
        await channel.send(RunEndEvent())
        channel.close()
        await task
        types = [type(e).__name__ for e in opt.events_seen]
        assert types == ["RunStartEvent", "ControllablePreCallEvent", "RunEndEvent"]


class TestOptimizerExceptionHandling:
    """Verifies _dispatch exception safety via envelope.reject()."""

    @pytest.mark.regression
    async def test_dispatch_rejects_on_event_exception(self) -> None:
        """If on_event raises, _dispatch rejects the envelope so the
        sender gets the exception instead of deadlocking."""

        class FailingOptimizer(Optimizer):
            async def initialize(
                self,
                goal: Goal,
                controllables: list[Controllable],
                observables: list[ObservableValue],
                llm_client: LLMClient,
            ) -> None:
                await super().initialize(goal, controllables, observables, llm_client)

            async def on_event(self, event: Event) -> EventResponse:
                raise ValueError("boom")

        opt = FailingOptimizer()
        channel = EventChannel()
        task = asyncio.create_task(opt.run(channel))
        # send() gets the exception from reject() — no deadlock
        with pytest.raises(ValueError, match="boom"):
            await channel.send(RunStartEvent(trajectory=Trajectory()))
        channel.close()
        with pytest.raises(ValueError, match="boom"):
            await task

    async def test_dispatch_maintains_trajectory_on_exception(self) -> None:
        """If on_event raises during RunEndEvent, trajectory is still archived."""

        class FailOnEnd(MinimalOptimizer):
            async def on_event(self, event: Event) -> EventResponse:
                if isinstance(event, RunEndEvent):
                    raise ValueError("end failed")
                return await super().on_event(event)

        opt = FailOnEnd()
        channel = EventChannel()
        task = asyncio.create_task(opt.run(channel))
        t = Trajectory()
        await channel.send(RunStartEvent(trajectory=t))
        with pytest.raises(ValueError, match="end failed"):
            await channel.send(RunEndEvent())
        assert opt.current_trajectory is None
        assert opt.past_trajectories[0] is t
        channel.close()
        with pytest.raises(ValueError, match="end failed"):
            await task

    async def test_dispatch_error_on_run_end_without_trajectory(self) -> None:
        """If on_event raises on RunEndEvent when _current_trajectory is None,
        _dispatch still handles lifecycle correctly (no crash, no archive)."""

        class FailOnEnd(MinimalOptimizer):
            async def on_event(self, event: Event) -> EventResponse:
                if isinstance(event, RunEndEvent):
                    raise ValueError("end failed")
                return await super().on_event(event)

        opt = FailOnEnd()
        channel = EventChannel()
        task = asyncio.create_task(opt.run(channel))
        # Send RunEndEvent WITHOUT a preceding RunStartEvent
        with pytest.raises(ValueError, match="end failed"):
            await channel.send(RunEndEvent())
        assert opt.current_trajectory is None
        assert opt.past_trajectories == []
        channel.close()
        with pytest.raises(ValueError, match="end failed"):
            await task

    @pytest.mark.regression
    async def test_channel_poisoned_after_optimizer_crash(self) -> None:
        """After on_event raises, subsequent channel.send() calls raise
        immediately (via channel.set_error) instead of deadlocking."""

        class FailOnControllable(MinimalOptimizer):
            async def on_event(self, event: Event) -> EventResponse:
                if isinstance(event, ControllablePreCallEvent):
                    raise ValueError("ctrl failed")
                return await super().on_event(event)

        opt = FailOnControllable()
        channel = EventChannel()

        async def run_with_poison() -> None:
            try:
                await opt.run(channel)
            except Exception as exc:
                channel.set_error(exc)
                raise

        task = asyncio.create_task(run_with_poison())
        tag = SecurityDomainTag("ext")
        ctrl = Controllable(name="x", security_domain=tag)
        t = Trajectory()
        await channel.send(RunStartEvent(trajectory=t))
        # This send gets the exception via reject()
        with pytest.raises(ValueError, match="ctrl failed"):
            await channel.send(
                ControllablePreCallEvent(
                    controllable=ctrl,
                    request="hi",
                )
            )
        # Subsequent sends raise immediately (channel poisoned)
        with pytest.raises(ValueError, match="ctrl failed"):
            await channel.send(RunEndEvent())
        with pytest.raises(ValueError, match="ctrl failed"):
            await task


class TestOptimizerTeardown:
    async def test_default_teardown_is_noop(self) -> None:
        """Base teardown() does nothing but must be callable."""
        opt = MinimalOptimizer()
        result = await opt.teardown()
        assert result is None
