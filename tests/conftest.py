"""Shared fixtures for the anonframework test suite."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from anonframework.core.controller import Controller
from anonframework.core.interfaces.optimizer import Optimizer
from anonframework.core.interfaces.target import Target
from anonframework.core.interfaces.task import NotApplicable, Task
from anonframework.core.llm import LLMClient
from anonframework.core.types.controllable import Controllable
from anonframework.core.types.evaluation import EvaluationResult, Score
from anonframework.core.types.event import Event, EventHandler, EventResponse, EventResponseHandler
from anonframework.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from anonframework.core.types.goal import Goal
from anonframework.core.types.llm import LLMConfig
from anonframework.core.types.observable import ObservableValue
from anonframework.core.types.security_domain import SecurityDomain, SecurityDomainTag
from anonframework.core.types.state import ConfigSpec, QuerySpec
from anonframework.core.types.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Persistence / reporting test safety rail
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _persistence_off_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Force every Controller built under pytest to ``persist=False`` and
    ``report=False`` unless the test opts in, and route any writes to
    ``tmp_path``.  Central so a single missed call site can neither leak
    sensitive trajectory content to ``./anonframework-results/`` nor race under
    ``pytest -n auto`` (each worker gets its own ``tmp_path``)."""
    original_init = Controller.__init__

    def patched_init(self: Controller, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("persist", False)
        kwargs.setdefault("report", False)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Controller, "__init__", patched_init)
    monkeypatch.setenv("ANONFRAMEWORK_RESULTS_DIR", str(tmp_path / "anonframework-results"))


# ---------------------------------------------------------------------------
# Security domain fixtures
# ---------------------------------------------------------------------------

STUB_LLM_CONFIG = LLMConfig(model="test-model", api_base="http://test", api_key="sk-test")

ROOT_TAG = SecurityDomainTag("root")
EXTERNAL_TAG = SecurityDomainTag("external", parent=ROOT_TAG)
INTERNAL_TAG = SecurityDomainTag("internal", parent=ROOT_TAG)


def make_domain() -> SecurityDomain:
    """Build the test domain. Factored into a function so that conftest
    import does not crash if SecurityDomain construction is mutated
    (mutmut would otherwise report these as 'survived' due to exit code 4)."""
    return SecurityDomain([ROOT_TAG, EXTERNAL_TAG, INTERNAL_TAG])


@pytest.fixture
def root_tag() -> SecurityDomainTag:
    return ROOT_TAG


@pytest.fixture
def external_tag() -> SecurityDomainTag:
    return EXTERNAL_TAG


@pytest.fixture
def internal_tag() -> SecurityDomainTag:
    return INTERNAL_TAG


@pytest.fixture
def domain() -> SecurityDomain:
    return make_domain()


# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------


class StubOptimizer(Optimizer):
    """Minimal optimizer that injects a fixed value."""

    def __init__(self, inject_value: str = "injected", done: bool = False) -> None:
        super().__init__()
        self.inject_value = inject_value
        self._done = done
        self.events_received: list[Event] = []
        self.initialized = False
        self.torn_down = False

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self.initialized = True

    async def on_event(self, event: Event) -> EventResponse:
        self.events_received.append(event)
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=self._done)
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self.inject_value,
        )

    async def teardown(self) -> None:
        self.torn_down = True


class StubTarget(Target):
    """Minimal target that emits one controllable event per run."""

    def __init__(self, tag: SecurityDomainTag = EXTERNAL_TAG) -> None:
        self._tag = tag
        self._config: dict[str, str] = {}
        self.run_count = 0
        self.torn_down = False
        self.reset_count = 0

    @property
    def security_domain(self) -> SecurityDomain:
        return make_domain()

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [ConfigSpec(name="prompt", security_domain=self._tag, description="The prompt")]

    def set_config(self, name: str, value: str) -> None:
        self._config[name] = value

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [QuerySpec(name="last_response", description="The last response")]

    def query(self, name: str, **params: str) -> str:
        return "target_response"

    def get_controllables(self) -> list[Controllable]:
        return [Controllable(name="user_input", security_domain=self._tag)]

    def get_observables(self) -> list[ObservableValue]:
        return []

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self.run_count += 1
        ctrl = Controllable(name="user_input", security_domain=self._tag)
        event = ControllablePreCallEvent(controllable=ctrl, request="hello")
        await send_event(event)

    async def reset_ephemeral_state(self) -> None:
        self.reset_count += 1

    async def teardown(self) -> None:
        self.torn_down = True


class ParallelTarget(Target):
    """Target that fires two events concurrently from parallel branches."""

    def __init__(self, tag: SecurityDomainTag = EXTERNAL_TAG) -> None:
        self._tag = tag

    @property
    def security_domain(self) -> SecurityDomain:
        return make_domain()

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return []

    def set_config(self, name: str, value: str) -> None:
        pass

    @property
    def query_specs(self) -> list[QuerySpec]:
        return []

    def query(self, name: str, **params: str) -> str:
        return ""

    def get_controllables(self) -> list[Controllable]:
        return []

    def get_observables(self) -> list[ObservableValue]:
        return []

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        ctrl = Controllable(name="parallel_input", security_domain=self._tag)

        async def branch(request: str) -> EventResponse:
            event = ControllablePreCallEvent(controllable=ctrl, request=request)
            return await send_event(event)

        r1, r2 = await asyncio.gather(branch("branch_a"), branch("branch_b"))
        assert isinstance(r1, ControllableInjection)
        assert isinstance(r2, ControllableInjection)

    async def reset_ephemeral_state(self) -> None:
        pass

    async def teardown(self) -> None:
        pass


class StubTask(Task[Target]):
    """Minimal task that returns a fixed evaluation result."""

    def __init__(
        self,
        score: float = 1.0,
        success: bool = True,
        goal_text: str = "Test goal",
    ) -> None:
        self._goal = Goal(description=goal_text)
        self._score = score
        self._success = success

    @property
    def goal(self) -> Goal:
        return self._goal

    async def configure_target(self, target: Target) -> None:
        pass

    async def evaluate(
        self,
        trajectory: Trajectory,
        target: Target,
    ) -> EvaluationResult:
        return EvaluationResult(
            success=self._success,
            primary_score=Score(value=self._score),
        )


class NotApplicableTask(Task[Target]):
    """Task that always raises NotApplicable."""

    @property
    def goal(self) -> Goal:
        return Goal(description="N/A task")

    async def configure_target(self, target: Target) -> None:
        raise NotApplicable("Not applicable to this target")

    async def evaluate(
        self,
        trajectory: Trajectory,
        target: Target,
    ) -> EvaluationResult:
        raise AssertionError("Should not be called")


class CountingOptimizer(Optimizer):
    """Optimizer that signals done after a configurable number of RunEndEvents."""

    def __init__(self, stop_after: int = 1, inject_value: str = "x") -> None:
        super().__init__()
        self._stop_after = stop_after
        self._inject_value = inject_value
        self._run_count = 0

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)
        if isinstance(event, RunEndEvent):
            self._run_count += 1
            return RunEndResponse(event=event, done=self._run_count >= self._stop_after)
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._inject_value,
        )

    async def teardown(self) -> None:
        pass


class NeverDoneOptimizer(Optimizer):
    """Optimizer that always returns done=False (never signals completion)."""

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, ControllablePreCallEvent):
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value="x",
            )
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=False)
        return EventResponse(event=event)

    async def teardown(self) -> None:
        pass


class FailingOnEventOptimizer(Optimizer):
    """Optimizer whose on_event raises on ControllablePreCallEvent."""

    def __init__(self) -> None:
        super().__init__()
        self.torn_down = False

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, ControllablePreCallEvent):
            raise RuntimeError("optimizer exploded")
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)

    async def teardown(self) -> None:
        self.torn_down = True
