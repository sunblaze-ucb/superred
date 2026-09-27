"""Optimizer interface.

An optimizer is an event-driven actor that runs as a concurrent task and
processes events delivered through an :class:`EventChannel`.

The optimizer receives events via the channel and responds to each one.
The default ``run()`` implementation processes events sequentially via
``on_event()``. Override ``run()`` for parallel consumption, continuous
execution, or custom scheduling.

The base class tracks trajectory lifecycle automatically via ``_dispatch()``:
``RunStartEvent`` sets ``_current_trajectory``, ``RunEndEvent`` archives it
to ``_past_trajectories``.

Optimizer authors implement :meth:`on_event` at minimum. Override
:meth:`run` for advanced consumption models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from anonframework.core.channel import EventChannel, EventEnvelope
from anonframework.core.llm import LLMClient
from anonframework.core.types.controllable import Controllable
from anonframework.core.types.event import Event, EventResponse
from anonframework.core.types.events import RunEndEvent, RunStartEvent
from anonframework.core.types.goal import Goal
from anonframework.core.types.observable import ObservableValue
from anonframework.core.types.trajectory import ReadableTrajectory


class Optimizer(ABC):
    """Base class for all optimizers.

    Optimizers are actors: the framework launches ``run(channel)`` as a
    concurrent ``asyncio.Task``. Events arrive through the channel;
    the optimizer pulls and responds at its own pace.

    The base class maintains trajectory state so implementations can
    query previous trajectories without bookkeeping.

    Consumption models (optimizer's choice):
        - **Sequential** (default): Inherit ``run()``, override ``on_event``.
          Events are processed one at a time.
        - **Parallel**: Override ``run()``, spawn tasks that call
          ``self._dispatch(envelope)`` for each envelope.
        - **Continuous with events**: Override ``run()``, do background
          work and pull from the channel concurrently.

    Lifecycle:
        1. Instantiate with configuration.
        2. ``initialize(goal, controllables, observables, llm_client)``
           — setup. Base class stores the LLM client.
        3. ``run(channel)`` — launched as concurrent task. Receives
           ``RunStartEvent``, controllable events, ``RunEndEvent``.
        4. ``teardown()`` — cleanup.
    """

    def __init__(self) -> None:
        self._past_trajectories: list[ReadableTrajectory] = []
        self._current_trajectory: ReadableTrajectory | None = None
        self._llm_client: LLMClient = None  # type: ignore[assignment]  # set via initialize()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @abstractmethod
    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        """Set up the optimizer before the first run.

        The controller passes a constrained :class:`LLMClient` that locks
        the model, API base, and API key. After this call, the client is
        also accessible via ``self.llm``.

        Subclasses **must** call ``await super().initialize(...)`` (or
        accept that ``self.llm`` won't work and store the client
        themselves).

        Args:
            goal: The adversarial goal.
            controllables: Available injection points — the surfaces the
                optimizer can inject into.  A surface that is visible but
                not injectable in this threat model (read-only) is not here;
                it appears in ``observables`` instead.
            observables: Readable surfaces describing the target system,
                including any read-only controllables re-presented as
                observables (with ``content=None`` — their values arrive at
                runtime on the trajectory).
            llm_client: The constrained LLM client for this task.
        """
        self._llm_client = llm_client

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, channel: EventChannel) -> None:
        """Main execution loop. Runs as a concurrent ``asyncio.Task``.

        The default implementation processes events sequentially via
        :meth:`_dispatch`, which handles trajectory lifecycle and
        delegates to :meth:`on_event`.

        If :meth:`on_event` raises, ``_dispatch`` rejects the envelope
        (propagating the exception to the sender) and re-raises. The
        exception exits ``run()`` and the controller detects the failure.
        The controller then poisons the channel via ``set_error()``,
        ensuring no other ``send()`` call deadlocks.

        Override for parallel consumption or continuous execution.
        Call :meth:`_dispatch` from custom implementations to retain
        automatic trajectory tracking.

        Args:
            channel: The event channel to receive events from.
        """
        async for envelope in channel:
            await self._dispatch(envelope)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    @abstractmethod
    async def on_event(self, event: Event) -> EventResponse:
        """Handle a single event.

        Called by :meth:`_dispatch` for each event. Dispatch on event
        type using ``isinstance``::

            if isinstance(event, ControllablePreCallEvent):
                return ControllableInjection(
                    event=event, controllable=event.controllable, value=...,
                )
            elif isinstance(event, RunStartEvent):
                return EventResponse(event=event)
            elif isinstance(event, RunEndEvent):
                return RunEndResponse(event=event, done=False)

        The current trajectory is available via :attr:`current_trajectory`
        (set automatically before this method is called for
        ``RunStartEvent``).

        May be called concurrently if ``run()`` is overridden for
        parallel consumption. The implementation must handle its own
        synchronization in that case.

        Args:
            event: The event to handle.

        Returns:
            A response appropriate for the event type.
        """
        ...

    async def _dispatch(self, envelope: EventEnvelope) -> None:
        """Lifecycle wrapper around :meth:`on_event`.

        Tracks trajectory state (``_current_trajectory``,
        ``_past_trajectories``), delegates to :meth:`on_event`, and
        delivers the response via the envelope.

        If :meth:`on_event` raises, the envelope is rejected with the
        exception (so the sender gets the error instead of deadlocking)
        and trajectory lifecycle is maintained. The exception is then
        re-raised.

        Call this from custom :meth:`run` implementations to retain
        automatic trajectory tracking.

        Args:
            envelope: The event envelope to process.
        """
        event = envelope.event

        # Pre-dispatch lifecycle
        if isinstance(event, RunStartEvent):
            self._current_trajectory = event.trajectory

        try:
            response = await self.on_event(event)
        except Exception as exc:
            # Post-dispatch lifecycle must still run
            if isinstance(event, RunEndEvent):
                if self._current_trajectory is not None:
                    self._past_trajectories.append(self._current_trajectory)
                self._current_trajectory = None
            # Reject so the sender gets the exception (no deadlock)
            envelope.reject(exc)
            raise

        # Post-dispatch lifecycle
        if isinstance(event, RunEndEvent):
            if self._current_trajectory is not None:
                self._past_trajectories.append(self._current_trajectory)
            self._current_trajectory = None

        envelope.respond(response)

    # ------------------------------------------------------------------
    # Trajectory access
    # ------------------------------------------------------------------

    @property
    def past_trajectories(self) -> list[ReadableTrajectory]:
        """All completed run trajectories, oldest first."""
        return list(self._past_trajectories)

    @property
    def current_trajectory(self) -> ReadableTrajectory | None:
        """The trajectory for the currently active run, or None."""
        return self._current_trajectory

    # ------------------------------------------------------------------
    # LLM access
    # ------------------------------------------------------------------

    @property
    def llm(self) -> LLMClient:
        """The controller-provided LLM client.

        Stored by the base-class :meth:`initialize`. Optimizers access
        it via ``self.llm.complete(messages)``.
        """
        return self._llm_client

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def teardown(self) -> None:
        """Release resources. Override for cleanup.

        Awaited to completion, except while the task is unwinding a
        cancellation (the wall-clock cap's, or an outer one): cleanup there is
        bounded and a teardown that overruns is cancelled.
        """
