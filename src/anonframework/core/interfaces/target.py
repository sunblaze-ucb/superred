"""Target interface.

A target is the AI system under test. It exposes configuration slots
(for task setup), queryable state (for evaluation), controllables
(runtime injection points), and observables (static context).

During a run the target calls ``send_event`` at each controllable point,
pausing until it receives a response, and ``emit`` to record one-way
trajectory entries.

Target authors implement this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from anonframework.core.types.controllable import Controllable
from anonframework.core.types.event import EventHandler, EventResponseHandler
from anonframework.core.types.observable import ObservableValue
from anonframework.core.types.security_domain import SecurityDomain
from anonframework.core.types.state import ConfigSpec, QuerySpec


class Target(ABC):
    """Base class for all targets (AI systems under test).

    Manual values (API keys, credentials) are passed directly to the
    target's constructor — not through the framework.

    Implementors override:
        - :attr:`config_specs` — declare pre-run configuration slots.
        - :meth:`set_config` — accept a configuration value.
        - :attr:`query_specs` — declare post-run interactions.
        - :meth:`query` — execute a post-run query.
        - :attr:`security_domain` — the security domain forest.
        - :meth:`get_controllables` — declare runtime injection points.
        - :meth:`get_observables` — provide static context.
        - :meth:`run` — execute one run.
        - :meth:`reset_ephemeral_state` — reset ephemeral (per-run) state after a run.
        - :meth:`teardown` — release resources.
    """

    # ------------------------------------------------------------------
    # Pre-run configuration (task sets these before a run)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def config_specs(self) -> list[ConfigSpec]:
        """Configuration slots this target accepts.

        Each spec declares a named, text-valued slot. The description
        documents the expected format — that is the contract.
        """
        ...

    @abstractmethod
    def set_config(self, name: str, value: str) -> None:
        """Set a configuration value before a run.

        Args:
            name: Must match a :attr:`ConfigSpec.name` from :attr:`config_specs`.
            value: Text value in the format the spec's description expects.
        """
        ...

    # ------------------------------------------------------------------
    # Post-run queries (evaluator uses these after a run)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def query_specs(self) -> list[QuerySpec]:
        """Interactions available after a run.

        Each spec declares a named query the evaluator can call,
        optionally with parameters. These are distinct from config specs.
        """
        ...

    @abstractmethod
    def query(self, name: str, **params: str) -> str:
        """Execute a post-run query.

        May be a simple getter (no params) or an action with parameters.

        Args:
            name: Must match a :attr:`QuerySpec.name` from :attr:`query_specs`.
            **params: Keyword arguments matching the spec's params.

        Returns:
            The text result of the query.
        """
        ...

    # ------------------------------------------------------------------
    # Security domain
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def security_domain(self) -> SecurityDomain:
        """The security domain forest defined by this target system.

        The domain classifies controllables and observables into a
        hierarchy of trust boundaries.
        """
        ...

    # ------------------------------------------------------------------
    # Controllables and observables
    # ------------------------------------------------------------------

    @abstractmethod
    def get_controllables(self) -> list[Controllable]:
        """Return the controllables (injection points) this target exposes."""
        ...

    @abstractmethod
    def get_observables(self) -> list[ObservableValue]:
        """Return static observables describing this target system."""
        ...

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(
        self,
        emit: EventHandler,
        send_event: EventResponseHandler,
    ) -> None:
        """Execute one run of the target system.

        The target records one-way entries via ``emit(entry)`` and
        pauses at controllable points by calling ``send_event(event)`` to
        get the optimizer's response.

        The target does **not** have access to the full trajectory — only
        the ``emit`` function for writing entries.

        Args:
            emit: Callback to record a trajectory entry (fire-and-forget).
            send_event: Callback to send an event and await a response.
        """
        ...

    @abstractmethod
    async def reset_ephemeral_state(self) -> None:
        """Reset the target's ephemeral (per-run) state after a run.

        Called by the controller after each run's evaluation, before the
        next run begins. Reset only ephemeral state here (for example the
        active conversation or the most recent response). Durable state
        must survive this call.

        A target has three state lifetimes:

        - **Ephemeral**: per-run state such as the active conversation.
          Reset by this method after every run.
        - **Durable**: state that persists across runs within a single
          task, such as a memory bank accumulated by a memory-injection
          attack. Not reset here; it is discarded only when the
          controller obtains a fresh instance from the ``TargetFactory``
          between tasks.
        - **Resources / identity**: the configured definition and
          external handles. Constant for the instance's lifetime and
          released in :meth:`teardown`.

        Implement to clear ephemeral databases, reset containers, etc.
        May be a no-op, but must be explicit.
        """
        ...

    @abstractmethod
    async def teardown(self) -> None:
        """Release resources. Called after all evaluation is done.

        Awaited to completion, except on a task the wall-clock cap cancelled:
        cleanup there is bounded and a teardown that overruns is cancelled.
        """
        ...
