"""Task interface.

A task defines an adversarial objective against a target. It is generic over
the target type: a task can be bound to a specific target class (accessing
its concrete API) or to the base ``Target`` (working with any target via
runtime discovery of config specs).

Tasks are stateless:
- ``configure_target`` sets initial config on the target via
  ``target.set_config()``.
- ``evaluate`` receives the trajectory and the target for ground-truth
  queries via ``target.query()``.

Config (pre-run) and query (post-run) are intentionally distinct.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from anonframework.core.interfaces.target import Target
from anonframework.core.types.evaluation import EvaluationResult
from anonframework.core.types.goal import Goal
from anonframework.core.types.trajectory import Trajectory

T_Target = TypeVar("T_Target", bound=Target)


class NotApplicable(Exception):  # noqa: N818
    """Raised by :meth:`Task.configure` when the task cannot work with the given target."""


class Task(ABC, Generic[T_Target]):
    """Base class for all tasks.

    Tasks are stateless — they do not hold a reference to the target.

    Bind to a specific target for type-safe access::

        class RAGSecretTask(Task[MyRAGTarget]):
            async def configure_target(self, target: MyRAGTarget) -> None:
                secret = generate_secret()
                target.set_config("db_seed", f"INSERT INTO docs VALUES ('{secret}')")

            async def evaluate(self, trajectory, target) -> EvaluationResult:
                response = target.query("last_response")
                ...

    Or bind to ``Target`` for a generic task::

        class GenericSecretTask(Task[Target]):
            async def configure_target(self, target: Target) -> None:
                spec = next(s for s in target.config_specs if "secret" in s.description.lower())
                target.set_config(spec.name, generate_secret())

            async def evaluate(self, trajectory, target) -> EvaluationResult:
                for spec in target.query_specs:
                    value = target.query(spec.name)
                    ...
    """

    @property
    @abstractmethod
    def goal(self) -> Goal:
        """The adversarial goal this task defines."""
        ...

    @abstractmethod
    async def configure_target(self, target: T_Target) -> None:
        """Configure the target's initial state for this task.

        Use ``target.set_config(name, value)`` to set config slots and
        ``target.config_specs`` to discover available slots.

        Args:
            target: The target to configure.

        Raises:
            NotApplicable: If this task cannot work with this target.
        """
        ...

    @abstractmethod
    async def evaluate(
        self,
        trajectory: Trajectory,
        target: T_Target,
    ) -> EvaluationResult:
        """Evaluate a run trajectory against this task's goal.

        The evaluator receives:
        - The full (unfiltered) run trajectory.
        - The target for on-demand post-run queries via
          ``target.query(name, **params)``. Use ``target.query_specs``
          to discover available queries and their parameters.

        Each :class:`Score` in the result carries a ``security_domain``.
        The controller filters ``sub_scores`` by the active scope before
        writing feedback to the trajectory, so the optimizer only sees
        scores within its security domain.

        The queries here access *post-run* ground truth, distinct from
        the initial config set during :meth:`configure`.

        Args:
            trajectory: The completed run trajectory (unfiltered).
            target: The target, for post-run queries.

        Returns:
            The evaluation result.
        """
        ...
