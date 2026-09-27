"""Security claim interface.

A security claim is an iterable of tasks. Claims compose:
- From tasks: ``SecurityClaim.from_tasks([task_a, task_b])``
- From other claims (chained lazily): ``SecurityClaim.from_claims([claim_1, claim_2])``

Because tasks are stateless, a claim can be iterated multiple times.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

from anonframework.core.interfaces.target import Target
from anonframework.core.interfaces.task import Task

T_Target = TypeVar("T_Target", bound=Target)


class SecurityClaim(Generic[T_Target]):
    """An iterable collection of tasks, composable from tasks or other claims.

    Construct via the factory classmethods::

        # From tasks
        claim = SecurityClaim.from_tasks([task_a, task_b])

        # From other claims (iterated lazily)
        combined = SecurityClaim.from_claims([claim_1, claim_2])

    Iterate repeatedly (tasks are stateless)::

        for task in claim:
            ...
        for task in claim:  # works again
            ...
    """

    _tasks: list[Task[T_Target]] | None
    _claims: list[SecurityClaim[T_Target]] | None

    def __init__(self) -> None:
        raise TypeError("Use SecurityClaim.from_tasks() or SecurityClaim.from_claims()")

    @classmethod
    def from_tasks(cls, tasks: list[Task[T_Target]]) -> SecurityClaim[T_Target]:
        """Create a claim from a non-empty list of tasks."""
        if not tasks:
            raise ValueError("At least one task required")
        claim: SecurityClaim[T_Target] = object.__new__(cls)
        claim._tasks = list(tasks)
        claim._claims = None
        return claim

    @classmethod
    def from_claims(cls, claims: list[SecurityClaim[T_Target]]) -> SecurityClaim[T_Target]:
        """Create a claim by composing other claims (iterated lazily)."""
        if not claims:
            raise ValueError("At least one claim required")
        claim: SecurityClaim[T_Target] = object.__new__(cls)
        claim._tasks = None
        claim._claims = list(claims)
        return claim

    def __iter__(self) -> Iterator[Task[T_Target]]:
        if self._tasks is not None:
            return iter(self._tasks)
        assert self._claims is not None
        return _chain(self._claims)


def _chain(claims: list[SecurityClaim[T_Target]]) -> Iterator[Task[T_Target]]:
    """Lazily chain iteration over multiple claims."""
    for claim in claims:
        yield from claim
