"""Trajectory: the ordered sequence of events and responses from a single run.

A run produces a trajectory storing :class:`Event` and
:class:`EventResponse` objects directly. The trajectory is the single
source of truth for a run — one-way log events from the target,
controllable events and their responses, and the :class:`RunEndEvent`
(which carries evaluation feedback) all live here. :class:`RunStartEvent`
is NOT persisted.

Each item's security domain is derived via :func:`get_domain`:
events carry ``security_domain`` directly, responses derive theirs
from the event they respond to.

All public methods are thread-safe.

:class:`FilteredTrajectory` provides a read-only view filtered by a
security domain scope.
"""

from __future__ import annotations

import threading

from anonframework.core.types.event import Event, EventResponse
from anonframework.core.types.security_domain import Scope, SecurityDomainTag, scope_includes

# Items stored on the trajectory: events and their responses.
TrajectoryItem = Event | EventResponse


def get_domain(item: TrajectoryItem) -> SecurityDomainTag | None:
    """Extract the security domain from a trajectory item.

    For :class:`Event` subclasses, returns ``item.security_domain``.
    For :class:`EventResponse` subclasses, derives the domain from the
    event the response belongs to (recursively).

    Returns:
        The security domain tag, or ``None`` if the item has no domain.
    """
    if isinstance(item, Event):
        return item.security_domain
    if isinstance(item, EventResponse):
        return get_domain(item.event)
    return None


class Trajectory:
    """A thread-safe stream of events and responses for one run.

    The target pushes one-way events via :meth:`emit`. The controller
    also writes controllable events/responses and feedback events.
    Consumers read via :meth:`drain` or :meth:`snapshot`.

    Pass *filtered_scope* to create a :class:`FilteredTrajectory` that
    receives in-scope items at emit time, accessible via :attr:`filtered`.

    All public methods are safe to call concurrently from multiple threads.
    """

    def __init__(
        self,
        filtered_scope: Scope | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._entries: list[TrajectoryItem] = []
        self._closed: bool = False
        self._drain_cursor: int = 0
        self._filter: tuple[Scope, FilteredTrajectory] | None = None
        if filtered_scope is not None:
            self._filter = (filtered_scope, FilteredTrajectory())

    def emit(self, item: TrajectoryItem) -> None:
        """Push an event or response onto the trajectory.

        The item's security domain (via :func:`get_domain`) must not be
        ``None`` — lifecycle events that lack a domain should not be
        persisted.

        Raises:
            RuntimeError: If the trajectory has already been closed.
            ValueError: If the item has no security domain.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot emit to a closed trajectory")
            domain = get_domain(item)
            if domain is None:
                raise ValueError(f"Cannot persist {type(item).__name__} without a security_domain")
            self._entries.append(item)
            if self._filter is not None:
                scope, view = self._filter
                if scope_includes(scope, domain):
                    view._push(item)

    def close(self) -> None:
        """Signal that no more items will be emitted."""
        with self._lock:
            self._closed = True

    def snapshot(self) -> list[TrajectoryItem]:
        """Return all items emitted so far without advancing the drain cursor."""
        with self._lock:
            return list(self._entries)

    def drain(self) -> list[TrajectoryItem]:
        """Return all items emitted since the last ``drain()`` call.

        Non-blocking: returns immediately with whatever is available.
        """
        with self._lock:
            new_entries = self._entries[self._drain_cursor :]
            self._drain_cursor = len(self._entries)
            return new_entries

    @property
    def filtered(self) -> FilteredTrajectory:
        """The filtered view, if *filtered_scope* was provided at construction.

        Raises:
            RuntimeError: If no filtered scope was configured.
        """
        if self._filter is None:
            raise RuntimeError("No filtered view — pass filtered_scope to Trajectory constructor")
        return self._filter[1]


class FilteredTrajectory:
    """Read-only view of trajectory items within a security domain scope.

    Created by passing *filtered_scope* to the :class:`Trajectory`
    constructor, then accessed via :attr:`Trajectory.filtered`. Items
    are pushed by the parent trajectory at emit time.

    **Encapsulation**: This object holds **no reference** to the
    underlying :class:`Trajectory`. ``__slots__`` prevents ``__dict__``.

    :meth:`snapshot` and :meth:`drain` are thread-safe.
    """

    __slots__ = ("_entries", "_lock", "_drain_cursor")

    def __init__(self) -> None:
        self._entries: list[TrajectoryItem] = []
        self._lock = threading.Lock()
        self._drain_cursor: int = 0

    def _push(self, item: TrajectoryItem) -> None:
        """Receive a pre-filtered item from the parent trajectory."""
        with self._lock:
            self._entries.append(item)

    def snapshot(self) -> list[TrajectoryItem]:
        """Return all in-scope items received so far."""
        with self._lock:
            return list(self._entries)

    def drain(self) -> list[TrajectoryItem]:
        """Return in-scope items received since the last ``drain()`` call."""
        with self._lock:
            new_entries = self._entries[self._drain_cursor :]
            self._drain_cursor = len(self._entries)
            return list(new_entries)


ReadableTrajectory = Trajectory | FilteredTrajectory
"""Type alias for objects that expose ``snapshot()`` and ``drain()``."""
