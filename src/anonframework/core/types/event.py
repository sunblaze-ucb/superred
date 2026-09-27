"""Event and EventResponse base classes, plus callback type aliases.

This module defines the abstract foundation of the event system.
Concrete event types live in :mod:`anonframework.core.types.events`.

Kept separate so that :mod:`~anonframework.core.types.trajectory` can
import ``Event`` / ``EventResponse`` without circular dependencies.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from anonframework.core.types.security_domain import SecurityDomainTag


@dataclass(frozen=True, kw_only=True)
class Event:
    """Base class for all events.

    Subclasses declare ``response_types`` as a :class:`ClassVar` tuple of
    allowed response classes.  Empty tuple means any :class:`EventResponse`
    is accepted (the default for base Event).

    Attributes:
        event_id: Unique identifier for this event instance.
        timestamp: When the event was created.
        security_domain: The security domain this event belongs to.
            ``None`` for lifecycle events that are not persisted.
    """

    response_types: ClassVar[tuple[type[EventResponse], ...]] = ()

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    security_domain: SecurityDomainTag | None = None


@dataclass(frozen=True, kw_only=True)
class EventResponse:
    """Base class for responses to events.

    Attributes:
        event: The event this response was produced for.
    """

    event: Event


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

EventHandler = Callable[[Event], None]
"""One-way event callback: fire-and-forget (e.g. target logging via ``emit``)."""

EventResponseHandler = Callable[[Event], Awaitable[EventResponse]]
"""Two-way event callback: sends an event and awaits a response
(e.g. ``send_event`` at controllable points)."""
