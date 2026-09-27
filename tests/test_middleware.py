"""Unit tests for middleware: compose() and security_domain_filter()."""

from __future__ import annotations

from anonframework.core.middleware import (
    Middleware,
    compose,
    security_domain_filter,
    trajectory_recorder,
)
from anonframework.core.types.controllable import Controllable
from anonframework.core.types.event import Event, EventResponse, EventResponseHandler
from anonframework.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunStartEvent,
)
from anonframework.core.types.security_domain import SecurityDomainTag
from anonframework.core.types.trajectory import Trajectory

PARENT_TAG = SecurityDomainTag("parent")
CHILD_TAG = SecurityDomainTag("child", parent=PARENT_TAG)
SIBLING_TAG = SecurityDomainTag("sibling", parent=PARENT_TAG)


# ---------------------------------------------------------------------------
# compose()
# ---------------------------------------------------------------------------


class TestCompose:
    async def test_identity(self) -> None:
        """compose() with no middlewares returns the handler unchanged."""
        calls: list[Event] = []

        async def handler(event: Event) -> EventResponse:
            calls.append(event)
            return EventResponse(event=event)

        wrapped = compose()(handler)
        event = RunStartEvent(trajectory=Trajectory())
        await wrapped(event)
        assert len(calls) == 1

    async def test_ordering_left_to_right(self) -> None:
        """Middlewares apply left-to-right (first listed = outermost)."""
        order: list[str] = []

        def make_mw(name: str) -> Middleware:
            def mw(handler: EventResponseHandler) -> EventResponseHandler:
                async def wrapped(event: Event) -> EventResponse:
                    order.append(f"{name}_before")
                    resp = await handler(event)
                    order.append(f"{name}_after")
                    return resp

                return wrapped

            return mw

        async def inner(event: Event) -> EventResponse:
            order.append("inner")
            return EventResponse(event=event)

        wrapped = compose(make_mw("a"), make_mw("b"))(inner)
        await wrapped(RunStartEvent(trajectory=Trajectory()))

        assert order == ["a_before", "b_before", "inner", "b_after", "a_after"]

    async def test_single_middleware(self) -> None:
        """compose(a)(handler) == a(handler)."""
        called = False

        def mw(handler: EventResponseHandler) -> EventResponseHandler:
            async def wrapped(event: Event) -> EventResponse:
                nonlocal called
                called = True
                return await handler(event)

            return wrapped

        async def handler(event: Event) -> EventResponse:
            return EventResponse(event=event)

        wrapped = compose(mw)(handler)
        await wrapped(Event())
        assert called


# ---------------------------------------------------------------------------
# security_domain_filter()
# ---------------------------------------------------------------------------


class TestSecurityDomainFilter:
    async def test_blocks_out_of_scope_pre_call(self) -> None:
        """Pre-call event for a domain outside scope is blocked."""

        async def handler(event: Event) -> EventResponse:
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value="x",
            )

        filtered = security_domain_filter(frozenset({CHILD_TAG}))(handler)

        # SIBLING_TAG is NOT included by CHILD_TAG
        c = Controllable(name="c", security_domain=SIBLING_TAG)
        event = ControllablePreCallEvent(controllable=c, request="hi")
        response = await filtered(event)

        assert isinstance(response, ControllableNoInjection)

    async def test_blocks_out_of_scope_post_call(self) -> None:
        """Post-call event outside scope is also blocked."""

        async def handler(event: Event) -> EventResponse:
            return EventResponse(event=event)

        filtered = security_domain_filter(frozenset({CHILD_TAG}))(handler)

        c = Controllable(name="c", security_domain=SIBLING_TAG)
        event = ControllablePostCallEvent(controllable=c, request="hi", answer="bye")
        response = await filtered(event)

        assert isinstance(response, ControllableNoInjection)

    async def test_passes_in_scope(self) -> None:
        """Events within scope are forwarded to the handler."""

        async def handler(event: Event) -> EventResponse:
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value="x",
            )

        filtered = security_domain_filter(frozenset({PARENT_TAG}))(handler)

        c = Controllable(name="c", security_domain=CHILD_TAG)
        event = ControllablePreCallEvent(controllable=c, request="hi")
        response = await filtered(event)

        assert isinstance(response, ControllableInjection)

    async def test_passes_non_controllable_events(self) -> None:
        """Non-controllable events always pass through."""

        async def handler(event: Event) -> EventResponse:
            return EventResponse(event=event)

        filtered = security_domain_filter(frozenset({CHILD_TAG}))(handler)
        event = RunStartEvent(trajectory=Trajectory())
        response = await filtered(event)

        assert isinstance(response, EventResponse)

    async def test_blocked_event_not_forwarded(self) -> None:
        """Blocked events do not reach the handler."""

        async def handler(event: Event) -> EventResponse:
            raise AssertionError("Should not be called")

        filtered = security_domain_filter(frozenset({CHILD_TAG}))(handler)

        c = Controllable(name="c", security_domain=SIBLING_TAG)
        event = ControllablePreCallEvent(controllable=c, request="hi")
        response = await filtered(event)

        assert isinstance(response, ControllableNoInjection)


# ---------------------------------------------------------------------------
# trajectory_recorder()
# ---------------------------------------------------------------------------


class TestTrajectoryRecorder:
    async def test_records_in_scope_event_and_response(self) -> None:
        """In-scope event and its response are both recorded."""
        trajectory = Trajectory(filtered_scope=frozenset({PARENT_TAG}))

        async def handler(event: Event) -> EventResponse:
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value="x",
            )

        wrapped = trajectory_recorder(trajectory)(handler)
        c = Controllable(name="c", security_domain=CHILD_TAG)
        await wrapped(ControllablePreCallEvent(controllable=c, request="hi"))

        entries = trajectory.snapshot()
        events = [e for e in entries if isinstance(e, ControllablePreCallEvent)]
        responses = [e for e in entries if isinstance(e, ControllableInjection)]
        assert len(events) == 1
        assert len(responses) == 1
        assert responses[0].value == "x"

    async def test_records_out_of_scope_event_and_noinjection(self) -> None:
        """Out-of-scope events blocked by the filter are still recorded
        by the recorder (because recorder is outermost in the compose chain)."""
        trajectory = Trajectory(filtered_scope=frozenset({CHILD_TAG}))

        # Compose in the correct order: recorder outermost, filter inner
        wrapped = compose(
            trajectory_recorder(trajectory),
            security_domain_filter(frozenset({CHILD_TAG})),
        )(self._unreachable_handler)

        # SIBLING is out of scope for CHILD
        c = Controllable(name="c", security_domain=SIBLING_TAG)
        await wrapped(ControllablePreCallEvent(controllable=c, request="hi"))

        entries = trajectory.snapshot()
        events = [e for e in entries if isinstance(e, ControllablePreCallEvent)]
        responses = [e for e in entries if isinstance(e, ControllableNoInjection)]
        # Both event and ControllableNoInjection response are recorded
        assert len(events) == 1
        assert len(responses) == 1

    async def test_wrong_compose_order_loses_out_of_scope_events(self) -> None:
        """If filter is outermost (wrong order), out-of-scope events are
        never seen by the recorder — they vanish from the trajectory."""
        trajectory = Trajectory(filtered_scope=frozenset({CHILD_TAG}))

        # WRONG order: filter outermost, recorder inner
        wrapped = compose(
            security_domain_filter(frozenset({CHILD_TAG})),
            trajectory_recorder(trajectory),
        )(self._unreachable_handler)

        c = Controllable(name="c", security_domain=SIBLING_TAG)
        await wrapped(ControllablePreCallEvent(controllable=c, request="hi"))

        # Nothing recorded — the filter returned ControllableNoInjection before
        # the recorder ever ran
        entries = trajectory.snapshot()
        assert len([e for e in entries if isinstance(e, ControllablePreCallEvent)]) == 0
        assert (
            len(
                [
                    e
                    for e in entries
                    if isinstance(e, (ControllableInjection, ControllableNoInjection))
                ]
            )
            == 0
        )

    @staticmethod
    async def _unreachable_handler(event: Event) -> EventResponse:
        raise AssertionError("Should not be called for out-of-scope events")


# ---------------------------------------------------------------------------
# Who declined: the scope filter, or the optimizer?
# ---------------------------------------------------------------------------


async def test_scope_filter_marks_its_own_declines() -> None:
    """A framework decline must be distinguishable from an attacker decline.

    Both are a ``ControllableNoInjection``. Without ``declined_by`` they are
    identical on the trajectory, and "the attacker had no access here" reads
    exactly like "the attacker had access and passed" -- opposite findings.
    """
    out_of_scope = Controllable(name="c", security_domain=SIBLING_TAG)

    async def handler(event: Event) -> EventResponse:  # pragma: no cover - never reached
        raise AssertionError("the optimizer must not be consulted for an out-of-scope event")

    filtered = security_domain_filter(frozenset({CHILD_TAG}))(handler)
    resp = await filtered(ControllablePreCallEvent(controllable=out_of_scope, request="q"))

    assert isinstance(resp, ControllableNoInjection)
    assert resp.declined_by == "scope"


async def test_an_optimizer_decline_defaults_to_optimizer() -> None:
    """In-scope events reach the handler, whose decline stays attributed to it."""
    in_scope = Controllable(name="c", security_domain=CHILD_TAG)

    async def handler(event: Event) -> EventResponse:
        assert isinstance(event, ControllablePreCallEvent)
        # Constructed the way every optimizer constructs it: no declined_by.
        return ControllableNoInjection(event=event, controllable=event.controllable)

    filtered = security_domain_filter(frozenset({CHILD_TAG}))(handler)
    resp = await filtered(ControllablePreCallEvent(controllable=in_scope, request="q"))

    assert isinstance(resp, ControllableNoInjection)
    assert resp.declined_by == "optimizer"
