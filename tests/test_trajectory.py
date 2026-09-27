"""Unit tests for Trajectory, FilteredTrajectory, and get_domain()."""

from __future__ import annotations

import threading

import pytest

from anonframework.core.types.controllable import Controllable
from anonframework.core.types.event import Event, EventResponse
from anonframework.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    ObservableEvent,
)
from anonframework.core.types.observable import Observable
from anonframework.core.types.security_domain import SecurityDomainTag
from anonframework.core.types.trajectory import (
    Trajectory,
    get_domain,
)

# Reusable tags for trajectory tests
_TAG = SecurityDomainTag("test")
_PARENT = SecurityDomainTag("parent")
_CHILD = SecurityDomainTag("child", parent=_PARENT)
_SIBLING = SecurityDomainTag("sibling", parent=_PARENT)


def _obs(content: str, tag: SecurityDomainTag = _TAG) -> ObservableEvent:
    """Helper to build an ObservableEvent with default tag."""
    return ObservableEvent(
        observable=Observable(name="test", security_domain=tag),
        content=content,
    )


# ---------------------------------------------------------------------------
# get_domain()
# ---------------------------------------------------------------------------


class TestGetDomain:
    def test_event_returns_security_domain(self) -> None:
        event = ObservableEvent(
            observable=Observable(name="x", security_domain=_TAG),
            content="x",
        )
        assert get_domain(event) is _TAG

    def test_event_response_derives_from_event(self) -> None:
        ctrl = Controllable(name="c", security_domain=_TAG)
        event = ControllablePreCallEvent(controllable=ctrl, request="hi")
        response = ControllableInjection(
            event=event,
            controllable=ctrl,
            value="x",
        )
        assert get_domain(response) is _TAG

    def test_lifecycle_event_returns_none(self) -> None:
        event = Event()  # security_domain defaults to None
        assert get_domain(event) is None

    def test_response_to_lifecycle_event_returns_none(self) -> None:
        event = Event()
        response = EventResponse(event=event)
        assert get_domain(response) is None

    def test_unknown_type_returns_none(self) -> None:
        assert get_domain("not an event") is None


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_empty_trajectory(self) -> None:
        t = Trajectory()
        assert t.snapshot() == []
        assert t.drain() == []

    def test_emit_and_snapshot(self) -> None:
        t = Trajectory()
        entry = _obs("Hello")
        t.emit(entry)
        snap = t.snapshot()
        assert len(snap) == 1
        assert snap[0] is entry

    def test_emit_after_close_raises(self) -> None:
        t = Trajectory()
        t.close()
        with pytest.raises(RuntimeError, match="closed"):
            t.emit(_obs("x"))

    def test_emit_rejects_none_domain(self) -> None:
        """Items without a security_domain cannot be persisted."""
        t = Trajectory()
        with pytest.raises(ValueError, match="without a security_domain"):
            t.emit(Event())  # Event() has security_domain=None

    def test_snapshot_does_not_advance_cursor(self) -> None:
        t = Trajectory()
        t.emit(_obs("a"))
        t.snapshot()
        # drain should still see the entry
        drained = t.drain()
        assert len(drained) == 1

    def test_drain_returns_new_entries_only(self) -> None:
        t = Trajectory()
        t.emit(_obs("a"))
        first = t.drain()
        assert len(first) == 1

        # No new entries
        assert t.drain() == []

        # Add another
        t.emit(_obs("b"))
        second = t.drain()
        assert len(second) == 1
        assert second[0].content == "b"

    def test_drain_returns_all_new_since_last_drain(self) -> None:
        t = Trajectory()
        t.emit(_obs("1"))
        t.emit(_obs("2"))
        t.drain()  # advance cursor past both
        t.emit(_obs("3"))
        t.emit(_obs("4"))
        drained = t.drain()
        assert [e.content for e in drained] == ["3", "4"]

    def test_snapshot_grows_with_emits(self) -> None:
        t = Trajectory()
        assert len(t.snapshot()) == 0
        t.emit(_obs("a"))
        assert len(t.snapshot()) == 1
        t.emit(_obs("b"))
        assert len(t.snapshot()) == 2

    def test_close_idempotent(self) -> None:
        t = Trajectory()
        t.close()
        t.close()  # should not raise
        # Still closed — emit should still fail
        with pytest.raises(RuntimeError, match="closed"):
            t.emit(_obs("x"))

    def test_snapshot_returns_copy(self) -> None:
        """Mutating snapshot list does not affect trajectory."""
        t = Trajectory()
        t.emit(_obs("a"))
        snap = t.snapshot()
        snap.clear()
        assert len(t.snapshot()) == 1

    def test_thread_safety_emit(self) -> None:
        """Multiple threads can emit concurrently without data loss."""
        t = Trajectory()
        n_threads = 10
        n_per_thread = 100
        barrier = threading.Barrier(n_threads)

        def emitter() -> None:
            barrier.wait()
            for i in range(n_per_thread):
                t.emit(_obs(str(i)))

        threads = [threading.Thread(target=emitter) for _ in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(t.snapshot()) == n_threads * n_per_thread

    def test_snapshot_after_close(self) -> None:
        """snapshot() still works correctly after close()."""
        t = Trajectory()
        t.emit(_obs("a"))
        t.emit(_obs("b"))
        t.close()
        snap = t.snapshot()
        assert len(snap) == 2
        assert snap[0].content == "a"
        assert snap[1].content == "b"

    def test_drain_after_close(self) -> None:
        """drain() still works correctly after close()."""
        t = Trajectory()
        t.emit(_obs("a"))
        t.close()
        drained = t.drain()
        assert len(drained) == 1
        assert drained[0].content == "a"
        # Second drain returns nothing
        assert t.drain() == []

    def test_drain_returns_copy(self) -> None:
        """Mutating the drained list does not affect the trajectory."""
        t = Trajectory()
        t.emit(_obs("a"))
        drained = t.drain()
        drained.clear()
        # snapshot should still have the entry
        assert len(t.snapshot()) == 1


# ---------------------------------------------------------------------------
# FilteredTrajectory
# ---------------------------------------------------------------------------


class TestFilteredTrajectory:
    def test_snapshot_filters_by_scope(self) -> None:
        t = Trajectory(filtered_scope=frozenset({_CHILD}))
        t.emit(_obs("child", _CHILD))
        t.emit(_obs("sibling", _SIBLING))
        snap = t.filtered.snapshot()
        assert len(snap) == 1
        assert snap[0].content == "child"

    def test_snapshot_includes_descendants(self) -> None:
        t = Trajectory(filtered_scope=frozenset({_PARENT}))
        t.emit(_obs("child", _CHILD))
        t.emit(_obs("sibling", _SIBLING))
        assert len(t.filtered.snapshot()) == 2

    def test_snapshot_empty_when_nothing_in_scope(self) -> None:
        t = Trajectory(filtered_scope=frozenset({_CHILD}))
        t.emit(_obs("x", _SIBLING))
        assert t.filtered.snapshot() == []

    def test_drain_returns_new_in_scope_entries(self) -> None:
        t = Trajectory(filtered_scope=frozenset({_CHILD}))
        t.emit(_obs("a", _CHILD))
        t.emit(_obs("b", _SIBLING))

        first = t.filtered.drain()
        assert len(first) == 1
        assert first[0].content == "a"

        # No new entries
        assert t.filtered.drain() == []

        # Add more entries — only in-scope ones returned
        t.emit(_obs("c", _CHILD))
        t.emit(_obs("d", _SIBLING))
        second = t.filtered.drain()
        assert len(second) == 1
        assert second[0].content == "c"

    def test_drain_cursor_independent_of_underlying(self) -> None:
        """FilteredTrajectory's drain cursor is independent of Trajectory's."""
        t = Trajectory(filtered_scope=frozenset({_CHILD}))
        t.emit(_obs("a", _CHILD))

        # Drain the underlying trajectory
        t.drain()
        # FilteredTrajectory should still see the entry on its first drain
        assert len(t.filtered.drain()) == 1

    def test_no_emit_or_close(self) -> None:
        """FilteredTrajectory is read-only — no emit() or close()."""
        t = Trajectory(filtered_scope=frozenset({_TAG}))
        assert not hasattr(t.filtered, "emit")
        assert not hasattr(t.filtered, "close")

    def test_no_trajectory_reference(self) -> None:
        """FilteredTrajectory holds no reference to the underlying Trajectory."""
        t = Trajectory(filtered_scope=frozenset({_TAG}))
        filtered = t.filtered
        for attr in dir(filtered):
            val = getattr(filtered, attr)
            assert not isinstance(val, Trajectory), (
                f"attribute {attr!r} references the underlying Trajectory"
            )
        # No __dict__ (slots-only)
        assert not hasattr(filtered, "__dict__")

    def test_thread_safety(self) -> None:
        """Concurrent drain() calls on FilteredTrajectory are safe."""
        t = Trajectory(filtered_scope=frozenset({_CHILD}))
        for i in range(100):
            t.emit(_obs(str(i), _CHILD))
        results: list[list[object]] = []
        lock = threading.Lock()

        def drainer() -> None:
            drained = t.filtered.drain()
            with lock:
                results.append(drained)

        threads = [threading.Thread(target=drainer) for _ in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        total = sum(len(r) for r in results)
        assert total == 100

    def test_empty_trajectory(self) -> None:
        """FilteredTrajectory on an empty trajectory returns empty results."""
        t = Trajectory(filtered_scope=frozenset({_CHILD}))
        assert t.filtered.snapshot() == []
        assert t.filtered.drain() == []

    def test_scope_matches_everything(self) -> None:
        """Parent scope includes all descendants."""
        t = Trajectory(filtered_scope=frozenset({_PARENT}))
        t.emit(_obs("child", _CHILD))
        t.emit(_obs("sibling", _SIBLING))
        t.emit(_obs("parent", _PARENT))
        assert len(t.filtered.snapshot()) == 3

    def test_live_updates_visible(self) -> None:
        """Entries emitted after construction are pushed to filtered view."""
        t = Trajectory(filtered_scope=frozenset({_CHILD}))
        assert t.filtered.snapshot() == []

        t.emit(_obs("new", _CHILD))
        assert len(t.filtered.snapshot()) == 1
        assert t.filtered.drain()[0].content == "new"

    def test_filtered_property_raises_without_scope(self) -> None:
        """Accessing .filtered without filtered_scope raises RuntimeError."""
        t = Trajectory()
        with pytest.raises(RuntimeError, match="No filtered view"):
            t.filtered

    def test_concurrent_emit_and_filtered_reads(self) -> None:
        """Thread safety: concurrent emit() and filtered snapshot()/drain()."""
        t = Trajectory(filtered_scope=frozenset({_CHILD}))
        n_emitters = 5
        n_per_emitter = 100
        barrier = threading.Barrier(n_emitters + 1)

        def emitter() -> None:
            barrier.wait()
            for i in range(n_per_emitter):
                t.emit(_obs(str(i), _CHILD))

        def reader() -> None:
            barrier.wait()
            for _ in range(50):
                t.filtered.snapshot()

        threads = [threading.Thread(target=emitter) for _ in range(n_emitters)]
        threads.append(threading.Thread(target=reader))
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        # All entries pushed through
        assert len(t.filtered.snapshot()) == n_emitters * n_per_emitter

    def test_event_response_domain_derived_from_event(self) -> None:
        """EventResponse domain is derived from its event's domain for filtering."""
        t = Trajectory(filtered_scope=frozenset({_CHILD}))

        ctrl = Controllable(name="c", security_domain=_CHILD)
        event = ControllablePreCallEvent(controllable=ctrl, request="hi")
        response = ControllableInjection(
            event=event,
            controllable=ctrl,
            value="x",
        )

        t.emit(event)
        t.emit(response)

        # Both event and response should appear in the filtered view
        assert len(t.filtered.snapshot()) == 2

    def test_out_of_scope_event_response_excluded(self) -> None:
        """EventResponse for out-of-scope event is excluded from filtered view."""
        t = Trajectory(filtered_scope=frozenset({_CHILD}))

        ctrl = Controllable(name="c", security_domain=_SIBLING)
        event = ControllablePreCallEvent(controllable=ctrl, request="hi")
        response = ControllableInjection(
            event=event,
            controllable=ctrl,
            value="x",
        )

        t.emit(event)
        t.emit(response)

        # Neither should appear in the filtered view (SIBLING not in CHILD scope)
        assert len(t.filtered.snapshot()) == 0
