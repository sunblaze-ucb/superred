"""Unit tests for EventChannel edge cases beyond the basic tests in test_controller.py."""

from __future__ import annotations

import asyncio
import threading

import pytest

from anonframework.core.channel import EventChannel, EventEnvelope
from anonframework.core.types.event import Event, EventResponse
from anonframework.core.types.events import RunEndEvent, RunEndResponse, RunStartEvent
from anonframework.core.types.trajectory import Trajectory


class TestEventEnvelope:
    async def test_respond_resolves_future(self) -> None:
        loop = asyncio.get_running_loop()
        event = Event()
        future: asyncio.Future[EventResponse] = loop.create_future()
        envelope = EventEnvelope(event=event, future=future, loop=loop)

        response = EventResponse(event=event)
        envelope.respond(response)
        result = await future
        assert result is response

    async def test_respond_after_the_sender_was_cancelled_is_quiet(self) -> None:
        """A cancelled sender leaves a done future; completing it must not throw.

        ``task_time_cap_s`` cancels a task that may be parked in
        ``channel.send``, which cancels the future.  The receiver -- a separate
        task, unaware -- then answers the envelope it already holds.  Setting a
        result on the cancelled future raised ``InvalidStateError`` from inside
        the ``call_soon_threadsafe`` callback, where no caller could catch it:
        it surfaced only as a loop-level "Exception in callback".
        """
        loop = asyncio.get_running_loop()
        errors: list[dict] = []
        loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx))

        event = Event()
        future: asyncio.Future[EventResponse] = loop.create_future()
        envelope = EventEnvelope(event=event, future=future, loop=loop)
        future.cancel()

        envelope.respond(EventResponse(event=event))
        await asyncio.sleep(0)
        assert errors == []

    async def test_reject_after_the_sender_was_cancelled_is_quiet(self) -> None:
        """Same for the error path: ``set_error`` drains and rejects queued
        envelopes, whose senders may already have been cancelled."""
        loop = asyncio.get_running_loop()
        errors: list[dict] = []
        loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx))

        event = Event()
        future: asyncio.Future[EventResponse] = loop.create_future()
        envelope = EventEnvelope(event=event, future=future, loop=loop)
        future.cancel()

        envelope.reject(RuntimeError("boom"))
        await asyncio.sleep(0)
        assert errors == []

    async def test_double_respond_raises_runtime_error(self) -> None:
        loop = asyncio.get_running_loop()
        event = Event()
        future: asyncio.Future[EventResponse] = loop.create_future()
        envelope = EventEnvelope(event=event, future=future, loop=loop)

        envelope.respond(EventResponse(event=event))
        with pytest.raises(RuntimeError, match="already responded"):
            envelope.respond(EventResponse(event=event))


class TestEventChannelEdgeCases:
    async def test_close_before_any_send_or_receive(self) -> None:
        """Closing before loop is captured puts sentinel via put_nowait."""
        channel = EventChannel()
        channel.close()
        result = await channel.receive()
        assert result is None

    async def test_receive_captures_loop(self) -> None:
        """receive() captures the event loop on first call."""
        channel = EventChannel()
        assert channel._loop is None

        # Close after receive captures the loop
        async def close_after_delay() -> None:
            await asyncio.sleep(0.01)
            channel.close()

        asyncio.create_task(close_after_delay())
        result = await channel.receive()
        assert result is None
        assert channel._loop is not None

    async def test_multiple_concurrent_sends(self) -> None:
        """Multiple senders can await concurrently; each gets its own response."""
        channel = EventChannel()
        n = 5
        events = [RunStartEvent(trajectory=Trajectory()) for _ in range(n)]

        async def consumer() -> None:
            for _ in range(n):
                envelope = await channel.receive()
                assert envelope is not None
                envelope.respond(EventResponse(event=envelope.event))

        consumer_task = asyncio.create_task(consumer())

        # Send all concurrently
        responses = await asyncio.gather(*(channel.send(e) for e in events))
        await consumer_task

        assert len(responses) == n
        # Each response references its corresponding event
        response_event_ids = {r.event.event_id for r in responses}
        sent_event_ids = {e.event_id for e in events}
        assert response_event_ids == sent_event_ids

    async def test_respond_from_thread(self) -> None:
        """respond() is thread-safe via call_soon_threadsafe."""
        channel = EventChannel()
        event = RunStartEvent(trajectory=Trajectory())

        async def receiver() -> None:
            envelope = await channel.receive()
            assert envelope is not None

            # Respond from a background thread
            def thread_respond() -> None:
                envelope.respond(EventResponse(event=event))

            t = threading.Thread(target=thread_respond)
            t.start()
            t.join()

        recv_task = asyncio.create_task(receiver())
        response = await channel.send(event)
        await recv_task
        assert response.event is event

    async def test_close_from_thread(self) -> None:
        """close() is thread-safe via call_soon_threadsafe."""
        channel = EventChannel()
        # First send to capture the loop
        event = RunStartEvent(trajectory=Trajectory())

        async def receiver() -> None:
            envelope = await channel.receive()
            assert envelope is not None
            envelope.respond(EventResponse(event=envelope.event))

        recv_task = asyncio.create_task(receiver())
        await channel.send(event)
        await recv_task

        # Now close from a thread
        def thread_close() -> None:
            channel.close()

        t = threading.Thread(target=thread_close)
        t.start()
        t.join()

        result = await channel.receive()
        assert result is None

    async def test_iteration_stops_on_close(self) -> None:
        """Async iteration terminates cleanly when channel is closed."""
        channel = EventChannel()
        collected: list[Event] = []

        async def iterate() -> None:
            async for envelope in channel:
                collected.append(envelope.event)
                envelope.respond(EventResponse(event=envelope.event))

        task = asyncio.create_task(iterate())
        e = RunStartEvent(trajectory=Trajectory())
        await channel.send(e)
        channel.close()
        await task

        assert len(collected) == 1
        assert collected[0] is e

    async def test_send_after_close_raises(self) -> None:
        """send() on a closed channel raises RuntimeError."""
        channel = EventChannel()
        channel.close()
        with pytest.raises(RuntimeError, match="closed"):
            await channel.send(Event())

    async def test_respond_validates_response_type(self) -> None:
        """respond() rejects responses not in event.response_types."""
        channel = EventChannel()
        event = RunEndEvent()

        async def receiver() -> None:
            envelope = await channel.receive()
            assert envelope is not None
            with pytest.raises(TypeError, match="RunEndResponse"):
                envelope.respond(EventResponse(event=event))

        recv_task = asyncio.create_task(receiver())
        # The sender gets the TypeError as an exception on the future
        with pytest.raises(TypeError, match="RunEndResponse"):
            await channel.send(event)
        await recv_task

    async def test_respond_accepts_valid_response_type(self) -> None:
        """respond() accepts responses declared in event.response_types."""

        channel = EventChannel()
        event = RunEndEvent()

        async def receiver() -> None:
            envelope = await channel.receive()
            assert envelope is not None
            envelope.respond(RunEndResponse(event=event, done=False))

        recv_task = asyncio.create_task(receiver())
        response = await channel.send(event)
        await recv_task
        assert isinstance(response, RunEndResponse)

    @pytest.mark.parametrize(
        "event_cls",
        [
            "ControllablePreCallEvent",
            "ControllablePostCallEvent",
        ],
    )
    async def test_respond_rejects_wrong_type_for_controllable(
        self,
        event_cls: str,
    ) -> None:
        """respond() rejects wrong response for controllable events."""
        from anonframework.core.types.controllable import Controllable
        from anonframework.core.types.events import (
            ControllablePostCallEvent,
            ControllablePreCallEvent,
        )
        from anonframework.core.types.security_domain import SecurityDomainTag

        channel = EventChannel()
        tag = SecurityDomainTag("t")
        ctrl = Controllable(name="c", security_domain=tag)
        if event_cls == "ControllablePreCallEvent":
            event = ControllablePreCallEvent(controllable=ctrl, request="hi")
        else:
            event = ControllablePostCallEvent(
                controllable=ctrl,
                request="hi",
                answer="bye",
            )

        async def receiver() -> None:
            envelope = await channel.receive()
            assert envelope is not None
            with pytest.raises(TypeError, match="ControllableInjection"):
                envelope.respond(EventResponse(event=event))

        recv_task = asyncio.create_task(receiver())
        with pytest.raises(TypeError, match="ControllableInjection"):
            await channel.send(event)
        await recv_task

    async def test_respond_allows_any_for_base_event(self) -> None:
        """Base Event has empty response_types — any EventResponse accepted."""
        channel = EventChannel()
        event = Event()

        async def receiver() -> None:
            envelope = await channel.receive()
            assert envelope is not None
            envelope.respond(EventResponse(event=event))

        recv_task = asyncio.create_task(receiver())
        response = await channel.send(event)
        await recv_task
        assert isinstance(response, EventResponse)


class TestEnvelopeReject:
    """Tests for envelope.reject() — exception propagation to sender."""

    async def test_reject_propagates_exception_to_sender(self) -> None:
        channel = EventChannel()
        event = Event()
        error = ValueError("test error")

        async def receiver() -> None:
            envelope = await channel.receive()
            assert envelope is not None
            envelope.reject(error)

        recv_task = asyncio.create_task(receiver())
        with pytest.raises(ValueError, match="test error"):
            await channel.send(event)
        await recv_task

    async def test_reject_after_respond_is_noop(self) -> None:
        """reject() after respond() is safe (no-op)."""
        channel = EventChannel()
        event = Event()

        async def receiver() -> None:
            envelope = await channel.receive()
            assert envelope is not None
            envelope.respond(EventResponse(event=event))
            envelope.reject(ValueError("ignored"))

        recv_task = asyncio.create_task(receiver())
        response = await channel.send(event)
        await recv_task
        assert isinstance(response, EventResponse)

    async def test_respond_after_reject_raises(self) -> None:
        """respond() after reject() raises RuntimeError."""
        channel = EventChannel()
        event = Event()

        async def receiver() -> None:
            envelope = await channel.receive()
            assert envelope is not None
            envelope.reject(ValueError("first"))
            with pytest.raises(RuntimeError, match="already responded"):
                envelope.respond(EventResponse(event=event))

        recv_task = asyncio.create_task(receiver())
        with pytest.raises(ValueError, match="first"):
            await channel.send(event)
        await recv_task


class TestChannelSetError:
    """Tests for channel.set_error() — channel poisoning."""

    async def test_set_error_rejects_queued_envelopes(self) -> None:
        """Envelopes in the queue get their futures rejected."""
        channel = EventChannel()
        event = Event()
        # Put an envelope on the queue (no receiver yet)
        send_task = asyncio.create_task(channel.send(event))
        await asyncio.sleep(0.01)  # let send() put envelope on queue
        channel.set_error(ValueError("poisoned"))
        with pytest.raises(ValueError, match="poisoned"):
            await send_task

    async def test_set_error_makes_future_sends_raise(self) -> None:
        """After set_error, send() raises immediately."""
        channel = EventChannel()
        channel.set_error(ValueError("dead"))
        with pytest.raises(ValueError, match="dead"):
            await channel.send(Event())

    async def test_set_error_closes_channel(self) -> None:
        """set_error also closes the channel (receiver gets sentinel)."""
        channel = EventChannel()
        channel.set_error(ValueError("done"))
        result = await channel.receive()
        assert result is None

    async def test_set_error_idempotent(self) -> None:
        """Second set_error is a no-op."""
        channel = EventChannel()
        channel.set_error(ValueError("first"))
        channel.set_error(ValueError("second"))
        with pytest.raises(ValueError, match="first"):
            await channel.send(Event())
