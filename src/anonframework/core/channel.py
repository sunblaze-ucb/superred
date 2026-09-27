"""Event channel: thread-safe bidirectional event-response communication.

The channel decouples event senders (controller/target side) from receivers
(optimizer side). The sender puts an event and awaits a response. The
receiver pulls events at its own pace and responds when ready.

Error propagation:
    If the receiver encounters an error, it calls ``envelope.reject(exc)``
    to propagate the exception to the sender. If the receiver task dies,
    ``channel.set_error(exc)`` poisons the channel — all pending and
    future ``send()`` calls raise the exception.

Thread safety:
    - ``send()`` and ``receive()`` run on the asyncio event loop thread.
    - ``respond()``, ``reject()``, ``close()``, and ``set_error()`` use
      ``call_soon_threadsafe`` so they can be called safely from any thread.
    - The internal ``asyncio.Queue`` is only accessed from the event loop.

Process safety:
    This is an in-process implementation. The interface (send/receive/
    respond/close) is designed so a future process-safe implementation
    (multiprocessing pipes, sockets, etc.) can provide the same contract.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

from anonframework.core.types.event import Event, EventResponse


class EventEnvelope:
    """An event paired with its response mechanism.

    The receiver calls :meth:`respond` exactly once to deliver a successful
    response, or :meth:`reject` to propagate an exception to the sender.
    Thread-safe — both methods may be called from any thread.

    Attributes:
        event: The event to handle.
    """

    __slots__ = ("event", "_future", "_loop", "_responded", "_lock")

    def __init__(
        self,
        event: Event,
        future: asyncio.Future[EventResponse],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.event = event
        self._future = future
        self._loop = loop
        self._responded = False
        self._lock = threading.Lock()

    def respond(self, response: EventResponse) -> None:
        """Deliver a successful response for this event.

        Thread-safe. Must be called exactly once (mutually exclusive with
        :meth:`reject`). The response type is validated against the
        event's ``response_types`` declaration.

        Raises:
            RuntimeError: If called more than once or after reject.
            TypeError: If the response type is not allowed for this event.
        """
        # Validate response type against event's declared response_types.
        # Empty tuple = any EventResponse accepted (base Event default).
        allowed = self.event.response_types
        if allowed and not isinstance(response, allowed):
            allowed_names = ", ".join(t.__name__ for t in allowed)
            error = TypeError(
                f"{type(self.event).__name__} requires response of type "
                f"{allowed_names}, got {type(response).__name__}"
            )
            # Reject the envelope so the sender doesn't deadlock
            self.reject(error)
            raise error
        with self._lock:
            if self._responded:
                raise RuntimeError("EventEnvelope already responded/rejected")
            self._responded = True
        self._loop.call_soon_threadsafe(self._settle, self._future.set_result, response)

    def _settle(self, complete: Callable[[Any], None], value: Any) -> None:
        """Complete the sender's future, unless the sender is already gone.

        The ``await`` in :meth:`EventChannel.send` can be cancelled out from
        under us -- ``task_time_cap_s`` does exactly that -- which cancels the
        future. Calling ``set_result``/``set_exception`` on it then raises
        ``InvalidStateError`` from inside a ``call_soon_threadsafe`` callback,
        where no caller can catch it: it surfaces only as a loop-level
        "Exception in callback" and the responding thread never learns of it.

        Checking ``done()`` here is race-free even though ``respond`` may be
        called from any thread, because this runs on the loop thread, which is
        the only thread that ever completes the future.
        """
        if not self._future.done():
            complete(value)

    def reject(self, error: BaseException) -> None:
        """Propagate an exception to the sender.

        Thread-safe. Must be called at most once (mutually exclusive with
        :meth:`respond`). The sender's ``await channel.send(...)`` will
        raise this exception.

        If already responded/rejected, this is a no-op (safe to call as
        cleanup in error paths).
        """
        with self._lock:
            if self._responded:
                return
            self._responded = True
        self._loop.call_soon_threadsafe(self._settle, self._future.set_exception, error)


class EventChannel:
    """Thread-safe bidirectional event-response channel.

    The send side puts events and awaits responses. The receive side
    pulls events at its own pace and responds via the envelope.

    Supports ``async for`` iteration on the receive side::

        async for envelope in channel:
            response = handle(envelope.event)
            envelope.respond(response)

    Iteration ends when the channel is closed and all envelopes have
    been consumed.

    Error propagation:
        Call :meth:`set_error` to poison the channel. All pending
        ``send()`` futures are resolved with the exception, and future
        ``send()`` calls raise it immediately.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[EventEnvelope | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._error: BaseException | None = None
        self._close_lock = threading.Lock()

    async def send(self, event: Event) -> EventResponse:
        """Send an event and wait for its response.

        Must be called from the event loop thread (i.e., from an async
        context on the loop that the channel is bound to).

        Args:
            event: The event to send.

        Returns:
            The response produced by the receiver.

        Raises:
            RuntimeError: If the channel is already closed.
            Exception: If the channel has been poisoned via :meth:`set_error`,
                re-raises the stored exception.
        """
        if self._error is not None:
            raise self._error
        if self._closed:
            raise RuntimeError("Cannot send on a closed EventChannel")
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        future: asyncio.Future[EventResponse] = loop.create_future()
        envelope = EventEnvelope(event=event, future=future, loop=loop)
        await self._queue.put(envelope)
        return await future

    async def receive(self) -> EventEnvelope | None:
        """Receive the next event envelope.

        Returns ``None`` when the channel has been closed and all pending
        envelopes have been consumed.

        Must be called from the event loop thread.
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return await self._queue.get()

    def close(self) -> None:
        """Signal that no more events will be sent.

        Thread-safe — may be called from any thread. Puts a sentinel
        (``None``) on the queue so the receiver knows to stop.
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._put_sentinel()

    def set_error(self, error: BaseException) -> None:
        """Poison the channel with an exception.

        Thread-safe. All queued envelopes have their futures rejected
        with the exception. Future ``send()`` calls raise it immediately.
        The channel is also closed (receiver gets sentinel).

        Idempotent — second call is a no-op.
        """
        with self._close_lock:
            if self._closed:
                return
            self._error = error
            self._closed = True
        # Drain queued envelopes — reject their pending futures
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not None:
                item.reject(error)
        self._put_sentinel()

    def _put_sentinel(self) -> None:
        """Put the close sentinel on the queue."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._queue.put_nowait, None)
        else:
            self._queue.put_nowait(None)

    # -- Async iteration ---------------------------------------------------

    def __aiter__(self) -> AsyncIterator[EventEnvelope]:
        return self

    async def __anext__(self) -> EventEnvelope:
        envelope = await self.receive()
        if envelope is None:
            raise StopAsyncIteration
        return envelope
