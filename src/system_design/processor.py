from __future__ import annotations

import heapq
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np
from manim import BLUE, Mobject, Square, VGroup

from shared.aggregators.moving_sum import MovingSum
from shared.constants import MEDIUM, SMALL
from system_design.message import Message, MessageType

if TYPE_CHECKING:
    from .connection import Connection


class RetryPolicy:
    def __init__(
        self,
        max_retry_attempts: int = 3,
        min_interval: float = 0.1,
        jitter_factor: float = 0.25,
    ):
        self.max_retry_attempts = max_retry_attempts
        self.min_interval = min_interval
        self.jitter_factor = jitter_factor

    def get_retry_interval(self, retry_attempt: int) -> float | None:
        """
        Returns the interval to wait before the next retry attempt. The interval increases
        exponentially with each attempt, with some random jitter.

        :param attempt: The retry attempt number (1-indexed).
        """
        jitter_factor = np.random.uniform(
            1 - self.jitter_factor, 1 + self.jitter_factor
        )
        if retry_attempt < 1:
            raise ValueError(f"Retry count must be 1 or greater, got {retry_attempt}.")
        if retry_attempt > self.max_retry_attempts:
            return None
        return self.min_interval * (2 ** (retry_attempt - 1)) * jitter_factor

# TODO: Split out client and server?
class Processor(VGroup):
    def __init__(
        self,
        req_rate: float = 5.0,
        retry_policy: RetryPolicy | None = None,
        failure_rate: float = 0.0,
        size: float = MEDIUM,
        max_concurrency: int = 10,
        max_queue_size: int = 10,
        **kwargs,
    ):
        super().__init__()

        square = Square(side_length=1.5 * size, color=BLUE, fill_opacity=0.2, **kwargs)
        self.add(square)

        self.time = 0.0

        # Client
        self.gen_req_rate = req_rate
        """Number of requests generated per second (on average) per connection."""
        self.actual_req_rate: dict[Connection, MovingSum] = defaultdict(MovingSum)
        """Number of requests sent in the last second, including retries."""
        self.retry_policy = retry_policy
        self.client_connections: list[tuple[Connection, int]] = []
        """Connections for which this processor is a client, with number of messages in flight."""

        self.retries: dict[Connection, list[tuple[float, Message]]] = defaultdict(list)
        """Messages which failed and are waiting to be retried."""

        self.unused_msgs: list[Message] = []
        """Messages which are not currently in use, ready for recycling."""

        # Server
        self.sim_failure_rate = failure_rate
        """Simulated failure rate, between 0 and 1."""
        self.actual_failure_rate: dict[Connection, MovingSum] = defaultdict(MovingSum)
        """Number of failure responses in the last second"""
        self.max_concurrency = max_concurrency
        self.processing: dict[Connection, list[tuple[float, Message]]] = defaultdict(
            list
        )
        """
        Messages currently being processed by this processor, along with their finish time.
        """
        self.rejections: dict[Connection, list[Message]] = defaultdict(list)
        """
        Messages rejected to be instantly returned.
        """

        self.max_queue_size = max_queue_size
        self.processing_queue: deque[tuple[Connection, Message]] = deque()
        """
        Messages which are queued up to be processed by this processor.
        """

    def _update(self, m: Mobject, dt: float):
        if dt <= 0:
            return
        self.time += dt
        self._generate_requests(dt)
        self._process_responses()

    def start(self):
        """
        Start the processor, allowing it to generate requests and process messages.
        """
        self.add_updater(self._update)

    def stop(self):
        """
        Stop the processor, preventing it from generating requests and processing messages.
        """
        self.remove_updater(self._update)

    # Client
    def _num_new_reqs(self, dt: float) -> int:
        """
        Returns the number of new requests which would have been created in the given period and
        request rate.
        """
        # Number of requests created per dt
        lam = self.gen_req_rate * dt

        return np.random.poisson(lam)

    # Client
    def gen_rps(self) -> float:
        """
        Returns the currently set request rate per second for this processor.
        """
        return self.gen_req_rate

    def actual_rps(self, conn: Connection) -> float:
        """
        Returns the actual request rate per second for the given connection, including retries.
        """
        return self.actual_req_rate[conn].get_value()

    def failure_rate(self, conn: Connection) -> float:
        """
        Returns the real failure rate.
        """
        return self.actual_failure_rate[conn].get_value()

    # Server
    def concurrency(self, include_queued: bool = True) -> int:
        """
        Returns the number of concurrent requests being processed (or queued for processing) by this
        processor.
        """
        processing = sum(len(v) for v in self.processing.values())
        return processing + (self.queued() if include_queued else 0)

    # Server
    def concurrency_by_type(
        self, include_queued: bool = True
    ) -> dict[MessageType, int]:
        """
        Returns the number of concurrent requests being processed (or queued for processing) by
        this processor, grouped by message type.
        """
        concurrency = defaultdict(int)
        for processing in self.processing.values():
            for _, msg in processing:
                concurrency[msg.type] += 1
        if include_queued:
            self._queued_by_type(concurrency)
        return concurrency

    # Server
    def queued(self) -> int:
        """
        Returns the number of messages currently queued for processing.
        """
        return len(self.processing_queue)

    # Server
    def queued_by_type(self) -> dict[MessageType, int]:
        """
        Returns the number of messages currently queued for processing, grouped by message type.
        """
        queued = defaultdict(int)
        return self._queued_by_type(queued)

    # Server
    def _queued_by_type(self, queued: dict[MessageType, int]) -> dict[MessageType, int]:
        for _, msg in self.processing_queue:
            queued[msg.type] += 1
        return queued

    # Client
    def add_client_connection(self, conn: Connection):
        """
        Add a connection for which this processor is a client.
        """
        self.client_connections.append((conn, 0))

    # Client
    def messages_in_flight(self) -> int:
        """
        Returns the total number of messages in flight for this processor.
        This includes both requests sent and responses received.
        """
        return sum(msgs_in_flight for _, msgs_in_flight in self.client_connections)

    # Client
    def _generate_requests(self, dt: float):
        for i, (conn, msgs_in_flight) in enumerate(self.client_connections):
            n = self._num_new_reqs(dt)

            total_reqs = n

            msgs: list[Message] = []

            # New messages
            for _ in range(n):
                req = (
                    # Try to recycle message object
                    self.unused_msgs.pop()
                    if len(self.unused_msgs) > 0
                    else Message(size=SMALL)
                )
                req.reset()
                req.unhide()
                msgs.append(req)

            # Retries
            while len(self.retries[conn]) > 0:
                retry_at, msg = self.retries[conn][0]
                if self.time >= retry_at:
                    heapq.heappop(self.retries[conn])
                    total_reqs += 1
                    msg.set_type(MessageType.RETRY_REQUEST)
                    msg.unhide()
                    msgs.append(msg)
                else:
                    break

            self.actual_req_rate[conn].add_value(total_reqs, dt)

            # Update the count of messages in flight
            self.client_connections[i] = (conn, msgs_in_flight + len(msgs))
            conn.send_requests(msgs, dt)

    # Client
    def _process_responses(self):
        for i, (conn, msgs_in_flight) in enumerate(self.client_connections):
            resps = conn.ready_responses()

            for msg in resps:
                msg.hide()

                if self._try_schedule_retry(msg, conn):
                    continue

                self._return_message_to_pool(msg)

            # Update the count of messages in flight
            self.client_connections[i] = (conn, msgs_in_flight - len(resps))

    # Client
    def _try_schedule_retry(self, msg: Message, conn: Connection) -> bool:
        """
        Attempt to schedule a retry for the message if it's a failure and we have a retry policy.

        Returns True if the message was scheduled for retry, False otherwise.
        """
        if not msg.failure or self.retry_policy is None:
            return False

        retry_interval = self.retry_policy.get_retry_interval(msg.attempt)
        if retry_interval is None:
            return False

        heapq.heappush(self.retries[conn], (self.time + retry_interval, msg))
        return True

    # Client
    def _return_message_to_pool(self, msg: Message):
        """
        Return a message to the unused message pool by hiding it and making it available for reuse.
        """
        self.unused_msgs.append(msg)

    # Server
    def process(self, msg: Message, return_to: Connection):
        """
        Process the given message.
        """
        msg.hide()

        if self.concurrency(include_queued=False) >= self.max_concurrency:
            # If we are at max concurrency...
            if len(self.processing_queue) >= self.max_queue_size:
                # ...and the queue is full, reject the message
                self._reject(msg, return_to)
            else:
                # ... otherwise, try queueing the message for later processing
                self.processing_queue.append((return_to, msg))
            return

        self._process(msg, return_to)

    # Server
    def _process(
        self,
        msg: Message,
        return_to: Connection,
    ):
        finished_at = self.time + Processor._processing_latency()

        msg.failure = np.random.uniform() < self.sim_failure_rate

        conn: list[tuple[float, Message]] = self.processing[return_to]
        heapq.heappush(conn, (finished_at, msg))

    def _reject(self, msg: Message, return_to: Connection):
        """
        Reject the message by scheduling it to instantly return to the connection as a failure.
        """
        self.rejections[return_to].append(msg)

    # Server
    def send_responses(self, conn: Connection, dt: float) -> list[Message]:
        """
        Remove and return any pending responses for the given connection.
        """
        responses = []
        failures = 0

        while len(self.processing[conn]) > 0:
            finished_at, msg = self.processing[conn][0]
            if self.time >= finished_at:
                heapq.heappop(self.processing[conn])

                msg.unhide()
                msg.set_as_response()

                if msg.type.is_failure():
                    failures += 1

                responses.append(msg)

                # If there are messages in the processing queue, start processing the next one
                if len(self.processing_queue) > 0:
                    conn, msg = self.processing_queue.popleft()
                    self._process(msg, conn)
            else:
                break

        # Handle any rejections
        while len(self.rejections[conn]) > 0:
            msg = self.rejections[conn].pop(0)
            msg.unhide()
            msg.set_as_response()
            failures += 1
            responses.append(msg)

        self.actual_failure_rate[conn].add_value(failures, dt)

        return responses

    # Server
    @staticmethod
    def _processing_latency() -> float:
        """
        Returns the processing latency of the processor, in seconds.
        """
        tasks = 3.0  # Number of tasks to simulate
        rate = 8.0  # Average rate of tasks per second
        return np.random.gamma(tasks, 1 / rate)
