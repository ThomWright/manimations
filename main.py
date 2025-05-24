from __future__ import annotations

import collections
import heapq
from enum import Enum
from typing import cast

import numpy as np
from manim import (
    BLUE,
    DEGREES,
    DOWN,
    GREEN,
    LEFT,
    LIGHTER_GRAY,
    RED,
    RIGHT,
    UP,
    X_AXIS,
    YELLOW,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    ManimColor,
    Mobject,
    Rectangle,
    Scene,
    Square,
    ValueTracker,
    Variable,
    VGroup,
    Write,
    linear,
    rotate_vector,
)
from manim.typing import Vector3D

MEDIUM = 1
SMALL = 0.5


def tex_escape_underscores(s: str) -> str:
    """
    Escape underscores in a string for LaTeX rendering.
    """
    return s.replace("_", "{\\_}")


class MessageType(Enum):
    REQUEST = "request"
    RETRY_REQUEST = "retry_request"
    RESPONSE = "response"
    FAILURE_RESPONSE = "failure_response"

    def color(self) -> ManimColor:
        if self == MessageType.REQUEST:
            return BLUE
        elif self == MessageType.RETRY_REQUEST:
            return YELLOW
        elif self == MessageType.RESPONSE:
            return GREEN
        elif self == MessageType.FAILURE_RESPONSE:
            return RED
        else:
            raise ValueError(f"Unknown message type: {self}")

    def is_request(self) -> bool:
        return self in (MessageType.REQUEST, MessageType.RETRY_REQUEST)

    def is_response(self) -> bool:
        return self in (MessageType.RESPONSE, MessageType.FAILURE_RESPONSE)

    def is_failure(self) -> bool:
        return self == MessageType.FAILURE_RESPONSE


class Message(Dot):
    radius_base = 0.15

    def __init__(
        self, size: float = MEDIUM, type: MessageType = MessageType.REQUEST, **kwargs
    ):
        super().__init__(
            radius=Message.radius_base * size,
            fill_opacity=0.8,
            color=type.color(),
            **kwargs,
        )
        self.id = np.random.randint(0, 1000000)
        self.type = type
        self.attempt = 1

        self.offset = 0
        """Scalar offset perpendicular to the line of flight."""
        self.time_alive = 0.0

        def update_time_alive(mob: Mobject, dt: float):
            self.time_alive += dt

        self.add_updater(update_time_alive)

    def hide(self):
        self.set_opacity(0.0)

    def set_type(self, type: MessageType):
        """
        Set the type of the message and update its color accordingly.
        """
        self.type = type
        self.set_color(type.color())
        self.set_opacity(0.8)
        self._reset_time_alive()
        if type == MessageType.REQUEST:
            self.offset = np.random.uniform(0.2, 1)
            self.attempt = 1
        elif type == MessageType.RETRY_REQUEST:
            self.offset = abs(self.offset)
            self.attempt += 1
        elif type.is_response():
            self.offset = -self.offset

    def _reset_time_alive(self):
        self.time_alive = 0.0


class Queue(Rectangle):
    def __init__(self, orientation: Vector3D = RIGHT, **kwargs):
        self.orientation = orientation
        if np.logical_and(orientation, X_AXIS).any():
            width = 2.0
            height = Message.radius_base * 4
        else:
            width = Message.radius_base * 4
            height = 2.0
        super().__init__(
            width=width, height=height, color=BLUE, fill_opacity=0.2, **kwargs
        )

    def queue_pos_to_point(self, queue_pos: int) -> Vector3D:
        """
        Returns the coordinates of the message at the given queue position.
        The queue position is 1-indexed, meaning that the first message is at position 1.
        """
        if queue_pos < 1:
            raise ValueError("Queue position must be 1 or greater.")

        if np.array_equal(self.orientation, RIGHT):
            reference_point = self.get_right()
            multiplier = -1
            axis = 0
        elif np.array_equal(self.orientation, LEFT):
            reference_point = self.get_left()
            multiplier = 1
            axis = 0
        elif np.array_equal(self.orientation, DOWN):
            reference_point = self.get_bottom()
            multiplier = 1
            axis = 1
        elif np.array_equal(self.orientation, UP):
            reference_point = self.get_top()
            multiplier = -1
            axis = 1
        else:
            raise ValueError(
                "Orientation must be one of the following: RIGHT, LEFT, UP, DOWN."
            )

        position = np.array(self.get_center())
        offset = (
            (queue_pos - 1) * Message.radius_base * 3 + Message.radius_base * 2
        ) * multiplier
        position[axis] = reference_point[axis] + offset

        return position


class Connection(VGroup):
    def __init__(
        self,
        start: Vector3D,
        end: Vector3D,
        server: Processor,
        rtt: float = 2.0,
    ):
        super().__init__()
        self.id = np.random.randint(0, 1000000)
        self.start = start
        self.end = end
        self.server = server
        self.rtt = rtt

        self.line = Line(start, end, color=LIGHTER_GRAY, stroke_opacity=0.8)
        self.reqs = VGroup()
        self.resps = VGroup()
        self.ready_msgs: list[Message] = []

        self.add(self.line)
        self.add(self.reqs)
        self.add(self.resps)

    def send_requests(self, reqs: list[Message], dt: float):
        """
        Send the given requests along this connection.

        Should be called by the client as part of the update loop, regardless of the number of
        requests.
        """
        for req in reqs:
            req.move_to(self.start)
            self.reqs.add(req)

        self._receive_server_responses()
        self._update_messages(self.reqs, is_request=True)
        self._update_messages(self.resps, is_request=False)

    def ready_responses(self) -> list[Message]:
        """
        Returns any responses which are ready to be sent back to the client.
        Should be called by the client as part of the update loop.
        """
        ret = self.ready_msgs
        self.ready_msgs = []
        return ret

    def _receive_server_responses(self):
        new_resps = self.server.send_responses(self)
        self.resps.add(new_resps)

    def _update_messages(self, messages, is_request: bool):
        """Update position of messages."""
        for msg in messages:
            msg = cast(Message, msg)
            proportion = self._proportion_of_flight_time(msg)
            if proportion > 1:
                messages.remove(msg)
                if is_request:
                    self.server.process(msg, self)
                else:
                    msg.hide()
                    self.ready_msgs.append(msg)
            else:
                self._move_msg_along_line(
                    msg, proportion if is_request else 1 - proportion
                )

    def _proportion_of_flight_time(self, msg: Message) -> float:
        return msg.time_alive / (self.rtt / 2)

    def _move_msg_along_line(self, req: Message, proportion: float):
        point = self.line.point_from_proportion(proportion)
        perp = rotate_vector(self.line.get_unit_vector(), 90 * DEGREES)
        offset_vec = perp * Message.radius_base * req.offset
        req.move_to(point + offset_vec)


# TODO: Split into Client and Server classes
class Processor(VGroup):
    def __init__(
        self,
        req_rate: float = 5.0,
        retry_policy: RetryPolicy | None = None,
        failure_rate: float = 0.0,
        size: float = MEDIUM,
        **kwargs,
    ):
        super().__init__()

        square = Square(side_length=1.5 * size, color=BLUE, fill_opacity=0.2, **kwargs)
        self.add(square)

        self.req_rate = req_rate
        self.retry_policy = retry_policy
        self.failure_rate = failure_rate

        self.client_connections: list[Connection] = []
        """Connections for which this processor is a client."""

        self.retries: dict[Connection, list[tuple[float, Message]]] = (
            collections.defaultdict(list)
        )
        """Messages which failed and are waiting to be retried."""

        self.unused_msgs: list[Message] = []
        """Messages which are not currently in use, ready for recycling."""

        self.time = 0.0

        def update(m: Mobject, dt: float):
            self.time += dt
            self._generate_requests(dt)
            self._process_responses()

        self.add_updater(update)

        self.processing: dict[Connection, list[tuple[float, Message]]] = (
            collections.defaultdict(list)
        )

    @staticmethod
    def _num_new_reqs(dt: float, req_rate: float) -> int:
        """
        Returns the number of new requests which would have been created in the given period and
        request rate.
        """
        # Number of requests created per dt
        lam = req_rate * dt

        return np.random.poisson(lam)

    def add_client_connection(self, conn: Connection):
        """
        Add a connection for which this processor is a client.
        """
        self.client_connections.append(conn)

    def _generate_requests(self, dt: float):
        for conn in self.client_connections:
            n = Processor._num_new_reqs(dt, self.req_rate)

            msgs: list[Message] = []
            for _ in range(n):
                if len(self.unused_msgs) > 0:
                    # Recycle message object
                    req = self.unused_msgs.pop()
                    req.set_type(MessageType.REQUEST)
                    msgs.append(req)
                else:
                    # Create a new message object
                    req = Message(size=SMALL)
                    req.set_type(MessageType.REQUEST)
                    msgs.append(req)

            while len(self.retries[conn]) > 0:
                retry_at, msg = self.retries[conn][0]
                if self.time >= retry_at:
                    heapq.heappop(self.retries[conn])
                    msg.set_type(MessageType.RETRY_REQUEST)
                    msgs.append(msg)
                else:
                    break

            conn.send_requests(msgs, dt)

    def _process_responses(self):
        for conn in self.client_connections:
            resps = conn.ready_responses()

            for msg in resps:
                if self._try_schedule_retry(msg, conn):
                    continue

                self._return_message_to_pool(msg)

    def _try_schedule_retry(self, msg: Message, conn: Connection) -> bool:
        """
        Attempt to schedule a retry for the message if it's a failure and we have a retry policy.

        Returns True if the message was scheduled for retry, False otherwise.
        """
        if msg.type.is_failure() is False or self.retry_policy is None:
            return False

        retry_interval = self.retry_policy.get_retry_interval(msg.attempt)
        if retry_interval is None:
            return False

        heapq.heappush(self.retries[conn], (self.time + retry_interval, msg))
        return True

    def _return_message_to_pool(self, msg: Message):
        """
        Return a message to the unused message pool by hiding it and making it available for reuse.
        """
        msg.hide()
        self.unused_msgs.append(msg)

    def process(self, msg: Message, return_to: Connection):
        """
        Process the given message.
        """
        msg.hide()

        finished_at = self.time + Processor._processing_latency()

        conn: list[tuple[float, Message]] = self.processing[return_to]
        heapq.heappush(conn, (finished_at, msg))

    def send_responses(self, conn: Connection) -> list[Message]:
        """
        Remove and return any pending responses for the given connection.
        """
        responses = []

        while len(self.processing[conn]) > 0:
            finished_at, msg = self.processing[conn][0]
            if self.time >= finished_at:
                heapq.heappop(self.processing[conn])
                if np.random.uniform() < self.failure_rate:
                    msg.set_type(MessageType.FAILURE_RESPONSE)
                else:
                    msg.set_type(MessageType.RESPONSE)
                responses.append(msg)
            else:
                break

        return responses

    @staticmethod
    def _processing_latency() -> float:
        """
        Returns the processing latency of the processor, in seconds.
        """
        tasks = 3.0  # Number of tasks to simulate
        rate = 8.0  # Average rate of tasks per second
        return np.random.gamma(tasks, 1 / rate)


class MessageQueueTest(Scene):
    def construct(self):
        queue = Queue().shift(LEFT)
        processor = Processor().shift(RIGHT * 2)
        msg1 = Message().move_to(queue.position(1))
        msg2 = Message().move_to(queue.position(2))

        self.play(
            FadeIn(queue, run_time=0.2),
            FadeIn(msg1, run_time=0.2),
            FadeIn(msg2, run_time=0.2),
        )
        self.play(FadeIn(processor, run_time=0.2))

        self.wait(2)


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


def create_label(
    m: Mobject, property_name: str, direction: Vector3D = RIGHT
) -> Variable:
    """
    Create a label for the given property of the Mobject.
    """
    label = Variable(
        getattr(m, property_name),
        tex_escape_underscores(property_name),
        num_decimal_places=2,
    )
    label.width = 3
    label.next_to(m, direction, buff=0.2)

    def update_label(v: Mobject):
        v = cast(Variable, v)
        v.tracker.set_value(getattr(m, property_name))

    label.add_updater(update_label)

    return label


class ClientServerTest(Scene):
    def construct(self):
        client = Processor(size=SMALL, retry_policy=RetryPolicy()).shift(LEFT * 1.5)
        server = Processor(size=SMALL, failure_rate=0).shift(RIGHT * 1.5)

        self.play(
            FadeIn(client, run_time=0.3),
            FadeIn(server, run_time=0.3),
        )
        self.wait(0.3)

        conn = Connection(client.get_right(), server.get_left(), server)

        self.play(
            Write(conn, run_time=0.3),
        )
        self.wait(0.3)

        client.add_client_connection(conn)

        self.wait(5)

        failure_rate = ValueTracker(0)
        server.add_updater(lambda m: m.set(failure_rate=failure_rate.get_value()))
        label = create_label(server, "failure_rate")
        self.play(Write(label), run_time=1)

        self.wait(1)

        self.play(failure_rate.animate.set_value(0.8), run_time=5.0, rate_func=linear)

        self.wait(5)

        self.play(failure_rate.animate.set_value(0), run_time=5.0, rate_func=linear)

        self.wait(2)

        client.set(req_rate=0.0)

        self.wait(2)

        self.play(FadeOut(client), FadeOut(server), FadeOut(conn), FadeOut(label))
