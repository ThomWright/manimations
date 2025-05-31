from __future__ import annotations

import collections
import heapq
import statistics
from enum import Enum
from typing import Callable, TypeVar, cast

import numpy as np
from manim import (
    BLUE,
    DEGREES,
    DOWN,
    GREEN,
    LEFT,
    LIGHTER_GRAY,
    ORANGE,
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
    VMobject,
    Write,
    linear,
    rotate_vector,
)
from manim.typing import Vector3D

MEDIUM = 1
""" Scaling factor for medium-sized objects. """
SMALL = 0.5
""" Scaling factor for small-sized objects. """

X_DIM = 0
""" Index into a vector for the x-coordinate. """
Y_DIM = 1
""" Index into a vector for the y-coordinate. """

# Copied from Manim internals.
STROKE_WIDTH_CONVERSION = 0.01
"""Conversion factor for stroke width to Manim units."""


def stroke_width_buffer(mob: VMobject, overlap=False) -> float:
    """
    Returns the buffer to use when positioning objects relative to the stroke width of the given
    Mobject.
    """
    return STROKE_WIDTH_CONVERSION * mob.stroke_width - (
        STROKE_WIDTH_CONVERSION if overlap else 0
    )


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

    def concurrency(self) -> int:
        """
        Returns the number of concurrent requests being processed by this processor.
        """
        return sum(len(v) for v in self.processing.values())

    def concurrency_by_type(self) -> dict[MessageType, int]:
        """
        Returns the number of concurrent requests being processed by this processor, grouped by
        message type.
        """
        concurrency = collections.defaultdict(int)
        for conn in self.processing.values():
            for _, msg in conn:
                concurrency[msg.type] += 1
        return dict(concurrency)

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


M = TypeVar("M", bound=Mobject)


def create_label(
    m: M,
    f: Callable[[M], int | float],
    name: str,
    direction: Vector3D = RIGHT,
    buff: float = 0.2,
) -> Variable:
    """
    Create a label for the given property of the Mobject.
    """
    label = Variable(
        f(m),
        tex_escape_underscores(name),
        num_decimal_places=2,
    )

    for sm in label.submobjects:
        sm.set(font_size=24)
    label.arrange_submobjects()

    label.next_to(m, direction, buff=buff)

    def update_label(v: Mobject):
        v = cast(Variable, v)
        v.tracker.set_value(f(m))

    label.add_updater(update_label)

    return label


class StackedBar(VGroup):
    """
    A stacked bar chart that can be used to visualize multiple values.
    """

    # Scaling the bars doesn't work if the size gets set to 0.
    min_bar_size = 0.01

    def __init__(
        self,
        max_value: float,
        values: list[float],
        colors: list[ManimColor] = [BLUE, YELLOW, ORANGE, RED],
        bar_length: float = 1.5,
        bar_width: float = 0.3,
        direction: Vector3D = UP,
        base_z_index: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.values = values
        self.colors = colors
        self.max_value = max_value
        self.bar_length = max(bar_length, StackedBar.min_bar_size)
        self.bar_width = max(bar_width, StackedBar.min_bar_size)
        self.direction = direction
        self.is_vertical = True if direction is UP or direction is DOWN else False
        self.base_z_index = base_z_index
        self.bars: list[Rectangle] = []

        self._create_bars()
        self._arrange_bars()

    def set_values(self, values: list[float]):
        """
        Set new values for the stacked bar and update the bars accordingly.
        """
        if len(values) != len(self.bars):
            raise ValueError(
                "Number of values must match the number of bars in the stacked bar."
            )
        self.values = values
        self._arrange_bars()

    def _create_bars(self):
        for i, (value, color) in enumerate(zip(self.values, self.colors)):
            # Scale the actual value to the length
            bar_length = (value / self.max_value) * self.bar_length

            # Set size based on orientation
            if self.is_vertical:
                width = max(self.bar_width, StackedBar.min_bar_size)
                height = max(bar_length, StackedBar.min_bar_size)
            else:
                width = max(bar_length, StackedBar.min_bar_size)
                height = max(self.bar_width, StackedBar.min_bar_size)

            # Create bar
            bar = Rectangle(
                width=width,
                height=height,
                fill_color=color,
                fill_opacity=1,
                stroke_color=color,
                # Draw from front to back
                z_index=self.base_z_index + len(self.values) - i,
            )

            self.bars.append(bar)
            self.add(bar)

    def _arrange_bars(self):
        """
        Resize and arrange the bars, stacking them along the specified direction.
        """
        prev: Rectangle | None = None
        for i, value in enumerate(self.values):
            bar = self.bars[i]

            # Scale the actual value to the length
            bar_length = max(
                (value / self.max_value) * self.bar_length, StackedBar.min_bar_size
            )

            if self.is_vertical:
                bar.stretch_to_fit_height(bar_length, about_edge=-self.direction)
            else:
                bar.stretch_to_fit_width(bar_length, about_edge=-self.direction)

            if prev is not None:
                # Position the bar relative to the previous one
                bar.next_to(
                    prev,
                    self.direction,
                    buff=stroke_width_buffer(prev, overlap=True),
                )

            prev = bar


class MovingAverageTracker(ValueTracker):
    def __init__(self, window_size: int = 10, **kwargs):
        super().__init__(0, **kwargs)
        self.window_size = window_size
        self.values: collections.deque[float] = collections.deque(maxlen=window_size)

    def add_value(self, value: float):
        self.values.append(value)
        self.set_value(statistics.fmean(self.values))


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

        avg_req_concurrency = MovingAverageTracker(window_size=30)
        avg_retry_concurrency = MovingAverageTracker(window_size=30)
        server.add_updater(
            lambda m: avg_req_concurrency.add_value(
                m.concurrency_by_type().get(MessageType.REQUEST, 0)
            )
        )
        server.add_updater(
            lambda m: avg_retry_concurrency.add_value(
                m.concurrency_by_type().get(MessageType.RETRY_REQUEST, 0)
            )
        )
        concurrency_bar = StackedBar(
            max_value=5,
            values=[
                avg_req_concurrency.get_value(),
                avg_retry_concurrency.get_value(),
            ],
            colors=[MessageType.REQUEST.color(), MessageType.RETRY_REQUEST.color()],
            bar_length=server.height,
            bar_width=server.height / 10,
        ).next_to(
            server,
            RIGHT,
            buff=stroke_width_buffer(server),
            aligned_edge=DOWN,
        )

        concurrency_bar.add_updater(
            lambda m: m.set_values(
                [
                    avg_req_concurrency.get_value(),
                    avg_retry_concurrency.get_value(),
                ]
            )
        )
        concurrency_label = create_label(
            server,
            lambda m: avg_req_concurrency.get_value()
            + avg_retry_concurrency.get_value(),
            "concurrency",
            direction=RIGHT,
            buff=0.2 + concurrency_bar.bar_width,
        ).shift(UP * 0.2)
        self.play(
            FadeIn(concurrency_bar),
            Write(concurrency_label),
            run_time=0.6,
        )
        self.wait(0.3)

        client.add_client_connection(conn)
        self.wait(5)

        failure_label = create_label(
            server, lambda m: getattr(m, "failure_rate"), "failure_rate"
        )
        failure_label.next_to(concurrency_label, DOWN, buff=0.2)
        self.play(Write(failure_label))
        self.wait(1)

        failure_rate = ValueTracker(0)
        server.add_updater(lambda m: m.set(failure_rate=failure_rate.get_value()))

        self.play(failure_rate.animate.set_value(0.8), run_time=5.0, rate_func=linear)
        self.wait(5)

        self.play(failure_rate.animate.set_value(0), run_time=5.0, rate_func=linear)
        self.wait(2)

        client.set(req_rate=0.0)
        self.wait(2)

        self.play(
            FadeOut(client),
            FadeOut(server),
            FadeOut(conn),
            FadeOut(concurrency_label),
            FadeOut(failure_label),
            FadeOut(concurrency_bar),
        )
