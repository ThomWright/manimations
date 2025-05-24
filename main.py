from __future__ import annotations
from typing import cast
from manim import (
    Dot,
    Rectangle,
    Square,
    Scene,
    FadeIn,
    VGroup,
    Line,
    ManimColor,
    Mobject,
    DEGREES,
    BLUE,
    GREEN,
    LIGHTER_GRAY,
    LEFT,
    RIGHT,
    DOWN,
    UP,
    X_AXIS,
    rotate_vector,
)
import numpy as np
from manim.typing import Vector3D
import heapq
import collections


MEDIUM = 1
SMALL = 0.5


class Message(Dot):
    radius_base = 0.15

    def __init__(self, size: float = MEDIUM, color: ManimColor = BLUE, **kwargs):
        super().__init__(
            radius=Message.radius_base * size, fill_opacity=0.8, color=color, **kwargs
        )
        self.id = np.random.randint(0, 1000000)
        self.offset = 0
        """Scalar offset perpendicular to the line of flight."""
        self.time_alive = 0.0

        def update_time_alive(mob: Mobject, dt: float):
            self.time_alive += dt

        self.add_updater(update_time_alive)

    def hide(self):
        self.set_opacity(0.0)

    def show_as_request(self):
        self.reset_time_alive()
        self.set_opacity(0.8)
        self.set_color(BLUE)
        self.offset = np.random.uniform(0.2, 1)

    def show_as_response(self):
        self.reset_time_alive()
        self.set_opacity(0.8)
        self.set_color(GREEN)
        self.offset = np.random.uniform(-1, -0.2)

    def reset_time_alive(self):
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
        self.unused_msgs: list[Message] = []

        self.add(self.line)
        self.add(self.reqs)
        self.add(self.resps)

    def send_requests(self, num_reqs: int, dt: float):
        """
        Send the given number of requests along this connection.

        Should be called by the client as part of the update loop. Should be called even when
        `num_reqs` is 0, to ensure that in flight messages are updated.
        """
        for _ in range(num_reqs):
            if len(self.unused_msgs) > 0:
                # Recycle message object
                req = self.unused_msgs.pop()
                req.show_as_request()
                req.move_to(self.start)
                self.reqs.add(req)
            else:
                # Create a new message object
                req = Message(size=SMALL).move_to(self.start)
                req.show_as_request()
                self.reqs.add(req)

        self.receive_responses()
        self.update_messages(self.reqs, is_request=True)
        self.update_messages(self.resps, is_request=False)

    def receive_responses(self):
        new_resps = self.server.take_responses(self)
        for resp in new_resps:
            resp.show_as_response()
        self.resps.add(new_resps)

    def update_messages(self, messages, is_request: bool):
        """Update position of messages."""
        for msg in messages:
            msg = cast(Message, msg)
            proportion = self.proportion_of_flight_time(msg)
            if proportion > 1:
                messages.remove(msg)
                if is_request:
                    self.server.process(msg, self)
                else:
                    msg.hide()
                    self.unused_msgs.append(msg)
            else:
                self.move_msg_along_line(
                    msg, proportion if is_request else 1 - proportion
                )

    def proportion_of_flight_time(self, msg: Message) -> float:
        return msg.time_alive / (self.rtt / 2)

    def move_msg_along_line(self, req: Message, proportion: float):
        point = self.line.point_from_proportion(proportion)
        perp = rotate_vector(self.line.get_unit_vector(), 90 * DEGREES)
        offset_vec = perp * Message.radius_base * req.offset
        req.move_to(point + offset_vec)


class Processor(VGroup):
    def __init__(
        self,
        req_rate: float = 5.0,
        size: float = MEDIUM,
        **kwargs,
    ):
        super().__init__()
        square = Square(side_length=1.5 * size, color=BLUE, fill_opacity=0.2, **kwargs)
        self.add(square)

        self.req_rate = req_rate

        self.client_connections: list[Connection] = []
        """Connections for which this processor is a client."""

        self.time = 0.0

        def update(m: Mobject, dt: float):
            self.time += dt
            self.generate_requests(dt)

        self.add_updater(update)

        self.processing: dict[Connection, list[tuple[float, Message]]] = (
            collections.defaultdict(list)
        )

    @staticmethod
    def num_new_reqs(dt: float, req_rate: float) -> int:
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

    def generate_requests(self, dt: float):
        for conn in self.client_connections:
            n = Processor.num_new_reqs(dt, self.req_rate)
            conn.send_requests(n, dt)

    def process(self, msg: Message, return_to: Connection):
        """
        Process the given message.
        """
        msg.hide()

        finished_at = self.time + Processor.processing_latency()

        conn: list[tuple[float, Message]] = self.processing[return_to]
        heapq.heappush(conn, (finished_at, msg))

    def take_responses(self, conn: Connection) -> list[Message]:
        """
        Remove and return any pending responses for the given connection.
        """
        responses = []

        msgs: list[tuple[float, Message]] = self.processing[conn]

        while True:
            if len(msgs) == 0:
                return responses

            finished_at, msg = msgs[0]
            if self.time > finished_at:
                heapq.heappop(msgs)
                responses.append(msg)
            else:
                return responses

    @staticmethod
    def processing_latency() -> float:
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


class ClientServerTest(Scene):
    def construct(self):
        client = Processor(size=SMALL).shift(LEFT * 1.5)
        server = Processor(size=SMALL).shift(RIGHT * 1.5)

        conn = Connection(client.get_right(), server.get_left(), server)

        client.add_client_connection(conn)

        self.play(
            FadeIn(client, run_time=0.2),
            FadeIn(server, run_time=0.2),
            FadeIn(conn, run_time=0.2),
        )

        self.wait(10)
