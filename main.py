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

MESSAGE_RADIUS_BASE = 0.15

MEDIUM = 1
SMALL = 0.5


class Message(Dot):
    def __init__(self, size: float = MEDIUM, color: ManimColor = BLUE, **kwargs):
        super().__init__(
            radius=MESSAGE_RADIUS_BASE * size, fill_opacity=0.8, color=color, **kwargs
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
            height = MESSAGE_RADIUS_BASE * 4
        else:
            width = MESSAGE_RADIUS_BASE * 4
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
            (queue_pos - 1) * MESSAGE_RADIUS_BASE * 3 + MESSAGE_RADIUS_BASE * 2
        ) * multiplier
        position[axis] = reference_point[axis] + offset

        return position


class Connection(VGroup):
    def __init__(
        self,
        server: Processor,
        start: Vector3D,
        end: Vector3D,
        req_rate: float = 5.0,
        rtt: float = 2.0,
    ):
        super().__init__()
        self.id = np.random.randint(0, 1000000)
        self.start = start
        self.end = end
        self.req_rate = req_rate
        self.rtt = rtt
        self.server = server

        self.line = Line(start, end, color=LIGHTER_GRAY, stroke_opacity=0.8)
        self.reqs = VGroup()
        self.resps = VGroup()
        self.unused_msgs: list[Message] = []

        self.add(self.line)
        self.add(self.reqs)
        self.add(self.resps)

        def update(m: Mobject, dt: float):
            self.update_reqs(dt)

        self.add_updater(lambda m, dt: update(m, dt))

    @staticmethod
    def num_new_reqs(dt: float, req_rate: float) -> int:
        """
        Returns the number of new requests which would have been created in the given period and
        request rate.
        """
        # Number of requests created per dt
        lam = req_rate * dt

        return np.random.poisson(lam)

    def update_reqs(self, dt: float):
        # Generate new requests
        # TODO: generate these in the client
        n_new_reqs = Connection.num_new_reqs(dt, self.req_rate)
        for _ in range(n_new_reqs):
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

        # Move requests along
        for req in self.reqs:
            req = cast(Message, req)
            proportion = self.proportion_of_flight_time(req)
            if proportion > 1:
                self.reqs.remove(req)
                self.server.process(req, self)
            else:
                self.move_msg_along_line(req, proportion)

        # Take new responses from the server
        new_resps = self.server.take_responses(self)
        for resp in new_resps:
            resp.show_as_response()
        self.resps.add(new_resps)

        # Move responses along
        for resp in self.resps:
            resp = cast(Message, resp)
            proportion = 1 - self.proportion_of_flight_time(resp)
            if proportion < 0:
                resp.hide()
                self.resps.remove(resp)
                self.unused_msgs.append(resp)
            else:
                self.move_msg_along_line(resp, proportion)

    def proportion_of_flight_time(self, msg: Message) -> float:
        return msg.time_alive / (self.rtt / 2)

    def move_msg_along_line(self, req: Message, proportion: float):
        point = self.line.point_from_proportion(proportion)
        perp = rotate_vector(self.line.get_unit_vector(), 90 * DEGREES)
        offset_vec = perp * MESSAGE_RADIUS_BASE * req.offset
        req.move_to(point + offset_vec)


class Processor(VGroup):
    def __init__(self, size: float = MEDIUM, **kwargs):
        super().__init__()
        square = Square(side_length=1.5 * size, color=BLUE, fill_opacity=0.2, **kwargs)
        self.add(square)

        self.time = 0.0

        def update_time(m: Mobject, dt: float):
            self.time += dt

        self.add_updater(update_time)

        self.processing: dict[Connection, list[tuple[float, Message]]] = (
            collections.defaultdict(list)
        )

    def process(self, msg: Message, return_to: Connection):
        """
        Process the given message.
        """
        msg.hide()

        processing_time = Processor.processing_latency()
        finished_at = self.time + processing_time

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
                break

        return responses

    @staticmethod
    def processing_latency() -> float:
        """
        Returns the processing latency of the processor, in seconds.
        """
        tasks = 3.0
        rate = 8.0
        return np.random.gamma(tasks, 1 / rate)


class MessageQueueTest(Scene):
    def construct(self):
        # queue = Queue().shift(LEFT)
        # processor = Processor().shift(RIGHT * 2)
        # msg1 = Message().move_to(queue.position(1))
        # msg2 = Message().move_to(queue.position(2))

        client = Processor(size=SMALL).shift(LEFT * 1.5)
        server = Processor(size=SMALL).shift(RIGHT * 1.5)

        conn = Connection(server, client.get_right(), server.get_left())

        # v_queue = Queue(orientation=DOWN).shift(DOWN * 2)
        # v_msg1 = Message().move_to(v_queue.position(1))
        # v_msg2 = Message().move_to(v_queue.position(2))

        self.play(
            # FadeIn(queue, run_time=0.2),
            # FadeIn(msg1, run_time=0.2),
            # FadeIn(msg2, run_time=0.2),
            FadeIn(client, run_time=0.2),
            FadeIn(server, run_time=0.2),
            FadeIn(conn, run_time=0.2),
            # FadeIn(v_queue, run_time=0.2),
            # FadeIn(v_msg1, run_time=0.2),
            # FadeIn(v_msg2, run_time=0.2),
        )
        # self.play(FadeIn(processor, run_time=0.2))

        self.wait(10)
