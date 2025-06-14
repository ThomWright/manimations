from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from manim import DEGREES, LIGHTER_GRAY, Line, VGroup, rotate_vector
from manim.typing import Vector3D

from components.message import Message

if TYPE_CHECKING:
    from components.processor import Processor


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
