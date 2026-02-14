from __future__ import annotations

from enum import Enum

import numpy as np
from manim import Dot, ManimColor, Mobject

from shared.constants import MEDIUM
from shared.theme import get_theme


class MessageType(Enum):
    REQUEST = "request"
    RETRY_REQUEST = "retry_request"
    RESPONSE = "response"
    FAILURE_RESPONSE = "failure_response"

    def color(self) -> ManimColor:
        theme = get_theme()
        if self == MessageType.REQUEST:
            return theme.primary
        elif self == MessageType.RETRY_REQUEST:
            return theme.secondary
        elif self == MessageType.RESPONSE:
            return theme.success
        elif self == MessageType.FAILURE_RESPONSE:
            return theme.error
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
        self.failure = False

        self.offset = 0
        """Scalar offset perpendicular to the line of flight."""
        self.time_alive = 0.0

        def update_time_alive(mob: Mobject, dt: float):
            self.time_alive += dt

        self.add_updater(update_time_alive)

    def hide(self):
        self.set_opacity(0.0)

    def unhide(self):
        self.set_opacity(0.8)
        self.time_alive = 0.0

    def reset(self):
        self.attempt = 1
        self.failure = False
        self.time_alive = 0.0
        self.set_type(MessageType.REQUEST)

    def set_as_response(self):
        if self.failure:
            self.set_type(MessageType.FAILURE_RESPONSE)
        else:
            self.set_type(MessageType.RESPONSE)

    def set_type(self, type: MessageType):
        """
        Set the type of the message and update its color accordingly.
        """
        self.type = type
        self.set_color(type.color())
        if type == MessageType.REQUEST:
            self.offset = np.random.uniform(0.2, 1)
        elif type == MessageType.RETRY_REQUEST:
            self.offset = abs(self.offset)
            self.attempt += 1
        elif type.is_response():
            self.offset = -self.offset

    def __lt__(self, other: Message) -> bool:
        return self.id < other.id

    def __le__(self, other: Message) -> bool:
        return self.id <= other.id
