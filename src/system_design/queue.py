from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, X_AXIS, Rectangle

from shared.theme import get_theme
from manim.typing import Vector3D

from system_design.message import Message


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
            width=width, height=height, color=get_theme().primary, fill_opacity=0.2, **kwargs
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
