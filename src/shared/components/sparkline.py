from __future__ import annotations

import math
from typing import Callable

import numpy as np
from manim import (
    BLUE,
    LEFT,
    RIGHT,
    Dot,
    ManimColor,
    Mobject,
    Rectangle,
    TracedPath,
    VMobject,
)
from manim.typing import Vector3D

from shared.constants import MEDIUM, X_DIM, Y_DIM


class Sparkline(VMobject):
    """A sparkline that traces a function over time."""

    def __init__(
        self,
        get_value: Callable[[], float],
        size: float = MEDIUM,
        start_y_bounds: tuple[float, float] = (-1, 1),
        dissipating_time: float = 1,
        stroke_color: ManimColor = BLUE,
        stroke_width: float = 2.0,
        dot_radius: float = 0.05,
        **kwargs,
    ):
        super().__init__(
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            **kwargs,
        )
        size = size * 0.6
        self.get_value = get_value
        self.sl_width = size * 3.0
        self.sl_height = size
        self.y_bounds = start_y_bounds
        self.dissipating_time = dissipating_time

        self.bounding_rect = Rectangle(
            width=self.sl_width,
            height=self.sl_height,
            stroke_opacity=0,
        )
        self.add(self.bounding_rect)

        self.dot = Dot(
            radius=dot_radius,
            color=stroke_color,
            fill_opacity=1.0,
            stroke_opacity=0.0,
        ).shift(RIGHT * (self.sl_width / 2))
        self.add(self.dot)

        self.update_dot_position()

        self.line = TracedPath(
            self.dot.get_center,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            dissipating_time=dissipating_time,
            stroke_opacity=[1, 0],  # type: ignore
        )
        self.line.remove_updater(self.line.update_path)
        self.line.start_new_path(self.dot.get_center())  # Initialize the path
        self.add(self.line)

    def _update(self, mob: Mobject, dt: float):
        if dt <= 0:
            return

        x_shift = self._x_shift(dt)
        self.line.shift(LEFT * x_shift)

        self.update_dot_position()
        self.line.update_path(self.line, dt)

    def update_dot_position(self):
        value = self.get_value()
        self._adjust_y_bounds(value)

        new_point = self._new_point(value)
        self.dot.move_to(new_point)

    def start(self):
        """
        Start the sparkline, allowing it to update its position and trace the function.
        """
        self.add_updater(self._update)

    def stop(self):
        """
        Stop the sparkline, preventing it from updating its position and tracing the function.
        """
        self.remove_updater(self._update)

    def _x_shift(self, dt: float) -> float:
        """
        Calculate the how much the x axis should be shifted.
        """
        shifts_per_dissipation_time = math.floor(self.dissipating_time / dt)
        return self.sl_width / shifts_per_dissipation_time

    def _adjust_y_bounds(self, value: float):
        """
        Adjust the y bounds based on the new value.
        """
        if value < self.y_bounds[0]:
            self.y_bounds = (value, self.y_bounds[1])
        elif value > self.y_bounds[1]:
            self.y_bounds = (self.y_bounds[0], value)

    def _new_point(self, value: float) -> Vector3D:
        """
        Create a new point for the sparkline based on the current time and value.
        """
        x: float = self.bounding_rect.get_right()[X_DIM]

        # Scale the y value to fit within the height of the sparkline
        y = (
            (value - self.y_bounds[0])
            / (self.y_bounds[1] - self.y_bounds[0])
            * self.sl_height
        ) + self.bounding_rect.get_bottom()[Y_DIM]
        return np.array([x, y, 0])
