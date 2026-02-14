from __future__ import annotations

from manim import (
    DOWN,
    UP,
    ManimColor,
    Rectangle,
    VGroup,
    VMobject,
)
from manim.typing import Vector3D

from shared.constants import STROKE_WIDTH_CONVERSION
from shared.theme import get_theme


def stroke_width_buffer(mob1: VMobject, mob2: VMobject) -> float:
    """
    Returns the buffer to use when positioning objects relative to the stroke width of the given
    Mobjects.
    """
    return (
        STROKE_WIDTH_CONVERSION * mob1.stroke_width
        + STROKE_WIDTH_CONVERSION * mob2.stroke_width
    ) / 2


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
        colors: list[ManimColor] | None = None,
        bar_length: float = 1.5,
        bar_width: float = 0.3,
        direction: Vector3D = UP,
        base_z_index: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.values = values
        if colors is None:
            theme = get_theme()
            colors = [theme.primary, theme.secondary, theme.accent, theme.error]
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
                fill_opacity=0 if value == 0 else 1,
                stroke_color=color,
                stroke_opacity=0 if value == 0 else 1,
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

            if value == 0:
                bar.set_opacity(0)
                bar.set_stroke(opacity=0)
            elif bar.get_stroke_opacity() == 0:
                bar.set_opacity(1)
                bar.set_stroke(opacity=1)

            if prev is not None:
                # Position the bar relative to the previous one
                bar.next_to(prev, self.direction, buff=0)
            if value != 0:
                prev = bar
