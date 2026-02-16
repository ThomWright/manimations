"""Reusable Poisson PMF bar chart component."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats
from manim import DOWN, LEFT, Axes, ManimColor, Rectangle, VGroup


def poisson_pmf(max_k: int, lam: float) -> np.ndarray:
    """Compute Poisson PMF (probability mass function) for k=0..max_k."""
    return stats.poisson.pmf(np.arange(max_k + 1), lam)


class PoissonPMFChart(VGroup):
    """A Poisson PMF bar chart with axes."""

    def __init__(
        self,
        max_k: int,
        y_max: float,
        x_length: float,
        y_length: float,
        bar_color: ManimColor,
        dim_opacity: float = 0.4,
        axis_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_k = max_k
        self.bar_color = bar_color
        self.dim_opacity = dim_opacity

        if axis_config is None:
            axis_config = {}

        self.axes = Axes(
            x_range=[0, max_k, 5],
            y_range=[0, y_max, 0.1],
            x_length=x_length,
            y_length=y_length,
            axis_config=axis_config,
            tips=False,
        )

        # Pre-compute scale factors (shift-invariant)
        origin = self.axes.c2p(0, 0)
        self._x_scale = self.axes.c2p(1, 0)[0] - origin[0]
        self._y_scale = self.axes.c2p(0, 1)[1] - origin[1]
        bar_width = self._x_scale * 0.85

        # Pre-allocate bars
        self.bars = VGroup()
        for k in range(max_k):
            bar = Rectangle(
                width=bar_width,
                height=0.001,
                fill_color=bar_color,
                fill_opacity=0,
                stroke_width=0,
            )
            if k == 0:
                bar.stretch_to_fit_width(bar_width / 2)
                bar.move_to(self.axes.c2p(0, 0), aligned_edge=DOWN + LEFT)
            else:
                bar.move_to(self.axes.c2p(k, 0), aligned_edge=DOWN)
            self.bars.add(bar)

        self.add(self.bars)
        # Render axes on top of bars, so the bars don't cover the ticks etc.
        self.add(self.axes)

    def set_lambda(
        self,
        lam: float,
        highlight_range: tuple[float, float] | None = None,
    ):
        """Update bars to show Poisson(lam) PMF.

        If highlight_range is given, bars within the range get full opacity,
        others get dim_opacity. Without a range, all visible bars get full opacity.
        """
        probs = poisson_pmf(self.max_k, lam)

        for k, bar in enumerate(self.bars):
            p = probs[k]
            height = abs(p * self._y_scale)
            bar.stretch_to_fit_height(max(height, 0.001))

            if k == 0:
                bar.move_to(self.axes.c2p(0, 0), aligned_edge=DOWN + LEFT)
            else:
                bar.move_to(self.axes.c2p(k, 0), aligned_edge=DOWN)

            if p < 1e-6:
                bar.set_fill(opacity=0)
            elif highlight_range and highlight_range[0] <= k <= highlight_range[1]:
                bar.set_fill(color=self.bar_color, opacity=1)
            elif highlight_range:
                bar.set_fill(color=self.bar_color, opacity=self.dim_opacity)
            else:
                bar.set_fill(color=self.bar_color, opacity=1)

    def x_to_point(self, x: float) -> np.ndarray:
        """Convert an x value to scene coordinates on this chart's x-axis."""
        origin = self.axes.c2p(0, 0)
        return np.array([
            origin[0] + x * self._x_scale,
            origin[1],
            0,
        ])
