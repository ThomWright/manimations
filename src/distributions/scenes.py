from __future__ import annotations

from typing import cast

import math

import numpy as np
import scipy.stats as stats
from manim import (
    UP,
    DOWN,
    MED_SMALL_BUFF,
    LEFT,
    RIGHT,
    SMALL_BUFF,
    MED_LARGE_BUFF,
    UR,
    Axes,
    Line,
    ManimColor,
    Mobject,
    Rectangle,
    ValueTracker,
    Variable,
    VGroup,
    always_redraw,
    linear,
)

from shared.theme import get_theme
from shared.themed_scene import ThemedScene

CAP_HEIGHT = 0.15


def _poisson_pmf(max_k: int, lam: float) -> np.ndarray:
    """Compute Poisson PMF (probability mass function) for k=0..max_k-1."""
    return stats.poisson.pmf(np.arange(max_k), lam)


MAX_K = 50
Y_MAX = 0.40
DIM_OPACITY = 0.4
STROKE_WIDTH = 4


class PoissonVariance(ThemedScene):
    def construct(self):
        theme = get_theme()
        STD_DEV_COLOR = theme.primary
        VARIANCE_COLOR = theme.accent

        lam_tracker = ValueTracker(1.0)

        # Axes
        axes = (
            Axes(
                x_range=[0, MAX_K, 5],
                y_range=[0, Y_MAX, 0.1],
                x_length=12.5,
                y_length=6,
                axis_config={
                    "include_numbers": True,
                    "font_size": 28,
                    "stroke_width": STROKE_WIDTH,
                },
                tips=False,
            )
            .shift(UP * MED_LARGE_BUFF)
            .shift(RIGHT * MED_SMALL_BUFF)
        )

        # Pre-compute axis coordinate transform (linear mapping)
        origin = axes.c2p(0, 0)
        x_scale = axes.c2p(1, 0)[0] - origin[0]
        y_scale = axes.c2p(0, 1)[1] - origin[1]
        bar_width = x_scale * 0.85

        # Pre-allocate bars
        bars = VGroup()
        for k in range(MAX_K):
            bar = Rectangle(
                width=bar_width,
                height=0.001,
                fill_color=STD_DEV_COLOR,
                fill_opacity=0,
                stroke_width=0,
            )
            x_pos = origin[0] + (k + 0.5) * x_scale
            bar.move_to(np.array([x_pos, origin[1], 0]), aligned_edge=DOWN)
            bars.add(bar)

        def update_bars(bars: Mobject) -> None:
            bars = cast(VGroup, bars)
            lam = lam_tracker.get_value()
            sigma = math.sqrt(lam)
            low = lam - sigma
            high = lam + sigma

            probs = _poisson_pmf(MAX_K, lam)

            for k, bar in enumerate(bars):
                p = probs[k]
                height = abs(p * y_scale)
                bar.stretch_to_fit_height(max(height, 0.001))
                x_pos = origin[0] + (k + 0.5) * x_scale
                bar.move_to(np.array([x_pos, origin[1], 0]), aligned_edge=DOWN)

                if p < 1e-6:
                    bar.set_fill(opacity=0)
                elif low <= k <= high:
                    bar.set_fill(color=STD_DEV_COLOR, opacity=1)
                else:
                    bar.set_fill(color=STD_DEV_COLOR, opacity=DIM_OPACITY)

        bars.add_updater(update_bars)

        # Labels
        lam_label = Variable(1.0, r"\lambda", num_decimal_places=1)
        var_label = Variable(1.0, r"\sigma^2", num_decimal_places=1)
        var_label.set_color(VARIANCE_COLOR)
        std_label = Variable(1.0, r"\sigma", num_decimal_places=2)
        std_label.set_color(STD_DEV_COLOR)
        cv_label = Variable(1.0, r"\sigma / \mu", num_decimal_places=2)

        all_labels = [lam_label, var_label, std_label, cv_label]
        for label in all_labels:
            label.scale(0.8)
        labels = VGroup(*all_labels).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        # Align the = signs by lining up value left edges
        max_value_left = max(label.value.get_left()[0] for label in all_labels)
        for label in all_labels:
            label.shift((max_value_left - label.value.get_left()[0]) * RIGHT)

        labels.to_corner(UR, buff=0.6)

        lam_label.add_updater(
            lambda v: cast(Variable, v).tracker.set_value(lam_tracker.get_value())
        )
        var_label.add_updater(
            lambda v: cast(Variable, v).tracker.set_value(lam_tracker.get_value())
        )
        std_label.add_updater(
            lambda v: cast(Variable, v).tracker.set_value(
                math.sqrt(lam_tracker.get_value())
            )
        )
        cv_label.add_updater(
            lambda v: cast(Variable, v).tracker.set_value(
                1.0 / math.sqrt(lam_tracker.get_value())
            )
        )

        # Spread indicators below x-axis
        indicators_shift = axes.x_axis.get_bottom()[1] - origin[1] - MED_SMALL_BUFF
        spread_indicators = always_redraw(
            lambda: _spread_indicators(
                origin, x_scale, lam_tracker, STD_DEV_COLOR, VARIANCE_COLOR
            ).shift(DOWN * abs(indicators_shift))
        )

        # Animate
        self.add(axes, bars, labels, spread_indicators)
        self.wait(1)
        self.play(
            lam_tracker.animate.set_value(25.0),
            run_time=12,
            rate_func=linear,
        )
        self.wait(2)


def _bar_indicator(left_x: float, right_x: float, y: float, color) -> VGroup:
    """Draw a |----| indicator between two x positions."""
    h_line = Line(
        np.array([left_x, y, 0]), np.array([right_x, y, 0]), color=color, stroke_width=STROKE_WIDTH
    )
    l_cap = Line(
        np.array([left_x, y - CAP_HEIGHT / 2, 0]),
        np.array([left_x, y + CAP_HEIGHT / 2, 0]),
        color=color,
        stroke_width=STROKE_WIDTH,
    )
    r_cap = Line(
        np.array([right_x, y - CAP_HEIGHT / 2, 0]),
        np.array([right_x, y + CAP_HEIGHT / 2, 0]),
        color=color,
        stroke_width=STROKE_WIDTH,
    )
    return VGroup(l_cap, h_line, r_cap)


def _spread_indicators(
    origin: np.ndarray,
    x_scale: float,
    lam_tracker: ValueTracker,
    std_dev_color: ManimColor,
    variance_color: ManimColor,
) -> VGroup:
    """Draw |----| indicators for ±σ and ±σ² ranges."""
    lam = lam_tracker.get_value()
    sigma = math.sqrt(lam)
    variance = lam

    y = origin[1]

    var_indicator = _bar_indicator(
        origin[0] + (lam - variance) * x_scale,
        origin[0] + (lam + variance) * x_scale,
        y,
        variance_color,
    ).shift(DOWN * SMALL_BUFF)
    sigma_indicator = (
        _bar_indicator(
            origin[0] + (lam - sigma) * x_scale,
            origin[0] + (lam + sigma) * x_scale,
            y,
            std_dev_color,
        )
        .shift(DOWN * SMALL_BUFF)
        .shift(DOWN * MED_SMALL_BUFF)
    )

    return VGroup(var_indicator, sigma_indicator)
