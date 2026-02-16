from __future__ import annotations

from typing import cast

import math

import numpy as np
from manim import (
    UP,
    DOWN,
    MED_SMALL_BUFF,
    LEFT,
    RIGHT,
    SMALL_BUFF,
    MED_LARGE_BUFF,
    UR,
    Line,
    ManimColor,
    Mobject,
    ValueTracker,
    Variable,
    VGroup,
    always_redraw,
    linear,
)

from shared.components.pmf_chart import PoissonPMFChart
from shared.theme import get_theme
from shared.themed_scene import ThemedScene


MAX_K = 50
Y_MAX = 0.40
STROKE_WIDTH = 4


class PoissonVariance(ThemedScene):
    def construct(self):
        theme = get_theme()
        STD_DEV_COLOR = theme.primary
        VARIANCE_COLOR = theme.accent

        lam_tracker = ValueTracker(1.0)

        chart = (
            PoissonPMFChart(
                max_k=MAX_K,
                y_max=Y_MAX,
                x_length=12.5,
                y_length=6,
                bar_color=STD_DEV_COLOR,
                axis_config={
                    "include_numbers": True,
                    "font_size": 28,
                    "stroke_width": STROKE_WIDTH,
                },
            )
            .shift(UP * MED_LARGE_BUFF)
            .shift(RIGHT * MED_SMALL_BUFF)
        )

        def update_chart(mob: Mobject):
            mob = cast(PoissonPMFChart, mob)
            lam = lam_tracker.get_value()
            sigma = math.sqrt(lam)
            mob.set_lambda(lam, highlight_range=(lam - sigma, lam + sigma))

        chart.add_updater(update_chart)

        # Labels
        lam_label = Variable(1.0, r"\lambda", num_decimal_places=1)
        var_label = Variable(1.0, r"\sigma^2", num_decimal_places=1)
        var_label.set_color(VARIANCE_COLOR)
        std_label = Variable(1.0, r"\sigma", num_decimal_places=2)
        std_label.set_color(STD_DEV_COLOR)
        cv_label = Variable(1.0, r"\sigma / \mu", num_decimal_places=2)

        all_labels = [lam_label, var_label, std_label, cv_label]
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
        indicators_shift = (
            chart.axes.x_axis.get_bottom()[1] - chart.x_to_point(0)[1] - MED_SMALL_BUFF
        )
        spread_indicators = always_redraw(
            lambda: _spread_indicators(
                chart, lam_tracker, STD_DEV_COLOR, VARIANCE_COLOR
            ).shift(DOWN * abs(indicators_shift))
        )

        # Animate
        self.add(chart, labels, spread_indicators)
        self.wait(1)
        self.play(
            lam_tracker.animate.set_value(25.0),
            run_time=12,
            rate_func=linear,
        )
        self.wait(2)


CAP_HEIGHT = 0.2


def _bar_indicator(left_x: float, right_x: float, y: float, color) -> VGroup:
    """Draw a |----| indicator between two x positions."""
    h_line = Line(
        np.array([left_x, y, 0]),
        np.array([right_x, y, 0]),
        color=color,
        stroke_width=STROKE_WIDTH,
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
    chart: PoissonPMFChart,
    lam_tracker: ValueTracker,
    std_dev_color: ManimColor,
    variance_color: ManimColor,
) -> VGroup:
    """Draw |----| indicators for ±σ and ±σ² ranges."""
    lam = lam_tracker.get_value()
    sigma = math.sqrt(lam)
    variance = lam

    y = chart.x_to_point(0)[1]

    var_indicator = _bar_indicator(
        chart.x_to_point(lam - variance)[0],
        chart.x_to_point(lam + variance)[0],
        y,
        variance_color,
    ).shift(DOWN * SMALL_BUFF)
    sigma_indicator = (
        _bar_indicator(
            chart.x_to_point(lam - sigma)[0],
            chart.x_to_point(lam + sigma)[0],
            y,
            std_dev_color,
        )
        .shift(DOWN * SMALL_BUFF * 2)
        .shift(DOWN * MED_SMALL_BUFF)
    )

    return VGroup(var_indicator, sigma_indicator)
