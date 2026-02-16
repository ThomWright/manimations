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
    LARGE_BUFF,
    UR,
    Brace,
    DashedLine,
    FadeIn,
    Line,
    ManimColor,
    MathTex,
    Mobject,
    ValueTracker,
    Variable,
    VGroup,
    Write,
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
        labels = VGroup(*all_labels).arrange(DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)

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


class PoissonSum(ThemedScene):
    """Demonstrate that pooling reduces relative variability.

    Three independent Poisson(5) distributions sum to Poisson(15).
    Means add linearly, but σ doesn't: 3(μ+σ) > μ_total + σ_total.
    """

    def construct(self):
        theme = get_theme()
        n = 3
        lam_small = 5
        lam_combined = n * lam_small
        sigma_small = math.sqrt(lam_small)
        sigma_combined = math.sqrt(lam_combined)

        mu_plus_sigma_each = lam_small + sigma_small
        naive_sum = n * mu_plus_sigma_each
        actual_mu_plus_sigma = lam_combined + sigma_combined

        # -- Charts --
        small_charts = VGroup()
        for _ in range(n):
            chart = PoissonPMFChart(
                max_k=15,
                y_max=0.20,
                x_length=4,
                y_length=1.6,
                bar_color=theme.primary,
                axis_config={
                    "include_numbers": True,
                    "stroke_width": 2,
                    "font_size": 18,
                },
            )
            chart.set_lambda(
                lam_small,
                highlight_range=(lam_small - sigma_small, lam_small + sigma_small),
            )
            small_charts.add(chart)
        small_charts.arrange(DOWN, buff=0.4)
        small_charts.shift(LEFT * 4)

        big_chart = PoissonPMFChart(
            max_k=30,
            y_max=0.12,
            x_length=7,
            y_length=5.5,
            bar_color=theme.primary,
            axis_config={
                "include_numbers": True,
                "stroke_width": 3,
                "font_size": 22,
            },
        )
        big_chart.set_lambda(
            lam_combined,
            highlight_range=(lam_combined - sigma_combined, lam_combined + sigma_combined),
        )
        big_chart.shift(RIGHT * 3)

        # -- Vertical marker lines --
        def _vline(chart: PoissonPMFChart, x: float, color, dashed: bool = True):
            """Create a vertical line at x on the given chart."""
            bottom = chart.x_to_point(x)
            top = np.array([bottom[0], chart.axes.c2p(0, chart.axes.y_range[1])[1], 0])
            if dashed:
                return DashedLine(bottom, top, color=color, stroke_width=2)
            return Line(bottom, top, color=color, stroke_width=2)

        # Small chart markers
        small_markers = VGroup()
        for chart in small_charts:
            chart = cast(PoissonPMFChart, chart)
            mu_line = _vline(chart, lam_small, theme.neutral)
            sigma_line = _vline(chart, mu_plus_sigma_each, theme.accent)
            small_markers.add(VGroup(mu_line, sigma_line))

        # Big chart markers
        big_mu_line = _vline(big_chart, lam_combined, theme.neutral)
        big_sigma_line = _vline(big_chart, actual_mu_plus_sigma, theme.primary)
        big_naive_line = _vline(big_chart, naive_sum, theme.accent)

        # -- Equations --
        small_eq = MathTex(
            r"\mu + \sigma = " + f"{mu_plus_sigma_each:.1f}",
            font_size=24,
            color=theme.accent,
        ).next_to(small_charts, RIGHT, buff=-LARGE_BUFF)

        big_sigma_eq = MathTex(
            r"\mu + \sigma = " + f"{actual_mu_plus_sigma:.1f}",
            font_size=28,
            color=theme.primary,
        )
        big_sigma_eq.next_to(big_sigma_line, UP, buff=SMALL_BUFF)

        big_naive_eq = MathTex(
            f"3(\\mu + \\sigma) = {naive_sum:.1f}",
            font_size=28,
            color=theme.accent,
        )
        big_naive_eq.next_to(big_naive_line, UP + RIGHT, buff=MED_SMALL_BUFF).shift(
            DOWN * MED_LARGE_BUFF
        )

        # -- Brace between the two σ lines --
        brace_bottom = big_chart.x_to_point(actual_mu_plus_sigma)
        brace_top = big_chart.x_to_point(naive_sum)
        # Vertical brace along x-axis (rendered as horizontal brace rotated)
        brace_line = Line(brace_bottom, brace_top)
        brace = Brace(brace_line, DOWN, buff=MED_LARGE_BUFF)
        gap_value = naive_sum - actual_mu_plus_sigma
        brace_label = MathTex(
            f"\\Delta = {gap_value:.1f}",
            font_size=24,
        ).next_to(brace, DOWN, buff=SMALL_BUFF)

        # -- Animation --
        # Phase 1: Small charts
        self.play(FadeIn(small_charts), run_time=0.8)
        self.wait(0.5)

        # Phase 2: Markers on small charts
        self.play(
            *[FadeIn(m) for m in small_markers],
            Write(small_eq),
            run_time=0.8,
        )
        self.wait(1)

        # Phase 3: Big chart
        self.play(FadeIn(big_chart), run_time=0.8)
        self.wait(0.5)

        # Phase 4: Big chart μ and μ+σ markers
        self.play(
            FadeIn(big_mu_line),
            FadeIn(big_sigma_line),
            Write(big_sigma_eq),
            run_time=0.8,
        )
        self.wait(1)

        # Phase 5: Naive sum marker
        self.play(
            FadeIn(big_naive_line),
            Write(big_naive_eq),
            run_time=0.8,
        )
        self.wait(0.5)

        # Phase 6: Brace showing the gap
        self.play(
            FadeIn(brace),
            Write(brace_label),
            run_time=0.8,
        )
        self.wait(3)
