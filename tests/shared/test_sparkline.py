import pytest

from shared.components.sparkline import Sparkline
from shared.constants import X_DIM


class TestSparkline:
    def test_sparkline(self):
        dt = 0.06666666666666665
        dissipating_time = 0.5

        sparkline = Sparkline(lambda: 0.5, size=2, dissipating_time=dissipating_time)

        total_time = 0.0
        while total_time < dissipating_time:
            sparkline._update(sparkline, dt)
            total_time += dt

        line_width = sparkline.line.length_over_dim(X_DIM)

        assert line_width == pytest.approx(sparkline.sl_width, rel=0.01), (
            "Sparkline width should be approximately equal to the specified width after dissipating time"
        )
