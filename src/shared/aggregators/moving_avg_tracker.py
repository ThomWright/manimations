from collections import deque
import statistics

from manim import ValueTracker


class MovingAverageTracker(ValueTracker):
    def __init__(self, window_size: int = 10, **kwargs):
        super().__init__(0, **kwargs)
        self.window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)

    def add_value(self, value: float):
        self.values.append(value)
        self.set_value(statistics.fmean(self.values))
