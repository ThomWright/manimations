from collections import deque


class MovingSum:
    def __init__(self, window_duration: float = 1.0):
        self.window_duration = window_duration
        self.values: deque[tuple[float, float]] = deque()
        self.total_time = 0.0

    def add_value(self, value: float, dt: float):
        self.total_time += dt
        self.values.append((value, self.total_time))
        self._remove_old_values()

    def _remove_old_values(self):
        while (
            self.values and self.values[0][1] < self.total_time - self.window_duration
        ):
            self.values.popleft()

    def get_value(self) -> float:
        if not self.values:
            return 0.0
        return sum(value for value, _ in self.values)
