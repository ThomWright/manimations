import pytest
from manim import DOWN, LEFT, RIGHT, UP

from components.processor import Processor, RetryPolicy
from components.queue import Queue
from components.sparkline import Sparkline
from constants import X_DIM
from components.label import tex_escape_underscores


class TestQueue:
    def test_queue_pos_to_point(self):
        # Test right-oriented queue
        q_right = Queue(orientation=RIGHT)
        pos1 = q_right.queue_pos_to_point(1)
        pos2 = q_right.queue_pos_to_point(2)
        assert pos1[0] > pos2[0], "x-coordinate of pos1 should be greater than pos2"

        # Test left-oriented queue
        q_left = Queue(orientation=LEFT)
        pos1 = q_left.queue_pos_to_point(1)
        pos2 = q_left.queue_pos_to_point(2)
        assert pos1[0] < pos2[0], "x-coordinate of pos1 should be less than pos2"

        # Test up-oriented queue
        q_up = Queue(orientation=UP)
        pos1 = q_up.queue_pos_to_point(1)
        pos2 = q_up.queue_pos_to_point(2)
        assert pos1[1] > pos2[1], "y-coordinate of pos1 should be greater than pos2"

        # Test down-oriented queue
        q_down = Queue(orientation=DOWN)
        pos1 = q_down.queue_pos_to_point(1)
        pos2 = q_down.queue_pos_to_point(2)
        assert pos1[1] < pos2[1], "y-coordinate of pos1 should be less than pos2"

    def test_queue_pos_invalid(self):
        q = Queue()
        with pytest.raises(ValueError):
            q.queue_pos_to_point(0)
        with pytest.raises(ValueError):
            q.queue_pos_to_point(-1)


class TestClient:
    # TODO: Make this test less flaky
    def test_num_new_reqs(self):
        req_rate = 2.0
        dt = 0.06666666666666665

        reqs = 0
        time = 0.0
        while time < 10.0:
            reqs += Processor._num_new_reqs(dt, req_rate)
            time += dt

        # Check that we get approximately 20 requests (20 = 2.0 requests/sec * 10 seconds)
        assert reqs == pytest.approx(20, rel=0.5), (
            "Expected ~20 requests to be created in 10 seconds"
        )


class TestRetryPolicy:
    def test_retry_policy(self):
        retry_policy = RetryPolicy(3, 0.1, 0)

        assert retry_policy.get_retry_interval(1) == 0.1, (
            "First retry interval should be 0.1"
        )
        assert retry_policy.get_retry_interval(2) == 0.2, (
            "Second retry interval should be 0.2 (doubled)"
        )
        assert retry_policy.get_retry_interval(3) == 0.4, (
            "Third retry interval should be 0.4 (doubled again)"
        )
        assert retry_policy.get_retry_interval(4) is None, (
            "Fourth retry should return None (no more retries allowed)"
        )
        with pytest.raises(
            ValueError, match="Retry count must be 1 or greater, got 0."
        ):
            retry_policy.get_retry_interval(0)


class TestTexEscapeUnderscores:
    def test_tex_escape_underscores(self):
        assert tex_escape_underscores("hello_world") == "hello{\\_}world", (
            "Should escape underscore in TeX"
        )
        assert (
            tex_escape_underscores("this_is_a_test") == "this{\\_}is{\\_}a{\\_}test"
        ), "Should escape multiple underscores in TeX"
        assert tex_escape_underscores("") == "", (
            "Should return empty string for empty input"
        )


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
