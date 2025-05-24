import pytest
from manim import UP, DOWN, LEFT, RIGHT
from main import Queue, Processor, RetryPolicy


class TestQueue:
    def test_queue_pos_to_point(self):
        # Test right-oriented queue
        q_right = Queue(orientation=RIGHT)
        pos1 = q_right.queue_pos_to_point(1)
        pos2 = q_right.queue_pos_to_point(2)
        assert pos1[0] < pos2[0], "x-coordinate of pos1 should be less than pos2"

        # Test left-oriented queue
        q_left = Queue(orientation=LEFT)
        pos1 = q_left.queue_pos_to_point(1)
        pos2 = q_left.queue_pos_to_point(2)
        assert pos1[0] > pos2[0], "x-coordinate of pos1 should be greater than pos2"

        # Test up-oriented queue
        q_up = Queue(orientation=UP)
        pos1 = q_up.queue_pos_to_point(1)
        pos2 = q_up.queue_pos_to_point(2)
        assert pos1[1] < pos2[1], "y-coordinate of pos1 should be less than pos2"

        # Test down-oriented queue
        q_down = Queue(orientation=DOWN)
        pos1 = q_down.queue_pos_to_point(1)
        pos2 = q_down.queue_pos_to_point(2)
        assert pos1[1] > pos2[1], "y-coordinate of pos1 should be greater than pos2"

    def test_queue_pos_invalid(self):
        q = Queue()
        with pytest.raises(ValueError):
            q.queue_pos_to_point(0)
        with pytest.raises(ValueError):
            q.queue_pos_to_point(-1)


class TestClient:
    def test_num_new_reqs(self):
        req_rate = 2.0
        dt = 0.06666666666666665

        reqs = 0
        time = 0.0
        while time < 10.0:
            reqs += Processor.num_new_reqs(dt, req_rate)
            time += dt

        # Check that we get approximately 20 requests (20 = 2.0 requests/sec * 10 seconds)
        assert reqs == pytest.approx(20, rel=0.25), (
            "Expected ~20 requests to be created in 10 seconds"
        )

class TestRetryPolicy:
    def test_retry_policy(self):
        retry_policy = RetryPolicy(3, 0.1, 0)

        assert retry_policy.get_retry_interval(1) == 0.1, "First retry interval should be 0.1"
        assert retry_policy.get_retry_interval(2) == 0.2, "Second retry interval should be 0.2 (doubled)"
        assert retry_policy.get_retry_interval(3) == 0.4, "Third retry interval should be 0.4 (doubled again)"
        with pytest.raises(ValueError, match="Retry count exceeds maximum retries"):
            retry_policy.get_retry_interval(4)
