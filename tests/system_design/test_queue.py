import pytest
from manim import DOWN, LEFT, RIGHT, UP

from system_design.queue import Queue


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
