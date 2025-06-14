from __future__ import annotations

import statistics
from collections import deque

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    Scene,
    ValueTracker,
    Write,
    linear,
)

from components.bar import StackedBar, stroke_width_buffer
from components.connection import Connection
from components.label import create_label
from components.message import Message, MessageType
from components.processor import Processor, RetryPolicy
from components.queue import Queue
from components.sparkline import Sparkline
from constants import QUEUEING_COLOR, SMALL


class MovingAverageTracker(ValueTracker):
    def __init__(self, window_size: int = 10, **kwargs):
        super().__init__(0, **kwargs)
        self.window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)

    def add_value(self, value: float):
        self.values.append(value)
        self.set_value(statistics.fmean(self.values))


class MessageQueueTest(Scene):
    def construct(self):
        queue = Queue().shift(LEFT)
        processor = Processor().shift(RIGHT * 2)
        msg1 = Message().move_to(queue.position(1))
        msg2 = Message().move_to(queue.position(2))

        self.play(
            FadeIn(queue, run_time=0.2),
            FadeIn(msg1, run_time=0.2),
            FadeIn(msg2, run_time=0.2),
        )
        self.play(FadeIn(processor, run_time=0.2))

        self.wait(2)


class ClientServerTest(Scene):
    def construct(self):
        client = Processor(size=SMALL, retry_policy=RetryPolicy()).shift(LEFT * 1.5)
        server = Processor(size=SMALL, failure_rate=0, max_concurrency=5).shift(
            RIGHT * 1.5
        )

        self.play(
            FadeIn(client, run_time=0.3),
            FadeIn(server, run_time=0.3),
        )
        self.wait(0.3)

        conn = Connection(client.get_right(), server.get_left(), server)
        client.add_client_connection(conn)

        self.play(
            Write(conn, run_time=0.3),
        )
        self.wait(0.3)

        avg_req_concurrency = MovingAverageTracker(window_size=30)
        avg_retry_concurrency = MovingAverageTracker(window_size=30)
        avg_queued_concurrency = MovingAverageTracker(window_size=30)

        server.add_updater(
            lambda m: avg_req_concurrency.add_value(
                m.concurrency_by_type(include_queued=False).get(MessageType.REQUEST, 0)
            )
        )
        server.add_updater(
            lambda m: avg_retry_concurrency.add_value(
                m.concurrency_by_type(include_queued=False).get(
                    MessageType.RETRY_REQUEST, 0
                )
            )
        )
        server.add_updater(lambda m: avg_queued_concurrency.add_value(m.queued()))

        concurrency_bar = StackedBar(
            max_value=server.max_concurrency,
            values=[
                avg_req_concurrency.get_value(),
                avg_retry_concurrency.get_value(),
                avg_queued_concurrency.get_value(),
            ],
            colors=[
                MessageType.REQUEST.color(),
                MessageType.RETRY_REQUEST.color(),
                QUEUEING_COLOR,
            ],
            bar_length=server.height,
            bar_width=server.height / 10,
        )
        concurrency_bar.next_to(
            server,
            RIGHT,
            buff=stroke_width_buffer(server, concurrency_bar),
            aligned_edge=DOWN,
        )
        concurrency_bar.add_updater(
            lambda m: m.set_values(
                [
                    avg_req_concurrency.get_value(),
                    avg_retry_concurrency.get_value(),
                    avg_queued_concurrency.get_value(),
                ]
            )
        )
        concurrency_label = create_label(
            server,
            lambda m: avg_req_concurrency.get_value()
            + avg_retry_concurrency.get_value(),
            "concurrency",
            direction=RIGHT,
            buff=0.2 + concurrency_bar.bar_width,
        ).shift(UP * 0.2)
        concurrency_sparkline = Sparkline(
            get_value=lambda: avg_req_concurrency.get_value()
            + avg_retry_concurrency.get_value(),
            start_y_bounds=(0, server.max_concurrency),
            stroke_color=MessageType.REQUEST.color(),
            size=SMALL,
        ).next_to(
            concurrency_label,
            RIGHT,
            buff=0.2,
        )
        queueing_label = create_label(
            server,
            lambda m: avg_queued_concurrency.get_value(),
            # + avg_queued_retry_concurrency.get_value(),
            "queueing",
        ).next_to(concurrency_label, DOWN, buff=0.2, aligned_edge=RIGHT)
        queueing_sparkline = Sparkline(
            get_value=lambda: avg_queued_concurrency.get_value(),
            # + avg_queued_retry_concurrency.get_value(),
            start_y_bounds=(0, server.max_queue_size),
            stroke_color=QUEUEING_COLOR,
            size=SMALL,
        ).next_to(
            queueing_label,
            RIGHT,
            buff=0.2,
        )
        self.play(
            FadeIn(concurrency_bar),
            Write(concurrency_label),
            FadeIn(concurrency_sparkline),
            Write(queueing_label),
            FadeIn(queueing_sparkline),
            run_time=0.6,
        )
        self.wait(0.3)

        client.start()
        server.start()
        concurrency_sparkline.start()
        queueing_sparkline.start()
        self.wait(5)

        failure_label = create_label(
            server, lambda m: getattr(m, "failure_rate"), "failure_rate"
        ).next_to(queueing_label, DOWN, buff=0.2, aligned_edge=RIGHT)
        self.play(Write(failure_label))
        self.wait(1)

        failure_rate = ValueTracker(0)
        server.add_updater(lambda m: m.set(failure_rate=failure_rate.get_value()))

        self.play(failure_rate.animate.set_value(0.8), run_time=5.0, rate_func=linear)
        self.wait(5)

        self.play(failure_rate.animate.set_value(0), run_time=5.0, rate_func=linear)
        self.wait(2)

        client.set(req_rate=0.0)
        self.wait(10, stop_condition=lambda: client.messages_in_flight() == 0)

        # Stop the processors and sparkline
        client.stop()
        server.stop()
        concurrency_sparkline.stop()

        self.play(
            FadeOut(client),
            FadeOut(conn),
            FadeOut(server),
            FadeOut(concurrency_bar),
            FadeOut(concurrency_label),
            FadeOut(concurrency_sparkline),
            FadeOut(queueing_label),
            FadeOut(queueing_sparkline),
            FadeOut(failure_label),
        )
