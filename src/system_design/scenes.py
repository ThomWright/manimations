from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    Indicate,
    ValueTracker,
    Write,
    linear,
)

from shared.themed_scene import ThemedScene
from shared.aggregators.moving_avg_tracker import MovingAverageTracker
from shared.components.bar import StackedBar, stroke_width_buffer
from shared.components.label import create_label
from shared.components.sparkline import Sparkline
from shared.constants import SMALL
from system_design.connection import Connection
from shared.theme import get_theme
from system_design.message import Message, MessageType
from system_design.processor import Processor, RetryPolicy
from system_design.queue import Queue


class MessageQueueTest(ThemedScene):
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


def immediate(t: float) -> float:
    return 1


class ClientServerTest(ThemedScene):
    def construct(self):
        # Introduce client and server
        client = Processor(size=SMALL, retry_policy=RetryPolicy(), req_rate=0.0).shift(
            LEFT * 1.5
        )
        server = Processor(
            size=SMALL, failure_rate=0, max_concurrency=5, max_queue_size=5
        ).shift(RIGHT * 1.5)
        self.play(
            FadeIn(client, run_time=0.3),
            FadeIn(server, run_time=0.3),
        )
        self.wait(0.3)

        # Add a connection
        conn = Connection(client.get_right(), server.get_left(), server)
        client.add_client_connection(conn)
        self.play(
            Write(conn, run_time=0.3),
        )
        self.wait(0.3)

        # Show RPS label
        rps_source = ValueTracker()
        rps_source.set_value(client.gen_rps())
        client.add_updater(lambda m: m.set(gen_req_rate=rps_source.get_value()))
        rps_label = create_label(
            client,
            lambda m: rps_source.get_value(),
            "rps",
            direction=LEFT,
            buff=0.2,
        ).shift(UP * 0.2)
        self.play(
            Write(rps_label),
            run_time=0.6,
        )
        self.wait(0.6)

        # Send some requests
        client.start()
        server.start()
        self.play(rps_source.animate.set_value(5.0), run_time=0.1, rate_func=immediate)
        self.play(Indicate(rps_label))
        self.wait(3)

        # Stop requests
        self.play(rps_source.animate.set_value(0.0), run_time=0.1, rate_func=immediate)
        self.play(Indicate(rps_label))
        self.wait(10, stop_condition=lambda: client.messages_in_flight() == 0)

        # Show actual RPS label
        actual_rps_tracker = ValueTracker()
        actual_rps_tracker.set_value(client.actual_rps(conn))
        client.add_updater(lambda m: actual_rps_tracker.set_value(m.actual_rps(conn)))
        actual_rps_label = create_label(
            client,
            lambda m: actual_rps_tracker.get_value(),
            "actual_rps",
            direction=LEFT,
            buff=0.2,
        ).next_to(rps_label, DOWN, buff=0.2, aligned_edge=RIGHT)
        self.play(
            Write(actual_rps_label),
            run_time=0.6,
        )
        self.wait(0.6)

        # Send some more requests
        self.play(rps_source.animate.set_value(5.0), run_time=0.1, rate_func=immediate)
        self.play(Indicate(rps_label))
        self.wait(3)
        self.play(rps_source.animate.set_value(0.0), run_time=0.1, rate_func=immediate)
        self.play(Indicate(rps_label))
        self.wait(10, stop_condition=lambda: client.messages_in_flight() == 0)

        # Show the server labels and sparklines

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
                get_theme().accent,
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
            "queueing",
        ).next_to(concurrency_label, DOWN, buff=0.2, aligned_edge=RIGHT)
        queueing_sparkline = Sparkline(
            get_value=lambda: avg_queued_concurrency.get_value(),
            start_y_bounds=(0, server.max_queue_size),
            stroke_color=get_theme().accent,
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

        # Send some requests
        concurrency_sparkline.start()
        queueing_sparkline.start()
        self.play(rps_source.animate.set_value(5.0), run_time=0.1, rate_func=immediate)
        self.play(Indicate(rps_label))
        self.wait(5)

        # Show failure rate labels
        sim_failure_label = create_label(
            server, lambda m: getattr(m, "sim_failure_rate"), "failure_rate"
        ).next_to(queueing_label, DOWN, buff=0.2, aligned_edge=RIGHT)
        actual_failure_tracker = ValueTracker()
        actual_failure_tracker.set_value(server.failure_rate(conn))
        server.add_updater(
            lambda m: actual_failure_tracker.set_value(m.failure_rate(conn))
        )
        actual_failure_label = create_label(
            server,
            lambda m: actual_failure_tracker.get_value(),
            "actual_failure_rate",
            direction=RIGHT,
            buff=0.2,
        ).next_to(sim_failure_label, DOWN, buff=0.2, aligned_edge=RIGHT)
        self.play(Write(sim_failure_label), Write(actual_failure_label))
        self.wait(1)

        # Ramp up failure rate
        failure_rate = ValueTracker(0)
        server.add_updater(lambda m: m.set(sim_failure_rate=failure_rate.get_value()))
        self.play(Indicate(sim_failure_label))
        self.play(failure_rate.animate.set_value(0.8), run_time=5.0, rate_func=linear)
        self.wait(10)

        # And ramp it back down again
        self.play(Indicate(sim_failure_label))
        self.play(failure_rate.animate.set_value(0), run_time=5.0, rate_func=linear)
        self.wait(2)

        #
        self.play(rps_source.animate.set_value(0.0), run_time=0.1, rate_func=immediate)
        self.play(Indicate(rps_label))
        self.wait(10, stop_condition=lambda: client.messages_in_flight() == 0)

        # Stop the processors and sparkline
        client.stop()
        server.stop()
        concurrency_sparkline.stop()

        self.play(
            FadeOut(client),
            FadeOut(conn),
            FadeOut(server),
            FadeOut(rps_label),
            FadeOut(actual_rps_label),
            FadeOut(concurrency_bar),
            FadeOut(concurrency_label),
            FadeOut(concurrency_sparkline),
            FadeOut(queueing_label),
            FadeOut(queueing_sparkline),
            FadeOut(sim_failure_label),
            FadeOut(actual_failure_label),
        )
