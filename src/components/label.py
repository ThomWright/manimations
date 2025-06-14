from __future__ import annotations

from typing import Callable, TypeVar, cast

from manim import RIGHT, Mobject, Variable
from manim.typing import Vector3D

M = TypeVar("M", bound=Mobject)


def tex_escape_underscores(s: str) -> str:
    """
    Escape underscores in a string for LaTeX rendering.
    """
    return s.replace("_", "{\\_}")


def create_label(
    m: M,
    f: Callable[[M], int | float],
    name: str,
    direction: Vector3D = RIGHT,
    buff: float = 0.2,
) -> Variable:
    """
    Create a label for the given property of the Mobject.
    """
    label = Variable(
        f(m),
        tex_escape_underscores(name),
        num_decimal_places=2,
    )

    for sm in label.submobjects:
        sm.set(font_size=24)
    label.arrange_submobjects()

    label.next_to(m, direction, buff=buff)

    def update_label(v: Mobject):
        v = cast(Variable, v)
        v.tracker.set_value(f(m))

    label.add_updater(update_label)

    return label
