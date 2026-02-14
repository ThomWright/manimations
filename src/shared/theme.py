"""Colour theme system for animations.

Set the MANIM_THEME environment variable to select a theme.
Available themes: "personal" (default), "manim".
"""

import os
from dataclasses import dataclass

from manim import (
    BLACK,
    BLUE,
    GREEN,
    LIGHTER_GRAY,
    ORANGE,
    RED,
    WHITE,
    YELLOW,
    ManimColor,
)


@dataclass(frozen=True)
class Theme:
    """A set of semantic colours for animations."""

    primary: ManimColor
    secondary: ManimColor
    accent: ManimColor
    success: ManimColor
    error: ManimColor
    neutral: ManimColor
    background: ManimColor
    foreground: ManimColor


MANIM_THEME = Theme(
    primary=BLUE,
    secondary=YELLOW,
    accent=ORANGE,
    success=GREEN,
    error=RED,
    neutral=LIGHTER_GRAY,
    background=BLACK,
    foreground=WHITE,
)

PERSONAL_THEME = Theme(
    primary=ManimColor("#477dca"),
    secondary=ManimColor("##20416f"),
    accent=ManimColor("###cc6699"),
    success=ManimColor("#a5c882"),
    error=ManimColor("#ed6145"),
    neutral=ManimColor("#666"),
    background=WHITE,
    foreground=ManimColor("#444"),
)

_THEMES: dict[str, Theme] = {
    "personal": PERSONAL_THEME,
    "manim": MANIM_THEME,
}

_DEFAULT_THEME_NAME = "personal"


def get_theme() -> Theme:
    """Get the active theme, selected by the MANIM_THEME environment variable."""
    name = os.environ.get("MANIM_THEME", _DEFAULT_THEME_NAME)
    theme = _THEMES.get(name)
    if theme is None:
        available = ", ".join(sorted(_THEMES.keys()))
        raise ValueError(f"Unknown theme: {name!r}. Available themes: {available}")
    return theme
