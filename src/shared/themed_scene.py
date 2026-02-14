"""Base scene that applies the active theme's background and foreground colours."""

from manim import Mobject, Scene

from shared.theme import get_theme


class ThemedScene(Scene):
    def setup(self):
        theme = get_theme()
        self.camera.background_color = theme.background
        Mobject.set_default(color=theme.foreground)
