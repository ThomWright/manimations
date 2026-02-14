from shared.components.label import tex_escape_underscores


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
