"""Tests for otter.config.utils module."""

from pydantic_core import PydanticUndefined

from otter.config.utils import (
    ROOT_DIR,
    tracked_field,
    untracked_field,
    coerce_empty_str,
)


# ── ROOT_DIR ──

class TestRootDir:
    def test_root_dir_is_absolute(self):
        assert ROOT_DIR.is_absolute()

    def test_root_dir_points_to_project_root(self):
        """ROOT_DIR should be 4 levels up from utils.py (config -> otter -> src -> ROOT)."""
        assert (ROOT_DIR / "pyproject.toml").exists()

    def test_root_dir_is_directory(self):
        assert ROOT_DIR.is_dir()


# ── tracked_field / untracked_field ──

class TestTrackedField:
    def test_tracked_field_sets_core_true(self):
        f = tracked_field(default="x", description="test")
        assert f.json_schema_extra["core"] is True

    def test_untracked_field_sets_core_false(self):
        f = untracked_field(default="x", description="test")
        assert f.json_schema_extra["core"] is False

    def test_tracked_field_preserves_default(self):
        f = tracked_field(default=42, description="test")
        assert f.default == 42

    def test_untracked_field_preserves_default(self):
        f = untracked_field(default="hello", description="test")
        assert f.default == "hello"

    def test_tracked_field_required_when_no_default(self):
        """When no default is given, the field should be PydanticUndefined."""
        f = tracked_field(description="required field")
        assert f.default is PydanticUndefined

    def test_untracked_field_required_when_no_default(self):
        f = untracked_field(description="required field")
        assert f.default is PydanticUndefined

    def test_tracked_field_passes_extra_kwargs(self):
        """Extra kwargs like ge, le should be forwarded to pydantic Field."""
        f = tracked_field(default=3, ge=1, description="with ge")
        # pydantic stores ge in metadata
        assert any(
            getattr(m, "ge", None) == 1
            for m in f.metadata
        )

    def test_tracked_field_preserves_existing_json_schema_extra(self):
        """If json_schema_extra is passed, it should be merged, not overwritten."""
        f = tracked_field(default=1, json_schema_extra={"custom": "value"})
        assert f.json_schema_extra["core"] is True
        assert f.json_schema_extra["custom"] == "value"

    def test_untracked_field_preserves_existing_json_schema_extra(self):
        f = untracked_field(default=1, json_schema_extra={"custom": "value"})
        assert f.json_schema_extra["core"] is False
        assert f.json_schema_extra["custom"] == "value"


# ── coerce_empty_str ──

class TestCoerceEmptyStr:
    def test_empty_string_becomes_none(self):
        assert coerce_empty_str("") is None

    def test_whitespace_only_becomes_none(self):
        assert coerce_empty_str("   ") is None

    def test_tab_only_becomes_none(self):
        assert coerce_empty_str("\t") is None

    def test_non_empty_string_unchanged(self):
        assert coerce_empty_str("hello") == "hello"

    def test_string_with_spaces_unchanged(self):
        assert coerce_empty_str("  hello  ") == "  hello  "

    def test_none_passthrough(self):
        assert coerce_empty_str(None) is None

    def test_int_passthrough(self):
        assert coerce_empty_str(42) == 42

    def test_zero_passthrough(self):
        assert coerce_empty_str(0) == 0

    def test_false_passthrough(self):
        assert coerce_empty_str(False) is False

    def test_list_passthrough(self):
        val = [1, 2, 3]
        assert coerce_empty_str(val) is val

    def test_float_passthrough(self):
        assert coerce_empty_str(3.14) == 3.14

