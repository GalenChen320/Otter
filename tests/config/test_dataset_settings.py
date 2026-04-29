"""Tests for otter.config.dataset_settings module."""

from pathlib import Path

from otter.config.dataset_settings import (
    DatasetSettings,
    EvalplusSettings,
    MbppplusSettings,
    DATASET_SETTINGS_REGISTRY,
)
from otter.config.utils import ROOT_DIR


# ── Registry ──

class TestDatasetSettingsRegistry:
    def test_registry_contains_evalplus(self):
        assert "evalplus" in DATASET_SETTINGS_REGISTRY

    def test_registry_contains_mbppplus(self):
        assert "mbppplus" in DATASET_SETTINGS_REGISTRY

    def test_registry_evalplus_maps_to_correct_class(self):
        assert DATASET_SETTINGS_REGISTRY["evalplus"] is EvalplusSettings

    def test_registry_mbppplus_maps_to_correct_class(self):
        assert DATASET_SETTINGS_REGISTRY["mbppplus"] is MbppplusSettings

    def test_all_registry_values_are_dataset_settings_subclasses(self):
        for name, cls in DATASET_SETTINGS_REGISTRY.items():
            assert issubclass(cls, DatasetSettings), f"{name} is not a DatasetSettings subclass"


# ── EvalplusSettings ──

class TestEvalplusSettings:
    def test_default_cache_dir(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        s = EvalplusSettings(_env_file=env_file, _env_prefix="DATASET__")
        assert s.cache_dir == ROOT_DIR / "data" / "cache"

    def test_custom_cache_dir_from_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("DATASET__cache_dir=/custom/path\n")
        s = EvalplusSettings(_env_file=env_file, _env_prefix="DATASET__")
        assert s.cache_dir == Path("/custom/path")

    def test_cache_dir_is_untracked(self):
        extra = EvalplusSettings.model_fields["cache_dir"].json_schema_extra
        assert extra["core"] is False


# ── MbppplusSettings ──

class TestMbppplusSettings:
    def test_default_cache_dir(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        s = MbppplusSettings(_env_file=env_file, _env_prefix="DATASET__")
        assert s.cache_dir == ROOT_DIR / "data" / "cache"

    def test_custom_cache_dir_from_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("DATASET__cache_dir=/another/path\n")
        s = MbppplusSettings(_env_file=env_file, _env_prefix="DATASET__")
        assert s.cache_dir == Path("/another/path")

    def test_cache_dir_is_untracked(self):
        extra = MbppplusSettings.model_fields["cache_dir"].json_schema_extra
        assert extra["core"] is False


# ── Extra fields ignored ──

class TestDatasetSettingsExtraIgnored:
    def test_unknown_env_vars_are_ignored(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "DATASET__cache_dir=/some/path\n"
            "DATASET__nonexistent_field=foobar\n"
        )
        s = EvalplusSettings(_env_file=env_file, _env_prefix="DATASET__")
        assert s.cache_dir == Path("/some/path")
        assert not hasattr(s, "nonexistent_field")
