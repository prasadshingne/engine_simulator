"""Tests for configuration loading and path resolution."""

import os
import pytest

from engine_sim.config.paths import (
    get_project_root, get_default_config,
    get_mechanism_dir, resolve_mechanism_path
)
from engine_sim.simulation.engine import EngineConfig


class TestPaths:
    """Tests for path resolution utilities."""

    def test_project_root_exists(self):
        """Project root should be a real directory containing pyproject.toml."""
        root = get_project_root()
        assert root.is_dir()
        assert (root / "pyproject.toml").exists()

    def test_default_config_exists(self):
        """Default config YAML should exist."""
        config = get_default_config()
        assert config.exists()
        assert config.name == "default_config.yaml"

    def test_mechanism_dir_exists(self):
        """Mechanisms directory should exist."""
        mech_dir = get_mechanism_dir()
        assert mech_dir.is_dir()

    def test_resolve_mechanism_relative(self):
        """resolve_mechanism_path should find mechanism via relative path."""
        path = resolve_mechanism_path("data/mechanisms/Nissan_chem.yaml")
        assert os.path.exists(path)
        assert "Nissan_chem.yaml" in path

    def test_resolve_mechanism_by_name(self):
        """resolve_mechanism_path should find mechanism by filename alone."""
        path = resolve_mechanism_path("Nissan_chem.yaml")
        assert os.path.exists(path)

    def test_resolve_mechanism_nonexistent_passthrough(self):
        """Nonexistent mechanism should pass through unchanged."""
        path = resolve_mechanism_path("nonexistent_mech.yaml")
        assert path == "nonexistent_mech.yaml"


class TestEngineConfig:
    """Tests for EngineConfig loading."""

    def test_from_yaml_default(self):
        """EngineConfig.from_yaml() with no args should load default config."""
        config = EngineConfig.from_yaml()
        assert isinstance(config, EngineConfig)

    def test_config_types(self):
        """All config fields should have correct types."""
        config = EngineConfig.from_yaml()
        assert isinstance(config.bore, float)
        assert isinstance(config.stroke, float)
        assert isinstance(config.comp_ratio, float)
        assert isinstance(config.speed, float)
        assert isinstance(config.phi, float)
        assert isinstance(config.egr, float)
        assert isinstance(config.adiabatic, bool)
        assert isinstance(config.model_type, str)
        assert isinstance(config.nzones, int)

    def test_config_reasonable_values(self):
        """Config values should be physically reasonable."""
        config = EngineConfig.from_yaml()
        assert 0.05 < config.bore < 0.2  # 50-200mm bore
        assert 0.05 < config.stroke < 0.2
        assert 5 < config.comp_ratio < 25
        assert 500 < config.speed < 10000  # RPM
        assert 0.0 < config.phi < 2.0
        assert 0.0 <= config.egr < 1.0
        assert config.pressure > 0
        assert config.temperature > 200

    def test_config_from_explicit_path(self, default_config_path):
        """EngineConfig should load from an explicit path."""
        config = EngineConfig.from_yaml(default_config_path)
        assert config.fuel == "C8H18"
