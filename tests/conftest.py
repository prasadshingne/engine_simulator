"""Shared pytest fixtures for engine simulator tests."""

import pytest
import numpy as np

from engine_sim.engine.geometry import GeometryParams
from engine_sim.engine.heat_transfer import HeatTransfer, WoschniParams
from engine_sim.config.paths import get_default_config, resolve_mechanism_path


@pytest.fixture
def standard_geometry():
    """Standard engine geometry matching default_config.yaml."""
    return GeometryParams(
        bore=0.086, stroke=0.086, con_rod=0.1455, comp_ratio=12.5
    )


@pytest.fixture
def standard_heat_transfer(standard_geometry):
    """Standard Woschni heat transfer model."""
    return HeatTransfer(geom=standard_geometry, params=WoschniParams())


@pytest.fixture
def nissan_mechanism_path():
    """Path to Nissan iso-octane mechanism."""
    return resolve_mechanism_path("data/mechanisms/Nissan_chem.yaml")


@pytest.fixture
def default_config_path():
    """Path to default configuration file."""
    return str(get_default_config())
