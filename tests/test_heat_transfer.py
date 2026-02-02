"""Tests for heat transfer model."""

import numpy as np
import pytest

from engine_sim.engine.geometry import GeometryParams
from engine_sim.engine.heat_transfer import HeatTransfer, WoschniParams


class TestHeatTransfer:
    """Tests for Woschni heat transfer model."""

    def test_zero_heat_transfer_when_equal_temps(self, standard_heat_transfer):
        """Q should be zero when gas temperature equals wall temperature."""
        ht = standard_heat_transfer
        T_wall = 500.0
        T_gas = 500.0  # Same as wall
        Q, h = ht.heat_transfer_rate(
            crank_angle=0.0, rpm=2000.0,
            pressure=1e6, temperature=T_gas,
            p_motored=1e6, p_ref=1e5, T_ref=400.0, V_ref=5e-4,
            T_wall=T_wall
        )
        assert Q == pytest.approx(0.0, abs=1e-6)
        assert h > 0  # Coefficient should still be positive

    def test_positive_heat_transfer_hot_gas(self, standard_heat_transfer):
        """Q should be positive (heat lost) when gas is hotter than wall."""
        ht = standard_heat_transfer
        Q, h = ht.heat_transfer_rate(
            crank_angle=0.0, rpm=2000.0,
            pressure=3e6, temperature=1500.0,
            p_motored=3e6, p_ref=1e5, T_ref=400.0, V_ref=5e-4,
            T_wall=400.0
        )
        assert Q > 0
        assert h > 0

    def test_heat_transfer_scales_with_temp_diff(self, standard_heat_transfer):
        """Q should scale linearly with (T_gas - T_wall) for fixed conditions."""
        ht = standard_heat_transfer
        kwargs = dict(
            crank_angle=np.pi / 4, rpm=2000.0,
            pressure=2e6, p_motored=2e6,
            p_ref=1e5, T_ref=400.0, V_ref=5e-4,
            T_wall=400.0
        )
        Q1, h1 = ht.heat_transfer_rate(temperature=800.0, **kwargs)  # dT=400
        Q2, h2 = ht.heat_transfer_rate(temperature=1200.0, **kwargs)  # dT=800

        # Q = h * A * (T - T_wall), but h depends on T so it's not exact 2x
        # However the ratio Q2/Q1 should be greater than 1
        assert Q2 > Q1

    def test_coefficient_always_positive(self, standard_heat_transfer):
        """Heat transfer coefficient should always be positive."""
        ht = standard_heat_transfer
        for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
            for P in [1e5, 1e6, 5e6]:
                for T in [400, 1000, 2000]:
                    _, h = ht.heat_transfer_rate(
                        crank_angle=theta, rpm=2000.0,
                        pressure=P, temperature=T,
                        p_motored=P, p_ref=1e5, T_ref=400.0,
                        V_ref=5e-4, T_wall=400.0
                    )
                    assert h > 0

    def test_custom_woschni_params(self, standard_geometry):
        """Test that custom WoschniParams are applied."""
        params = WoschniParams(C_scale=3.0)  # Double the default 1.5
        ht_custom = HeatTransfer(geom=standard_geometry, params=params)
        ht_default = HeatTransfer(geom=standard_geometry, params=WoschniParams())

        kwargs = dict(
            crank_angle=0.0, rpm=2000.0,
            pressure=2e6, temperature=1000.0,
            p_motored=2e6, p_ref=1e5, T_ref=400.0,
            V_ref=5e-4, T_wall=400.0
        )
        Q_custom, _ = ht_custom.heat_transfer_rate(**kwargs)
        Q_default, _ = ht_default.heat_transfer_rate(**kwargs)

        assert Q_custom == pytest.approx(2.0 * Q_default, rel=1e-10)
