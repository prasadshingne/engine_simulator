"""Tests for engine geometry calculations."""

import numpy as np
import pytest

from engine_sim.engine.geometry import GeometryParams


class TestGeometryParams:
    """Tests for GeometryParams dataclass and derived quantities."""

    def test_derived_quantities(self, standard_geometry):
        """Verify derived quantities are computed correctly in __post_init__."""
        g = standard_geometry
        assert g.crank_radius == pytest.approx(0.043, abs=1e-6)
        assert g.bore_area == pytest.approx(np.pi * (0.086 / 2) ** 2, rel=1e-10)
        assert g.displacement == pytest.approx(g.bore_area * 0.086, rel=1e-10)
        assert g.clearance_volume == pytest.approx(
            g.displacement / (12.5 - 1), rel=1e-10
        )

    def test_compression_ratio(self, standard_geometry):
        """V_BDC / V_TDC should equal the compression ratio."""
        g = standard_geometry
        V_tdc = g.cylinder_volume(0.0)  # TDC at theta=0
        V_bdc = g.cylinder_volume(np.pi)  # BDC at theta=pi
        assert V_bdc / V_tdc == pytest.approx(g.comp_ratio, rel=1e-3)

    def test_volume_at_tdc(self, standard_geometry):
        """Volume at TDC should equal clearance volume."""
        g = standard_geometry
        V_tdc = g.cylinder_volume(0.0)
        assert V_tdc == pytest.approx(g.clearance_volume, rel=1e-3)

    def test_volume_at_bdc(self, standard_geometry):
        """Volume at BDC should equal clearance + displacement."""
        g = standard_geometry
        V_bdc = g.cylinder_volume(np.pi)
        assert V_bdc == pytest.approx(
            g.clearance_volume + g.displacement, rel=1e-3
        )

    def test_volume_always_positive(self, standard_geometry):
        """Cylinder volume must be positive at all crank angles."""
        g = standard_geometry
        angles = np.linspace(0, 2 * np.pi, 360)
        for theta in angles:
            assert g.cylinder_volume(theta) > 0

    def test_volume_symmetric(self, standard_geometry):
        """Volume should be symmetric about TDC: V(theta) == V(-theta)."""
        g = standard_geometry
        for theta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            assert g.cylinder_volume(theta) == pytest.approx(
                g.cylinder_volume(-theta), rel=1e-10
            )

    def test_piston_position_symmetric(self, standard_geometry):
        """Piston position should be symmetric: pos(theta) == pos(-theta)."""
        g = standard_geometry
        for theta in [0.3, 0.7, 1.2, 2.0]:
            assert g.piston_position(theta) == pytest.approx(
                g.piston_position(-theta), rel=1e-10
            )

    def test_volume_rate_zero_at_tdc_bdc(self, standard_geometry):
        """Volume rate should be approximately zero at TDC and BDC."""
        g = standard_geometry
        rpm = 2000.0
        # At TDC (theta=0) and BDC (theta=pi), piston velocity is zero
        assert g.volume_rate(0.0, rpm) == pytest.approx(0.0, abs=1e-10)
        assert g.volume_rate(np.pi, rpm) == pytest.approx(0.0, abs=1e-10)

    def test_volume_rate_scales_with_rpm(self, standard_geometry):
        """Volume rate should scale linearly with RPM."""
        g = standard_geometry
        theta = np.pi / 3  # 60 degrees
        rate_1000 = g.volume_rate(theta, 1000.0)
        rate_2000 = g.volume_rate(theta, 2000.0)
        assert rate_2000 == pytest.approx(2.0 * rate_1000, rel=1e-10)

    def test_surface_area_components(self, standard_geometry):
        """Surface area should return three positive components."""
        g = standard_geometry
        head, piston, liner = g.surface_area(np.pi / 4)
        assert head > 0
        assert piston > 0
        assert liner >= 0  # Can be zero at TDC

    def test_surface_area_head_piston_constant(self, standard_geometry):
        """Head and piston areas should be constant (equal to bore area)."""
        g = standard_geometry
        for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
            head, piston, _ = g.surface_area(theta)
            assert head == pytest.approx(g.bore_area, rel=1e-10)
            assert piston == pytest.approx(g.bore_area, rel=1e-10)

    def test_surface_area_liner_varies(self, standard_geometry):
        """Liner area should vary with crank angle (min at TDC, max at BDC)."""
        g = standard_geometry
        _, _, liner_tdc = g.surface_area(0.0)
        _, _, liner_bdc = g.surface_area(np.pi)
        assert liner_bdc > liner_tdc

    def test_different_geometries(self):
        """Test with a different engine geometry."""
        g = GeometryParams(bore=0.1, stroke=0.1, con_rod=0.2, comp_ratio=10.0)
        assert g.displacement == pytest.approx(np.pi * 0.05**2 * 0.1, rel=1e-10)
        V_tdc = g.cylinder_volume(0.0)
        V_bdc = g.cylinder_volume(np.pi)
        assert V_bdc / V_tdc == pytest.approx(10.0, rel=1e-3)
