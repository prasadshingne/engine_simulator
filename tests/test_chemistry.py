"""Tests for chemistry interface (requires Cantera)."""

import numpy as np
import pytest

from engine_sim.models.chemistry import Chemistry, ChemistryParams
from engine_sim.config.paths import resolve_mechanism_path

pytestmark = pytest.mark.slow


@pytest.fixture
def chemistry():
    """Create Chemistry object with Nissan mechanism."""
    mech = resolve_mechanism_path("data/mechanisms/Nissan_chem.yaml")
    params = ChemistryParams(mechanism=mech, fuel="C8H18", phi=0.7, egr=0.3)
    return Chemistry(params)


class TestChemistry:
    """Tests for Chemistry class."""

    def test_initialization(self, chemistry):
        """Chemistry should initialize without error."""
        assert chemistry.gas is not None
        assert len(chemistry.gas.species_names) == 33

    def test_setup_initial_mixture(self, chemistry):
        """Initial mixture mass fractions should sum to 1."""
        Y0 = chemistry.setup_initial_mixture(T=475.0, P=1.013e5)
        assert Y0.sum() == pytest.approx(1.0, abs=1e-10)
        assert len(Y0) == 33
        assert np.all(Y0 >= 0)

    def test_fuel_present_in_mixture(self, chemistry):
        """Fuel should be present in initial mixture."""
        Y0 = chemistry.setup_initial_mixture(T=475.0, P=1.013e5)
        fuel_idx = chemistry.gas.species_index("C8H18")
        assert Y0[fuel_idx] > 0

    def test_get_properties(self, chemistry):
        """get_properties should return dict with expected keys."""
        chemistry.setup_initial_mixture(T=475.0, P=1.013e5)
        props = chemistry.get_properties()
        expected_keys = ['cp', 'cv', 'h', 'u', 'rho', 'MW', 'gamma']
        for key in expected_keys:
            assert key in props
            assert np.isfinite(props[key])

    def test_gamma_reasonable(self, chemistry):
        """Gamma should be between 1.1 and 1.5 for combustion gases."""
        chemistry.setup_initial_mixture(T=475.0, P=1.013e5)
        props = chemistry.get_properties()
        assert 1.1 < props['gamma'] < 1.5

    def test_get_reaction_rates_shape(self, chemistry):
        """Reaction rates should have correct shape."""
        chemistry.setup_initial_mixture(T=475.0, P=1.013e5)
        mdot, Q = chemistry.get_reaction_rates()
        assert len(mdot) == 33
        assert np.isfinite(Q)

    def test_update_state(self, chemistry):
        """update_state should set gas to specified conditions."""
        Y0 = chemistry.setup_initial_mixture(T=475.0, P=1.013e5)
        chemistry.update_state(T=800.0, P=2e6, Y=Y0)
        assert chemistry.gas.T == pytest.approx(800.0, rel=1e-6)
        assert chemistry.gas.P == pytest.approx(2e6, rel=1e-6)

    def test_egr_composition(self, chemistry):
        """EGR composition should contain combustion products."""
        X_egr = chemistry.calculate_egr_composition()
        # Should have CO2, H2O, N2
        assert 'N2' in X_egr
        assert 'CO2' in X_egr
        assert 'H2O' in X_egr
