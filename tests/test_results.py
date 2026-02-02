"""Tests for simulation results processing.

Uses synthetic data to avoid requiring Cantera.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for testing
import pytest

from engine_sim.simulation.results import SimulationResults


@pytest.fixture
def synthetic_results():
    """Create synthetic simulation results for testing."""
    n_points = 100
    n_species = 5
    species_names = ['C8H18', 'O2', 'N2', 'CO2', 'H2O']

    # Simulate a compression-combustion-expansion cycle
    ca = np.linspace(-180, 180, n_points)
    t = np.linspace(0, 0.01, n_points)

    # Simple profiles
    temperature = 500 + 1500 * np.exp(-0.5 * (ca / 20) ** 2)  # Peak at TDC
    pressure = 1e5 + 4e6 * np.exp(-0.5 * (ca / 15) ** 2)
    volume = 5e-4 - 4.5e-4 * np.exp(-0.5 * (ca / 50) ** 2)  # Min at TDC
    volume = np.maximum(volume, 4e-5)  # Clearance volume
    mass = np.full(n_points, 5e-4)  # Constant mass

    # Species: fuel decreases, products increase
    species = np.zeros((n_species, n_points))
    species[0] = 0.05 * (1 - 0.5 * (1 + np.tanh(ca / 10)))  # C8H18 decreasing
    species[1] = 0.20 * (1 - 0.3 * (1 + np.tanh(ca / 10)))  # O2 decreasing
    species[2] = np.full(n_points, 0.65)  # N2 constant
    species[3] = 0.05 * (1 + np.tanh(ca / 10))  # CO2 increasing
    species[4] = 0.05 * (1 + np.tanh(ca / 10))  # H2O increasing

    return SimulationResults(
        time=t,
        crank_angle=ca,
        temperature=temperature,
        pressure=pressure,
        volume=volume,
        mass=mass,
        species=species,
        species_names=species_names,
    )


class TestSimulationResults:
    """Tests for SimulationResults dataclass."""

    def test_fields_stored(self, synthetic_results):
        """All fields should be stored correctly."""
        r = synthetic_results
        assert len(r.time) == 100
        assert len(r.crank_angle) == 100
        assert len(r.temperature) == 100
        assert len(r.species_names) == 5
        assert r.species.shape == (5, 100)

    def test_calculate_performance_keys(self, synthetic_results):
        """Performance dict should have expected keys."""
        perf = synthetic_results.calculate_performance()
        assert 'indicated_work' in perf
        assert 'imep' in perf
        assert 'peak_pressure' in perf
        assert 'peak_temperature' in perf

    def test_peak_values(self, synthetic_results):
        """Peak values should match max of arrays."""
        perf = synthetic_results.calculate_performance()
        assert perf['peak_pressure'] == pytest.approx(
            np.max(synthetic_results.pressure) / 1e5, rel=1e-10
        )
        assert perf['peak_temperature'] == pytest.approx(
            np.max(synthetic_results.temperature), rel=1e-10
        )

    def test_plot_methods_run(self, synthetic_results, tmp_path):
        """Plotting methods should run without error."""
        r = synthetic_results
        r.plot_pressure_volume(str(tmp_path / "pv.png"))
        r.plot_temperature_ca(str(tmp_path / "temp.png"))
        r.plot_pressure_ca(str(tmp_path / "press.png"))
        r.plot_mass_ca(str(tmp_path / "mass.png"))
        r.plot_species(species_indices=[0, 1], save_path=str(tmp_path / "species.png"))

        # Check files were created
        assert (tmp_path / "pv.png").exists()
        assert (tmp_path / "temp.png").exists()
        assert (tmp_path / "press.png").exists()

    def test_from_solver_output_single(self):
        """from_solver_output should parse single-zone format."""
        n_points = 50
        nsp = 3
        y = np.random.rand(4 + nsp, n_points)
        output = {
            't': np.linspace(0, 0.01, n_points),
            'y': y,
            'ca': np.linspace(-180, 180, n_points),
            'success': True,
            'message': 'OK',
            'nfev': 100,
            'njev': 10,
        }
        species_names = ['A', 'B', 'C']
        results = SimulationResults.from_solver_output(
            output, species_names, model_type="single"
        )
        assert len(results.temperature) == n_points
        assert results.species.shape == (nsp, n_points)
        # Verify correct index mapping: T=0, V=1, P=2, m=3, Y=4:
        np.testing.assert_array_equal(results.temperature, y[0])
        np.testing.assert_array_equal(results.volume, y[1])
        np.testing.assert_array_equal(results.pressure, y[2])
        np.testing.assert_array_equal(results.mass, y[3])
