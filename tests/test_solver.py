"""Tests for the ODE solver (requires Cantera)."""

import numpy as np
import pytest

from engine_sim.simulation.engine import EngineConfig, EngineSimulation

pytestmark = pytest.mark.slow


@pytest.fixture
def single_zone_config():
    """Single-zone adiabatic config for fast testing."""
    config = EngineConfig.from_yaml()
    config.adiabatic = True
    config.model_type = "single"
    return config


@pytest.fixture
def multizone_config():
    """10-zone non-adiabatic config using CVODE."""
    config = EngineConfig.from_yaml()
    config.adiabatic = False
    config.model_type = "multi"
    config.nzones = 10
    config.method = "CVODE"
    config.rtol = 1e-4
    config.atol = 1e-10
    return config


has_cvode = False
try:
    from engine_sim.simulation.solver_ida import check_cvode_available
    has_cvode = check_cvode_available()
except ImportError:
    pass


class TestSingleZoneSolver:
    """Tests for single-zone solver."""

    def test_simulation_completes(self, single_zone_config):
        """Single-zone simulation should complete successfully."""
        sim = EngineSimulation(single_zone_config)
        results = sim.run()
        assert results is not None
        assert len(results.temperature) > 0

    def test_solution_dict_keys(self, single_zone_config):
        """Solver output should have required keys."""
        sim = EngineSimulation(single_zone_config)
        y0 = sim.setup_initial_state()
        sol = sim.solver.solve_closed_cycle(
            rpm=single_zone_config.speed,
            T_wall=single_zone_config.wall_temp,
            ca_start=single_zone_config.start_ca,
            ca_end=single_zone_config.end_ca,
            y0=y0,
        )
        for key in ['t', 'y', 'ca', 'success', 'message', 'nfev']:
            assert key in sol

    def test_state_vector_size_single(self, single_zone_config):
        """Single-zone state vector should be 4 + n_species."""
        sim = EngineSimulation(single_zone_config)
        y0 = sim.setup_initial_state()
        nsp = len(sim.chemistry.gas.species_names)
        assert len(y0) == 4 + nsp

    def test_peak_temperature_above_initial(self, single_zone_config):
        """Peak temperature should exceed initial (combustion occurs)."""
        sim = EngineSimulation(single_zone_config)
        results = sim.run()
        assert np.max(results.temperature) > single_zone_config.temperature * 2

    def test_imep_positive(self, single_zone_config):
        """IMEP should be positive (net positive work)."""
        sim = EngineSimulation(single_zone_config)
        results = sim.run()
        perf = results.calculate_performance()
        assert perf['imep'] > 0


@pytest.mark.cvode
@pytest.mark.skipif(not has_cvode, reason="scikits.odes/CVODE not installed")
class TestMultiZoneSolver:
    """Tests for multi-zone solver with CVODE."""

    def test_multizone_completes(self, multizone_config):
        """10-zone non-adiabatic simulation should complete with CVODE."""
        sim = EngineSimulation(multizone_config)
        results = sim.run()
        assert results is not None
        assert len(results.temperature) > 0

    def test_state_vector_size_multi(self, multizone_config):
        """Multizone state vector: 3 + nzones*(nsp+1) + nsp."""
        sim = EngineSimulation(multizone_config)
        y0 = sim.setup_initial_state()
        nsp = len(sim.chemistry.gas.species_names)
        nzones = multizone_config.nzones
        expected = 3 + nzones * (nsp + 1) + nsp
        assert len(y0) == expected

    def test_multizone_imep_positive(self, multizone_config):
        """Multizone IMEP should be positive."""
        sim = EngineSimulation(multizone_config)
        results = sim.run()
        perf = results.calculate_performance()
        assert perf['imep'] > 0

    def test_multizone_temperature_stratification(self, multizone_config):
        """Non-adiabatic multizone should show temperature stratification."""
        sim = EngineSimulation(multizone_config)
        y0 = sim.setup_initial_state()
        sol = sim.solver.solve_closed_cycle(
            rpm=multizone_config.speed,
            T_wall=multizone_config.wall_temp,
            ca_start=multizone_config.start_ca,
            ca_end=multizone_config.end_ca,
            y0=y0,
        )
        # Extract zone temperatures at last timestep
        nsp = len(sim.chemistry.gas.species_names)
        nzones = multizone_config.nzones
        zone_temps_final = []
        for i in range(nzones):
            idx = 3 + i * (nsp + 1)
            zone_temps_final.append(sol['y'][idx, -1])

        # With heat transfer, zones should have different final temperatures
        temp_spread = max(zone_temps_final) - min(zone_temps_final)
        assert temp_spread > 10  # At least some stratification
