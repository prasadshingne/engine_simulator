"""
Cantera ReactorNet solver for engine simulation.

This module provides a fast solver using Cantera's built-in ReactorNet,
which uses CVODES internally with analytical Jacobians. This is much faster
than the manual ODE approach for large mechanisms (100+ species).

Key advantages:
- Analytical Jacobian from Cantera (not numerical approximation)
- CVODES solver optimized for chemical kinetics
- Handles stiff chemistry naturally
"""

import numpy as np
import cantera as ct
from typing import Dict, Optional
from dataclasses import dataclass

from ..engine.geometry import GeometryParams
from ..engine.heat_transfer import HeatTransfer, WoschniParams
from ..models.chemistry import Chemistry
from .multizone_profiles import (
    validate_multizone_count,
    zone_heat_loss_multipliers,
    zone_mass_fractions,
)


@dataclass
class CanteraSolverParams:
    """Parameters for Cantera ReactorNet solver."""
    rtol: float = 1.0e-6          # Relative tolerance
    atol: float = 1.0e-12         # Absolute tolerance
    max_steps: int = 10000        # Maximum steps between outputs
    adiabatic: bool = False       # Whether to run in adiabatic mode
    verbose: bool = True          # Print solver progress


class CanteraEngineSolver:
    """
    Engine cycle solver using Cantera's ReactorNet.

    This solver uses Cantera's IdealGasReactor with a velocity-controlled
    Wall to simulate piston motion. It leverages Cantera's built-in CVODES
    solver with analytical Jacobians for fast, robust integration.

    Suitable for:
    - Large mechanisms (100+ species) where numerical Jacobian is too slow
    - Single-zone simulations
    - Both adiabatic and non-adiabatic cases
    """

    def __init__(self, geom: GeometryParams, chemistry: Chemistry,
                 heat_transfer: Optional[HeatTransfer] = None,
                 params: CanteraSolverParams = None):
        """
        Initialize Cantera engine solver.

        Parameters
        ----------
        geom : GeometryParams
            Engine geometry parameters
        chemistry : Chemistry
            Chemistry interface (Cantera gas already initialized)
        heat_transfer : HeatTransfer, optional
            Heat transfer model (for non-adiabatic simulations)
        params : CanteraSolverParams, optional
            Solver parameters
        """
        self.geom = geom
        self.chemistry = chemistry
        self.heat_transfer = heat_transfer
        self.params = params or CanteraSolverParams()

        # Store reference to Cantera gas object
        self.gas = chemistry.gas

    def solve_closed_cycle(self, rpm: float, T_wall: float,
                          ca_start: float, ca_end: float,
                          y0: np.ndarray) -> Dict:
        """
        Solve closed cycle from IVC to EVO using Cantera ReactorNet.

        Parameters
        ----------
        rpm : float
            Engine speed [rpm]
        T_wall : float
            Wall temperature [K]
        ca_start : float
            Start crank angle [deg] (relative to TDC, e.g., -180 for BDC)
        ca_end : float
            End crank angle [deg]
        y0 : np.ndarray
            Initial state vector: [T, V, P, m, Y...] for single zone

        Returns
        -------
        Dict
            Solution dictionary with keys:
            - 't': time array [s]
            - 'y': state array (each row is a state variable over time)
            - 'ca': crank angle array [deg]
            - 'success': bool
            - 'message': str
            - 'nfev': int (not tracked, set to 0)
        """
        # Extract initial conditions from y0
        # Single zone format: [T, V, P, m, Y...]
        T0 = y0[0]
        V0 = y0[1]
        P0 = y0[2]
        m0 = y0[3]
        Y0 = y0[4:]

        # Calculate angular velocity
        omega = rpm * 2 * np.pi / 60  # [rad/s]

        # Time span
        theta_start = np.deg2rad(ca_start)
        theta_end = np.deg2rad(ca_end)
        t_end = (theta_end - theta_start) / omega

        # Set up gas with initial conditions
        self.gas.TPY = T0, P0, Y0

        # Store reference conditions for heat transfer
        self.p_ref = P0
        self.T_ref = T0
        self.V_ref = V0
        self.gamma = self.gas.cp / self.gas.cv

        if self.params.verbose:
            print(f"\nCantera ReactorNet Solver:")
            print(f"  Species: {len(Y0)}")
            print(f"  Time span: 0 to {t_end*1000:.3f} ms")
            print(f"  CA range: {ca_start:.1f}° to {ca_end:.1f}°")
            print(f"  Initial: T={T0:.1f} K, P={P0/1e5:.2f} bar, V={V0*1e6:.2f} cm³")

        # Create reactor
        reactor = ct.IdealGasReactor(self.gas)
        reactor.volume = V0

        # Piston velocity: volume_rate returns -dV/dt, negate for Cantera convention
        def piston_velocity(t):
            theta = omega * t + theta_start
            return -self.geom.volume_rate(theta, rpm)

        # Environment reservoir
        env = ct.Reservoir(ct.Solution('air.yaml'))

        if not self.params.adiabatic:
            # Woschni parameters
            if self.heat_transfer is not None:
                C = self.heat_transfer.params.C
                C_scale = self.heat_transfer.params.C_scale
            else:
                C = 130.0
                C_scale = 1.5

            Up = 2 * self.geom.stroke * rpm / 60

            # Woschni heat flux: q(t) = h(t) * A_surface(t) * (T_gas - T_wall)
            # Wall.heat_flux accepts a callable, unlike Wall.U which must be constant
            def woschni_heat_flux(t):
                theta = omega * t + theta_start
                V = self.geom.cylinder_volume(theta)
                T_gas = reactor.T
                P_gas = reactor.thermo.P
                h = (C_scale * C *
                     V**0.6 *
                     (P_gas / 1000)**0.8 *
                     T_gas**(-0.4) *
                     (Up + 1.4)**0.8)
                head_a, piston_a, liner_a = self.geom.surface_area(theta)
                A_total = head_a + piston_a + liner_a
                return h * A_total * (T_gas - T_wall)

            if self.params.verbose:
                q0 = woschni_heat_flux(0)
                print(f"  Heat transfer: Woschni C={C}, q_init={q0:.1f} W")

            # Single wall: piston motion via velocity, heat loss via heat_flux
            # A=1 so heat_rate = 1 * heat_flux(t) = total heat rate [W]
            wall = ct.Wall(reactor, env, velocity=piston_velocity, A=1.0)
            wall.heat_flux = woschni_heat_flux
        else:
            wall = ct.Wall(reactor, env, velocity=piston_velocity)

        # Create reactor network
        net = ct.ReactorNet([reactor])
        net.rtol = self.params.rtol
        net.atol = self.params.atol
        net.max_steps = self.params.max_steps

        # Set up output arrays
        n_points = 1200  # Number of output points (0.3°/pt for 360° cycle)
        t_eval = np.linspace(0, t_end, n_points)

        # State storage: [T, V, P, m, Y...]
        n_species = len(Y0)
        n_states = 4 + n_species
        y_out = np.zeros((n_states, n_points))
        t_out = np.zeros(n_points)
        q_wall_out = np.zeros(n_points)  # Track wall heat rate [W]

        # Store initial state
        t_out[0] = 0
        y_out[0, 0] = reactor.T
        y_out[1, 0] = reactor.volume
        y_out[2, 0] = reactor.thermo.P
        y_out[3, 0] = reactor.mass
        y_out[4:, 0] = reactor.Y
        q_wall_out[0] = wall.heat_rate

        success = True
        message = "Integration successful"

        if self.params.verbose:
            print("  Integrating...", end="", flush=True)

        try:
            for i, t_target in enumerate(t_eval[1:], start=1):
                # Advance to next time point
                net.advance(t_target)

                # Store state
                t_out[i] = net.time
                y_out[0, i] = reactor.T
                y_out[1, i] = reactor.volume
                y_out[2, i] = reactor.thermo.P
                y_out[3, i] = reactor.mass
                y_out[4:, i] = reactor.Y
                q_wall_out[i] = wall.heat_rate

        except Exception as e:
            success = False
            message = f"Integration failed: {str(e)}"
            if self.params.verbose:
                print(f" FAILED at t={net.time*1000:.3f} ms")
                print(f"  Error: {e}")
            # Truncate arrays to valid data
            valid_idx = np.where(t_out > 0)[0]
            if len(valid_idx) > 0:
                last_valid = valid_idx[-1] + 1
                t_out = t_out[:last_valid]
                y_out = y_out[:, :last_valid]

        if success and self.params.verbose:
            print(" done!")
            print(f"  Final: T={reactor.T:.1f} K, P={reactor.thermo.P/1e5:.2f} bar")
            print(f"  Peak T: {np.max(y_out[0]):.1f} K")
            print(f"  Peak P: {np.max(y_out[2])/1e5:.2f} bar")

        # Calculate crank angles
        crank_angles = ca_start + np.rad2deg(omega * t_out)

        return {
            't': t_out,
            'y': y_out,
            'ca': crank_angles,
            'q_wall': q_wall_out,  # Wall heat rate [W] at each output point
            'success': success,
            'message': message,
            'nfev': 0,  # Not tracked by Cantera
            'njev': 0
        }


@dataclass
class CanteraMultizoneSolverParams:
    """Parameters for Cantera multizone solver using N coupled IdealGasReactors."""
    rtol: float = 1.0e-6
    # 1e-12 can trigger CVODES tiny-step collapse (h -> 0) for large
    # low-temperature gasoline mechanisms near ignition onset; 3e-12 is a
    # more robust default with negligible impact on validated 475 K cases.
    atol: float = 3.0e-12
    max_steps: int = 50000
    adiabatic: bool = False
    verbose: bool = True
    nzones: int = 10  # Supported: 10, 20, 40
    pressure_coupling_coeff: float = 0.0  # 0 = auto; expansion_rate_coeff for chain coupling walls


class CanteraMultizoneEngineSolver:
    """
    Multizone engine solver using N standard Cantera IdealGasReactors.

    Each zone is a separate IdealGasReactor with Cantera's analytical Jacobian.
    Zones are connected by coupling walls (expansion_rate_coeff) for pressure
    equilibration, matching the shared-pressure multizone model.

    Architecture:
    - N IdealGasReactors (one per zone)
    - N piston walls: proportional velocity v_i = (V_i/V_total) * v_piston
    - N-1 coupling walls: pressure equilibration between adjacent zones
    - Per-zone heat flux: Ai-weighted Woschni (zone's own T and P)

    This preserves Cantera's analytical chemistry Jacobian (computed in C++),
    making it orders of magnitude faster than approaches that use
    finite-difference Jacobian for large mechanisms (100+ species).
    """

    def __init__(self, geom: GeometryParams, chemistry: Chemistry,
                 heat_transfer: Optional[HeatTransfer] = None,
                 params: CanteraMultizoneSolverParams = None):
        self.geom = geom
        self.chemistry = chemistry
        self.heat_transfer = heat_transfer
        self.params = params or CanteraMultizoneSolverParams()
        self.mechanism_file = chemistry.params.mechanism

    def solve_closed_cycle(self, rpm: float, T_wall: float,
                          ca_start: float, ca_end: float,
                          y0: np.ndarray) -> Dict:
        """
        Solve closed cycle using N coupled Cantera IdealGasReactors.

        Parameters
        ----------
        rpm : float
            Engine speed [rpm]
        T_wall : float
            Wall temperature [K]
        ca_start, ca_end : float
            Crank angle range [deg] (relative to TDC)
        y0 : np.ndarray
            Initial state vector (multizone format):
            [T_bulk, P_bulk, V_total, T_z0, Y_z0, ..., T_zN, Y_zN, Y_bulk]

        Returns
        -------
        Dict with 't', 'y', 'ca', 'q_wall', 'success', 'message', 'nfev', 'njev'
        """
        nsp = self.chemistry.gas.n_species
        nzones = validate_multizone_count(self.params.nzones)
        geom = self.geom
        mass_fracs = zone_mass_fractions(nzones)

        # Extract initial state
        P0 = y0[1]
        V0 = y0[2]

        T_zones0 = np.zeros(nzones)
        Y_zones0 = np.zeros((nzones, nsp))
        for i in range(nzones):
            T_zones0[i] = y0[3 + i * (nsp + 1)]
            Y_zones0[i] = y0[3 + i * (nsp + 1) + 1: 3 + (i + 1) * (nsp + 1)]

        # Time parameters
        omega = rpm * 2 * np.pi / 60
        theta_start = np.deg2rad(ca_start)
        t_end = (np.deg2rad(ca_end) - theta_start) / omega

        # --- Create N gas objects and reactors ---
        # Use MoleReactor for large systems (required for GMRES + preconditioner)
        n_vars_est = nzones * (3 + nsp)
        use_mole_reactor = (n_vars_est > 500)
        ReactorClass = ct.IdealGasMoleReactor if use_mole_reactor else ct.IdealGasReactor

        reactors = []
        for i in range(nzones):
            gas_i = ct.Solution(self.mechanism_file)
            gas_i.TPY = T_zones0[i], P0, Y_zones0[i]
            r_i = ReactorClass(gas_i)
            r_i.volume = V0 * mass_fracs[i]
            reactors.append(r_i)

        zone_masses = np.array([r.mass for r in reactors], dtype=float)
        M_total = float(np.sum(zone_masses))
        zone_mass_fracs = zone_masses / M_total
        heat_mult = zone_heat_loss_multipliers(nzones, zone_mass_fracs)

        # Environment reservoir
        env = ct.Reservoir(ct.Solution('air.yaml'))

        # Fallback geometric heat weights (core=0, wall=max, sum=1.0)
        # Quadratic distribution: Ai[i] = i^2 / sum(j^2, j=0..N-1)
        # Zone 0 (core) has no wall contact, zone N-1 (wall) has maximum
        base_heat_weights = np.zeros(nzones)
        if nzones > 1:
            denom = nzones * (nzones - 1) * (2 * nzones - 1) / 6  # sum of i^2
            for i in range(nzones):
                base_heat_weights[i] = i**2 / denom
        else:
            base_heat_weights[0] = 1.0

        # Piston velocity (total)
        def total_piston_velocity(t):
            theta = omega * t + theta_start
            return -geom.volume_rate(theta, rpm)

        # Heat transfer parameters
        if self.heat_transfer:
            hp = self.heat_transfer.params
        else:
            hp = WoschniParams()
        Up = 2 * geom.stroke * rpm / 60
        if heat_mult is not None:
            fixed_zone_heat_weights = zone_mass_fracs * heat_mult
        else:
            fixed_zone_heat_weights = base_heat_weights.copy()
        wsum = float(np.sum(fixed_zone_heat_weights))
        if wsum <= 0.0:
            fixed_zone_heat_weights = np.full(nzones, 1.0 / nzones)
        else:
            fixed_zone_heat_weights = fixed_zone_heat_weights / wsum

        def _bulk_temperature():
            temps = np.array([r.T for r in reactors], dtype=float)
            return float(np.dot(zone_mass_fracs, temps))

        def _overall_heat_loss_rate(t):
            theta = omega * t + theta_start
            V = geom.cylinder_volume(theta)
            T_bulk = _bulk_temperature()
            P_bulk = reactors[0].thermo.P
            h = (hp.C_scale * hp.C *
                 V**hp.vol_exp *
                 (P_bulk / 1000)**hp.press_exp *
                 T_bulk**hp.temp_exp *
                 (Up + hp.vel_offset)**hp.vel_exp)
            head_a, piston_a, liner_a = geom.surface_area(theta)
            A_total = head_a + piston_a + liner_a
            return h * A_total * (T_bulk - T_wall)

        # --- Create walls ---
        piston_walls = []

        for i in range(nzones):
            # Piston wall: proportional velocity v_i = (V_i / V_total) * v_total
            def _make_piston_v(idx):
                def v(t):
                    V_total = sum(r.volume for r in reactors)
                    V_i = reactors[idx].volume
                    return (V_i / V_total) * total_piston_velocity(t)
                return v

            w = ct.Wall(reactors[i], env, velocity=_make_piston_v(i), A=1.0)

            if not self.params.adiabatic:
                def _make_hf(idx):
                    def hf(t):
                        q_overall = _overall_heat_loss_rate(t)
                        if q_overall <= 0.0:
                            return 0.0
                        return fixed_zone_heat_weights[idx] * q_overall
                    return hf
                w.heat_flux = _make_hf(i)

            piston_walls.append(w)

        # Chain coupling walls for pressure equilibration: zone i ↔ zone i+1.
        # Adjacent zones are coupled only, which limits Jacobian density and
        # avoids the step-underflow stiffness that all-to-all topology produces
        # for small zones (zone 0 = 2% mass with Kodavasal profiles).
        gamma_est = 1.35
        V_TDC = geom.cylinder_volume(0)
        P_TDC_est = P0 * (V0 / V_TDC) ** gamma_est
        K = self.params.pressure_coupling_coeff
        if K <= 0:
            V_TDC_zone_avg = V_TDC / nzones
            target_eigenvalue = 5e6
            K = target_eigenvalue * V_TDC_zone_avg / (gamma_est * P_TDC_est)
        n_coupling_walls = nzones - 1
        coupling_walls = []
        for i in range(nzones - 1):
            w = ct.Wall(reactors[i], reactors[i + 1], A=1.0)
            w.expansion_rate_coeff = K
            coupling_walls.append(w)

        # --- Create ReactorNet ---
        net = ct.ReactorNet(reactors)
        net.rtol = self.params.rtol
        effective_atol = self.params.atol
        if nsp >= 200 and effective_atol < 3.0e-12:
            # Guard against CVODES step underflow in large, stiff low-T
            # chemistry regimes (e.g., 312-species gasoline at 470 K).
            effective_atol = 3.0e-12
        net.atol = effective_atol
        net.max_steps = self.params.max_steps

        # Use GMRES + preconditioner for large systems (avoids dense O(N³) linear solve)
        n_vars = sum(r.n_vars for r in reactors)
        if n_vars > 500:
            net.linear_solver_type = "GMRES"
            if use_mole_reactor:
                preconditioner = ct.AdaptivePreconditioner()
                preconditioner.threshold = 0
                net.preconditioner = preconditioner

        has_precond = n_vars > 500 and use_mole_reactor

        if self.params.verbose:
            lsolver = "GMRES + AdaptivePreconditioner" if has_precond else (
                "GMRES" if n_vars > 500 else "DENSE")
            print(f"\nCantera N-Reactor Multizone Solver:")
            print(f"  Zones: {nzones}, Species: {nsp}")
            print(f"  Reactor type: {'IdealGasMoleReactor' if use_mole_reactor else 'IdealGasReactor'}")
            print(f"  Internal state: {n_vars} variables (analytical Jacobian)")
            print(f"  Linear solver: {lsolver}")
            print(f"  Tolerances: rtol={self.params.rtol:.1e}, atol={effective_atol:.1e}")
            print(f"  Coupling: chain ({n_coupling_walls} walls), K={K:.4e}")
            print(f"  CA: {ca_start:.1f}° to {ca_end:.1f}°")
            print(f"  M_total = {M_total*1000:.4f} g")
            print(f"  Zone mass fractions = {[f'{z:.3f}' for z in zone_mass_fracs]}")
            if heat_mult is not None:
                print(f"  Zone heat multipliers = {[f'{c:.3f}' for c in heat_mult]}")
            else:
                print(f"  Geometric heat weights = {[f'{a:.4f}' for a in base_heat_weights]}")
            if not self.params.adiabatic and self.heat_transfer:
                print(f"  Heat transfer: Woschni C={hp.C}, C_scale={hp.C_scale}")
            print(f"  Integrating...", end="", flush=True)

        # Output arrays
        n_points = 1200
        t_eval = np.linspace(0, t_end, n_points)
        n_out = 3 + nzones * (nsp + 1) + nsp
        y_out = np.zeros((n_out, n_points))
        t_out = np.zeros(n_points)
        q_wall_out = np.zeros(n_points)

        # Store initial state
        t_out[0] = 0.0
        self._store_state(0, reactors, nzones, nsp, zone_masses, M_total,
                         y_out, q_wall_out, piston_walls)

        success = True
        message = "Integration successful"

        try:
            for k in range(1, n_points):
                net.advance(t_eval[k])
                t_out[k] = net.time
                self._store_state(k, reactors, nzones, nsp, zone_masses,
                                 M_total, y_out, q_wall_out, piston_walls)

                if self.params.verbose and k % 100 == 0:
                    ca_now = ca_start + np.rad2deg(omega * t_eval[k])
                    T_pk = np.max(y_out[0, :k+1])
                    # Pressure divergence diagnostic
                    zone_Ps = [r.thermo.P for r in reactors]
                    P_spread = (max(zone_Ps) - min(zone_Ps)) / np.mean(zone_Ps) * 100
                    print(f" CA={ca_now:.0f}°(T={T_pk:.0f},dP={P_spread:.2f}%)",
                          end="", flush=True)

        except Exception as e:
            success = False
            message = f"Integration failed: {str(e)}"
            if self.params.verbose:
                ca_fail = ca_start + np.rad2deg(omega * t_out[max(0, k - 1)])
                print(f"\n  FAILED at CA={ca_fail:.1f}°")
                print(f"  Error: {e}")
            valid = np.where(t_out > 0)[0]
            if len(valid) > 0:
                last = valid[-1] + 1
                t_out = t_out[:last]
                y_out = y_out[:, :last]
                q_wall_out = q_wall_out[:last]

        if success and self.params.verbose:
            print(" done!")
            print(f"  Peak T_bulk: {np.max(y_out[0]):.1f} K")
            print(f"  Peak P: {np.max(y_out[1])/1e5:.2f} bar")
            for i in range(nzones):
                label = "core" if i == 0 else (
                    "wall" if i == nzones - 1 else f"  {i}")
                peak_Ti = np.max(y_out[3 + i * (nsp + 1)])
                print(f"    Zone {i} ({label}) peak T: {peak_Ti:.1f} K")

        crank_angles = ca_start + np.rad2deg(omega * t_out)

        return {
            't': t_out, 'y': y_out, 'ca': crank_angles,
            'q_wall': q_wall_out, 'success': success,
            'message': message, 'nfev': 0, 'njev': 0
        }

    def _store_state(self, idx, reactors, nzones, nsp, zone_masses, M_total,
                    y_out, q_wall_out, piston_walls):
        """Extract multizone state from N reactors into output format.

        Output: [T_bulk, P_bulk, V_total, T_z0, Y_z0, ..., T_zN, Y_zN, Y_bulk]
        """
        T_bulk_sum = 0.0
        Y_bulk = np.zeros(nsp)
        V_total = 0.0
        Q_total = 0.0

        for i in range(nzones):
            r = reactors[i]
            T_i = r.T
            Y_i = r.Y
            V_total += r.volume
            T_bulk_sum += zone_masses[i] * T_i
            Y_bulk += zone_masses[i] * Y_i

            y_out[3 + i * (nsp + 1), idx] = T_i
            y_out[3 + i * (nsp + 1) + 1: 3 + (i + 1) * (nsp + 1), idx] = Y_i

            Q_total += piston_walls[i].heat_rate

        y_out[0, idx] = T_bulk_sum / M_total
        y_out[1, idx] = reactors[0].thermo.P
        y_out[2, idx] = V_total
        y_out[3 + nzones * (nsp + 1):, idx] = Y_bulk / M_total
        q_wall_out[idx] = Q_total


def check_cantera_solver_available() -> bool:
    """Check if Cantera ReactorNet solver is available."""
    try:
        import cantera as ct
        gas = ct.Solution('gri30.yaml')
        r = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([r])
        return True
    except Exception:
        return False
