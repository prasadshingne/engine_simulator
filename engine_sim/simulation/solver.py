"""ODE solver for engine simulation."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Callable
from scipy.integrate import solve_ivp
from tqdm import tqdm
import cantera as ct

from ..engine.geometry import GeometryParams
from ..engine.heat_transfer import HeatTransfer
from ..models.chemistry import Chemistry
from .multizone_profiles import (
    validate_multizone_count,
    zone_heat_loss_multipliers,
    zone_mass_fractions,
)

# Try to import CVODE solver (SUNDIALS) for better handling of stiff systems
try:
    from .solver_ida import MultizoneCVODE, check_cvode_available
    HAS_CVODE = check_cvode_available()
except ImportError:
    HAS_CVODE = False

# Try to import Cantera ReactorNet solver for large mechanisms
try:
    from .solver_cantera import (
        CanteraEngineSolver, CanteraSolverParams,
        CanteraMultizoneEngineSolver, CanteraMultizoneSolverParams,
        check_cantera_solver_available
    )
    HAS_CANTERA_SOLVER = check_cantera_solver_available()
except ImportError:
    HAS_CANTERA_SOLVER = False

# Threshold for switching to Cantera solver (species count)
CANTERA_SOLVER_THRESHOLD = 100

@dataclass
class SolverParams:
    """Solver parameters."""
    method: str = "LSODA"     # Integration method
    rtol: float = 1.0e-8      # Relative tolerance
    atol: float = 1.0e-10     # Absolute tolerance
    max_step: float = 1.0e-3  # Maximum step size
    first_step: float = 1.0e-6  # First step size
    adiabatic: bool = False   # Whether to run in adiabatic mode
    min_temp: float = 200.0   # Minimum allowed temperature [K]
    max_temp: float = 3500.0  # Maximum allowed temperature [K]
    min_press: float = 1e4    # Minimum allowed pressure [Pa]
    max_press: float = 1e8    # Maximum allowed pressure [Pa]
    min_mass_fraction: float = -1e-10  # Minimum allowed mass fraction
    mass_fraction_threshold: float = 1e-12  # Threshold below which to set to zero
    max_rate_limit: float = 1000.0  # Maximum allowed fractional change in mass fraction per step
    show_progress: bool = True  # Whether to show progress bar
    model_type: str = "single"  # Model type (single or multi-zone)
    nzones: int = 10  # Number of zones (supported: 10, 20, 40)
    use_manual_solver: bool = False  # Use legacy scipy ODE solver instead of Cantera ReactorNet

    @classmethod
    def from_yaml(cls, config: Dict) -> 'SolverParams':
        """Create SolverParams from YAML config."""
        solver_config = config['solver']
        return cls(
            method=solver_config['method'],
            rtol=solver_config['rtol'],
            atol=solver_config['atol'],
            max_step=solver_config['max_step'],
            first_step=solver_config['first_step'],
            adiabatic=solver_config['adiabatic'],
            # Keep defaults for other parameters
            min_temp=200.0,
            max_temp=3500.0,
            min_press=1e4,
            max_press=1e8,
            min_mass_fraction=-1e-10,
            mass_fraction_threshold=1e-12,
            max_rate_limit=1000.0,
            show_progress=True,
            model_type=solver_config['model_type'],
            nzones=solver_config['nzones'],
            use_manual_solver=bool(solver_config.get('use_manual_solver', False)),
        )

class EngineSolver:
    """Engine cycle ODE solver."""
    
    def __init__(self, geom: GeometryParams, heat_transfer: HeatTransfer,
                 chemistry: Chemistry, params: SolverParams = SolverParams()):
        """
        Initialize solver.
        
        Parameters
        ----------
        geom : GeometryParams
            Engine geometry
        heat_transfer : HeatTransfer
            Heat transfer model
        chemistry : Chemistry
            Chemistry interface
        params : SolverParams, optional
            Solver parameters
        """
        self.geom = geom
        self.heat_transfer = heat_transfer
        self.chemistry = chemistry
        self.params = params
        self.progress_bar = None

        # Intermediate result storage for partial result plotting
        self.intermediate_results = {'t': [], 'y': [], 'ca': []}
        self.save_interval = 100  # Save every N ODE evaluations
        self.eval_count = 0
        self.save_path = None
        self.last_t = None
        self.t_eval = None
        self.gamma = 1.35
        self.ca_start = None
        self.theta_start = None
        self.rpm = None

        # Store reference conditions
        self.p_ref = None
        self.T_ref = None
        self.V_ref = None
        
    def _calculate_motored_pressure(self, crank_angle: float) -> float:
        """Calculate motored pressure assuming polytropic compression."""
        V = self.geom.cylinder_volume(crank_angle)
        return self.p_ref * (self.V_ref/V)**self.gamma
    
    def _ode_system(self, t: float, y: np.ndarray, rpm: float, T_wall: float) -> np.ndarray:
        """Calculate derivatives for engine ODE system."""
        # Save intermediate results for partial plotting
        self.eval_count += 1
        if self.save_path and self.eval_count % self.save_interval == 0:
            omega = self.rpm * 2 * np.pi / 60
            ca = self.ca_start + np.rad2deg(omega * t)
            self.intermediate_results['t'].append(t)
            self.intermediate_results['y'].append(y.copy())
            self.intermediate_results['ca'].append(ca)
            # Periodically save to file
            if len(self.intermediate_results['t']) % 10 == 0:
                np.savez(self.save_path,
                         t=np.array(self.intermediate_results['t']),
                         y=np.array(self.intermediate_results['y']).T,
                         ca=np.array(self.intermediate_results['ca']))

        if self.params.model_type == "single":
            return self._ode_system_single(t, y, rpm, T_wall)
        else:
            return self._ode_system_multi(t, y, rpm, T_wall)
            
    def _ode_system_single(self, t: float, y: np.ndarray, rpm: float, T_wall: float) -> np.ndarray:
        """Calculate derivatives for single-zone engine ODE system."""
        # Extract state variables
        T = y[0]              # Temperature [K]
        V = y[1]              # Volume [m³]
        P = y[2]              # Pressure [Pa]
        m = y[3]              # Mass [kg]
        Y = y[4:]            # Species mass fractions [-]
        
        # Quick bounds check before expensive operations
        if not (200 <= T <= 3500 and 1e4 <= P <= 1e8):
            if T < 200 or T > 3500:
                raise ValueError(f"Temperature {T} K out of bounds [200, 3500]")
            else:
                raise ValueError(f"Pressure {P} Pa out of bounds [1e4, 1e8]")
        
        # Calculate crank angle and volume change
        theta = (rpm * 2 * np.pi / 60) * t + self.theta_start  # [rad]
        # volume_rate() returns bore_area * dx/dt = -dV/dt, negate to get actual dV/dt
        dVdt = -self.geom.volume_rate(theta, rpm)

        # Update gas state and get properties
        self.chemistry.update_state(T, P, Y)
        props = self.chemistry.get_properties()
        
        # Get reaction rates and heat release
        mdot, Q_chem = self.chemistry.get_reaction_rates()  # Q_chem in [W/m³]
        ydot = mdot / props['rho']
        
        # Calculate heat transfer
        if self.params.adiabatic:
            Q_wall = 0.0
        else:
            p_motored = self._calculate_motored_pressure(theta)
            Q_wall, _ = self.heat_transfer.heat_transfer_rate(
                theta, rpm, P, T, p_motored,
                self.p_ref, self.T_ref, self.V_ref, T_wall
            )
        
        # Energy equation
        cv = props['cv']
        dTdt = 1.0/(m * cv) * (-P*dVdt + Q_chem*V - Q_wall)
        
        # Pressure equation from ideal gas law
        dPdt = P * (dTdt/T - dVdt/V)
        
        # Combine derivatives
        dydt = np.zeros_like(y)
        dydt[0] = dTdt
        dydt[1] = dVdt
        dydt[2] = dPdt
        dydt[4:] = ydot
        
        # Update progress bar if enabled
        if self.progress_bar is not None:
            current_ca = np.rad2deg(theta)  # theta already includes theta_start
            progress = current_ca - self.ca_start  # Progress from start
            if progress > self.progress_bar.n:  # Only update if we've made progress
                self.progress_bar.update(progress - self.progress_bar.n)
                self.progress_bar.set_postfix({'CA': f"{current_ca:.1f}°"})

        return dydt
        
    def _ode_system_multi(self, t: float, y: np.ndarray, rpm: float, T_wall: float) -> np.ndarray:
        """Calculate derivatives for multi-zone engine ODE system.

        State vector structure (matching Matlab HCCI_sim_dec038.m):
        [T, P, V, T_zone1, Y_zone1, ..., T_zoneN, Y_zoneN, Y_bulk]

        Note: Mass M is constant for closed cycle (no valve flow)
        """
        # CRITICAL: For non-adiabatic, clip state vector to physical bounds BEFORE checking for NaN
        # This prevents Jacobian perturbations from creating unphysical values
        # For adiabatic cases, skip this as they work fine without it
        if not self.params.adiabatic:
            y_clipped = y.copy()

            # Temperature bounds: 100 K - 5000 K
            y_clipped[0] = np.clip(y_clipped[0], 100.0, 5000.0)  # Bulk T

            # Pressure bounds: 0.1 bar - 500 bar
            y_clipped[1] = np.clip(y_clipped[1], 1.0e4, 5.0e7)  # Bulk P

            # Volume bounds: positive
            if y_clipped[2] < 1.0e-10:
                y_clipped[2] = 1.0e-10

            # Get dimensions
            nsp = self.chemistry.gas.n_species
            nzones = validate_multizone_count(self.params.nzones)

            # Clip zone temperatures and mass fractions
            for i in range(nzones):
                # Zone temperature
                zone_T_idx = 3 + i*(nsp+1)
                y_clipped[zone_T_idx] = np.clip(y_clipped[zone_T_idx], 100.0, 5000.0)

                # Zone species mass fractions
                zone_Y_start = 3 + i*(nsp+1) + 1
                zone_Y_end = 3 + (i+1)*(nsp+1)
                y_clipped[zone_Y_start:zone_Y_end] = np.clip(
                    y_clipped[zone_Y_start:zone_Y_end],
                    self.params.min_mass_fraction,
                    1.0
                )
                # Renormalize
                Y_sum = np.sum(y_clipped[zone_Y_start:zone_Y_end])
                if Y_sum > 1.0e-10:
                    y_clipped[zone_Y_start:zone_Y_end] /= Y_sum

            # Clip bulk species mass fractions
            bulk_Y_start = 3 + nzones*(nsp+1)
            y_clipped[bulk_Y_start:] = np.clip(
                y_clipped[bulk_Y_start:],
                self.params.min_mass_fraction,
                1.0
            )
            Y_bulk_sum = np.sum(y_clipped[bulk_Y_start:])
            if Y_bulk_sum > 1.0e-10:
                y_clipped[bulk_Y_start:] /= Y_bulk_sum

            # Now check for NaN in clipped state
            if not np.all(np.isfinite(y_clipped)):
                theta = (rpm * 2 * np.pi / 60) * t
                ca = self.ca_start + np.rad2deg(theta)
                nan_indices = np.where(~np.isfinite(y_clipped))[0]
                print(f"\n=== NaN in input state at CA={ca:.1f}° ===", flush=True)
                print(f"NaN at indices: {nan_indices[:10]}", flush=True)  # First 10
                print(f"y_clipped[0:5] = {y_clipped[0:5]}", flush=True)
                raise ValueError(f"NaN in input state vector at CA={ca:.1f}°")

            # Use clipped state for all subsequent calculations
            y = y_clipped
        else:
            # For adiabatic, just check for NaN in input
            if not np.all(np.isfinite(y)):
                theta = (rpm * 2 * np.pi / 60) * t
                ca = self.ca_start + np.rad2deg(theta)
                nan_indices = np.where(~np.isfinite(y))[0]
                print(f"\n=== NaN in input state at CA={ca:.1f}° ===", flush=True)
                print(f"NaN at indices: {nan_indices[:10]}", flush=True)
                print(f"y[0:5] = {y[0:5]}", flush=True)
                raise ValueError(f"NaN in input state vector at CA={ca:.1f}°")

        # Extract state variables (Matlab lines 31-37)
        T = y[0]              # Bulk Temperature [K]
        P = y[1]              # Bulk Pressure [Pa]
        V = y[2]              # Total Volume [m³]
        # M is CONSTANT for closed cycle - stored in self.M_total
        M = self.M_total

        # Get number of species and zones
        nsp = self.chemistry.gas.n_species
        nzones = validate_multizone_count(self.params.nzones)
        zone_mass_fracs = zone_mass_fractions(nzones)
        zone_heat_mult = zone_heat_loss_multipliers(nzones, zone_mass_fracs)

        # Extract zone temperatures and compositions (Matlab lines 80-82)
        tzone = np.zeros(nzones)
        yzone = np.zeros((nsp, nzones))
        for i in range(nzones):
            tzone[i] = y[3 + i*(nsp+1)]  # Changed from y[4+...] to y[3+...]
            yzone[:,i] = y[3 + i*(nsp+1) + 1:3 + (i+1)*(nsp+1)]  # Changed indexing
            # Ensure mass fractions stay within bounds
            yzone[:,i] = np.clip(yzone[:,i], self.params.min_mass_fraction, 1.0)
            yzone[:,i] /= np.sum(yzone[:,i])  # Renormalize

        # Safety check: ensure temperatures stay physical
        min_tzone = np.min(tzone)
        if min_tzone < 100.0:
            theta = (rpm * 2 * np.pi / 60) * t
            ca = self.ca_start + np.rad2deg(theta)
            # Print diagnostic information
            print(f"\n=== DIAGNOSTIC INFO AT FAILURE ===", flush=True)
            print(f"CA = {ca:.1f}°, t = {t:.6f} s", flush=True)
            print(f"Bulk: P={P/1e5:.2f} bar, V={V*1e6:.2f} cm³, M={M*1000:.3f} g", flush=True)
            print(f"Zone temperatures: {tzone}", flush=True)
            raise ValueError(f"Zone temperature too low ({min_tzone:.1f} K) at CA={ca:.1f}°")

        # Extract bulk composition (Matlab line 41)
        Ybulk = y[3 + nzones*(nsp+1):]  # Changed from y[4+...] to y[3+...]

        # Calculate crank angle and volume change
        theta = (rpm * 2 * np.pi / 60) * t + self.theta_start  # [rad]
        # volume_rate() returns bore_area * dx/dt = -dV/dt, negate to get actual dV/dt
        dVdt = -self.geom.volume_rate(theta, rpm)

        # Update progress bar if enabled
        if self.progress_bar is not None:
            current_ca = np.rad2deg(theta)  # theta already includes theta_start
            progress = current_ca - self.ca_start
            if progress > self.progress_bar.n:
                self.progress_bar.update(progress - self.progress_bar.n)
                self.progress_bar.set_postfix({
                    'CA': f"{current_ca:.1f}°",
                    'Zones': nzones
                })

        # Calculate heat transfer (zero if adiabatic)
        Q_wall = 0.0
        if not self.params.adiabatic:
            p_motored = self._calculate_motored_pressure(theta)
            Q_wall, _ = self.heat_transfer.heat_transfer_rate(
                theta, rpm, P, T, p_motored,
                self.p_ref, self.T_ref, self.V_ref, T_wall
            )

        # Initialize arrays (following Matlab Cycle_ode_multi.asv structure)
        Ygen = np.zeros((nsp, nzones))
        unsp = np.zeros((nsp, nzones))  # Internal energy per species [J/kg]
        hnsp = np.zeros((nsp, nzones))  # Enthalpy per species [J/kg]
        Vi = np.zeros(nzones)
        Mi = np.zeros(nzones)
        Cvi = np.zeros(nzones)
        Cpi = np.zeros(nzones)
        Ri = np.zeros(nzones)

        # Get species-specific properties (constant across all zones)
        # Just need to set state once to get MW
        self.chemistry.gas.TPY = T, P, Ybulk
        MW = self.chemistry.gas.molecular_weights  # [kg/kmol]
        Rk = ct.gas_constant / MW  # [J/kg/K] - Matlab line 90

        # Accumulators for bulk equations
        sumgen = 0.0      # Matlab line 114: sum(mi*Ygen'*unsp)
        sumRYdot = 0.0    # Matlab line 116: sum(Rk'*Ygen)
        sumRY = 0.0       # Matlab line 119: sum(Rk'*Y)

        # Per-zone calculations (Matlab lines 79-122)
        for i in range(nzones):
            Mi[i] = M * zone_mass_fracs[i]

            try:
                # Set zone state (Matlab line 85)
                self.chemistry.gas.TPY = tzone[i], P, yzone[:,i]

                # Get thermodynamic properties (Matlab lines 86-89)
                Cpi[i] = self.chemistry.gas.cp_mass
                Cvi[i] = self.chemistry.gas.cv_mass
                Ri[i] = Cpi[i] - Cvi[i]

                # Zone volume (Matlab line 93)
                Vi[i] = Mi[i] / self.chemistry.gas.density

                # Species properties (Matlab lines 90-92)
                # unsp = ((h_k/(R*T))*T - T)*Rk = u_k (internal energy per unit mass)
                # hnsp = (h_k/(R*T))*T*Rk = h_k (enthalpy per unit mass)
                # where h_k/(R*T) is dimensionless, Rk = R/MW
                h_RT = self.chemistry.gas.standard_enthalpies_RT  # Dimensionless: H_k/(R*T)

                # unsp = (h_RT * T - T) * Rk = R*T*(h_RT - 1)/MW [J/kg]
                # This is internal energy: u = h - RT
                unsp[:,i] = ct.gas_constant * tzone[i] * (h_RT - 1.0) / MW

                # hnsp = h_RT * T * Rk = R*T*h_RT/MW [J/kg]
                # This is enthalpy: h
                hnsp[:,i] = ct.gas_constant * tzone[i] * h_RT / MW

                # Species production rates (Matlab line 97)
                wdot = self.chemistry.gas.net_production_rates  # [kmol/m³/s]
                prody = wdot * MW  # [kg/m³/s]
                Ygen[:,i] = prody / self.chemistry.gas.density  # [1/s]

                # Check for non-finite chemistry results
                if not np.all(np.isfinite(Ygen[:,i])):
                    # If chemistry gives bad results, use zero production rates
                    Ygen[:,i] = np.zeros(nsp)

            except Exception as e:
                # If Cantera fails (e.g., due to bad state from Jacobian perturbation),
                # use safe default values and zero production rates
                theta = (rpm * 2 * np.pi / 60) * t
                ca = self.ca_start + np.rad2deg(theta)
                # Use approximate values to allow continuation
                Cpi[i] = 1000.0  # Approximate cp
                Cvi[i] = 700.0   # Approximate cv
                Ri[i] = 300.0    # Approximate R
                Vi[i] = V / nzones  # Approximate zone volume
                unsp[:,i] = np.zeros(nsp)  # Zero internal energy changes
                hnsp[:,i] = np.zeros(nsp)  # Zero enthalpy changes
                Ygen[:,i] = np.zeros(nsp)  # Zero production rates

            # Accumulate for bulk equations (Matlab lines 114, 116, 119)
            sumgen += Mi[i] * np.dot(Ygen[:,i], unsp[:,i])
            sumRYdot += np.dot(Rk, Ygen[:,i])
            sumRY += np.dot(Rk, yzone[:,i])

        # Bulk gas properties for temperature equation (Matlab lines 42-46)
        try:
            self.chemistry.gas.TPY = T, P, Ybulk
            CVbulk = self.chemistry.gas.cv_mass
            CPbulk = self.chemistry.gas.cp_mass
            Rbulk = CPbulk - CVbulk  # Initial value before accumulation (Matlab line 45)
        except Exception as e:
            # Use safe defaults if bulk properties fail
            CVbulk = 700.0  # Approximate cv for air
            CPbulk = 1000.0  # Approximate cp for air
            Rbulk = 300.0  # Approximate R for air

        #  IMPORTANT: Matlab accumulates Rbulk over zones (line 120)
        # Even though this seems dimensionally inconsistent, match it exactly
        for i in range(nzones):
            Rbulk += Mi[i] * np.dot(Rk, yzone[:,i])  # Matlab line 120: Rbulk = Rbulk + Rk(:,i)'*yzone(:,i).*init.mzone(i)

        # Bulk temperature derivative (Matlab line 88-89)
        # dT/dt = 1/(M*CV)*(-M*R*T/V*dV/dt - q - sumgen)
        compression_work = -M * Rbulk * T / V * dVdt
        dTdt = (1.0 / (M * CVbulk)) * (compression_work - Q_wall - sumgen)

        # Pressure derivative (Matlab line 95) - CRITICAL: Use ideal gas law form, NOT composition-dependent!
        # The composition-dependent form (line 93 in Matlab) is commented out in the working version
        # dP/dt = P*((dTdt/T) - (dVdt/V))  - Simple ideal gas law
        if abs(T) < 1.0:
            theta = (rpm * 2 * np.pi / 60) * t
            ca = self.ca_start + np.rad2deg(theta)
            print(f"\n=== DIAGNOSTIC: T too low ===", flush=True)
            print(f"CA={ca:.1f}°, T={T:.1f} K, P={P/1e5:.2f} bar", flush=True)
            print(f"dTdt={dTdt:.2e}, dVdt={dVdt:.2e}", flush=True)
            raise ValueError(f"Bulk temperature too low: {T} K at CA={ca:.1f}°")
        if abs(V) < 1e-10:
            raise ValueError(f"Volume too small: {V} m³")

        # Use IDEAL GAS LAW form (Matlab line 95), not composition-dependent form
        dPdt = P * ((dTdt / T) - (dVdt / V))

        # Zone-wise heat loss weighting.
        if self.params.adiabatic or abs(Q_wall) < 1e-16:
            zone_heat_weights = np.zeros(nzones)
        else:
            if nzones > 1:
                denom = nzones * (nzones - 1) * (2 * nzones - 1) / 6
                geometric_w = np.array([(i**2) / denom for i in range(nzones)])
            else:
                geometric_w = np.array([1.0])

            if zone_heat_mult is not None:
                # Equation (7): q_i = C_i * (m_i/m_tot) * q_overall
                zone_heat_weights = zone_mass_fracs * zone_heat_mult
                wsum = float(np.sum(zone_heat_weights))
                if wsum <= 0.0:
                    zone_heat_weights = geometric_w
                else:
                    zone_heat_weights = zone_heat_weights / wsum
            else:
                zone_heat_weights = geometric_w

        # Zone temperature derivatives (Matlab lines 145-150)
        dTidt_array = np.zeros(nzones)
        for i in range(nzones):
            # dTi/dt = 1/(mi*cpi)*(dP/dt*Vi - q*Ai - mi*Ygen'*hnsp)  (Matlab line 146)
            chem_term = Mi[i] * np.dot(Ygen[:,i], hnsp[:,i])
            dTidt_array[i] = (1.0 / (Mi[i] * Cpi[i])) * (
                dPdt * Vi[i] - Q_wall * zone_heat_weights[i] - chem_term
            )

            # Check for non-finite values
            if not np.isfinite(dTidt_array[i]):
                theta = (rpm * 2 * np.pi / 60) * t
                ca = self.ca_start + np.rad2deg(theta)
                print(f"\n=== Zone {i} dTidt NaN/Inf ===", flush=True)
                print(f"CA={ca:.1f}°, dTidt[{i}]={dTidt_array[i]}", flush=True)
                print(f"Mi={Mi[i]}, Cpi={Cpi[i]}, Vi={Vi[i]}", flush=True)
                print(f"dPdt={dPdt}, Q_wall={Q_wall}, w_i={zone_heat_weights[i]}", flush=True)
                print(f"chem_term={chem_term}", flush=True)
                print(f"max|Ygen|={np.max(np.abs(Ygen[:,i]))}, max|hnsp|={np.max(np.abs(hnsp[:,i]))}", flush=True)
                raise ValueError(f"Non-finite zone temperature derivative")

        # Calculate bulk composition as mass-weighted average (Matlab line 121-124)
        dYdt = np.zeros(nsp)
        for i in range(nzones):
            dYdt += Mi[i] * Ygen[:,i]
        dYdt /= M

        # Initialize derivative vector
        dydt = np.zeros_like(y)

        # Check for NaN or Inf in critical derivatives
        if not np.isfinite(dTdt) or not np.isfinite(dPdt):
            theta = (rpm * 2 * np.pi / 60) * t
            ca = self.ca_start + np.rad2deg(theta)
            print(f"\n=== NaN/Inf DIAGNOSTIC ===", flush=True)
            print(f"CA={ca:.1f}°, t={t:.6e} s", flush=True)
            print(f"dTdt={dTdt}, dPdt={dPdt}, dVdt={dVdt}", flush=True)
            print(f"T={T}, P={P}, V={V}", flush=True)
            print(f"sumgen={sumgen}, sumRY={sumRY}, sumRYdot={sumRYdot}", flush=True)
            print(f"CVbulk={CVbulk}, Rbulk={Rbulk}", flush=True)
            raise ValueError(f"Non-finite derivative at CA={ca:.1f}°: dTdt={dTdt}, dPdt={dPdt}")

        # Set bulk derivatives (Matlab line 152: x = [dTdt;dPdt;vdt;dm])
        # NOTE: M is constant for closed cycle, so dM/dt not included in state vector
        dydt[0] = dTdt
        dydt[1] = dPdt
        dydt[2] = dVdt
        # dydt[3] removed - no dM/dt

        # Set zone derivatives (Matlab lines 154-157)
        for i in range(nzones):
            dydt[3 + i*(nsp+1)] = dTidt_array[i]  # Changed from dydt[4+...] to dydt[3+...]
            # Zone composition derivatives
            dydt[3 + i*(nsp+1) + 1:3 + (i+1)*(nsp+1)] = Ygen[:,i]  # Changed indexing

        # Set bulk composition derivatives (Matlab line 160)
        dydt[3 + nzones*(nsp+1):] = dYdt  # Changed from dydt[4+...] to dydt[3+...]

        return dydt
    
    def _compute_jac_sparsity(self, y0: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian sparsity pattern for multizone system.

        This tells the solver which Jacobian elements are non-zero,
        dramatically reducing the number of function evaluations needed
        for numerical Jacobian approximation.

        For multizone: state = [T, P, V, T_z1, Y_z1, ..., T_zN, Y_zN, Y_bulk]

        Key coupling patterns:
        - Bulk (T,P,V) affects all zones (shared pressure)
        - Each zone affects bulk through mass-weighted averaging
        - Zone species are coupled through chemistry within that zone
        - Zones are weakly coupled through bulk temperature/pressure
        """
        n = len(y0)

        if self.params.model_type == "single":
            # For single zone, assume dense Jacobian (it's small anyway)
            return np.ones((n, n), dtype=int)

        # Initialize sparse pattern (mostly zeros)
        jac_pattern = np.zeros((n, n), dtype=int)

        # Get dimensions
        nsp = self.chemistry.gas.n_species
        nzones = validate_multizone_count(self.params.nzones)

        # ===== Bulk variables (T, P, V) =====
        # All bulk variables couple to each other
        jac_pattern[0:3, 0:3] = 1

        # Bulk variables affect all zone temperatures and species
        for i in range(nzones):
            zone_start = 3 + i*(nsp+1)
            zone_end = 3 + (i+1)*(nsp+1)
            jac_pattern[zone_start:zone_end, 0:3] = 1  # Zones depend on bulk
            jac_pattern[0:3, zone_start:zone_end] = 1  # Bulk depends on zones

        # Bulk variables affect bulk species
        bulk_species_start = 3 + nzones*(nsp+1)
        jac_pattern[0:3, bulk_species_start:] = 1
        jac_pattern[bulk_species_start:, 0:3] = 1

        # ===== Zone variables =====
        for i in range(nzones):
            zone_start = 3 + i*(nsp+1)
            zone_end = 3 + (i+1)*(nsp+1)

            # Within zone: temperature and all species couple (dense block)
            jac_pattern[zone_start:zone_end, zone_start:zone_end] = 1

            # Zone affects bulk species (mass-weighted average)
            jac_pattern[bulk_species_start:, zone_start:zone_end] = 1
            jac_pattern[zone_start:zone_end, bulk_species_start:] = 1

        return jac_pattern

    def solve_closed_cycle(self, rpm: float, T_wall: float,
                          ca_start: float, ca_end: float,
                          y0: np.ndarray) -> Dict:
        """
        Solve closed cycle from IVC to EVO.

        Parameters
        ----------
        rpm : float
            Engine speed [rpm]
        T_wall : float
            Wall temperature [K]
        ca_start : float
            Start crank angle [deg]
        ca_end : float
            End crank angle [deg]
        y0 : np.ndarray
            Initial state vector

        Returns
        -------
        Dict
            Solution dictionary with time, states, and crank angles
        """
        if self.params.model_type == "multi":
            self.params.nzones = validate_multizone_count(self.params.nzones)

        # Dispatch to Cantera ReactorNet solver unless the manual solver is explicitly requested
        n_species = self.chemistry.gas.n_species
        use_cantera_reactor = (
            HAS_CANTERA_SOLVER and
            self.params.model_type == "single" and
            not self.params.use_manual_solver
        )

        if use_cantera_reactor:
            print(f"\nUsing Cantera ReactorNet solver ({n_species} species)")
            # Cantera CVODES requires tight tolerances; clamp GUI values (1e-4/1e-10)
            ct_rtol = min(self.params.rtol, 1e-6)
            ct_atol = min(self.params.atol, 1e-12)
            cantera_solver = CanteraEngineSolver(
                geom=self.geom,
                chemistry=self.chemistry,
                heat_transfer=self.heat_transfer if not self.params.adiabatic else None,
                params=CanteraSolverParams(
                    rtol=ct_rtol,
                    atol=ct_atol,
                    adiabatic=self.params.adiabatic,
                    verbose=self.params.show_progress
                )
            )
            return cantera_solver.solve_closed_cycle(
                rpm=rpm, T_wall=T_wall,
                ca_start=ca_start, ca_end=ca_end,
                y0=y0
            )

        # Check if we should use Cantera multizone solver
        use_cantera_multizone = (
            HAS_CANTERA_SOLVER and
            self.params.model_type == "multi" and
            not self.params.use_manual_solver
        )

        if use_cantera_multizone:
            print(f"\nUsing Cantera Multizone ReactorNet solver ({n_species} species, {self.params.nzones} zones)")
            # Cantera CVODES requires tight tolerances; clamp GUI values (1e-4/1e-10).
            # solver_cantera.py then applies the nsp-dependent atol guard (>= 3e-12).
            mz_rtol = min(self.params.rtol, 1e-6)
            mz_atol = min(self.params.atol, 1e-12)
            cantera_mz_solver = CanteraMultizoneEngineSolver(
                geom=self.geom,
                chemistry=self.chemistry,
                heat_transfer=self.heat_transfer if not self.params.adiabatic else None,
                params=CanteraMultizoneSolverParams(
                    rtol=mz_rtol,
                    atol=mz_atol,
                    adiabatic=self.params.adiabatic,
                    verbose=self.params.show_progress,
                    nzones=self.params.nzones
                )
            )
            return cantera_mz_solver.solve_closed_cycle(
                rpm=rpm, T_wall=T_wall,
                ca_start=ca_start, ca_end=ca_end,
                y0=y0
            )

        # Store crank angle range for progress bar
        self.ca_start = ca_start
        self.rpm = rpm

        # Convert crank angles to radians
        theta_start = np.deg2rad(ca_start)
        theta_end = np.deg2rad(ca_end)
        self.theta_start = theta_start

        # Calculate time span
        omega = rpm * 2 * np.pi / 60  # [rad/s]
        t_span = (0, (theta_end - theta_start)/omega)

        # Store reference conditions at IVC
        # Single zone: [T, V, P, m, Y...]
        # Multi zone:  [T, P, V, T_i, Y_i..., Y_bulk] (M is constant, not in state)
        if self.params.model_type == "single":
            self.T_ref = y0[0]
            self.V_ref = y0[1]
            self.p_ref = y0[2]
        else:
            self.T_ref = y0[0]
            self.p_ref = y0[1]
            self.V_ref = y0[2]

            # For multizone, compute total mass from initial conditions (Matlab line 92: init.m=init.V*density(gas))
            # Extract bulk composition from state vector
            nsp = self.chemistry.gas.n_species
            nzones = validate_multizone_count(self.params.nzones)
            Ybulk_init = y0[3 + nzones*(nsp+1):]

            # Set gas state and compute initial mass
            self.chemistry.gas.TPY = self.T_ref, self.p_ref, Ybulk_init
            rho_init = self.chemistry.gas.density
            self.M_total = self.V_ref * rho_init  # Constant mass for closed cycle
        
        # Create time evaluation points (increased for smoothness)
        self.t_eval = np.linspace(t_span[0], t_span[1], 1200)  # 0.3°/pt for smooth zoom
        
        # Initialize progress bar if requested
        if self.params.show_progress:
            print("\nSolving engine cycle:")
            # Calculate total crank angle range
            total_ca = ca_end - ca_start
            self.progress_bar = tqdm(total=total_ca, initial=0,
                                   desc="Progress", unit="°CA",
                                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.1f}°CA [{elapsed}<{remaining}, {rate_fmt}]')

        # Enable intermediate result saving for multizone
        if self.params.model_type == "multi":
            import tempfile
            self.save_path = tempfile.mktemp(suffix='_partial.npz', prefix='multizone_')
            self.intermediate_results = {'t': [], 'y': [], 'ca': []}
            self.eval_count = 0
            print(f"Saving intermediate results to: {self.save_path}")

        # Compute Jacobian sparsity pattern for non-adiabatic cases ONLY
        # (Adiabatic cases work fine without it)
        jac_sparsity = None
        if self.params.model_type == "multi" and self.params.nzones > 1 and not self.params.adiabatic:
            print(f"Computing Jacobian sparsity pattern for {len(y0)} state variables...")
            jac_sparsity = self._compute_jac_sparsity(y0)
            nnz = np.sum(jac_sparsity)
            sparsity_pct = 100.0 * nnz / (len(y0)**2)
            print(f"  Jacobian: {len(y0)}×{len(y0)} matrix, {nnz} non-zero ({sparsity_pct:.1f}%)")

        # Choose solver method
        # For multizone with heat transfer (non-adiabatic), prefer CVODE if available, else Radau
        method = self.params.method
        rtol_adjusted = self.params.rtol
        atol_adjusted = self.params.atol
        use_cvode = False

        # Use CVODE (SUNDIALS) when available, fall back to scipy otherwise
        if HAS_CVODE and method in ["BDF", "Radau", "CVODE", "IDA", "LSODA"]:
            method = "CVODE"
            use_cvode = True
            print(f"Using CVODE solver (SUNDIALS)")
        elif method in ["CVODE", "IDA"]:
            # CVODE requested but not available, fall back to LSODA
            method = "LSODA"
            print(f"CVODE not available, falling back to LSODA")
            print(f"Note: Install scikits.odes for better performance: conda install -c conda-forge scikits.odes sundials")

        # Respect configured tolerances directly for both single- and multi-zone.

        try:
            if use_cvode and HAS_CVODE:
                # Use CVODE solver from SUNDIALS (similar to Matlab's ode15s)
                print(f"Initializing CVODE solver...")

                # Create CVODE solver
                cvode_solver = MultizoneCVODE(
                    ode_function=lambda t, y: self._ode_system(t, y, rpm, T_wall),
                    n_states=len(y0),
                    verbose=True
                )

                # Solve with CVODE
                solution = cvode_solver.solve(
                    y0=y0,
                    t_span=t_span,
                    t_eval=self.t_eval,
                    rtol=rtol_adjusted,
                    atol=atol_adjusted,
                    max_steps=50000,
                    max_order=5,  # BDF order 1-5
                    lmm_type='cvode',  # Use CVODE (BDF) for stiff systems
                    max_step_size=self.params.max_step
                )

            else:
                # Use scipy's solve_ivp
                solution = solve_ivp(
                    fun=lambda t, y: self._ode_system(t, y, rpm, T_wall),
                    t_span=t_span,
                    y0=y0,
                    method=method,
                    t_eval=self.t_eval,
                    rtol=rtol_adjusted,
                    atol=atol_adjusted,
                    max_step=self.params.max_step,
                    first_step=self.params.first_step,
                    jac_sparsity=jac_sparsity,  # Critical for large multizone systems!
                    vectorized=False  # Ensure we can clip values in each call
                )
        finally:
            # Save final intermediate results
            if self.save_path and len(self.intermediate_results['t']) > 0:
                np.savez(self.save_path,
                         t=np.array(self.intermediate_results['t']),
                         y=np.array(self.intermediate_results['y']).T,
                         ca=np.array(self.intermediate_results['ca']))
                print(f"\nPartial results saved to: {self.save_path}")

            # Clean up progress bar if used
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None
            self.t_eval = None
            self.last_t = None
            self.ca_start = None
            self.theta_start = None
            self.rpm = None
        
        # Handle both dict (CVODE) and object (scipy) solution formats
        if isinstance(solution, dict):
            # CVODE returns a dict
            t = solution['t']
            y = solution['y']
            success = solution.get('success', True)
            message = solution.get('message', '')
            nfev = solution.get('nfev', 0)
            njev = 0  # CVODE doesn't track Jacobian evaluations separately
        else:
            # scipy returns an object
            t = solution.t
            y = solution.y
            success = solution.success
            message = solution.message
            nfev = solution.nfev
            njev = getattr(solution, 'njev', 0)

        # Calculate crank angles
        crank_angles = ca_start + np.rad2deg(omega * t)

        return {
            't': t,
            'y': y,
            'ca': crank_angles,
            'success': success,
            'message': message,
            'nfev': nfev,
            'njev': njev
        } 
