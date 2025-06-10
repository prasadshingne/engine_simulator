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

@dataclass
class SolverParams:
    """Solver parameters."""
    method: str = "LSODA"     # Integration method
    rtol: float = 1.0e-4      # Relative tolerance
    atol: float = 1.0e-6      # Absolute tolerance
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
    nzones: int = 5  # Number of zones

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
            nzones=solver_config['nzones']
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
        self.last_t = None
        self.t_eval = None
        self.gamma = 1.35
        self.ca_start = None
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
        theta = (rpm * 2 * np.pi / 60) * t  # [rad]
        dVdt = self.geom.volume_rate(theta, rpm)
        
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
            current_ca = self.ca_start + np.rad2deg(theta)  # Current crank angle
            progress = current_ca - self.ca_start  # Progress from start
            if progress > self.progress_bar.n:  # Only update if we've made progress
                self.progress_bar.update(progress - self.progress_bar.n)
                # Display shifted crank angle
                self.progress_bar.set_postfix({'CA': f"{current_ca - 180:.1f}°"})
        
        return dydt
        
    def _ode_system_multi(self, t: float, y: np.ndarray, rpm: float, T_wall: float) -> np.ndarray:
        """Calculate derivatives for multi-zone engine ODE system."""
        # Extract state variables
        T = y[0]              # Bulk Temperature [K]
        P = y[1]              # Bulk Pressure [Pa]
        V = y[2]              # Total Volume [m³]
        M = y[3]              # Total Mass [kg]
        
        # Get number of species and zones
        nsp = self.chemistry.gas.n_species
        nzones = min(max(1, self.params.nzones), 20)  # Limit between 1 and 20 zones
        
        # Extract zone temperatures and compositions
        tzone = np.zeros(nzones)
        yzone = np.zeros((nsp, nzones))
        for i in range(nzones):
            tzone[i] = y[4 + i*(nsp+1)]
            yzone[:,i] = y[4 + i*(nsp+1) + 1:4 + (i+1)*(nsp+1)]
            # Ensure mass fractions stay within bounds
            yzone[:,i] = np.clip(yzone[:,i], self.params.min_mass_fraction, 1.0)
            yzone[:,i] /= np.sum(yzone[:,i])  # Renormalize
        
        # Extract bulk composition
        Ybulk = y[4 + nzones*(nsp+1):]
        
        # Calculate crank angle and volume change
        theta = (rpm * 2 * np.pi / 60) * t  # [rad]
        dVdt = self.geom.volume_rate(theta, rpm)
        
        # Update progress bar if enabled
        if self.progress_bar is not None:
            current_ca = self.ca_start + np.rad2deg(theta)
            progress = current_ca - self.ca_start
            if progress > self.progress_bar.n:
                self.progress_bar.update(progress - self.progress_bar.n)
                self.progress_bar.set_postfix({
                    'CA': f"{current_ca - 180:.1f}°",
                    'Zones': nzones
                })
        
        # Calculate surface areas
        head_area, piston_area, liner_area = self.geom.surface_area(theta)
        total_area = head_area + piston_area + liner_area
        
        # Calculate heat transfer (zero if adiabatic)
        Q_wall = 0.0
        if not self.params.adiabatic:
            p_motored = self._calculate_motored_pressure(theta)
            Q_wall, _ = self.heat_transfer.heat_transfer_rate(
                theta, rpm, P, T, p_motored,
                self.p_ref, self.T_ref, self.V_ref, T_wall
            )
        
        # Initialize arrays for zone calculations
        Ygen = np.zeros((nsp, nzones))
        unsp = np.zeros((nsp, nzones))
        hnsp = np.zeros((nsp, nzones))
        Cvi = np.zeros(nzones)
        Ri = np.zeros(nzones)
        Vi = np.zeros(nzones)
        dVidt = np.zeros(nzones)
        Ai = np.zeros(nzones)
        hi = np.zeros(nzones)
        Mi = np.zeros(nzones)
        
        # Calculate bulk properties
        sumgen = 0.0
        sumRYdot = 0.0
        sumhYdot = 0.0
        sumCv = 0.0
        sumR = 0.0
        sumPV = 0.0
        
        # Calculate per-zone properties
        for i in range(nzones):
            # Calculate zone mass based on composition
            Mi[i] = M * np.sum(yzone[:,i])
            
            # Calculate zone volume using ideal gas law
            # V = mRT/P
            Vi[i] = Mi[i] * ct.gas_constant * tzone[i] / (P * self.chemistry.gas.mean_molecular_weight)
            dVidt[i] = dVdt * Vi[i]/V  # Zone volume change proportional to volume fraction
            Ai[i] = 3*(2*(i-1))**2/(2*nzones*(nzones-1)*(2*nzones-1)) * total_area
            
            # Now that we have valid mass and volume, set zone state
            self.chemistry.gas.TPY = tzone[i], P, yzone[:,i]
            
            # Get species properties
            Rk = ct.gas_constant * self.chemistry.gas.molecular_weights
            hk = self.chemistry.gas.partial_molar_enthalpies
            unsp[:,i] = (hk - tzone[i] * Rk) / self.chemistry.gas.molecular_weights
            hnsp[:,i] = hk / self.chemistry.gas.molecular_weights
            
            # Calculate species production rates
            mdot, _ = self.chemistry.get_reaction_rates()
            Ygen[:,i] = mdot / self.chemistry.gas.density
            
            # Limit species production rates to prevent negative mass fractions
            for j in range(nsp):
                if yzone[j,i] + Ygen[j,i] * self.params.max_step < self.params.min_mass_fraction:
                    Ygen[j,i] = (self.params.min_mass_fraction - yzone[j,i]) / self.params.max_step
            
            # Calculate zone properties
            Cvi[i] = self.chemistry.gas.cv_mass
            Ri[i] = ct.gas_constant / self.chemistry.gas.mean_molecular_weight
            hi[i] = self.chemistry.gas.enthalpy_mass
            
            # Accumulate terms for bulk equations
            sumgen += Mi[i] * Ygen[:,i].T @ unsp[:,i]
            sumRYdot += Rk.T @ Ygen[:,i]
            sumhYdot += Mi[i] * Ygen[:,i].T @ hnsp[:,i]
            sumCv += Mi[i] * Cvi[i]
            sumR += Mi[i] * Ri[i]
            sumPV += P * Vi[i]
        
        # Calculate bulk derivatives
        dTdt = (sumgen - P*dVdt) / sumCv  # No Q_wall term in adiabatic case
        
        # Calculate pressure derivative using ideal gas law
        dPdt = P * (dTdt/T - dVdt/V)
        
        # Mass is conserved
        dMdt = 0.0
        
        # Calculate bulk composition derivatives
        dYdt = np.zeros(nsp)
        for i in range(nzones):
            dYdt += Mi[i] * Ygen[:,i]
        dYdt /= M
        
        # Initialize derivative vector
        dydt = np.zeros_like(y)
        
        # Set bulk derivatives
        dydt[0] = dTdt
        dydt[1] = dPdt
        dydt[2] = dVdt
        dydt[3] = dMdt
        
        # Set zone derivatives
        for i in range(nzones):
            # Get reaction rates and heat release for zone i
            mdot, Q_chem = self.chemistry.get_reaction_rates()  # Q_chem in [W/m³]
            
            # Calculate zone temperature derivative using energy equation
            # dT/dt = 1/(m*cv) * (-P*dV + Q_chem*V)
            dTidt = 1.0/(Mi[i] * Cvi[i]) * (-P*dVidt[i] + Q_chem*Vi[i])
            
            dydt[4 + i*(nsp+1)] = dTidt
            
            # Zone composition derivatives
            dydt[4 + i*(nsp+1) + 1:4 + (i+1)*(nsp+1)] = Ygen[:,i]
        
        # Set bulk composition derivatives
        dydt[4 + nzones*(nsp+1):] = dYdt
        
        return dydt
    
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
        # Store crank angle range for progress bar
        self.ca_start = ca_start
        self.rpm = rpm
        
        # Convert crank angles to radians
        theta_start = np.deg2rad(ca_start)
        theta_end = np.deg2rad(ca_end)
        
        # Calculate time span
        omega = rpm * 2 * np.pi / 60  # [rad/s]
        t_span = (0, (theta_end - theta_start)/omega)
        
        # Store reference conditions at IVC
        self.p_ref = y0[2]
        self.T_ref = y0[0]
        self.V_ref = y0[1]
        
        # Create time evaluation points (increased for smoothness)
        self.t_eval = np.linspace(t_span[0], t_span[1], 700)  # Increased from 100 to 700 points
        
        # Initialize progress bar if requested
        if self.params.show_progress:
            print("\nSolving engine cycle:")
            # Calculate total crank angle range
            total_ca = ca_end - ca_start
            self.progress_bar = tqdm(total=total_ca, initial=0, 
                                   desc="Progress", unit="°CA",
                                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.1f}°CA [{elapsed}<{remaining}, {rate_fmt}]')
        
        try:
            # Solve ODE system
            solution = solve_ivp(
                fun=lambda t, y: self._ode_system(t, y, rpm, T_wall),
                t_span=t_span,
                y0=y0,
                method=self.params.method,
                t_eval=self.t_eval,
                rtol=self.params.rtol,
                atol=self.params.atol,
                max_step=self.params.max_step,
                first_step=self.params.first_step
            )
        finally:
            # Clean up progress bar if used
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None
            self.t_eval = None
            self.last_t = None
            self.ca_start = None
            self.rpm = None
        
        # Calculate crank angles
        crank_angles = ca_start + np.rad2deg(omega * solution.t)
        
        return {
            't': solution.t,
            'y': solution.y,
            'ca': crank_angles,
            'success': solution.success,
            'message': solution.message,
            'nfev': solution.nfev,
            'njev': solution.njev
        } 