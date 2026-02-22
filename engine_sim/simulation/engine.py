"""Main engine simulation class."""

import numpy as np
import yaml
from dataclasses import dataclass
from typing import Dict, Optional, Union

from ..engine.geometry import GeometryParams
from ..engine.heat_transfer import HeatTransfer, WoschniParams
from ..models.chemistry import Chemistry, ChemistryParams
from ..config.paths import get_default_config, resolve_mechanism_path
from .solver import EngineSolver, SolverParams
from .results import SimulationResults
from .multizone_profiles import validate_multizone_count

@dataclass
class EngineConfig:
    """Engine configuration parameters."""
    # Geometry
    bore: float           # Bore diameter [m]
    stroke: float         # Stroke length [m]
    con_rod: float       # Connecting rod length [m]
    comp_ratio: float    # Compression ratio [-]
    
    # Operating conditions
    speed: float         # Engine speed [rpm]
    wall_temp: float     # Wall temperature [K]
    
    # Simulation range
    start_ca: float      # Start crank angle [deg]
    end_ca: float        # End crank angle [deg]
    
    # Chemistry
    mechanism: str                          # Mechanism file path
    fuel: Union[str, Dict[str, float]]     # Fuel species name or multi-component dict
    phi: float                              # Equivalence ratio
    egr: float          # EGR fraction
    
    # Initial conditions
    pressure: float     # Initial pressure [Pa]
    temperature: float  # Initial temperature [K]
    
    # Solver settings
    method: str         # Solver method
    rtol: float        # Relative tolerance
    atol: float        # Absolute tolerance
    max_step: float    # Maximum step size
    first_step: float  # First step size
    adiabatic: bool    # Whether to run in adiabatic mode
    model_type: str    # Model type (single or multi)
    nzones: int        # Number of zones for multi-zone model (10/20/40)

    # Output settings
    save_path: str     # Path to save results
    plot_format: str   # Plot file format
    dpi: int          # Plot resolution

    # Optional overrides (have defaults — must come after all non-default fields)
    use_manual_solver: bool = False  # Use legacy scipy ODE solver instead of Cantera ReactorNet
    
    @classmethod
    def from_yaml(cls, config_path: str = None):
        """Load configuration from YAML file.

        Parameters
        ----------
        config_path : str, optional
            Path to YAML config file. If None, uses the bundled default config.
        """
        if config_path is None:
            config_path = str(get_default_config())
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return cls(
            # Geometry
            bore=float(config['engine']['geometry']['bore']),
            stroke=float(config['engine']['geometry']['stroke']),
            con_rod=float(config['engine']['geometry']['con_rod']),
            comp_ratio=float(config['engine']['geometry']['comp_ratio']),
            
            # Operating conditions
            speed=float(config['engine']['operating_conditions']['speed']),
            wall_temp=float(config['engine']['operating_conditions']['wall_temp']),
            
            # Simulation range
            start_ca=float(config['engine']['simulation']['start_ca']),
            end_ca=float(config['engine']['simulation']['end_ca']),
            
            # Chemistry
            mechanism=str(config['chemistry']['mechanism']),
            fuel=config['chemistry']['fuel'],  # str or dict for multi-component
            phi=float(config['chemistry']['phi']),
            egr=float(config['chemistry']['egr']),
            
            # Initial conditions
            pressure=float(config['initial_conditions']['pressure']),
            temperature=float(config['initial_conditions']['temperature']),
            
            # Solver settings
            method=str(config['solver']['method']),
            rtol=float(config['solver']['rtol']),
            atol=float(config['solver']['atol']),
            max_step=float(config['solver']['max_step']),
            first_step=float(config['solver']['first_step']),
            adiabatic=bool(config['solver']['adiabatic']),
            model_type=str(config['solver']['model_type']),
            nzones=int(config['solver']['nzones']),
            use_manual_solver=bool(config['solver'].get('use_manual_solver', False)),
            
            # Output settings
            save_path=str(config['output']['save_path']),
            plot_format=str(config['output']['plot_format']),
            dpi=int(config['output']['dpi'])
        )

class EngineSimulation:
    """Main engine simulation class."""
    
    def __init__(self, config: EngineConfig):
        """
        Initialize engine simulation.
        
        Parameters
        ----------
        config : EngineConfig
            Simulation configuration
        """
        self.config = config
        if self.config.model_type == "multi":
            self.config.nzones = validate_multizone_count(self.config.nzones)

        # Auto-resolve mechanism path
        config.mechanism = resolve_mechanism_path(config.mechanism)

        # Initialize geometry
        self.geom = GeometryParams(
            bore=config.bore,
            stroke=config.stroke,
            con_rod=config.con_rod,
            comp_ratio=config.comp_ratio
        )
        
        # Initialize heat transfer
        self.heat_transfer = HeatTransfer(
            geom=self.geom,
            params=WoschniParams()
        )
        
        # Initialize chemistry
        self.chemistry = Chemistry(
            ChemistryParams(
                mechanism=config.mechanism,
                fuel=config.fuel,
                phi=config.phi,
                egr=config.egr
            )
        )
        
        # Initialize solver
        self.solver = EngineSolver(
            geom=self.geom,
            heat_transfer=self.heat_transfer,
            chemistry=self.chemistry,
            params=SolverParams(
                method=config.method,
                rtol=config.rtol,
                atol=config.atol,
                max_step=config.max_step,
                first_step=config.first_step,
                adiabatic=config.adiabatic,
                model_type=config.model_type,
                nzones=config.nzones,
                use_manual_solver=config.use_manual_solver,
            )
        )
        
    def setup_initial_state(self) -> np.ndarray:
        """Set up initial state vector."""
        # Set up initial mixture
        Y0 = self.chemistry.setup_initial_mixture(
            self.config.temperature,
            self.config.pressure
        )
        
        # Calculate initial volume
        theta0 = np.deg2rad(self.config.start_ca)
        V0 = self.geom.cylinder_volume(theta0)
        
        # Calculate initial mass (ideal gas)
        props = self.chemistry.get_properties()
        m0 = self.config.pressure * V0 / (
            props['cv'] * self.config.temperature * (props['gamma'] - 1.0))  # PV = mRT where R = cv*(gamma-1)
        
        if self.config.model_type == "single":
            # Single zone model
            y0 = np.zeros(4 + len(Y0))
            y0[0] = self.config.temperature
            y0[1] = V0
            y0[2] = self.config.pressure
            y0[3] = m0
            y0[4:] = Y0
        else:
            # Multi-zone model
            # State vector structure (matching Matlab HCCI_sim_dec038.m lines 121-130):
            # [T, P, V, T_zone1, Y_zone1, ..., T_zoneN, Y_zoneN, Y_bulk]
            # Note: Mass M is NOT in state vector (constant for closed cycle)
            nzones = validate_multizone_count(self.config.nzones)
            nsp = len(Y0)
            y0 = np.zeros(3 + nzones*(nsp+1) + nsp)  # [T,P,V] + [T_i,Y_i for each zone] + [Y_bulk]
            y0[0] = self.config.temperature  # Bulk T
            y0[1] = self.config.pressure    # Bulk P
            y0[2] = V0                      # Volume
            # y0[3] = m0  # REMOVED - M is constant, not in state vector

            # Initialize zone temperatures and compositions (Matlab lines 124-129)
            # IMPORTANT: Use same temperature for all zones initially (Matlab line 45)
            # oper.tzone=oper.temp*ones(1,oper.nzone);
            for i in range(nzones):
                # Zone temperature (all zones start at bulk temperature)
                T_zone = self.config.temperature
                y0[3 + i*(nsp+1)] = T_zone  # Changed from y0[4+...] to y0[3+...]
                # Zone composition (same as bulk initially)
                y0[3 + i*(nsp+1) + 1:3 + (i+1)*(nsp+1)] = Y0  # Changed indexing

            # Bulk composition (Matlab line 130)
            y0[3 + nzones*(nsp+1):] = Y0  # Changed from y0[4+...] to y0[3+...]
        
        return y0
    
    def run(self) -> SimulationResults:
        """
        Run engine simulation.
        
        Returns
        -------
        SimulationResults
            Simulation results
        """
        # Set up initial state
        y0 = self.setup_initial_state()
        
        # Run simulation
        solution = self.solver.solve_closed_cycle(
            rpm=self.config.speed,
            T_wall=self.config.wall_temp,
            ca_start=self.config.start_ca,
            ca_end=self.config.end_ca,
            y0=y0
        )
        
        # Process results
        results = SimulationResults.from_solver_output(
            solution,
            self.chemistry.gas.species_names,
            model_type=self.config.model_type,
            mechanism=self.config.mechanism  # Pass mechanism for mass calculation
        )
        
        return results
    
    def run_and_plot(self):
        """Run simulation and generate plots."""
        # Run simulation
        results = self.run()
        
        # Calculate performance metrics
        metrics = results.calculate_performance()
        print("\nPerformance Metrics:")
        print(f"IMEP: {metrics['imep']:.1f} bar")
        print(f"Peak Pressure: {metrics['peak_pressure']:.1f} bar")
        print(f"Peak Temperature: {metrics['peak_temperature']:.0f} K")
        
        # Generate plots
        results.plot_pressure_volume(
            save_path=f"{self.config.save_path}/pv_diagram.{self.config.plot_format}",
            dpi=self.config.dpi
        )
        
        results.plot_temperature_ca(
            save_path=f"{self.config.save_path}/temperature.{self.config.plot_format}",
            dpi=self.config.dpi
        )
        
        results.plot_pressure_ca(
            save_path=f"{self.config.save_path}/pressure.{self.config.plot_format}",
            dpi=self.config.dpi
        )
        
        results.plot_mass_ca(
            save_path=f"{self.config.save_path}/mass.{self.config.plot_format}",
            dpi=self.config.dpi
        )
        
        # Plot major species
        major_species = ['C8H18', 'O2', 'CO2', 'H2O']
        species_indices = [
            i for i, name in enumerate(results.species_names)
            if name in major_species
        ]
        
        results.plot_species(
            species_indices=species_indices,
            save_path=f"{self.config.save_path}/species.{self.config.plot_format}",
            dpi=self.config.dpi
        )
        
        return results 
