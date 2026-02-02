"""Chemical kinetics interface using Cantera."""

import numpy as np
import cantera as ct
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class ChemistryParams:
    """Chemistry parameters."""
    mechanism: str        # Mechanism file path
    fuel: str            # Fuel species name
    phi: float           # Equivalence ratio
    egr: float           # EGR fraction
    
class Chemistry:
    """Interface for chemical kinetics calculations."""
    
    def __init__(self, params: ChemistryParams):
        """
        Initialize chemistry interface.
        
        Parameters
        ----------
        params : ChemistryParams
            Chemistry parameters
        """
        self.params = params
        self.gas = ct.Solution(params.mechanism)
        
        # Explicitly enable all reactions
        self.gas.set_multiplier(1.0)
        
        print("\nInitializing chemistry:")
        print("Available species:", self.gas.species_names)
        print("Fuel species:", self.params.fuel)
        print("Number of reactions:", len(self.gas.reactions()))
        
        # Check multipliers for first few reactions
        print("\nReaction multipliers:")
        for i in range(min(5, len(self.gas.reactions()))):
            print(f"Reaction {i}: {self.gas.multiplier(i)}")
            print(f"  {self.gas.reactions()[i]}")
        
    def calculate_egr_composition(self, T_burned: float = 2000.0, T_egr: float = 480.0) -> Dict[str, float]:
        """Calculate EGR composition from equilibrium products."""
        # Create a new gas object for equilibrium calculation
        gas_eq = ct.Solution(self.params.mechanism)
        
        # Set up initial mixture at stoichiometric conditions
        gas_eq.set_equivalence_ratio(
            phi=self.params.phi,
            fuel=self.params.fuel,
            oxidizer={'O2': 1.0, 'N2': 3.76}
        )
        
        # Set to adiabatic flame temperature and let equilibrate
        gas_eq.TPX = T_burned, ct.one_atm, gas_eq.X
        gas_eq.equilibrate('HP')
        
        # Cool down to EGR temperature at constant pressure
        gas_eq.TP = T_egr, ct.one_atm
        gas_eq.equilibrate('HP')
        
        print("\nCalculated EGR composition (equilibrium):")
        X_egr = {}
        for name, X in zip(gas_eq.species_names, gas_eq.X):
            if X > 1e-6:  # Only store significant species
                X_egr[name] = X
                print(f"{name}: {X:.4f}")
            
        return X_egr
        
    def setup_initial_mixture(self, T: float, P: float) -> np.ndarray:
        """Set up initial mixture composition."""
        print("\nSetting up fuel-air mixture with EGR:")
        print(f"T = {T:.1f} K, P = {P/1e5:.1f} bar")
        print(f"Phi = {self.params.phi:.2f}")
        print(f"EGR = {self.params.egr:.2f}")
        
        # Set up fresh charge at desired equivalence ratio
        self.gas.set_equivalence_ratio(
            phi=self.params.phi,
            fuel=self.params.fuel,
            oxidizer={'O2': 1.0, 'N2': 3.76}
        )
        Y_fresh = self.gas.Y.copy()
        
        # Calculate and set up EGR composition at the initial temperature
        X_egr = self.calculate_egr_composition(T_egr=T)
        self.gas.TPX = T, P, X_egr
        Y_egr = self.gas.Y.copy()
        
        # Mix fresh charge with EGR
        Y0 = (1 - self.params.egr) * Y_fresh + self.params.egr * Y_egr
        
        # Set final state
        self.gas.TPY = T, P, Y0
        
        print("\nInitial mixture composition:")
        for name, Y in zip(self.gas.species_names, Y0):
            if Y > 1e-6:  # Only print significant species
                print(f"{name}: {Y:.6f}")
        
        return Y0
    
    def get_properties(self) -> Dict[str, float]:
        """
        Get current gas properties.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of gas properties
        """
        return {
            'cp': self.gas.cp_mass,        # [J/kg/K]
            'cv': self.gas.cv_mass,        # [J/kg/K]
            'h': self.gas.h,               # [J/kg]
            'u': self.gas.int_energy_mass, # [J/kg]
            'rho': self.gas.density,       # [kg/m³]
            'MW': self.gas.mean_molecular_weight,  # [kg/kmol]
            'gamma': self.gas.cp/self.gas.cv
        }
    
    def get_reaction_rates(self) -> Tuple[np.ndarray, float]:
        """
        Get species production rates and heat release.
        
        Returns
        -------
        Tuple[np.ndarray, float]
            Species mass production rates [kg/m³/s], Heat release [W/m³]
        """
        # Get net production rates [kmol/m³/s]
        wdot = self.gas.net_production_rates
        
        # Convert to mass rates [kg/m³/s]
        mdot = wdot * self.gas.molecular_weights
        
        # Calculate heat release [W/m³]
        # Get reaction enthalpies and rates
        dH_rxn = np.zeros(len(self.gas.reactions()))
        for i in range(len(self.gas.reactions())):
            dH_rxn[i] = self.gas.delta_enthalpy[i]  # [J/kmol]
        
        # Heat release is sum of reaction enthalpies times net rates
        Q = -np.sum(dH_rxn * self.gas.net_rates_of_progress)
        
        return mdot, Q
    
    def update_state(self, T: float, P: float, Y: np.ndarray):
        """
        Update gas state.
        
        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Pressure [Pa]
        Y : np.ndarray
            Mass fractions
        """
        try:
            # Clean up mass fractions
            Y = np.array(Y, dtype=np.float64)
            Y[np.abs(Y) < 1e-12] = 0.0
            Y = np.maximum(Y, 0.0)
            Y = Y / np.sum(Y)
            
            # Update state
            self.gas.TPY = T, P, Y
            
            # Ensure reactions stay enabled
            self.gas.set_multiplier(1.0)
            
            # Verify state is valid
            if not np.isfinite(self.gas.cp_mass):
                raise ValueError("Invalid thermodynamic state (non-finite cp)")
                
        except Exception as e:
            print("\nError in update_state:")
            print(f"T = {T:.1f} K, P = {P/1e5:.1f} bar")
            print(f"Exception: {str(e)}")
            raise 