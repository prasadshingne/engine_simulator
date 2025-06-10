"""Results processing for engine simulation."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

@dataclass
class SimulationResults:
    """Container for simulation results."""
    time: np.ndarray           # Time [s]
    crank_angle: np.ndarray    # Crank angle [deg]
    temperature: np.ndarray    # Temperature [K]
    pressure: np.ndarray       # Pressure [Pa]
    volume: np.ndarray         # Volume [m³]
    mass: np.ndarray           # Mass [kg]
    species: np.ndarray        # Species mass fractions [-]
    species_names: List[str]   # Species names
    
    @classmethod
    def from_solver_output(cls, output: Dict, species_names: List[str], model_type: str = "single"):
        """Create results from solver output."""
        if model_type == "single":
            # Single zone model: [T, V, P, m, Y...]
            return cls(
                time=output['t'],
                crank_angle=output['ca'],
                temperature=output['y'][0],
                pressure=output['y'][2],
                volume=output['y'][1],
                mass=output['y'][3],
                species=output['y'][4:],
                species_names=species_names
            )
        else:
            # Multi-zone model: [T, P, V, m, T_i, Y_i..., Y_bulk]
            nsp = len(species_names)
            nzones = (output['y'].shape[0] - 4 - nsp) // (nsp + 1)
            return cls(
                time=output['t'],
                crank_angle=output['ca'],
                temperature=output['y'][0],
                pressure=output['y'][1],
                volume=output['y'][2],
                mass=output['y'][3],
                species=output['y'][4 + nzones*(nsp+1):],  # Use bulk composition
                species_names=species_names
            )
    
    def plot_pressure_volume(self, save_path: str = None, dpi: int = 300):
        """Plot P-V diagram."""
        plt.figure(figsize=(8, 6))
        plt.plot(self.volume * 1e6, self.pressure / 1e5)
        plt.xlabel('Volume [cm³]')
        plt.ylabel('Pressure [bar]')
        plt.title('P-V Diagram')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_temperature_ca(self, save_path: str = None, dpi: int = 300):
        """Plot temperature vs crank angle."""
        plt.figure(figsize=(8, 6))
        plt.plot(self.crank_angle, self.temperature)
        plt.xlabel('Crank Angle [deg]')
        plt.ylabel('Temperature [K]')
        plt.title('Temperature Profile')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_pressure_ca(self, save_path: str = None, dpi: int = 300):
        """Plot pressure vs crank angle."""
        plt.figure(figsize=(8, 6))
        plt.plot(self.crank_angle, self.pressure / 1e5)
        plt.xlabel('Crank Angle [deg]')
        plt.ylabel('Pressure [bar]')
        plt.title('Pressure Profile')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_species(self, species_indices: List[int] = None,
                    save_path: str = None, dpi: int = 300):
        """Plot species mass fractions."""
        if species_indices is None:
            species_indices = range(len(self.species_names))
            
        plt.figure(figsize=(10, 6))
        for i in species_indices:
            plt.plot(self.crank_angle, self.species[i],
                    label=self.species_names[i])
        
        plt.xlabel('Crank Angle [deg]')
        plt.ylabel('Mass Fraction [-]')
        plt.title('Species Evolution')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def calculate_performance(self) -> Dict[str, float]:
        """Calculate engine performance metrics."""
        # Calculate indicated work
        # Convention: clockwise P-V loop gives positive work
        dW = np.diff(self.volume) * (self.pressure[1:] + self.pressure[:-1])/2
        W_indicated = np.sum(dW)
        
        # Calculate IMEP
        V_displacement = np.max(self.volume) - np.min(self.volume)
        imep = W_indicated / V_displacement
        
        # Find peak pressure and temperature
        p_max = np.max(self.pressure)
        T_max = np.max(self.temperature)
        
        return {
            'indicated_work': W_indicated,  # [J]
            'imep': imep / 1e5,            # [bar]
            'peak_pressure': p_max / 1e5,   # [bar]
            'peak_temperature': T_max       # [K]
        }

    def plot_mass_ca(self, save_path: str = None, dpi: int = 300):
        """Plot total mass vs crank angle."""
        plt.figure(figsize=(8, 6))
        plt.plot(self.crank_angle, self.mass * 1000)  # Convert to grams
        plt.xlabel('Crank Angle [deg]')
        plt.ylabel('Mass [g]')
        plt.title('Total Mass')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_interactive(self, save_path=None):
        """Create interactive plots of simulation results."""
        # Create subplots - 2x3 grid
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
        
        # P-V diagram
        ax1.plot(self.volume * 1e6, self.pressure / 1e5)
        ax1.set_xlabel('Volume [cm³]')
        ax1.set_ylabel('Pressure [bar]')
        ax1.set_title('P-V Diagram')
        ax1.grid(True)
        
        # Temperature vs crank angle
        ax2.plot(self.crank_angle, self.temperature)
        ax2.set_xlabel('Crank Angle [deg]')
        ax2.set_ylabel('Temperature [K]')
        ax2.set_title('Temperature Profile')
        ax2.grid(True)
        
        # Pressure vs crank angle
        ax3.plot(self.crank_angle, self.pressure / 1e5)
        ax3.set_xlabel('Crank Angle [deg]')
        ax3.set_ylabel('Pressure [bar]')
        ax3.set_title('Pressure Profile')
        ax3.grid(True)
        
        # Mass vs crank angle
        ax4.plot(self.crank_angle, self.mass * 1000)  # Convert to grams
        ax4.set_xlabel('Crank Angle [deg]')
        ax4.set_ylabel('Mass [g]')
        ax4.set_title('Total Mass')
        ax4.grid(True)
        
        # Major species evolution
        major_species = ['C8H18', 'O2', 'CO2', 'H2O']
        for name in major_species:
            if name in self.species_names:
                i = self.species_names.index(name)
                ax5.plot(self.crank_angle, self.species[i], label=name)
        ax5.set_xlabel('Crank Angle [deg]')
        ax5.set_ylabel('Mass Fraction [-]')
        ax5.set_title('Major Species Evolution')
        ax5.grid(True)
        ax5.legend()
        
        # Minor species evolution
        minor_species = ['C8H17', 'C8H16', 'C8H17O2', 'C8H16OOH', 'O2C8H16OOH', 
                        'C8KET', 'CH2O', 'HCO', 'CO', 'H2O2', 'HO2', 'OH']
        for name in minor_species:
            if name in self.species_names:
                i = self.species_names.index(name)
                if np.max(self.species[i]) > 1e-6:  # Only plot if species appears
                    ax6.plot(self.crank_angle, self.species[i], label=name)
        ax6.set_xlabel('Crank Angle [deg]')
        ax6.set_ylabel('Mass Fraction [-]')
        ax6.set_title('Minor Species Evolution')
        ax6.grid(True)
        ax6.legend()
        ax6.set_yscale('log')  # Use log scale for better visibility
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_minor_species(self, save_path=None):
        """Plot evolution of minor species."""
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Select minor species (intermediates and radicals)
        minor_species = ['C8H17', 'C8H16', 'C8H17O2', 'C8H16OOH', 'O2C8H16OOH', 
                        'C8KET', 'CH2O', 'HCO', 'CO', 'H2O2', 'HO2', 'OH']
        
        for name in minor_species:
            if name in self.species_names:
                i = self.species_names.index(name)
                if np.max(self.species[i]) > 1e-6:  # Only plot if species appears
                    plt.plot(self.crank_angle, self.species[i], label=name)
        
        plt.xlabel('Crank Angle [deg]')
        plt.ylabel('Mass Fraction [-]')
        plt.title('Minor Species Evolution')
        plt.grid(True)
        plt.legend()
        plt.yscale('log')  # Use log scale for better visibility
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 