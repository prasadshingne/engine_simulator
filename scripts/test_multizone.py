#!/usr/bin/env python3
"""Test script for multi-zone model."""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path and set project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.simulation.engine import EngineConfig, EngineSimulation

def main():
    """Run multi-zone model and plot results."""
    # Create results directory
    results_dir = project_root / 'data' / 'output' / 'multizone_test'
    results_dir.mkdir(exist_ok=True)
    
    # Load base configuration
    config_path = project_root / 'src' / 'config' / 'default_config.yaml'
    config = EngineConfig.from_yaml(str(config_path))
    
    # Update mechanism path to absolute path
    config.mechanism = str(project_root / 'data' / 'mechanisms' / 'Nissan_chem.yaml')
    
    # Configure for multi-zone model
    config.model_type = "multi"
    config.nzones = 3  # Start with 3 zones for testing
    config.adiabatic = True  # Run in adiabatic mode

    # Use BDF solver - Python's equivalent of Matlab's ode15s
    # Match Matlab settings: ode15s with RelTol=1e-5, AbsTol=1e-12
    config.method = "BDF"  # Equivalent to Matlab's ode15s
    config.rtol = 1.0e-5  # Match Matlab RelTol
    config.atol = 1.0e-12  # Match Matlab AbsTol
    config.max_step = 1.0e-3  # Reasonable max step
    config.first_step = 1.0e-6  # Small initial step
    
    # Run multi-zone model
    print("\nRunning multi-zone model...")
    sim = EngineSimulation(config)
    results = sim.run()
    
    # Plot results
    print("\nPlotting results...")
    
    # Plot bulk temperature
    plt.figure(figsize=(10, 6))
    plt.plot(results.crank_angle, results.temperature)
    plt.xlabel('Crank Angle [deg]')
    plt.ylabel('Bulk Temperature [K]')
    plt.title('Bulk Temperature Evolution')
    plt.grid(True)
    plt.savefig(results_dir / 'bulk_temperature.png', dpi=300)
    plt.close()
    
    # Plot bulk pressure
    plt.figure(figsize=(10, 6))
    plt.plot(results.crank_angle, results.pressure/1e5)
    plt.xlabel('Crank Angle [deg]')
    plt.ylabel('Bulk Pressure [bar]')
    plt.title('Bulk Pressure Evolution')
    plt.grid(True)
    plt.savefig(results_dir / 'bulk_pressure.png', dpi=300)
    plt.close()
    
    # Plot P-V diagram
    plt.figure(figsize=(10, 6))
    plt.plot(results.volume*1e6, results.pressure/1e5)
    plt.xlabel('Volume [cm³]')
    plt.ylabel('Pressure [bar]')
    plt.title('P-V Diagram')
    plt.grid(True)
    plt.savefig(results_dir / 'pv_diagram.png', dpi=300)
    plt.close()
    
    # Plot major species
    major_species = ['C8H18', 'O2', 'CO2', 'H2O']
    for species in major_species:
        if species in results.species_names:
            idx = results.species_names.index(species)
            plt.figure(figsize=(10, 6))
            plt.plot(results.crank_angle, results.species[idx])
            plt.xlabel('Crank Angle [deg]')
            plt.ylabel('Mass Fraction [-]')
            plt.title(f'{species} Mass Fraction Evolution')
            plt.grid(True)
            plt.savefig(results_dir / f'{species}_evolution.png', dpi=300)
            plt.close()
    
    # Calculate performance metrics
    perf = results.calculate_performance()
    
    print("\nPerformance Metrics:")
    print(f"IMEP: {perf['imep']:.2f} bar")
    print(f"Peak Pressure: {perf['peak_pressure']:.2f} bar")
    print(f"Peak Temperature: {perf['peak_temperature']:.0f} K")
    
    print(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main() 