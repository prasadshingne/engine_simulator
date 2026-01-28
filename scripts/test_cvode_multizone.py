#!/usr/bin/env python3
"""
Test 10-zone non-adiabatic multizone model using CVODE (SUNDIALS).

Run with: conda activate py39_sundials && python scripts/test_cvode_multizone.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Check if CVODE is available
try:
    from scikits.odes import ode
    print("✓ scikits.odes available - CVODE will be used")
except ImportError:
    print("✗ scikits.odes NOT available")
    print("  Install with: conda install -c conda-forge scikits.odes sundials")
    sys.exit(1)

from src.simulation.engine import EngineConfig, EngineSimulation
from src.simulation.results import SimulationResults


def main():
    """Run 10-zone non-adiabatic simulation with CVODE."""
    # Load configuration
    config_path = project_root / 'src' / 'config' / 'default_config.yaml'
    config = EngineConfig.from_yaml(str(config_path))

    # Update mechanism path
    config.mechanism = str(project_root / 'data' / 'mechanisms' / 'Nissan_chem.yaml')

    # Configure for 10-zone WITH HEAT TRANSFER (non-adiabatic)
    config.model_type = "multi"
    config.nzones = 10  # 10 zones as requested
    config.adiabatic = False  # Heat transfer enabled - the main point!
    config.method = "CVODE"  # Use CVODE from SUNDIALS
    config.rtol = 1.0e-5
    config.atol = 1.0e-12
    config.show_progress = True

    print("="*70)
    print("10-ZONE NON-ADIABATIC MULTIZONE MODEL WITH CVODE (SUNDIALS)")
    print("="*70)
    print(f"Zones: {config.nzones}")
    print(f"Adiabatic: {config.adiabatic}")
    print(f"Solver: CVODE (SUNDIALS) - equivalent to Matlab's ode15s")
    print(f"Heat transfer: ENABLED")
    print("="*70)
    print("\nThis demonstrates temperature stratification from heat transfer.")
    print("="*70)

    # Run simulation
    sim = EngineSimulation(config)
    y0 = sim.setup_initial_state()

    print(f"\nState vector size: {len(y0)} variables")
    print(f"  (3 bulk + 10 zones × 34 vars + 33 bulk species = 376)")

    solver_output = sim.solver.solve_closed_cycle(
        y0=y0,
        rpm=sim.config.speed,
        T_wall=sim.config.wall_temp,
        ca_start=sim.config.start_ca,
        ca_end=sim.config.end_ca
    )

    if not solver_output.get('success', False):
        print(f"\n⚠ Solver did not complete successfully: {solver_output.get('message', 'Unknown error')}")
        print("Plotting partial results...")

    # Create results object
    results = SimulationResults.from_solver_output(
        solver_output,
        sim.chemistry.gas.species_names,
        model_type=sim.config.model_type,
        mechanism=sim.config.mechanism
    )

    # Extract zone temperatures
    nsp = len(results.species_names)
    nzones = config.nzones

    zone_temps = np.zeros((nzones, len(results.crank_angle)))
    for i in range(nzones):
        zone_idx = 3 + i * (nsp + 1)
        zone_temps[i, :] = solver_output['y'][zone_idx, :]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('10-Zone Non-Adiabatic Multizone Model (CVODE/SUNDIALS)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Bulk Pressure
    ax = axes[0, 0]
    ax.plot(results.crank_angle, results.pressure / 1e5, 'b-', linewidth=2)
    ax.set_xlabel('Crank Angle [°]')
    ax.set_ylabel('Pressure [bar]')
    ax.set_title('Bulk Pressure')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='TDC')
    ax.legend()

    # Plot 2: Bulk Temperature
    ax = axes[0, 1]
    ax.plot(results.crank_angle, results.temperature, 'r-', linewidth=2)
    ax.set_xlabel('Crank Angle [°]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Bulk Temperature')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='TDC')
    ax.legend()

    # Plot 3: Zone Temperatures - SHOWS STRATIFICATION!
    ax = axes[1, 0]
    colors = plt.cm.plasma(np.linspace(0, 1, nzones))
    for i in range(nzones):
        label = f'Zone {i+1}'
        if i == 0:
            label += ' (core)'
        elif i == nzones - 1:
            label += ' (wall)'
        ax.plot(results.crank_angle, zone_temps[i, :], color=colors[i],
                linewidth=1.5, label=label, alpha=0.9)
    ax.set_xlabel('Crank Angle [°]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Zone Temperatures (10 Zones)')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=7, ncol=2, loc='best')

    # Plot 4: Temperature Stratification
    ax = axes[1, 1]
    temp_diff = zone_temps[0, :] - zone_temps[-1, :]  # Core minus wall
    ax.plot(results.crank_angle, temp_diff, 'g-', linewidth=2)
    ax.set_xlabel('Crank Angle [°]')
    ax.set_ylabel('ΔT (Core - Wall) [K]')
    ax.set_title('Temperature Stratification')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='TDC')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend()

    # Annotate max stratification
    max_diff_idx = np.argmax(np.abs(temp_diff))
    max_diff = temp_diff[max_diff_idx]
    max_diff_ca = results.crank_angle[max_diff_idx]
    ax.plot(max_diff_ca, max_diff, 'ro', markersize=8)
    ax.annotate(f'Max: {max_diff:.1f} K\nat {max_diff_ca:.1f}°',
                xy=(max_diff_ca, max_diff),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    # Save figure
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "10zone_cvode_stratification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    # Print statistics
    print("\n" + "="*70)
    print("ZONE TEMPERATURE STATISTICS (10 ZONES, NON-ADIABATIC)")
    print("="*70)

    # Find TDC index
    tdc_idx = np.argmin(np.abs(results.crank_angle))
    print(f"\nAt TDC ({results.crank_angle[tdc_idx]:.1f}°):")
    for i in range(nzones):
        print(f"  Zone {i+1:2d}: {zone_temps[i, tdc_idx]:.1f} K")
    temp_spread_tdc = zone_temps[:, tdc_idx].max() - zone_temps[:, tdc_idx].min()
    print(f"  Temperature spread: {temp_spread_tdc:.1f} K")

    # Find peak temperature location
    peak_idx = np.argmax(zone_temps.max(axis=0))
    print(f"\nAt Peak ({results.crank_angle[peak_idx]:.1f}°):")
    for i in range(nzones):
        print(f"  Zone {i+1:2d}: {zone_temps[i, peak_idx]:.1f} K")
    temp_spread_peak = zone_temps[:, peak_idx].max() - zone_temps[:, peak_idx].min()
    print(f"  Temperature spread: {temp_spread_peak:.1f} K")

    print(f"\nMaximum stratification: {max_diff:.1f} K at {max_diff_ca:.1f}°")
    print("="*70)

    # Calculate performance metrics
    try:
        perf = results.calculate_performance()
        print("\nPerformance Metrics:")
        print(f"IMEP: {perf['imep']:.2f} bar")
        print(f"Peak Pressure: {perf['peak_pressure']:.2f} bar")
        print(f"Peak Temperature: {perf['peak_temperature']:.0f} K")
    except Exception as e:
        print(f"\nCould not calculate performance metrics: {e}")

    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()
