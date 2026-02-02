"""Example script to run engine simulation."""

from pathlib import Path

from engine_sim.simulation.engine import EngineConfig, EngineSimulation
from engine_sim.config.paths import get_project_root


def main():
    """Run engine simulation example."""
    # Create results directory
    results_dir = get_project_root() / 'data' / 'output'
    results_dir.mkdir(exist_ok=True)

    # Load configuration (uses bundled default config, auto-resolves mechanism path)
    config = EngineConfig.from_yaml()

    # Create and run simulation
    sim = EngineSimulation(config)
    results = sim.run()

    # Plot results
    print("\nSimulation completed. Creating plots...")
    results.plot_interactive(str(results_dir / 'interactive_plots.png'))
    results.plot_minor_species(str(results_dir / 'minor_species.png'))

    # Calculate and display performance metrics
    perf = results.calculate_performance()
    print("\nEngine Performance Metrics:")
    print(f"Indicated Work: {perf['indicated_work']:.2f} J")
    print(f"IMEP: {perf['imep']:.2f} bar")
    print(f"Peak Pressure: {perf['peak_pressure']:.2f} bar")
    print(f"Peak Temperature: {perf['peak_temperature']:.2f} K")

    # Print summary of simulation settings
    print("\nSimulation Settings:")
    print(f"Fuel: {config.fuel}")
    print(f"Equivalence Ratio: {config.phi:.2f}")
    print(f"EGR Fraction: {config.egr:.2f}")
    print(f"Initial Temperature: {config.temperature:.0f} K")
    print(f"Initial Pressure: {config.pressure/1e5:.1f} bar")
    print(f"Compression Ratio: {config.comp_ratio:.1f}")
    print(f"Engine Speed: {config.speed:.0f} rpm")

if __name__ == '__main__':
    main()
