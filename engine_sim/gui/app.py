"""Streamlit GUI for the 0D HCCI Engine Simulator."""

import streamlit as st
import numpy as np
import pandas as pd

from engine_sim.simulation.engine import EngineConfig, EngineSimulation
from engine_sim.simulation.results import SimulationResults
from engine_sim.gui.plotting import plot_results_dashboard, plot_zone_temperatures, plot_heat_release
from engine_sim.config.mechanisms import MECHANISM_PRESETS, DEFAULT_MECHANISM

MAX_SAVED_RUNS = 4


def _make_run_label(config, mech_name: str = "") -> str:
    """Generate a short descriptive label from the config."""
    model = "SZ" if config.model_type == "single" else f"MZ-{config.nzones}"
    adi = "adi" if config.adiabatic else "non-adi"
    if "Gasoline" in mech_name:
        mech_short = "GS"
    else:
        # Derive octane number from fuel blend (C8H18 mole fraction × 100)
        if isinstance(config.fuel, dict):
            on = round(config.fuel.get("C8H18", 1.0) * 100)
        else:
            on = 100  # pure iso-octane string fallback
        mech_short = f"PRF{on}"
    return f"{mech_short}, {model}, phi={config.phi:.2f}, {adi}"


def main():
    st.set_page_config(
        page_title="Engine Simulator",
        page_icon="🔧",
        layout="wide",
    )

    st.title("Engine Simulator")
    st.caption("Interactive closed-cycle simulation with detailed chemical kinetics")

    # Initialise session state
    if 'saved_runs' not in st.session_state:
        st.session_state['saved_runs'] = []

    # ── Sidebar: all inputs ──────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        # --- Mechanism ---
        st.subheader("Mechanism")
        mech_names = list(MECHANISM_PRESETS.keys())
        mech_name = st.selectbox("Reaction Mechanism", mech_names, index=0)
        preset = MECHANISM_PRESETS[mech_name]

        # PRF blend control — only shown for Nissan PRF mechanism
        prf_octane = 85  # default; only used when Nissan is selected
        if "Nissan" in mech_name:
            prf_octane = st.slider(
                "Octane Number (PRF)", min_value=0, max_value=100, value=85, step=1,
                help="Mole fraction blend: iso-Octane % + n-Heptane % = 100%",
            )
            st.caption(
                f"PRF{prf_octane}: {prf_octane}% iso-Octane (C8H18) / "
                f"{100 - prf_octane}% n-Heptane (C7H16)"
            )
        else:
            st.caption(preset["fuel_label"])

        if "Gasoline" in mech_name:
            st.info("312 species — auto-dispatches to Cantera ReactorNet solver with GMRES + preconditioner.")

        # --- Mixture ---
        st.subheader("Mixture")
        phi = st.number_input("Equivalence Ratio (phi)", value=0.70, min_value=0.10, max_value=1.00, step=0.01, format="%.2f")
        egr = st.number_input("EGR Fraction", value=0.30, min_value=0.00, max_value=0.80, step=0.01, format="%.2f")

        # --- Engine Geometry ---
        st.subheader("Engine Geometry")
        bore_mm = st.number_input("Bore [mm]", value=86.0, min_value=50.0, max_value=200.0, step=1.0)
        stroke_mm = st.number_input("Stroke [mm]", value=86.0, min_value=50.0, max_value=200.0, step=1.0)
        con_rod_mm = st.number_input("Con Rod Length [mm]", value=145.5, min_value=80.0, max_value=400.0, step=0.5)
        comp_ratio = st.number_input("Compression Ratio", value=12.5, min_value=6.0, max_value=25.0, step=0.5)

        # --- Operating Conditions ---
        st.subheader("Operating Conditions")
        rpm = st.number_input("Engine Speed [RPM]", value=2000, min_value=500, max_value=7000, step=1)
        T_init = st.number_input("Initial Temperature [K]", value=475.0, min_value=300.0, max_value=700.0, step=5.0)
        P_init_bar = st.number_input("Initial Pressure [bar]", value=1.013, min_value=0.5, max_value=5.0, step=0.1, format="%.3f")
        T_wall = st.number_input("Wall Temperature [K]", value=358.15, min_value=250.0, max_value=600.0, step=5.0)

        # --- Model ---
        st.subheader("Model")
        model_type = st.radio("Model Type", ["Single Zone HCCI", "Multi Zone HCCI"])
        adiabatic = st.checkbox("Adiabatic (no heat transfer)", value=False)

        nzones = 1
        if model_type == "Multi Zone HCCI":
            nzones = st.selectbox("Number of Zones", options=[10, 20, 40], index=0)
            if "Gasoline" in mech_name:
                st.caption(f"{nzones} zones x 312 species — runtime grows with zone count")

        # --- Run button ---
        st.divider()
        run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

        # --- Saved runs (read-only list) ---
        if st.session_state['saved_runs']:
            st.divider()
            st.subheader("Saved Runs")
            for i, run in enumerate(st.session_state['saved_runs']):
                st.text(f"{i + 1}. {run['label']}")

    # ── Main area ────────────────────────────────────────────────────────
    if run_clicked:
        _run_simulation(
            phi=phi, egr=egr,
            bore_mm=bore_mm, stroke_mm=stroke_mm,
            con_rod_mm=con_rod_mm, comp_ratio=comp_ratio,
            rpm=rpm, T_init=T_init, P_init_bar=P_init_bar, T_wall=T_wall,
            model_type=model_type, adiabatic=adiabatic, nzones=nzones,
            mech_name=mech_name, prf_octane=prf_octane,
        )
    elif 'results' in st.session_state:
        # Show cached results if available
        _display_results()
    else:
        st.info("Configure parameters in the sidebar and click **Run Simulation**.")


def _run_simulation(*, phi, egr, bore_mm, stroke_mm, con_rod_mm, comp_ratio,
                    rpm, T_init, P_init_bar, T_wall, model_type, adiabatic, nzones,
                    mech_name, prf_octane=85):
    """Build config, run simulation, cache and display results."""
    preset = MECHANISM_PRESETS[mech_name]

    # Build config
    config = EngineConfig.from_yaml()
    config.mechanism = preset["file"]
    if "Nissan" in mech_name:
        frac = prf_octane / 100.0
        config.fuel = {"C8H18": frac, "C7H16": 1.0 - frac}
    else:
        config.fuel = preset["fuel"]
    config.phi = phi
    config.egr = egr
    config.bore = bore_mm / 1000.0
    config.stroke = stroke_mm / 1000.0
    config.con_rod = con_rod_mm / 1000.0
    config.comp_ratio = comp_ratio
    config.speed = float(rpm)
    config.temperature = T_init
    config.pressure = P_init_bar * 1e5
    config.wall_temp = T_wall
    config.adiabatic = adiabatic
    config.model_type = "single" if model_type == "Single Zone HCCI" else "multi"
    config.nzones = nzones

    # Use CVODE (SUNDIALS) for all cases
    config.method = "CVODE"

    progress_bar = st.progress(0, text="Initializing simulation...")

    try:
        progress_bar.progress(5, text="Setting up chemistry...")
        sim = EngineSimulation(config)
        y0 = sim.setup_initial_state()

        progress_bar.progress(10, text="Solving ODE system...")
        sol = sim.solver.solve_closed_cycle(
            y0=y0, rpm=config.speed, T_wall=config.wall_temp,
            ca_start=config.start_ca, ca_end=config.end_ca,
        )

        progress_bar.progress(90, text="Processing results...")
        results = SimulationResults.from_solver_output(
            sol, sim.chemistry.gas.species_names,
            model_type=config.model_type, mechanism=config.mechanism,
        )

        # Cache results in session state
        st.session_state['results'] = results
        st.session_state['config'] = config
        st.session_state['solver_output'] = sol
        st.session_state['species_names'] = sim.chemistry.gas.species_names
        st.session_state['mech_name'] = mech_name
        st.session_state['major_species'] = preset["major_species"]

        progress_bar.progress(100, text="Done!")
        progress_bar.empty()

        _display_results()

    except Exception as e:
        progress_bar.empty()
        st.error(f"Simulation failed: {e}")
        st.exception(e)


def _display_results():
    """Display cached simulation results."""
    results = st.session_state['results']
    config = st.session_state['config']
    saved_runs = st.session_state.get('saved_runs', [])

    mech_name = st.session_state.get('mech_name', DEFAULT_MECHANISM)
    major_species = st.session_state.get('major_species', MECHANISM_PRESETS[DEFAULT_MECHANISM]["major_species"])

    # Build the runs list for plotting
    current_label = _make_run_label(config, mech_name)
    runs = [(r['label'], r['results']) for r in saved_runs]
    runs.append((current_label, results))

    # Collect all major species across saved runs + current
    all_major = list(major_species)
    for r in saved_runs:
        for sp in r.get('major_species', []):
            if sp not in all_major:
                all_major.append(sp)

    has_comparison = len(runs) > 1

    # ── Performance metrics ──────────────────────────────────────────
    if has_comparison:
        rows = []
        for label, res in runs:
            perf = res.calculate_performance()
            rows.append({
                'Run': label,
                'IMEP [bar]': f"{perf['imep']:.2f}",
                'Peak P [bar]': f"{perf['peak_pressure']:.1f}",
                'Peak T [K]': f"{perf['peak_temperature']:.0f}",
                'Work [J]': f"{perf['indicated_work']:.1f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index('Run'), use_container_width=True)
    else:
        perf = results.calculate_performance()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("IMEP", f"{perf['imep']:.2f} bar")
        col2.metric("Peak Pressure", f"{perf['peak_pressure']:.1f} bar")
        col3.metric("Peak Temperature", f"{perf['peak_temperature']:.0f} K")
        col4.metric("Indicated Work", f"{perf['indicated_work']:.1f} J")

    # ── Comparison controls ─────────────────────────────────────────
    btn_cols = st.columns([1, 1, 4])
    with btn_cols[0]:
        save_clicked = st.button("Save Run for Comparison")
    with btn_cols[1]:
        clear_clicked = st.button("Clear All Saved") if saved_runs else False

    if save_clicked:
        if len(saved_runs) >= MAX_SAVED_RUNS:
            st.warning(f"Maximum {MAX_SAVED_RUNS} saved runs. Clear some first.")
        else:
            saved_runs.append({
                'label': current_label,
                'results': results,
                'major_species': list(major_species),
            })
            st.session_state['saved_runs'] = saved_runs
            st.rerun()
    if clear_clicked:
        st.session_state['saved_runs'] = []
        st.rerun()

    # ── Main results dashboard (overlay all runs) ────────────────────
    fig = plot_results_dashboard(runs, major_species=all_major)
    st.plotly_chart(fig, use_container_width=True)

    # ── Heat release rate and cumulative heat release ─────────────────
    st.subheader("Heat Release")
    fig_hr = plot_heat_release(runs)
    st.plotly_chart(fig_hr, use_container_width=True)

    # ── Multizone zone temperatures (current run only) ───────────────
    if config.model_type == "multi":
        sol = st.session_state['solver_output']
        species_names = st.session_state['species_names']
        nsp = len(species_names)
        nzones = config.nzones

        zone_temps = np.zeros((nzones, len(results.crank_angle)))
        for i in range(nzones):
            zone_temps[i, :] = sol['y'][3 + i * (nsp + 1), :]

        st.subheader("Zone Temperature Stratification")
        fig_zones = plot_zone_temperatures(results.crank_angle, zone_temps, nzones)
        st.plotly_chart(fig_zones, use_container_width=True)

        # Zone stats at TDC
        tdc_idx = np.argmin(np.abs(results.crank_angle))
        stratification = zone_temps[0, :] - zone_temps[-1, :]
        max_strat = np.max(np.abs(stratification))

        col1, col2 = st.columns(2)
        col1.metric("Max Stratification", f"{max_strat:.0f} K")
        col2.metric(
            "Spread at TDC",
            f"{zone_temps[:, tdc_idx].max() - zone_temps[:, tdc_idx].min():.0f} K",
        )


if __name__ == "__main__":
    main()
