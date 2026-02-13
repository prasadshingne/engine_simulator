"""Plotly-based interactive plotting for Streamlit GUI."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine_sim.simulation.results import SimulationResults

DASH_STYLES = ['solid', 'dash', 'dot', 'dashdot']


DEFAULT_MAJOR_SPECIES = ['C8H18', 'O2', 'CO2', 'H2O']

# Colors for species (cover both Nissan PRF and LLNL Gasoline Surrogate names)
SPECIES_COLORS = {
    'C8H18': '#ff7f0e', 'IC8H18': '#ff7f0e',
    'NC7H16': '#8c564b', 'C7H16': '#8c564b',
    'C6H5CH3': '#e377c2',
    'C5H10-2': '#bcbd22',
    'O2': '#2ca02c',
    'CO2': '#9467bd',
    'H2O': '#17becf',
}


def plot_results_dashboard(runs, major_species=None) -> go.Figure:
    """Create a 2x2 dashboard of simulation results.

    Parameters
    ----------
    runs : SimulationResults or list of (label, SimulationResults)
        A single result or a list of labelled results to overlay.
    major_species : list of str, optional
        Species to plot in the species subplot. If None, uses default.
    """
    # Normalise input
    if isinstance(runs, SimulationResults):
        runs = [('Run', runs)]

    if major_species is None:
        major_species = DEFAULT_MAJOR_SPECIES

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('P-V Diagram', 'Temperature', 'Pressure', 'Major Species'),
        vertical_spacing=0.18, horizontal_spacing=0.08,
    )

    single = len(runs) == 1

    for run_idx, (label, results) in enumerate(runs):
        dash = DASH_STYLES[run_idx % len(DASH_STYLES)]

        # P-V diagram -- show legend entry once per run in comparison mode
        fig.add_trace(
            go.Scatter(
                x=results.volume * 1e6, y=results.pressure / 1e5,
                mode='lines', name=label if not single else 'P-V',
                line=dict(color='#1f77b4', width=2, dash=dash),
                showlegend=not single,
                legendgroup=label,
                legend='legend' if not single else None,
            ),
            row=1, col=1,
        )

        # Temperature vs CA
        fig.add_trace(
            go.Scatter(
                x=results.crank_angle, y=results.temperature,
                mode='lines', name='Temperature',
                line=dict(color='#d62728', width=2, dash=dash),
                showlegend=False,
                legendgroup=label,
            ),
            row=1, col=2,
        )

        # Pressure vs CA
        fig.add_trace(
            go.Scatter(
                x=results.crank_angle, y=results.pressure / 1e5,
                mode='lines', name='Pressure',
                line=dict(color='#1f77b4', width=2, dash=dash),
                showlegend=False,
                legendgroup=label,
            ),
            row=2, col=1,
        )

        # Major species
        for sp_name in major_species:
            if sp_name not in results.species_names:
                continue
            idx = results.species_names.index(sp_name)
            # Show species legend for first run that has this species
            show_sp = (run_idx == 0)
            fig.add_trace(
                go.Scatter(
                    x=results.crank_angle, y=results.species[idx],
                    mode='lines',
                    name=sp_name,
                    line=dict(
                        color=SPECIES_COLORS.get(sp_name, '#333333'),
                        width=2, dash=dash,
                    ),
                    showlegend=show_sp,
                    legendgroup=sp_name,
                    legend='legend2',
                ),
                row=2, col=2,
            )

    fig.update_xaxes(title_text='Volume [cm3]', row=1, col=1)
    fig.update_yaxes(title_text='Pressure [bar]', row=1, col=1)
    fig.update_xaxes(title_text='Crank Angle [deg]', row=1, col=2)
    fig.update_yaxes(title_text='Temperature [K]', row=1, col=2)
    fig.update_xaxes(title_text='Crank Angle [deg]', row=2, col=1)
    fig.update_yaxes(title_text='Pressure [bar]', row=2, col=1)
    fig.update_xaxes(title_text='Crank Angle [deg]', row=2, col=2)
    fig.update_yaxes(title_text='Mass Fraction [-]', row=2, col=2)

    # Species legend -- always near the species subplot (bottom-right)
    species_legend = dict(
        x=0.58, y=0.42,
        xanchor='left', yanchor='top',
        font=dict(size=11, color='black'),
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='rgba(0,0,0,0.2)',
        borderwidth=1,
    )

    if single:
        fig.update_layout(
            height=720, template='plotly_white',
            margin=dict(t=40, b=20, r=20),
            legend2=species_legend,
            showlegend=True,
        )
    else:
        # Runs legend -- top-right outside the plot area
        runs_legend = dict(
            x=1.0, y=1.0,
            xanchor='left', yanchor='top',
            font=dict(size=10, color='black'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
        )
        fig.update_layout(
            height=720, template='plotly_white',
            margin=dict(t=40, b=20, r=180),
            legend=runs_legend,
            legend2=species_legend,
            showlegend=True,
        )

    return fig


def compute_heat_release(crank_angle, pressure, volume, gamma=1.30):
    """Compute net heat release rate and cumulative heat release.

    Uses first-law analysis:
        dQ/dθ = (γ/(γ-1)) * P * dV/dθ + (1/(γ-1)) * V * dP/dθ

    Parameters
    ----------
    crank_angle : array
        Crank angle [deg]
    pressure : array
        Pressure [Pa]
    volume : array
        Volume [m³]
    gamma : float
        Ratio of specific heats (default 1.30)

    Returns
    -------
    ca_mid : array
        Crank angles at midpoints [deg]
    dQdtheta : array
        Net heat release rate [J/deg]
    Q_cum : array
        Cumulative heat release [J] (same length as ca_mid)
    Q_norm : array
        Normalized cumulative heat release [0-1]
    """
    dtheta = np.diff(crank_angle)  # [deg]
    dV = np.diff(volume)
    dP = np.diff(pressure)
    P_mid = (pressure[:-1] + pressure[1:]) / 2
    V_mid = (volume[:-1] + volume[1:]) / 2
    ca_mid = (crank_angle[:-1] + crank_angle[1:]) / 2

    g1 = gamma / (gamma - 1)
    g2 = 1 / (gamma - 1)
    dQdtheta = g1 * P_mid * dV / dtheta + g2 * V_mid * dP / dtheta

    Q_cum = np.cumsum(dQdtheta * dtheta)
    # Normalize: 0 at start of combustion, 1 at end
    Q_min = np.min(Q_cum)
    Q_max = np.max(Q_cum)
    Q_range = Q_max - Q_min
    Q_norm = (Q_cum - Q_min) / Q_range if Q_range > 0 else np.zeros_like(Q_cum)

    return ca_mid, dQdtheta, Q_cum, Q_norm


def plot_heat_release(runs) -> go.Figure:
    """Plot heat release rate and normalized cumulative heat release.

    Parameters
    ----------
    runs : list of (label, SimulationResults)
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Heat Release Rate', 'Normalized Cumulative Heat Release'),
        horizontal_spacing=0.1,
    )

    single = len(runs) == 1

    for run_idx, (label, results) in enumerate(runs):
        dash = DASH_STYLES[run_idx % len(DASH_STYLES)]
        ca_mid, dQdtheta, Q_cum, Q_norm = compute_heat_release(
            results.crank_angle, results.pressure, results.volume,
        )

        fig.add_trace(
            go.Scatter(
                x=ca_mid, y=dQdtheta,
                mode='lines', name=label if not single else 'HRR',
                line=dict(width=2, dash=dash),
                showlegend=True,
                legendgroup=label,
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=ca_mid, y=Q_norm,
                mode='lines', name=label if not single else 'Cumulative',
                line=dict(width=2, dash=dash),
                showlegend=False,
                legendgroup=label,
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text='Crank Angle [deg]', row=1, col=1)
    fig.update_yaxes(title_text='dQ/dθ [J/deg]', row=1, col=1)
    fig.update_xaxes(title_text='Crank Angle [deg]', row=1, col=2)
    fig.update_yaxes(title_text='Normalized Heat Release [-]', row=1, col=2)

    fig.update_layout(
        height=350, template='plotly_white',
        margin=dict(t=40, b=20),
        legend=dict(font=dict(size=10)),
    )
    return fig


def plot_zone_temperatures(crank_angle, zone_temps, nzones) -> go.Figure:
    """Plot individual zone temperatures for multizone simulation."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Zone Temperatures', 'Temperature Stratification'),
        horizontal_spacing=0.1,
    )

    # Generate plasma-like colors
    import plotly.express as px
    colors = px.colors.sample_colorscale('Plasma', np.linspace(0, 1, nzones))

    for i in range(nzones):
        label = f'Zone {i + 1}'
        if i == 0:
            label += ' (core)'
        elif i == nzones - 1:
            label += ' (wall)'
        fig.add_trace(
            go.Scatter(
                x=crank_angle, y=zone_temps[i, :],
                mode='lines', name=label,
                line=dict(color=colors[i], width=1.5),
            ),
            row=1, col=1,
        )

    fig.update_xaxes(title_text='Crank Angle [deg]', row=1, col=1)
    fig.update_yaxes(title_text='Temperature [K]', row=1, col=1)

    # Stratification (core - wall)
    stratification = zone_temps[0, :] - zone_temps[-1, :]
    fig.add_trace(
        go.Scatter(
            x=crank_angle, y=stratification,
            mode='lines', name='Core - Wall',
            line=dict(color='#2ca02c', width=2),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text='Crank Angle [deg]', row=1, col=2)
    fig.update_yaxes(title_text='dT (Core - Wall) [K]', row=1, col=2)

    fig.update_layout(
        height=400, template='plotly_white',
        margin=dict(t=40, b=20),
        legend=dict(font=dict(size=10)),
    )
    return fig
