"""Plotly-based interactive plotting for Streamlit GUI."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine_sim.simulation.results import SimulationResults

DASH_STYLES = ['solid', 'dash', 'dot', 'dashdot']


def plot_results_dashboard(runs) -> go.Figure:
    """Create a 2x2 dashboard of simulation results.

    Parameters
    ----------
    runs : SimulationResults or list of (label, SimulationResults)
        A single result or a list of labelled results to overlay.
    """
    # Normalise input
    if isinstance(runs, SimulationResults):
        runs = [('Run', runs)]

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
        species_colors = {
            'C8H18': '#ff7f0e', 'O2': '#2ca02c',
            'CO2': '#9467bd', 'H2O': '#17becf',
        }
        for sp_name in ['C8H18', 'O2', 'CO2', 'H2O']:
            if sp_name not in results.species_names:
                continue
            idx = results.species_names.index(sp_name)
            # Show species legend for first run only (color key)
            show_sp = (run_idx == 0)
            fig.add_trace(
                go.Scatter(
                    x=results.crank_angle, y=results.species[idx],
                    mode='lines',
                    name=sp_name,
                    line=dict(
                        color=species_colors.get(sp_name),
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
