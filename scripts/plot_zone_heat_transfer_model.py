#!/usr/bin/env python3
"""
Generate figure explaining multizone heat transfer model.

This script creates a visualization showing:
1. Concentric zone arrangement (core to wall)
2. Surface area fraction distribution across zones
3. How this leads to temperature stratification
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap


def calculate_area_fractions(nzones):
    """Calculate surface area fractions for each zone (matching solver.py)."""
    Ai = np.zeros(nzones)
    for i in range(nzones):
        if nzones > 1:
            # Formula from solver.py line 463
            Ai[i] = 3 * (2 * (i - 1)) ** 2 / (2 * nzones * (nzones - 1) * (2 * nzones - 1))
        else:
            Ai[i] = 1.0
    return Ai


def main():
    nzones = 10

    # Calculate area fractions
    Ai = calculate_area_fractions(nzones)

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(14, 6))

    # Left: Schematic of concentric zones
    ax1 = fig.add_subplot(121)

    # Create concentric circles representing zones
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, nzones))
    max_radius = 1.0

    for i in range(nzones - 1, -1, -1):
        # Outer zones have larger radii
        radius = max_radius * (i + 1) / nzones
        circle = Circle((0, 0), radius, facecolor=colors[i], edgecolor='black',
                        linewidth=0.5, alpha=0.8)
        ax1.add_patch(circle)

    # Add zone labels
    for i in range(nzones):
        radius = max_radius * (i + 0.5) / nzones
        if i == 0:
            label = f'Zone 1\n(Core)\nAi=0'
        elif i == nzones - 1:
            label = f'Zone {nzones}\n(Wall)\nAi={Ai[i]:.3f}'
        else:
            label = f'{i+1}'
        ax1.text(0, radius * 0.7 if i < 5 else -radius * 0.7, label,
                ha='center', va='center', fontsize=8 if i in [0, nzones-1] else 7,
                fontweight='bold' if i in [0, nzones-1] else 'normal')

    # Add wall indication
    wall_circle = Circle((0, 0), max_radius * 1.08, facecolor='none',
                         edgecolor='brown', linewidth=8, linestyle='-')
    ax1.add_patch(wall_circle)
    ax1.text(0, 1.2, 'Cylinder Wall (Twall)', ha='center', va='bottom',
            fontsize=10, color='brown', fontweight='bold')

    # Add heat transfer arrows
    for angle in [45, 135, 225, 315]:
        rad = np.deg2rad(angle)
        x_start = max_radius * 0.95 * np.cos(rad)
        y_start = max_radius * 0.95 * np.sin(rad)
        x_end = max_radius * 1.15 * np.cos(rad)
        y_end = max_radius * 1.15 * np.sin(rad)
        arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                               arrowstyle='->', mutation_scale=15,
                               color='red', linewidth=2)
        ax1.add_patch(arrow)

    ax1.text(1.3, 0, 'Q_wall', ha='left', va='center', fontsize=11,
            color='red', fontweight='bold')

    ax1.set_xlim(-1.5, 1.8)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Concentric Zone Arrangement\n(Cross-section view)', fontsize=12, fontweight='bold')

    # Right: Area fraction distribution
    ax2 = fig.add_subplot(122)

    zones = np.arange(1, nzones + 1)
    bars = ax2.bar(zones, Ai, color=colors, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Zone Number', fontsize=11)
    ax2.set_ylabel('Surface Area Fraction (Ai)', fontsize=11)
    ax2.set_title('Heat Transfer Area Distribution\n', fontsize=12, fontweight='bold')
    ax2.set_xticks(zones)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add annotations
    ax2.annotate('Core zone:\nNo wall contact\n(Ai = 0)',
                xy=(1, 0), xytext=(2.5, 0.15),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax2.annotate('Wall zone:\nMaximum heat loss',
                xy=(nzones, Ai[-1]), xytext=(nzones - 2, Ai[-1] + 0.05),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Add formula text box
    formula_text = (
        r"$A_i = \frac{3(2(i-1))^2}{2 N_{zones}(N_{zones}-1)(2N_{zones}-1)}$"
        "\n\n"
        "Zone 1 (core): Ai = 0 (no wall contact)\n"
        "Zone N (wall): Maximum wall exposure\n"
        "\n"
        "Result: Core zones stay hot,\n"
        "wall zones cool faster"
    )
    ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    # Add total check
    total = np.sum(Ai)
    ax2.text(0.98, 0.02, f'Sum of Ai = {total:.4f}', transform=ax2.transAxes,
            fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Save figure
    from pathlib import Path
    output_path = Path(__file__).parent.parent / 'data' / 'output' / 'multizone_heat_transfer_model.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
