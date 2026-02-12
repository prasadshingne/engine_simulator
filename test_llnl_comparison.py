"""Compare single-zone vs multizone Cantera solver with LLNL 312sp mechanism."""
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine_sim.engine.geometry import GeometryParams
from engine_sim.engine.heat_transfer import HeatTransfer, WoschniParams
from engine_sim.models.chemistry import Chemistry, ChemistryParams
from engine_sim.simulation.solver_cantera import (
    CanteraEngineSolver, CanteraSolverParams,
    CanteraMultizoneEngineSolver, CanteraMultizoneSolverParams
)

# Engine parameters
geom = GeometryParams(bore=0.086, stroke=0.086, con_rod=0.143, comp_ratio=14.0)
T_init, P_init = 450.0, 1.5e5
rpm, T_wall = 1200, 450.0
nzones = 10
ht = HeatTransfer(geom, WoschniParams(C_scale=15.0))

llnl_params = ChemistryParams(
    mechanism="data/mechanisms/gasoline_surrogate_323.yaml",
    fuel={"IC8H18": 0.56, "NC7H16": 0.17, "C6H5CH3": 0.27},
    phi=0.4, egr=0.0
)

results = {}

# ========== Case 1: Single-zone Cantera Woschni ==========
print("\n" + "="*60)
print("Case 1: Single-zone Cantera Woschni (LLNL 312sp)")
print("="*60)
chem1 = Chemistry(llnl_params)
Y0 = chem1.setup_initial_mixture(T=T_init, P=P_init)
nsp = len(Y0)
V0 = geom.cylinder_volume(np.deg2rad(-180))
chem1.gas.TPY = T_init, P_init, Y0
rho0 = chem1.gas.density
m0 = rho0 * V0

# Single-zone y0: [T, V, P, m, Y...]
y0_single = np.zeros(4 + nsp)
y0_single[0] = T_init
y0_single[1] = V0
y0_single[2] = P_init
y0_single[3] = m0
y0_single[4:] = Y0

t0 = time.time()
solver1 = CanteraEngineSolver(
    geom=geom, chemistry=chem1, heat_transfer=ht,
    params=CanteraSolverParams(adiabatic=False, verbose=True, rtol=1e-6, atol=1e-12)
)
r1 = solver1.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0_single.copy()
)
t_single = time.time() - t0
results['Single-zone Woschni'] = r1
print(f"  Wall time: {t_single:.1f} s")

# ========== Case 2: Multizone Cantera Woschni ==========
print("\n" + "="*60)
print(f"Case 2: Multizone Cantera Woschni (LLNL 312sp, {nzones} zones)")
print("="*60)
chem2 = Chemistry(llnl_params)
Y0_mz = chem2.setup_initial_mixture(T=T_init, P=P_init)

# Multizone y0: [T_bulk, P_bulk, V, T_z0, Y_z0, ..., T_zN, Y_zN, Y_bulk]
y0_multi = np.zeros(3 + nzones * (nsp + 1) + nsp)
y0_multi[0] = T_init
y0_multi[1] = P_init
y0_multi[2] = V0
for i in range(nzones):
    y0_multi[3 + i * (nsp + 1)] = T_init
    y0_multi[3 + i * (nsp + 1) + 1:3 + (i + 1) * (nsp + 1)] = Y0_mz
y0_multi[3 + nzones * (nsp + 1):] = Y0_mz

print(f"  Multizone state vector: {len(y0_multi)} variables")

t0 = time.time()
solver2 = CanteraMultizoneEngineSolver(
    geom=geom, chemistry=chem2, heat_transfer=ht,
    params=CanteraMultizoneSolverParams(
        adiabatic=False, nzones=nzones, verbose=True,
        rtol=1e-6, atol=1e-12, max_steps=50000
    )
)
r2 = solver2.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0_multi.copy()
)
t_multi = time.time() - t0
results['Multizone Woschni'] = r2
print(f"  Wall time: {t_multi:.1f} s")

# ========== Results Summary ==========
print("\n" + "="*60)
print(f"Results Summary — LLNL 312sp Woschni ({nzones} zones)")
print("="*60)
for name, r in results.items():
    ca = r['ca']
    if name.startswith('Single'):
        T = r['y'][0]
        P = r['y'][2]
    else:
        T = r['y'][0]
        P = r['y'][1]
    print(f"\n  {name}:")
    print(f"    Peak T = {np.max(T):.1f} K @ CA={ca[np.argmax(T)]:.1f}°")
    print(f"    Peak P = {np.max(P)/1e5:.2f} bar")
    print(f"    T_EVO = {T[-1]:.1f} K")
    print(f"    Success: {r['success']}")

if 'Multizone Woschni' in results and results['Multizone Woschni']['success']:
    r = results['Multizone Woschni']
    ca = r['ca']
    print(f"\n  Multizone zone detail:")
    tdc_idx = np.argmin(np.abs(ca))
    for i in range(nzones):
        T_zi = r['y'][3 + i * (nsp + 1)]
        label = "core" if i == 0 else ("wall" if i == nzones - 1 else f"  {i}")
        print(f"    Zone {i} ({label}): TDC={T_zi[tdc_idx]:.1f} K, peak={np.max(T_zi):.1f} K")
    strat = r['y'][3] - r['y'][3 + (nzones-1)*(nsp+1)]
    print(f"    Max stratification: {np.max(np.abs(strat)):.1f} K")

print(f"\n  Timing: single={t_single:.1f}s, multi={t_multi:.1f}s")

# ========== Plot ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Column 0: Temperature vs CA ---
ax = axes[0, 0]
# Single-zone
ca_s = results['Single-zone Woschni']['ca']
T_s = results['Single-zone Woschni']['y'][0]
ax.plot(ca_s, T_s, 'b-', linewidth=2, label='Single-zone')
# Multizone bulk
if results['Multizone Woschni']['success']:
    ca_m = results['Multizone Woschni']['ca']
    T_m = results['Multizone Woschni']['y'][0]
    ax.plot(ca_m, T_m, 'r--', linewidth=2, label='Multizone (bulk)')
ax.set_xlabel('Crank Angle [deg]')
ax.set_ylabel('Temperature [K]')
ax.set_title('Bulk Temperature')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Column 1: Pressure vs CA ---
ax = axes[0, 1]
P_s = results['Single-zone Woschni']['y'][2]
ax.plot(ca_s, P_s / 1e5, 'b-', linewidth=2, label='Single-zone')
if results['Multizone Woschni']['success']:
    P_m = results['Multizone Woschni']['y'][1]
    ax.plot(ca_m, P_m / 1e5, 'r--', linewidth=2, label='Multizone (bulk)')
ax.set_xlabel('Crank Angle [deg]')
ax.set_ylabel('Pressure [bar]')
ax.set_title('Cylinder Pressure')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Column 2: Multizone zone temperatures ---
ax = axes[0, 2]
if results['Multizone Woschni']['success']:
    r = results['Multizone Woschni']
    ca_m = r['ca']
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, nzones))
    for i in range(nzones):
        T_zi = r['y'][3 + i * (nsp + 1)]
        label = f"Zone {i}" + (" (core)" if i == 0 else " (wall)" if i == nzones - 1 else "")
        ax.plot(ca_m, T_zi, color=colors[i], linewidth=1.5, label=label)
    ax.set_xlabel('Crank Angle [deg]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Multizone Zone Temperatures ({nzones} zones)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Multizone failed', transform=ax.transAxes, ha='center')

# --- Row 1, Col 0: P-V diagram ---
ax = axes[1, 0]
V_s = results['Single-zone Woschni']['y'][1]
ax.plot(V_s * 1e6, P_s / 1e5, 'b-', linewidth=2, label='Single-zone')
if results['Multizone Woschni']['success']:
    V_m = results['Multizone Woschni']['y'][2]
    ax.plot(V_m * 1e6, P_m / 1e5, 'r--', linewidth=2, label='Multizone')
ax.set_xlabel('Volume [cm³]')
ax.set_ylabel('Pressure [bar]')
ax.set_title('P-V Diagram')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Row 1, Col 1: Major species (single-zone) ---
ax = axes[1, 1]
species_names = chem1.gas.species_names
major = ['IC8H18', 'NC7H16', 'C6H5CH3', 'O2', 'CO2', 'H2O', 'CO']
for name in major:
    if name in species_names:
        idx = species_names.index(name)
        Y_sp = results['Single-zone Woschni']['y'][4 + idx]
        if np.max(Y_sp) > 1e-6:
            ax.plot(ca_s, Y_sp, linewidth=1.5, label=name)
ax.set_xlabel('Crank Angle [deg]')
ax.set_ylabel('Mass Fraction')
ax.set_title('Single-zone Species')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Row 1, Col 2: Heat loss comparison ---
ax = axes[1, 2]
if 'q_wall' in results['Single-zone Woschni']:
    ax.plot(ca_s, results['Single-zone Woschni']['q_wall'], 'b-',
            linewidth=1.5, label='Single-zone Q_wall')
if results['Multizone Woschni']['success'] and 'q_wall' in results['Multizone Woschni']:
    ax.plot(ca_m, results['Multizone Woschni']['q_wall'], 'r--',
            linewidth=1.5, label='Multizone Q_wall')
ax.set_xlabel('Crank Angle [deg]')
ax.set_ylabel('Heat Loss Rate [W]')
ax.set_title('Wall Heat Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(f'LLNL Gasoline Surrogate (312 species) — Single-zone vs Multizone Woschni\n'
             f'CR={geom.comp_ratio}, phi=0.4, RPM={rpm}, {nzones} zones, '
             f't_single={t_single:.1f}s, t_multi={t_multi:.1f}s',
             fontsize=13)
plt.tight_layout()
out = 'data/output/llnl_singlezone_vs_multizone_woschni.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nPlot saved to {out}")
