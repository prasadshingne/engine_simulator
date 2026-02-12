"""Test Cantera multizone solver against manual multizone solver."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine_sim.engine.geometry import GeometryParams
from engine_sim.engine.heat_transfer import HeatTransfer, WoschniParams
from engine_sim.models.chemistry import Chemistry, ChemistryParams
from engine_sim.simulation.solver_cantera import (
    CanteraMultizoneEngineSolver, CanteraMultizoneSolverParams
)
from engine_sim.simulation.solver import EngineSolver, SolverParams

# Engine parameters (same as validation test)
geom = GeometryParams(bore=0.086, stroke=0.086, con_rod=0.143, comp_ratio=14.0)
T_init, P_init = 450.0, 1.5e5
rpm, T_wall = 1200, 450.0
nzones = 5
ht = HeatTransfer(geom, WoschniParams(C_scale=15.0))

nissan_params = ChemistryParams(
    mechanism="data/mechanisms/Nissan_chem.yaml",
    fuel="C8H18", phi=0.4, egr=0.0
)

# Build multizone initial state: [T, P, V, T_z0, Y_z0, ..., T_zN, Y_zN, Y_bulk]
chem_ref = Chemistry(nissan_params)
Y0 = chem_ref.setup_initial_mixture(T=T_init, P=P_init)
nsp = len(Y0)
V0 = geom.cylinder_volume(np.deg2rad(-180))

y0 = np.zeros(3 + nzones * (nsp + 1) + nsp)
y0[0] = T_init
y0[1] = P_init
y0[2] = V0
for i in range(nzones):
    y0[3 + i * (nsp + 1)] = T_init
    y0[3 + i * (nsp + 1) + 1:3 + (i + 1) * (nsp + 1)] = Y0
y0[3 + nzones * (nsp + 1):] = Y0

print(f"State vector size: {len(y0)} ({nzones} zones, {nsp} species)")
print(f"Initial: T={T_init} K, P={P_init/1e5} bar, V={V0*1e6:.2f} cm3")

results = {}

# --- Case 1: Cantera Multizone Adiabatic ---
print("\n" + "="*60)
print("Case 1: Cantera Multizone Adiabatic (Nissan PRF)")
print("="*60)
chem1 = Chemistry(nissan_params)
chem1.setup_initial_mixture(T=T_init, P=P_init)
solver1 = CanteraMultizoneEngineSolver(
    geom=geom, chemistry=chem1, heat_transfer=ht,
    params=CanteraMultizoneSolverParams(
        adiabatic=True, nzones=nzones, verbose=True,
        rtol=1e-6, atol=1e-12
    )
)
results['Cantera Adiabatic'] = solver1.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0.copy()
)

# --- Case 2: Cantera Multizone Woschni ---
print("\n" + "="*60)
print("Case 2: Cantera Multizone Woschni (Nissan PRF)")
print("="*60)
chem2 = Chemistry(nissan_params)
chem2.setup_initial_mixture(T=T_init, P=P_init)
solver2 = CanteraMultizoneEngineSolver(
    geom=geom, chemistry=chem2, heat_transfer=ht,
    params=CanteraMultizoneSolverParams(
        adiabatic=False, nzones=nzones, verbose=True,
        rtol=1e-6, atol=1e-12
    )
)
results['Cantera Woschni'] = solver2.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0.copy()
)

# --- Case 3: Manual Multizone Adiabatic (reference) ---
print("\n" + "="*60)
print("Case 3: Manual Multizone Adiabatic (reference)")
print("="*60)
chem3 = Chemistry(nissan_params)
chem3.setup_initial_mixture(T=T_init, P=P_init)
solver3 = EngineSolver(
    geom=geom, heat_transfer=ht, chemistry=chem3,
    params=SolverParams(adiabatic=True, model_type="multi", nzones=nzones, show_progress=True)
)
results['Manual Adiabatic'] = solver3.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0.copy()
)

# --- Case 4: Manual Multizone Woschni (reference) ---
print("\n" + "="*60)
print("Case 4: Manual Multizone Woschni (reference)")
print("="*60)
chem4 = Chemistry(nissan_params)
chem4.setup_initial_mixture(T=T_init, P=P_init)
solver4 = EngineSolver(
    geom=geom, heat_transfer=ht, chemistry=chem4,
    params=SolverParams(adiabatic=False, model_type="multi", nzones=nzones, show_progress=True)
)
results['Manual Woschni'] = solver4.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0.copy()
)

# --- Analysis ---
print("\n" + "="*60)
print("Results Summary")
print("="*60)
for name, r in results.items():
    ca = r['ca']
    T_bulk = r['y'][0]
    P_bulk = r['y'][1]
    print(f"\n  {name}:")
    print(f"    Bulk Peak T = {np.max(T_bulk):.1f} K @ CA={ca[np.argmax(T_bulk)]:.1f}")
    print(f"    Bulk Peak P = {np.max(P_bulk)/1e5:.2f} bar")
    print(f"    T_EVO = {T_bulk[-1]:.1f} K")

    # Zone temperatures
    zone_temps = np.zeros((nzones, len(ca)))
    for i in range(nzones):
        zone_temps[i, :] = r['y'][3 + i * (nsp + 1)]

    tdc_idx = np.argmin(np.abs(ca))
    print(f"    Zone temps at TDC (CA={ca[tdc_idx]:.1f}):")
    for i in range(nzones):
        label = "core" if i == 0 else ("wall" if i == nzones - 1 else f"  {i}")
        print(f"      Zone {i} ({label}): {zone_temps[i, tdc_idx]:.1f} K  "
              f"(peak: {np.max(zone_temps[i]):.1f} K)")

    strat = zone_temps[0, :] - zone_temps[-1, :]
    print(f"    Max stratification (core-wall): {np.max(np.abs(strat)):.1f} K")

    # Heat loss
    if 'q_wall' in r and r['q_wall'] is not None:
        dt = np.diff(r['t'])
        Q_total = np.sum(0.5 * (r['q_wall'][1:] + r['q_wall'][:-1]) * dt)
        print(f"    Total Q_wall = {Q_total:.2f} J")

# --- Validation Checks ---
print("\n" + "="*60)
print("Validation Checks")
print("="*60)

# 1. Woschni < Adiabatic for both solvers
for solver_name in ['Cantera', 'Manual']:
    T_adia = np.max(results[f'{solver_name} Adiabatic']['y'][0])
    T_wos = np.max(results[f'{solver_name} Woschni']['y'][0])
    status = 'PASS' if T_wos < T_adia else 'FAIL'
    print(f"  {solver_name} Woschni < Adiabatic? {status} ({T_wos:.1f} vs {T_adia:.1f})")

# 2. Cantera vs Manual agreement (within 10%)
for case in ['Adiabatic', 'Woschni']:
    T_cantera = np.max(results[f'Cantera {case}']['y'][0])
    T_manual = np.max(results[f'Manual {case}']['y'][0])
    diff_pct = abs(T_cantera - T_manual) / T_manual * 100
    status = 'PASS' if diff_pct < 10 else 'FAIL'
    print(f"  {case} Cantera vs Manual peak T: {status} "
          f"({T_cantera:.1f} vs {T_manual:.1f}, diff={diff_pct:.1f}%)")

# 3. Zone ordering for Woschni (core > wall)
for solver_name in ['Cantera', 'Manual']:
    r = results[f'{solver_name} Woschni']
    ca = r['ca']
    tdc_idx = np.argmin(np.abs(ca))
    core_T = r['y'][3, tdc_idx]
    wall_T = r['y'][3 + (nzones - 1) * (nsp + 1), tdc_idx]
    status = 'PASS' if core_T > wall_T else 'FAIL'
    print(f"  {solver_name} core > wall at TDC? {status} ({core_T:.1f} vs {wall_T:.1f})")

# --- Plot ---
fig, axes = plt.subplots(2, 4, figsize=(22, 10))

for col, (name, r) in enumerate(results.items()):
    ca = r['ca']
    zone_temps = np.zeros((nzones, len(ca)))
    for i in range(nzones):
        zone_temps[i, :] = r['y'][3 + i * (nsp + 1)]

    # Zone temperatures
    for i in range(nzones):
        label = f"Zone {i}" + (" (core)" if i == 0 else " (wall)" if i == nzones - 1 else "")
        axes[0, col].plot(ca, zone_temps[i], linewidth=1.2, label=label)
    axes[0, col].set_ylabel('Temperature [K]')
    axes[0, col].set_title(f'{name}\nZone Temperatures')
    axes[0, col].legend(fontsize=7)
    axes[0, col].grid(True, alpha=0.3)

    # Bulk T and P
    axes[1, col].plot(ca, r['y'][0], 'b-', linewidth=1.5, label='T_bulk')
    ax2 = axes[1, col].twinx()
    ax2.plot(ca, r['y'][1] / 1e5, 'r-', linewidth=1.5, label='P_bulk')
    axes[1, col].set_ylabel('Temperature [K]', color='blue')
    ax2.set_ylabel('Pressure [bar]', color='red')
    axes[1, col].set_xlabel('Crank Angle [deg]')
    axes[1, col].set_title(f'{name}\nBulk T & P')
    axes[1, col].grid(True, alpha=0.3)

plt.suptitle(f'Cantera vs Manual Multizone — {nzones} zones, Nissan PRF, CR={geom.comp_ratio}, '
             f'phi=0.4, RPM={rpm}', fontsize=13)
plt.tight_layout()
import os
out = 'data/output/multizone_test/cantera_vs_manual_multizone.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nPlot saved to {out}")
