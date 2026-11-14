"""Validate manual multizone solver with all bug fixes (theta, dVdt, tolerances)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine_sim.engine.geometry import GeometryParams
from engine_sim.engine.heat_transfer import HeatTransfer, WoschniParams
from engine_sim.models.chemistry import Chemistry, ChemistryParams
from engine_sim.simulation.solver import EngineSolver, SolverParams

geom = GeometryParams(bore=0.086, stroke=0.086, con_rod=0.143, comp_ratio=14.0)
T_init, P_init = 450.0, 1.5e5
rpm, T_wall = 1200, 450.0
nzones = 10
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
y0[0] = T_init   # Bulk T
y0[1] = P_init   # Bulk P
y0[2] = V0       # Volume
for i in range(nzones):
    y0[3 + i * (nsp + 1)] = T_init
    y0[3 + i * (nsp + 1) + 1:3 + (i + 1) * (nsp + 1)] = Y0
y0[3 + nzones * (nsp + 1):] = Y0

print(f"State vector size: {len(y0)} ({nzones} zones, {nsp} species)")
print(f"Initial: T={T_init} K, P={P_init/1e5} bar, V={V0*1e6:.2f} cm3")

results = {}

# --- Case 1: Multizone Adiabatic ---
print("\n" + "="*60)
print(f"Case 1: Multizone Adiabatic ({nzones} zones, Nissan PRF)")
print("="*60)
chem1 = Chemistry(nissan_params)
chem1.setup_initial_mixture(T=T_init, P=P_init)
solver1 = EngineSolver(
    geom=geom, heat_transfer=ht, chemistry=chem1,
    params=SolverParams(adiabatic=True, model_type="multi", nzones=nzones, show_progress=True)
)
results['Adiabatic'] = solver1.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0.copy()
)

# --- Case 2: Multizone Woschni ---
print("\n" + "="*60)
print(f"Case 2: Multizone Woschni ({nzones} zones, Nissan PRF)")
print("="*60)
chem2 = Chemistry(nissan_params)
chem2.setup_initial_mixture(T=T_init, P=P_init)
solver2 = EngineSolver(
    geom=geom, heat_transfer=ht, chemistry=chem2,
    params=SolverParams(adiabatic=False, model_type="multi", nzones=nzones, show_progress=True)
)
results['Woschni'] = solver2.solve_closed_cycle(
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

# Validation checks
print("\n" + "="*60)
print("Validation Checks")
print("="*60)
T_adia = np.max(results['Adiabatic']['y'][0])
T_wos = np.max(results['Woschni']['y'][0])
print(f"  Woschni peak T < Adiabatic peak T? "
      f"{'PASS' if T_wos < T_adia else 'FAIL'} ({T_wos:.1f} vs {T_adia:.1f})")

# Check zone ordering for Woschni (core should be hottest)
wos_zones = np.zeros((nzones, len(results['Woschni']['ca'])))
for i in range(nzones):
    wos_zones[i, :] = results['Woschni']['y'][3 + i * (nsp + 1)]
tdc_idx = np.argmin(np.abs(results['Woschni']['ca']))
core_T = wos_zones[0, tdc_idx]
wall_T = wos_zones[-1, tdc_idx]
print(f"  Core > Wall at TDC (Woschni)? "
      f"{'PASS' if core_T > wall_T else 'FAIL'} ({core_T:.1f} vs {wall_T:.1f})")

# Adiabatic: all zones should be equal (no heat transfer differentiation)
adia_zones = np.zeros((nzones, len(results['Adiabatic']['ca'])))
for i in range(nzones):
    adia_zones[i, :] = results['Adiabatic']['y'][3 + i * (nsp + 1)]
adia_spread = np.max(adia_zones[:, tdc_idx]) - np.min(adia_zones[:, tdc_idx])
print(f"  Adiabatic zone spread at TDC? {adia_spread:.2f} K (should be ~0)")

# --- Plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
    axes[0, col].set_title(f'{name} — Zone Temperatures')
    axes[0, col].legend(fontsize=8)
    axes[0, col].grid(True, alpha=0.3)

    # Bulk T and P
    axes[1, col].plot(ca, r['y'][0], 'b-', linewidth=1.5, label='T_bulk')
    ax2 = axes[1, col].twinx()
    ax2.plot(ca, r['y'][1] / 1e5, 'r-', linewidth=1.5, label='P_bulk')
    axes[1, col].set_ylabel('Temperature [K]', color='blue')
    ax2.set_ylabel('Pressure [bar]', color='red')
    axes[1, col].set_xlabel('Crank Angle [deg]')
    axes[1, col].set_title(f'{name} — Bulk T & P')
    axes[1, col].grid(True, alpha=0.3)

plt.suptitle(f'Multizone Validation — {nzones} zones, Nissan PRF, CR={geom.comp_ratio}, '
             f'phi=0.4, RPM={rpm}', fontsize=13)
plt.tight_layout()
out = 'data/output/multizone_test/multizone_validation.png'
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nPlot saved to {out}")
