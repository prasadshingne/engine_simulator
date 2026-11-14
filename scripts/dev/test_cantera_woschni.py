"""Combustion comparison: Manual solver (Nissan PRF 33sp) vs Cantera solver (LLNL 312sp).
Adiabatic and Woschni heat transfer for both, reactions ON."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine_sim.engine.geometry import GeometryParams
from engine_sim.engine.heat_transfer import HeatTransfer, WoschniParams
from engine_sim.models.chemistry import Chemistry, ChemistryParams
from engine_sim.simulation.solver import EngineSolver, SolverParams
from engine_sim.simulation.solver_cantera import CanteraEngineSolver, CanteraSolverParams

# Engine geometry
geom = GeometryParams(
    bore=0.086, stroke=0.086, con_rod=0.143, comp_ratio=14.0
)

T_init, P_init = 450.0, 1.5e5
rpm, T_wall = 1200, 450.0
ht = HeatTransfer(geom, WoschniParams(C_scale=15.0))

# --- Nissan PRF setup (for manual solver) ---
nissan_params = ChemistryParams(
    mechanism="data/mechanisms/Nissan_chem.yaml",
    fuel="C8H18", phi=0.4, egr=0.0
)

# Get initial state from Nissan mechanism
chem_ref = Chemistry(nissan_params)
Y0_nissan = chem_ref.setup_initial_mixture(T=T_init, P=P_init)
T0, P0 = chem_ref.gas.T, chem_ref.gas.P
V0 = geom.cylinder_volume(np.deg2rad(-180))
m0 = chem_ref.gas.density * V0
y0_nissan = np.concatenate([[T0, V0, P0, m0], Y0_nissan])

print(f"\nInitial: T={T0:.1f} K, P={P0/1e5:.2f} bar, V={V0*1e6:.2f} cm3, m={m0*1e3:.3f} g")

# --- LLNL Gasoline Surrogate setup (for Cantera solver) ---
llnl_params = ChemistryParams(
    mechanism="data/mechanisms/gasoline_surrogate_323.yaml",
    fuel={"IC8H18": 0.5413, "NC7H16": 0.1488, "C6H5CH3": 0.2738, "C5H10-2": 0.0361},
    phi=0.4, egr=0.0
)

chem_llnl_ref = Chemistry(llnl_params)
Y0_llnl = chem_llnl_ref.setup_initial_mixture(T=T_init, P=P_init)
T0_l, P0_l = chem_llnl_ref.gas.T, chem_llnl_ref.gas.P
m0_l = chem_llnl_ref.gas.density * V0
y0_llnl = np.concatenate([[T0_l, V0, P0_l, m0_l], Y0_llnl])

print(f"LLNL Initial: T={T0_l:.1f} K, P={P0_l/1e5:.2f} bar, m={m0_l*1e3:.3f} g")

results = {}

# --- Case 1: Manual Nissan PRF, Adiabatic ---
print("\n" + "="*60)
print("Case 1: Manual solver (Nissan PRF 33sp) — Adiabatic")
print("="*60)
chem1 = Chemistry(nissan_params)
chem1.setup_initial_mixture(T=T_init, P=P_init)

manual_adia = EngineSolver(
    geom=geom, heat_transfer=ht, chemistry=chem1,
    params=SolverParams(adiabatic=True, model_type="single", show_progress=True)
)
results['Nissan adiabatic'] = manual_adia.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0_nissan
)

# --- Case 2: Manual Nissan PRF, Woschni ---
print("\n" + "="*60)
print("Case 2: Manual solver (Nissan PRF 33sp) — Woschni")
print("="*60)
chem2 = Chemistry(nissan_params)
chem2.setup_initial_mixture(T=T_init, P=P_init)

manual_ht = EngineSolver(
    geom=geom, heat_transfer=ht, chemistry=chem2,
    params=SolverParams(adiabatic=False, model_type="single", show_progress=True)
)
results['Nissan Woschni'] = manual_ht.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0_nissan
)

# --- Case 3: Cantera LLNL Gasoline Surrogate, Adiabatic ---
print("\n" + "="*60)
print("Case 3: Cantera solver (LLNL 312sp) — Adiabatic")
print("="*60)
chem3 = Chemistry(llnl_params)
chem3.setup_initial_mixture(T=T_init, P=P_init)

cantera_adia = CanteraEngineSolver(
    geom=geom, chemistry=chem3,
    params=CanteraSolverParams(adiabatic=True, verbose=True)
)
results['LLNL adiabatic'] = cantera_adia.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0_llnl
)

# --- Case 4: Cantera LLNL Gasoline Surrogate, Woschni ---
print("\n" + "="*60)
print("Case 4: Cantera solver (LLNL 312sp) — Woschni")
print("="*60)
chem4 = Chemistry(llnl_params)
chem4.setup_initial_mixture(T=T_init, P=P_init)

cantera_ht = CanteraEngineSolver(
    geom=geom, chemistry=chem4, heat_transfer=ht,
    params=CanteraSolverParams(adiabatic=False, verbose=True)
)
results['LLNL Woschni'] = cantera_ht.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0_llnl
)

# --- Compute cumulative heat loss for Woschni cases ---
omega = rpm * 2 * np.pi / 60
for name, r in results.items():
    ca = r['ca']
    T = r['y'][0]
    P = r['y'][2]
    theta_arr = np.deg2rad(ca)
    # Time from crank angle
    t_arr = (theta_arr - theta_arr[0]) / omega

    Q_wall_arr = np.zeros_like(ca)
    if 'Woschni' in name:
        for i, (th, Ti, Pi) in enumerate(zip(theta_arr, T, P)):
            p_mot = P[0] * (geom.cylinder_volume(theta_arr[0]) / geom.cylinder_volume(th))**1.35
            Q_wall_arr[i], _ = ht.heat_transfer_rate(
                th, rpm, Pi, Ti, p_mot, P[0], T[0],
                geom.cylinder_volume(theta_arr[0]), T_wall
            )
    # Integrate Q_wall over time using trapezoidal rule → total heat loss in Joules
    Q_total = np.trapz(Q_wall_arr, t_arr)
    r['Q_wall_total_J'] = Q_total
    r['Q_wall_arr'] = Q_wall_arr
    r['t_arr'] = t_arr

# --- Summary ---
print("\n" + "="*60)
print("Summary (with combustion)")
print("="*60)
for name, r in results.items():
    T = r['y'][0]
    P = r['y'][2]
    ca = r['ca']
    Q_str = f"Q_loss = {r['Q_wall_total_J']:.2f} J" if r['Q_wall_total_J'] > 0 else "Q_loss = 0 (adiabatic)"
    print(f"  {name:22s}: Peak T = {np.max(T):.1f} K @ {ca[np.argmax(T)]:.1f} CA, "
          f"Peak P = {np.max(P)/1e5:.2f} bar, T_EVO = {T[-1]:.1f} K, {Q_str}")

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

colors = {
    'Nissan adiabatic': '#1f77b4',
    'Nissan Woschni':   '#d62728',
    'LLNL adiabatic':   '#2ca02c',
    'LLNL Woschni':     '#ff7f0e',
}
styles = {
    'Nissan adiabatic': '-',
    'Nissan Woschni':   '-',
    'LLNL adiabatic':   '--',
    'LLNL Woschni':     '--',
}

for name, r in results.items():
    ca = r['ca']
    T = r['y'][0]
    P = r['y'][2] / 1e5
    axes[0].plot(ca, T, styles[name], color=colors[name], linewidth=1.5, label=name)
    axes[1].plot(ca, P, styles[name], color=colors[name], linewidth=1.5, label=name)

axes[0].set_ylabel('Temperature [K]')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('HCCI Combustion — Manual (Nissan PRF 33sp) vs Cantera (LLNL 312sp)\n'
                   f'CR={geom.comp_ratio}, phi=0.4, RPM={rpm}, T_IVC={T_init} K, P_IVC={P_init/1e5} bar')

axes[1].set_ylabel('Pressure [bar]')
axes[1].set_xlabel('Crank Angle [deg]')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = 'data/output/cantera_vs_manual_combustion.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nPlot saved to: {out_path}")
