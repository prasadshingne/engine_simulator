"""Fine K sweep between 1e-6 and 1e-5 for coupling walls."""
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from engine_sim.engine.geometry import GeometryParams
from engine_sim.engine.heat_transfer import HeatTransfer, WoschniParams
from engine_sim.models.chemistry import Chemistry, ChemistryParams
from engine_sim.simulation.solver_cantera import (
    CanteraEngineSolver, CanteraSolverParams,
    CanteraMultizoneEngineSolver, CanteraMultizoneSolverParams
)

# Engine parameters (same as test_llnl_comparison.py)
geom = GeometryParams(bore=0.086, stroke=0.086, con_rod=0.143, comp_ratio=14.0)
T_init, P_init = 450.0, 1.5e5
rpm, T_wall = 1200, 450.0
nzones = 10
ht = HeatTransfer(geom, WoschniParams(C_scale=15.0))

llnl_params = ChemistryParams(
    mechanism="data/mechanisms/gasoline_surrogate_323.yaml",
    fuel={"IC8H18": 0.5413, "NC7H16": 0.1488, "C6H5CH3": 0.2738, "C5H10-2": 0.0361},
    phi=0.4, egr=0.0
)

# First get single-zone reference
print("="*60)
print("Reference: Single-zone Cantera Woschni (LLNL 312sp)")
print("="*60)
chem_ref = Chemistry(llnl_params)
Y0 = chem_ref.setup_initial_mixture(T=T_init, P=P_init)
nsp = len(Y0)
V0 = geom.cylinder_volume(np.deg2rad(-180))
chem_ref.gas.TPY = T_init, P_init, Y0
m0 = chem_ref.gas.density * V0
y0_single = np.zeros(4 + nsp)
y0_single[0] = T_init; y0_single[1] = V0; y0_single[2] = P_init; y0_single[3] = m0
y0_single[4:] = Y0

solver_ref = CanteraEngineSolver(
    geom=geom, chemistry=chem_ref, heat_transfer=ht,
    params=CanteraSolverParams(adiabatic=False, verbose=False, rtol=1e-6, atol=1e-12)
)
r_ref = solver_ref.solve_closed_cycle(
    rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0_single.copy()
)
ref_peak_P = np.max(r_ref['y'][2]) / 1e5
ref_peak_T = np.max(r_ref['y'][0])
print(f"  Single-zone: Peak P = {ref_peak_P:.2f} bar, Peak T = {ref_peak_T:.1f} K")

# K sweep
K_values = [1e-6, 2e-6, 3e-6, 5e-6, 7e-6, 1e-5]

print(f"\n{'='*60}")
print(f"K Sweep: {nzones} zones, LLNL 312sp")
print(f"{'='*60}")
print(f"{'K':>10s}  {'Status':>10s}  {'Peak P':>10s}  {'dP vs SZ':>10s}  {'Peak T':>10s}  {'Time':>8s}  {'dP_zone%':>10s}")
print("-" * 80)

for K in K_values:
    chem = Chemistry(llnl_params)
    Y0_mz = chem.setup_initial_mixture(T=T_init, P=P_init)

    y0_multi = np.zeros(3 + nzones * (nsp + 1) + nsp)
    y0_multi[0] = T_init; y0_multi[1] = P_init; y0_multi[2] = V0
    for i in range(nzones):
        y0_multi[3 + i * (nsp + 1)] = T_init
        y0_multi[3 + i * (nsp + 1) + 1:3 + (i + 1) * (nsp + 1)] = Y0_mz
    y0_multi[3 + nzones * (nsp + 1):] = Y0_mz

    t0 = time.time()
    solver = CanteraMultizoneEngineSolver(
        geom=geom, chemistry=chem, heat_transfer=ht,
        params=CanteraMultizoneSolverParams(
            adiabatic=False, nzones=nzones, verbose=False,
            rtol=1e-6, atol=1e-12, max_steps=50000,
            pressure_coupling_coeff=K
        )
    )
    try:
        r = solver.solve_closed_cycle(
            rpm=rpm, T_wall=T_wall, ca_start=-180, ca_end=180, y0=y0_multi.copy()
        )
        elapsed = time.time() - t0
        peak_P = np.max(r['y'][1]) / 1e5
        peak_T = np.max(r['y'][0])
        dP_pct = (peak_P - ref_peak_P) / ref_peak_P * 100

        # Check pressure spread between zones at peak pressure
        # (We can't easily get zone pressures from output, but we can check
        # if the run succeeded and how close peak P is to reference)
        status = "OK" if r['success'] else f"FAIL@{r['ca'][-1]:.0f}"
        print(f"{K:>10.1e}  {status:>10s}  {peak_P:>8.2f} bar  {dP_pct:>+8.1f}%  {peak_T:>8.1f} K  {elapsed:>6.1f} s")
    except Exception as e:
        elapsed = time.time() - t0
        err_msg = str(e)[:40]
        print(f"{K:>10.1e}  {'ERROR':>10s}  {'--':>10s}  {'--':>10s}  {'--':>10s}  {elapsed:>6.1f} s  {err_msg}")

print(f"\nReference single-zone: Peak P = {ref_peak_P:.2f} bar, Peak T = {ref_peak_T:.1f} K")
