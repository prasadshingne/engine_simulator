"""Test pressure equalization approach at T_init=470K and 475K."""
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

llnl_params_base = dict(
    mechanism="data/mechanisms/gasoline_surrogate_323.yaml",
    fuel={"IC8H18": 0.5413, "NC7H16": 0.1488, "C6H5CH3": 0.2738, "C5H10-2": 0.0361},
)

geom = GeometryParams(bore=0.086, stroke=0.086, con_rod=0.1455, comp_ratio=12.5)
nzones = 10

for T_init in [470.0, 475.0]:
    print(f"\n{'='*70}")
    print(f"  T_init = {T_init} K")
    print(f"{'='*70}")

    ht = HeatTransfer(geom, WoschniParams(C_scale=1.5))
    chem_params = ChemistryParams(**llnl_params_base, phi=0.5, egr=0.3)

    # --- Single-zone ---
    chem_s = Chemistry(chem_params)
    Y0 = chem_s.setup_initial_mixture(T=T_init, P=1.013e5)
    nsp = len(Y0)
    V0 = geom.cylinder_volume(np.deg2rad(-180))
    chem_s.gas.TPY = T_init, 1.013e5, Y0
    m0 = chem_s.gas.density * V0
    y0_s = np.zeros(4 + nsp)
    y0_s[0] = T_init; y0_s[1] = V0; y0_s[2] = 1.013e5; y0_s[3] = m0
    y0_s[4:] = Y0

    t0 = time.time()
    solver_s = CanteraEngineSolver(
        geom=geom, chemistry=chem_s, heat_transfer=ht,
        params=CanteraSolverParams(adiabatic=False, verbose=False, rtol=1e-6, atol=1e-12)
    )
    r_s = solver_s.solve_closed_cycle(
        rpm=2000, T_wall=358.15, ca_start=-180, ca_end=180, y0=y0_s.copy()
    )
    sz_P = np.max(r_s['y'][2]) / 1e5
    sz_T = np.max(r_s['y'][0])
    print(f"\n  Single-zone: Peak P={sz_P:.2f} bar, Peak T={sz_T:.1f} K ({time.time()-t0:.1f}s)")

    # --- Multizone ---
    chem_m = Chemistry(chem_params)
    Y0_m = chem_m.setup_initial_mixture(T=T_init, P=1.013e5)
    y0_m = np.zeros(3 + nzones * (nsp + 1) + nsp)
    y0_m[0] = T_init; y0_m[1] = 1.013e5; y0_m[2] = V0
    for i in range(nzones):
        y0_m[3 + i * (nsp + 1)] = T_init
        y0_m[3 + i * (nsp + 1) + 1:3 + (i + 1) * (nsp + 1)] = Y0_m
    y0_m[3 + nzones * (nsp + 1):] = Y0_m

    t0 = time.time()
    solver_m = CanteraMultizoneEngineSolver(
        geom=geom, chemistry=chem_m, heat_transfer=ht,
        params=CanteraMultizoneSolverParams(
            adiabatic=False, nzones=nzones, verbose=True,
            rtol=1e-6, atol=1e-12, max_steps=50000,
        )
    )
    try:
        r_m = solver_m.solve_closed_cycle(
            rpm=2000, T_wall=358.15, ca_start=-180, ca_end=180, y0=y0_m.copy()
        )
        elapsed = time.time() - t0
        mz_P = np.max(r_m['y'][1]) / 1e5
        mz_T = np.max(r_m['y'][0])
        dP = (mz_P - sz_P) / sz_P * 100
        status = "OK" if r_m['success'] else f"FAIL@CA={r_m['ca'][-1]:.0f}"
        print(f"\n  Multizone: Peak P={mz_P:.2f} bar ({dP:+.1f}%), Peak T={mz_T:.1f} K, {elapsed:.1f}s [{status}]")
        print(f"  SZ-MZ gap: {abs(dP):.1f}%")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  Multizone: EXCEPTION after {elapsed:.1f}s — {str(e)[:100]}")
