"""Test adaptive K with both sets of conditions."""
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

conditions = {
    "Test (CR=14, phi=0.4, no EGR)": {
        "geom": GeometryParams(bore=0.086, stroke=0.086, con_rod=0.143, comp_ratio=14.0),
        "T_init": 450.0, "P_init": 1.5e5, "rpm": 1200, "T_wall": 450.0,
        "C_scale": 15.0, "phi": 0.4, "egr": 0.0,
    },
    "GUI (CR=12.5, phi=0.5, EGR=0.3)": {
        "geom": GeometryParams(bore=0.086, stroke=0.086, con_rod=0.1455, comp_ratio=12.5),
        "T_init": 475.0, "P_init": 1.013e5, "rpm": 2000, "T_wall": 358.15,
        "C_scale": 1.5, "phi": 0.5, "egr": 0.3,
    },
}

nzones = 10

for cond_name, cond in conditions.items():
    print(f"\n{'='*60}")
    print(f"Condition: {cond_name}")
    print(f"{'='*60}")

    geom = cond["geom"]
    ht = HeatTransfer(geom, WoschniParams(C_scale=cond["C_scale"]))
    chem_params = ChemistryParams(**llnl_params_base, phi=cond["phi"], egr=cond["egr"])

    # --- Single-zone reference ---
    chem_ref = Chemistry(chem_params)
    Y0 = chem_ref.setup_initial_mixture(T=cond["T_init"], P=cond["P_init"])
    nsp = len(Y0)
    V0 = geom.cylinder_volume(np.deg2rad(-180))
    chem_ref.gas.TPY = cond["T_init"], cond["P_init"], Y0
    m0 = chem_ref.gas.density * V0
    y0_s = np.zeros(4 + nsp)
    y0_s[0] = cond["T_init"]; y0_s[1] = V0; y0_s[2] = cond["P_init"]; y0_s[3] = m0
    y0_s[4:] = Y0

    solver_ref = CanteraEngineSolver(
        geom=geom, chemistry=chem_ref, heat_transfer=ht,
        params=CanteraSolverParams(adiabatic=False, verbose=False, rtol=1e-6, atol=1e-12)
    )
    r_ref = solver_ref.solve_closed_cycle(
        rpm=cond["rpm"], T_wall=cond["T_wall"], ca_start=-180, ca_end=180, y0=y0_s.copy()
    )
    ref_P = np.max(r_ref['y'][2]) / 1e5
    ref_T = np.max(r_ref['y'][0])
    print(f"  Single-zone: Peak P={ref_P:.2f} bar, Peak T={ref_T:.1f} K")

    # --- Multizone with adaptive K ---
    chem_mz = Chemistry(chem_params)
    Y0_mz = chem_mz.setup_initial_mixture(T=cond["T_init"], P=cond["P_init"])
    y0_m = np.zeros(3 + nzones * (nsp + 1) + nsp)
    y0_m[0] = cond["T_init"]; y0_m[1] = cond["P_init"]; y0_m[2] = V0
    for i in range(nzones):
        y0_m[3 + i * (nsp + 1)] = cond["T_init"]
        y0_m[3 + i * (nsp + 1) + 1:3 + (i + 1) * (nsp + 1)] = Y0_mz
    y0_m[3 + nzones * (nsp + 1):] = Y0_mz

    t0 = time.time()
    solver_mz = CanteraMultizoneEngineSolver(
        geom=geom, chemistry=chem_mz, heat_transfer=ht,
        params=CanteraMultizoneSolverParams(
            adiabatic=False, nzones=nzones, verbose=True,
            rtol=1e-6, atol=1e-12, max_steps=50000,
            pressure_coupling_coeff=0.0  # adaptive
        )
    )
    try:
        r_mz = solver_mz.solve_closed_cycle(
            rpm=cond["rpm"], T_wall=cond["T_wall"], ca_start=-180, ca_end=180, y0=y0_m.copy()
        )
        elapsed = time.time() - t0
        mz_P = np.max(r_mz['y'][1]) / 1e5
        mz_T = np.max(r_mz['y'][0])
        dP = (mz_P - ref_P) / ref_P * 100
        status = "OK" if r_mz['success'] else f"FAIL@CA={r_mz['ca'][-1]:.0f}"
        print(f"\n  Multizone: Peak P={mz_P:.2f} bar ({dP:+.1f}%), Peak T={mz_T:.1f} K, {elapsed:.1f}s [{status}]")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  Multizone: EXCEPTION after {elapsed:.1f}s — {str(e)[:60]}")
