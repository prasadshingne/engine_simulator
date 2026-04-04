"""
Multi-Zone sweep for AC model calibration.

Generates a Latin Hypercube Sampling (LHS) grid over:
  RPM:   800  - 4000
  phi:   0.2  - 0.8
  RGF:   0.20 - 0.60   (set via config.egr)
  P_IVC: 1.0  - 2.5 bar

Runs 10-zone LLNL 4-component Gasoline Surrogate (312 species) in parallel.
MFB is derived from fuel depletion of the 4 tracked fuel species:
  IC8H18, NC7H16, C6H5CH3, C5H10-2

Output: data/calibration/mz_sweep_raw.csv
"""

import os
import sys
import time
import io
import contextlib
import multiprocessing as mp
import traceback

import numpy as np
import pandas as pd
from scipy.stats import qmc

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── constants ────────────────────────────────────────────────────────────────
N_SAMPLES  = 120
N_WORKERS  = 14
SEED       = 42

T_IVC_K    = 475.0
T_WALL_K   = 400.0
IVC_CA     = -180.0
EVO_CA     =  180.0

MECH_FILE  = "data/mechanisms/gasoline_surrogate_323.yaml"
FUEL_BLEND = {"IC8H18": 0.5413, "NC7H16": 0.1488,
              "C6H5CH3": 0.2738, "C5H10-2": 0.0361}
FUEL_SPECIES = list(FUEL_BLEND.keys())

# Misfire: less than 5% of fuel consumed
MISFIRE_CEFF = 0.05

# MFB percentile used as "ignition timing" (1% burned)
MFB_IGN_THRESHOLD = 0.01

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "calibration")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "mz_sweep_raw.csv")


def generate_lhs_grid(n: int, seed: int) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    raw     = sampler.random(n=n)
    scaled  = qmc.scale(raw,
                        [800.0, 0.20, 0.20, 1.0],
                        [4000.0, 0.80, 0.60, 2.5])
    return pd.DataFrame(scaled, columns=["rpm", "phi", "rgf", "p_ivc_bar"])


# ── worker ───────────────────────────────────────────────────────────────────

def _run_point(row_dict: dict) -> dict:
    """Run one 10-zone LLNL simulation. Returns extracted calibration targets."""
    result = dict(row_dict)
    result["misfire"] = False
    result["error"]   = None

    # Capture all solver chatter — 14 workers printing simultaneously is unreadable
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            import cantera as ct
            from engine_sim.simulation.engine import EngineConfig, EngineSimulation
            from engine_sim.simulation.results import SimulationResults

            rpm      = float(row_dict["rpm"])
            phi      = float(row_dict["phi"])
            rgf      = float(row_dict["rgf"])
            p_ivc_pa = float(row_dict["p_ivc_bar"]) * 1e5

            config = EngineConfig.from_yaml()
            config.mechanism   = MECH_FILE
            config.fuel        = dict(FUEL_BLEND)
            config.phi         = phi
            config.egr         = rgf
            config.speed       = rpm
            config.temperature = T_IVC_K
            config.pressure    = p_ivc_pa
            config.wall_temp   = T_WALL_K
            config.start_ca    = IVC_CA
            config.end_ca      = EVO_CA
            config.model_type  = "multi"
            config.nzones      = 10
            config.adiabatic   = False

            sim = EngineSimulation(config)
            y0  = sim.setup_initial_state()
            sol = sim.solver.solve_closed_cycle(
                y0=y0, rpm=config.speed, T_wall=config.wall_temp,
                ca_start=config.start_ca, ca_end=config.end_ca,
            )
            res = SimulationResults.from_solver_output(
                sol, sim.chemistry.gas.species_names,
                model_type="multi", mechanism=MECH_FILE,
            )

        # ── extract from results ─────────────────────────────────────────────
        ca     = res.crank_angle          # shape (1200,) uniform -180 to 180
        T_bulk = res.temperature          # bulk temperature
        P_bulk = res.pressure
        V      = res.volume
        Y      = res.species              # (n_sp, 1200)
        snames = res.species_names

        # Fuel depletion MFB — sum across 4 fuel species
        y_fuel = np.zeros(len(ca))
        for sp in FUEL_SPECIES:
            if sp in snames:
                y_fuel += Y[snames.index(sp), :]

        y_fuel_ivc = float(y_fuel[0])
        y_fuel_evo = float(y_fuel[-1])

        if y_fuel_ivc < 1e-10:
            result["misfire"] = True
            result["_log"] = buf.getvalue()
            return result

        C_eff = float(np.clip(1.0 - y_fuel_evo / y_fuel_ivc, 0.0, 1.0))

        if C_eff < MISFIRE_CEFF:
            result["misfire"] = True
            result["_log"] = buf.getvalue()
            return result

        # Normalised cumulative MFB from fuel depletion
        # MFB(theta) = (y_fuel_ivc - y_fuel(theta)) / (y_fuel_ivc - y_fuel_evo)
        burned    = y_fuel_ivc - y_fuel
        mfb       = np.clip(burned / max(y_fuel_ivc - y_fuel_evo, 1e-15), 0.0, 1.0)

        def find_ca_at_mfb(target: float) -> float:
            """Linear interpolation: first CA where MFB >= target."""
            for i in range(1, len(mfb)):
                if mfb[i - 1] <= target <= mfb[i]:
                    f = (target - mfb[i - 1]) / max(mfb[i] - mfb[i - 1], 1e-15)
                    return float(ca[i - 1] + f * (ca[i] - ca[i - 1]))
            # If target never reached in combustion region, return last CA
            return float(ca[np.argmax(mfb >= target)] if np.any(mfb >= target) else ca[-1])

        theta_IGN = find_ca_at_mfb(MFB_IGN_THRESHOLD)
        theta_25  = find_ca_at_mfb(0.25)
        theta_50  = find_ca_at_mfb(0.50)
        theta_75  = find_ca_at_mfb(0.75)
        theta_90  = find_ca_at_mfb(0.90)   # extra: useful for burn duration tail

        # Sanity: milestones must be sequential and before EVO
        if not (theta_IGN < theta_25 < theta_50 < theta_75 < EVO_CA - 5):
            result["misfire"] = True
            result["_log"] = buf.getvalue()
            return result

        # P_TDC polytropic (same definition as AC model)
        gas_tmp = ct.Solution(MECH_FILE)
        Y_ivc   = Y[:, 0]
        gas_tmp.TPY = T_IVC_K, p_ivc_pa, Y_ivc
        gamma_ivc = float(gas_tmp.cp / gas_tmp.cv)
        V_ivc_val = float(V[0])
        V_tdc_val = float(np.min(V))
        P_TDC_bar = float(p_ivc_pa / 1e5 * (V_ivc_val / V_tdc_val) ** gamma_ivc)

        # phi' = phi * (1 - RGF)
        phi_prime = phi * (1.0 - rgf)

        # IMEP (closed-cycle)
        V_d      = float(np.max(V) - np.min(V))
        dW       = np.diff(V) * (P_bulk[:-1] + P_bulk[1:]) / 2.0
        IMEP_bar = float(np.sum(dW) / V_d / 1e5)

        # Peak bulk temperature
        T_peak   = float(np.max(T_bulk))

        result.update({
            "theta_IGN":  theta_IGN,
            "theta_25":   theta_25,
            "theta_50":   theta_50,
            "theta_75":   theta_75,
            "theta_90":   theta_90,
            "C_eff":      C_eff,
            "T_peak_K":   T_peak,
            "P_TDC_bar":  P_TDC_bar,
            "phi_prime":  phi_prime,
            "gamma_ivc":  gamma_ivc,
            "IMEP_bar":   IMEP_bar,
        })

    except Exception:
        result["error"] = traceback.format_exc()
        result["_log"]  = buf.getvalue()

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 62)
    print("AC Model Calibration — 10-Zone LLNL Sweep")
    print(f"  Mechanism : LLNL Gasoline Surrogate 4-component (312 sp)")
    print(f"  Fuel      : IC8H18/NC7H16/C6H5CH3/C5H10-2")
    print(f"  Samples   : {N_SAMPLES}  (LHS seed={SEED})")
    print(f"  Workers   : {N_WORKERS}")
    print(f"  T_IVC     : {T_IVC_K} K  (fixed)")
    print(f"  T_wall    : {T_WALL_K} K  (fixed)")
    print(f"  Output    : {OUTPUT_CSV}")
    print("=" * 62)

    grid = generate_lhs_grid(N_SAMPLES, SEED)
    rows = grid.to_dict(orient="records")
    print(f"\nSampling ranges:")
    print(f"  RPM  : {grid['rpm'].min():.0f} – {grid['rpm'].max():.0f}")
    print(f"  phi  : {grid['phi'].min():.2f} – {grid['phi'].max():.2f}")
    print(f"  RGF  : {grid['rgf'].min():.2f} – {grid['rgf'].max():.2f}")
    print(f"  P_IVC: {grid['p_ivc_bar'].min():.2f} – {grid['p_ivc_bar'].max():.2f} bar")
    print()

    t_start  = time.time()
    results  = []
    n_ok     = 0
    n_misf   = 0
    n_err    = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_point, rows), 1):
            elapsed = time.time() - t_start

            if res.get("error"):
                n_err += 1
                tag    = "ERR "
                extra  = str(res["error"]).splitlines()[-1][:40]
            elif res.get("misfire"):
                n_misf += 1
                tag     = "MFIR"
                extra   = ""
            else:
                n_ok  += 1
                tag    = "OK  "
                extra  = (f"θ_IGN={res['theta_IGN']:+5.1f}° "
                          f"θ_50={res['theta_50']:+5.1f}° "
                          f"C_eff={res['C_eff']:.2f} "
                          f"IMEP={res['IMEP_bar']:.2f}bar")

            print(f"[{i:3d}/{N_SAMPLES}] {elapsed:5.0f}s {tag}  "
                  f"rpm={res['rpm']:5.0f} φ={res['phi']:.2f} "
                  f"rgf={res['rgf']:.2f} P={res['p_ivc_bar']:.2f}  {extra}")

            # Drop internal log field before saving
            res.pop("_log", None)
            results.append(res)

            # Partial save every 10 points
            if i % 10 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    elapsed = time.time() - t_start

    print()
    print("=" * 62)
    print(f"Sweep complete: {elapsed/60:.1f} min")
    print(f"  OK      : {n_ok}")
    print(f"  Misfire : {n_misf}")
    print(f"  Error   : {n_err}")
    print(f"  Saved   : {OUTPUT_CSV}")
    print("=" * 62)


if __name__ == "__main__":
    main()
