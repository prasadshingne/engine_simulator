"""
Fit AC model correlations from the MZ sweep results.

Reads:  data/calibration/mz_sweep_raw.csv
Fits:
  1. δE_AC polynomial  (7 coefficients: dE_n0 … dE_n6)
  2. Burn duration Eqs 1-3 (6 coefficients each: a1-a3, x1-x3 per equation)
  3. Combustion efficiency Eq. 17 (b0-b5 + eta0)

Writes: data/calibration/fitted_ac_params.yaml
        data/calibration/fit_diagnostics.png
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq

# ── path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

INPUT_CSV    = os.path.join(PROJECT_ROOT, "data", "calibration", "mz_sweep_raw.csv")
OUTPUT_YAML  = os.path.join(PROJECT_ROOT, "data", "calibration", "fitted_ac_params.yaml")
OUTPUT_DIAG  = os.path.join(PROJECT_ROOT, "data", "calibration", "fit_diagnostics.png")

# Goldsborough / AC model constants (must match solver_adiabatic_core.py)
R_u     = 8.314462          # J/mol/K
E_AC    = 125357.0          # J/mol
A_GOLD  = 1.29e-12
N_FUEL  = -0.13
N_O2    = -0.77

MECH_FILE = os.path.join(PROJECT_ROOT,
                          "data", "mechanisms", "gasoline_surrogate_323.yaml")
FUEL_SPECIES = ["IC8H18", "NC7H16", "C6H5CH3", "C5H10-2"]

# ── paper initial-guess values (Shingne 2016 Table 3) ───────────────────────
P0_BURN1 = [6.43e-2,  1.58e0,  2.47e1, -4.24e-3, 1.88e-1, -4.21e-1]
P0_BURN2 = [8.21e-3,  2.67e-2, 2.47e1, -9.90e-1, 1.73e-1, -6.04e-1]
P0_BURN3 = [7.03e-3,  2.16e-2, 1.80e0, -1.24e0,  1.96e-1, -7.56e-1]
P0_DEAC  = [1.06e0, -3.49e-2, -5.36e-2, -1.08e-1, 2.78e-1, -1.05e-1, 3.13e-2]
P0_CEFF  = [5.44e-2, 1.61e3, 3.58e-2, 0.0, 1.77e-2, 6.55e-2]   # b0-b5
ETA0     = 0.97


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Load and validate sweep data
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    n_total = len(df)

    # Keep only successful, non-misfire rows
    ok = (
        df["misfire"].fillna(False).astype(bool) == False
    ) & (
        df["error"].isna()
    )
    df = df[ok].copy()

    # Drop rows where critical targets are NaN
    targets = ["theta_IGN", "theta_25", "theta_50", "theta_75",
               "C_eff", "T_peak_K", "P_TDC_bar", "phi_prime"]
    df = df.dropna(subset=targets)

    # Remove physically impossible points (negative IMEP or C_eff below threshold)
    n_before = len(df)
    df = df[df["IMEP_bar"] > 0.0].copy()
    df = df[df["C_eff"] > 0.50].copy()
    n_removed = n_before - len(df)
    if n_removed:
        print(f"  Removed {n_removed} outlier row(s) (IMEP≤0 or C_eff≤0.5)")

    print(f"  Total points : {n_total}")
    print(f"  Valid points : {len(df)}")
    print(f"  IMEP range   : {df['IMEP_bar'].min():.2f} – "
          f"{df['IMEP_bar'].max():.2f} bar")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  δE_AC fitting
#     For each row, find δE_AC such that the Livengood-Wu integral = 1
#     at θ_IGN_MZ, then fit the polynomial.
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_EAC_for_row(row: pd.Series) -> float:
    """
    Binary search for δE_AC such that LW integral = 1.0 at theta_IGN_MZ.
    Uses the AC model's polytropic T_ad and Goldsborough τ.
    Returns np.nan if no solution found.
    """
    import cantera as ct

    T_ivc    = 475.0                       # fixed IVC temperature (K)
    p_ivc_pa = float(row["p_ivc_bar"]) * 1e5
    phi      = float(row["phi"])
    rgf      = float(row["rgf"])
    rpm      = float(row["rpm"])
    theta_IGN_MZ = float(row["theta_IGN"])
    IVC_CA   = -180.0
    omega    = rpm * 2.0 * np.pi / 60.0

    # ── IVC composition ──────────────────────────────────────────────────
    gas = ct.Solution(MECH_FILE)

    # Build fuel/air/diluent mixture.
    # egr/RGF is handled by EngineSimulation.setup_initial_state; here we
    # replicate the composition it would produce.
    # Simplification: set pure air + fuel for Goldsborough (RGF enters via
    # concentration—it dilutes O2 and fuel equally).
    phi_prime = phi * (1.0 - rgf)
    # Set at equivalence ratio phi_prime using fuel blend
    fuel_str = " ".join(f"{k}:{v}" for k, v in zip(
        FUEL_SPECIES, [0.5413, 0.1488, 0.2738, 0.0361]))
    gas.set_equivalence_ratio(phi_prime, fuel=fuel_str, oxidizer="O2:1, N2:3.76")
    gas.TP = T_ivc, p_ivc_pa

    gamma_ivc = float(gas.cp / gas.cv)
    s_ivc     = float(gas.entropy_mass)
    X_ivc     = gas.X.copy()

    # ── CA arrays for LW integration ─────────────────────────────────────
    n_pts  = 400
    ca_arr = np.linspace(IVC_CA, theta_IGN_MZ + 0.5, n_pts)
    gas_v  = ct.Solution(MECH_FILE)
    gas_v.TPX = T_ivc, p_ivc_pa, X_ivc
    V_ivc = float(gas_v.volume_mass) * float(gas_v.density) ** (-1)
    # Use geometry-agnostic approach: scale volume by isentropic pressure ratio
    # P_arr[i] = p_ivc * (V_ivc/V_i)^gamma → we work backwards from P
    # Since V proportional to 1/P^(1/gamma):
    # Use polytropic: P_arr[i] directly from the CR profile.
    # For this simplified computation we just need relative volumes; use a
    # linear placeholder for the pressure array — the actual solver uses
    # cylinder_volume(), but here we only need tau vs CA which is dominated
    # by T_ad.  Re-use the s_ivc isentropic path:
    P_arr = np.zeros(n_pts)
    T_arr = np.zeros(n_pts)
    gas_tmp = ct.Solution(MECH_FILE)
    # We need actual volumes to get P(CA) — use the solver's geometry.
    # Rather than replicating the geometry, use the stored P_TDC_bar to
    # reconstruct a polytropic profile.
    P_TDC_pa = float(row["P_TDC_bar"]) * 1e5
    # sin-based volume profile: V(CA) = V_tdc * (1 + (CR-1)/2*(1-cos(CA*pi/180)))
    # Avoid importing geometry; reconstruct CR from P_TDC_bar / p_ivc_pa
    CR = (P_TDC_pa / p_ivc_pa) ** (1.0 / gamma_ivc)
    def V_norm(ca_deg):
        """Normalised volume: 1 at TDC, CR at BDC."""
        theta = np.deg2rad(ca_deg)
        return 1.0 + (CR - 1.0) / 2.0 * (1.0 - np.cos(theta))
    V_arr_norm = np.array([V_norm(ca) for ca in ca_arr])
    P_arr = p_ivc_pa * (V_norm(IVC_CA) / V_arr_norm) ** gamma_ivc
    for i, P_pt in enumerate(P_arr):
        gas_tmp.SPX = s_ivc, P_pt, X_ivc
        T_arr[i] = gas_tmp.T

    # ── fuel and O2 concentrations at IVC (constant composition assumed) ─
    gas.TPX = T_ivc, p_ivc_pa, X_ivc
    rho_molar_ivc = gas.density_mole   # kmol/m³  (matches solver's gas.density_mole)
    conc_fuel = sum(
        rho_molar_ivc * float(gas.X[gas.species_index(sp)])
        for sp in FUEL_SPECIES if sp in gas.species_names
    )
    conc_O2 = rho_molar_ivc * float(gas.X[gas.species_index("O2")])
    conc_fuel = max(conc_fuel, 1e-30)
    conc_O2   = max(conc_O2,   1e-30)

    # Goldsborough pre-factor (independent of δE_AC)
    A0 = A_GOLD * conc_fuel ** N_FUEL * conc_O2 ** N_O2

    def lw_integral(delta_EAC: float) -> float:
        """LW integral from IVC to theta_IGN_MZ for a given δE_AC."""
        integral = 0.0
        for i in range(1, n_pts):
            if ca_arr[i] > theta_IGN_MZ:
                break
            E_eff1 = E_AC / (delta_EAC * R_u * T_arr[i - 1])
            E_eff2 = E_AC / (delta_EAC * R_u * T_arr[i])
            tau1   = A0 * np.exp(E_eff1)
            tau2   = A0 * np.exp(E_eff2)
            dca_rad = np.deg2rad(ca_arr[i] - ca_arr[i - 1])
            dt      = dca_rad / omega
            integral += 0.5 * (1.0 / tau1 + 1.0 / tau2) * dt
        return integral - 1.0   # = 0 when ignition occurs at theta_IGN_MZ

    try:
        # Check signs: integral(0.1) and integral(5.0) should bracket 0
        f_lo = lw_integral(0.1)
        f_hi = lw_integral(5.0)
        if f_lo * f_hi > 0:
            return np.nan
        delta_EAC = brentq(lw_integral, 0.1, 5.0, xtol=1e-4, maxiter=50)
        return float(delta_EAC)
    except Exception:
        return np.nan


def fit_delta_EAC(df: pd.DataFrame) -> dict:
    print("\n── Fitting δE_AC ──────────────────────────────────────────────")
    print("  Finding required δE_AC per point via LW root-finding...")

    delta_EAC_vals = []
    for i, (_, row) in enumerate(df.iterrows()):
        dEAC = compute_delta_EAC_for_row(row)
        delta_EAC_vals.append(dEAC)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(df)} done "
                  f"(last δE_AC={dEAC:.3f})" if not np.isnan(dEAC) else
                  f"  {i+1}/{len(df)} done (last=NaN)")

    df["delta_EAC_target"] = delta_EAC_vals
    df_fit = df.dropna(subset=["delta_EAC_target"]).copy()
    print(f"  Valid inversion points: {len(df_fit)} / {len(df)}")

    # Features: rgf%/45, rpm/2000, phi_prime/0.35, P_TDC/25
    # rgf is stored as fraction (0.20-0.60); solver converts to % before /45
    rgf_n  = df_fit["rgf"].values * 100.0 / 45.0   # same as solver: egr*100/45
    spd_n  = df_fit["rpm"].values / 2000.0
    phi_n  = df_fit["phi_prime"].values / 0.35
    P_r    = df_fit["P_TDC_bar"].values / 25.0
    y      = df_fit["delta_EAC_target"].values

    # Extended 10-param model: n7=phi, n8=phi^2, n9=phi*spd
    def model(X, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9):
        rgf, spd, phi, Pr = X
        poly = (n0 + n1*rgf + n2*spd + n3*rgf**2
                + n4*rgf*spd + n5*spd**2
                + n7*phi + n8*phi**2 + n9*phi*spd)
        return poly * Pr**n6

    p0_ext = list(P0_DEAC) + [0.0, 0.0, 0.0]   # n7, n8, n9 initial guess = 0

    try:
        popt, _ = curve_fit(
            model, (rgf_n, spd_n, phi_n, P_r), y,
            p0=p0_ext, maxfev=30000,
        )
    except Exception as e:
        print(f"  WARNING: curve_fit failed ({e}), using paper defaults.")
        popt = p0_ext

    y_pred = model((rgf_n, spd_n, phi_n, P_r), *popt)
    r2 = 1 - np.var(y - y_pred) / np.var(y)
    print(f"  R² = {r2:.3f}")
    print(f"  Coefficients: {popt}")

    return {
        "dE_n0": float(popt[0]),
        "dE_n1": float(popt[1]),
        "dE_n2": float(popt[2]),
        "dE_n3": float(popt[3]),
        "dE_n4": float(popt[4]),
        "dE_n5": float(popt[5]),
        "dE_n6": float(popt[6]),
        "dE_n7": float(popt[7]),
        "dE_n8": float(popt[8]),
        "dE_n9": float(popt[9]),
        "_r2_delta_EAC": round(float(r2), 4),
        "_df_deac": df_fit,       # returned for diagnostics
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Burn duration correlations (Eqs 1-3)
# ═══════════════════════════════════════════════════════════════════════════

def _burn_model(X, a1, a2, a3, x1, x2, x3):
    """Generic burn-duration model: (a1*theta² + a2*theta + a3) × φ_r^x1 × spd^x2 × P_r^x3"""
    theta, phi_r, spd, P_r = X
    poly = a1 * theta**2 + a2 * theta + a3
    return np.abs(poly) * phi_r**x1 * spd**x2 * P_r**x3


def fit_burn_durations(df: pd.DataFrame) -> dict:
    print("\n── Fitting burn duration correlations (Eqs 1-3) ───────────────")

    phi_r = (df["phi_prime"].values / 0.35)
    spd   = (df["rpm"].values / 2000.0)
    P_r   = (df["P_TDC_bar"].values / 25.0)

    d1 = df["theta_25"].values - df["theta_IGN"].values  # Δθ_IGN→25
    d2 = df["theta_50"].values - df["theta_25"].values   # Δθ_25→50
    d3 = df["theta_75"].values - df["theta_50"].values   # Δθ_50→75

    results = {}

    for label, dur, theta_in, p0 in [
        ("Eq1 (θ_IGN→25)", d1, df["theta_IGN"].values, P0_BURN1),
        ("Eq2 (θ_25→50)",  d2, df["theta_25"].values,  P0_BURN2),
        ("Eq3 (θ_50→75)",  d3, df["theta_50"].values,  P0_BURN3),
    ]:
        # Only fit rows where the duration is positive
        mask = dur > 0.5
        if mask.sum() < 10:
            print(f"  {label}: too few positive-duration points ({mask.sum()}), "
                  "using paper defaults.")
            results[label] = dict(zip(
                ["a", "b", "c", "x1", "x2", "x3"], p0))
            continue

        X = (theta_in[mask], phi_r[mask], spd[mask], P_r[mask])
        y = dur[mask]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _burn_model, X, y,
                    p0=p0, maxfev=20000,
                    bounds=([-np.inf, -np.inf, 0, -5, -5, -5],
                            [ np.inf,  np.inf, 50,  5,  5,  5]),
                )
        except Exception as e:
            print(f"  {label}: fit failed ({e}), using paper defaults.")
            popt = p0

        y_pred = _burn_model(X, *popt)
        r2 = 1 - np.var(y - y_pred) / np.var(y)
        print(f"  {label}: R²={r2:.3f}  coeffs=[{', '.join(f'{v:.3e}' for v in popt)}]")
        results[label] = {"popt": popt, "r2": r2, "theta": theta_in[mask], "y": y, "y_pred": y_pred}

    eq1 = results["Eq1 (θ_IGN→25)"]["popt"] if "popt" in results["Eq1 (θ_IGN→25)"] else P0_BURN1
    eq2 = results["Eq2 (θ_25→50)"]["popt"]  if "popt" in results["Eq2 (θ_25→50)"]  else P0_BURN2
    eq3 = results["Eq3 (θ_50→75)"]["popt"]  if "popt" in results["Eq3 (θ_50→75)"]  else P0_BURN3

    return {
        "a1": float(eq1[0]), "a2": float(eq1[1]), "a3": float(eq1[2]),
        "x1": float(eq1[3]), "x2": float(eq1[4]), "x3": float(eq1[5]),
        "a4": float(eq2[0]), "a5": float(eq2[1]), "a6": float(eq2[2]),
        "x4": float(eq2[3]), "x5": float(eq2[4]), "x6": float(eq2[5]),
        "a7": float(eq3[0]), "a8": float(eq3[1]), "a9": float(eq3[2]),
        "x7": float(eq3[3]), "x8": float(eq3[4]), "x9": float(eq3[5]),
        "_r2_burn1": round(float(results["Eq1 (θ_IGN→25)"].get("r2", 0.0)), 4),
        "_r2_burn2": round(float(results["Eq2 (θ_25→50)"].get("r2", 0.0)), 4),
        "_r2_burn3": round(float(results["Eq3 (θ_50→75)"].get("r2", 0.0)), 4),
        "_fit_results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Combustion efficiency (Eq. 17)
# ═══════════════════════════════════════════════════════════════════════════

def fit_combustion_efficiency(df: pd.DataFrame) -> dict:
    """
    Fit C_eff = Fn1(T_peak) × (φ'/0.35)^b3 × (spd)^b4 × (P_TDC/25)^b5
    where Fn1 is the hyperbolic function (Eqs 18-20).
    Strategy: fix eta0=0.97, fit b0-b5 jointly with a nonlinear optimizer.
    """
    print("\n── Fitting combustion efficiency (Eq. 17) ─────────────────────")

    T_peak = df["T_peak_K"].values
    C_eff  = df["C_eff"].values.clip(0.01, 0.99)
    phi_r  = df["phi_prime"].values / 0.35
    spd    = df["rpm"].values / 2000.0
    P_r    = df["P_TDC_bar"].values / 25.0
    eta0   = ETA0

    def fn1(T, b0, b1, b2):
        Fn2 = -b0 * (T - b1) - 2.0 * eta0
        Fn3 = eta0 * (eta0 + b0 * (T - b1)) - b2
        disc = Fn2**2 - 4.0 * Fn3
        disc = np.where(disc < 0, 0.0, disc)
        return (-Fn2 - np.sqrt(disc)) / 2.0

    def model(X, b0, b1, b2, b3, b4, b5):
        T, phi_r_, spd_, P_r_ = X
        f1 = fn1(T, b0, b1, b2)
        return f1 * phi_r_**b3 * spd_**b4 * P_r_**b5

    X = (T_peak, phi_r, spd, P_r)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                model, X, C_eff,
                p0=P0_CEFF,
                maxfev=20000,
                bounds=([0, 1000, 0, -2, -2, -2],
                        [1,  2500, 1,  2,  2,  2]),
            )
    except Exception as e:
        print(f"  WARNING: C_eff fit failed ({e}), using paper defaults.")
        popt = P0_CEFF

    y_pred = model(X, *popt)
    r2 = 1 - np.var(C_eff - y_pred) / np.var(C_eff)
    print(f"  R²={r2:.3f}  b0={popt[0]:.3e} b1={popt[1]:.1f} b2={popt[2]:.3e} "
          f"b3={popt[3]:.3e} b4={popt[4]:.3e} b5={popt[5]:.3e}")

    return {
        "b0":   float(popt[0]),
        "b1":   float(popt[1]),
        "b2":   float(popt[2]),
        "b3":   float(popt[3]),
        "b4":   float(popt[4]),
        "b5":   float(popt[5]),
        "eta0": ETA0,
        "_r2_ceff": round(float(r2), 4),
        "_y_pred": y_pred,
        "_C_eff": C_eff,
        "_T_peak": T_peak,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Direct θ_IGN polynomial (Option B)
#     log(−θ_IGN) = 2nd-order poly in (RPM/2000, phi/0.5, RGF/0.4, P_TDC/50)
# ═══════════════════════════════════════════════════════════════════════════

def fit_direct_ign(df: pd.DataFrame) -> dict:
    print("\n── Fitting direct θ_IGN polynomial ────────────────────────────")

    r  = df["rpm"].values      / 2000.0
    ph = df["phi"].values      / 0.5
    rg = df["rgf"].values      / 0.4
    pt = df["P_TDC_bar"].values / 50.0
    y  = df["theta_IGN"].values

    X = np.column_stack([
        np.ones(len(y)),
        r, ph, rg, pt,
        r**2, ph**2, rg**2, pt**2,
        r*ph, r*rg, r*pt, ph*rg, ph*pt, rg*pt,
    ])
    y_log = np.log(-y)

    from scipy.linalg import lstsq as scipy_lstsq
    coeffs, *_ = scipy_lstsq(X, y_log)

    y_pred_log = X @ coeffs
    y_pred = -np.exp(y_pred_log)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    rms    = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"  R²={r2:.4f}  RMS={rms:.3f} °CA  Max|err|={np.abs(y-y_pred).max():.2f} °CA")

    names = [f"ig_c{i}" for i in range(15)]
    result = {n: float(c) for n, c in zip(names, coeffs)}
    result["use_direct_ign_fit"] = True
    result["_r2_direct_ign"]    = round(float(r2), 4)
    result["_rms_direct_ign"]   = round(float(rms), 3)
    result["_y_pred_ign"]       = y_pred
    result["_y_ign"]            = y
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Diagnostics plot
# ═══════════════════════════════════════════════════════════════════════════

def make_diagnostics(df, deac_result, burn_result, ceff_result, ign_result, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("AC Model Calibration — Fit Diagnostics", fontsize=13)

    # ── 1. δE_AC parity ─────────────────────────────────────────────────
    ax = axes[0, 0]
    df_d = deac_result.get("_df_deac", pd.DataFrame())
    if "delta_EAC_target" in df_d.columns:
        y_t   = df_d["delta_EAC_target"].values
        rgf_n = df_d["rgf"].values * 100.0 / 45.0
        spd_n = df_d["rpm"].values / 2000.0
        phi_n = df_d["phi_prime"].values / 0.35
        P_r   = df_d["P_TDC_bar"].values / 25.0
        n = [deac_result[k] for k in ["dE_n0","dE_n1","dE_n2","dE_n3","dE_n4","dE_n5","dE_n6","dE_n7","dE_n8","dE_n9"]]
        poly = (n[0] + n[1]*rgf_n + n[2]*spd_n + n[3]*rgf_n**2
                + n[4]*rgf_n*spd_n + n[5]*spd_n**2
                + n[7]*phi_n + n[8]*phi_n**2 + n[9]*phi_n*spd_n)
        y_p  = poly * P_r**n[6]
        ax.scatter(y_t, y_p, s=15, alpha=0.7)
        lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("δE_AC target")
        ax.set_ylabel("δE_AC predicted")
        r2 = deac_result.get("_r2_delta_EAC", 0)
        ax.set_title(f"δE_AC  (R²={r2:.3f})")

    # ── 2. θ_50 parity ──────────────────────────────────────────────────
    ax = axes[0, 1]
    fr = burn_result.get("_fit_results", {})
    if "Eq2 (θ_25→50)" in fr and "y" in fr["Eq2 (θ_25→50)"]:
        d2_true = fr["Eq2 (θ_25→50)"]["y"]
        d2_pred = fr["Eq2 (θ_25→50)"]["y_pred"]
        ax.scatter(d2_true, d2_pred, s=15, alpha=0.7)
        lims = [min(d2_true.min(), d2_pred.min()), max(d2_true.max(), d2_pred.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Δθ_25→50 MZ [°CA]")
        ax.set_ylabel("Δθ_25→50 fitted [°CA]")
        r2 = burn_result.get("_r2_burn2", 0)
        ax.set_title(f"Burn Δθ_25→50  (R²={r2:.3f})")

    # ── 3. θ_IGN parity ─────────────────────────────────────────────────
    ax = axes[0, 2]
    if "Eq1 (θ_IGN→25)" in fr and "y" in fr["Eq1 (θ_IGN→25)"]:
        d1_true = fr["Eq1 (θ_IGN→25)"]["y"]
        d1_pred = fr["Eq1 (θ_IGN→25)"]["y_pred"]
        ax.scatter(d1_true, d1_pred, s=15, alpha=0.7)
        lims = [min(d1_true.min(), d1_pred.min()), max(d1_true.max(), d1_pred.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Δθ_IGN→25 MZ [°CA]")
        ax.set_ylabel("Δθ_IGN→25 fitted [°CA]")
        r2 = burn_result.get("_r2_burn1", 0)
        ax.set_title(f"Burn Δθ_IGN→25  (R²={r2:.3f})")

    # ── 4. C_eff parity ─────────────────────────────────────────────────
    ax = axes[1, 0]
    y_ce = ceff_result.get("_C_eff")
    y_cp = ceff_result.get("_y_pred")
    if y_ce is not None:
        ax.scatter(y_ce, y_cp, s=15, alpha=0.7)
        lims = [min(y_ce.min(), y_cp.min()), max(y_ce.max(), y_cp.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("C_eff MZ [-]")
        ax.set_ylabel("C_eff fitted [-]")
        r2 = ceff_result.get("_r2_ceff", 0)
        ax.set_title(f"C_eff  (R²={r2:.3f})")

    # ── 5. θ_IGN parity (direct polynomial) ─────────────────────────────
    ax = axes[1, 1]
    y_ign  = ign_result.get("_y_ign")
    yp_ign = ign_result.get("_y_pred_ign")
    if y_ign is not None:
        ax.scatter(y_ign, yp_ign, s=15, alpha=0.7)
        lims = [min(y_ign.min(), yp_ign.min()), max(y_ign.max(), yp_ign.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("θ_IGN MZ [°CA]")
        ax.set_ylabel("θ_IGN predicted [°CA]")
        r2 = ign_result.get("_r2_direct_ign", 0)
        rms = ign_result.get("_rms_direct_ign", 0)
        ax.set_title(f"θ_IGN direct-fit  (R²={r2:.3f}, RMS={rms:.2f}°CA)")

    # ── 6. IMEP distribution ─────────────────────────────────────────────
    ax = axes[1, 2]
    ax.hist(df["IMEP_bar"], bins=20, edgecolor="white")
    ax.set_xlabel("IMEP [bar]")
    ax.set_ylabel("Count")
    ax.set_title(f"IMEP coverage  ({df['IMEP_bar'].min():.1f}–{df['IMEP_bar'].max():.1f} bar)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nDiagnostics saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6.  YAML output
# ═══════════════════════════════════════════════════════════════════════════

def save_yaml(deac, burn, ceff, ign, n_valid, output_path):
    # Strip internal diagnostic keys (prefixed with _)
    def clean(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    doc = {
        "_metadata": {
            "description": (
                "AC model parameters refitted using 10-zone LLNL Gasoline Surrogate MZ model. "
                f"Fit from {n_valid} valid operating points."
            ),
            "sweep_RPM":         "800 – 4000 rpm",
            "sweep_phi":         "0.20 – 0.80",
            "sweep_RGF":         "0.20 – 0.60",
            "sweep_P_IVC":       "1.0 – 2.5 bar",
            "mechanism":         "LLNL Gasoline Surrogate 323-species",
            "R2_direct_ign":     ign.get("_r2_direct_ign", "n/a"),
            "RMS_direct_ign_CA": ign.get("_rms_direct_ign", "n/a"),
            "R2_delta_EAC":      deac.get("_r2_delta_EAC", "n/a"),
            "R2_burn1":          burn.get("_r2_burn1", "n/a"),
            "R2_burn2":          burn.get("_r2_burn2", "n/a"),
            "R2_burn3":          burn.get("_r2_burn3", "n/a"),
            "R2_C_eff":          ceff.get("_r2_ceff", "n/a"),
        },
        "ignition_direct_fit":  clean(ign),
        "ignition_delta_EAC":   clean(deac),
        "burn_duration":        clean(burn),
        "combustion_efficiency": clean(ceff),
    }

    with open(output_path, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

    print(f"\nFitted parameters saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AC Model Calibration — Parameter Fitting")
    print("=" * 60)

    df = load_data()

    ign_result  = fit_direct_ign(df)
    deac_result = fit_delta_EAC(df)
    burn_result = fit_burn_durations(df)
    ceff_result = fit_combustion_efficiency(df)

    make_diagnostics(df, deac_result, burn_result, ceff_result, ign_result, OUTPUT_DIAG)
    save_yaml(deac_result, burn_result, ceff_result, ign_result, len(df), OUTPUT_YAML)

    print("\nDone.")
    print(f"  R² θ_IGN  : {ign_result.get('_r2_direct_ign', 'n/a')}  "
          f"(RMS {ign_result.get('_rms_direct_ign', '?')} °CA)")
    print(f"  R² δE_AC  : {deac_result.get('_r2_delta_EAC', 'n/a')}")
    print(f"  R² burn1  : {burn_result.get('_r2_burn1', 'n/a')}")
    print(f"  R² burn2  : {burn_result.get('_r2_burn2', 'n/a')}")
    print(f"  R² burn3  : {burn_result.get('_r2_burn3', 'n/a')}")
    print(f"  R² C_eff  : {ceff_result.get('_r2_ceff', 'n/a')}")
    print(f"\nNext step: load fitted params in the AC model with:")
    print(f"  AdiabaticCoreParams.from_yaml('{OUTPUT_YAML}')")


if __name__ == "__main__":
    main()
