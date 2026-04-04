"""
Adiabatic Core Ignition and Three-Step Combustion Model.

Based on:
- Shingne et al. (2016) Part 1: Adiabatic core ignition model (thermodynamic1)
  - Eq. 4: entropy at IVC from Cantera
  - Eq. 5: isentropic adiabatic-core temperature T_ad
  - Eq. 6: Livengood-Wu autoignition integral
  - Eq. 8: δE_AC calibration factor (Table 12 coefficients)
- Shingne et al. (2016) Part 2: Three-step combustion profile (thermodynamic2)
  - Eq. 1–3: burn duration correlations (θ_IGN-25, θ_25-50, θ_50-75)
  - Eq. 4–6: MFB2 Wiebe function
  - Eq. 7–9: MFB1 exponential function
  - Eq. 10–12: MFB3 linear function
  - Eq. 13–15: Bezier smoothing at stage transitions
  - Eq. 16–20: combustion efficiency

Ignition delay: Goldsborough (2009) iso-octane correlation
  Ref: Goldsborough (2009) Combust Flame 156(6):1248–1262
  High-temperature Arrhenius form:
    τ = A_gold * [C8H18 (kmol/m³)]^n_fuel * [O2 (kmol/m³)]^n_O2
          * exp(E_AC / (R_u * T_ad))
  E_AC is calibrated by δE_AC: E_AC_used = E_AC / δE_AC
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cantera as ct
from scipy.integrate import solve_ivp

from ..engine.geometry import GeometryParams
from ..engine.heat_transfer import HeatTransfer

# Universal gas constant [J/mol/K]
R_u = 8.314462


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdiabaticCoreParams:
    """
    All empirical coefficients for the Shingne adiabatic-core model.

    Ignition model (Part 1):
      δE_AC coefficients — Table 12, Shingne 2016 (thermodynamic1)
      Goldsborough (2009) iso-octane HT-branch coefficients

    Combustion model (Part 2):
      Burn duration + MFB shape coefficients — Table 3, Shingne 2016 (thermodynamic2)
    """

    # ── Ignition model ──────────────────────────────────────────────────────

    # Direct θ_IGN polynomial (Option B).
    # Predicts log(−θ_IGN) as a 2nd-order polynomial in normalised inputs:
    #   r=RPM/2000, ph=phi/0.5, rg=RGF/0.4, pt=P_TDC_bar/50
    # θ_IGN = −exp(ig_c0 + ig_c1·r + ig_c2·ph + ig_c3·rg + ig_c4·pt
    #              + ig_c5·r² + ig_c6·ph² + ig_c7·rg² + ig_c8·pt²
    #              + ig_c9·r·ph + ig_c10·r·rg + ig_c11·r·pt
    #              + ig_c12·ph·rg + ig_c13·ph·pt + ig_c14·rg·pt)
    # Fitted to 10-zone MZ/LLNL sweep (114 points); R²=0.80, RMS≈3.4 °CA.
    use_direct_ign_fit: bool = False   # set True to use polynomial; False for LW
    ig_c0:  float = 0.0;  ig_c1:  float = 0.0;  ig_c2:  float = 0.0
    ig_c3:  float = 0.0;  ig_c4:  float = 0.0;  ig_c5:  float = 0.0
    ig_c6:  float = 0.0;  ig_c7:  float = 0.0;  ig_c8:  float = 0.0
    ig_c9:  float = 0.0;  ig_c10: float = 0.0;  ig_c11: float = 0.0
    ig_c12: float = 0.0;  ig_c13: float = 0.0;  ig_c14: float = 0.0

    # δE_AC calibration (Eq. 8, Table 12)
    dE_n0: float = 1.06e+00
    dE_n1: float = -3.49e-02
    dE_n2: float = -5.36e-02
    dE_n3: float = -1.08e-01
    dE_n4: float = 2.78e-01
    dE_n5: float = -1.05e-01
    dE_n6: float = 3.13e-02
    dE_n7: float = 0.0            # φ'/0.35 linear term
    dE_n8: float = 0.0            # (φ'/0.35)² term
    dE_n9: float = 0.0            # (φ'/0.35)*(rpm/2000) cross term

    # Goldsborough (2009) iso-octane HT-branch Arrhenius coefficients
    # τ [s] = A_gold * [C8H18 (kmol/m³)]^n_fuel * [O2 (kmol/m³)]^n_O2
    #            * exp(E_AC / (R_u [J/mol/K] * T_ad [K]))
    # Ref: Goldsborough (2009) Combust Flame 156(6):1248-1262, Table 1 HT branch
    A_gold:  float = 1.29e-12   # pre-exponential factor
    n_fuel:  float = -0.13      # iso-octane (C8H18) concentration exponent
    n_O2:    float = -0.77      # O2 concentration exponent
    E_AC:    float = 125357.0   # activation energy [J/mol]  (= R_u * 15073 K)

    # Polytropic index for pre-ignition compression (used to estimate P(θ))
    # Used only if pressure profile is not available from a preceding ODE solve.
    n_poly: float = 1.35

    # ── Combustion model ────────────────────────────────────────────────────

    # Burn duration: θ_IGN→25 (Eq. 1, Table 3)
    a1: float = 6.43e-02;  a2: float = 1.58e+00;  a3: float = 2.47e+01
    x1: float = -4.24e-03; x2: float = 1.88e-01;  x3: float = -4.21e-01

    # Burn duration: θ_25→50 (Eq. 2, Table 3)
    a4: float = 8.21e-03;  a5: float = 2.67e-02;  a6: float = 2.47e+01
    x4: float = -9.90e-01; x5: float = 1.73e-01;  x6: float = -6.04e-01

    # Burn duration: θ_50→75 (Eq. 3, Table 3)
    a7: float = 7.03e-03;  a8: float = 2.16e-02;  a9: float = 1.80e+00
    x7: float = -1.24e+00; x8: float = 1.96e-01;  x9: float = -7.56e-01

    # MFB1 exponential A-coefficient (Eq. 8, Table 3)
    c1: float = -2.50e-01; c2: float = -6.56e-01; c3: float = 6.14e-01
    d1: float = 2.92e-03;  d2: float = 2.63e-01

    # MFB1 exponential B-coefficient (Eq. 9, Table 3)
    c4: float = -3.26e-02
    d3: float = -6.56e-04; d4: float = -1.22e-02; d5: float = 2.47e-01

    # MFB3 linear slope M (Eq. 11, Table 3)
    e1: float = -1.75e-03; e2: float = 6.08e-01;  e3: float = -3.57e-01
    f1: float = 1.24e-03;  f2: float = 6.78e-02

    # MFB3 linear intercept C (Eq. 12, Table 3)
    e4: float = -6.17e-03; e5: float = -3.94e-02
    f3: float = -1.37e-01; f4: float = 7.50e-02

    # Combustion efficiency (Eq. 17–20, Table 3)
    b0:   float = 5.44e-02
    b1:   float = 1.61e+03   # [K] temperature at maximum efficiency inflection
    b2:   float = 3.58e-02
    b3:   float = 0.0        # φ' exponent — update if available from paper
    b4:   float = 1.77e-02
    b5:   float = 6.55e-02
    eta0: float = 0.97       # maximum combustion efficiency (plateau)

    # Wiebe function shape (Eq. 6)
    k_wiebe: float = 2.0   # shape factor (exponent = k+1 = 3)
    BM:      float = 0.50  # MFB fraction at Wiebe midpoint (θ_50)
    BS:      float = 0.10  # MFB fraction at Wiebe start  (lower Bezier boundary)
    BE:      float = 0.90  # MFB fraction at Wiebe end    (upper Bezier boundary)

    # Robust bounds for late-burn transition placement.
    # MFB3 coefficients are fitted over a narrow 95-100% range; unconstrained
    # extrapolation can produce unrealistically long tails for some conditions.
    mfb23_cap_factor: float = 2.0  # max (θ_23 - θ_75) / (θ_75 - θ_50)
    mfb_end_cap_factor: float = 6.0  # max (θ_end - θ_75) / (θ_75 - θ_50)

    # Fuel lower heating value [J/kg] — iso-octane default
    # (used to scale heat release from MFB profile)
    LHV: float = 4.43e7   # iso-octane: 44.3 MJ/kg

    # Solver settings
    rtol:      float = 1.0e-8
    atol:      float = 1.0e-10
    verbose:   bool  = True

    @classmethod
    def from_yaml(cls, path: str) -> "AdiabaticCoreParams":
        """
        Load fitted parameters from a calibration YAML file.

        The YAML is expected to have top-level sections
        (ignition_delta_EAC, burn_duration, combustion_efficiency) whose
        keys map directly to dataclass field names.  Unknown keys and
        metadata keys (prefixed with '_') are silently ignored.
        """
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        params = cls()
        for section in data.values():
            if not isinstance(section, dict):
                continue
            for key, val in section.items():
                if key.startswith("_"):
                    continue
                if hasattr(params, key):
                    field_type = type(getattr(params, key))
                    if field_type is bool:
                        setattr(params, key, bool(val))
                    else:
                        setattr(params, key, float(val))
        return params


# ---------------------------------------------------------------------------
# Isentropic T_ad helper
# ---------------------------------------------------------------------------

def _tad_from_pressure(
    gas_ref: ct.Solution,
    s_ivc: float,
    X_ivc: np.ndarray,
    P: float,
) -> float:
    """
    Find T_ad such that entropy(T_ad, P, X_IVC) = s_IVC  (Eq. 5).

    Parameters
    ----------
    gas_ref : ct.Solution
        Cantera gas object (will be modified temporarily).
    s_ivc : float
        Specific entropy at IVC [J/kg/K].
    X_ivc : np.ndarray
        Mole fractions at IVC (composition fixed for isentropic core).
    P : float
        Current cylinder pressure [Pa].

    Returns
    -------
    float
        Adiabatic core temperature [K].
    """
    gas_ref.SPX = s_ivc, P, X_ivc
    return gas_ref.T


# ---------------------------------------------------------------------------
# Goldsborough ignition delay
# ---------------------------------------------------------------------------

def _goldsborough_tau(
    T_ad: float,
    P: float,
    gas: ct.Solution,
    params: AdiabaticCoreParams,
    delta_EAC: float,
    fuel_species: Optional[List[str]] = None,
) -> float:
    """
    Goldsborough (2009) iso-octane HT-branch ignition delay [s].

    τ = A_gold * [fuel_total (kmol/m³)]^n_fuel * [O2]^n_O2
            * exp(E_AC / (delta_EAC * R_u * T_ad))

    fuel_total is the sum of molar concentrations of all fuel species so the
    correlation works for PRF blends and multi-component gasoline surrogates.

    Parameters
    ----------
    T_ad : float
        Adiabatic core temperature [K].
    P : float
        Cylinder pressure [Pa].
    gas : ct.Solution
        Cantera gas object set to IVC composition.
    params : AdiabaticCoreParams
        Model coefficients.
    delta_EAC : float
        Calibration factor (Eq. 8).
    fuel_species : list of str, optional
        Names of fuel species in the mechanism.  Defaults to ['C8H18'].

    Returns
    -------
    float
        Ignition delay [s].  Returns 1e30 if conditions are out of range.
    """
    if T_ad < 600.0 or T_ad > 1400.0:
        return 1.0e30   # outside correlation range → no ignition

    if fuel_species is None:
        fuel_species = ["C8H18"]

    # Molar concentrations [kmol/m³]
    rho_molar = gas.density_mole
    X = gas.X

    # Sum over all fuel species in the blend
    conc_fuel = 0.0
    for fsp in fuel_species:
        try:
            conc_fuel += rho_molar * X[gas.species_index(fsp)]
        except ValueError:
            pass
    conc_fuel = max(conc_fuel, 1.0e-30)

    try:
        idx_O2 = gas.species_index("O2")
    except ValueError:
        return 1.0e30
    conc_O2 = max(rho_molar * X[idx_O2], 1.0e-30)

    E_eff = params.E_AC / delta_EAC       # effective activation energy [J/mol]
    exp_term = np.exp(E_eff / (R_u * T_ad))

    tau = (params.A_gold
           * conc_fuel ** params.n_fuel
           * conc_O2   ** params.n_O2
           * exp_term)

    return max(tau, 1.0e-10)


# ---------------------------------------------------------------------------
# Fuel / equivalence helpers
# ---------------------------------------------------------------------------

def _fuel_species_from_param(fuel_param: object) -> List[str]:
    """
    Return a list of fuel species names from chemistry fuel input.
    """
    if isinstance(fuel_param, str):
        return [fuel_param]
    if isinstance(fuel_param, dict):
        return list(fuel_param.keys())
    return ["C8H18"]


def _compute_phi_prime_from_ivc(
    gas: ct.Solution,
    Y_ivc: np.ndarray,
    fuel_param: object,
    fuel_species: List[str],
    phi_fallback: float,
    egr_fallback: float,
) -> Tuple[float, float, float, str]:
    """
    Compute fuel-to-charge equivalence ratio used in Eq. 1-3 correlations.

    Paper definition:
      phi' = (m_fuel / (m_total - m_fuel)) / (m_fuel / m_air)_st

    Returns
    -------
    phi_prime : float
        Fuel-to-charge equivalence ratio.
    y_fuel : float
        Fuel mass fraction at IVC (sum over input fuel species).
    afr_st : float
        Stoichiometric air/fuel mass ratio for the configured fuel blend.
    method : str
        "ivc_fuel_to_charge" or a fallback tag.
    """
    phi_proxy = float(phi_fallback) * (1.0 - float(egr_fallback))

    y_fuel = 0.0
    for sp in fuel_species:
        try:
            y_fuel += float(Y_ivc[gas.species_index(sp)])
        except ValueError:
            pass

    y_fuel = float(np.clip(y_fuel, 0.0, 1.0 - 1.0e-12))

    try:
        afr_st = float(
            gas.stoich_air_fuel_ratio(
                fuel=fuel_param,
                oxidizer={"O2": 1.0, "N2": 3.76},
                basis="mass",
            )
        )
        if afr_st <= 0.0 or y_fuel <= 0.0:
            return phi_proxy, y_fuel, afr_st, "fallback_phi_times_1_minus_egr"
        fa_charge = y_fuel / max(1.0 - y_fuel, 1.0e-12)   # m_fuel / (m_total - m_fuel)
        phi_prime = fa_charge * afr_st                     # divide by (m_fuel/m_air)_st
        return float(phi_prime), y_fuel, afr_st, "ivc_fuel_to_charge"
    except Exception:
        return phi_proxy, y_fuel, float("nan"), "fallback_phi_times_1_minus_egr"


def _compute_blend_heat_release_from_mechanism(
    gas: ct.Solution,
    fuel_param: object,
    fuel_species: List[str],
    lhv_fallback: float,
    T_ref: float = 298.15,
    P_ref: float = ct.one_atm,
) -> Tuple[float, str]:
    """
    Compute blend chemical energy per kg fuel from mechanism thermochemistry.

    Uses complete-combustion stoichiometry for 1 kmol of the configured fuel
    blend with products CO2/H2O/N2 (water vapor basis, i.e. LHV-style).
    """
    # Fuel blend basis (mole fractions in the same format used by Cantera)
    if isinstance(fuel_param, str):
        x_fuel = {fuel_param: 1.0}
    elif isinstance(fuel_param, dict):
        x_fuel = {k: float(v) for k, v in fuel_param.items() if float(v) > 0.0}
        x_sum = sum(x_fuel.values())
        if x_sum <= 0.0:
            return float(lhv_fallback), "fallback_config_lhv"
        x_fuel = {k: v / x_sum for k, v in x_fuel.items()}
    else:
        if not fuel_species:
            return float(lhv_fallback), "fallback_config_lhv"
        x_fuel = {fuel_species[0]: 1.0}

    # Preserve gas state (this helper temporarily changes TPX)
    T0, P0, X0 = gas.T, gas.P, gas.X.copy()
    try:
        # Required oxidizer / product species
        for sp in ("O2", "N2", "CO2", "H2O"):
            gas.species_index(sp)  # raises if missing

        # Cache species molar enthalpies at reference state [J/kmol]
        h_cache: Dict[str, float] = {}

        def _h_molar(sp: str) -> float:
            if sp not in h_cache:
                gas.TPX = T_ref, P_ref, {sp: 1.0}
                h_cache[sp] = float(gas.enthalpy_mole)
            return h_cache[sp]

        # Element totals for 1 kmol fuel blend
        nC = 0.0
        nH = 0.0
        nO = 0.0
        nN = 0.0
        mw_fuel = 0.0
        h_fuel = 0.0

        for sp, x in x_fuel.items():
            idx = gas.species_index(sp)
            mw = float(gas.molecular_weights[idx])  # [kg/kmol]
            nC += x * float(gas.n_atoms(sp, "C"))
            nH += x * float(gas.n_atoms(sp, "H"))
            nO += x * float(gas.n_atoms(sp, "O"))
            nN += x * float(gas.n_atoms(sp, "N"))
            mw_fuel += x * mw
            h_fuel += x * _h_molar(sp)

        if mw_fuel <= 0.0:
            return float(lhv_fallback), "fallback_config_lhv"

        # Stoichiometric O2 requirement for complete oxidation
        nu_O2 = nC + 0.25 * nH - 0.5 * nO
        if nu_O2 <= 0.0:
            return float(lhv_fallback), "fallback_config_lhv"

        # Reactant / product enthalpy for 1 kmol fuel blend
        h_react = h_fuel + nu_O2 * _h_molar("O2") + 3.76 * nu_O2 * _h_molar("N2")
        h_prod = (
            nC * _h_molar("CO2")
            + 0.5 * nH * _h_molar("H2O")
            + (3.76 * nu_O2 + 0.5 * nN) * _h_molar("N2")
        )

        delta_h = h_prod - h_react          # [J/kmol fuel blend]
        q_fuel = -delta_h / mw_fuel         # [J/kg fuel], positive for exothermic
        if not np.isfinite(q_fuel) or q_fuel <= 0.0:
            return float(lhv_fallback), "fallback_config_lhv"
        return float(q_fuel), "mechanism_complete_combustion"
    except Exception:
        return float(lhv_fallback), "fallback_config_lhv"
    finally:
        gas.TPX = T0, P0, X0


# ---------------------------------------------------------------------------
# δE_AC calibration factor  (Eq. 8, Table 12)
# ---------------------------------------------------------------------------

def _compute_delta_EAC(
    egr: float,
    rpm: float,
    P_TDC_bar: float,
    params: AdiabaticCoreParams,
    phi_prime: float = 0.35,
) -> float:
    """
    Compute calibration factor δE_AC (Eq. 8, Table 12, Shingne 2016 Part 1),
    extended with an optional φ_prime term (dE_n7).

    δE_AC = (n0 + n1*(RGF%/45) + n2*(rpm/2000)
               + n3*(RGF%/45)^2 + n4*(RGF%/45)*(rpm/2000)
               + n5*(rpm/2000)^2 + n7*(φ'/0.35))
             * (P_TDC/25)^n6

    Parameters
    ----------
    egr : float
        EGR fraction (0–1).
    rpm : float
        Engine speed [rpm].
    P_TDC_bar : float
        TDC pressure [bar].
    phi_prime : float
        Effective equivalence ratio φ' = φ·(1-RGF). Default 0.35.
    """
    RGF  = egr * 100.0          # convert fraction → percent
    spd  = rpm / 2000.0
    rgf  = RGF / 45.0
    phi_n = phi_prime / 0.35

    poly = (params.dE_n0
            + params.dE_n1 * rgf
            + params.dE_n2 * spd
            + params.dE_n3 * rgf**2
            + params.dE_n4 * rgf * spd
            + params.dE_n5 * spd**2
            + params.dE_n7 * phi_n
            + params.dE_n8 * phi_n**2
            + params.dE_n9 * phi_n * spd)

    return poly * (P_TDC_bar / 25.0) ** params.dE_n6


# ---------------------------------------------------------------------------
# Direct θ_IGN polynomial (Option B — fitted to MZ/LLNL sweep)
# ---------------------------------------------------------------------------

def _direct_ign_timing(
    rpm: float,
    phi: float,
    rgf: float,
    P_TDC_bar: float,
    p: "AdiabaticCoreParams",
    ca_start: float = -170.0,
    ca_end: float = 80.0,
) -> float:
    """Predict θ_IGN directly from a 2nd-order polynomial fitted to the MZ sweep.

    Features are normalised: r=RPM/2000, ph=phi/0.5, rg=RGF/0.4, pt=P_TDC_bar/50.
    Returns θ_IGN [°CA] clamped to [ca_start, ca_end-1].
    """
    r  = rpm       / 2000.0
    ph = phi       / 0.5
    rg = rgf       / 0.4
    pt = P_TDC_bar / 50.0

    log_neg_theta = (
        p.ig_c0
        + p.ig_c1  * r   + p.ig_c2  * ph  + p.ig_c3  * rg  + p.ig_c4  * pt
        + p.ig_c5  * r**2 + p.ig_c6 * ph**2 + p.ig_c7 * rg**2 + p.ig_c8 * pt**2
        + p.ig_c9  * r*ph + p.ig_c10* r*rg  + p.ig_c11* r*pt
        + p.ig_c12 * ph*rg + p.ig_c13* ph*pt + p.ig_c14* rg*pt
    )
    theta = -np.exp(log_neg_theta)
    return float(np.clip(theta, ca_start, ca_end - 1.0))


# ---------------------------------------------------------------------------
# Livengood-Wu autoignition integral
# ---------------------------------------------------------------------------

def _find_ignition_timing(
    ca_array: np.ndarray,
    T_ad_array: np.ndarray,
    P_array: np.ndarray,
    gas: ct.Solution,
    params: AdiabaticCoreParams,
    delta_EAC: float,
    omega: float,
    fuel_species: Optional[List[str]] = None,
) -> float:
    """
    Integrate Livengood-Wu integral (Eq. 6) over CA to find θ_IGN.

    ∫₀^t_IGN (1/τ) dt = 1

    Uses trapezoidal integration over pre-computed (CA, T_ad, P) arrays.

    Parameters
    ----------
    ca_array : np.ndarray
        Crank angle array [deg].
    T_ad_array : np.ndarray
        Adiabatic core temperature array [K].
    P_array : np.ndarray
        Cylinder pressure array [Pa].
    gas : ct.Solution
        Cantera gas object at IVC composition.
    params : AdiabaticCoreParams
        Model coefficients.
    delta_EAC : float
        δE_AC calibration factor.
    omega : float
        Angular velocity [rad/s].
    fuel_species : list of str, optional
        Names of fuel species in the mechanism.

    Returns
    -------
    float
        Ignition crank angle [deg].  Returns ca_array[-1] if no ignition found.
    """
    integral = 0.0
    n = len(ca_array)

    for i in range(1, n):
        T1 = T_ad_array[i - 1]
        T2 = T_ad_array[i]
        P1 = P_array[i - 1]
        P2 = P_array[i]

        tau1 = _goldsborough_tau(T1, P1, gas, params, delta_EAC, fuel_species)
        tau2 = _goldsborough_tau(T2, P2, gas, params, delta_EAC, fuel_species)

        # dt = d(CA) / omega  (convert deg → rad, divide by omega)
        dca_rad = np.deg2rad(ca_array[i] - ca_array[i - 1])
        dt      = dca_rad / omega

        integral += 0.5 * (1.0/tau1 + 1.0/tau2) * dt

        if integral >= 1.0:
            # Linear interpolation for precise θ_IGN
            f = (1.0 - (integral - 0.5*(1.0/tau1+1.0/tau2)*dt)) / (0.5*(1.0/tau1+1.0/tau2)*dt)
            theta_ign = ca_array[i-1] + f * (ca_array[i] - ca_array[i-1])
            return float(theta_ign)

    return float(ca_array[-1])   # no ignition detected


# ---------------------------------------------------------------------------
# Burn duration correlations  (Eq. 1–3, Table 3)
# ---------------------------------------------------------------------------

def _burn_durations(
    theta_IGN: float,
    phi_prime: float,
    rpm: float,
    P_TDC_bar: float,
    params: AdiabaticCoreParams,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute CA milestones from burn-duration correlations (Eq. 1–3).

    Returns
    -------
    Tuple: (Δθ_IGN-25, Δθ_25-50, Δθ_50-75, θ_25, θ_50, θ_75)
    """
    spd = rpm / 2000.0
    phi_r = phi_prime / 0.35
    P_r   = P_TDC_bar / 25.0

    # θ_IGN → 25% (Eq. 1)
    d_IGN_25 = abs((params.a1 * theta_IGN**2 + params.a2 * theta_IGN + params.a3)
                   * phi_r**params.x1 * spd**params.x2 * P_r**params.x3)
    theta_25 = theta_IGN + d_IGN_25

    # θ_25 → 50% (Eq. 2)
    d_25_50 = abs((params.a4 * theta_25**2 + params.a5 * theta_25 + params.a6)
                  * phi_r**params.x4 * spd**params.x5 * P_r**params.x6)

    theta_50 = theta_25 + d_25_50

    # θ_50 → 75% (Eq. 3)
    d_50_75 = abs((params.a7 * theta_50**2 + params.a8 * theta_50 + params.a9)
                  * phi_r**params.x7 * spd**params.x8 * P_r**params.x9)

    theta_75 = theta_50 + d_50_75

    return d_IGN_25, d_25_50, d_50_75, theta_25, theta_50, theta_75


# ---------------------------------------------------------------------------
# MFB1: exponential  (Eq. 7–9)
# ---------------------------------------------------------------------------

def _mfb1_coefficients(
    theta_IGN: float,
    phi_prime: float,
    rpm: float,
    P_TDC_bar: float,
    params: AdiabaticCoreParams,
) -> Tuple[float, float]:
    """
    Compute MFB1 exponential coefficients A and B (Eq. 8–9).

    MFB1(θ) = A * exp(B * θ)   [θ in deg CA]
    """
    spd = rpm / 2000.0
    phi_r = phi_prime / 0.35
    P_r   = P_TDC_bar / 25.0

    # Coefficient A (Eq. 8)
    A = (params.d1 * np.exp(-params.d2 * theta_IGN)
         * phi_r**params.c1 * spd**params.c2 * P_r**params.c3)

    # Coefficient B (Eq. 9)
    B = ((params.d3 * theta_IGN**2 + params.d4 * theta_IGN + params.d5)
         * phi_r**params.c4)

    return A, B


def _mfb1(theta: np.ndarray, A: float, B: float) -> np.ndarray:
    """Evaluate MFB1 exponential: MFB1(θ) = A * exp(B * θ)."""
    return A * np.exp(B * theta)


# ---------------------------------------------------------------------------
# MFB2: Wiebe function  (Eq. 4–6)
# ---------------------------------------------------------------------------

def _fit_wiebe(
    theta_25: float,
    theta_50: float,
    theta_75: float,
    k: float = 2.0,
    BM: float = 0.50,
    BS: float = 0.10,
    BE: float = 0.90,
) -> Tuple[float, float]:
    """
    Compute Wiebe parameters per Shingne (2016) Part II Eqs 4-6.

    The Wiebe duration is defined as (θ_75 - θ_25) × 9/5 so that
    θ_25 maps to MFB = BS = 10%  (not 25%)
    θ_50 maps to MFB = BM = 50%  (midpoint, exact)
    θ_75 maps to MFB = BE = 90%  (not 75%)

    Parameters
    ----------
    theta_25 : float
        CA at 25% MFB [deg] — used only to define the Wiebe duration.
    theta_50 : float
        CA at 50% MFB [deg] — Wiebe midpoint (BM).
    theta_75 : float
        CA at 75% MFB [deg] — used only to define the Wiebe duration.
    k : float
        Wiebe shape factor (exponent = k+1, typically 2).
    BM, BS, BE : float
        Wiebe midpoint, start, end MFB fractions.

    Returns
    -------
    Tuple[float, float]
        (w, theta_0) — rate parameter [1/deg^(k+1)] and start CA [deg].
    """
    m = k + 1  # exponent (= 3 for k=2)

    # Eq. 4: Wiebe burn duration scaled from (25-75) window to (BS-BE) window
    theta_bdu = (theta_75 - theta_25) * (9.0 / 5.0)

    # Eq. 5-6 intermediate terms
    lnBM = (-np.log(1.0 - BM)) ** (1.0 / m)   # = ln(2)^(1/3) ≈ 0.8845
    lnBE = (-np.log(1.0 - BE)) ** (1.0 / m)   # = ln(10)^(1/3) ≈ 1.3197
    lnBS = (-np.log(1.0 - BS)) ** (1.0 / m)   # = ln(10/9)^(1/3) ≈ 0.4724
    denom = lnBE - lnBS                         # ≈ 0.8473

    if abs(denom) < 1e-10:
        denom = 1e-10

    # Eq. 5: start crank angle
    theta_0 = theta_50 - theta_bdu * lnBM / denom

    # Eq. 6: Wiebe rate parameter
    w = (theta_bdu / denom) ** (-m)

    return w, theta_0


def _mfb2(theta: np.ndarray, w: float, theta_0: float, k: float = 2.0) -> np.ndarray:
    """
    Evaluate MFB2 Wiebe function.

    MFB2(θ) = 1 - exp[-w * (θ - θ_0)^(k+1)]
    """
    m = k + 1
    arg = np.where(theta > theta_0, (theta - theta_0)**m, 0.0)
    return 1.0 - np.exp(-w * arg)


# ---------------------------------------------------------------------------
# MFB3: linear  (Eq. 10–12)
# ---------------------------------------------------------------------------

def _mfb3_coefficients(
    theta_50: float,
    phi_prime: float,
    rpm: float,
    P_TDC_bar: float,
    params: AdiabaticCoreParams,
) -> Tuple[float, float]:
    """
    Compute MFB3 slope M and intercept C (Eq. 11–12).

    MFB3(θ) = M * θ + C
    """
    spd = rpm / 2000.0
    phi_r = phi_prime / 0.35
    P_r   = P_TDC_bar / 25.0

    # Slope M (Eq. 11)
    M = (params.f1 * np.exp(-params.f2 * theta_50)
         * phi_r**params.e1 * spd**params.e2 * P_r**params.e3)

    # Intercept C (Eq. 12)
    C = ((params.f3 * np.exp(-params.f4 * theta_50) + 1.0)
         * phi_r**params.e4 * spd**params.e5)

    return M, C


def _mfb3(theta: np.ndarray, M: float, C: float) -> np.ndarray:
    """Evaluate MFB3 linear: MFB3(θ) = M * θ + C."""
    return M * theta + C


# ---------------------------------------------------------------------------
# Combustion efficiency  (Eq. 17–20)
# ---------------------------------------------------------------------------

def _combustion_efficiency(
    T_peak: float,
    phi_prime: float,
    rpm: float,
    P_TDC_bar: float,
    params: AdiabaticCoreParams,
) -> float:
    """
    Compute combustion efficiency C_eff (Eq. 17–20).

    Hyperbolic (quadratic formula) fit based on T_peak.
    """
    spd = rpm / 2000.0
    phi_r = phi_prime / 0.35
    P_r   = P_TDC_bar / 25.0

    eta0 = params.eta0
    b0, b1, b2 = params.b0, params.b1, params.b2
    b3, b4, b5 = params.b3, params.b4, params.b5

    # Hyperbolic fit (Eq. 17–20)
    Fn2 = -b0 * (T_peak - b1) - 2.0 * eta0
    Fn3 = eta0 * (eta0 + b0 * (T_peak - b1)) - b2

    discriminant = Fn2**2 - 4.0 * Fn3
    if discriminant < 0.0:
        Fn1 = eta0   # fallback: maximum efficiency
    else:
        Fn1 = (-Fn2 - np.sqrt(discriminant)) / 2.0

    # Scale by operating condition exponents
    C_eff = Fn1 * phi_r**b3 * spd**b4 * P_r**b5

    # Below b1 the quadratic formula gives Fn1 < 0 — invalid.
    # Use a linear ramp from 0 at T_low=1000 K up to C_eff(b1).
    if C_eff < 0.0 or T_peak < b1:
        T_low = 1000.0
        if T_peak <= T_low:
            return 0.0
        # Evaluate the formula at exactly b1 (where it is valid)
        Fn2_b1 = -2.0 * eta0
        Fn3_b1 = eta0 ** 2 - b2
        disc_b1 = max(Fn2_b1 ** 2 - 4.0 * Fn3_b1, 0.0)
        Fn1_b1  = (-Fn2_b1 - np.sqrt(disc_b1)) / 2.0
        C_eff_b1 = Fn1_b1 * phi_r**b3 * spd**b4 * P_r**b5
        C_eff = C_eff_b1 * (T_peak - T_low) / (b1 - T_low)

    # Clamp to physical bounds
    return float(np.clip(C_eff, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Quadratic Bezier blending  (Eq. 13–15)
# ---------------------------------------------------------------------------

def _bezier_blend(
    t_param: np.ndarray,
    P0: Tuple[float, float],
    P1: Tuple[float, float],
    P2: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quadratic Bezier curve parametrized by t ∈ [0, 1] (Eq. 13–15).

    B(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2

    Returns (theta_bez, mfb_bez) arrays.
    """
    t = t_param
    s = 1.0 - t

    theta_bez = s**2 * P0[0] + 2*s*t * P1[0] + t**2 * P2[0]
    mfb_bez   = s**2 * P0[1] + 2*s*t * P1[1] + t**2 * P2[1]

    return theta_bez, mfb_bez


def _find_intersection(
    theta: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
) -> float:
    """
    Find first crank angle where f1(θ) ≈ f2(θ) (sign change of f1-f2).

    Returns the interpolated theta at crossing, or theta[-1] if not found.
    """
    diff = f1 - f2
    for i in range(1, len(diff)):
        if diff[i - 1] * diff[i] <= 0.0:
            # Linear interpolation
            t_cross = theta[i-1] - diff[i-1] * (theta[i] - theta[i-1]) / (diff[i] - diff[i-1])
            return float(t_cross)
    return float(theta[-1])


# ---------------------------------------------------------------------------
# Full MFB profile builder
# ---------------------------------------------------------------------------

def build_mfb_profile(
    ca_array: np.ndarray,
    theta_IGN: float,
    theta_25: float,
    theta_50: float,
    theta_75: float,
    phi_prime: float,
    rpm: float,
    P_TDC_bar: float,
    params: AdiabaticCoreParams,
) -> np.ndarray:
    """
    Assemble the normalized cumulative MFB profile over ca_array.

    Stages:
      MFB1 (exponential): θ_IGN → transition with MFB2
      MFB2 (Wiebe):       transition → transition with MFB3
      MFB3 (linear):      transition → end of combustion
    Bezier smoothing at stage transitions.

    If MFB2 already exceeds MFB1 at θ_IGN (no crossing), MFB1 is skipped and
    MFB2 is normalized to start at 0 at θ_IGN while preserving its shape.

    Returns
    -------
    np.ndarray
        Normalized MFB ∈ [0, 1], same length as ca_array. Zero before θ_IGN.
    """
    # ── 1. Stage coefficients ───────────────────────────────────────────────
    A_1, B_1 = _mfb1_coefficients(theta_IGN, phi_prime, rpm, P_TDC_bar, params)
    w_2, th0_2 = _fit_wiebe(theta_25, theta_50, theta_75,
                             k=params.k_wiebe, BM=params.BM,
                             BS=params.BS, BE=params.BE)
    M_3, C_3   = _mfb3_coefficients(theta_50, phi_prime, rpm, P_TDC_bar, params)

    th_fine = np.linspace(theta_IGN, max(theta_75 + 120.0, 150.0), 4000)
    f1_fine = _mfb1(th_fine, A_1, B_1)
    f2_fine = _mfb2(th_fine, w_2, th0_2, params.k_wiebe)
    # ── 2. MFB1/MFB2 crossing ──────────────────────────────────────────────
    # Skip MFB1 only when MFB2 already exceeds MFB1 at ignition.
    # "No crossing found" alone is not sufficient, because MFB1 can also stay
    # above MFB2 over the search window.
    f1_ign = float(_mfb1(np.array([theta_IGN]), A_1, B_1)[0])
    f2_ign = float(_mfb2(np.array([theta_IGN]), w_2, th0_2, params.k_wiebe)[0])

    if f2_ign >= f1_ign:
        no_mfb1 = True
        th_trans12 = theta_IGN
    else:
        th_trans12 = _find_intersection(th_fine, f1_fine, f2_fine)
        no_mfb1 = abs(th_trans12 - th_fine[-1]) < 0.5
        if no_mfb1:
            # Robust fallback outside calibration range: use closest approach.
            idx_closest_12 = int(np.argmin(np.abs(f1_fine - f2_fine)))
            th_trans12 = float(th_fine[idx_closest_12])
            no_mfb1 = False

    # ── 3. Effective MFB2 (normalized only when MFB1 is skipped) ───────────
    if no_mfb1:
        # Shift and scale MFB2 so that MFB(theta_IGN)=0 and end value stays 1.
        v0 = f2_ign
        scale = max(1.0 - v0, 1.0e-12)
        f2_eff = np.clip((f2_fine - v0) / scale, 0.0, 1.0)
    else:
        v0, scale = 0.0, 1.0
        f2_eff = f2_fine

    # ── 4. MFB2 remap + MFB2→MFB3 transition anchor ───────────────────────
    # Raw MFB2 can drift from the predicted θ25/θ50/θ75 markers for skewed
    # combinations of (θ25, θ50, θ75).  Remap in MFB-space so stage-2 follows
    # those milestones, then start stage-3 at θ75 (late-burn boundary).
    def _mfb2_eff_raw(ca_pt):
        raw = float(_mfb2(np.array([ca_pt]), w_2, th0_2, params.k_wiebe)[0])
        return float(np.clip((raw - v0) / scale, 0.0, 1.0))

    f25 = _mfb2_eff_raw(theta_25)
    f50 = _mfb2_eff_raw(theta_50)
    f75 = _mfb2_eff_raw(theta_75)
    f25 = max(f25, 1.0e-12)
    f50 = max(f50, f25 + 1.0e-12)
    f75 = max(f75, f50 + 1.0e-12)
    eps = 1.0e-12

    def _mfb2_eff(ca_pt):
        x = _mfb2_eff_raw(ca_pt)
        if no_mfb1:
            # Keep legacy behavior for the no-MFB1 fallback path.
            y = x
        elif x <= f25:
            # First-transition anchor: force MFB(theta_25)=0.25.
            y = 0.25
        elif x <= f50:
            # Preserve the predicted 25→50 window.
            y = 0.25 + (x - f25) * 0.25 / max(f50 - f25, eps)
        elif x <= f75:
            # Preserve the predicted 50→75 window.
            y = 0.50 + (x - f50) * 0.25 / max(f75 - f50, eps)
        else:
            # Keep stage-2 below unity; stage-3 handles the final late tail.
            y = 0.75 + (x - f75) * 0.20 / max(1.0 - f75, eps)
        return float(np.clip(y, 0.0, 1.0))

    # First-transition anchor from ignition to θ25.
    f1_ign_eff = float(_mfb1(np.array([theta_IGN]), A_1, B_1)[0])
    f1_25_eff = float(_mfb1(np.array([theta_25]), A_1, B_1)[0])
    denom_12 = max(f1_25_eff - f1_ign_eff, 1.0e-12)

    def _mfb1_eff(ca_pt):
        x = float(_mfb1(np.array([ca_pt]), A_1, B_1)[0])
        y = 0.25 * (x - f1_ign_eff) / denom_12
        return float(np.clip(y, 0.0, 0.25))

    d_50_75 = max(theta_75 - theta_50, 1.0e-6)
    th_trans23 = float(theta_75)

    mfb_trans23 = 0.5 * (_mfb2_eff(th_trans23)
                         + float(_mfb3(np.array([th_trans23]), M_3, C_3)[0]))
    mfb_trans23 = float(np.clip(mfb_trans23, 0.0, 1.0))

    # ── 5. Bezier control points (transition 2→3) ──────────────────────────
    t_bez  = np.linspace(0.0, 1.0, 50)
    th_P3  = float(theta_75)
    P3_23  = (th_P3,      _mfb2_eff(th_P3))
    P4_23  = (th_trans23, mfb_trans23)
    # Stage-3 is a short late-burn tail (empirically near the 75→~99 window).
    th_end = theta_75 + 2.0 * d_50_75
    if th_end <= th_trans23 + 1.0e-6:
        th_end = th_trans23 + 1.0e-3
    P5_23  = (th_end,     1.0)
    bez_23_th, bez_23_mfb = _bezier_blend(t_bez, P3_23, P4_23, P5_23)

    # ── 6. Assemble MFB ────────────────────────────────────────────────────
    mfb = np.zeros_like(ca_array)

    if no_mfb1:
        # Skip MFB1 — normalized MFB2 from theta_IGN, then Bezier tail
        for idx, ca in enumerate(ca_array):
            if ca <= theta_IGN:
                mfb[idx] = 0.0
            elif ca <= P3_23[0]:
                mfb[idx] = _mfb2_eff(ca)
            elif ca <= P5_23[0]:
                mfb[idx] = float(np.interp(ca, bez_23_th, bez_23_mfb))
            else:
                mfb[idx] = 1.0
    else:
        # Normal 3-stage: anchored MFB1 (IGN→θ25) → anchored MFB2 → late tail.
        for idx, ca in enumerate(ca_array):
            if ca <= theta_IGN:
                mfb[idx] = 0.0
            elif ca <= theta_25:
                mfb[idx] = _mfb1_eff(ca)
            elif ca <= P3_23[0]:
                mfb[idx] = _mfb2_eff(ca)
            elif ca <= P5_23[0]:
                mfb[idx] = float(np.interp(ca, bez_23_th, bez_23_mfb))
            else:
                mfb[idx] = 1.0

    # Clamp and enforce monotonicity
    mfb = np.clip(mfb, 0.0, 1.0)
    for i in range(1, len(mfb)):
        mfb[i] = max(mfb[i], mfb[i - 1])

    return mfb


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

@dataclass
class AdiabaticCoreSolverConfig:
    """Configuration for the adiabatic core engine solver."""
    params: AdiabaticCoreParams = field(default_factory=AdiabaticCoreParams)
    rtol: float = 1.0e-8
    atol: float = 1.0e-10
    verbose: bool = True


class AdiabaticCoreEngineSolver:
    """
    Single-zone engine ODE solver using the Shingne adiabatic-core model.

    Workflow
    --------
    1. Pre-ignition: polytropic compression + Cantera isentropic solve for T_ad.
    2. Livengood-Wu integral (Goldsborough correlation) → θ_IGN.
    3. Three-step MFB profile (exponential + Wiebe + linear) → prescribed dQ/dt.
    4. Full-cycle single-zone ODE integration with prescribed heat release.

    Output format matches the single-zone Cantera/manual solver so that
    SimulationResults.from_solver_output() works without modification.
    """

    def __init__(
        self,
        geom: GeometryParams,
        chemistry,                  # engine_sim.models.chemistry.Chemistry
        heat_transfer: Optional[HeatTransfer],
        params: AdiabaticCoreSolverConfig,
    ):
        self.geom = geom
        self.chemistry = chemistry
        self.heat_transfer = heat_transfer
        self.params = params
        self.p = params.params    # shortcut to AdiabaticCoreParams

    # ------------------------------------------------------------------
    def solve_closed_cycle(
        self,
        rpm: float,
        T_wall: float,
        ca_start: float,
        ca_end: float,
        y0: np.ndarray,
    ) -> Dict:
        """
        Solve the closed cycle (IVC → EVO) using the adiabatic core model.

        Parameters
        ----------
        rpm : float
            Engine speed [rpm].
        T_wall : float
            Wall temperature [K].
        ca_start : float
            IVC crank angle [deg].
        ca_end : float
            EVO crank angle [deg].
        y0 : np.ndarray
            Initial state: [T, V, P, m, Y₁...Yₙ].

        Returns
        -------
        Dict with keys 't', 'ca', 'y'.
          y shape: (4 + n_species, n_points)
          Row order: [T, V, P, m, Y₁...Yₙ]
        """
        p = self.p
        omega = rpm * 2.0 * np.pi / 60.0   # [rad/s]

        # ── Extract IVC conditions from y0 ─────────────────────────────────
        T_ivc  = float(y0[0])
        V_ivc  = float(y0[1])
        P_ivc  = float(y0[2])
        m_tot  = float(y0[3])
        Y_ivc  = y0[4:]

        gas = self.chemistry.gas
        gas.TPY = T_ivc, P_ivc, Y_ivc
        X_ivc = gas.X.copy()

        # ── Geometry arrays ─────────────────────────────────────────────────
        n_pts  = 1200
        ca_arr = np.linspace(ca_start, ca_end, n_pts)
        t_arr  = np.deg2rad(ca_arr - ca_start) / omega   # time from IVC

        V_arr = np.array([self.geom.cylinder_volume(np.deg2rad(ca)) for ca in ca_arr])

        # ── Polytropic pressure profile for pre-ignition ────────────────────
        gamma_ivc = gas.cp / gas.cv
        n_poly    = gamma_ivc   # isentropic pre-ignition (adiabatic core assumption)
        P_arr_poly = P_ivc * (V_ivc / V_arr) ** n_poly

        # ── Isentropic T_ad profile (Eq. 5) ────────────────────────────────
        s_ivc  = gas.entropy_mass   # specific entropy at IVC [J/kg/K]
        gas_tmp = ct.Solution(gas.source)
        gas_tmp.TPY = T_ivc, P_ivc, Y_ivc

        T_ad_arr = np.zeros(n_pts)
        for i, (P_pt,) in enumerate(zip(P_arr_poly)):
            gas_tmp.SPX = s_ivc, P_pt, X_ivc
            T_ad_arr[i] = gas_tmp.T

        # ── P_TDC (polytropic to TDC) for combustion correlations ───────────
        V_tdc     = self.geom.cylinder_volume(0.0)
        P_TDC     = P_ivc * (V_ivc / V_tdc) ** gamma_ivc   # [Pa]
        P_TDC_bar = P_TDC / 1.0e5                           # [bar]

        # Determine fuel species from chemistry params (str or dict blend)
        fuel_param = self.chemistry.params.fuel
        fuel_species = _fuel_species_from_param(fuel_param)

        # ── φ' for burn correlations (Eq. 1–3 fuel-to-charge definition) ───
        phi_prime, y_fuel_ivc, afr_st, phi_method = _compute_phi_prime_from_ivc(
            gas=gas,
            Y_ivc=Y_ivc,
            fuel_param=fuel_param,
            fuel_species=fuel_species,
            phi_fallback=self.chemistry.params.phi,
            egr_fallback=self.chemistry.params.egr,
        )
        q_fuel, q_method = _compute_blend_heat_release_from_mechanism(
            gas=gas,
            fuel_param=fuel_param,
            fuel_species=fuel_species,
            lhv_fallback=p.LHV,
        )

        # ── δE_AC calibration factor (Eq. 8) ────────────────────────────────
        delta_EAC = _compute_delta_EAC(self.chemistry.params.egr, rpm, P_TDC_bar, p,
                                       phi_prime=phi_prime)

        if p.verbose:
            print(f"\n[AdiabaticCore] δE_AC = {delta_EAC:.4f}")
            print(
                f"[AdiabaticCore] P_TDC = {P_TDC_bar:.2f} bar, φ' = {phi_prime:.3f} "
                f"(method={phi_method}, Y_fuel={y_fuel_ivc:.5f}, AFR_st={afr_st:.3f})"
            )
            print(f"[AdiabaticCore] q_fuel = {q_fuel/1.0e6:.2f} MJ/kg ({q_method})")

        # ── Find ignition timing ─────────────────────────────────────────────
        if p.use_direct_ign_fit:
            # Direct 2nd-order polynomial fitted to MZ/LLNL sweep (R²=0.80)
            phi_input = float(self.chemistry.params.phi)
            rgf_input = float(self.chemistry.params.egr)
            theta_IGN = _direct_ign_timing(
                rpm, phi_input, rgf_input, P_TDC_bar, p,
                ca_start=ca_start, ca_end=ca_end,
            )
            if p.verbose:
                print(f"[AdiabaticCore] θ_IGN = {theta_IGN:.2f} °CA (direct-fit polynomial)")
        else:
            # Livengood-Wu + Goldsborough
            gas.TPY = T_ivc, P_ivc, Y_ivc
            theta_IGN = _find_ignition_timing(
                ca_arr, T_ad_arr, P_arr_poly,
                gas, p, delta_EAC, omega,
                fuel_species=fuel_species,
            )
            if p.verbose:
                print(f"[AdiabaticCore] θ_IGN = {theta_IGN:.2f} °CA (Livengood-Wu)")

        if theta_IGN >= ca_end - 1.0:
            # No ignition — return motoring result
            if p.verbose:
                print("[AdiabaticCore] WARNING: No ignition detected — returning motoring.")
            return self._solve_motoring(
                ca_arr, t_arr, V_arr, T_ivc, P_ivc, m_tot, Y_ivc,
                T_wall, rpm, omega,
            )

        # ── Burn duration correlations (Eq. 1–3) ────────────────────────────
        d_IGN_25, d_25_50, d_50_75, theta_25, theta_50, theta_75 = _burn_durations(
            theta_IGN, phi_prime, rpm, P_TDC_bar, p,
        )

        if p.verbose:
            print(f"[AdiabaticCore] θ_25={theta_25:.2f}, θ_50={theta_50:.2f}, θ_75={theta_75:.2f} °CA")

        # ── Fuel mass (sum over all blend components) ────────────────────────
        m_fuel = 0.0
        for _fsp in fuel_species:
            try:
                m_fuel += m_tot * float(Y_ivc[gas.species_index(_fsp)])
            except ValueError:
                pass
        if m_fuel <= 0.0:
            m_fuel = m_tot * 0.03   # fallback estimate

        # ── MFB profile (shape only — independent of C_eff) ─────────────────
        mfb_norm = build_mfb_profile(
            ca_arr, theta_IGN, theta_25, theta_50, theta_75,
            phi_prime, rpm, P_TDC_bar, p,
        )
        dmfb = np.gradient(mfb_norm, ca_arr)

        if p.verbose:
            def _ca_at(frac: float) -> float:
                idx = np.where(mfb_norm >= frac)[0]
                return float(ca_arr[idx[0]]) if len(idx) else float('nan')

            print(
                "[AdiabaticCore] MFB markers pred→ach: "
                f"θ25 {theta_25:.2f}→{_ca_at(0.25):.2f}, "
                f"θ50 {theta_50:.2f}→{_ca_at(0.50):.2f}, "
                f"θ75 {theta_75:.2f}→{_ca_at(0.75):.2f}, "
                f"θ90 {_ca_at(0.90):.2f}, θ99 {_ca_at(0.99):.2f} °CA"
            )

        # ── C_eff iteration (adiabatic T_peak) ───────────────────────────────
        # The combustion efficiency formula (Eq. 17-20) was derived from the
        # adiabatic-core temperature — the peak temperature the core would reach
        # without wall heat loss during the fast combustion event.  Wall heat
        # transfer is a slower process that cools the bulk gas after combustion
        # and must NOT feed back into the C_eff determination.
        # Procedure:
        #   1. Iterate C_eff against T_peak from an ADIABATIC forward integration.
        #   2. Once converged, do one final integration with actual heat transfer
        #      to produce the physical T(θ), P(θ) output.
        C_eff = 1.0
        sol   = None
        for _iter in range(12):
            dQ_dca = (C_eff * m_fuel * q_fuel) * dmfb
            # Always use adiabatic T_peak — C_eff formula needs the core temperature
            # without wall losses (the "adiabatic core" concept from Shingne)
            sol_adi = self._forward_integrate(
                ca_arr, t_arr, V_arr, y0, T_wall, rpm, omega,
                dQ_dca, mfb_norm,
                force_adiabatic=True,
            )
            T_peak    = float(np.max(sol_adi['y'][0]))
            C_eff_new = _combustion_efficiency(T_peak, phi_prime, rpm, P_TDC_bar, p)

            if p.verbose:
                print(f"[AdiabaticCore]   iter {_iter + 1}: "
                      f"T_peak_adi={T_peak:.0f} K, C_eff={C_eff:.4f} → {C_eff_new:.4f}")

            if abs(C_eff_new - C_eff) < 0.005:
                C_eff = C_eff_new
                break
            C_eff = C_eff_new

        # Final integration with actual heat transfer (if any)
        dQ_dca = (C_eff * m_fuel * q_fuel) * dmfb
        sol = self._forward_integrate(
            ca_arr, t_arr, V_arr, y0, T_wall, rpm, omega,
            dQ_dca, mfb_norm,
        )

        if p.verbose:
            print(f"[AdiabaticCore] Peak T = {np.max(sol['y'][0]):.0f} K, "
                  f"Peak P = {np.max(sol['y'][2])/1e5:.1f} bar")

        return sol

    # ------------------------------------------------------------------
    def _forward_integrate(
        self,
        ca_arr: np.ndarray,
        t_arr: np.ndarray,
        V_arr: np.ndarray,
        y0: np.ndarray,
        T_wall: float,
        rpm: float,
        omega: float,
        dQ_dca: np.ndarray,
        mfb: np.ndarray,
        force_adiabatic: bool = False,   # True → ignore self.heat_transfer (used for C_eff iteration)
    ) -> Dict:
        """
        Forward step integration of the single-zone first-law energy equation.

        Given prescribed heat release dQ/dCA [J/deg], integrate at each CA step:

            m·cv·dT = dQ_chem - dQ_wall - P·dV
            P_new    = P_old · (T_new/T_old) · (V_old/V_new)   [ideal gas]

        No stiff ODE solver is needed because heat release is prescribed,
        not coupled to fast chemical kinetics.
        """
        gas   = self.chemistry.gas
        T_ivc = float(y0[0])
        P_ivc = float(y0[2])
        m_tot = float(y0[3])
        Y_ivc = y0[4:].copy()
        nsp   = len(Y_ivc)

        # Equilibrium products for species post-processing (MFB blending)
        gas_prod = ct.Solution(gas.source)
        try:
            gas_prod.set_equivalence_ratio(
                phi=self.chemistry.params.phi,
                fuel=self.chemistry.params.fuel,
                oxidizer={'O2': 1.0, 'N2': 3.76},
            )
            gas_prod.TPX = 2200.0, P_ivc, gas_prod.X
            gas_prod.equilibrate('HP')
            Y_prod = gas_prod.Y.copy()
        except Exception:
            Y_prod = Y_ivc.copy()

        n     = len(ca_arr)
        T_out = np.zeros(n)
        P_out = np.zeros(n)
        T_out[0] = T_ivc
        P_out[0] = P_ivc
        T = T_ivc
        P = P_ivc

        for i in range(1, n):
            dca = ca_arr[i] - ca_arr[i - 1]        # [deg]
            dt  = t_arr[i]  - t_arr[i - 1]         # [s]
            dV  = V_arr[i]  - V_arr[i - 1]         # [m³]

            # Chemical heat [J] — trapezoidal
            dQ_chem = 0.5 * (dQ_dca[i - 1] + dQ_dca[i]) * dca

            # Wall heat loss [J]
            _ht = None if force_adiabatic else self.heat_transfer
            if _ht is not None:
                theta_mid = np.deg2rad(0.5 * (ca_arr[i - 1] + ca_arr[i]))
                Q_wall_rate, _ = _ht.heat_transfer_rate(
                    theta_mid, rpm, P, T,
                    p_motored=P, p_ref=P_ivc,
                    T_ref=T_ivc, V_ref=V_arr[0],
                    T_wall=T_wall,
                )
                dQ_wall = Q_wall_rate * dt
            else:
                dQ_wall = 0.0

            # cv at current state
            gas.TPY = T, P, Y_ivc
            cv = gas.cv_mass

            # First law: dU = dQ_chem - dQ_wall - P·dV
            dT    = (dQ_chem - dQ_wall - P * dV) / (m_tot * cv)
            T_new = max(T + dT, 300.0)
            P_new = max(P * (T_new / T) * (V_arr[i - 1] / V_arr[i]), 1.0e4)

            T = T_new
            P = P_new
            T_out[i] = T
            P_out[i] = P

        # Assemble output: state vector [T, V, P, m, Y...]
        y_out = np.zeros((4 + nsp, n))
        y_out[0] = T_out
        y_out[1] = V_arr
        y_out[2] = P_out
        y_out[3] = m_tot
        for i in range(n):
            x = float(mfb[i])
            y_out[4:, i] = (1.0 - x) * Y_ivc + x * Y_prod

        return {
            't': t_arr,
            'ca': ca_arr,
            'y': y_out,
            'ca_heat_release': ca_arr,
            'dQ_dca_prescribed': dQ_dca,
            'mfb_profile': mfb,
        }

    # ------------------------------------------------------------------
    def _solve_motoring(
        self,
        ca_arr, t_arr, V_arr,
        T0, P0, m0, Y0,
        T_wall, rpm, omega,
    ) -> Dict:
        """Return a motoring (non-firing) result with no heat release."""
        nsp   = len(Y0)
        n_pts = len(ca_arr)
        gas   = self.chemistry.gas
        gas.TPY = T0, P0, Y0

        gamma = gas.cp / gas.cv
        P_arr = P0 * (V_arr[0] / V_arr) ** gamma
        T_arr = T0 * (V_arr[0] / V_arr) ** (gamma - 1.0)

        y_out = np.zeros((4 + nsp, n_pts))
        y_out[0] = T_arr
        y_out[1] = V_arr
        y_out[2] = P_arr
        y_out[3] = m0
        y_out[4:] = Y0[:, np.newaxis]   # species constant

        return {
            't': t_arr,
            'ca': ca_arr,
            'y': y_out,
            'ca_heat_release': ca_arr,
            'dQ_dca_prescribed': np.zeros_like(ca_arr),
            'mfb_profile': np.zeros_like(ca_arr),
        }
