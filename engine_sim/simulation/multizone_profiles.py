"""Reference multizone profiles from Kodavasal et al. (2013, JER 10.1177/1468087413482480).

This module provides practical 10-zone profiles derived from:
- Figure 2: prescribed zone mass fractions
- Figure 3(b): zone heat-loss multiplier trend
"""

from __future__ import annotations

import numpy as np


# Figure 2 (10-zone) mass fractions, ordered from hottest (zone 1/core)
# to coldest (zone 10/wall-adjacent). Values read from the published curve.
_FIG2_MASS_FRACTIONS_10 = np.array(
    [0.02, 0.08, 0.10, 0.10, 0.20, 0.20, 0.10, 0.10, 0.08, 0.02],
    dtype=float,
)

# Figure 3(b) is shown for 40 zones. These 10-zone values preserve the same
# strongly increasing shape from hot to cold zones and are normalized so that
# sum_i(m_i * C_i) = 1 (Equation 7 consistency).
_FIG3B_HEAT_MULT_10_RAW = np.array(
    [0.08, 0.22, 0.38, 0.55, 0.75, 1.00, 1.35, 1.85, 2.90, 7.70],
    dtype=float,
)


def _normalize_mass_weighted_multipliers(mass_fracs: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Normalize multipliers so mass-weighted average is 1."""
    denom = float(np.dot(mass_fracs, multipliers))
    if denom <= 0.0:
        return np.ones_like(multipliers)
    return multipliers / denom


def zone_mass_fractions(nzones: int) -> np.ndarray:
    """Return per-zone mass fractions (sum=1)."""
    if nzones == 10:
        return _FIG2_MASS_FRACTIONS_10.copy()
    return np.full(nzones, 1.0 / nzones, dtype=float)


def zone_heat_loss_multipliers(nzones: int, mass_fracs: np.ndarray) -> np.ndarray | None:
    """Return per-zone heat-loss multipliers C_i, or None if no profile."""
    if nzones == 10:
        return _normalize_mass_weighted_multipliers(
            mass_fracs, _FIG3B_HEAT_MULT_10_RAW
        )
    return None

