"""Derive orbital inclination from Gaia Thiele-Innes elements (Halbwachs et al. 2023, App. A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

# Gaia DR3: inclination is not published for astrometric binaries; derive from A,B,F,G.
_TI_META_KEYS = (
    ("A", ("A_Thiele_Innes", "a_thiele_innes")),
    ("A_err", ("A_Thiele_Innes_Error", "a_thiele_innes_error")),
    ("B", ("B_Thiele_Innes", "b_thiele_innes")),
    ("B_err", ("B_Thiele_Innes_Error", "b_thiele_innes_error")),
    ("F", ("F_Thiele_Innes", "f_thiele_innes")),
    ("F_err", ("F_Thiele_Innes_Error", "f_thiele_innes_error")),
    ("G", ("G_Thiele_Innes", "g_thiele_innes")),
    ("G_err", ("G_Thiele_Innes_Error", "g_thiele_innes_error")),
)


@dataclass(frozen=True)
class ThieleInnesElements:
    A: float
    B: float
    F: float
    G: float
    A_err: float = np.nan
    B_err: float = np.nan
    F_err: float = np.nan
    G_err: float = np.nan

    def has_errors(self) -> bool:
        errs = (self.A_err, self.B_err, self.F_err, self.G_err)
        return any(np.isfinite(e) and e > 0 for e in errs)


@dataclass(frozen=True)
class CampbellElements:
    """Astrometric Campbell elements from Thiele-Innes (Halbwachs App. A)."""

    a_mas: float
    i_deg: float
    omega_deg: float
    Omega_deg: float
    a_mas_err: float = np.nan
    i_deg_err: float = np.nan
    omega_deg_err: float = np.nan
    Omega_deg_err: float = np.nan


def _meta_float(meta: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key not in meta:
            continue
        try:
            val = float(meta[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            return val
    return None


def campbell_to_thiele_innes(
    a_mas: float,
    i_deg: float,
    omega_deg: float,
    Omega_deg: float,
) -> tuple[float, float, float, float]:
    """Forward Binnendijk/Halbwachs A.1 (mas); used for validation and tests."""
    i = np.deg2rad(i_deg)
    omega = np.deg2rad(omega_deg)
    Omega = np.deg2rad(Omega_deg)
    A = a_mas * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i))
    B = a_mas * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(i))
    F = -a_mas * (np.sin(omega) * np.cos(Omega) + np.cos(omega) * np.sin(Omega) * np.cos(i))
    G = -a_mas * (np.sin(omega) * np.sin(Omega) - np.cos(omega) * np.cos(Omega) * np.cos(i))
    return float(A), float(B), float(F), float(G)


def thiele_innes_from_metadata(meta: Mapping[str, Any]) -> Optional[ThieleInnesElements]:
    """Parse A,B,F,G (+ errors) from Gaia metadata or ADQL row dict."""
    vals: dict[str, Optional[float]] = {}
    for slot, keys in _TI_META_KEYS:
        vals[slot] = _meta_float(meta, *keys)
    if any(vals[k] is None for k in ("A", "B", "F", "G")):
        return None
    return ThieleInnesElements(
        A=float(vals["A"]),  # type: ignore[arg-type]
        B=float(vals["B"]),  # type: ignore[arg-type]
        F=float(vals["F"]),  # type: ignore[arg-type]
        G=float(vals["G"]),  # type: ignore[arg-type]
        A_err=float(vals["A_err"]) if vals["A_err"] is not None else np.nan,
        B_err=float(vals["B_err"]) if vals["B_err"] is not None else np.nan,
        F_err=float(vals["F_err"]) if vals["F_err"] is not None else np.nan,
        G_err=float(vals["G_err"]) if vals["G_err"] is not None else np.nan,
    )


def _wrap_deg_0_360(deg: float) -> float:
    return float(np.mod(deg, 360.0))


def _circular_std_deg(values: np.ndarray) -> float:
    ang = np.deg2rad(np.asarray(values, dtype=float))
    c = float(np.mean(np.cos(ang)))
    s = float(np.mean(np.sin(ang)))
    r = float(np.hypot(c, s))
    if r <= 1e-15:
        return 180.0
    return float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(min(1.0, r))))))


def invert_thiele_innes(A: float, B: float, F: float, G: float) -> Optional[tuple[float, float, float, float]]:
    """
    Invert Thiele-Innes A,B,F,G to (a_mas, i_rad, omega_rad, Omega_rad).

    Uses the Binnendijk/Halbwachs inversion (same arccos branch as ``rv_fits.astrometric_orbit``).
    """
    u = (A * A + B * B + F * F + G * G) / 2.0
    v = A * G - B * F
    disc = u * u - v * v
    if disc < 0:
        if disc > -1e-18 * max(1.0, u * u):
            disc = 0.0
        else:
            return None
    a0 = np.sqrt(u + np.sqrt(disc))
    if a0 <= 0 or not np.isfinite(a0):
        return None

    o_minus_O = np.arctan2(B + F, G - A)
    o_plus_O = np.arctan2(B - F, A + G)
    omega = 0.5 * (o_minus_O + o_plus_O)
    Omega = o_plus_O - omega

    if (-B - F) * np.sin(omega - Omega) < 0:
        o_minus_O += np.pi
        omega = 0.5 * (o_minus_O + o_plus_O)
        Omega = o_plus_O - omega

    cos_wpO = np.cos(omega + Omega)
    if abs(cos_wpO) < 1e-15:
        return None
    arg = ((A + G) / a0) / cos_wpO - 1.0
    if not np.isfinite(arg):
        return None
    arg = float(np.clip(arg, -1.0, 1.0))
    i_rad = float(np.arccos(arg))
    if not np.isfinite(i_rad) or i_rad <= 0 or i_rad >= np.pi:
        return None
    return float(a0), i_rad, float(omega), float(Omega)


def inclination_rad_from_thiele_innes(A: float, B: float, F: float, G: float) -> Optional[float]:
    """Orbital inclination (rad) from Thiele-Innes elements."""
    inv = invert_thiele_innes(A, B, F, G)
    if inv is None:
        return None
    return float(inv[1])


def inclination_deg_from_thiele_innes(A: float, B: float, F: float, G: float) -> Optional[float]:
    i_rad = inclination_rad_from_thiele_innes(A, B, F, G)
    if i_rad is None:
        return None
    return float(np.rad2deg(i_rad))


def _ti_sigmas(ti: ThieleInnesElements) -> tuple[float, float, float, float]:
    return (
        ti.A_err if np.isfinite(ti.A_err) and ti.A_err > 0 else 0.0,
        ti.B_err if np.isfinite(ti.B_err) and ti.B_err > 0 else 0.0,
        ti.F_err if np.isfinite(ti.F_err) and ti.F_err > 0 else 0.0,
        ti.G_err if np.isfinite(ti.G_err) and ti.G_err > 0 else 0.0,
    )


def _mc_inclination_error_deg(
    ti: ThieleInnesElements,
    i_point_deg: float,
    *,
    n_samples: int,
    seed: int,
) -> Optional[float]:
    del i_point_deg
    camp = campbell_from_thiele_innes(ti, mc_samples=n_samples, seed=seed)
    if camp is None:
        return None
    err = float(camp.i_deg_err)
    if not np.isfinite(err) or err <= 0:
        return None
    return err


def campbell_from_thiele_innes(
    ti: ThieleInnesElements,
    *,
    mc_samples: int = 512,
    seed: int = 0,
) -> Optional[CampbellElements]:
    """
    Point-estimate Campbell elements from A,B,F,G.

    Uncertainties are Monte Carlo over uncorrelated Gaussian TI errors when present.
    """
    inv = invert_thiele_innes(ti.A, ti.B, ti.F, ti.G)
    if inv is None:
        return None
    a0, i_rad, omega, Omega = inv
    i_deg = float(np.rad2deg(i_rad))
    omega_deg = _wrap_deg_0_360(float(np.rad2deg(omega)))
    Omega_deg = _wrap_deg_0_360(float(np.rad2deg(Omega)))

    sigmas = _ti_sigmas(ti)
    a_err = i_err = omega_err = Omega_err = np.nan
    if any(s > 0 for s in sigmas):
        rng = np.random.default_rng(seed)
        a_draws: list[float] = []
        i_draws: list[float] = []
        om_draws: list[float] = []
        Om_draws: list[float] = []
        for _ in range(mc_samples):
            a = rng.normal(ti.A, sigmas[0] or 0.0)
            b = rng.normal(ti.B, sigmas[1] or 0.0)
            f = rng.normal(ti.F, sigmas[2] or 0.0)
            g = rng.normal(ti.G, sigmas[3] or 0.0)
            d = invert_thiele_innes(a, b, f, g)
            if d is None:
                continue
            aa, ii, oo, OO = d
            i_d = float(np.rad2deg(ii))
            if not (0.0 < i_d < 180.0):
                continue
            a_draws.append(float(aa))
            i_draws.append(i_d)
            om_draws.append(_wrap_deg_0_360(float(np.rad2deg(oo))))
            Om_draws.append(_wrap_deg_0_360(float(np.rad2deg(OO))))
        n_ok = len(i_draws)
        if n_ok >= max(50, mc_samples // 20):
            a_err = float(np.std(np.asarray(a_draws)))
            i_err = float(np.std(np.asarray(i_draws)))
            omega_err = _circular_std_deg(np.asarray(om_draws))
            Omega_err = _circular_std_deg(np.asarray(Om_draws))

    return CampbellElements(
        a_mas=float(a0),
        i_deg=i_deg,
        omega_deg=omega_deg,
        Omega_deg=Omega_deg,
        a_mas_err=float(a_err),
        i_deg_err=float(i_err),
        omega_deg_err=float(omega_err),
        Omega_deg_err=float(Omega_err),
    )


def inclination_from_thiele_innes(
    ti: ThieleInnesElements,
    *,
    mc_samples: int = 4096,
    seed: int = 0,
) -> tuple[Optional[float], Optional[float]]:
    """
    Return (inclination_deg, inclination_error_deg).

    Point estimate from Thiele-Innes inversion; σ_i from Monte Carlo over Gaussian TI
    uncertainties (uncorrelated when covariances are unavailable).
    """
    i_deg = inclination_deg_from_thiele_innes(ti.A, ti.B, ti.F, ti.G)
    if i_deg is None:
        return None, None
    i_err = _mc_inclination_error_deg(ti, i_deg, n_samples=mc_samples, seed=seed)
    return i_deg, i_err


def metadata_inclination_is_missing(meta: Mapping[str, Any]) -> bool:
    incl = _meta_float(meta, "Inclination", "inclination", "Inclination_Deg")
    if incl is None:
        return True
    return incl <= 0.05 or incl >= 179.95


def fill_inclination_in_metadata(meta: dict[str, Any]) -> bool:
    """
    If Inclination is missing/invalid, derive from Thiele-Innes and update *meta* in place.

    Returns True when inclination (and optional error) were written.
    """
    if not metadata_inclination_is_missing(meta):
        return False
    ti = thiele_innes_from_metadata(meta)
    if ti is None:
        return False
    i_deg, i_err = inclination_from_thiele_innes(ti)
    if i_deg is None or i_deg <= 0.05 or i_deg >= 179.95:
        return False
    meta["Inclination"] = float(i_deg)
    if i_err is not None and np.isfinite(i_err) and i_err > 0:
        meta["Inclination_Error"] = float(i_err)
    return True


def inclination_from_row_dict(row: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Convenience for ADQL / cache rows with lowercase nss column names."""
    ti = thiele_innes_from_metadata(row)
    if ti is None:
        return None, None
    return inclination_from_thiele_innes(ti)
