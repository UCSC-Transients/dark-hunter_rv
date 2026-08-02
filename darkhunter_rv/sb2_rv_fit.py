"""Independent and joint Keplerian RV fits for SB2 component time series."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from fit_apf_rv_keplerian import (
    RVPoint,
    estimate_m1_msun_from_hrd,
    fit_all_variants,
    fit_keplerian,
    load_nss_priors_from_summary,
    rv_model,
)

logger = logging.getLogger(__name__)

_PRIOR_WEIGHT = 0.4
_MASS_PRIOR_SIGMA_FRAC = 0.2
_MASS_RATIO_SIGMA = 0.25


@dataclass
class Sb2EpochRow:
    """One epoch from ``sb2_epochs.csv``."""

    basename: str
    mjd: float
    rv1_kms: float
    rv1_err_kms: float
    rv2_kms: float
    rv2_err_kms: float
    summary_rv_kms: float
    summary_rv_err_kms: float
    sb2_candidate: bool
    median_vel_kms: np.ndarray | None = None
    median_ccf: np.ndarray | None = None


@dataclass
class Sb2TemplateMasses:
    teff1: float
    teff2: float
    logg1: float
    logg2: float
    m1_prior_msun: float
    m2_prior_msun: float


@dataclass
class GaussianPrior:
    value: float
    sigma: float


@dataclass
class OrbitPriors:
    period_days: GaussianPrior | None = None
    eccentricity: GaussianPrior | None = None
    gamma1_kms: GaussianPrior | None = None
    k1_kms: GaussianPrior | None = None
    omega_deg: GaussianPrior | None = None


def load_sb2_epochs_csv(path: Path, summary_path: Path | None = None) -> list[Sb2EpochRow]:
    """Load epoch table; fill summary RV errors from summary when missing."""
    df = pd.read_csv(path)
    summary_err: dict[str, float] = {}
    if summary_path is not None and summary_path.is_file():
        from darkhunter_rv.sb2 import parse_pipeline_rvs_from_summary

        for bn, seed in parse_pipeline_rvs_from_summary(summary_path).items():
            summary_err[bn] = float(seed.rv_err_kms)
    rows: list[Sb2EpochRow] = []
    for _, r in df.iterrows():
        bn = str(r["basename"])
        s_err = summary_err.get(bn, 0.5)
        rows.append(
            Sb2EpochRow(
                basename=bn,
                mjd=float(r.get("mjd", np.nan)),
                rv1_kms=float(r.get("rv1_kms", np.nan)),
                rv1_err_kms=float(r.get("rv1_err_kms", np.nan)) if np.isfinite(r.get("rv1_err_kms", np.nan)) else s_err,
                rv2_kms=float(r.get("rv2_kms", np.nan)),
                rv2_err_kms=float(r.get("rv2_err_kms", np.nan)) if np.isfinite(r.get("rv2_err_kms", np.nan)) else s_err,
                summary_rv_kms=float(r.get("summary_rv_kms", np.nan)),
                summary_rv_err_kms=s_err,
                sb2_candidate=bool(r.get("sb2_candidate", False)),
            )
        )
    return rows


def load_template_masses(sb2_fit_json: Path) -> Sb2TemplateMasses:
    """HRD mass priors from ``sb2_fit.json`` stellar parameters."""
    data = json.loads(sb2_fit_json.read_text(encoding="utf-8"))
    p = data["params"]
    m1 = estimate_m1_msun_from_hrd({"Teff": p["teff1"], "logg": p["logg1"]})
    m2 = estimate_m1_msun_from_hrd({"Teff": p["teff2"], "logg": p["logg2"]})
    if m1 is None:
        m1 = 1.0
    if m2 is None:
        m2 = 0.5
    return Sb2TemplateMasses(
        teff1=float(p["teff1"]),
        teff2=float(p["teff2"]),
        logg1=float(p["logg1"]),
        logg2=float(p["logg2"]),
        m1_prior_msun=float(m1),
        m2_prior_msun=float(m2),
    )


def _star1_rv(row: Sb2EpochRow) -> tuple[float, float]:
    """Pipeline summary RV unless SB2 detection yielded a primary CCF rv1."""
    if row.sb2_candidate and np.isfinite(row.rv1_kms):
        err = row.rv1_err_kms if np.isfinite(row.rv1_err_kms) else row.summary_rv_err_kms
        return float(row.rv1_kms), max(float(err), 1e-3)
    if np.isfinite(row.summary_rv_kms):
        return float(row.summary_rv_kms), max(float(row.summary_rv_err_kms), 1e-3)
    if np.isfinite(row.rv1_kms):
        err = row.rv1_err_kms if np.isfinite(row.rv1_err_kms) else row.summary_rv_err_kms
        return float(row.rv1_kms), max(float(err), 1e-3)
    return float("nan"), float("nan")


def build_rv_points_star1(epochs: list[Sb2EpochRow]) -> list[RVPoint]:
    pts: list[RVPoint] = []
    for row in epochs:
        if not np.isfinite(row.mjd):
            continue
        rv, err = _star1_rv(row)
        if not np.isfinite(rv):
            continue
        pts.append(
            RVPoint(
                mjd=row.mjd,
                rv=rv,
                rv_err=err,
                rms=err,
                file=row.basename,
                telescope="APF",
            )
        )
    return pts


def build_rv_points_star2(epochs: list[Sb2EpochRow]) -> list[RVPoint]:
    pts: list[RVPoint] = []
    for row in epochs:
        if not row.sb2_candidate or not np.isfinite(row.mjd):
            continue
        if not np.isfinite(row.rv2_kms):
            continue
        err = row.rv2_err_kms if np.isfinite(row.rv2_err_kms) else row.summary_rv_err_kms
        pts.append(
            RVPoint(
                mjd=row.mjd,
                rv=float(row.rv2_kms),
                rv_err=max(float(err), 1e-3),
                rms=max(float(err), 1e-3),
                file=row.basename,
                telescope="APF",
            )
        )
    return pts


def orbital_unit_amplitude(
    log_p: float,
    h: float,
    k: float,
    m0: float,
    t: np.ndarray,
    t_ref: float,
) -> np.ndarray:
    """Keplerian orbital term with K=1 and gamma=0."""
    params = np.array([log_p, 1.0, h, k, m0, 0.0], dtype=float)
    return rv_model(params, np.asarray(t, float), float(t_ref))


def _joint_resid(
    x: np.ndarray,
    t1: np.ndarray,
    rv1: np.ndarray,
    err1: np.ndarray,
    t2: np.ndarray,
    rv2: np.ndarray,
    err2: np.ndarray,
    t_ref: float,
    m1_prior: float,
    m2_prior: float,
    *,
    fix_period: float | None,
    fix_e: float | None,
    mass_sigma_frac: float,
    orbit_priors: OrbitPriors | None = None,
) -> np.ndarray:
    log_p, h, k, m0, g1, log_k1, g2, log_k2 = x
    if fix_period is not None:
        log_p = float(np.log(fix_period))
    if fix_e is not None:
        omega = math.atan2(k, h)
        h = float(fix_e) * math.cos(omega)
        k = float(fix_e) * math.sin(omega)
    k1 = float(np.exp(log_k1))
    k2 = float(np.exp(log_k2))
    chunks: list[float] = []
    if t1.size:
        u1 = orbital_unit_amplitude(log_p, h, k, m0, t1, t_ref)
        chunks.extend(((rv1 - (g1 + k1 * u1)) / err1).tolist())
    if t2.size:
        u2 = orbital_unit_amplitude(log_p, h, k, m0, t2, t_ref)
        chunks.extend(((rv2 - (g2 + k2 * u2)) / err2).tolist())
    if m1_prior > 0 and m2_prior > 0 and k2 > 0:
        q_prior = m2_prior / m1_prior
        q_fit = k1 / k2
        m2_fit = m1_prior * q_fit
        chunks.append(_PRIOR_WEIGHT * (q_fit - q_prior) / _MASS_RATIO_SIGMA)
        chunks.append(_PRIOR_WEIGHT * (m2_fit - m2_prior) / max(mass_sigma_frac * m2_prior, 0.05))
    if fix_period is not None:
        chunks.append(1e3 * (x[0] - np.log(fix_period)))
    if orbit_priors is not None:
        p_days = float(np.exp(log_p))
        ecc = float(np.hypot(h, k))
        omega_deg = float(np.degrees(np.arctan2(k, h)))
        if orbit_priors.period_days is not None and orbit_priors.period_days.sigma > 0:
            chunks.append((p_days - orbit_priors.period_days.value) / orbit_priors.period_days.sigma)
        if orbit_priors.eccentricity is not None and orbit_priors.eccentricity.sigma > 0:
            chunks.append((ecc - orbit_priors.eccentricity.value) / orbit_priors.eccentricity.sigma)
        if orbit_priors.gamma1_kms is not None and orbit_priors.gamma1_kms.sigma > 0:
            chunks.append((g1 - orbit_priors.gamma1_kms.value) / orbit_priors.gamma1_kms.sigma)
        if orbit_priors.k1_kms is not None and orbit_priors.k1_kms.sigma > 0:
            chunks.append((k1 - orbit_priors.k1_kms.value) / orbit_priors.k1_kms.sigma)
        if orbit_priors.omega_deg is not None and orbit_priors.omega_deg.sigma > 0:
            d_omega = ((omega_deg - orbit_priors.omega_deg.value + 180.0) % 360.0) - 180.0
            chunks.append(d_omega / orbit_priors.omega_deg.sigma)
    return np.asarray(chunks, float)


def fit_joint_sb2_variant(
    epochs: list[Sb2EpochRow],
    masses: Sb2TemplateMasses,
    *,
    fix_period: float | None = None,
    fix_e: float | None = None,
    mass_sigma_frac: float = _MASS_PRIOR_SIGMA_FRAC,
    orbit_priors: OrbitPriors | None = None,
) -> tuple[np.ndarray, dict] | None:
    """Fit shared orbit to star1 (all epochs) and star2 (sb2_candidate epochs)."""
    pts1 = build_rv_points_star1(epochs)
    pts2 = build_rv_points_star2(epochs)
    if len(pts1) < 3:
        return None
    t1 = np.array([p.mjd for p in pts1], float)
    rv1 = np.array([p.rv for p in pts1], float)
    e1 = np.array([p.rv_err for p in pts1], float)
    t2 = np.array([p.mjd for p in pts2], float) if pts2 else np.array([], float)
    rv2 = np.array([p.rv for p in pts2], float) if pts2 else np.array([], float)
    e2 = np.array([p.rv_err for p in pts2], float) if pts2 else np.array([], float)
    t_all = np.concatenate([t1, t2]) if t2.size else t1
    t_ref = float(np.median(t_all))
    span = max(float(np.ptp(rv1)), 1.0)
    g1_0 = float(np.median(rv1))
    g2_0 = float(np.median(rv2)) if rv2.size else g1_0
    k1_0 = max(1.0, 0.5 * span)
    k2_0 = max(1.0, k1_0 * masses.m2_prior_msun / max(masses.m1_prior_msun, 0.1))
    p0 = 1000.0
    e0 = 0.1
    m0_0 = 0.0
    h0, k0 = e0 * 0.8, 0.0
    if fix_period is not None:
        p0 = float(fix_period)
    if fix_e is not None:
        e0 = float(fix_e)
        h0, k0 = e0 * 0.8, 0.0
    try:
        _, rep1 = fit_keplerian(t1, rv1, e1, fix_period=fix_period, fix_e=fix_e)
        if rep1.get("converged", True) and np.isfinite(rep1.get("P_days", float("nan"))):
            p0 = float(rep1["P_days"])
            e0 = float(rep1.get("e", e0))
            omega0 = float(rep1.get("omega_rad", 0.0))
            h0 = e0 * math.cos(omega0)
            k0 = e0 * math.sin(omega0)
            m0_0 = float(rep1.get("params_raw", [0, 0, 0, 0, 0, 0])[4]) if isinstance(rep1.get("params_raw"), list) else 0.0
            g1_0 = float(rep1.get("gamma_kms", g1_0))
            k1_0 = float(rep1.get("K_kms", k1_0))
    except Exception:
        pass
    if rv2.size and k2_0 <= 0:
        k2_0 = max(1.0, k1_0 * masses.m2_prior_msun / max(masses.m1_prior_msun, 0.1))
    x0 = np.array(
        [
            np.log(max(p0, 0.5)),
            h0,
            k0,
            m0_0,
            g1_0,
            np.log(max(k1_0, 0.1)),
            g2_0,
            np.log(max(k2_0, 0.1)),
        ],
        float,
    )
    lb = np.array([np.log(0.5), -0.95, -0.95, -np.pi, -500.0, np.log(0.1), -500.0, np.log(0.1)])
    ub = np.array([np.log(5000.0), 0.95, 0.95, np.pi, 500.0, np.log(300.0), 500.0, np.log(300.0)])

    def resid(x: np.ndarray) -> np.ndarray:
        r = _joint_resid(
            x,
            t1,
            rv1,
            e1,
            t2,
            rv2,
            e2,
            t_ref,
            masses.m1_prior_msun,
            masses.m2_prior_msun,
            fix_period=fix_period,
            fix_e=fix_e,
            mass_sigma_frac=mass_sigma_frac,
            orbit_priors=orbit_priors,
        )
        if r.size == 0:
            return np.array([1e6])
        return r

    try:
        sol = least_squares(resid, np.clip(x0, lb + 1e-6, ub - 1e-6), bounds=(lb, ub), max_nfev=6000)
    except Exception:
        return None
    if not sol.success:
        return None
    log_p, h, k, m0, g1, log_k1, g2, log_k2 = sol.x
    if fix_period is not None:
        log_p = float(np.log(fix_period))
    if fix_e is not None:
        omega = math.atan2(k, h)
        h = float(fix_e) * math.cos(omega)
        k = float(fix_e) * math.sin(omega)
    p_days = float(np.exp(log_p))
    e = float(np.hypot(h, k))
    omega = float(math.atan2(k, h))
    k1 = float(np.exp(log_k1))
    k2 = float(np.exp(log_k2))
    m2_fit = masses.m1_prior_msun * k1 / k2 if k2 > 0 else float("nan")
    chi2 = float(np.sum(sol.fun**2))
    n_data = len(t1) + len(t2)
    dof = max(n_data - len(sol.x), 1)
    report = {
        "converged": True,
        "message": str(sol.message),
        "n_points_star1": int(len(t1)),
        "n_points_star2": int(len(t2)),
        "P_days": p_days,
        "e": e,
        "omega_rad": omega,
        "omega_deg": float(np.degrees(omega)),
        "gamma1_kms": float(g1),
        "gamma2_kms": float(g2),
        "K1_kms": k1,
        "K2_kms": k2,
        "M1_prior_msun": masses.m1_prior_msun,
        "M2_prior_msun": masses.m2_prior_msun,
        "M1_fit_msun": masses.m1_prior_msun,
        "M2_fit_msun": float(m2_fit) if np.isfinite(m2_fit) else None,
        "t_ref_mjd": t_ref,
        "chi2": chi2,
        "dof": int(dof),
        "chi2_red": chi2 / dof,
        "fixed_period_days": fix_period,
        "fixed_eccentricity": fix_e,
        "orbit_priors": {
            "period_days": None
            if orbit_priors is None or orbit_priors.period_days is None
            else {"value": orbit_priors.period_days.value, "sigma": orbit_priors.period_days.sigma},
            "eccentricity": None
            if orbit_priors is None or orbit_priors.eccentricity is None
            else {"value": orbit_priors.eccentricity.value, "sigma": orbit_priors.eccentricity.sigma},
            "gamma1_kms": None
            if orbit_priors is None or orbit_priors.gamma1_kms is None
            else {"value": orbit_priors.gamma1_kms.value, "sigma": orbit_priors.gamma1_kms.sigma},
            "k1_kms": None
            if orbit_priors is None or orbit_priors.k1_kms is None
            else {"value": orbit_priors.k1_kms.value, "sigma": orbit_priors.k1_kms.sigma},
            "omega_deg": None
            if orbit_priors is None or orbit_priors.omega_deg is None
            else {"value": orbit_priors.omega_deg.value, "sigma": orbit_priors.omega_deg.sigma},
        },
        "params_raw": sol.x.tolist(),
    }
    return sol.x, report


def fit_joint_sb2_all_variants(
    epochs: list[Sb2EpochRow],
    masses: Sb2TemplateMasses,
    gaia_nss: dict[str, float] | None,
    *,
    mass_sigma_frac: float = _MASS_PRIOR_SIGMA_FRAC,
    orbit_priors: OrbitPriors | None = None,
) -> dict[str, tuple[np.ndarray, dict]]:
    """Four joint SB2 variants matching ``fit_all_variants`` naming."""
    out: dict[str, tuple[np.ndarray, dict]] = {}
    free = fit_joint_sb2_variant(epochs, masses, mass_sigma_frac=mass_sigma_frac, orbit_priors=orbit_priors)
    if free is None:
        return out
    out["free"] = free
    free[1]["fit_variant"] = "free"
    nss_p = gaia_nss.get("period_days") if gaia_nss else None
    nss_e = gaia_nss.get("eccentricity") if gaia_nss else None
    if nss_p is not None and nss_e is not None and nss_p > 0 and 0 <= nss_e < 1:
        for key, fp, fe in (
            ("fix_period", float(nss_p), None),
            ("fix_ecc", None, float(nss_e)),
            ("fix_period_ecc", float(nss_p), float(nss_e)),
        ):
            hit = fit_joint_sb2_variant(
                epochs,
                masses,
                fix_period=fp,
                fix_e=fe,
                mass_sigma_frac=mass_sigma_frac,
                orbit_priors=orbit_priors,
            )
            if hit is not None:
                hit[1]["fit_variant"] = key
                out[key] = hit
    return out


def predict_joint_rvs(
    report: dict,
    params: np.ndarray,
    mjd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rv1, rv2) from a joint fit at given MJDs."""
    log_p, h, k, m0, g1, log_k1, g2, log_k2 = params
    if report.get("fixed_period_days") is not None:
        log_p = float(np.log(float(report["fixed_period_days"])))
    if report.get("fixed_eccentricity") is not None:
        fe = float(report["fixed_eccentricity"])
        omega = math.atan2(k, h)
        h = fe * math.cos(omega)
        k = fe * math.sin(omega)
    t_ref = float(report["t_ref_mjd"])
    u = orbital_unit_amplitude(log_p, h, k, m0, mjd, t_ref)
    return g1 + np.exp(log_k1) * u, g2 + np.exp(log_k2) * u


def fit_independent_sb2(
    epochs: list[Sb2EpochRow],
    summary_path: Path,
    *,
    period_min: float | None = None,
    period_max: float | None = None,
) -> tuple[dict[str, dict[str, tuple[np.ndarray, dict]]], dict[str, Any]]:
    """Independent Keplerian fits for star1 and star2.

    Returns ``(variants, json_reports)`` where *variants* holds scipy params for plotting.
    """
    gaia_nss = load_nss_priors_from_summary(summary_path)
    pts1 = build_rv_points_star1(epochs)
    pts2 = build_rv_points_star2(epochs)
    variants: dict[str, dict[str, tuple[np.ndarray, dict]]] = {"star1": {}, "star2": {}}
    json_out: dict[str, Any] = {"star1": {}, "star2": {}}
    if len(pts1) >= 3:
        t = np.array([p.mjd for p in pts1], float)
        y = np.array([p.rv for p in pts1], float)
        e = np.array([p.rv_err for p in pts1], float)
        vars1 = fit_all_variants(t, y, e, gaia_nss, period_min=period_min, period_max=period_max, period_prior_sigma=0.15)
        variants["star1"] = vars1
        json_out["star1"] = {k: v[1] for k, v in vars1.items()}
    if len(pts2) >= 3:
        t = np.array([p.mjd for p in pts2], float)
        y = np.array([p.rv for p in pts2], float)
        e = np.array([p.rv_err for p in pts2], float)
        vars2 = fit_all_variants(t, y, e, gaia_nss, period_min=period_min, period_max=period_max, period_prior_sigma=0.15)
        variants["star2"] = vars2
        json_out["star2"] = {k: v[1] for k, v in vars2.items()}
    return variants, json_out


def print_orbit_report(title: str, variants: dict[str, dict]) -> None:
    """Console summary matching Keplerian fit style."""
    print(f"\n=== {title} ===")
    for key, rep in variants.items():
        if not rep:
            continue
        p = rep.get("P_days")
        k = rep.get("K_kms", rep.get("K1_kms"))
        e = rep.get("e")
        g = rep.get("gamma_kms", rep.get("gamma1_kms"))
        chi = rep.get("chi2_red")
        print(
            f"  [{key}] P={p:.4f} d  K={k:.3f} km/s  e={e:.4f}  gamma={g:.3f}  chi2_red={chi:.3f}"
            if p is not None and k is not None and e is not None and g is not None and chi is not None
            else f"  [{key}] {rep}"
        )
        if "M1_fit_msun" in rep:
            print(
                f"         M1={rep.get('M1_fit_msun')} Msun (prior {rep.get('M1_prior_msun')})  "
                f"M2={rep.get('M2_fit_msun')} Msun (prior {rep.get('M2_prior_msun')})  "
                f"K1={rep.get('K1_kms'):.3f}  K2={rep.get('K2_kms'):.3f}"
            )


def run_sb2_orbit_fits(
    sb2_dir: Path,
    summary_path: Path,
    out_dir: Path,
    *,
    gaia_id: str,
    mass_sigma_frac: float = _MASS_PRIOR_SIGMA_FRAC,
    epoch_analyses: list[Any] | None = None,
    orbit_priors: OrbitPriors | None = None,
) -> dict[str, Any]:
    """Run independent + joint fits, write JSON, plots, and CCF diagnostics."""
    sb2_dir = Path(sb2_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs_csv = sb2_dir / "sb2_epochs.csv"
    fit_json = sb2_dir / "sb2_fit.json"
    epochs = load_sb2_epochs_csv(epochs_csv, summary_path)
    masses = load_template_masses(fit_json) if fit_json.is_file() else Sb2TemplateMasses(5500, 4000, 4.5, 4.0, 1.0, 0.5)
    gaia_nss = load_nss_priors_from_summary(summary_path)

    indep_variants, indep = fit_independent_sb2(epochs, summary_path)
    joint_raw = fit_joint_sb2_all_variants(
        epochs,
        masses,
        gaia_nss,
        mass_sigma_frac=mass_sigma_frac,
        orbit_priors=orbit_priors,
    )
    joint = {k: v[1] for k, v in joint_raw.items()}
    joint_params = {k: v[0].tolist() for k, v in joint_raw.items()}

    predictions: list[dict[str, Any]] = []
    best_key = "fix_period_ecc" if "fix_period_ecc" in joint_raw else ("free" if "free" in joint_raw else next(iter(joint_raw), None))
    if best_key is not None:
        rep = joint[best_key]
        params = np.array(joint_params[best_key], float)
        for row in epochs:
            if not np.isfinite(row.mjd):
                continue
            rv1_p, rv2_p = predict_joint_rvs(rep, params, np.array([row.mjd]))
            predictions.append(
                {
                    "basename": row.basename,
                    "mjd": row.mjd,
                    "rv1_pred_kms": float(rv1_p[0]),
                    "rv2_pred_kms": float(rv2_p[0]),
                    "sb2_candidate": row.sb2_candidate,
                }
            )

    result = {
        "gaia_id": gaia_id,
        "summary_path": str(summary_path),
        "sb2_dir": str(sb2_dir),
        "masses": {
            "m1_prior_msun": masses.m1_prior_msun,
            "m2_prior_msun": masses.m2_prior_msun,
            "teff1": masses.teff1,
            "teff2": masses.teff2,
            "logg1": masses.logg1,
            "logg2": masses.logg2,
        },
        "independent": indep,
        "joint": joint,
        "joint_params": joint_params,
        "predictions": predictions,
        "best_joint_variant": best_key,
        "orbit_priors": None
        if orbit_priors is None
        else {
            "period_days": None
            if orbit_priors.period_days is None
            else {"value": orbit_priors.period_days.value, "sigma": orbit_priors.period_days.sigma},
            "eccentricity": None
            if orbit_priors.eccentricity is None
            else {"value": orbit_priors.eccentricity.value, "sigma": orbit_priors.eccentricity.sigma},
            "gamma1_kms": None
            if orbit_priors.gamma1_kms is None
            else {"value": orbit_priors.gamma1_kms.value, "sigma": orbit_priors.gamma1_kms.sigma},
            "k1_kms": None
            if orbit_priors.k1_kms is None
            else {"value": orbit_priors.k1_kms.value, "sigma": orbit_priors.k1_kms.sigma},
            "omega_deg": None
            if orbit_priors.omega_deg is None
            else {"value": orbit_priors.omega_deg.value, "sigma": orbit_priors.omega_deg.sigma},
        },
    }
    (out_dir / "sb2_independent_fit.json").write_text(json.dumps(indep, indent=2), encoding="utf-8")
    (out_dir / "sb2_joint_fit.json").write_text(json.dumps({**result, "joint": joint}, indent=2), encoding="utf-8")

    print_orbit_report("Independent star1", indep.get("star1", {}))
    print_orbit_report("Independent star2", indep.get("star2", {}))
    print_orbit_report("Joint SB2", joint)

    from darkhunter_rv.sb2_rv_plots import (
        plot_joint_sb2_fit,
        plot_missing_epoch_ccf,
        plot_sb2_independent_fit,
    )

    pts1 = build_rv_points_star1(epochs)
    pts2 = build_rv_points_star2(epochs)
    if indep_variants.get("star1") or indep_variants.get("star2"):
        plot_sb2_independent_fit(
            gaia_id,
            pts1,
            pts2,
            indep_variants,
            out_dir / f"Gaia_DR3_{gaia_id}_sb2_independent_fit.png",
            masses=masses,
        )
    if joint_raw:
        plot_joint_sb2_fit(
            gaia_id,
            epochs,
            joint_raw,
            predictions,
            out_dir / f"Gaia_DR3_{gaia_id}_sb2_joint_fit.png",
        )
    if epoch_analyses is not None and best_key is not None:
        rep = joint[best_key]
        params = np.array(joint_params[best_key], float)
        for analysis in epoch_analyses:
            if analysis.sb2_candidate:
                continue
            if analysis.median_vel_kms is None or len(analysis.median_vel_kms) < 10:
                continue
            rv1_p, rv2_p = predict_joint_rvs(rep, params, np.array([analysis.mjd]))
            stem = Path(analysis.basename).stem
            plot_missing_epoch_ccf(
                analysis.median_vel_kms,
                analysis.median_ccf,
                rv1_pred=float(rv1_p[0]),
                rv2_pred=float(rv2_p[0]),
                out_path=out_dir / f"{stem}_ccf_missing.png",
                title=f"{stem} (no sb2_candidate)",
            )

    return result
