"""The Joker SB1 RV orbit fits (rejection sampling, optional pymc MCMC)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from astropy.time import Time
import astropy.units as u

from fit_apf_rv_keplerian import (
    mass_function_msun,
    rv_model,
    solve_m2_with_inclination_msun,
    solve_m2sini_msun,
)

JOKER_VARIANT_ORDER: Tuple[str, ...] = ("rv_only", "period", "ecc", "full")
JOKER_VARIANT_LABEL: Dict[str, str] = {
    "rv_only": "RV only",
    "period": "RV + P",
    "ecc": "RV + e",
    "full": "RV + NSS",
}

P_MIN_DAYS = 20.0
P_MAX_DAYS = 2000.0
SIGMA_K0_FLOOR_KMS = 30.0
SIGMA_V_FLOOR_KMS = 100.0
DEFAULT_PRIOR_SIZE = 100_000
DEFAULT_MAX_POSTERIOR = 256
MCMC_MAX_SURVIVORS = 16
GAIA_T0_EPOCH = Time(2016.0, format="jyear", scale="tcb")

try:
    import thejoker as tj

    HAS_THEJOKER = True
except Exception:  # pragma: no cover - optional CI dep
    tj = None  # type: ignore[assignment]
    HAS_THEJOKER = False


def sigma_k0_kms(rvs: np.ndarray) -> float:
    """Scale of the Joker K prior: max(mean(RV), 30 km/s)."""
    finite = np.asarray(rvs, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return SIGMA_K0_FLOOR_KMS
    return float(max(float(np.mean(finite)), SIGMA_K0_FLOOR_KMS))


def sigma_v_kms(rvs: np.ndarray) -> float:
    """Scale of the Joker v0 prior: max(max(|RV|), 100 km/s)."""
    finite = np.asarray(rvs, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return SIGMA_V_FLOOR_KMS
    return float(max(float(np.max(np.abs(finite))), SIGMA_V_FLOOR_KMS))


def period_bounds_days(nss: Optional[Mapping[str, Any]]) -> Tuple[float, float]:
    """Default 20–2000 d, expanded to cover Gaia P ± 5σ when needed."""
    p_min = P_MIN_DAYS
    p_max = P_MAX_DAYS
    if not nss:
        return p_min, p_max
    p = _finite(nss.get("period_days"))
    if p is None:
        return p_min, p_max
    perr = _finite(nss.get("period_days_error")) or 0.0
    lo = p - 5.0 * perr
    hi = p + 5.0 * perr
    if lo < p_min:
        p_min = max(1.0, float(lo))
    if hi > p_max:
        p_max = float(hi)
    return p_min, p_max


def t_periastron_gaia_to_mjd(t_periastron_gaia: float) -> float:
    """Gaia NSS t_periastron is days from J2016.0; already-MJD values pass through."""
    t = float(t_periastron_gaia)
    if t > 40000.0:
        return t
    return float((GAIA_T0_EPOCH + t * u.day).mjd)


def mean_anomaly_rad(t_peri_mjd: float, period_days: float, t_ref_mjd: float) -> float:
    """M0 at t_ref for The Joker, wrapped to [0, 2π)."""
    return float((2.0 * np.pi * ((t_ref_mjd - t_peri_mjd) / period_days)) % (2.0 * np.pi))


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _gaussian_field(
    nss: Optional[Mapping[str, Any]],
    value_key: str,
    err_key: str,
    *,
    min_err: float = 0.0,
) -> Optional[Tuple[float, float]]:
    if not nss:
        return None
    mu = _finite(nss.get(value_key))
    sig = _finite(nss.get(err_key))
    if mu is None or sig is None or sig <= min_err:
        return None
    return mu, sig


def prior_spec_for_variant(
    variant: str,
    nss: Optional[Mapping[str, Any]],
    *,
    t_ref_mjd: float,
) -> Dict[str, Any]:
    """
    Describe which Gaia/Thiele-Innes Gaussians apply to a Joker prior variant.

    Returns skip_reason when the variant cannot be built.
    """
    spec: Dict[str, Any] = {"variant": variant, "skip_reason": None, "fields": {}}
    if variant == "rv_only":
        return spec
    if variant == "period":
        p = _gaussian_field(nss, "period_days", "period_days_error")
        if p is None:
            spec["skip_reason"] = "missing_period_prior"
            return spec
        spec["fields"]["P"] = {"mu": p[0], "sigma": p[1], "unit": "day"}
        return spec
    if variant == "ecc":
        e = _gaussian_field(nss, "eccentricity", "eccentricity_error")
        if e is None:
            spec["skip_reason"] = "missing_eccentricity_prior"
            return spec
        spec["fields"]["e"] = {"mu": e[0], "sigma": e[1], "truncated": True}
        return spec
    if variant == "full":
        fields: Dict[str, Any] = {}
        p = _gaussian_field(nss, "period_days", "period_days_error")
        e = _gaussian_field(nss, "eccentricity", "eccentricity_error")
        om = _gaussian_field(nss, "omega_deg", "omega_deg_error")
        if p is None:
            spec["skip_reason"] = "missing_period_prior"
            return spec
        if e is None:
            spec["skip_reason"] = "missing_eccentricity_prior"
            return spec
        if om is None:
            spec["skip_reason"] = "missing_omega_prior"
            return spec
        fields["P"] = {"mu": p[0], "sigma": p[1], "unit": "day"}
        fields["e"] = {"mu": e[0], "sigma": e[1], "truncated": True}
        fields["omega"] = {
            "mu": math.radians(om[0]),
            "sigma": math.radians(om[1]),
            "unit": "rad",
        }
        t0_g = _finite(nss.get("t_periastron_gaia") if nss else None)
        t0_err = _finite(nss.get("t_periastron_error_days") if nss else None)
        if t0_g is not None and t0_err is not None and t0_err > 0 and p[0] > 0:
            t_peri = t_periastron_gaia_to_mjd(t0_g)
            m0 = mean_anomaly_rad(t_peri, p[0], t_ref_mjd)
            m0_sig = float(2.0 * np.pi * t0_err / p[0])
            fields["M0"] = {"mu": m0, "sigma": max(m0_sig, 1e-3), "unit": "rad"}
            fields["t_periastron_mjd"] = t_peri
        spec["fields"] = fields
        return spec
    spec["skip_reason"] = f"unknown_variant:{variant}"
    return spec


def percentile_block(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    p16, p50, p84 = np.percentile(arr, [16.0, 50.0, 84.0])
    return {
        "p16": float(p16),
        "median": float(p50),
        "p84": float(p84),
    }


def params_from_keplerian(
    p_days: float,
    k_kms: float,
    e: float,
    omega_rad: float,
    m0_rad: float,
    gamma_kms: float,
) -> np.ndarray:
    """Pack into fit_apf_rv_keplerian.rv_model parameter vector."""
    e_clip = float(np.clip(e, 0.0, 0.999))
    return np.array(
        [
            math.log(max(p_days, 1e-6)),
            float(k_kms),
            e_clip * math.cos(omega_rad),
            e_clip * math.sin(omega_rad),
            float(m0_rad),
            float(gamma_kms),
        ],
        dtype=float,
    )


def median_params_from_arrays(
    p_days: np.ndarray,
    k_kms: np.ndarray,
    e: np.ndarray,
    omega_rad: np.ndarray,
    m0_rad: np.ndarray,
    gamma_kms: np.ndarray,
) -> np.ndarray:
    """One Keplerian using the sample nearest median P (consistent orbit)."""
    p = np.asarray(p_days, dtype=float)
    if p.size == 0:
        raise ValueError("empty posterior")
    med = float(np.median(p))
    idx = int(np.argmin(np.abs(p - med)))
    return params_from_keplerian(
        float(p[idx]),
        float(k_kms[idx]),
        float(e[idx]),
        float(omega_rad[idx]),
        float(m0_rad[idx]),
        float(gamma_kms[idx]),
    )


def random_param_rows(
    p_days: np.ndarray,
    k_kms: np.ndarray,
    e: np.ndarray,
    omega_rad: np.ndarray,
    m0_rad: np.ndarray,
    gamma_kms: np.ndarray,
    *,
    n_draw: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    n = int(np.asarray(p_days).size)
    if n == 0:
        return []
    gen = rng if rng is not None else np.random.default_rng()
    take = min(n_draw, n)
    idx = gen.choice(n, size=take, replace=False)
    rows: List[np.ndarray] = []
    for i in idx:
        rows.append(
            params_from_keplerian(
                float(p_days[i]),
                float(k_kms[i]),
                float(e[i]),
                float(omega_rad[i]),
                float(m0_rad[i]),
                float(gamma_kms[i]),
            )
        )
    return rows


def summarize_sample_arrays(
    *,
    p_days: np.ndarray,
    k_kms: np.ndarray,
    e: np.ndarray,
    omega_rad: np.ndarray,
    m0_rad: np.ndarray,
    gamma_kms: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    t_ref_mjd: float,
    sampler: str,
    variant: str,
    m1_msun: Optional[float],
    inclination_deg: Optional[float],
) -> Dict[str, Any]:
    """JSON block for one variant from posterior arrays (no thejoker required)."""
    params = median_params_from_arrays(p_days, k_kms, e, omega_rad, m0_rad, gamma_kms)
    logp, k, h, kk, m0, gamma = params
    p_med = float(np.exp(logp))
    e_med = float(np.hypot(h, kk))
    om_med = float(math.atan2(kk, h))
    model = rv_model(params, t, t_ref_mjd)
    chi2 = float(np.sum(((y - model) / np.maximum(yerr, 1e-6)) ** 2))
    fm = mass_function_msun(p_med, float(k), e_med)
    m2sini = None
    m2_at_i = None
    if fm is not None and m1_msun is not None and m1_msun > 0:
        m2sini = float(solve_m2sini_msun(float(fm), float(m1_msun)))
        if inclination_deg is not None:
            m2_at_i = solve_m2_with_inclination_msun(float(fm), float(m1_msun), float(inclination_deg))
    n = 2.0 * np.pi / p_med
    t_peri = float(t_ref_mjd - m0 / n)
    out: Dict[str, Any] = {
        "fit_variant": variant,
        "sampler": sampler,
        "n_samples": int(np.asarray(p_days).size),
        "n_points": int(t.size),
        "chi2": chi2,
        "P_days": p_med,
        "K_kms": float(k),
        "e": e_med,
        "omega_rad": om_med,
        "omega_deg": float(np.degrees(om_med)),
        "gamma_kms": float(gamma),
        "t_periastron_mjd": t_peri,
        "t_ref_mjd": float(t_ref_mjd),
        "mass_function_msun": None if fm is None else float(fm),
        "m2sini_msun": m2sini,
        "m2_at_i_msun": None if m2_at_i is None else float(m2_at_i),
        "percentiles": {
            "P_days": percentile_block(p_days),
            "K_kms": percentile_block(k_kms),
            "e": percentile_block(e),
            "omega_deg": percentile_block(np.degrees(omega_rad)),
            "gamma_kms": percentile_block(gamma_kms),
        },
        "params_raw": params.tolist(),
        "skip_reason": None,
    }
    return out


def should_skip_refit(
    json_path: Path,
    summary_path: Path,
    n_rv: int,
    *,
    force: bool,
) -> bool:
    """Cron FIT_FORCE=0: skip when JSON is newer than the summary and n_rv matches."""
    if force or not json_path.is_file():
        return False
    try:
        if float(summary_path.stat().st_mtime) > float(json_path.stat().st_mtime):
            return False
        report = json.loads(json_path.read_text())
    except Exception:
        return False
    prev_n = report.get("n_points")
    try:
        return int(prev_n) == int(n_rv)
    except (TypeError, ValueError):
        return False


def is_unimodal_enough(p_days: np.ndarray) -> bool:
    """MCMC continuation when few survivors or a tight period peak."""
    p = np.asarray(p_days, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return False
    if p.size <= MCMC_MAX_SURVIVORS:
        return True
    med = float(np.median(p))
    if med <= 0:
        return False
    p16, p84 = np.percentile(p, [16.0, 84.0])
    return float(p84 - p16) / med < 0.05


def skipped_variant_block(variant: str, reason: str, n_points: int) -> Dict[str, Any]:
    return {
        "fit_variant": variant,
        "skip_reason": reason,
        "n_points": int(n_points),
        "n_samples": 0,
        "sampler": None,
    }


def arrays_from_joker_samples(samples: Any) -> Dict[str, np.ndarray]:
    """Pull P, K, e, omega, M0, v0 from a JokerSamples-like mapping."""
    def _to(name: str, unit: Optional[u.Unit] = None) -> np.ndarray:
        val = samples[name]
        if hasattr(val, "to_value") and unit is not None:
            return np.asarray(val.to_value(unit), dtype=float)
        if hasattr(val, "to_value"):
            return np.asarray(val.to_value(), dtype=float)
        return np.asarray(val, dtype=float)

    p = _to("P", u.day)
    k = _to("K", u.km / u.s)
    e = _to("e")
    omega = _to("omega", u.radian)
    if "M0" in samples.par_names if hasattr(samples, "par_names") else True:
        try:
            m0 = _to("M0", u.radian)
        except Exception:
            m0 = np.zeros_like(p)
    else:
        m0 = np.zeros_like(p)
    try:
        v0 = _to("v0", u.km / u.s)
    except Exception:
        v0 = _to("v0")
    k = np.abs(k)
    return {
        "P_days": p,
        "K_kms": k,
        "e": e,
        "omega_rad": omega,
        "M0_rad": m0,
        "gamma_kms": v0,
    }


def _build_joker_prior(
    variant: str,
    nss: Optional[Mapping[str, Any]],
    *,
    rvs: np.ndarray,
    t_ref_mjd: float,
) -> Tuple[Any, Dict[str, Any]]:
    if not HAS_THEJOKER:
        raise RuntimeError("thejoker is not installed")
    import pymc as pm
    import thejoker.units as xu

    spec = prior_spec_for_variant(variant, nss, t_ref_mjd=t_ref_mjd)
    if spec.get("skip_reason"):
        return None, spec
    p_min, p_max = period_bounds_days(nss)
    sigma_k0 = sigma_k0_kms(rvs) * u.km / u.s
    sigma_v = sigma_v_kms(rvs) * u.km / u.s
    fields = spec.get("fields") or {}
    pars: Dict[str, Any] = {}
    model = pm.Model()
    with model:
        if "P" in fields:
            f = fields["P"]
            pars["P"] = xu.with_unit(pm.Normal("P", f["mu"], f["sigma"]), u.day)
        if "e" in fields:
            f = fields["e"]
            mu = float(np.clip(f["mu"], 1e-4, 0.999))
            pars["e"] = xu.with_unit(
                pm.TruncatedNormal("e", mu=mu, sigma=max(float(f["sigma"]), 1e-4), lower=0.0, upper=0.999),
                u.one,
            )
        if "omega" in fields:
            f = fields["omega"]
            pars["omega"] = xu.with_unit(
                pm.Normal("omega", f["mu"], max(float(f["sigma"]), 1e-4)),
                u.radian,
            )
        if "M0" in fields:
            f = fields["M0"]
            pars["M0"] = xu.with_unit(
                pm.Normal("M0", f["mu"], max(float(f["sigma"]), 1e-4)),
                u.radian,
            )
        kwargs: Dict[str, Any] = {
            "sigma_K0": sigma_k0,
            "sigma_v": sigma_v,
            "model": model,
        }
        if "P" not in pars:
            kwargs["P_min"] = p_min * u.day
            kwargs["P_max"] = p_max * u.day
        if pars:
            kwargs["pars"] = pars
        prior = tj.JokerPrior.default(**kwargs)
    return prior, spec


def run_joker_variant(
    variant: str,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    nss: Optional[Mapping[str, Any]],
    *,
    t_ref_mjd: float,
    prior_size: int,
    chain_path: Path,
    old_chain_path: Optional[Path],
    rng: np.random.Generator,
    mcmc_tune: int,
    mcmc_draws: int,
    mcmc_chains: int,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any], str]:
    """
    Rejection sample, optionally continue with MCMC, write HDF5.

    Returns (arrays, prior_spec, sampler_name).
    """
    spec = prior_spec_for_variant(variant, nss, t_ref_mjd=t_ref_mjd)
    if spec.get("skip_reason"):
        return None, spec, "skipped"
    if not HAS_THEJOKER:
        spec = dict(spec)
        spec["skip_reason"] = "thejoker_not_installed"
        return None, spec, "skipped"

    prior, spec = _build_joker_prior(variant, nss, rvs=y, t_ref_mjd=t_ref_mjd)
    data = tj.RVData(t=Time(t, format="mjd"), rv=y * u.km / u.s, rv_err=yerr * u.km / u.s)
    joker = tj.TheJoker(prior, rng=rng)
    prior_samples = prior.sample(size=int(prior_size), rng=rng)
    if old_chain_path is not None and old_chain_path.is_file():
        try:
            old = tj.JokerSamples.read(str(old_chain_path))
            prior_samples = prior_samples.concatenate(old) if hasattr(prior_samples, "concatenate") else prior_samples
        except Exception:
            pass
    samples = joker.rejection_sample(
        data,
        prior_samples,
        max_posterior_samples=DEFAULT_MAX_POSTERIOR,
    )
    sampler_name = "rejection"
    arr = arrays_from_joker_samples(samples)
    if is_unimodal_enough(arr["P_days"]):
        try:
            import pymc as pm

            with prior.model:
                mcmc_init = joker.setup_mcmc(data, samples)
                idata = pm.sample(
                    tune=int(mcmc_tune),
                    draws=int(mcmc_draws),
                    chains=int(mcmc_chains),
                    start=mcmc_init,
                    cores=1,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
            samples = tj.JokerSamples.from_inference_data(prior, idata, data)
            if hasattr(samples, "wrap_K"):
                samples.wrap_K()
            arr = arrays_from_joker_samples(samples)
            sampler_name = "mcmc"
        except Exception:
            sampler_name = "rejection"
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        samples.write(str(chain_path), overwrite=True)
    except Exception:
        pass
    return arr, spec, sampler_name


def masses_from_report_variants(
    variants: Mapping[str, dict],
    *,
    m1_msun: Optional[float],
    inclination_deg: Optional[float],
    used_m2_msun: Optional[float],
) -> Dict[str, Optional[float]]:
    rv_only = variants.get("rv_only") if isinstance(variants.get("rv_only"), dict) else None
    full = variants.get("full") if isinstance(variants.get("full"), dict) else None
    m2sini = None
    m2_at_i = None
    m2_rv_ast = None
    if rv_only and m1_msun:
        fm = rv_only.get("mass_function_msun")
        if fm is None:
            fm = mass_function_msun(rv_only["P_days"], rv_only["K_kms"], rv_only["e"])
        if fm is not None and float(fm) > 0:
            m2sini = float(solve_m2sini_msun(float(fm), float(m1_msun)))
            if inclination_deg is not None:
                m2_at_i = solve_m2_with_inclination_msun(
                    float(fm), float(m1_msun), float(inclination_deg)
                )
    if full and m1_msun and inclination_deg is not None:
        fm = full.get("mass_function_msun")
        if fm is None and full.get("P_days") is not None:
            fm = mass_function_msun(full["P_days"], full["K_kms"], full["e"])
        if fm is not None and float(fm) > 0:
            m2_rv_ast = solve_m2_with_inclination_msun(
                float(fm), float(m1_msun), float(inclination_deg)
            )
    return {
        "used_m1_msun": m1_msun,
        "used_m2_msun": used_m2_msun,
        "m2sini_msun": m2sini,
        "m2_at_i_msun": None if m2_at_i is None else float(m2_at_i),
        "m2_rv_astrometry_msun": None if m2_rv_ast is None else float(m2_rv_ast),
        "inclination_deg_used": inclination_deg,
    }


def envelope_report(
    *,
    gaia_source_id: Optional[str],
    summary_file: str,
    n_points: int,
    t_ref_mjd: float,
    now_mjd: float,
    gaia_nss: Optional[dict],
    variants: Dict[str, dict],
    masses: Mapping[str, Optional[float]],
    observability_window: Optional[dict],
) -> Dict[str, Any]:
    rv_only = variants.get("rv_only") if isinstance(variants.get("rv_only"), dict) else {}
    out: Dict[str, Any] = {
        "gaia_source_id": gaia_source_id,
        "summary_file": summary_file,
        "n_points": int(n_points),
        "t_ref_mjd": float(t_ref_mjd),
        "now_mjd": float(now_mjd),
        "gaia_nss": gaia_nss,
        "fit_engine": "thejoker",
        "fit_variants": variants,
        "observability_window": observability_window,
    }
    out.update(dict(masses))
    if rv_only.get("P_days") is not None:
        out["P_days"] = rv_only.get("P_days")
        out["K_kms"] = rv_only.get("K_kms")
        out["e"] = rv_only.get("e")
        out["omega_deg"] = rv_only.get("omega_deg")
        out["gamma_kms"] = rv_only.get("gamma_kms")
        out["t_periastron_mjd"] = rv_only.get("t_periastron_mjd")
        out["params_raw"] = rv_only.get("params_raw")
        out["chi2"] = rv_only.get("chi2")
    # Keep chi2-fit website helpers working.
    if "rv_only" in variants:
        variants_alias = dict(variants)
        variants_alias["free"] = variants["rv_only"]
        out["fit_variants"] = variants_alias
    return out
