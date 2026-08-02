"""SB2 detection via mask CCF and two-template spectral separation."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from darkhunter_rv import config, continuum, gaia_utils, io_utils, physics, qc, rv_core, templates
from darkhunter_rv.blaze import BlazeCalibration
from darkhunter_rv.ccf_rv_estimators import (
    BiGaussCcfResult,
    EstimatorConfig,
    estimate_ccf_bi_gauss_from_arrays,
    estimate_ccf_secondary_seeded,
)
from darkhunter_rv.instruments import InstrumentProfile
from darkhunter_rv.rv_point_filters import rv_epoch_is_valid
from darkhunter_rv.summary_paths import discover_primary_epoch_files

logger = logging.getLogger(__name__)

_DEFAULT_TEFF1_WINDOW_K = 1200.0
_DEFAULT_DELTA_CHI2_MIN = 2.5
_DEFAULT_MIN_RV_SEP_KMS = 10.0
_DEFAULT_SECONDARY_SNR_MIN = 0.75
_DEFAULT_EXCLUDE_WIDTH_KMS = 12.0
_DEFAULT_MAX_SECONDARY_SEP_KMS = 50.0
_PRIOR_WEIGHT = 0.35
_PHOENIX_RAW_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray] | None] = {}
_COARSE_ORDER_STRIDE = 10
_REFINE_ORDER_STRIDE = 3


@dataclass
class EpochRvSeed:
    """Pipeline RV from star summary for one epoch."""

    basename: str
    mjd: float
    rv_kms: float
    rv_err_kms: float


@dataclass
class StarContext:
    """Gaia priors and per-epoch RV seeds loaded disk-first from summary."""

    gaia_id: str
    summary_path: Path
    teff: float
    logg: float
    mh: float
    epoch_rvs: dict[str, EpochRvSeed] = field(default_factory=dict)
    gaia_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderCcfRecord:
    """Bias-corrected mask CCF for one echelle order."""

    order: int
    chunk_key: str
    vel_kms: np.ndarray
    ccf: np.ndarray
    b0_kms: float
    peak_snr: float
    rv_kms: float


@dataclass
class EpochCcfAnalysis:
    """Median-CCF SB2 score for one exposure."""

    spectrum_path: str
    basename: str
    mjd: float
    order_records: list[OrderCcfRecord]
    median_vel_kms: np.ndarray
    median_ccf: np.ndarray
    bi_gauss: BiGaussCcfResult
    summary_rv_kms: float
    sb2_candidate: bool


@dataclass
class Sb2FitParams:
    """Shared stellar parameters from multi-epoch template fit."""

    teff1: float
    teff2: float
    feh: float
    logg1: float
    logg2: float
    vsini1: float
    vsini2: float


@dataclass
class EpochSb2Fit:
    """Per-epoch RVs from template fit."""

    basename: str
    vel1_kms: float
    vel2_kms: float
    vel1_err_kms: float
    vel2_err_kms: float


@dataclass
class Sb2TemplateFitResult:
    """Full multi-epoch two-template fit."""

    params: Sb2FitParams
    epochs: list[EpochSb2Fit]
    chi2_red: float
    success: bool
    n_data: int = 0
    n_dof: int = 0


@dataclass
class Sb2FitSettings:
    """Template-fit continuum, priors, and pixel masking."""

    continuum_mode: str = "sinc_blaze"
    blaze_calibration: BlazeCalibration | None = None
    order_edge_trim_frac: float = 0.0
    rv_prior_sigma_kms: float = 1.0
    vsini1_init_kms: float | None = None
    vsini1_max_kms: float | None = None


def load_sb2_blaze_calibration(
    continuum_mode: str,
    blaze_path: Path | None = None,
) -> tuple[str, BlazeCalibration | None]:
    """Resolve blaze calibration; fall back to spline when blaze file is missing."""
    mode = str(continuum_mode)
    if mode not in ("sinc_blaze", "sinc_blaze_only", "spline"):
        mode = "sinc_blaze"
    if mode == "spline":
        return mode, None
    path = Path(blaze_path) if blaze_path is not None else config.BLAZE_CALIBRATION_FILE
    if path.is_file():
        return mode, BlazeCalibration.load(path)
    logger.warning("Blaze calibration missing at %s; SB2 template fit uses spline", path)
    return "spline", None


def _continuum_kw(hot: bool, order: int) -> dict[str, Any]:
    return {
        "continuum_mode": "spline",
        "echelle_order": int(order),
        "exclude_near_lines_width": float(
            config.HOT_SPLINE_EXCLUDE_NEAR_LINES_WIDTH if hot else config.COOL_SPLINE_EXCLUDE_NEAR_LINES_WIDTH
        ),
    }


def _sb2_continuum_kw(hot: bool, order: int, settings: Sb2FitSettings) -> dict[str, Any]:
    """Continuum kwargs for SB2 template fit (blaze-aware when calibrated)."""
    mode = str(settings.continuum_mode)
    kw: dict[str, Any] = {"continuum_mode": mode, "echelle_order": int(order)}
    blaze_cal = settings.blaze_calibration
    if mode in ("sinc_blaze", "sinc_blaze_only") and blaze_cal is not None:
        model = blaze_cal.model_for_order(int(order))
        if model is not None:
            kw["blaze_model"] = model
        else:
            kw["continuum_mode"] = "spline"
            mode = "spline"
    if mode in ("spline", "sinc_blaze"):
        kw["exclude_near_lines_width"] = float(
            config.HOT_SPLINE_EXCLUDE_NEAR_LINES_WIDTH if hot else config.COOL_SPLINE_EXCLUDE_NEAR_LINES_WIDTH
        )
    return kw


def _order_edge_mask(n_pixels: int, edge_trim_frac: float) -> np.ndarray:
    """True for interior pixels; exclude first/last fraction when trim > 0."""
    n = int(n_pixels)
    ok = np.ones(n, dtype=bool)
    frac = float(edge_trim_frac)
    if n < 20 or frac <= 0.0:
        return ok
    n_trim = int(np.ceil(frac * n))
    n_trim = min(max(n_trim, 0), n // 2 - 5)
    if n_trim > 0:
        ok[:n_trim] = False
        ok[-n_trim:] = False
    return ok


def _fit_pixel_mask(
    wave: np.ndarray,
    obs_nf: np.ndarray,
    model_nf: np.ndarray,
    settings: Sb2FitSettings,
) -> np.ndarray:
    """Pixels entering SB2 template chi² (telluric, finite, optional edge trim)."""
    n = min(len(wave), len(obs_nf), len(model_nf))
    if n < 10:
        return np.zeros(0, dtype=bool)
    w = np.asarray(wave[:n], float)
    obs = np.asarray(obs_nf[:n], float)
    model = np.asarray(model_nf[:n], float)
    mask = (
        _telluric_mask(w)
        & np.isfinite(obs)
        & np.isfinite(model)
        & _order_edge_mask(n, settings.order_edge_trim_frac)
    )
    return mask


def parse_pipeline_rvs_from_summary(path: Path) -> dict[str, EpochRvSeed]:
    """Parse ``[PIPELINE RESULTS]`` RV epochs keyed by spectrum basename."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    if "[PIPELINE RESULTS]" in text:
        lines = text.split("[PIPELINE RESULTS]", 1)[-1].splitlines()
    else:
        lines = text.splitlines()
    out: dict[str, EpochRvSeed] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or (line.startswith("[") and line.endswith("]")):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if len(parts) >= 6 and parts[-1] in ("True", "False"):
            parts = parts[:-1]
        if len(parts) < 5:
            continue
        try:
            mjd = float(parts[1])
            rv = float(parts[2])
            rv_err = max(float(parts[3]), 1e-4)
        except ValueError:
            continue
        if not rv_epoch_is_valid(mjd, rv):
            continue
        bn = Path(parts[0]).name
        out[bn] = EpochRvSeed(basename=bn, mjd=mjd, rv_kms=rv, rv_err_kms=rv_err)
    return out


def load_star_context(
    gaia_id: str,
    summary_path: Path,
    *,
    force_gaia: bool = False,
) -> StarContext:
    """
    Load Gaia priors and pipeline RV seeds from ``summary_path``.

    Queries the Gaia archive only when the summary is missing or lacks complete
    ``[GAIA METADATA]`` (unless ``force_gaia``).
    """
    gid = str(gaia_id).strip()
    summary_path = Path(summary_path)
    gaia_data = gaia_utils.resolve_gaia_data(int(gid), summary_path, force_query=bool(force_gaia))
    meta = (gaia_data or {}).get("metadata") or {}
    teff = float(meta.get("Teff", meta.get("teff", 5500.0)) or 5500.0)
    logg = float(meta.get("logg", meta.get("Logg", 4.5)) or 4.5)
    mh = float(meta.get("MH", meta.get("[M/H]", meta.get("mh", 0.0))) or 0.0)
    epoch_rvs = parse_pipeline_rvs_from_summary(summary_path)
    return StarContext(
        gaia_id=gid,
        summary_path=summary_path,
        teff=teff,
        logg=logg,
        mh=mh,
        epoch_rvs=epoch_rvs,
        gaia_metadata=dict(meta),
    )


def select_stellar_mask(
    spec_data: dict,
    instrument: InstrumentProfile,
    test_orders: list[int],
    hot_continuum: bool,
) -> tuple[dict[str, np.ndarray] | None, str]:
    """Mask tournament: return (``{w,s}``, mask_stem) or (None, ``none``)."""
    mask_dir = Path(instrument.mask_directory)
    inst_lower = instrument.name.lower()
    mask_files = list(mask_dir.glob(f"*_{inst_lower}.txt"))
    if not mask_files:
        mask_files = list(mask_dir.glob("*_espresso.txt"))
    if not mask_files:
        return None, "none"

    scores: list[tuple[str, float]] = []
    best_peak = -np.inf
    best_pack: dict[str, np.ndarray] | None = None
    best_name = ""

    for mf in sorted(mask_files):
        try:
            md = np.loadtxt(mf)
            mw, ms = md[:, 0], md[:, 1]
        except Exception:
            continue
        peak_sum = 0.0
        for o in test_orders:
            if o not in spec_data:
                continue
            d = spec_data[o]
            w = np.array(d["wavelength"], float)
            f = np.array(d["flux"], float)
            e = np.array(d["eflux"], float)
            if len(w) < 10:
                continue
            try:
                nw, nf, _ = continuum.fit_continuum(w, f, e, **_continuum_kw(hot_continuum, int(o)))
            except Exception:
                continue
            if nw[-1] < mw[0] or nw[0] > mw[-1]:
                continue
            line_t = rv_core.mask_line_flux_in_excluded_wavelengths(nw, 1.0 - nf)
            _, _, _, _, p, _, _ = rv_core.cross_correlate_stellar_mask(nw, line_t, mw, ms)
            peak_sum += float(p)
        scores.append((mf.stem, peak_sum))
        if peak_sum > best_peak:
            best_peak = peak_sum
            best_pack = {"w": mw, "s": ms}
            best_name = mf.stem

    if not best_pack:
        return None, "none"
    return best_pack, best_name


def run_mask_ccf_per_order(
    spec_data: dict,
    mask_pack: dict[str, np.ndarray],
    instrument: InstrumentProfile,
    bias_table: dict,
    *,
    hot_continuum: bool,
    apply_bias: bool = True,
) -> list[OrderCcfRecord]:
    """One mask CCF per echelle order with optional ``b0`` velocity debias."""
    mw = mask_pack["w"]
    ms = mask_pack["s"]
    records: list[OrderCcfRecord] = []
    valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
    for order in valid_orders:
        d = spec_data[order]
        w = np.array(d["wavelength"], float)
        f = np.array(d["flux"], float)
        e = np.array(d["eflux"], float)
        if len(w) < 10:
            continue
        try:
            nw, nf, _ = continuum.fit_continuum(w, f, e, **_continuum_kw(hot_continuum, int(order)))
        except Exception:
            continue
        if nw[-1] < mw[0] or nw[0] > mw[-1]:
            continue
        line_t = rv_core.mask_line_flux_in_excluded_wavelengths(nw, 1.0 - nf)
        rv_m, _, vels_m, ccf_m, _, _, peak_snr_m = rv_core.cross_correlate_stellar_mask(nw, line_t, mw, ms)
        if vels_m is None or ccf_m is None:
            continue
        chunk_key = str(order)
        b0 = 0.0
        if apply_bias:
            bvec = io_utils.lookup_bias(bias_table, chunk_key)
            if isinstance(bvec, (list, tuple)) and len(bvec) >= 1:
                b0 = float(bvec[0])
        vel_corr = np.asarray(vels_m, float) - b0
        rv_corr = float(rv_m) - b0 if np.isfinite(rv_m) else float("nan")
        records.append(
            OrderCcfRecord(
                order=int(order),
                chunk_key=chunk_key,
                vel_kms=vel_corr,
                ccf=np.asarray(ccf_m, float),
                b0_kms=b0,
                peak_snr=float(peak_snr_m),
                rv_kms=rv_corr,
            )
        )
    return records


def median_ccf_across_orders(order_records: list[OrderCcfRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate per-order CCFs onto a common velocity grid and take the median."""
    if not order_records:
        return np.array([]), np.array([])
    v_lo = max(float(np.min(r.vel_kms)) for r in order_records)
    v_hi = min(float(np.max(r.vel_kms)) for r in order_records)
    if v_hi <= v_lo:
        v_lo = float(np.min([np.min(r.vel_kms) for r in order_records]))
        v_hi = float(np.max([np.max(r.vel_kms) for r in order_records]))
    n_grid = 401
    vel_grid = np.linspace(v_lo, v_hi, n_grid)
    stack = []
    for rec in order_records:
        ccf_i = np.interp(vel_grid, rec.vel_kms, rec.ccf, left=np.nan, right=np.nan)
        stack.append(ccf_i)
    mat = np.vstack(stack)
    with np.errstate(all="ignore"):
        median_ccf = np.nanmedian(mat, axis=0)
    ok = np.isfinite(median_ccf)
    if int(np.sum(ok)) < 10:
        return vel_grid, median_ccf
    fill = float(np.nanmedian(median_ccf[ok]))
    median_ccf = np.where(ok, median_ccf, fill)
    return vel_grid, median_ccf


def score_sb2_from_median_ccf(
    vel_kms: np.ndarray,
    ccf: np.ndarray,
    *,
    cfg: EstimatorConfig | None = None,
    rv_primary_seed: float | None = None,
    exclude_width_kms: float = _DEFAULT_EXCLUDE_WIDTH_KMS,
    max_secondary_sep_kms: float = _DEFAULT_MAX_SECONDARY_SEP_KMS,
) -> BiGaussCcfResult:
    """Bi-Gaussian or primary-seeded secondary fit on the bias-corrected median CCF."""
    cfg = cfg or EstimatorConfig()
    bi = estimate_ccf_bi_gauss_from_arrays(vel_kms, ccf, cfg=cfg)
    if rv_primary_seed is not None and np.isfinite(rv_primary_seed):
        seeded = estimate_ccf_secondary_seeded(
            vel_kms,
            ccf,
            float(rv_primary_seed),
            cfg=cfg,
            exclude_width_kms=float(exclude_width_kms),
            max_secondary_sep_kms=float(max_secondary_sep_kms),
        )
        if not bi.fit_ok:
            return seeded
        if seeded.fit_ok and (
            not np.isfinite(bi.delta_chi2)
            or (np.isfinite(seeded.delta_chi2) and seeded.delta_chi2 > bi.delta_chi2)
        ):
            return seeded
    return bi


def sb2_candidate_from_score(
    bi: BiGaussCcfResult,
    *,
    delta_chi2_min: float | None = None,
    min_rv_sep_kms: float = _DEFAULT_MIN_RV_SEP_KMS,
    secondary_snr_min: float = _DEFAULT_SECONDARY_SNR_MIN,
) -> bool:
    """True when fit succeeded and Δχ², RV separation, and secondary S/N pass thresholds."""
    if not bi.fit_ok or not np.isfinite(bi.delta_chi2):
        return False
    dmin = _DEFAULT_DELTA_CHI2_MIN if delta_chi2_min is None else float(delta_chi2_min)
    if bi.rv2_method == "seeded_peak":
        # Primary-seeded fits on weak secondaries can have small Δχ² in the narrow slice.
        dmin = min(dmin, 0.75)
    if bi.delta_chi2 < dmin:
        return False
    if not np.isfinite(bi.rv1_kms) or not np.isfinite(bi.rv2_kms):
        return False
    if abs(bi.rv1_kms - bi.rv2_kms) < float(min_rv_sep_kms):
        return False
    snr2 = bi.secondary_peak_snr
    if not np.isfinite(snr2) or snr2 < float(secondary_snr_min):
        return False
    return True


def analyze_epoch_ccf(
    spectrum_path: str | Path,
    spec_data: dict,
    mask_pack: dict[str, np.ndarray],
    instrument: InstrumentProfile,
    bias_table: dict,
    ctx: StarContext,
    *,
    hot_continuum: bool,
    apply_bias: bool = True,
    delta_chi2_min: float | None = None,
    cfg: EstimatorConfig | None = None,
) -> EpochCcfAnalysis:
    """Run per-order CCF, median stack, and SB2 scoring for one exposure."""
    spectrum_path = str(spectrum_path)
    basename = Path(spectrum_path).name
    order_records = run_mask_ccf_per_order(
        spec_data,
        mask_pack,
        instrument,
        bias_table,
        hot_continuum=hot_continuum,
        apply_bias=apply_bias,
    )
    vel_med, ccf_med = median_ccf_across_orders(order_records)
    seed = ctx.epoch_rvs.get(basename)
    summary_rv = float(seed.rv_kms) if seed is not None else float("nan")
    mjd = float(seed.mjd) if seed is not None else float("nan")
    rv_seed = summary_rv if np.isfinite(summary_rv) else None
    bi = score_sb2_from_median_ccf(vel_med, ccf_med, cfg=cfg, rv_primary_seed=rv_seed)
    candidate = sb2_candidate_from_score(bi, delta_chi2_min=delta_chi2_min)
    return EpochCcfAnalysis(
        spectrum_path=spectrum_path,
        basename=basename,
        mjd=mjd,
        order_records=order_records,
        median_vel_kms=vel_med,
        median_ccf=ccf_med,
        bi_gauss=bi,
        summary_rv_kms=summary_rv,
        sb2_candidate=candidate,
    )


def _vel_prior_sigma(err_kms: float, *, rv_prior_sigma_kms: float = 1.0) -> float:
    """Gaussian width for CCF velocity priors (fixed by default, not CCF err)."""
    _ = err_kms
    return max(float(rv_prior_sigma_kms), 1e-6)


def build_sb2_fit_settings(
    *,
    continuum_mode: str = "sinc_blaze",
    blaze_calibration_path: Path | None = None,
    order_edge_trim_frac: float = 0.0,
    rv_prior_sigma_kms: float = 1.0,
    vsini1_init_kms: float | None = None,
    vsini1_max_kms: float | None = None,
) -> Sb2FitSettings:
    """Construct fit settings with blaze loaded when requested."""
    mode, blaze = load_sb2_blaze_calibration(continuum_mode, blaze_calibration_path)
    return Sb2FitSettings(
        continuum_mode=mode,
        blaze_calibration=blaze,
        order_edge_trim_frac=float(order_edge_trim_frac),
        rv_prior_sigma_kms=float(rv_prior_sigma_kms),
        vsini1_init_kms=vsini1_init_kms,
        vsini1_max_kms=vsini1_max_kms,
    )


def _doppler_flux_on_grid(
    wave_obs: np.ndarray,
    wave_tpl: np.ndarray,
    flux_tpl: np.ndarray,
    vel_kms: float,
    resolving_power: float,
) -> np.ndarray:
    """Broadened, Doppler-shifted PHOENIX flux interpolated onto ``wave_obs``."""
    beta = 1.0 + float(vel_kms) / config.C_KMS
    w_shift = np.asarray(wave_tpl, float) * beta
    order = np.argsort(w_shift)
    flux_i = np.interp(wave_obs, w_shift[order], np.asarray(flux_tpl, float)[order], left=np.nan, right=np.nan)
    if resolving_power > 1.0:
        flux_i = rv_core.degrade_template_flux_lsf(wave_obs, flux_i, resolving_power)
    return np.asarray(flux_i, dtype=float)


def _load_phoenix_raw_cached(
    teff: float,
    logg: float,
    feh: float,
    wave_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    key = (round(float(teff)), round(float(logg), 1), round(float(feh), 2), round(wave_range[0]), round(wave_range[1]))
    if key in _PHOENIX_RAW_CACHE:
        hit = _PHOENIX_RAW_CACHE[key]
        return None if hit is None else (hit[0], hit[1])
    loaded = templates.load_hires_phoenix_raw_nearest(
        teff,
        logg,
        feh,
        wave_range,
        air=True,
        mh_tol=0.85,
        dg_max=2.5,
        dt_max=12000.0,
    )
    if loaded is None:
        _PHOENIX_RAW_CACHE[key] = None
        return None
    wave_tpl, flux_raw, _ = loaded
    _PHOENIX_RAW_CACHE[key] = (wave_tpl, flux_raw)
    return wave_tpl, flux_raw


def _load_phoenix_component(
    teff: float,
    logg: float,
    feh: float,
    vsini: float,
    vel_kms: float,
    wave_obs: np.ndarray,
    resolving_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (wave_obs, flux_on_obs, flux_rest_on_obs) for one stellar component."""
    wlo = float(np.nanmin(wave_obs))
    whi = float(np.nanmax(wave_obs))
    raw = _load_phoenix_raw_cached(teff, logg, feh, (wlo - 50.0, whi + 50.0))
    if raw is None:
        return None
    wave_tpl, flux_raw = raw
    flux_broad = physics.broaden_spectrum(wave_tpl, flux_raw, float(vsini))
    flux_shift = _doppler_flux_on_grid(wave_obs, wave_tpl, flux_broad, vel_kms, resolving_power)
    flux_rest = _doppler_flux_on_grid(wave_obs, wave_tpl, flux_broad, 0.0, resolving_power)
    return wave_obs, flux_shift, flux_rest


def _normalize_blend_order(
    wave: np.ndarray,
    flux_blend: np.ndarray,
    hot: bool,
    order: int,
    settings: Sb2FitSettings | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Continuum-normalize a flux blend; returns aligned (wavelength, norm_flux)."""
    fit_settings = settings or Sb2FitSettings(continuum_mode="spline")
    n = min(len(wave), len(flux_blend))
    if n < 10:
        return None
    w = np.asarray(wave[:n], float)
    f = np.asarray(flux_blend[:n], float)
    try:
        dummy_err = np.ones(n, float) * 0.01
        nw, nf, _ = continuum.fit_continuum(
            w,
            f,
            dummy_err,
            **_sb2_continuum_kw(hot, order, fit_settings),
        )
        m = min(len(nw), len(nf))
        if m < 10:
            return None
        return np.asarray(nw[:m], float), np.asarray(nf[:m], float)
    except Exception:
        return None


def flux_fraction_components(
    f1: np.ndarray,
    f2: np.ndarray,
    total_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose a jointly normalized blend into per-star contributions.

    With blend = f1 + f2 and total = norm(blend), returns
    (f1/blend)*total and (f2/blend)*total so components sum to total.
    """
    n = min(len(f1), len(f2), len(total_norm))
    f1 = np.asarray(f1[:n], float)
    f2 = np.asarray(f2[:n], float)
    total_norm = np.asarray(total_norm[:n], float)
    blend = f1 + f2
    with np.errstate(divide="ignore", invalid="ignore"):
        frac1 = np.where(np.abs(blend) > 0.0, f1 / blend, np.nan)
        frac2 = np.where(np.abs(blend) > 0.0, f2 / blend, np.nan)
    return frac1 * total_norm, frac2 * total_norm


def _interp_flux_on_wave(wave_out: np.ndarray, wave_in: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Interpolate flux onto ``wave_out`` when blaze continuum trims pixels."""
    w_out = np.asarray(wave_out, float)
    w_in = np.asarray(wave_in, float)
    f_in = np.asarray(flux, float)
    if len(w_out) == len(w_in) and np.allclose(w_out, w_in, rtol=0.0, atol=1e-9):
        return f_in.copy()
    return np.interp(w_out, w_in, f_in, left=np.nan, right=np.nan)


def _normalized_blend_components(
    wave: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    hot: bool,
    order: int,
    settings: Sb2FitSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (wave_out, star1, star2, total) from raw fluxes using fit-consistent normalization."""
    n = min(len(wave), len(f1), len(f2))
    if n < 10:
        return None
    wave = np.asarray(wave[:n], float)
    f1 = np.asarray(f1[:n], float)
    f2 = np.asarray(f2[:n], float)
    blend = f1 + f2
    norm_pack = _normalize_blend_order(wave, blend, hot, order, settings=settings)
    if norm_pack is None:
        return None
    nw, total = norm_pack
    f1_on = _interp_flux_on_wave(nw, wave, f1)
    f2_on = _interp_flux_on_wave(nw, wave, f2)
    star1, star2 = flux_fraction_components(f1_on, f2_on, total)
    return nw, star1, star2, total


def _telluric_mask(wave: np.ndarray) -> np.ndarray:
    ok = np.ones(len(wave), dtype=bool)
    for lo, hi in qc.rv_contamination_bands():
        ok &= ~((wave >= lo) & (wave <= hi))
    return ok


def _order_obs_norm(
    spec_data: dict,
    order: int,
    hot: bool,
    settings: Sb2FitSettings | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    fit_settings = settings or Sb2FitSettings(continuum_mode="spline")
    d = spec_data[order]
    w = np.array(d["wavelength"], float)
    f = np.array(d["flux"], float)
    e = np.array(d["eflux"], float)
    if len(w) < 10:
        return None
    try:
        nw, nf, _ = continuum.fit_continuum(w, f, e, **_sb2_continuum_kw(hot, int(order), fit_settings))
        return nw, nf
    except Exception:
        return None


@dataclass
class _EpochSpec:
    basename: str
    spec_data: dict
    vel1_seed: float
    vel2_seed: float
    vel1_sigma: float
    vel2_sigma: float
    summary_rv: float


def _teff2_grid(teff1: float, *, step: float = 300.0) -> np.ndarray:
    """Secondary Teff coarse grid (cooler than primary)."""
    hi = min(5500.0, float(teff1) - 50.0)
    if hi <= 3000.0:
        return np.array([], float)
    return np.arange(3000.0, hi + 1.0, step)


def _sb2_n_dof(n_data: int, n_epochs: int) -> int:
    """Degrees of freedom: spectral pixels minus shared + per-epoch velocity params."""
    n_params = 7 + 2 * int(n_epochs)
    return max(int(n_data) - n_params, 1)


def _sb2_chi2_red(chi2_spec: float, n_data: int, n_epochs: int) -> float:
    return float(chi2_spec) / float(_sb2_n_dof(n_data, n_epochs))


def _teff_grid(teff_center: float, half_width: float, teff_max: float | None = None, *, step: float = 400.0) -> np.ndarray:
    lo = max(3200.0, teff_center - half_width)
    hi = teff_center + half_width
    if teff_max is not None:
        hi = min(hi, teff_max)
    if hi <= lo:
        return np.array([], float)
    return np.arange(lo, hi + 1.0, step)


def _resolve_vsini_init_max(
    *,
    vsini_guess: float,
    vsini_init_kms: float | None,
    vsini_max_kms: float | None,
) -> tuple[float, float]:
    init = float(vsini_init_kms) if vsini_init_kms is not None else float(vsini_guess)
    vmax = float(vsini_max_kms) if vsini_max_kms is not None else 80.0
    init = max(init, 0.0)
    vmax = max(vmax, 0.0)
    if init > vmax:
        init = vmax
    return init, vmax


def _coarse_sb2_grid(
    ctx: StarContext,
    epochs: list[_EpochSpec],
    instrument: InstrumentProfile,
    settings: Sb2FitSettings,
    *,
    teff1_window: float,
) -> tuple[Sb2FitParams, list[float], list[float]] | None:
    """Grid search Teff1/Teff2 on a single reference order; return best params and vel seeds."""
    hot = ctx.teff > config.HOT_STAR_TEFF_THRESHOLD
    valid_orders = sorted(o for o in epochs[0].spec_data if o not in instrument.bad_orders)
    if not valid_orders:
        return None
    ref_order = valid_orders[len(valid_orders) // 2]
    obs_pack = _order_obs_norm(epochs[0].spec_data, ref_order, hot, settings=settings)
    if obs_pack is None:
        return None
    ow, onf = obs_pack
    tpl0 = templates.load_hires_phoenix_raw_nearest(ctx.teff, ctx.logg, ctx.mh, (ow[0], ow[-1]), mh_tol=0.85)
    vsini_guess = 10.0
    if tpl0 is not None:
        tw, tf, _ = tpl0
        vb, _ = rv_core.estimate_broadening(ow, onf, tw, tf)
        if vb is not None and np.isfinite(vb):
            vsini_guess = float(max(vb, 0.5))

    vsini1_init, _ = _resolve_vsini_init_max(
        vsini_guess=vsini_guess,
        vsini_init_kms=settings.vsini1_init_kms,
        vsini_max_kms=settings.vsini1_max_kms,
    )
    vsini2_init = float(vsini_guess)

    teff1_vals = _teff_grid(ctx.teff, teff1_window, step=600.0)
    logg2_vals = [3.5, 4.0, 4.5]
    best_chi2 = np.inf
    best: tuple[Sb2FitParams, list[float], list[float]] | None = None
    rp = float(instrument.resolving_power)
    sig_prior = float(settings.rv_prior_sigma_kms)

    for t1 in teff1_vals:
        teff2_vals = _teff2_grid(float(t1), step=300.0)
        for t2 in teff2_vals:
            for g2 in logg2_vals:
                params = Sb2FitParams(
                    teff1=float(t1),
                    teff2=float(t2),
                    feh=float(ctx.mh),
                    logg1=float(ctx.logg),
                    logg2=float(g2),
                    vsini1=vsini1_init,
                    vsini2=vsini2_init,
                )
                chi2 = 0.0
                v1_list = [e.vel1_seed for e in epochs]
                v2_list = [e.vel2_seed for e in epochs]
                for ep, v1, v2 in zip(epochs, v1_list, v2_list):
                    obs = _order_obs_norm(ep.spec_data, ref_order, hot, settings=settings)
                    if obs is None:
                        continue
                    wave, obs_nf = obs
                    c1 = _load_phoenix_component(
                        params.teff1, params.logg1, params.feh, params.vsini1, v1, wave, rp
                    )
                    c2 = _load_phoenix_component(
                        params.teff2, params.logg2, params.feh, params.vsini2, v2, wave, rp
                    )
                    if c1 is None or c2 is None:
                        chi2 = np.inf
                        break
                    _, f1, _ = c1
                    _, f2, _ = c2
                    norm_pack = _normalize_blend_order(wave, f1 + f2, hot, ref_order, settings=settings)
                    if norm_pack is None:
                        chi2 = np.inf
                        break
                    nw, model_nf = norm_pack
                    n = min(len(wave), len(obs_nf), len(nw), len(model_nf))
                    obs_nf = np.asarray(obs_nf[:n], float)
                    model_nf = np.asarray(model_nf[:n], float)
                    mask = _fit_pixel_mask(nw[:n], obs_nf, model_nf, settings)
                    if int(np.sum(mask)) < 20:
                        continue
                    chi2 += float(np.sum((obs_nf[mask] - model_nf[mask]) ** 2))
                    chi2 += (_PRIOR_WEIGHT * (v1 - ep.vel1_seed) / sig_prior) ** 2
                    chi2 += (_PRIOR_WEIGHT * (v2 - ep.vel2_seed) / sig_prior) ** 2
                if np.isfinite(chi2) and chi2 < best_chi2:
                    best_chi2 = chi2
                    best = (params, v1_list, v2_list)
    return best


def _residual_vector_sb2(
    params: Sb2FitParams,
    epochs: list[_EpochSpec],
    instrument: InstrumentProfile,
    hot: bool,
    vel1_list: list[float],
    vel2_list: list[float],
    settings: Sb2FitSettings,
    *,
    order_stride: int = 1,
) -> tuple[np.ndarray, int]:
    """Stacked data and prior residuals for ``least_squares``; returns (residuals, n_spec_pixels)."""
    chunks: list[float] = []
    n_data = 0
    rp = float(instrument.resolving_power)
    sig_prior = float(settings.rv_prior_sigma_kms)
    for ep, v1, v2 in zip(epochs, vel1_list, vel2_list):
        valid_orders = sorted(o for o in ep.spec_data if o not in instrument.bad_orders)
        if order_stride > 1:
            valid_orders = valid_orders[::order_stride]
        for order in valid_orders:
            obs = _order_obs_norm(ep.spec_data, order, hot, settings=settings)
            if obs is None:
                continue
            wave, obs_nf = obs
            c1 = _load_phoenix_component(
                params.teff1, params.logg1, params.feh, params.vsini1, v1, wave, rp
            )
            c2 = _load_phoenix_component(
                params.teff2, params.logg2, params.feh, params.vsini2, v2, wave, rp
            )
            if c1 is None or c2 is None:
                continue
            _, f1, _ = c1
            _, f2, _ = c2
            blend = f1 + f2
            norm_pack = _normalize_blend_order(wave, blend, hot, order, settings=settings)
            if norm_pack is None:
                continue
            nw, model_nf = norm_pack
            n = min(len(wave), len(obs_nf), len(nw), len(model_nf))
            obs_nf = np.asarray(obs_nf[:n], float)
            model_nf = np.asarray(model_nf[:n], float)
            mask = _fit_pixel_mask(nw[:n], obs_nf, model_nf, settings)
            if int(np.sum(mask)) < 20:
                continue
            n_data += int(np.sum(mask))
            chunks.extend((obs_nf[mask] - model_nf[mask]).tolist())
        chunks.append(_PRIOR_WEIGHT * (v1 - ep.vel1_seed) / sig_prior)
        chunks.append(_PRIOR_WEIGHT * (v2 - ep.vel2_seed) / sig_prior)
    if not chunks:
        return np.array([1e6]), 0
    return np.asarray(chunks, float), n_data


def _evaluate_sb2_params(
    params: Sb2FitParams,
    epochs: list[_EpochSpec],
    instrument: InstrumentProfile,
    hot: bool,
    settings: Sb2FitSettings,
    vel1_list: list[float] | None = None,
    vel2_list: list[float] | None = None,
    *,
    order_stride: int = 1,
) -> tuple[float, int, list[float], list[float]]:
    """Return (spectral chi2, n_data pixels, per-epoch vel1, per-epoch vel2)."""
    if vel1_list is None:
        vel1_list = [e.vel1_seed for e in epochs]
    if vel2_list is None:
        vel2_list = [e.vel2_seed for e in epochs]
    res, n_data = _residual_vector_sb2(
        params, epochs, instrument, hot, vel1_list, vel2_list, settings, order_stride=order_stride
    )
    n_ep = len(epochs)
    n_prior = 2 * n_ep
    spec_res = res[:-n_prior] if n_prior > 0 and res.size > n_prior else res
    chi2 = float(np.sum(spec_res**2))
    return chi2, n_data, vel1_list, vel2_list


def fit_sb2_templates_multi_epoch(
    ctx: StarContext,
    epoch_analyses: list[EpochCcfAnalysis],
    spec_paths: list[Path],
    instrument: InstrumentProfile,
    settings: Sb2FitSettings,
    *,
    teff1_window: float = _DEFAULT_TEFF1_WINDOW_K,
    refine: bool = False,
) -> Sb2TemplateFitResult | None:
    """
    Simultaneous two-template fit across epochs.

    Shared: Teff1, Teff2, feh, logg1, logg2, vsini1, vsini2. Per epoch: vel1, vel2 with
    Gaussian priors from median-CCF seeds (width ``settings.rv_prior_sigma_kms``).
    """
    epochs: list[_EpochSpec] = []
    for path, analysis in zip(spec_paths, epoch_analyses):
        header, spec_data = io_utils.read_spectrum(str(path))
        bi = analysis.bi_gauss
        seed = ctx.epoch_rvs.get(analysis.basename)
        v1_seed = bi.rv1_kms if bi.fit_ok and np.isfinite(bi.rv1_kms) else (
            seed.rv_kms if seed is not None else float("nan")
        )
        v2_seed = bi.rv2_kms if bi.fit_ok and np.isfinite(bi.rv2_kms) else float("nan")
        if not np.isfinite(v1_seed) and seed is not None:
            v1_seed = seed.rv_kms
        if not np.isfinite(v2_seed):
            v2_seed = v1_seed + 20.0 if np.isfinite(v1_seed) else 0.0
        epochs.append(
            _EpochSpec(
                basename=analysis.basename,
                spec_data=spec_data,
                vel1_seed=float(v1_seed),
                vel2_seed=float(v2_seed),
                vel1_sigma=float(bi.rv1_err_kms),
                vel2_sigma=float(bi.rv2_err_kms),
                summary_rv=float(seed.rv_kms) if seed is not None else float("nan"),
            )
        )
    if not epochs:
        return None

    coarse = _coarse_sb2_grid(
        ctx,
        epochs,
        instrument,
        settings,
        teff1_window=teff1_window,
    )
    if coarse is None:
        return None
    params0, v1_list, v2_list = coarse

    n_ep = len(epochs)
    x0 = np.array(
        [
            params0.teff1,
            params0.teff2,
            params0.feh,
            params0.logg1,
            params0.logg2,
            params0.vsini1,
            params0.vsini2,
            *v1_list,
            *v2_list,
        ],
        float,
    )
    hot = ctx.teff > config.HOT_STAR_TEFF_THRESHOLD

    def _pack(x: np.ndarray) -> tuple[Sb2FitParams, list[float], list[float]]:
        p = Sb2FitParams(
            teff1=float(x[0]),
            teff2=float(x[1]),
            feh=float(x[2]),
            logg1=float(x[3]),
            logg2=float(x[4]),
            vsini1=float(max(x[5], 0.0)),
            vsini2=float(max(x[6], 0.0)),
        )
        v1 = [float(v) for v in x[7 : 7 + n_ep]]
        v2 = [float(v) for v in x[7 + n_ep : 7 + 2 * n_ep]]
        return p, v1, v2

    def resid(x: np.ndarray) -> np.ndarray:
        p, v1, v2 = _pack(x)
        if p.teff2 >= p.teff1 - 50.0:
            return np.full(1, 1e6)
        r, _ = _residual_vector_sb2(p, epochs, instrument, hot, v1, v2, settings, order_stride=1)
        return r

    _, vsini1_max = _resolve_vsini_init_max(
        vsini_guess=float(params0.vsini1),
        vsini_init_kms=settings.vsini1_init_kms,
        vsini_max_kms=settings.vsini1_max_kms,
    )

    lb = np.array(
        [
            2800.0,
            2800.0,
            -2.5,
            0.0,
            0.0,
            0.0,
            0.0,
            *([-500.0] * n_ep),
            *([-500.0] * n_ep),
        ],
        float,
    )
    ub = np.array(
        [
            12000.0,
            10000.0,
            0.8,
            5.5,
            5.5,
            vsini1_max,
            80.0,
            *([500.0] * n_ep),
            *([500.0] * n_ep),
        ],
        float,
    )
    ub[1] = min(ub[1], x0[0] - 50.0)
    p_final, v1_f, v2_f = params0, v1_list, v2_list
    success = coarse is not None
    chi2_red = float("nan")
    n_data = 0
    n_dof = 0
    if refine:
        try:
            sol = least_squares(
                resid,
                np.clip(x0, lb + 1e-6, ub - 1e-6),
                bounds=(lb, ub),
                max_nfev=25,
            )
        except Exception:
            sol = None
        if sol is not None and sol.success:
            p_final, v1_f, v2_f = _pack(sol.x)
            chi2, n_data, _, _ = _evaluate_sb2_params(
                p_final, epochs, instrument, hot, settings, v1_f, v2_f, order_stride=_REFINE_ORDER_STRIDE
            )
            n_dof = _sb2_n_dof(n_data, n_ep)
            chi2_red = _sb2_chi2_red(chi2, n_data, n_ep)
            success = True
    else:
        chi2, n_data, _, _ = _evaluate_sb2_params(
            params0, epochs, instrument, hot, settings, v1_list, v2_list, order_stride=_REFINE_ORDER_STRIDE
        )
        n_dof = _sb2_n_dof(n_data, n_ep)
        chi2_red = _sb2_chi2_red(chi2, n_data, n_ep)

    sig_prior = float(settings.rv_prior_sigma_kms)
    epoch_fits = [
        EpochSb2Fit(
            basename=ep.basename,
            vel1_kms=float(v1),
            vel2_kms=float(v2),
            vel1_err_kms=sig_prior,
            vel2_err_kms=sig_prior,
        )
        for ep, v1, v2 in zip(epochs, v1_f, v2_f)
    ]
    return Sb2TemplateFitResult(
        params=p_final,
        epochs=epoch_fits,
        chi2_red=chi2_red,
        success=success,
        n_data=n_data,
        n_dof=n_dof,
    )


def separated_spectra_for_epoch(
    params: Sb2FitParams,
    spec_data: dict,
    instrument: InstrumentProfile,
    *,
    hot: bool,
    vel1_kms: float,
    vel2_kms: float,
    settings: Sb2FitSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Return (wave, primary_contrib, secondary_contrib) concatenated over orders.

    Components are flux-fraction decompositions of norm(f1+f2) at the epoch RVs,
    so primary + secondary equals the fit total model (not separately normalized).
    """
    fit_settings = settings or Sb2FitSettings(continuum_mode="spline")
    waves: list[np.ndarray] = []
    prim: list[np.ndarray] = []
    sec: list[np.ndarray] = []
    rp = float(instrument.resolving_power)
    valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
    for order in valid_orders:
        obs = _order_obs_norm(spec_data, order, hot, settings=fit_settings)
        if obs is None:
            continue
        wave, _ = obs
        c1 = _load_phoenix_component(
            params.teff1, params.logg1, params.feh, params.vsini1, vel1_kms, wave, rp
        )
        c2 = _load_phoenix_component(
            params.teff2, params.logg2, params.feh, params.vsini2, vel2_kms, wave, rp
        )
        if c1 is None or c2 is None:
            continue
        _, f1, _ = c1
        _, f2, _ = c2
        decomp = _normalized_blend_components(wave, f1, f2, hot, order, settings=fit_settings)
        if decomp is None:
            continue
        nw, n1, n2, _total = decomp
        m = min(len(nw), len(n1), len(n2))
        if m < 20:
            continue
        waves.append(np.asarray(nw[:m], float))
        prim.append(np.asarray(n1[:m], float))
        sec.append(np.asarray(n2[:m], float))
    if not waves:
        return None
    return np.concatenate(waves), np.concatenate(prim), np.concatenate(sec)


def separated_spectra_at_rest(
    params: Sb2FitParams,
    spec_data: dict,
    instrument: InstrumentProfile,
    *,
    hot: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Backward-compatible wrapper at v=0."""
    return separated_spectra_for_epoch(
        params, spec_data, instrument, hot=hot, vel1_kms=0.0, vel2_kms=0.0
    )


def order_records_to_dataframe(records: list[OrderCcfRecord]) -> pd.DataFrame:
    """Flatten per-order CCF peaks for CSV export."""
    rows = []
    for r in records:
        rows.append(
            {
                "order": r.order,
                "chunk_key": r.chunk_key,
                "b0_kms": r.b0_kms,
                "peak_snr": r.peak_snr,
                "rv_kms": r.rv_kms,
            }
        )
    return pd.DataFrame(rows)


def epoch_analysis_to_row(analysis: EpochCcfAnalysis) -> dict[str, Any]:
    """Summary row for ``sb2_epochs.csv``."""
    bi = analysis.bi_gauss
    return {
        "basename": analysis.basename,
        "spectrum_path": analysis.spectrum_path,
        "mjd": analysis.mjd,
        "summary_rv_kms": analysis.summary_rv_kms,
        "rv1_kms": bi.rv1_kms,
        "rv2_kms": bi.rv2_kms,
        "rv1_err_kms": bi.rv1_err_kms,
        "rv2_err_kms": bi.rv2_err_kms,
        "delta_chi2": bi.delta_chi2,
        "secondary_peak_snr": bi.secondary_peak_snr,
        "rv2_method": bi.rv2_method,
        "exclude_width_kms": bi.exclude_width_kms,
        "sb2_candidate": analysis.sb2_candidate,
        "n_orders": len(analysis.order_records),
    }


def bi_gauss_to_dict(bi: BiGaussCcfResult) -> dict[str, Any]:
    """JSON-serializable bi-Gauss result."""
    d = asdict(bi)
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            d[k] = float(v)
    return d


def discover_epoch_paths(spec_root: Path, gaia_id: str) -> list[Path]:
    """Primary epoch spectra for a Gaia source."""
    return discover_primary_epoch_files(spec_root, gaia_id)


def write_separated_spectrum_txt(path: Path, wave: np.ndarray, flux: np.ndarray) -> None:
    """Two-column wavelength / normalized flux text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([wave, flux])
    np.savetxt(path, data, fmt="%.6f %.8f", header="wavelength_A flux_norm")


def fit_result_to_json(fit: Sb2TemplateFitResult, settings: Sb2FitSettings | None = None) -> dict[str, Any]:
    """Serialize template fit for ``sb2_fit.json``."""
    out: dict[str, Any] = {
        "params": asdict(fit.params),
        "epochs": [asdict(e) for e in fit.epochs],
        "chi2_red": float(fit.chi2_red) if np.isfinite(fit.chi2_red) else None,
        "n_data": int(fit.n_data),
        "n_dof": int(fit.n_dof),
        "success": bool(fit.success),
    }
    if settings is not None:
        out["fit_settings"] = {
            "continuum_mode": settings.continuum_mode,
            "order_edge_trim_frac": float(settings.order_edge_trim_frac),
            "rv_prior_sigma_kms": float(settings.rv_prior_sigma_kms),
            "vsini1_init_kms": settings.vsini1_init_kms,
            "vsini1_max_kms": settings.vsini1_max_kms,
            "blaze_loaded": settings.blaze_calibration is not None,
        }
    return out


def run_sb2_star(
    gaia_id: str,
    spec_root: Path,
    summary_path: Path,
    instrument: InstrumentProfile,
    out_dir: Path,
    *,
    force_gaia: bool = False,
    force_fit: bool = False,
    apply_bias: bool = True,
    teff1_window: float = _DEFAULT_TEFF1_WINDOW_K,
    delta_chi2_min: float | None = None,
    write_plots: bool = False,
    refine_fit: bool = False,
    vsini1_init_kms: float | None = None,
    vsini1_max_kms: float | None = None,
    continuum_mode: str = "sinc_blaze",
    order_edge_trim_frac: float = 0.0,
    rv_prior_sigma_kms: float = 1.0,
    blaze_calibration_path: Path | None = None,
    return_epoch_analyses: bool = False,
) -> dict[str, Any]:
    """
    End-to-end SB2 search for one Gaia star: CCF detection, optional template fit, outputs.

    Writes ``sb2_report.json``, ``sb2_orders.csv``, ``sb2_epochs.csv``, and when SB2
  detected (or ``force_fit``): ``sb2_fit.json`` plus per-epoch separated spectra.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_star_context(gaia_id, summary_path, force_gaia=force_gaia)
    spec_paths = discover_epoch_paths(spec_root, gaia_id)
    if not spec_paths:
        raise FileNotFoundError(f"No spectra under {spec_root} for Gaia {gaia_id}")

    hot = ctx.teff > config.HOT_STAR_TEFF_THRESHOLD
    bias_table: dict = {}
    if apply_bias and instrument.bias_file:
        bias_table = io_utils.read_bias(instrument.bias_file)

    epoch_analyses: list[EpochCcfAnalysis] = []
    order_rows: list[dict[str, Any]] = []

    for spec_path in spec_paths:
        _, spec_data = io_utils.read_spectrum(str(spec_path))
        valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
        if not valid_orders:
            continue
        mid = len(valid_orders) // 2
        test_orders = valid_orders[max(0, mid - 2) : min(len(valid_orders), mid + 2)]
        mask_pack, mask_name = select_stellar_mask(spec_data, instrument, test_orders, hot)
        if mask_pack is None:
            logger.warning("No mask for %s", spec_path)
            continue
        analysis = analyze_epoch_ccf(
            spec_path,
            spec_data,
            mask_pack,
            instrument,
            bias_table,
            ctx,
            hot_continuum=hot,
            apply_bias=apply_bias,
            delta_chi2_min=delta_chi2_min,
        )
        epoch_analyses.append(analysis)
        df_ord = order_records_to_dataframe(analysis.order_records)
        df_ord.insert(0, "basename", analysis.basename)
        df_ord.insert(1, "mask_name", mask_name)
        order_rows.extend(df_ord.to_dict(orient="records"))

    if not epoch_analyses:
        raise RuntimeError(f"No valid epoch CCF analyses for Gaia {gaia_id}")

    epoch_df = pd.DataFrame([epoch_analysis_to_row(a) for a in epoch_analyses])
    epoch_df.to_csv(out_dir / "sb2_epochs.csv", index=False)
    if order_rows:
        pd.DataFrame(order_rows).to_csv(out_dir / "sb2_orders.csv", index=False)

    any_sb2 = any(a.sb2_candidate for a in epoch_analyses)
    median_delta = float(np.nanmedian([a.bi_gauss.delta_chi2 for a in epoch_analyses]))
    fit_result: Sb2TemplateFitResult | None = None
    fit_settings: Sb2FitSettings | None = None

    base_report = {
        "gaia_id": gaia_id,
        "summary_path": str(summary_path),
        "teff_gaia": ctx.teff,
        "logg_gaia": ctx.logg,
        "mh_gaia": ctx.mh,
        "n_epochs": len(epoch_analyses),
        "median_delta_chi2": median_delta,
        "any_sb2_candidate": any_sb2,
        "epochs": [epoch_analysis_to_row(a) for a in epoch_analyses],
        "fit": None,
    }
    (out_dir / "sb2_report.json").write_text(json.dumps(base_report, indent=2), encoding="utf-8")

    if any_sb2 or force_fit:
        fit_settings = build_sb2_fit_settings(
            continuum_mode=continuum_mode,
            blaze_calibration_path=blaze_calibration_path,
            order_edge_trim_frac=order_edge_trim_frac,
            rv_prior_sigma_kms=rv_prior_sigma_kms,
            vsini1_init_kms=vsini1_init_kms,
            vsini1_max_kms=vsini1_max_kms,
        )
        fit_analyses = [a for a in epoch_analyses if a.sb2_candidate] if any_sb2 else epoch_analyses
        fit_paths = [
            p for p, a in zip(spec_paths, epoch_analyses) if a in fit_analyses
        ]
        fit_result = fit_sb2_templates_multi_epoch(
            ctx,
            fit_analyses,
            fit_paths,
            instrument,
            fit_settings,
            teff1_window=teff1_window,
            refine=bool(refine_fit or force_fit),
        )
        if fit_result is not None:
            (out_dir / "sb2_fit.json").write_text(
                json.dumps(fit_result_to_json(fit_result, fit_settings), indent=2),
                encoding="utf-8",
            )
            hot_fit = ctx.teff > config.HOT_STAR_TEFF_THRESHOLD
            for spec_path, ep_fit in zip(fit_paths, fit_result.epochs):
                _, spec_data = io_utils.read_spectrum(str(spec_path))
                sep = separated_spectra_for_epoch(
                    fit_result.params,
                    spec_data,
                    instrument,
                    hot=hot_fit,
                    vel1_kms=float(ep_fit.vel1_kms),
                    vel2_kms=float(ep_fit.vel2_kms),
                    settings=fit_settings,
                )
                if sep is None:
                    continue
                wave, pflux, sflux = sep
                stem = Path(ep_fit.basename).stem
                write_separated_spectrum_txt(out_dir / f"{stem}_primary_v0.txt", wave, pflux)
                write_separated_spectrum_txt(out_dir / f"{stem}_secondary_v0.txt", wave, sflux)

    report = dict(base_report)
    report["fit"] = fit_result_to_json(fit_result, fit_settings) if fit_result is not None else None
    (out_dir / "sb2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if return_epoch_analyses:
        report["epoch_analyses"] = epoch_analyses

    if write_plots:
        try:
            import matplotlib.pyplot as plt

            for analysis in epoch_analyses:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(analysis.median_vel_kms, analysis.median_ccf, "k-", lw=1.2, label="median CCF")
                bi = analysis.bi_gauss
                if bi.fit_ok:
                    ax.axvline(bi.rv1_kms, color="C0", ls="--", label=f"rv1={bi.rv1_kms:.1f}")
                    ax.axvline(bi.rv2_kms, color="C1", ls="--", label=f"rv2={bi.rv2_kms:.1f}")
                ax.set_xlabel("Velocity (km/s)")
                ax.set_ylabel("CCF")
                ax.legend(fontsize=8)
                ax.set_title(f"{analysis.basename} Δχ²={bi.delta_chi2:.1f}")
                fig.savefig(out_dir / f"{Path(analysis.basename).stem}_median_ccf.png", dpi=120)
                plt.close(fig)
        except Exception as ex:
            logger.warning("Plotting failed: %s", ex)

    return report
