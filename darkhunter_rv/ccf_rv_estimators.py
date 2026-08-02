"""Mask-CCF RV estimators: grid, parabolic, Gaussian, smoothed peak, bi-Gaussian."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, least_squares

from darkhunter_rv import config, qc

_DEFAULT_MIN_CCF_PEAK_SNR = 3.2
_DEFAULT_MAX_GAUSS_OFFSET_FROM_GRID_KMS = 35.0

ESTIMATOR_NAMES = (
    "grid",
    "parabolic_3pt",
    "parabolic_ls",
    "gauss_offset",
    "smooth_peak",
    "bi_gauss",
    "auto",
)

LOWER_IS_BETTER_METRICS = frozenset(
    {
        "median_sigma_rv_kms",
        "p90_sigma_rv_kms",
        "median_chunk_scatter_kms",
        "bias_curve_rms_kms",
        "stellar_bias_cv_rmse_kms",
    }
)


@dataclass(frozen=True)
class CcfRvResult:
    rv_kms: float
    rv_err_kms: float
    estimator: str
    peak_snr: float
    v_grid_kms: float
    gauss_popt: tuple[float, ...] | None
    asymmetry: float
    fit_ok: bool


@dataclass
class CcfFitSlice:
    vel_shifts: np.ndarray
    ccf: np.ndarray
    peak_idx: int
    peak_val: float
    peak_snr: float
    v_grid: float
    v_lo: float
    v_hi: float
    x_m: np.ndarray
    y_m: np.ndarray
    scale_y: float


@dataclass(frozen=True)
class EstimatorConfig:
    fit_width: int = 50
    min_peak_snr: float = _DEFAULT_MIN_CCF_PEAK_SNR
    max_gauss_offset_kms: float = _DEFAULT_MAX_GAUSS_OFFSET_FROM_GRID_KMS
    ccf_neg_spike_sigma: float = 6.0
    low_snr_parabolic_threshold: float = 4.5
    high_asymmetry_bi_gauss_threshold: float = 0.22


def prepare_ccf_fit_slice(
    vel_shifts: np.ndarray,
    ccf: np.ndarray,
    *,
    peak_idx: int,
    peak_val: float,
    peak_snr: float,
    fit_width: int,
    ccf_neg_spike_sigma: float,
) -> CcfFitSlice | None:
    """Window around grid argmax with CR spike masking for sub-pixel fits."""
    vel_shifts = np.asarray(vel_shifts, float)
    ccf = np.asarray(ccf, float)
    v_grid = float(vel_shifts[peak_idx])
    v_lo = float(np.min(vel_shifts))
    v_hi = float(np.max(vel_shifts))

    sl = slice(max(0, peak_idx - fit_width), min(len(ccf), peak_idx + fit_width + 1))
    x_fit, y_fit = vel_shifts[sl], ccf[sl]
    if len(x_fit) < 5:
        return None

    y = np.asarray(y_fit, float)
    x_arr = np.asarray(x_fit, float)
    med_y = float(np.median(y))
    mad_y = float(np.median(np.abs(y - med_y))) + 1e-12
    scale_y = 1.4826 * mad_y
    neg_floor = med_y - float(ccf_neg_spike_sigma) * scale_y
    use = y >= neg_floor
    if int(np.sum(use)) < 5:
        use = np.ones_like(y, dtype=bool)
    x_m = x_arr[use]
    y_m = y[use]
    if len(x_m) < 5:
        return None

    span = float(max(np.ptp(y_m), 1e-9))
    sig0 = float(max(2.0, min(80.0, 0.5 * (v_hi - v_lo) * fit_width / max(len(ccf), 1))))
    rough = med_y + span * 0.5 * np.exp(-0.5 * ((x_arr - v_grid) / max(sig0, 2.0)) ** 2)
    use = use & (y >= (rough - 4.0 * scale_y))
    if int(np.sum(use)) < 5:
        use = np.ones_like(y, dtype=bool)
        x_m = x_arr[use]
        y_m = y[use]
    else:
        x_m = x_arr[use]
        y_m = y[use]

    return CcfFitSlice(
        vel_shifts=vel_shifts,
        ccf=ccf,
        peak_idx=peak_idx,
        peak_val=peak_val,
        peak_snr=peak_snr,
        v_grid=v_grid,
        v_lo=v_lo,
        v_hi=v_hi,
        x_m=np.asarray(x_m, float),
        y_m=np.asarray(y_m, float),
        scale_y=float(scale_y),
    )


def _ccf_asymmetry(slice_: CcfFitSlice) -> float:
    _, asym = qc.ccf_shape_metrics(slice_.vel_shifts, slice_.ccf)
    return float(asym) if np.isfinite(asym) else float("nan")


def _estimate_grid(slice_: CcfFitSlice) -> CcfRvResult:
    return CcfRvResult(
        rv_kms=slice_.v_grid,
        rv_err_kms=float("nan"),
        estimator="grid",
        peak_snr=slice_.peak_snr,
        v_grid_kms=slice_.v_grid,
        gauss_popt=None,
        asymmetry=_ccf_asymmetry(slice_),
        fit_ok=True,
    )


def _parabolic_vertex(_x: np.ndarray, a: float, b: float, _c: float) -> float:
    if abs(a) < 1e-15:
        return float("nan")
    return float(-b / (2.0 * a))


def _estimate_parabolic_3pt(slice_: CcfFitSlice) -> CcfRvResult:
    i = slice_.peak_idx
    ccf = slice_.ccf
    vel = slice_.vel_shifts
    if i <= 0 or i >= len(ccf) - 1:
        return _estimate_grid(slice_)
    x3 = vel[i - 1 : i + 2]
    y3 = ccf[i - 1 : i + 2]
    try:
        coef = np.polyfit(x3, y3, 2)
        mu = _parabolic_vertex(x3, *coef)
    except Exception:
        mu = float("nan")
    if not np.isfinite(mu) or mu < slice_.v_lo or mu > slice_.v_hi:
        return _estimate_grid(slice_)
    return CcfRvResult(
        rv_kms=mu,
        rv_err_kms=float("nan"),
        estimator="parabolic_3pt",
        peak_snr=slice_.peak_snr,
        v_grid_kms=slice_.v_grid,
        gauss_popt=None,
        asymmetry=_ccf_asymmetry(slice_),
        fit_ok=True,
    )


def _estimate_parabolic_ls(slice_: CcfFitSlice) -> CcfRvResult:
    x_m, y_m = slice_.x_m, slice_.y_m
    if len(x_m) < 5:
        return _estimate_grid(slice_)
    try:
        coef = np.polyfit(x_m, y_m, 2)
        mu = _parabolic_vertex(x_m, *coef)
    except Exception:
        mu = float("nan")
    if not np.isfinite(mu) or mu < slice_.v_lo or mu > slice_.v_hi:
        return _estimate_grid(slice_)
    return CcfRvResult(
        rv_kms=mu,
        rv_err_kms=float("nan"),
        estimator="parabolic_ls",
        peak_snr=slice_.peak_snr,
        v_grid_kms=slice_.v_grid,
        gauss_popt=None,
        asymmetry=_ccf_asymmetry(slice_),
        fit_ok=True,
    )


def _estimate_smooth_peak(slice_: CcfFitSlice, *, smooth_sigma_pts: float = 1.2) -> CcfRvResult:
    y = gaussian_filter1d(slice_.ccf.astype(float), sigma=smooth_sigma_pts)
    peak_i = int(np.argmax(y))
    if peak_i <= 0 or peak_i >= len(y) - 1:
        return _estimate_grid(slice_)
    x3 = slice_.vel_shifts[peak_i - 1 : peak_i + 2]
    y3 = y[peak_i - 1 : peak_i + 2]
    try:
        coef = np.polyfit(x3, y3, 2)
        mu = _parabolic_vertex(x3, *coef)
    except Exception:
        mu = float("nan")
    if not np.isfinite(mu) or mu < slice_.v_lo or mu > slice_.v_hi:
        return _estimate_grid(slice_)
    return CcfRvResult(
        rv_kms=mu,
        rv_err_kms=float("nan"),
        estimator="smooth_peak",
        peak_snr=slice_.peak_snr,
        v_grid_kms=slice_.v_grid,
        gauss_popt=None,
        asymmetry=_ccf_asymmetry(slice_),
        fit_ok=True,
    )


def _gauss_bounds_and_p0(slice_: CcfFitSlice) -> tuple[tuple, list]:
    x_m, y_m = slice_.x_m, slice_.y_m
    v_grid, v_lo, v_hi = slice_.v_grid, slice_.v_lo, slice_.v_hi
    span = float(max(np.ptp(y_m), 1e-9))
    c0_0 = float(np.median(y_m))
    amp0 = float(max(float(np.max(y_m)) - c0_0, span * 0.02, 1e-9))
    sig_est = float(max(2.0, min(80.0, 0.5 * (v_hi - v_lo) * len(x_m) / max(len(slice_.ccf), 1))))
    c_lo = float(np.min(y_m) - 0.75 * span)
    c_hi = float(np.max(y_m) + 0.25 * span)
    span_v = float(v_hi - v_lo)
    if span_v > 25.0:
        wing_sep = max(28.0, min(0.11 * span_v, 140.0))
        wing_mask = np.abs(slice_.vel_shifts - v_grid) >= wing_sep
        n_wing = int(np.sum(wing_mask))
        if n_wing >= 6:
            y_w = np.asarray(slice_.ccf[wing_mask], float)
            c0_wing = float(np.median(y_w))
            mad_w = float(np.median(np.abs(y_w - c0_wing)) * 1.4826) + 1e-12
            ptp_w = float(np.ptp(y_w))
            band = max(10.0 * mad_w, 0.025 * abs(slice_.peak_val - c0_wing), 0.05 * max(ptp_w, mad_w))
            band = max(band, 1e-9)
            c_lo_n = max(c_lo, c0_wing - band)
            c_hi_n = min(c_hi, c0_wing + band)
            if c_lo_n < c_hi_n - 1e-6:
                c_lo, c_hi = c_lo_n, c_hi_n
                c0_0 = float(np.clip(c0_wing, c_lo + 1e-9, c_hi - 1e-9))
                amp0 = float(max(float(np.max(y_m)) - c0_0, span * 0.02, 1e-9))
    sig_max = min(400.0, max(10.0, v_hi - v_lo))
    amp_hi = 10.0 * span + abs(c0_0)
    bounds = (
        [c_lo, 1e-12, v_lo, 0.5],
        [c_hi, amp_hi, v_hi, sig_max],
    )
    p0 = [c0_0, amp0, v_grid, sig_est]
    return bounds, p0


def _fit_soft_l1_gauss(
    x_m: np.ndarray,
    y_m: np.ndarray,
    bounds: tuple,
    p0: list,
    span: float,
) -> tuple[float, float, float, float] | None:
    lb = np.array(bounds[0], dtype=float)
    ub = np.array(bounds[1], dtype=float)
    p0a = np.clip(np.array(p0, dtype=float), lb + 1e-12, ub - 1e-12)
    f_scale = max(span * 0.12, 1e-9)

    def resid(p):
        c0p, amp, mu, sig = p
        return (c0p + amp * np.exp(-0.5 * ((x_m - mu) / sig) ** 2)) - y_m

    try:
        sol = least_squares(
            resid,
            p0a,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=f_scale,
            max_nfev=8000,
        )
        if not np.all(np.isfinite(sol.x)):
            return None
        return tuple(float(v) for v in sol.x)
    except Exception:
        return None


def _failed_gauss(slice_: CcfFitSlice) -> CcfRvResult:
    return CcfRvResult(
        rv_kms=slice_.v_grid,
        rv_err_kms=float("nan"),
        estimator="gauss_offset",
        peak_snr=slice_.peak_snr,
        v_grid_kms=slice_.v_grid,
        gauss_popt=None,
        asymmetry=_ccf_asymmetry(slice_),
        fit_ok=False,
    )


def _estimate_gauss_offset(slice_: CcfFitSlice, cfg: EstimatorConfig) -> CcfRvResult:
    x_m, y_m = slice_.x_m, slice_.y_m
    v_grid, v_lo, v_hi = slice_.v_grid, slice_.v_lo, slice_.v_hi
    span = float(max(np.ptp(y_m), 1e-9))

    def ccf_offset_gauss(x, c0p, amp, mu, sig):
        return c0p + amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    bounds, p0 = _gauss_bounds_and_p0(slice_)
    gauss_popt = None
    rv = v_grid
    rv_err = float("nan")

    try:
        popt, pcov = curve_fit(ccf_offset_gauss, x_m, y_m, p0=p0, bounds=bounds, maxfev=8000)
        c_fit, amp_fit, mu_fit, sig_fit = (float(v) for v in popt)
        if amp_fit < 1e-11:
            sl = _fit_soft_l1_gauss(x_m, y_m, bounds, p0, span)
            if sl is None:
                return _failed_gauss(slice_)
            c_fit, amp_fit, mu_fit, sig_fit = sl
            pcov = None
        if amp_fit < 1e-11:
            return _failed_gauss(slice_)
        if abs(mu_fit - v_grid) > cfg.max_gauss_offset_kms:
            return _failed_gauss(slice_)
        if mu_fit < v_lo - 1e-3 or mu_fit > v_hi + 1e-3:
            return _failed_gauss(slice_)
        if float(sig_fit) < float(config.MASK_CCF_MIN_GAUSS_SIGMA_KMS):
            return _failed_gauss(slice_)
        rv = mu_fit
        if pcov is not None:
            perr = np.sqrt(np.diag(pcov))
            rv_err = float(perr[2]) if np.isfinite(perr[2]) else float("nan")
        gauss_popt = (c_fit, amp_fit, mu_fit, sig_fit)
    except Exception:
        sl = _fit_soft_l1_gauss(x_m, y_m, bounds, p0, span)
        if sl is None:
            return _failed_gauss(slice_)
        c_fit, amp_fit, mu_fit, sig_fit = sl
        if amp_fit < 1e-11 or abs(mu_fit - v_grid) > cfg.max_gauss_offset_kms:
            return _failed_gauss(slice_)
        if mu_fit < v_lo - 1e-3 or mu_fit > v_hi + 1e-3:
            return _failed_gauss(slice_)
        if float(sig_fit) < float(config.MASK_CCF_MIN_GAUSS_SIGMA_KMS):
            return _failed_gauss(slice_)
        rv = mu_fit
        gauss_popt = (c_fit, amp_fit, mu_fit, sig_fit)

    return CcfRvResult(
        rv_kms=rv,
        rv_err_kms=rv_err,
        estimator="gauss_offset",
        peak_snr=slice_.peak_snr,
        v_grid_kms=slice_.v_grid,
        gauss_popt=gauss_popt,
        asymmetry=_ccf_asymmetry(slice_),
        fit_ok=True,
    )


def _estimate_bi_gauss(slice_: CcfFitSlice, cfg: EstimatorConfig) -> CcfRvResult:
    x_m, y_m = slice_.x_m, slice_.y_m
    v_grid, v_lo, v_hi = slice_.v_grid, slice_.v_lo, slice_.v_hi
    span = float(max(np.ptp(y_m), 1e-9))
    c0_0 = float(np.median(y_m))
    amp0 = float(max(float(np.max(y_m)) - c0_0, span * 0.02, 1e-9))
    sig0 = float(max(2.0, min(40.0, 0.25 * (v_hi - v_lo))))

    def bi_gauss(x, c0p, a1, mu1, s1, a2, mu2, s2):
        g1 = a1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
        g2 = a2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
        return c0p + g1 + g2

    c_lo = float(np.min(y_m) - span)
    c_hi = float(np.max(y_m) + 0.5 * span)
    sig_max = min(200.0, max(8.0, v_hi - v_lo))
    amp_hi = 8.0 * span + abs(c0_0)
    bounds = (
        [c_lo, 1e-12, v_lo, 0.5, 1e-12, v_lo, 0.5],
        [c_hi, amp_hi, v_hi, sig_max, amp_hi, v_hi, sig_max],
    )
    p0 = [c0_0, 0.65 * amp0, v_grid - 3.0, sig0, 0.35 * amp0, v_grid + 3.0, sig0 * 1.2]
    try:
        popt, pcov = curve_fit(bi_gauss, x_m, y_m, p0=p0, bounds=bounds, maxfev=12000)
        c0p, a1, mu1, s1, a2, mu2, s2 = (float(v) for v in popt)
        if a1 < 1e-11 and a2 < 1e-11:
            return _estimate_gauss_offset(slice_, cfg)
        if a1 >= a2:
            mu_fit, sig_fit = mu1, s1
            amp_fit = a1
            idx = 2
        else:
            mu_fit, sig_fit = mu2, s2
            amp_fit = a2
            idx = 5
        if abs(mu_fit - v_grid) > cfg.max_gauss_offset_kms * 1.5:
            return _estimate_gauss_offset(slice_, cfg)
        if float(sig_fit) < float(config.MASK_CCF_MIN_GAUSS_SIGMA_KMS):
            return _estimate_gauss_offset(slice_, cfg)
        rv_err = float("nan")
        if pcov is not None:
            perr = np.sqrt(np.diag(pcov))
            if np.isfinite(perr[idx]):
                rv_err = float(perr[idx])
        gauss_popt = (c0p, amp_fit, mu_fit, sig_fit)
        return CcfRvResult(
            rv_kms=mu_fit,
            rv_err_kms=rv_err,
            estimator="bi_gauss",
            peak_snr=slice_.peak_snr,
            v_grid_kms=slice_.v_grid,
            gauss_popt=gauss_popt,
            asymmetry=_ccf_asymmetry(slice_),
            fit_ok=True,
        )
    except Exception:
        return _estimate_gauss_offset(slice_, cfg)


_ESTIMATORS: dict[str, Callable[..., CcfRvResult]] = {
    "grid": lambda s, c: _estimate_grid(s),
    "parabolic_3pt": lambda s, c: _estimate_parabolic_3pt(s),
    "parabolic_ls": lambda s, c: _estimate_parabolic_ls(s),
    "gauss_offset": lambda s, c: _estimate_gauss_offset(s, c),
    "smooth_peak": lambda s, c: _estimate_smooth_peak(s),
    "bi_gauss": lambda s, c: _estimate_bi_gauss(s, c),
}


def select_ccf_estimator(
    peak_snr: float,
    asymmetry: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> str:
    """Route to parabolic (low S/N), bi-Gaussian (high asymmetry), else Gaussian."""
    cfg = cfg or EstimatorConfig()
    if not np.isfinite(peak_snr) or peak_snr < cfg.low_snr_parabolic_threshold:
        return "parabolic_3pt"
    if np.isfinite(asymmetry) and asymmetry >= cfg.high_asymmetry_bi_gauss_threshold and peak_snr >= 6.0:
        return "bi_gauss"
    return "gauss_offset"


def estimate_ccf_rv(
    estimator: str,
    slice_: CcfFitSlice,
    *,
    cfg: EstimatorConfig | None = None,
) -> CcfRvResult:
    cfg = cfg or EstimatorConfig()
    name = estimator
    if name == "auto":
        asym = _ccf_asymmetry(slice_)
        name = select_ccf_estimator(slice_.peak_snr, asym, cfg=cfg)
    fn = _ESTIMATORS.get(name)
    if fn is None:
        raise ValueError(f"unknown estimator: {estimator}")
    result = fn(slice_, cfg)
    if estimator in _ESTIMATORS and estimator != result.estimator:
        return CcfRvResult(
            rv_kms=result.rv_kms,
            rv_err_kms=result.rv_err_kms,
            estimator=estimator,
            peak_snr=result.peak_snr,
            v_grid_kms=result.v_grid_kms,
            gauss_popt=result.gauss_popt,
            asymmetry=result.asymmetry,
            fit_ok=result.fit_ok,
        )
    return result



@dataclass(frozen=True)
class BiGaussCcfResult:
    """Two-component CCF fit for SB2 scoring (primary + secondary)."""

    rv1_kms: float
    rv2_kms: float
    rv1_err_kms: float
    rv2_err_kms: float
    amp1: float
    amp2: float
    sigma1: float
    sigma2: float
    c0: float
    delta_chi2: float
    secondary_peak_snr: float
    asymmetry: float
    fit_ok: bool
    peak_snr: float
    v_grid_kms: float
    rv2_method: str = "bi_gauss"
    exclude_width_kms: float = float("nan")


def _failed_bi_gauss(
    *,
    peak_snr: float,
    v_grid_kms: float,
    asymmetry: float = float("nan"),
    rv2_method: str = "bi_gauss",
    exclude_width_kms: float = float("nan"),
) -> BiGaussCcfResult:
    nan = float("nan")
    return BiGaussCcfResult(
        rv1_kms=nan,
        rv2_kms=nan,
        rv1_err_kms=nan,
        rv2_err_kms=nan,
        amp1=nan,
        amp2=nan,
        sigma1=nan,
        sigma2=nan,
        c0=nan,
        delta_chi2=nan,
        secondary_peak_snr=nan,
        asymmetry=asymmetry,
        fit_ok=False,
        peak_snr=peak_snr,
        v_grid_kms=v_grid_kms,
        rv2_method=rv2_method,
        exclude_width_kms=exclude_width_kms,
    )


def _ccf_peak_metrics(vel_kms: np.ndarray, ccf: np.ndarray) -> tuple[int, float, float, float]:
    """Return peak_idx, peak_val, peak_snr, asymmetry on full arrays."""
    vel_kms = np.asarray(vel_kms, float)
    ccf = np.asarray(ccf, float)
    peak_idx = int(np.nanargmax(ccf))
    peak_val = float(ccf[peak_idx])
    c_med = float(np.nanmedian(ccf))
    c_mad = float(np.nanmedian(np.abs(ccf - c_med))) + 1e-12
    peak_snr = float((peak_val - c_med) / (1.4826 * c_mad))
    # Local asymmetry around peak for reporting
    half = min(40, peak_idx, len(ccf) - 1 - peak_idx)
    if half < 3:
        asym = float("nan")
    else:
        left = ccf[peak_idx - half : peak_idx]
        right = ccf[peak_idx + 1 : peak_idx + half + 1]
        n = min(len(left), len(right))
        if n < 3:
            asym = float("nan")
        else:
            lsum = float(np.nansum(left[-n:]))
            rsum = float(np.nansum(right[:n]))
            denom = abs(lsum) + abs(rsum) + 1e-12
            asym = abs(rsum - lsum) / denom
    return peak_idx, peak_val, peak_snr, asym


def _chi2(y: np.ndarray, model: np.ndarray) -> float:
    resid = np.asarray(y, float) - np.asarray(model, float)
    return float(np.nansum(resid * resid))


def estimate_ccf_bi_gauss_from_arrays(
    vel_kms: np.ndarray,
    ccf: np.ndarray,
    *,
    cfg: EstimatorConfig | None = None,
) -> BiGaussCcfResult:
    """
    Fit one- and two-Gaussian models on a CCF velocity grid.

    ``delta_chi2`` is ``chi2_1gauss - chi2_2gauss`` (positive => two-component preferred).
    Component 1 is the stronger amplitude; component 2 is the secondary.
    """
    cfg = cfg or EstimatorConfig()
    vel = np.asarray(vel_kms, float)
    y = np.asarray(ccf, float)
    if vel.size < 10 or y.size != vel.size or not np.any(np.isfinite(y)):
        return _failed_bi_gauss(peak_snr=float("nan"), v_grid_kms=float("nan"))

    peak_idx, peak_val, peak_snr, asym = _ccf_peak_metrics(vel, y)
    v_grid = float(vel[peak_idx])
    v_lo = float(np.nanmin(vel))
    v_hi = float(np.nanmax(vel))
    span = float(max(np.ptp(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0, 1e-9))
    c0_0 = float(np.nanmedian(y))
    amp0 = float(max(peak_val - c0_0, span * 0.02, 1e-9))
    sig0 = float(max(2.0, min(40.0, 0.08 * (v_hi - v_lo))))

    def one_gauss(x, c0p, a1, mu1, s1):
        return c0p + a1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)

    def bi_gauss(x, c0p, a1, mu1, s1, a2, mu2, s2):
        g1 = a1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
        g2 = a2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
        return c0p + g1 + g2

    c_lo = float(np.nanmin(y) - span)
    c_hi = float(np.nanmax(y) + 0.5 * span)
    sig_max = min(200.0, max(8.0, v_hi - v_lo))
    amp_hi = 8.0 * span + abs(c0_0)
    try:
        p1, _ = curve_fit(
            one_gauss,
            vel,
            y,
            p0=[c0_0, amp0, v_grid, sig0],
            bounds=(
                [c_lo, 1e-12, v_lo, 0.5],
                [c_hi, amp_hi, v_hi, sig_max],
            ),
            maxfev=12000,
        )
        chi1 = _chi2(y, one_gauss(vel, *p1))
    except Exception:
        return _failed_bi_gauss(peak_snr=peak_snr, v_grid_kms=v_grid, asymmetry=asym)

    # Seed secondary near residual max away from primary
    resid = y - one_gauss(vel, *p1)
    mask_pri = np.abs(vel - float(p1[2])) >= 8.0
    if np.any(mask_pri) and np.nanmax(resid[mask_pri]) > 0:
        i2 = int(np.nanargmax(np.where(mask_pri, resid, -np.inf)))
        mu2_0 = float(vel[i2])
        a2_0 = float(max(resid[i2], span * 0.05, 1e-9))
    else:
        mu2_0 = float(np.clip(v_grid + 12.0, v_lo + 1.0, v_hi - 1.0))
        a2_0 = 0.35 * amp0

    p0 = [c0_0, 0.7 * amp0, float(p1[2]), float(p1[3]), a2_0, mu2_0, sig0 * 1.1]
    bounds = (
        [c_lo, 1e-12, v_lo, 0.5, 1e-12, v_lo, 0.5],
        [c_hi, amp_hi, v_hi, sig_max, amp_hi, v_hi, sig_max],
    )
    try:
        p2, pcov = curve_fit(bi_gauss, vel, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return _failed_bi_gauss(peak_snr=peak_snr, v_grid_kms=v_grid, asymmetry=asym)

    c0p, a1, mu1, s1, a2, mu2, s2 = (float(v) for v in p2)
    chi2 = _chi2(y, bi_gauss(vel, *p2))
    dchi = float(chi1 - chi2)

    # Order by amplitude: primary = stronger
    if a2 > a1:
        a1, a2 = a2, a1
        mu1, mu2 = mu2, mu1
        s1, s2 = s2, s1
        idx1, idx2 = 5, 2
    else:
        idx1, idx2 = 2, 5

    e1 = e2 = float("nan")
    if pcov is not None:
        perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        if np.isfinite(perr[idx1]):
            e1 = float(perr[idx1])
        if np.isfinite(perr[idx2]):
            e2 = float(perr[idx2])

    # Secondary peak S/N vs continuum MAD
    c_med = float(np.nanmedian(y))
    c_mad = float(np.nanmedian(np.abs(y - c_med))) + 1e-12
    snr2 = float(a2 / (1.4826 * c_mad))

    ok = bool(a1 > 1e-11 and a2 > 1e-11 and np.isfinite(dchi))
    return BiGaussCcfResult(
        rv1_kms=mu1,
        rv2_kms=mu2,
        rv1_err_kms=e1,
        rv2_err_kms=e2,
        amp1=a1,
        amp2=a2,
        sigma1=s1,
        sigma2=s2,
        c0=c0p,
        delta_chi2=dchi,
        secondary_peak_snr=snr2,
        asymmetry=asym,
        fit_ok=ok,
        peak_snr=peak_snr,
        v_grid_kms=v_grid,
        rv2_method="bi_gauss",
        exclude_width_kms=float("nan"),
    )


def estimate_ccf_secondary_seeded(
    vel_kms: np.ndarray,
    ccf: np.ndarray,
    rv_primary_seed: float,
    *,
    cfg: EstimatorConfig | None = None,
    exclude_width_kms: float = 12.0,
    max_secondary_sep_kms: float = 50.0,
) -> BiGaussCcfResult:
    """
    Primary-seeded secondary search: blank the primary core, find residual peak,
    then fit a two-Gaussian model seeded at (primary, secondary).

    Used when free bi-Gaussian under-fits a weak secondary near a strong primary.
    """
    cfg = cfg or EstimatorConfig()
    vel = np.asarray(vel_kms, float)
    y = np.asarray(ccf, float)
    excl = float(exclude_width_kms)
    max_sep = float(max_secondary_sep_kms)
    peak_idx, peak_val, peak_snr, asym = _ccf_peak_metrics(vel, y)
    v_grid = float(vel[peak_idx])
    if not np.isfinite(rv_primary_seed) or vel.size < 10:
        return _failed_bi_gauss(
            peak_snr=peak_snr,
            v_grid_kms=v_grid,
            asymmetry=asym,
            rv2_method="seeded_peak",
            exclude_width_kms=excl,
        )

    mu1_0 = float(rv_primary_seed)
    # Residual after blanking primary core for secondary hunt
    search = np.ones(vel.shape, dtype=bool)
    search &= np.abs(vel - mu1_0) >= excl
    search &= np.abs(vel - mu1_0) <= max_sep
    if not np.any(search):
        return _failed_bi_gauss(
            peak_snr=peak_snr,
            v_grid_kms=v_grid,
            asymmetry=asym,
            rv2_method="seeded_peak",
            exclude_width_kms=excl,
        )

    # Continuum-subtracted CCF for peak pick
    c0_0 = float(np.nanmedian(y))
    y_sub = y - c0_0
    y_search = np.where(search, y_sub, -np.inf)
    i2 = int(np.nanargmax(y_search))
    if not np.isfinite(y_search[i2]) or y_search[i2] <= 0:
        return _failed_bi_gauss(
            peak_snr=peak_snr,
            v_grid_kms=v_grid,
            asymmetry=asym,
            rv2_method="seeded_peak",
            exclude_width_kms=excl,
        )
    mu2_0 = float(vel[i2])

    v_lo = float(np.nanmin(vel))
    v_hi = float(np.nanmax(vel))
    span = float(max(np.ptp(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0, 1e-9))
    amp0 = float(max(float(np.nanmax(y)) - c0_0, span * 0.02, 1e-9))
    a2_0 = float(max(y_sub[i2], span * 0.05, 1e-9))
    sig0 = float(max(2.0, min(40.0, 0.08 * (v_hi - v_lo))))
    c_lo = float(np.nanmin(y) - span)
    c_hi = float(np.nanmax(y) + 0.5 * span)
    sig_max = min(200.0, max(8.0, v_hi - v_lo))
    amp_hi = 8.0 * span + abs(c0_0)

    def one_gauss(x, c0p, a1, mu1, s1):
        return c0p + a1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)

    def bi_gauss(x, c0p, a1, mu1, s1, a2, mu2, s2):
        return (
            c0p
            + a1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
            + a2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
        )

    try:
        p1, _ = curve_fit(
            one_gauss,
            vel,
            y,
            p0=[c0_0, amp0, mu1_0, sig0],
            bounds=(
                [c_lo, 1e-12, v_lo, 0.5],
                [c_hi, amp_hi, v_hi, sig_max],
            ),
            maxfev=12000,
        )
        chi1 = _chi2(y, one_gauss(vel, *p1))
    except Exception:
        return _failed_bi_gauss(
            peak_snr=peak_snr,
            v_grid_kms=v_grid,
            asymmetry=asym,
            rv2_method="seeded_peak",
            exclude_width_kms=excl,
        )

    # Keep primary near seed; secondary near residual peak
    p0 = [c0_0, 0.75 * amp0, mu1_0, sig0, a2_0, mu2_0, sig0 * 1.1]
    # Soft bounds: primary within exclude_width of seed; secondary in sep window
    mu1_lo = max(v_lo, mu1_0 - excl)
    mu1_hi = min(v_hi, mu1_0 + excl)
    mu2_lo = max(v_lo, mu1_0 - max_sep)
    mu2_hi = min(v_hi, mu1_0 + max_sep)
    if mu1_lo >= mu1_hi - 1e-6 or mu2_lo >= mu2_hi - 1e-6:
        return _failed_bi_gauss(
            peak_snr=peak_snr,
            v_grid_kms=v_grid,
            asymmetry=asym,
            rv2_method="seeded_peak",
            exclude_width_kms=excl,
        )
    bounds = (
        [c_lo, 1e-12, mu1_lo, 0.5, 1e-12, mu2_lo, 0.5],
        [c_hi, amp_hi, mu1_hi, sig_max, amp_hi, mu2_hi, sig_max],
    )
    try:
        p2, pcov = curve_fit(bi_gauss, vel, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return _failed_bi_gauss(
            peak_snr=peak_snr,
            v_grid_kms=v_grid,
            asymmetry=asym,
            rv2_method="seeded_peak",
            exclude_width_kms=excl,
        )

    c0p, a1, mu1, s1, a2, mu2, s2 = (float(v) for v in p2)
    chi2 = _chi2(y, bi_gauss(vel, *p2))
    dchi = float(chi1 - chi2)

    e1 = e2 = float("nan")
    if pcov is not None:
        perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        if np.isfinite(perr[2]):
            e1 = float(perr[2])
        if np.isfinite(perr[5]):
            e2 = float(perr[5])

    c_med = float(np.nanmedian(y))
    c_mad = float(np.nanmedian(np.abs(y - c_med))) + 1e-12
    snr2 = float(a2 / (1.4826 * c_mad))
    ok = bool(a1 > 1e-11 and a2 > 1e-11 and np.isfinite(dchi) and abs(mu1 - mu2) >= 1.0)

    return BiGaussCcfResult(
        rv1_kms=mu1,
        rv2_kms=mu2,
        rv1_err_kms=e1,
        rv2_err_kms=e2,
        amp1=a1,
        amp2=a2,
        sigma1=s1,
        sigma2=s2,
        c0=c0p,
        delta_chi2=dchi,
        secondary_peak_snr=snr2,
        asymmetry=asym,
        fit_ok=ok,
        peak_snr=peak_snr,
        v_grid_kms=v_grid,
        rv2_method="seeded_peak",
        exclude_width_kms=excl,
    )


def estimate_all_ccf_rvs(
    slice_: CcfFitSlice,
    *,
    cfg: EstimatorConfig | None = None,
    estimators: tuple[str, ...] | None = None,
) -> dict[str, CcfRvResult]:
    cfg = cfg or EstimatorConfig()
    names = estimators or tuple(n for n in ESTIMATOR_NAMES if n != "auto")
    return {name: estimate_ccf_rv(name, slice_, cfg=cfg) for name in names}
