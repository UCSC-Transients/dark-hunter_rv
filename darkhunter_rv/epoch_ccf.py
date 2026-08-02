"""Epoch–epoch CCF relative RVs and WLS absolute fill.

Pair API measures ``Δv_ij = v_i - v_j`` via log-λ FFT cross-correlation of two
continuum-normalized spectra. The WLS combiner folds pairwise relatives with a
sparse absolute-RV vector: when ≥1 anchor exists, missing epochs get absolute
fills; with zero anchors, ``v_0`` is fixed at 0 and only relatives are reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from darkhunter_rv import config
from darkhunter_rv.ccf_rv_estimators import (
    EstimatorConfig,
    estimate_ccf_rv,
    prepare_ccf_fit_slice,
)


@dataclass(frozen=True)
class EpochPairCcfResult:
    """Result of ``epoch_pair_ccf`` for spectra ``i`` vs ``j``.

    ``dv_kms`` is ``v_i - v_j`` (positive ⇒ epoch ``i`` redshifted relative to ``j``).
    """

    dv_kms: float
    err_kms: float
    peak: float
    width_kms: float
    peak_snr: float
    fit_ok: bool
    qc: dict[str, float | bool | str]


@dataclass(frozen=True)
class EpochAbsFillResult:
    """WLS combination of relative matrix rows + optional absolute anchors."""

    v_hat_kms: np.ndarray
    sigma_kms: np.ndarray
    n_abs_anchors: int
    float_zeropoint: bool
    relative_only: bool
    cov: np.ndarray




def _common_log_grid(
    wave_i: np.ndarray,
    wave_j: np.ndarray,
    *,
    min_pixels: int = 256,
) -> np.ndarray | None:
    """Overlap log10-λ grid sized to next power of two (≥ ``min_pixels``)."""
    wi = np.asarray(wave_i, float)
    wj = np.asarray(wave_j, float)
    lo = max(float(np.nanmin(wi)), float(np.nanmin(wj)))
    hi = min(float(np.nanmax(wi)), float(np.nanmax(wj)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    n_est = int(min(len(wi), len(wj)))
    npts = max(2 ** int(np.ceil(np.log2(max(n_est, min_pixels)))), min_pixels)
    return np.linspace(np.log10(lo), np.log10(hi), npts)


def _resample_absorption(log_grid: np.ndarray, wave: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Interpolate continuum-normalized flux onto log grid; return absorption ``1 - f``."""
    w = np.asarray(wave, float)
    f = np.asarray(flux, float)
    fr = np.interp(log_grid, np.log10(w), f, left=np.nan, right=np.nan)
    return 1.0 - fr


def _fft_pair_ccf(
    abs_i: np.ndarray,
    abs_j: np.ndarray,
    log_grid: np.ndarray,
    *,
    rv_search_half_width_kms: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Log-λ FFT CCF of epoch ``i`` (obs) vs epoch ``j`` (reference).

    Positive lag ⇒ ``i`` redshifted relative to ``j`` (``Δv_ij = v_i - v_j``).
    """
    valid = np.isfinite(abs_i) & np.isfinite(abs_j)
    if int(np.sum(valid)) < 32:
        return None
    med_i = float(np.nanmedian(abs_i[valid]))
    med_j = float(np.nanmedian(abs_j[valid]))
    yi = np.where(np.isfinite(abs_i), abs_i, med_i)
    yj = np.where(np.isfinite(abs_j), abs_j, med_j)
    window = np.hanning(len(yi))
    zi = (yi - float(np.mean(yi))) / (float(np.std(yi)) + 1e-9)
    zj = (yj - float(np.mean(yj))) / (float(np.std(yj)) + 1e-9)
    ccf = np.fft.fftshift(np.fft.ifft(np.fft.fft(zi * window) * np.conj(np.fft.fft(zj * window))).real)
    delta_lnlam = (log_grid[1] - log_grid[0]) * np.log(10.0)
    dv_pix = float(config.C_KMS) * delta_lnlam
    npts = len(log_grid)
    vel_axis = (np.arange(npts) - npts // 2) * dv_pix
    half = float(rv_search_half_width_kms)
    mask = (vel_axis >= -half) & (vel_axis <= half)
    if int(np.sum(mask)) < 16:
        return None
    return np.asarray(vel_axis[mask], float), np.asarray(ccf[mask], float)


def _fit_ccf_peak(
    vel: np.ndarray,
    ccf: np.ndarray,
    *,
    fit_width: int = 40,
) -> tuple[float, float, float, float, float, bool]:
    """Return ``(dv, err, peak, width, peak_snr, fit_ok)`` from CCF curve."""
    peak_idx = int(np.argmax(ccf))
    peak_val = float(ccf[peak_idx])
    med = float(np.median(ccf))
    mad = float(np.median(np.abs(ccf - med))) + 1e-12
    peak_snr = float((peak_val - med) / (1.4826 * mad))
    sl = prepare_ccf_fit_slice(
        vel,
        ccf,
        peak_idx=peak_idx,
        peak_val=peak_val,
        peak_snr=peak_snr,
        fit_width=fit_width,
        ccf_neg_spike_sigma=6.0,
    )
    if sl is None:
        return float(vel[peak_idx]), float("nan"), peak_val, float("nan"), peak_snr, False

    cfg = EstimatorConfig(fit_width=fit_width, max_gauss_offset_kms=80.0)
    res = estimate_ccf_rv("gauss_offset", sl, cfg=cfg)
    width = float("nan")
    if res.gauss_popt is not None and len(res.gauss_popt) >= 4:
        width = float(res.gauss_popt[3])
    if not res.fit_ok or not np.isfinite(res.rv_kms):
        # Parabolic fallback on 3-pt around grid peak
        i = peak_idx
        if 0 < i < len(ccf) - 1:
            y0, y1, y2 = float(ccf[i - 1]), float(ccf[i]), float(ccf[i + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-15:
                dv_pix = 0.5 * (y0 - y2) / denom
                dv = float(vel[i] + dv_pix * (vel[1] - vel[0]))
            else:
                dv = float(vel[i])
        else:
            dv = float(vel[i])
        err = float(abs(vel[1] - vel[0])) if len(vel) > 1 else float("nan")
        return dv, err, peak_val, width, peak_snr, False

    err = float(res.rv_err_kms) if np.isfinite(res.rv_err_kms) else float("nan")
    if not np.isfinite(err) or err <= 0:
        # Floor from peak S/N and Gaussian width when formal cov fails
        if np.isfinite(width) and width > 0 and peak_snr > 1.0:
            err = float(width / max(peak_snr, 1.0))
        else:
            err = float(abs(vel[1] - vel[0])) if len(vel) > 1 else float("nan")
    return float(res.rv_kms), err, peak_val, width, peak_snr, True


def epoch_pair_ccf(
    wave_i: np.ndarray,
    flux_i: np.ndarray,
    wave_j: np.ndarray,
    flux_j: np.ndarray,
    *,
    rv_search_half_width_kms: float = 500.0,
    fit_width: int = 40,
) -> EpochPairCcfResult:
    """
    Cross-correlate continuum-normalized epoch ``i`` against epoch ``j``.

    Parameters
    ----------
    wave_i, flux_i
        Wavelength (Å) and continuum-normalized flux for epoch ``i``.
    wave_j, flux_j
        Same for epoch ``j`` (reference).
    rv_search_half_width_kms
        Half-width of the lag search window about 0.
    fit_width
        Half-window (samples) for Gaussian peak fit around the grid argmax.

    Returns
    -------
    EpochPairCcfResult
        ``dv_kms`` = ``v_i - v_j``; ``err_kms`` formal/floor uncertainty;
        ``peak`` / ``width_kms`` / ``peak_snr`` describe the CCF peak;
        ``qc`` carries flags (overlap, auto-corr hint, fit status).

    Limitations
    -----------
    Assumes both spectra share the same barycentric frame and continuum
    convention. Does not mask tellurics. Best on well-overlapped 1D products;
    per-order IVW stacking is out of spike scope.
    """
    log_grid = _common_log_grid(wave_i, wave_j)
    if log_grid is None:
        return EpochPairCcfResult(
            dv_kms=float("nan"),
            err_kms=float("nan"),
            peak=float("nan"),
            width_kms=float("nan"),
            peak_snr=float("nan"),
            fit_ok=False,
            qc={"ok": False, "reason": "no_overlap"},
        )
    abs_i = _resample_absorption(log_grid, wave_i, flux_i)
    abs_j = _resample_absorption(log_grid, wave_j, flux_j)
    pair = _fft_pair_ccf(
        abs_i,
        abs_j,
        log_grid,
        rv_search_half_width_kms=float(rv_search_half_width_kms),
    )
    if pair is None:
        return EpochPairCcfResult(
            dv_kms=float("nan"),
            err_kms=float("nan"),
            peak=float("nan"),
            width_kms=float("nan"),
            peak_snr=float("nan"),
            fit_ok=False,
            qc={"ok": False, "reason": "ccf_failed"},
        )
    vel, ccf = pair
    dv, err, peak, width, peak_snr, fit_ok = _fit_ccf_peak(vel, ccf, fit_width=fit_width)
    auto = bool(np.allclose(np.asarray(wave_i, float), np.asarray(wave_j, float)) and np.allclose(
        np.asarray(flux_i, float), np.asarray(flux_j, float)
    ))
    qc: dict[str, float | bool | str] = {
        "ok": bool(fit_ok and np.isfinite(dv)),
        "n_grid": float(len(log_grid)),
        "auto_correlation": auto,
        "estimator": "gauss_offset",
    }
    if auto and np.isfinite(dv):
        qc["diag_near_zero"] = bool(abs(dv) < max(3.0 * (err if np.isfinite(err) else 1.0), 1.0))
    return EpochPairCcfResult(
        dv_kms=float(dv),
        err_kms=float(err),
        peak=float(peak),
        width_kms=float(width),
        peak_snr=float(peak_snr),
        fit_ok=bool(fit_ok),
        qc=qc,
    )


def combine_relative_and_absolute(
    dv_ij: np.ndarray,
    sigma_ij: np.ndarray,
    abs_rv: np.ndarray,
    abs_sigma: np.ndarray,
    *,
    use_upper_triangle_only: bool = True,
) -> EpochAbsFillResult:
    """
    Weighted least-squares fill of epoch velocities from relatives + absolutes.

    Parameters
    ----------
    dv_ij, sigma_ij
        ``(N, N)`` matrices with ``dv_ij[i, j] ≈ v_i - v_j``. Diagonal ignored.
        When ``use_upper_triangle_only``, only ``i < j`` rows enter (antisymmetry
        assumed for the lower triangle).
    abs_rv, abs_sigma
        Length-``N`` absolute RVs; non-finite or non-positive ``abs_sigma`` ⇒ no
        anchor at that epoch.
    use_upper_triangle_only
        If True (default), use ``i < j`` pairs only.

    Returns
    -------
    EpochAbsFillResult
        ``v_hat_kms`` / ``sigma_kms`` for all epochs; ``float_zeropoint`` True when
        zero absolute anchors (``v_0`` fixed at 0, ``relative_only`` True).

    Limitations
    -----------
    Formal covariance only; no short-pair inflation. Assumes independent pair
    errors (overcounts information if both triangles supplied with
    ``use_upper_triangle_only=False``).
    """
    dv = np.asarray(dv_ij, float)
    sig = np.asarray(sigma_ij, float)
    a = np.asarray(abs_rv, float)
    sa = np.asarray(abs_sigma, float)
    if dv.ndim != 2 or dv.shape[0] != dv.shape[1]:
        raise ValueError("dv_ij must be square")
    n = int(dv.shape[0])
    if sig.shape != dv.shape:
        raise ValueError("sigma_ij shape must match dv_ij")
    if a.shape != (n,) or sa.shape != (n,):
        raise ValueError("abs_rv / abs_sigma must be length N")

    anchor = np.isfinite(a) & np.isfinite(sa) & (sa > 0)
    n_abs = int(np.sum(anchor))
    float_zp = n_abs == 0

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    wts: list[float] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if use_upper_triangle_only and i >= j:
                continue
            dij = float(dv[i, j])
            sij = float(sig[i, j])
            if not np.isfinite(dij) or not np.isfinite(sij) or sij <= 0:
                continue
            row = np.zeros(n, dtype=float)
            row[i] = 1.0
            row[j] = -1.0
            rows.append(row)
            rhs.append(dij)
            wts.append(1.0 / (sij * sij))

    for i in range(n):
        if not anchor[i]:
            continue
        row = np.zeros(n, dtype=float)
        row[i] = 1.0
        rows.append(row)
        rhs.append(float(a[i]))
        wts.append(1.0 / (float(sa[i]) ** 2))

    if not rows:
        v_hat = np.full(n, np.nan)
        if float_zp:
            v_hat[0] = 0.0
        return EpochAbsFillResult(
            v_hat_kms=v_hat,
            sigma_kms=np.full(n, np.nan),
            n_abs_anchors=n_abs,
            float_zeropoint=float_zp,
            relative_only=float_zp,
            cov=np.full((n, n), np.nan),
        )

    A = np.vstack(rows)
    b = np.asarray(rhs, float)
    w = np.asarray(wts, float)
    sw = np.sqrt(w)

    if float_zp:
        # Fix v_0 = 0: drop column 0; free parameters are v_1..v_{N-1}
        if n == 1:
            return EpochAbsFillResult(
                v_hat_kms=np.array([0.0]),
                sigma_kms=np.array([0.0]),
                n_abs_anchors=0,
                float_zeropoint=True,
                relative_only=True,
                cov=np.array([[0.0]]),
            )
        A_free = A[:, 1:]
        Aw = A_free * sw[:, None]
        bw = b * sw
        # Solve (A^T W A) x = A^T W b
        ata = Aw.T @ Aw
        atb = Aw.T @ bw
        try:
            x_free = np.linalg.solve(ata, atb)
            cov_free = np.linalg.inv(ata)
        except np.linalg.LinAlgError:
            x_free, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
            cov_free = np.full((n - 1, n - 1), np.nan)
        v_hat = np.zeros(n, dtype=float)
        v_hat[1:] = x_free
        cov = np.zeros((n, n), dtype=float)
        if np.all(np.isfinite(cov_free)):
            cov[1:, 1:] = cov_free
        sig_out = np.sqrt(np.maximum(np.diag(cov), 0.0))
        sig_out[0] = 0.0
        return EpochAbsFillResult(
            v_hat_kms=v_hat,
            sigma_kms=sig_out,
            n_abs_anchors=0,
            float_zeropoint=True,
            relative_only=True,
            cov=cov,
        )

    Aw = A * sw[:, None]
    bw = b * sw
    ata = Aw.T @ Aw
    atb = Aw.T @ bw
    try:
        x = np.linalg.solve(ata, atb)
        cov = np.linalg.inv(ata)
    except np.linalg.LinAlgError:
        x, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
        cov = np.full((n, n), np.nan)
    sig_out = np.sqrt(np.maximum(np.diag(cov), 0.0)) if np.all(np.isfinite(cov)) else np.full(n, np.nan)
    return EpochAbsFillResult(
        v_hat_kms=np.asarray(x, float),
        sigma_kms=np.asarray(sig_out, float),
        n_abs_anchors=n_abs,
        float_zeropoint=False,
        relative_only=False,
        cov=np.asarray(cov, float),
    )


def build_relative_matrix_from_pairs(
    n_epochs: int,
    pair_results: dict[tuple[int, int], EpochPairCcfResult],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble antisymmetric ``(dv, sigma)`` matrices from upper-triangle pair results.

    Missing pairs stay NaN. Diagonal set to 0 with tiny σ for QC bookkeeping.
    """
    dv = np.full((n_epochs, n_epochs), np.nan)
    sig = np.full((n_epochs, n_epochs), np.nan)
    np.fill_diagonal(dv, 0.0)
    np.fill_diagonal(sig, 1e-6)
    for (i, j), res in pair_results.items():
        if i == j:
            continue
        if not np.isfinite(res.dv_kms):
            continue
        e = float(res.err_kms) if np.isfinite(res.err_kms) and res.err_kms > 0 else float("nan")
        dv[i, j] = float(res.dv_kms)
        sig[i, j] = e
        dv[j, i] = -float(res.dv_kms)
        sig[j, i] = e
    return dv, sig
