"""Epoch–epoch relative RVs via production mask CCF with spectrum-as-mask.

Same chunking, continuum, ``cross_correlate_stellar_mask``, chunk bias (b0/b1/b2),
QC, and IVW as the stellar-mask lane — except the mask is the continuum-normalized
absorption of another epoch. Auto-correlation uses a Gaussian-smoothed copy of the
same epoch as the mask (obs stays unsmoothed) to avoid noise matching.

This is the production pair engine for ``validation.epoch_ccf_matrix`` (default
``engine=mask``). The legacy log-λ FFT path remains available as ``engine=fft``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from darkhunter_rv import chunking, config, continuum, io_utils, qc, rv_core
from darkhunter_rv.blaze import BlazeCalibration
from darkhunter_rv.epoch_ccf import EpochPairCcfResult
from darkhunter_rv.instruments import get_instrument_profile
from validation.chunk_layout import iter_order_chunks_from_layout, load_chunk_layout

logger = logging.getLogger(__name__)

DEFAULT_AUTO_SMOOTH_SIGMA = 3.0

# Match ``pipeline._exposure_stack_keep_mask`` (avoid importing heavy pipeline module).
_STACK_CLIP_MIN_ORDERS = 12
_STACK_CLIP_SIGMA = 3.2
_STACK_CLIP_MAXITERS = 2
_STACK_CLIP_MIN_KEEP_FRAC = 0.35
_STACK_CLIP_MIN_KEEP_ABS = 4


@dataclass(frozen=True)
class ChunkNorm:
    """Continuum-normalized chunk ready for mask CCF."""

    chunk_key: str
    order: int
    wave: np.ndarray
    flux_norm: np.ndarray
    eflux_norm: np.ndarray


@dataclass(frozen=True)
class PairResult:
    """Exposure-level relative RV and stacked CCF for obs vs ref."""

    epoch_x: int
    epoch_y: int
    dv_kms: float
    err_kms: float
    n_chunks: int
    auto_correlation: bool
    lag_sample_kms: float
    vel_stack: np.ndarray
    ccf_stack: np.ndarray
    peak_from_stack_kms: float
    n_chunks_raw: int = 0
    n_chunks_clipped: int = 0


def _exposure_stack_keep_mask(rv_arr: np.ndarray) -> np.ndarray:
    """Median + MAD sigma rejection; mirrors ``pipeline._exposure_stack_keep_mask``."""
    n = len(rv_arr)
    all_true = np.ones(n, dtype=bool)
    if n < _STACK_CLIP_MIN_ORDERS:
        return all_true
    keep = np.ones(n, dtype=bool)
    for _ in range(_STACK_CLIP_MAXITERS):
        m = float(np.median(rv_arr[keep]))
        mad = float(np.median(np.abs(rv_arr[keep] - m))) + 1e-12
        scale = 1.4826 * mad
        if scale < 1e-8:
            break
        new_keep = np.abs(rv_arr - m) < (_STACK_CLIP_SIGMA * scale)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    min_keep = max(_STACK_CLIP_MIN_KEEP_ABS, int(np.ceil(_STACK_CLIP_MIN_KEEP_FRAC * n)))
    if np.sum(keep) < min_keep:
        return all_true
    return keep


def spectrum_mask_from_norm(
    wave: np.ndarray,
    flux_norm: np.ndarray,
    *,
    smooth_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (mask_wave, mask_strength) from continuum-normalized flux.

    Parameters
    ----------
    wave, flux_norm
        Continuum-normalized spectrum (flux ~1 outside lines).
    smooth_sigma
        If set and ``> 0``, Gaussian-smooth ``flux_norm`` (pixels) before forming
        absorption strengths. Used for auto-correlation masks.

    Returns
    -------
    mask_wave, mask_strength
        ``strength = max(0, 1 - f)`` suitable for ``cross_correlate_stellar_mask``.
    """
    w = np.asarray(wave, float)
    f = np.asarray(flux_norm, float)
    if smooth_sigma is not None and float(smooth_sigma) > 0:
        f = gaussian_filter1d(f, sigma=float(smooth_sigma), mode="nearest")
    strength = np.maximum(0.0, 1.0 - f)
    return w, strength


def spectrum_mask_feature_count(
    strength: np.ndarray,
    *,
    min_depth: float = 0.05,
    min_sep_pix: int = 5,
) -> int:
    """
    Count absorption features in a dense spectrum-as-mask.

    Local maxima of ``strength`` above ``min_depth``, thinned to ``min_sep_pix``
    separation. Plays the same QC role as ``qc.mask_line_count_in_chunk`` for a
    sparse stellar mask: require enough real line cores in the chunk.
    """
    s = np.asarray(strength, float)
    if s.size < 3:
        return int(np.sum(np.isfinite(s) & (s >= float(min_depth))))
    core = s[1:-1]
    left = s[:-2]
    right = s[2:]
    peaks = (
        np.isfinite(core)
        & (core >= float(min_depth))
        & (core >= left)
        & (core >= right)
    )
    idx = np.where(peaks)[0] + 1
    if idx.size == 0:
        return 0
    sep = max(int(min_sep_pix), 1)
    kept = [int(idx[0])]
    for i in idx[1:]:
        if int(i) - kept[-1] >= sep:
            kept.append(int(i))
    return len(kept)


def _continuum_kw(
    order: int,
    *,
    continuum_mode: str,
    blaze_cal: BlazeCalibration | None = None,
) -> dict:
    """Mask-lane continuum kwargs (blaze model when mode needs it)."""
    mode = str(continuum_mode)
    kw: dict = {"continuum_mode": mode, "echelle_order": int(order)}
    if mode in ("sinc_blaze", "sinc_blaze_only"):
        if blaze_cal is not None:
            model = blaze_cal.model_for_order(int(order))
            if model is not None:
                kw["blaze_model"] = model
            else:
                kw["continuum_mode"] = "spline"
                mode = "spline"
        else:
            kw["continuum_mode"] = "spline"
            mode = "spline"
    if mode in ("spline", "sinc_blaze"):
        kw["exclude_near_lines_width"] = float(config.COOL_SPLINE_EXCLUDE_NEAR_LINES_WIDTH)
    return kw


def prepare_epoch_chunks(
    spectrum_path: Path,
    *,
    instrument_name: str = "APF",
    continuum_mode: str | None = None,
    blaze_cal: BlazeCalibration | None = None,
    chunk_layout: Path | None = None,
    subchunks: int = 8,
) -> list[ChunkNorm]:
    """
    Load spectrum and continuum-normalize every production chunk.

    Defaults to mask-lane continuum (``config.MASK_CONTINUUM_MODE`` + blaze) and
    ``config.DEFAULT_CHUNK_LAYOUT`` (``subchunks_8``).
    """
    instrument = get_instrument_profile(instrument_name)
    _hdr, spec_data = io_utils.read_spectrum(str(spectrum_path))
    mode = str(continuum_mode) if continuum_mode is not None else str(config.MASK_CONTINUUM_MODE)

    layout_path = chunk_layout if chunk_layout is not None else config.DEFAULT_CHUNK_LAYOUT
    if layout_path is not None and Path(layout_path).is_file():
        layout = load_chunk_layout(Path(layout_path))
        chunk_iter = iter_order_chunks_from_layout(spec_data, instrument.bad_orders, layout)
    else:
        chunk_iter = chunking.iter_order_chunks(spec_data, instrument.bad_orders, int(subchunks))

    out: list[ChunkNorm] = []
    for chunk_key, w, f, e in chunk_iter:
        ord_num, _, _ = chunking.parse_chunk_key(str(chunk_key))
        if ord_num is None:
            continue
        try:
            nw, nf, ne = continuum.fit_continuum(
                w,
                f,
                e,
                **_continuum_kw(int(ord_num), continuum_mode=mode, blaze_cal=blaze_cal),
            )
            nw, nf, ne = continuum.despike_normalized_pre_ccf(nw, nf, ne)
        except Exception as ex:
            logger.debug("continuum fail %s %s: %s", spectrum_path.name, chunk_key, ex)
            continue
        if len(nw) < 10:
            continue
        out.append(
            ChunkNorm(
                chunk_key=str(chunk_key),
                order=int(ord_num),
                wave=np.asarray(nw, float),
                flux_norm=np.asarray(nf, float),
                eflux_norm=np.asarray(ne, float),
            )
        )
    return out


def _mask_ccf_stack_error_inflated(
    rv: np.ndarray,
    er: np.ndarray,
    formal_combined: float,
    *,
    mu_weighted: float,
) -> float:
    """Match pipeline mask-stack error floor (MAD / weighted RMS / std)."""
    rv = np.asarray(rv, float).ravel()
    er = np.asarray(er, float).ravel()
    formal = float(formal_combined)
    n = int(rv.size)
    if n < 2 or er.size != n:
        return formal
    mu = float(mu_weighted)
    w = 1.0 / (er**2 + 1e-9)
    resid_rms = float(np.sqrt(np.average((rv - mu) ** 2, weights=w)))
    from_rms = resid_rms / np.sqrt(n)
    med = float(np.median(rv))
    mad = float(np.median(np.abs(rv - med))) * 1.4826
    from_mad = mad / np.sqrt(n)
    if n >= 3:
        from_std = float(np.std(rv, ddof=1) / np.sqrt(n))
    else:
        from_std = 0.5 * float(abs(rv[0] - rv[1]))
    return float(max(formal, from_rms, from_mad, from_std, 1e-9))


def _median_stack_ccfs(
    vel_list: list[np.ndarray],
    ccf_list: list[np.ndarray],
    *,
    n_grid: int = 801,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate chunk CCFs onto a common velocity grid; take nanmedian."""
    if not vel_list:
        return np.array([]), np.array([])
    v_lo = max(float(np.min(v)) for v in vel_list)
    v_hi = min(float(np.max(v)) for v in vel_list)
    if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi <= v_lo:
        v_lo = float(np.min([np.min(v) for v in vel_list]))
        v_hi = float(np.max([np.max(v) for v in vel_list]))
    vel_grid = np.linspace(v_lo, v_hi, int(n_grid))
    stack = []
    for vel, ccf in zip(vel_list, ccf_list):
        stack.append(np.interp(vel_grid, vel, ccf, left=np.nan, right=np.nan))
    mat = np.vstack(stack)
    with np.errstate(all="ignore"):
        med = np.nanmedian(mat, axis=0)
    ok = np.isfinite(med)
    if int(np.sum(ok)) < 10:
        return vel_grid, med
    fill = float(np.nanmedian(med[ok]))
    return vel_grid, np.where(ok, med, fill)


def pair_spectrum_mask_ccf(
    chunks_obs: list[ChunkNorm],
    chunks_ref: list[ChunkNorm],
    *,
    epoch_x: int,
    epoch_y: int,
    bias: dict,
    auto_smooth_sigma: float = DEFAULT_AUTO_SMOOTH_SIGMA,
    max_chunk_err_kms: float = 50.0,
    qc_thresholds: dict | None = None,
    min_chunks_for_stack: int | None = None,
    feature_min_depth: float = 0.05,
    feature_min_sep_pix: int = 5,
) -> PairResult:
    """
    Relative ``Δv = v_obs - v_ref`` via per-chunk mask CCF with spectrum-as-mask.

    Auto-correlation (``epoch_x == epoch_y``) smooths the ref flux before building
    mask strengths. Off-diagonal pairs use unsmoothed ref absorption.

    QC mirrors the mask lane (``evaluate_chunk_qc``) with ``mask_line_count``
    replaced by ``spectrum_mask_feature_count`` on the dense ref mask.
    """
    thresholds = dict(qc.DEFAULT_QC["global"])
    if qc_thresholds:
        thresholds.update(qc_thresholds)
    thresholds["max_chunk_err_kms"] = float(
        min(float(thresholds.get("max_chunk_err_kms", max_chunk_err_kms)), float(max_chunk_err_kms))
    )
    min_chunks = (
        int(config.MIN_MASK_CCF_CHUNKS_FOR_STACK)
        if min_chunks_for_stack is None
        else int(min_chunks_for_stack)
    )
    min_features = int(thresholds.get("min_mask_line_count", 8))

    ref_by_key = {c.chunk_key: c for c in chunks_ref}
    is_auto = int(epoch_x) == int(epoch_y)
    smooth = float(auto_smooth_sigma) if is_auto else None

    rvs: list[float] = []
    errs: list[float] = []
    vel_list: list[np.ndarray] = []
    ccf_list: list[np.ndarray] = []
    lag_samples: list[float] = []

    for obs in chunks_obs:
        ref = ref_by_key.get(obs.chunk_key)
        if ref is None:
            continue
        mw, ms = spectrum_mask_from_norm(ref.wave, ref.flux_norm, smooth_sigma=smooth)
        n_feat = spectrum_mask_feature_count(
            ms, min_depth=float(feature_min_depth), min_sep_pix=int(feature_min_sep_pix)
        )
        if n_feat < min_features:
            continue
        if obs.wave[-1] < mw[0] or obs.wave[0] > mw[-1]:
            continue

        line_obs = rv_core.mask_line_flux_in_excluded_wavelengths(obs.wave, 1.0 - obs.flux_norm)
        rv_m, err_m, vels_m, ccf_m, peak_m, _gauss_p, peak_snr_m = rv_core.cross_correlate_stellar_mask(
            obs.wave, line_obs, mw, ms
        )
        if vels_m is None or ccf_m is None:
            continue

        bvec = io_utils.lookup_bias(bias, obs.chunk_key)
        if isinstance(bvec, (list, tuple)) and len(bvec) >= 3:
            b0, b1, b2 = float(bvec[0]), float(bvec[1]), float(bvec[2])
        else:
            b0, b1, b2 = 0.0, 0.0, 0.0

        rv_corr = float(rv_m) - b0 if np.isfinite(rv_m) else float("nan")
        err_corr = (
            float(np.sqrt(float(err_m) ** 2 + b1**2 + b2**2)) if np.isfinite(err_m) else float("nan")
        )
        vel_corr = np.asarray(vels_m, float) - b0

        tell_frac = qc.telluric_fraction(obs.wave)
        _w_ccf, asym_ccf = qc.ccf_shape_metrics(vels_m, ccf_m)
        qc_ok, _reason = qc.evaluate_chunk_qc(
            {
                "rv_err_kms": err_corr,
                "mask_line_count": n_feat,
                "telluric_fraction": tell_frac,
                "ccf_asymmetry": asym_ccf,
                "ccf_peak": peak_m,
                "ccf_peak_snr": peak_snr_m,
            },
            thresholds,
        )
        if not qc_ok:
            continue
        if not (np.isfinite(rv_corr) and np.isfinite(err_corr) and err_corr > 0):
            continue
        if err_corr > float(thresholds["max_chunk_err_kms"]):
            continue

        rvs.append(rv_corr)
        errs.append(err_corr)
        vel_list.append(vel_corr)
        ccf_list.append(np.asarray(ccf_m, float))
        if len(vel_corr) > 1:
            lag_samples.append(float(np.median(np.diff(vel_corr))))

    n_raw = len(rvs)
    if n_raw == 0:
        empty = np.array([])
        return PairResult(
            epoch_x=int(epoch_x),
            epoch_y=int(epoch_y),
            dv_kms=float("nan"),
            err_kms=float("nan"),
            n_chunks=0,
            auto_correlation=is_auto,
            lag_sample_kms=float("nan"),
            vel_stack=empty,
            ccf_stack=empty,
            peak_from_stack_kms=float("nan"),
            n_chunks_raw=0,
            n_chunks_clipped=0,
        )

    rv_a = np.asarray(rvs, float)
    er_a = np.asarray(errs, float)
    keep = _exposure_stack_keep_mask(rv_a)
    n_clip = int(n_raw - int(np.sum(keep)))
    rv_a = rv_a[keep]
    er_a = er_a[keep]
    vel_kept = [v for v, k in zip(vel_list, keep) if k]
    ccf_kept = [c for c, k in zip(ccf_list, keep) if k]

    vel_stack, ccf_stack = _median_stack_ccfs(vel_kept, ccf_kept)
    peak_stack = float("nan")
    if len(ccf_stack) and np.any(np.isfinite(ccf_stack)):
        peak_stack = float(vel_stack[int(np.nanargmax(ccf_stack))])

    lag_med = float(np.nanmedian(lag_samples)) if lag_samples else float("nan")
    if len(rv_a) < min_chunks:
        return PairResult(
            epoch_x=int(epoch_x),
            epoch_y=int(epoch_y),
            dv_kms=float("nan"),
            err_kms=float("nan"),
            n_chunks=len(rv_a),
            auto_correlation=is_auto,
            lag_sample_kms=lag_med,
            vel_stack=vel_stack,
            ccf_stack=ccf_stack,
            peak_from_stack_kms=peak_stack,
            n_chunks_raw=n_raw,
            n_chunks_clipped=n_clip,
        )

    w = 1.0 / (er_a**2 + 1e-18)
    mu = float(np.sum(w * rv_a) / np.sum(w))
    sig = float(np.sqrt(1.0 / np.sum(w)))
    sig = _mask_ccf_stack_error_inflated(rv_a, er_a, sig, mu_weighted=mu)

    return PairResult(
        epoch_x=int(epoch_x),
        epoch_y=int(epoch_y),
        dv_kms=mu,
        err_kms=sig,
        n_chunks=len(rv_a),
        auto_correlation=is_auto,
        lag_sample_kms=lag_med,
        vel_stack=vel_stack,
        ccf_stack=ccf_stack,
        peak_from_stack_kms=peak_stack,
        n_chunks_raw=n_raw,
        n_chunks_clipped=n_clip,
    )


def pair_result_to_epoch_pair(res: PairResult) -> EpochPairCcfResult:
    """Convert spectrum-as-mask ``PairResult`` to the WLS matrix ``EpochPairCcfResult``."""
    peak = float("nan")
    if len(res.ccf_stack) and np.any(np.isfinite(res.ccf_stack)):
        peak = float(np.nanmax(res.ccf_stack))
    ok = bool(np.isfinite(res.dv_kms) and np.isfinite(res.err_kms) and res.err_kms > 0)
    qc_dict: dict[str, float | bool | str] = {
        "ok": ok,
        "auto_correlation": bool(res.auto_correlation),
        "n_chunks": float(res.n_chunks),
        "n_chunks_raw": float(res.n_chunks_raw),
        "n_chunks_clipped": float(res.n_chunks_clipped),
        "lag_sample_kms": float(res.lag_sample_kms) if np.isfinite(res.lag_sample_kms) else float("nan"),
        "peak_from_stack_kms": (
            float(res.peak_from_stack_kms) if np.isfinite(res.peak_from_stack_kms) else float("nan")
        ),
        "engine": "mask",
    }
    if res.auto_correlation and np.isfinite(res.dv_kms):
        qc_dict["diag_near_zero"] = bool(
            abs(res.dv_kms) < max(3.0 * (res.err_kms if np.isfinite(res.err_kms) else 1.0), 1.0)
        )
    return EpochPairCcfResult(
        dv_kms=float(res.dv_kms) if np.isfinite(res.dv_kms) else float("nan"),
        err_kms=float(res.err_kms) if np.isfinite(res.err_kms) else float("nan"),
        peak=peak,
        width_kms=float("nan"),
        peak_snr=float("nan"),
        fit_ok=ok,
        qc=qc_dict,
    )


def compute_mask_pair_matrix(
    prepared: list[list[ChunkNorm]],
    epoch_indices: list[int],
    *,
    bias: dict,
    auto_smooth_sigma: float = DEFAULT_AUTO_SMOOTH_SIGMA,
    max_chunk_err_kms: float = 50.0,
    qc_thresholds: dict | None = None,
) -> tuple[dict[tuple[int, int], EpochPairCcfResult], list[dict], list[PairResult]]:
    """
    Compute all pairs with ``i <= j`` (includes auto-correlation diagonal).

    Lower triangle long-form rows are filled via antisymmetry for CSV consumers;
    the WLS assembler uses upper-triangle ``EpochPairCcfResult`` entries only.
    """
    n = len(prepared)
    if n != len(epoch_indices):
        raise ValueError("prepared and epoch_indices length mismatch")
    pairs: dict[tuple[int, int], EpochPairCcfResult] = {}
    rows: list[dict] = []
    detailed: list[PairResult] = []
    for i in range(n):
        for j in range(i, n):
            res = pair_spectrum_mask_ccf(
                prepared[i],
                prepared[j],
                epoch_x=int(epoch_indices[i]),
                epoch_y=int(epoch_indices[j]),
                bias=bias,
                auto_smooth_sigma=float(auto_smooth_sigma),
                max_chunk_err_kms=float(max_chunk_err_kms),
                qc_thresholds=qc_thresholds,
            )
            detailed.append(res)
            epr = pair_result_to_epoch_pair(res)
            pairs[(i, j)] = epr
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "dv_kms": epr.dv_kms,
                    "err_kms": epr.err_kms,
                    "peak": epr.peak,
                    "width_kms": epr.width_kms,
                    "peak_snr": epr.peak_snr,
                    "fit_ok": epr.fit_ok,
                    "auto_correlation": bool(epr.qc.get("auto_correlation", False)),
                    "qc_ok": bool(epr.qc.get("ok", False)),
                    "n_chunks": res.n_chunks,
                    "lag_sample_kms": res.lag_sample_kms,
                }
            )
            if i != j:
                rows.append(
                    {
                        "i": j,
                        "j": i,
                        "dv_kms": -epr.dv_kms if np.isfinite(epr.dv_kms) else np.nan,
                        "err_kms": epr.err_kms,
                        "peak": epr.peak,
                        "width_kms": epr.width_kms,
                        "peak_snr": epr.peak_snr,
                        "fit_ok": epr.fit_ok,
                        "auto_correlation": False,
                        "qc_ok": bool(epr.qc.get("ok", False)),
                        "n_chunks": res.n_chunks,
                        "lag_sample_kms": res.lag_sample_kms,
                    }
                )
    return pairs, rows, detailed
