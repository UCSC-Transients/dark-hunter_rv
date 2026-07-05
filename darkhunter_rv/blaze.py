"""
Parameterized echelle blaze profile for a single order (Hβ lane first).

Model: amplitude × [sinc(π (λ − λ₀) / w) / (π (λ − λ₀) / w)]^p  with sinc(0) = 1.

Fit on many weak-line spectra: per-star amplitude, shared (λ₀, w, p). Pixels near strong
lines are masked before stacking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

from darkhunter_rv import config, continuum, io_utils, rv_core

logger = logging.getLogger(__name__)

HB_REST_A = float(rv_core.HB_REST_A)

BLAZE_ITERATIVE_THRESHOLDS = tuple(
    getattr(
        config,
        "BLAZE_ITERATIVE_THRESHOLDS",
        (0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98),
    )
)
BLAZE_ITERATIVE_NSIGMA = float(getattr(config, "BLAZE_ITERATIVE_NSIGMA", 3.5))


FIT_MODEL_SINC2 = "sinc2"
FIT_MODEL_SINC_POLY = "sinc_poly"


@dataclass(frozen=True)
class IterativeBlazeFitResult:
    center_angstrom: float
    width_angstrom: float
    power: float
    amplitude: float
    continuum_mask: np.ndarray
    stage_counts: tuple[tuple[float, int], ...]
    mask_expand_pixels: int = 0
    fit_model: str = FIT_MODEL_SINC2
    poly_coef: tuple[float, ...] | None = None
    poly_w_center: float = 0.0
    poly_w_scale: float = 1.0


@dataclass(frozen=True)
class SharedBlazeOrderResult:
    """Shared blaze shape with per-spectrum scale (optional log-poly multiplier)."""

    envelope: np.ndarray
    continuum_mask: np.ndarray
    scale: float
    fit_model: str = FIT_MODEL_SINC2
    poly_coef: tuple[float, ...] | None = None
    poly_w_center: float = 0.0
    poly_w_scale: float = 1.0


def _sinc_pi(x: np.ndarray) -> np.ndarray:
    y = np.ones_like(x)
    m = np.abs(x) > 1e-12
    y[m] = np.sin(np.pi * x[m]) / (np.pi * x[m])
    return y


def eval_blaze_sinc2(
    wavelength: np.ndarray,
    center: float,
    width: float,
    *,
    power: float = 2.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Parameterized blaze: amplitude × |sinc(πx)|^power via (sinc²)^(power/2), x=(λ−center)/width."""
    w = np.asarray(wavelength, float)
    x = (w - float(center)) / max(float(width), 1e-9)
    sinc2 = _sinc_pi(x) ** 2
    return float(amplitude) * (sinc2 ** (float(power) / 2.0))


def eval_poly_multiplier(
    wavelength: np.ndarray,
    coef: tuple[float, ...] | list[float],
    w_center: float,
    w_scale: float,
) -> np.ndarray:
    """Positive polynomial multiplier: exp(log-poly) in centered/scaled wavelength."""
    w = np.asarray(wavelength, float)
    scale = max(float(w_scale), 1e-9)
    x = (w - float(w_center)) / scale
    log_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        log_poly += float(c) * (x**i)
    return np.exp(log_poly)


def _poly_w_center_scale(grid: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    g = np.asarray(grid, float)[np.asarray(mask, bool)]
    if g.size == 0:
        return 0.0, 1.0
    w_center = float(np.median(g))
    w_scale = float(np.std(g))
    if not np.isfinite(w_scale) or w_scale < 1e-6:
        w_scale = max(float(np.max(g) - np.min(g)) / 2.0, 1.0)
    return w_center, w_scale


def _fit_poly_multiplier_on_mask(
    grid: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
    *,
    center: float,
    width: float,
    power: float,
    amplitude: float,
    poly_order: int,
) -> tuple[tuple[float, ...], float, float] | None:
    """Linear LSQ fit of log(flux/sinc²) vs centered polynomial on ``mask`` pixels."""
    g = np.asarray(grid, float)
    f = np.asarray(flux, float)
    m = np.asarray(mask, bool) & np.isfinite(f) & (f > 0)
    order = int(poly_order)
    if order < 1 or int(np.sum(m)) < order + 2:
        return None
    sinc = eval_blaze_sinc2(g[m], center, width, power=power, amplitude=amplitude)
    ratio = f[m] / sinc
    pos = ratio > 0
    if int(np.sum(pos)) < order + 2:
        return None
    log_ratio = np.log(ratio[pos])
    w_center, w_scale = _poly_w_center_scale(g[m], pos)
    x = (g[m][pos] - w_center) / w_scale
    design = np.vander(x, order + 1, increasing=True)
    coef, _, _, _ = np.linalg.lstsq(design, log_ratio, rcond=None)
    return tuple(float(c) for c in coef), w_center, w_scale


def eval_fit_envelope(
    result: IterativeBlazeFitResult,
    wavelength: np.ndarray,
) -> np.ndarray:
    """Combined sinc²×poly envelope from an iterative fit result."""
    w = np.asarray(wavelength, float)
    sinc = eval_blaze_sinc2(
        w,
        result.center_angstrom,
        result.width_angstrom,
        power=result.power,
        amplitude=result.amplitude,
    )
    if result.fit_model == FIT_MODEL_SINC_POLY and result.poly_coef is not None:
        poly = eval_poly_multiplier(
            w,
            result.poly_coef,
            result.poly_w_center,
            result.poly_w_scale,
        )
        return sinc * poly
    return sinc


def blaze_line_mask(
    wavelength: np.ndarray,
    *,
    rest_lines: list[float] | None = None,
    half_width_angstrom: float = 22.0,
) -> np.ndarray:
    """True where pixel is usable for blaze fitting (far from catalogued strong lines)."""
    rests = rest_lines if rest_lines is not None else list(continuum.STRONG_LINES)
    w = np.asarray(wavelength, float)
    ok = np.isfinite(w)
    hw = float(half_width_angstrom)
    for rest in rests:
        ok &= ~((w >= float(rest) - hw) & (w <= float(rest) + hw))
    return ok


def load_blaze_stellar_mask(
    mask_name: str | None = None,
    mask_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load rest-frame stellar mask used to exclude line cores from blaze fits."""
    name = mask_name or str(getattr(config, "BLAZE_STELLAR_MASK_NAME", "G8_espresso"))
    mdir = Path(mask_dir or config.MASK_DIRECTORY)
    path = mdir / f"{name}.txt"
    if not path.is_file():
        logger.warning("blaze stellar mask not found: %s", path)
        return None
    md = np.loadtxt(path)
    return np.asarray(md[:, 0], float), np.asarray(md[:, 1], float)


def blaze_stellar_mask(
    wavelength: np.ndarray,
    mask_wave: np.ndarray,
    mask_strength: np.ndarray,
    *,
    half_width_angstrom: float | None = None,
    min_strength: float | None = None,
    max_lines_per_span: int | None = None,
) -> np.ndarray:
    """
    True where pixel is usable (not within half-width of stellar-mask lines).

    Dense masks (e.g. G8) can cover an entire order if every line is excluded;
    ``max_lines_per_span`` keeps only the strongest lines in the order window.
    """
    w = np.asarray(wavelength, float)
    mw = np.asarray(mask_wave, float)
    ms = np.asarray(mask_strength, float)
    ok = np.isfinite(w)
    if mw.size == 0 or ms.shape != mw.shape:
        return ok
    hw = float(
        half_width_angstrom
        if half_width_angstrom is not None
        else getattr(config, "BLAZE_STELLAR_MASK_HALF_WIDTH_A", 3.0)
    )
    thr = float(
        min_strength
        if min_strength is not None
        else getattr(config, "BLAZE_STELLAR_MASK_MIN_STRENGTH", 0.15)
    )
    max_lines = max_lines_per_span
    if max_lines is None:
        max_lines = getattr(config, "BLAZE_STELLAR_MASK_MAX_LINES_PER_SPAN", 35)
    wmn, wmx = float(np.min(w)), float(np.max(w))
    in_span = (mw >= wmn - hw) & (mw <= wmx + hw) & (ms >= thr)
    line_w = mw[in_span]
    line_s = ms[in_span]
    if line_w.size == 0:
        return ok
    if max_lines is not None and int(max_lines) > 0 and line_w.size > int(max_lines):
        order = np.argsort(-line_s)[: int(max_lines)]
        line_w = line_w[order]
    for rest in line_w:
        lam = float(rest)
        ok &= ~((w >= lam - hw) & (w <= lam + hw))
    return ok


def _uniform_order_grid(profiles: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Uniform Å grid for stacking profiles (one pixel per native sampling step).

    Avoids ``np.unique(concatenate(w))`` which inflates pixel count when epochs have
  slightly different wavelength solutions and breaks local absorption masking.
    """
    all_w = np.concatenate([np.asarray(w, float) for w, _f in profiles])
    wmn, wmx = float(np.min(all_w)), float(np.max(all_w))
    ref_w = np.asarray(profiles[0][0], float)
    step = float(np.median(np.diff(np.sort(ref_w))))
    if not np.isfinite(step) or step <= 0:
        step = max((wmx - wmn) / 120.0, 0.05)
    n_pix = max(40, int(np.ceil((wmx - wmn) / step)) + 1)
    return np.linspace(wmn, wmx, n_pix)


def _fit_sinc2_on_mask(
    grid: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Fit sinc² to ``flux`` on pixels where ``mask`` is True."""
    g = np.asarray(grid, float)
    f = np.asarray(flux, float)
    m = np.asarray(mask, bool) & np.isfinite(f) & (f > 0)
    if int(np.sum(m)) < 8:
        return None
    wg, yg = g[m], f[m]
    i_peak = int(np.argmax(yg))
    center0 = float(wg[i_peak])
    half = 0.5 * float(yg[i_peak])
    above = wg[yg >= half]
    width0 = float(0.5 * (np.max(above) - np.min(above))) if above.size > 4 else 80.0
    width0 = max(width0, 15.0)

    def model(lam, center, width, power, amp):
        return eval_blaze_sinc2(lam, center, width, power=power, amplitude=amp)

    try:
        popt, _pcov = curve_fit(
            model,
            wg,
            yg,
            p0=[center0, width0, 2.0, float(np.max(yg))],
            bounds=(
                [float(np.min(wg)), 5.0, 2.0, 0.01 * float(np.max(yg))],
                [float(np.max(wg)), 500.0, 4.5, 100.0 * float(np.max(yg))],
            ),
            maxfev=20000,
        )
    except Exception as ex:
        logger.warning("blaze curve_fit failed: %s", ex)
        return None
    return tuple(map(float, popt))


def _blaze_normalized_flux(
    grid: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
    *,
    center: float,
    width: float,
    power: float,
    amplitude: float,
    poly_coef: tuple[float, ...] | None = None,
    poly_w_center: float = 0.0,
    poly_w_scale: float = 1.0,
) -> np.ndarray:
    """``flux / (sinc² [× poly])`` median-scaled on ``mask`` pixels."""
    blaze = eval_blaze_sinc2(
        grid,
        center,
        width,
        power=power,
        amplitude=amplitude,
    )
    if poly_coef is not None:
        poly = eval_poly_multiplier(grid, poly_coef, poly_w_center, poly_w_scale)
        blaze = blaze * poly
    norm = np.asarray(flux, float) / blaze
    ok = np.asarray(mask, bool) & np.isfinite(norm) & (norm > 0)
    level = float(np.nanmedian(norm[ok])) if np.any(ok) else 1.0
    if not np.isfinite(level) or level <= 0:
        level = 1.0
    return norm / level


def _continuum_snr_per_pixel(
    grid: np.ndarray,
    norm: np.ndarray,
    mask: np.ndarray,
    *,
    window_angstrom: float | None = None,
) -> np.ndarray:
    """
    Continuum S/N per pixel from residual scatter of blaze-normalized flux about 1.0.

    Local sigma is estimated in a rolling window over continuum-mask pixels, then
    interpolated across the full wavelength grid (including line cores).
    """
    g = np.asarray(grid, float)
    n = np.asarray(norm, float)
    m = np.asarray(mask, bool) & np.isfinite(n)
    if window_angstrom is None:
        span = float(np.max(g) - np.min(g)) if g.size else 80.0
        window_angstrom = max(span / 12.0, 3.0)
    hw = 0.5 * float(window_angstrom)
    r = n - 1.0

    cont_g = g[m]
    cont_r = r[m]
    if cont_g.size < 3:
        sigma_glob = float(np.nanstd(cont_r)) if cont_r.size > 1 else 0.01
        sigma_glob = max(sigma_glob, 1e-6)
        return np.full(g.shape, 1.0 / sigma_glob, dtype=float)

    local_sigma = np.empty(cont_g.shape, dtype=float)
    for i, wi in enumerate(cont_g):
        in_win = (cont_g >= wi - hw) & (cont_g <= wi + hw)
        local_sigma[i] = float(np.nanstd(cont_r[in_win]))
    local_sigma = np.maximum(local_sigma, 1e-6)

    order = np.argsort(cont_g)
    wg = cont_g[order]
    sg = local_sigma[order]
    sigma_all = np.interp(g, wg, sg, left=float(sg[0]), right=float(sg[-1]))
    sigma_all = np.maximum(sigma_all, 1e-6)
    snr = 1.0 / sigma_all
    return np.where(np.isfinite(snr) & (snr > 0), snr, 1.0)


def _expand_excluded_mask(
    mask: np.ndarray,
    n_pix: int,
    *,
    bounds: np.ndarray | None = None,
) -> np.ndarray:
    """Grow excluded (False) regions by ``n_pix`` pixels on each side along the 1D grid."""
    m = np.asarray(mask, bool).copy()
    if n_pix <= 0:
        if bounds is not None:
            m &= np.asarray(bounds, bool)
        return m
    excluded = ~m
    for _ in range(int(n_pix)):
        grow = excluded.copy()
        if grow.size > 1:
            grow[1:] |= excluded[:-1]
            grow[:-1] |= excluded[1:]
        excluded = grow
    out = ~excluded
    if bounds is not None:
        out &= np.asarray(bounds, bool)
    return out


def _pull_distribution_consistency_score(pulls: np.ndarray) -> float:
    """
    Score asymmetry of continuum pulls (norm - 1): lower means <1 and >1 tails are consistent.

    Combines count balance, median magnitude, 95th percentile, and a two-sample KS statistic
    between |negative pulls| and positive pulls.
    """
    p = np.asarray(pulls, float)
    p = p[np.isfinite(p)]
    if p.size < 6:
        return 0.0
    below = p[p < 0.0]
    above = p[p > 0.0]
    if below.size < 2 or above.size < 2:
        return 1.0 + abs(below.size - above.size) / max(p.size, 1)

    count_pen = abs(np.log((below.size + 0.5) / (above.size + 0.5)))
    med_below = float(np.median(np.abs(below)))
    med_above = float(np.median(above))
    scale = max(med_below, med_above, 1e-6)
    med_pen = abs(med_below - med_above) / scale
    p95_below = float(np.percentile(np.abs(below), 95))
    p95_above = float(np.percentile(above, 95))
    p95_pen = abs(p95_below - p95_above) / max(p95_below, p95_above, 1e-6)
    ks_stat = float(ks_2samp(np.abs(below), above).statistic)
    return 0.25 * (count_pen + med_pen + p95_pen + ks_stat)


def _choose_mask_expansion_pixels(
    grid: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
    *,
    base_mask: np.ndarray,
    center: float,
    width: float,
    power: float,
    amplitude: float,
    min_pixels: int,
    max_expand: int,
    fixed_line_mask: np.ndarray | None = None,
    fixed_cont_mask: np.ndarray | None = None,
    cr_mask: np.ndarray | None = None,
    poly_coef: tuple[float, ...] | None = None,
    poly_w_center: float = 0.0,
    poly_w_scale: float = 1.0,
) -> int:
    """Pick excluded-region dilation (pixels per side) that best balances pull tails."""
    if max_expand <= 0:
        return 0
    g = np.asarray(grid, float)
    f = np.asarray(flux, float)
    post_mask = np.asarray(mask, bool)
    bounds = np.asarray(base_mask, bool)

    def _score_for_expand(n_pix: int) -> float | None:
        trial = _expand_excluded_mask(post_mask, n_pix, bounds=bounds)
        trial = _apply_fixed_mask_constraints(
            trial,
            fixed_line_mask=fixed_line_mask,
            fixed_cont_mask=fixed_cont_mask,
            cr_mask=cr_mask,
        )
        if int(np.sum(trial)) < int(min_pixels):
            return None
        norm = _blaze_normalized_flux(
            g,
            f,
            trial,
            center=center,
            width=width,
            power=power,
            amplitude=amplitude,
            poly_coef=poly_coef,
            poly_w_center=poly_w_center,
            poly_w_scale=poly_w_scale,
        )
        pulls = norm[trial] - 1.0
        return _pull_distribution_consistency_score(pulls)

    best_x = 0
    best_score = _score_for_expand(0)
    if best_score is None:
        return 0
    prev_score = best_score
    for xp in range(1, int(max_expand) + 1):
        score = _score_for_expand(xp)
        if score is None:
            break
        if score < best_score - 1e-6:
            best_score = score
            best_x = xp
        elif xp > best_x and score > prev_score + 0.02:
            break
        prev_score = score
    return best_x


def _apply_fixed_mask_constraints(
    mask: np.ndarray,
    *,
    fixed_line_mask: np.ndarray | None = None,
    fixed_cont_mask: np.ndarray | None = None,
    cr_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Enforce manual line/continuum and CR constraints on a continuum mask."""
    m = np.asarray(mask, bool).copy()
    if cr_mask is not None:
        m &= np.asarray(cr_mask, bool)
    if fixed_line_mask is not None:
        m &= ~np.asarray(fixed_line_mask, bool)
    if fixed_cont_mask is not None:
        m |= np.asarray(fixed_cont_mask, bool)
    return m


def _manual_expand_bounds(
    base_mask: np.ndarray,
    *,
    cr_mask: np.ndarray | None = None,
    fixed_line_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Dilation bounds: excluded regions cannot grow through fixed lines."""
    bounds = np.asarray(base_mask, bool)
    if cr_mask is not None:
        bounds &= np.asarray(cr_mask, bool)
    if fixed_line_mask is not None:
        bounds &= ~np.asarray(fixed_line_mask, bool)
    return bounds


def _iterative_mask_snr_fixed_blaze(
    grid: np.ndarray,
    flux: np.ndarray,
    base_mask: np.ndarray,
    *,
    center: float,
    width: float,
    power: float,
    amplitude: float = 1.0,
    thresholds: tuple[float, ...] | None = None,
    nsigma: float | None = None,
    window_angstrom: float | None = None,
    max_mask_expand: int | None = None,
    min_pixels: int | None = None,
    fixed_line_mask: np.ndarray | None = None,
    fixed_cont_mask: np.ndarray | None = None,
    cr_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    S/N-significance continuum mask with fixed sinc² shape (no blaze refit).

    Used at ingest: calibrated blaze divides flux; mask selects pixels for median scale.
    """
    g = np.asarray(grid, float)
    f = np.asarray(flux, float)
    base = np.asarray(base_mask, bool)
    thr_list = tuple(
        thresholds
        if thresholds is not None
        else getattr(config, "BLAZE_ITERATIVE_THRESHOLDS", BLAZE_ITERATIVE_THRESHOLDS)
    )
    min_pix = int(
        min_pixels
        if min_pixels is not None
        else getattr(config, "BLAZE_ITERATIVE_MIN_PIXELS", 18)
    )
    sig = float(
        nsigma
        if nsigma is not None
        else getattr(config, "BLAZE_ITERATIVE_NSIGMA", BLAZE_ITERATIVE_NSIGMA)
    )
    mask = base.copy()
    mask = _apply_fixed_mask_constraints(
        mask,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
    )
    if int(np.sum(mask)) < min_pix:
        return mask
    if window_angstrom is None:
        span = float(np.max(g) - np.min(g)) if g.size else 80.0
        window_angstrom = max(span / 12.0, 3.0)

    for thr in thr_list:
        norm = _blaze_normalized_flux(
            g, f, mask, center=center, width=width, power=power, amplitude=amplitude
        )
        sigma_mask = mask & (norm >= float(thr))
        if int(np.sum(sigma_mask)) < 3:
            sigma_mask = mask
        snr_cont = _continuum_snr_per_pixel(
            g, norm, sigma_mask, window_angstrom=window_angstrom
        )
        dev = 1.0 - norm
        significant = dev > (sig / np.maximum(snr_cont, 1e-9))
        mask = base & ~((norm < float(thr)) & significant)
        mask = _apply_fixed_mask_constraints(
            mask,
            fixed_line_mask=fixed_line_mask,
            fixed_cont_mask=fixed_cont_mask,
            cr_mask=cr_mask,
        )
        if int(np.sum(mask)) < min_pix:
            break

    max_expand = int(
        max_mask_expand
        if max_mask_expand is not None
        else getattr(config, "BLAZE_MASK_EXPAND_MAX_PIXELS", 8)
    )
    expand_bounds = _manual_expand_bounds(
        base,
        cr_mask=cr_mask,
        fixed_line_mask=fixed_line_mask,
    )
    expand_px = _choose_mask_expansion_pixels(
        g,
        f,
        mask,
        base_mask=expand_bounds,
        center=center,
        width=width,
        power=power,
        amplitude=amplitude,
        min_pixels=min_pix,
        max_expand=max_expand,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
    )
    if expand_px > 0:
        mask = _expand_excluded_mask(mask, expand_px, bounds=expand_bounds)
    mask = _apply_fixed_mask_constraints(
        mask,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
    )
    return mask


def continuum_mask_for_blaze_model(
    wavelength: np.ndarray,
    flux: np.ndarray,
    blaze_model: OrderBlazeModel,
    *,
    rest_lines: list[float] | None = None,
    half_width_angstrom: float | None = None,
    thresholds: tuple[float, ...] | None = None,
    nsigma: float | None = None,
    max_mask_expand: int | None = None,
    base_mask: np.ndarray | None = None,
    fixed_line_mask: np.ndarray | None = None,
    fixed_cont_mask: np.ndarray | None = None,
    cr_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Iterative S/N mask using a fixed calibrated ``OrderBlazeModel`` (ingest path)."""
    w = np.asarray(wavelength, float)
    f = np.asarray(flux, float)
    if base_mask is None:
        hw = float(
            half_width_angstrom
            if half_width_angstrom is not None
            else blaze_model.line_mask_half_width_angstrom
        )
        if rest_lines is None:
            rests = strong_lines_in_span(float(np.min(w)), float(np.max(w)))
        else:
            rests = rest_lines
        if rests:
            base_mask = blaze_line_mask(w, rest_lines=rests, half_width_angstrom=hw)
        else:
            base_mask = np.isfinite(w) & np.isfinite(f) & (f > 0)
    else:
        base_mask = np.asarray(base_mask, bool)
    return _iterative_mask_snr_fixed_blaze(
        w,
        f,
        base_mask,
        center=float(blaze_model.center_angstrom),
        width=float(blaze_model.width_angstrom),
        power=float(blaze_model.power),
        amplitude=1.0,
        thresholds=thresholds,
        nsigma=nsigma,
        max_mask_expand=max_mask_expand,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
    )


def shared_blaze_order_envelope(
    wavelength: np.ndarray,
    flux: np.ndarray,
    blaze_model: OrderBlazeModel,
    *,
    rest_lines: list[float] | None = None,
    base_mask: np.ndarray | None = None,
    fixed_line_mask: np.ndarray | None = None,
    fixed_cont_mask: np.ndarray | None = None,
    cr_mask: np.ndarray | None = None,
    fit_model: str = FIT_MODEL_SINC2,
    poly_order: int = 2,
) -> SharedBlazeOrderResult | None:
    """
    Shared per-order blaze shape with per-spectrum amplitude scale.

    ``envelope = scale × blaze_model(w) [× exp(poly)]``; continuum mask uses fixed
    calibrated sinc² shape (no per-epoch sinc refit).
    """
    w = np.asarray(wavelength, float)
    f = np.asarray(flux, float)
    cont_mask = continuum_mask_for_blaze_model(
        w,
        f,
        blaze_model,
        rest_lines=rest_lines,
        base_mask=base_mask,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
    )
    shape = blaze_model.blaze_on_grid(w)
    ok = cont_mask & np.isfinite(f) & np.isfinite(shape) & (shape > 0)
    if not np.any(ok):
        return None
    scale = float(np.nanmedian(f[ok] / shape[ok]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    envelope = scale * shape
    effective_model = FIT_MODEL_SINC2
    poly_coef: tuple[float, ...] | None = None
    poly_w_center = 0.0
    poly_w_scale = 1.0

    if fit_model == FIT_MODEL_SINC_POLY:
        p_order = max(1, min(int(poly_order), 4))
        poly_fit = _fit_poly_multiplier_on_mask(
            w,
            f,
            cont_mask,
            center=float(blaze_model.center_angstrom),
            width=float(blaze_model.width_angstrom),
            power=float(blaze_model.power),
            amplitude=scale,
            poly_order=p_order,
        )
        if poly_fit is not None:
            poly_coef, poly_w_center, poly_w_scale = poly_fit
            poly = eval_poly_multiplier(w, poly_coef, poly_w_center, poly_w_scale)
            envelope = scale * shape * poly
            effective_model = FIT_MODEL_SINC_POLY

    return SharedBlazeOrderResult(
        envelope=envelope,
        continuum_mask=cont_mask,
        scale=scale,
        fit_model=effective_model,
        poly_coef=poly_coef,
        poly_w_center=poly_w_center,
        poly_w_scale=poly_w_scale,
    )


def normalize_order_sinc_blaze_only(
    wavelength: np.ndarray,
    flux: np.ndarray,
    eflux: np.ndarray,
    blaze_model: OrderBlazeModel,
    *,
    rest_lines: list[float] | None = None,
    base_mask: np.ndarray | None = None,
    fixed_line_mask: np.ndarray | None = None,
    fixed_cont_mask: np.ndarray | None = None,
    cr_mask: np.ndarray | None = None,
    fit_model: str = FIT_MODEL_SINC2,
    poly_order: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blaze-correct and median-scale flux for uberMS ingest (``sinc_blaze_only``).

    Uses shared ``OrderBlazeModel`` shape with per-spectrum scale; optional log-poly
    multiplier when ``fit_model=sinc_poly``.
    """
    w = np.asarray(wavelength, float)
    f = np.asarray(flux, float)
    e = np.asarray(eflux, float)

    result = shared_blaze_order_envelope(
        w,
        f,
        blaze_model,
        rest_lines=rest_lines,
        base_mask=base_mask,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
        fit_model=fit_model,
        poly_order=poly_order,
    )
    if result is None:
        use = np.isfinite(f) & (f > 0) & np.isfinite(e)
        return w[use], f[use], e[use]

    norm_f = f / result.envelope
    ok = result.continuum_mask & np.isfinite(norm_f) & (norm_f > 0)
    level = float(np.nanmedian(norm_f[ok])) if np.any(ok) else 1.0
    if not np.isfinite(level) or level <= 0:
        level = 1.0
    nf = norm_f / level
    ne = e / result.envelope / level
    use = np.isfinite(nf) & (nf > 0) & np.isfinite(ne)
    return w[use], nf[use], ne[use]


def fit_order_blaze_iterative(
    grid: np.ndarray,
    flux: np.ndarray,
    *,
    thresholds: tuple[float, ...] | None = None,
    initial_mask: np.ndarray | None = None,
    min_pixels: int | None = None,
    nsigma: float | None = None,
    window_angstrom: float | None = None,
    max_mask_expand: int | None = None,
    fixed_line_mask: np.ndarray | None = None,
    fixed_cont_mask: np.ndarray | None = None,
    cr_mask: np.ndarray | None = None,
    fit_model: str = FIT_MODEL_SINC2,
    poly_order: int = 2,
) -> IterativeBlazeFitResult | None:
    """
    Iterative sinc² blaze fit with S/N-significance line masking.

    After each fit, rebuild the mask from ``initial_mask`` (no persistence) and
    exclude pixels where blaze-normalized flux is below threshold ``t`` and the
    deviation from continuum is significant at ``nsigma`` given local continuum S/N.

    When ``fit_model=sinc_poly``, fit a polynomial multiplier on ``flux/sinc²`` each round.
    """
    g = np.asarray(grid, float)
    f = np.asarray(flux, float)
    if g.size != f.size:
        raise ValueError("grid and flux must have the same length")
    use_poly = fit_model == FIT_MODEL_SINC_POLY
    p_order = max(1, min(int(poly_order), 4))
    thr_list = tuple(
        thresholds
        if thresholds is not None
        else getattr(config, "BLAZE_ITERATIVE_THRESHOLDS", BLAZE_ITERATIVE_THRESHOLDS)
    )
    min_pix = int(
        min_pixels
        if min_pixels is not None
        else getattr(config, "BLAZE_ITERATIVE_MIN_PIXELS", 18)
    )
    sig = float(
        nsigma
        if nsigma is not None
        else getattr(config, "BLAZE_ITERATIVE_NSIGMA", BLAZE_ITERATIVE_NSIGMA)
    )
    if initial_mask is None:
        base_mask = np.isfinite(g) & np.isfinite(f) & (f > 0)
    else:
        base_mask = np.asarray(initial_mask, bool).copy()
        base_mask &= np.isfinite(g) & np.isfinite(f) & (f > 0)
    mask = base_mask.copy()
    mask = _apply_fixed_mask_constraints(
        mask,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
    )
    if int(np.sum(mask)) < min_pix:
        return None

    fit = _fit_sinc2_on_mask(g, f, mask)
    if fit is None:
        return None
    center, width, power, amp = fit
    poly_coef: tuple[float, ...] | None = None
    poly_w_center = 0.0
    poly_w_scale = 1.0
    if use_poly:
        poly_fit = _fit_poly_multiplier_on_mask(
            g,
            f,
            mask,
            center=center,
            width=width,
            power=power,
            amplitude=amp,
            poly_order=p_order,
        )
        if poly_fit is not None:
            poly_coef, poly_w_center, poly_w_scale = poly_fit

    stage_counts: list[tuple[float, int]] = []
    if window_angstrom is None:
        span = float(np.max(g) - np.min(g)) if g.size else 80.0
        window_angstrom = max(span / 12.0, 3.0)

    def _norm_kw() -> dict[str, Any]:
        if use_poly and poly_coef is not None:
            return {
                "poly_coef": poly_coef,
                "poly_w_center": poly_w_center,
                "poly_w_scale": poly_w_scale,
            }
        return {}

    for thr in thr_list:
        norm = _blaze_normalized_flux(
            g,
            f,
            mask,
            center=center,
            width=width,
            power=power,
            amplitude=amp,
            **_norm_kw(),
        )
        sigma_mask = mask & (norm >= float(thr))
        if int(np.sum(sigma_mask)) < 3:
            sigma_mask = mask
        snr_cont = _continuum_snr_per_pixel(
            g, norm, sigma_mask, window_angstrom=window_angstrom
        )
        dev = 1.0 - norm
        significant = dev > (sig / np.maximum(snr_cont, 1e-9))
        mask = base_mask & ~((norm < float(thr)) & significant)
        mask = _apply_fixed_mask_constraints(
            mask,
            fixed_line_mask=fixed_line_mask,
            fixed_cont_mask=fixed_cont_mask,
            cr_mask=cr_mask,
        )
        n_ok = int(np.sum(mask))
        stage_counts.append((float(thr), n_ok))
        if n_ok < min_pix:
            break
        fit_next = _fit_sinc2_on_mask(g, f, mask)
        if fit_next is None:
            break
        center, width, power, amp = fit_next
        if use_poly:
            poly_fit = _fit_poly_multiplier_on_mask(
                g,
                f,
                mask,
                center=center,
                width=width,
                power=power,
                amplitude=amp,
                poly_order=p_order,
            )
            if poly_fit is not None:
                poly_coef, poly_w_center, poly_w_scale = poly_fit

    if int(np.sum(mask)) < min_pix:
        return None

    max_expand = int(
        max_mask_expand
        if max_mask_expand is not None
        else getattr(config, "BLAZE_MASK_EXPAND_MAX_PIXELS", 8)
    )
    expand_bounds = _manual_expand_bounds(
        base_mask,
        cr_mask=cr_mask,
        fixed_line_mask=fixed_line_mask,
    )
    expand_px = _choose_mask_expansion_pixels(
        g,
        f,
        mask,
        base_mask=expand_bounds,
        center=center,
        width=width,
        power=power,
        amplitude=amp,
        min_pixels=min_pix,
        max_expand=max_expand,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
        poly_coef=poly_coef if use_poly else None,
        poly_w_center=poly_w_center,
        poly_w_scale=poly_w_scale,
    )
    if expand_px > 0:
        mask = _expand_excluded_mask(mask, expand_px, bounds=expand_bounds)
        fit_final = _fit_sinc2_on_mask(g, f, mask)
        if fit_final is not None:
            center, width, power, amp = fit_final
            if use_poly:
                poly_fit = _fit_poly_multiplier_on_mask(
                    g,
                    f,
                    mask,
                    center=center,
                    width=width,
                    power=power,
                    amplitude=amp,
                    poly_order=p_order,
                )
                if poly_fit is not None:
                    poly_coef, poly_w_center, poly_w_scale = poly_fit

    mask = _apply_fixed_mask_constraints(
        mask,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
    )

    return IterativeBlazeFitResult(
        center_angstrom=center,
        width_angstrom=width,
        power=power,
        amplitude=amp,
        continuum_mask=mask,
        stage_counts=tuple(stage_counts),
        mask_expand_pixels=expand_px,
        fit_model=fit_model if use_poly and poly_coef is not None else FIT_MODEL_SINC2,
        poly_coef=poly_coef if use_poly else None,
        poly_w_center=poly_w_center,
        poly_w_scale=poly_w_scale,
    )


def blaze_fit_continuum_mask(
    wavelength: np.ndarray,
    flux: np.ndarray,
    *,
    rest_lines: list[float] | None = None,
    half_width_angstrom: float = 22.0,
    thresholds: tuple[float, ...] | None = None,
    min_pixels: int | None = None,
    nsigma: float | None = None,
    stage_counts: list[tuple[float, int]] | None = None,
    initial_mask: np.ndarray | None = None,
    fixed_line_mask: np.ndarray | None = None,
    fixed_cont_mask: np.ndarray | None = None,
    cr_mask: np.ndarray | None = None,
    fit_model: str = FIT_MODEL_SINC2,
    poly_order: int = 2,
    **kwargs: Any,
) -> np.ndarray:
    """
    Pixels used to fit shared sinc² blaze (S/N-significance iterative mask).

    Optional ``rest_lines`` applies static Balmer/He/Ca exclusion before round 0.
    If ``stage_counts`` is provided, it is filled with (threshold, n_pixels) per round.
    """
    del kwargs  # legacy stellar/absorption kwargs ignored
    w = np.asarray(wavelength, float)
    f = np.asarray(flux, float)
    if initial_mask is not None:
        mask0 = np.asarray(initial_mask, bool)
        mask0 &= np.isfinite(w) & np.isfinite(f) & (f > 0)
    elif rest_lines:
        mask0 = blaze_line_mask(
            w,
            rest_lines=rest_lines,
            half_width_angstrom=half_width_angstrom,
        )
    else:
        mask0 = np.isfinite(w) & np.isfinite(f) & (f > 0)
    result = fit_order_blaze_iterative(
        w,
        f,
        thresholds=thresholds,
        initial_mask=mask0,
        min_pixels=min_pixels,
        nsigma=nsigma,
        fixed_line_mask=fixed_line_mask,
        fixed_cont_mask=fixed_cont_mask,
        cr_mask=cr_mask,
        fit_model=fit_model,
        poly_order=poly_order,
    )
    if result is None:
        return mask0
    if stage_counts is not None:
        stage_counts.clear()
        stage_counts.extend(result.stage_counts)
    return result.continuum_mask


def find_order_covering_rest(
    spec_data: dict,
    rest: float,
    *,
    bad_orders: list[int] | None = None,
) -> int | None:
    bad = set(bad_orders or [])
    for o in sorted(spec_data.keys()):
        if int(o) in bad:
            continue
        w = np.asarray(spec_data[o]["wavelength"], float)
        if w.size < 10:
            continue
        if float(np.min(w)) <= rest <= float(np.max(w)):
            return int(o)
    return None


def order_covers_strong_line(
    wavelength_min: float,
    wavelength_max: float,
    *,
    rest_lines: list[float] | None = None,
) -> bool:
    """True if any entry in STRONG_LINES falls inside the order wavelength span."""
    rests = rest_lines if rest_lines is not None else list(continuum.STRONG_LINES)
    wmn, wmx = float(wavelength_min), float(wavelength_max)
    return any(wmn <= float(rest) <= wmx for rest in rests)


def list_clean_orders(
    spec_data: dict,
    *,
    bad_orders: list[int] | None = None,
    rest_lines: list[float] | None = None,
) -> list[tuple[int, float, float]]:
    """Echelle orders with no STRONG_LINES hit; returns (order, wmin, wmax)."""
    bad = set(bad_orders or [])
    out: list[tuple[int, float, float]] = []
    for o in sorted(spec_data.keys()):
        if int(o) in bad:
            continue
        w = np.asarray(spec_data[o]["wavelength"], float)
        if w.size < 10:
            continue
        wmn, wmx = float(np.min(w)), float(np.max(w))
        if not order_covers_strong_line(wmn, wmx, rest_lines=rest_lines):
            out.append((int(o), wmn, wmx))
    return out


def pick_clean_order_near_wavelength(
    spec_data: dict,
    target_angstrom: float,
    *,
    bad_orders: list[int] | None = None,
) -> tuple[int, float, float] | None:
    """Choose the clean order whose midpoint is closest to ``target_angstrom``."""
    clean = list_clean_orders(spec_data, bad_orders=bad_orders)
    if not clean:
        return None
    target = float(target_angstrom)
    o, wmn, wmx = min(clean, key=lambda row: abs(0.5 * (row[1] + row[2]) - target))
    return o, wmn, wmx


def hbeta_absorption_depth_raw(
    wavelength: np.ndarray,
    flux: np.ndarray,
    *,
    rest: float = HB_REST_A,
    core_half_angstrom: float = 18.0,
    wing_half_angstrom: float = 70.0,
) -> float:
    """
    Shallow-line proxy on raw (blaze-shaped) flux.

    Fits a local linear continuum in the wings (excluding the line core), then
    returns 1 − min(core flux) / median(continuum in core).
    """
    w = np.asarray(wavelength, float)
    f = np.asarray(flux, float)
    m_wing = (w >= rest - wing_half_angstrom) & (w <= rest + wing_half_angstrom) & np.isfinite(f)
    m_core = (w >= rest - core_half_angstrom) & (w <= rest + core_half_angstrom) & np.isfinite(f)
    m_fit = m_wing & ~m_core
    if int(np.sum(m_core)) < 5 or int(np.sum(m_fit)) < 12:
        return float("nan")
    coef = np.polyfit(w[m_fit], f[m_fit], 1)
    cont_core = np.polyval(coef, w[m_core])
    env = float(np.nanmedian(cont_core))
    if not np.isfinite(env) or env <= 0:
        return float("nan")
    return float(1.0 - np.nanmin(f[m_core]) / env)


@dataclass
class OrderBlazeModel:
    """Shared blaze for one echelle order (wavelengths in Å)."""

    echelle_order: int
    model: str  # "sinc2"
    center_angstrom: float
    width_angstrom: float
    power: float
    n_spectra_fit: int
    wavelength_min: float
    wavelength_max: float
    rest_line_angstrom: float = HB_REST_A
    line_mask_half_width_angstrom: float = 22.0

    def blaze_on_grid(self, wavelength: np.ndarray) -> np.ndarray:
        return eval_blaze_sinc2(
            wavelength,
            self.center_angstrom,
            self.width_angstrom,
            power=self.power,
            amplitude=1.0,
        )

    def correct_flux(self, wavelength: np.ndarray, flux: np.ndarray) -> np.ndarray:
        w = np.asarray(wavelength, float)
        f = np.asarray(flux, float)
        b = self.blaze_on_grid(w)
        b = np.maximum(b, 1e-9 * float(np.nanmax(b)))
        return f / b

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> OrderBlazeModel:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> OrderBlazeModel:
        return cls.from_json_dict(json.loads(Path(path).read_text()))


def _interp_to_grid(w: np.ndarray, f: np.ndarray, grid: np.ndarray) -> np.ndarray:
    o = np.argsort(w)
    ws, fs = w[o], f[o]
    m = np.isfinite(ws) & np.isfinite(fs)
    if int(np.sum(m)) < 8:
        return np.full(grid.shape, np.nan, dtype=float)
    return np.interp(grid, ws[m], fs[m], left=np.nan, right=np.nan)


def fit_order_blaze_from_profiles(
    profiles: list[tuple[np.ndarray, np.ndarray]],
    echelle_order: int,
    *,
    line_mask_half_width: float = 22.0,
    rest_lines: list[float] | None = None,
    mask_wave: np.ndarray | None = None,
    mask_strength: np.ndarray | None = None,
    use_stellar_mask: bool = True,
) -> OrderBlazeModel | None:
    """
    Fit shared sinc² blaze to multiple raw-flux profiles on the same order.

    Uses iterative refit with progressive blaze-normalized thresholds. Stellar-mask
    kwargs are accepted for API compatibility but ignored.
    """
    del mask_wave, mask_strength, use_stellar_mask
    if len(profiles) < 3:
        logger.warning("need at least 3 profiles to fit blaze; got %d", len(profiles))
        return None

    grid = _uniform_order_grid(profiles)
    if grid.size < 40:
        return None

    stack = np.vstack([_interp_to_grid(w, f, grid) for w, f in profiles])
    median_f = np.nanmedian(stack, axis=0)
    if rest_lines:
        initial_mask = blaze_line_mask(
            grid,
            rest_lines=rest_lines,
            half_width_angstrom=line_mask_half_width,
        )
    else:
        initial_mask = np.isfinite(median_f) & (median_f > 0)

    result = fit_order_blaze_iterative(grid, median_f, initial_mask=initial_mask)
    if result is None:
        return None

    return OrderBlazeModel(
        echelle_order=int(echelle_order),
        model="sinc2",
        center_angstrom=result.center_angstrom,
        width_angstrom=result.width_angstrom,
        power=result.power,
        n_spectra_fit=len(profiles),
        wavelength_min=float(grid[0]),
        wavelength_max=float(grid[-1]),
        rest_line_angstrom=HB_REST_A,
        line_mask_half_width_angstrom=float(line_mask_half_width),
    )


def median_profile_and_rms(
    profiles: list[tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
    *,
    line_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.vstack([_interp_to_grid(w, f, grid) for w, f in profiles])
    if line_mask is not None:
        stack[:, ~line_mask] = np.nan
    med = np.nanmedian(stack, axis=0)
    rms = np.nanstd(stack, axis=0)
    return med, rms


def strong_lines_in_span(wavelength_min: float, wavelength_max: float) -> list[float]:
    """STRONG_LINES entries that fall inside an order wavelength span."""
    wmn, wmx = float(wavelength_min), float(wavelength_max)
    return [float(r) for r in continuum.STRONG_LINES if wmn <= float(r) <= wmx]


@dataclass
class BlazeCalibration:
    """Per-order sinc² blaze models for one instrument."""

    instrument: str
    n_spectra_fit: int
    min_snr: float
    orders: dict[int, OrderBlazeModel]

    def model_for_order(self, echelle_order: int) -> OrderBlazeModel | None:
        return self.orders.get(int(echelle_order))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "n_spectra_fit": self.n_spectra_fit,
            "min_snr": self.min_snr,
            "orders": {str(k): v.to_json_dict() for k, v in sorted(self.orders.items())},
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> BlazeCalibration:
        orders = {
            int(k): OrderBlazeModel.from_json_dict(v)
            for k, v in (d.get("orders") or {}).items()
        }
        return cls(
            instrument=str(d.get("instrument", "APF")),
            n_spectra_fit=int(d.get("n_spectra_fit", 0)),
            min_snr=float(d.get("min_snr", 0.0)),
            orders=orders,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> BlazeCalibration:
        return cls.from_json_dict(json.loads(Path(path).read_text()))


def build_blaze_calibration(
    spec_paths: list[Path],
    instrument,
    *,
    min_snr: float = 3.5,
    overlap: dict[str, dict] | None = None,
    line_mask_half_width: float = 22.0,
    min_profiles: int = 8,
) -> BlazeCalibration:
    """
    Fit a shared sinc² blaze per echelle order from many spectra.

    Uses iterative progressive masking; static STRONG_LINES mask when lines fall in span.
    """
    from collections import defaultdict

    overlap = overlap or {}
    bad = set(instrument.bad_orders or [])
    by_order: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    n_used = 0

    for spec_path in spec_paths:
        stem = Path(spec_path).stem
        meta = overlap.get(stem, {})
        snr = float(meta.get("median_mask_ccf_peak_snr", np.nan))
        if overlap and np.isfinite(snr) and snr < float(min_snr):
            continue
        try:
            _hdr, spec_data = io_utils.read_spectrum(str(spec_path))
        except Exception as ex:
            logger.debug("skip %s: %s", stem, ex)
            continue
        n_used += 1
        for o in spec_data:
            if int(o) in bad:
                continue
            w = np.asarray(spec_data[o]["wavelength"], float)
            f = np.asarray(spec_data[o]["flux"], float)
            if w.size < 20 or not np.any(np.isfinite(f) & (f > 0)):
                continue
            by_order[int(o)].append((w, f))

    models: dict[int, OrderBlazeModel] = {}
    for o in sorted(by_order):
        profiles = by_order[o]
        if len(profiles) < int(min_profiles):
            logger.debug("order %d: only %d profiles", o, len(profiles))
            continue
        wmins = [float(np.min(w)) for w, _f in profiles]
        wmaxs = [float(np.max(w)) for w, _f in profiles]
        wmn, wmx = float(np.min(wmins)), float(np.max(wmaxs))
        rests = strong_lines_in_span(wmn, wmx)
        model = fit_order_blaze_from_profiles(
            profiles,
            o,
            line_mask_half_width=float(line_mask_half_width),
            rest_lines=rests if rests else [],
        )
        if model is not None:
            models[o] = model

    return BlazeCalibration(
        instrument=str(getattr(instrument, "name", "APF")),
        n_spectra_fit=n_used,
        min_snr=float(min_snr),
        orders=models,
    )
