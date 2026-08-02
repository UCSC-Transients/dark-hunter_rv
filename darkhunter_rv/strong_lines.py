"""
Strong-line product helpers: candidate list metadata, inclusion gates, debias + IVW combine.

Issues: #91 (wire metals), #92 (inclusion), #93 (debias/weights).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from darkhunter_rv import config, qc

logger = logging.getLogger(__name__)

# Rest wavelengths (air Å) for product keep-candidates (#91).
MG_IB2_REST_A = 5172.68
MG_IB3_REST_A = 5183.60
CA_I_6122_REST_A = 6122.22
CA_I_6162_REST_A = 6162.17
CA_I_4227_REST_A = 4226.73

# Prefer Balmer Hβ first; then optical metals from the 114-stem keep list; then other Balmer.
# Excludes Ca H&K, red IR, Fe I unhelpful, Mg I b₁ (blend).
PRODUCT_STRONG_LINE_ORDER: tuple[tuple[str, float, bool], ...] = (
    # name, rest_a, is_balmer (broad_lines path when hot)
    ("Hbeta", 4861.3, True),
    ("MgIb2", MG_IB2_REST_A, False),
    ("CaI6122", CA_I_6122_REST_A, False),
    ("CaI6162", CA_I_6162_REST_A, False),
    ("MgIb3", MG_IB3_REST_A, False),
    ("CaI4227", CA_I_4227_REST_A, False),
    ("Hgamma", 4340.5, True),
    ("Hdelta", 4101.7, True),
    ("Halpha", 6562.8, True),
)


@dataclass(frozen=True)
class StrongLineInclusionConfig:
    """Gates for including a fitted line in the exposure strong_lines RV (#92)."""

    min_depth: float = 0.05
    max_err_kms: float = 40.0
    min_snr: float = 8.0  # continuum flux/eflux near the line
    min_width_kms: float = 3.0
    max_width_kms: float = 250.0
    max_abs_rv_kms: float = 400.0
    max_telluric_frac: float = 0.08


DEFAULT_INCLUSION = StrongLineInclusionConfig()


def product_strong_line_rests() -> list[tuple[str, float]]:
    """Ordered (name, rest_Å) for the product strong-lines loop."""
    return [(n, float(r)) for n, r, _balmer in PRODUCT_STRONG_LINE_ORDER]


def line_uses_broad_profile(line_name: str, *, hot_spectrum: bool) -> bool:
    """Balmer lines use the broad Voigt window on hot stars; metals stay narrow."""
    meta = {n: balmer for n, _r, balmer in PRODUCT_STRONG_LINE_ORDER}
    if not meta.get(str(line_name), False):
        return False
    return bool(hot_spectrum) or str(line_name) == "Hbeta"


def local_core_depth(wave: np.ndarray, flux_norm: np.ndarray, rest: float) -> float:
    w = np.asarray(wave, float)
    f = np.asarray(flux_norm, float)
    m = (w >= rest - 25.0) & (w <= rest + 25.0) & np.isfinite(f)
    if int(np.sum(m)) < 30:
        return float("nan")
    ww, ff = w[m], f[m]
    core = (ww >= rest - 4.5) & (ww <= rest + 4.5)
    wing = ~core
    if int(np.sum(wing)) < 15 or int(np.sum(core)) < 5:
        return float("nan")
    cont = float(np.nanmedian(ff[wing]))
    if not np.isfinite(cont) or cont <= 0.05:
        return float("nan")
    return float(1.0 - np.nanmin(ff[core]) / cont)


def local_snr_near_line(
    wave: np.ndarray,
    flux: np.ndarray,
    eflux: np.ndarray,
    rest: float,
    *,
    half_window_a: float = 25.0,
    core_half_a: float = 4.5,
) -> float:
    """
    Continuum S/N **near** ``rest``: median(flux / eflux) in the local wings.

    Uses raw (or blaze-corrected) flux and uncertainty in ±``half_window_a`` Å,
    excluding ±``core_half_a`` Å about the line. This tracks instrument sensitivity
    and stellar SED at that wavelength — both vary strongly across the echellogram
    and across stars of different color.
    """
    w = np.asarray(wave, float)
    f = np.asarray(flux, float)
    e = np.asarray(eflux, float)
    if w.size != f.size or w.size != e.size or w.size < 20:
        return float("nan")
    m = (
        (w >= rest - half_window_a)
        & (w <= rest + half_window_a)
        & np.isfinite(f)
        & np.isfinite(e)
        & (e > 0)
        & (f > 0)
    )
    if int(np.sum(m)) < 20:
        return float("nan")
    ww, ff, ee = w[m], f[m], e[m]
    wing = ~((ww >= rest - core_half_a) & (ww <= rest + core_half_a))
    if int(np.sum(wing)) < 10:
        return float("nan")
    snr_pix = ff[wing] / ee[wing]
    snr_pix = snr_pix[np.isfinite(snr_pix) & (snr_pix > 0)]
    if snr_pix.size < 8:
        return float("nan")
    return float(np.nanmedian(snr_pix))


def local_continuum_snr(wave: np.ndarray, flux_norm: np.ndarray, rest: float) -> float:
    """
    Legacy normalized-flux scatter S/N (1/rms about continuum).

    Prefer :func:`local_snr_near_line` with flux/eflux for weighting — normalized
    spectra erase absolute sensitivity and stellar color.
    """
    w = np.asarray(wave, float)
    f = np.asarray(flux_norm, float)
    m = (w >= rest - 25.0) & (w <= rest + 25.0) & np.isfinite(f)
    if int(np.sum(m)) < 30:
        return float("nan")
    ww, ff = w[m], f[m]
    wing = ~((ww >= rest - 4.5) & (ww <= rest + 4.5))
    if int(np.sum(wing)) < 15:
        return float("nan")
    rms = float(np.nanstd(ff[wing] - np.nanmedian(ff[wing])))
    if not np.isfinite(rms) or rms < 1e-6:
        return float("nan")
    return float(1.0 / rms)


def fit_width_kms(bundle: Mapping, rest: float) -> float:
    """Gaussian σ of joint Voigt+Lorentz fit converted to km/s."""
    p = bundle.get("hb_joint_fit_params")
    if p is None or len(p) < 8:
        return float("nan")
    sig_a = abs(float(p[3]))
    r = float(rest)
    if r <= 0:
        return float("nan")
    return float(config.C_KMS * sig_a / r)


def telluric_fraction_near(wave: np.ndarray, rest: float, half_a: float = 40.0) -> float:
    w = np.asarray(wave, float)
    m = (w >= rest - half_a) & (w <= rest + half_a) & np.isfinite(w)
    if int(np.sum(m)) < 5:
        return float("nan")
    bad = qc.wavelength_band_mask(w[m], qc.rv_contamination_bands())
    return float(np.mean(bad))


def strong_line_fit_metrics(
    *,
    wave: np.ndarray,
    flux_norm: np.ndarray,
    rest: float,
    rv_kms: float,
    err_kms: float,
    bundle: Mapping | None = None,
    flux: np.ndarray | None = None,
    eflux: np.ndarray | None = None,
    wave_native: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Line metrics for inclusion and weighting.

    ``snr`` is continuum S/N **near the line** from flux/eflux when provided
    (:func:`local_snr_near_line`); otherwise falls back to normalized-flux scatter.
    """
    depth = local_core_depth(wave, flux_norm, rest)
    snr_norm = local_continuum_snr(wave, flux_norm, rest)
    snr_flux = float("nan")
    if flux is not None and eflux is not None:
        w_use = wave_native if wave_native is not None else wave
        snr_flux = local_snr_near_line(w_use, flux, eflux, rest)
    # Prefer photon/instrument S/N at the line; normalized scatter is a last resort.
    if np.isfinite(snr_flux) and snr_flux > 0:
        snr = float(snr_flux)
    elif np.isfinite(depth) and np.isfinite(snr_norm):
        snr = float(depth * snr_norm)
    else:
        snr = float("nan")
    width = fit_width_kms(bundle or {}, rest) if bundle is not None else float("nan")
    tfrac = telluric_fraction_near(wave, rest)
    return {
        "depth": float(depth) if np.isfinite(depth) else float("nan"),
        "snr": snr,
        "snr_near_line": float(snr_flux) if np.isfinite(snr_flux) else float("nan"),
        "continuum_snr": float(snr_norm) if np.isfinite(snr_norm) else float("nan"),
        "width_kms": float(width) if np.isfinite(width) else float("nan"),
        "err_kms": float(err_kms) if np.isfinite(err_kms) else float("nan"),
        "rv_kms": float(rv_kms) if np.isfinite(rv_kms) else float("nan"),
        "telluric_frac": float(tfrac) if np.isfinite(tfrac) else float("nan"),
    }


def strong_line_passes_inclusion(
    metrics: Mapping[str, float],
    cfg: StrongLineInclusionConfig | None = None,
) -> tuple[bool, str]:
    """Return (pass, reason). Empty reason when pass (#92)."""
    c = cfg or DEFAULT_INCLUSION
    depth = float(metrics.get("depth", np.nan))
    snr = float(metrics.get("snr", np.nan))
    width = float(metrics.get("width_kms", np.nan))
    err = float(metrics.get("err_kms", np.nan))
    rv = float(metrics.get("rv_kms", np.nan))
    tfrac = float(metrics.get("telluric_frac", np.nan))

    if not np.isfinite(rv) or abs(rv) > c.max_abs_rv_kms:
        return False, "bad_rv"
    if not np.isfinite(err) or err <= 0 or err > c.max_err_kms:
        return False, "bad_err"
    if not np.isfinite(depth) or depth < c.min_depth:
        return False, "shallow_depth"
    if not np.isfinite(snr) or snr < c.min_snr:
        return False, "low_snr"
    if not np.isfinite(width) or width < c.min_width_kms:
        return False, "too_narrow"
    if width > c.max_width_kms:
        return False, "too_broad"
    if np.isfinite(tfrac) and tfrac > c.max_telluric_frac:
        return False, "telluric"
    return True, ""


def estimate_strong_line_offsets_and_qualities(
    rows: Sequence[Mapping],
    *,
    mask_key: str = "mask_rv_kms",
    rv_key: str = "rv_kms",
    line_key: str = "line",
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Estimate per-line offsets and relative qualities from a campaign table.

    Procedure (#93 / user requirement):
      1. offset_L = median(RV_L − mask) for each line
      2. debiased residual = (RV_L − offset_L) − mask
      3. quality_L ∝ 1 / MAD(|debiased residual|)  (normalized so max = 1)

    Quality is how well the **debiased** line recovers the reference RV relative to
    other lines. It must not be inferred from raw (biased) RVs.
    """
    by_line: dict[str, list[float]] = {}
    for row in rows:
        name = str(row[line_key])
        rv = float(row[rv_key])
        mask = float(row[mask_key])
        if not np.isfinite(rv) or not np.isfinite(mask):
            continue
        by_line.setdefault(name, []).append(rv - mask)

    offsets: dict[str, float] = {}
    for name, diffs in by_line.items():
        arr = np.asarray(diffs, dtype=float)
        offsets[name] = float(np.median(arr))

    mads: dict[str, float] = {}
    for name, diffs in by_line.items():
        arr = np.asarray(diffs, dtype=float)
        resid = arr - float(offsets[name])  # debiased vs mask
        mad = float(np.median(np.abs(resid)))
        mads[name] = mad if mad > 1e-6 else 1e-6

    mad_ref = min(mads.values()) if mads else 1.0
    qualities = {name: float(mad_ref / mad) for name, mad in mads.items()}
    return offsets, qualities


def read_strong_line_calibration(path: Path | None) -> tuple[dict[str, float], dict[str, float]]:
    """
    Load per-line RV offsets (km/s) and species quality priors (#93).

    File format: ``line_name offset_kms [quality]`` (``#`` comments).
    Missing quality defaults to 1.0. Missing file → ({}, {}).
    """
    if path is None or not Path(path).is_file():
        return {}, {}
    offsets: dict[str, float] = {}
    qualities: dict[str, float] = {}
    for raw in Path(path).read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        name = str(parts[0])
        try:
            offsets[name] = float(parts[1])
        except ValueError:
            continue
        if len(parts) >= 3:
            try:
                q = float(parts[2])
            except ValueError:
                q = 1.0
            qualities[name] = q if np.isfinite(q) and q > 0 else 1.0
        else:
            qualities[name] = 1.0
    return offsets, qualities


def read_strong_line_offsets(path: Path | None) -> dict[str, float]:
    """Backward-compatible: offsets only."""
    offsets, _qualities = read_strong_line_calibration(path)
    return offsets


def combine_strong_line_rvs(
    measurements: Sequence[Mapping],
    offsets: Mapping[str, float] | None = None,
    *,
    qualities: Mapping[str, float] | None = None,
    min_snr: float = 0.5,
    default_quality: float = 1.0,
) -> dict:
    """
    Debias each line, then IVW with separated quality and S/N weights (#93).

    For each included measurement::

        rv_c = rv_raw − offset_line
        w    = Q_line × snr_at_line²

    - ``Q_line`` — species quality from **debiased** campaign residuals (how well the
      line recovers the correct RV after removing its systematic offset).
    - ``snr_at_line`` — per-exposure S/N at that line (from ``snr`` in the measurement;
      falls back to ``depth / err_kms`` if needed).

    Formal fit error alone is not used as the quality term.
    """
    off = dict(offsets or {})
    qual = dict(qualities or {})
    rvs: list[float] = []
    wts: list[float] = []
    used: list[str] = []
    details: list[dict] = []
    for m in measurements:
        if not bool(m.get("included", True)):
            continue
        name = str(m["line"])
        rv = float(m["rv_kms"])
        if not np.isfinite(rv):
            continue
        debias = float(off.get(name, 0.0))
        rv_c = rv - debias
        q = float(qual.get(name, default_quality))
        if not np.isfinite(q) or q <= 0:
            q = float(default_quality)
        snr = float(m.get("snr", np.nan))
        if not np.isfinite(snr) or snr <= 0:
            depth = float(m.get("depth", np.nan))
            err = float(m.get("err_kms", np.nan))
            if np.isfinite(depth) and depth > 0 and np.isfinite(err) and err > 0:
                snr = float(depth / err)
        if not np.isfinite(snr) or snr < float(min_snr):
            continue
        w = float(q) * float(snr) * float(snr)
        if not np.isfinite(w) or w <= 0:
            continue
        rvs.append(rv_c)
        wts.append(w)
        used.append(name)
        details.append(
            {
                "line": name,
                "rv_raw_kms": rv,
                "offset_kms": debias,
                "rv_debiased_kms": rv_c,
                "snr_at_line": snr,
                "quality": q,
                "weight": w,
                "err_kms": float(m.get("err_kms", np.nan)),
                "depth": float(m.get("depth", np.nan)),
            }
        )
    if not rvs:
        return {
            "rv_kms": float("nan"),
            "err_kms": float("nan"),
            "n_lines": 0,
            "lines_used": [],
            "details": [],
        }
    w_arr = np.asarray(wts, dtype=float)
    rv_arr = np.asarray(rvs, dtype=float)
    rv_ivw = float(np.sum(w_arr * rv_arr) / np.sum(w_arr))
    err_ivw = float(1.0 / np.sqrt(np.sum(w_arr)))
    return {
        "rv_kms": rv_ivw,
        "err_kms": err_ivw,
        "n_lines": int(len(used)),
        "lines_used": used,
        "details": details,
    }


def default_strong_line_offsets_path() -> Path:
    env = str(getattr(config, "STRONG_LINE_OFFSETS_FILE", "") or "").strip()
    if env:
        return Path(env)
    return Path(config.REPO_ROOT) / "calibration" / "strong_line_offsets.txt"
