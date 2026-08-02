#!/usr/bin/env python3
"""
Build epoch–epoch CCF relative-RV matrix for one Gaia ID (step 11b/c).

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.epoch_ccf_matrix \\
    --gaia-id 468391369318487040 \\
    --data-root /Users/rfoley/darkhunter/rvs/data \\
    --abs-diagnostics-glob '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/output/Gaia_DR3_468391369318487040_epoch_*_diagnostics.csv' \\
    --out-dir validation_output/epoch_ccf/468391369318487040

  Enrich diagnostics so cascade can use epoch_ccf_abs_fill (after strong_lines)::

  PYTHONPATH=. python -m validation.epoch_ccf_matrix \\
    --gaia-id 468391369318487040 \\
    --data-root /Users/rfoley/darkhunter/rvs/data \\
    --out-dir validation_output/epoch_ccf/468391369318487040 \\
    --abs-diagnostics-glob 'output/Gaia_DR3_468391369318487040_epoch_*_diagnostics.csv' \\
    --enrich-diagnostics-glob 'output/Gaia_DR3_468391369318487040_epoch_*_diagnostics.csv' \\
    --enrich-out-dir output
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import continuum, io_utils
from darkhunter_rv.epoch_ccf import (
    EpochPairCcfResult,
    abs_rel_delta_discordant,
    build_relative_matrix_from_pairs,
    combine_relative_and_absolute,
    epoch_pair_ccf,
    inflate_sigma_ij,
)

logger = logging.getLogger(__name__)

_EPOCH_RE = re.compile(r"^Gaia_DR3_(\d+)_epoch_(\d+)\.txt$")
_DIAG_EPOCH_RE = re.compile(r"Gaia_DR3_(\d+)_epoch_(\d+)_diagnostics\.csv$")


def discover_epoch_spectra(data_root: Path, gaia_id: str) -> list[tuple[int, Path]]:
    """
    Find ``Gaia_DR3_<id>_epoch_*.txt`` under ``data_root``.

    Returns
    -------
    list of (epoch_index, path) sorted by epoch index.
    """
    gid = str(gaia_id).strip()
    out: list[tuple[int, Path]] = []
    for path in sorted(data_root.glob(f"Gaia_DR3_{gid}_epoch_*.txt")):
        m = _EPOCH_RE.match(path.name)
        if not m or m.group(1) != gid:
            continue
        out.append((int(m.group(2)), path))
    out.sort(key=lambda t: t[0])
    return out


def spectrum_to_normalized_1d(
    spectrum_path: Path,
    *,
    wave_min: float | None = None,
    wave_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load APF-style multi-order spectrum; quick-normalize each order; stitch 1D.

    Parameters
    ----------
    spectrum_path
        Path to ``Gaia_DR3_*_epoch_*.txt``.
    wave_min, wave_max
        Optional Å cuts applied after stitch (inclusive).

    Returns
    -------
    wave, flux
        Continuum-normalized 1D arrays sorted by wavelength.

    Limitations
    -----------
    Overlapping orders are concatenated (duplicate λ regions kept). Telluric
    masking and production blaze continuum are out of scope for matrix v1.
    """
    _hdr, spec_data = io_utils.read_spectrum(str(spectrum_path))
    waves: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    for order in sorted(spec_data):
        w = np.asarray(spec_data[order]["wavelength"], float)
        f = np.asarray(spec_data[order]["flux"], float)
        ok = np.isfinite(w) & np.isfinite(f) & (f > 0)
        if int(np.sum(ok)) < 50:
            continue
        wn, fn = continuum.quick_normalize(w[ok], f[ok])
        waves.append(np.asarray(wn, float))
        fluxes.append(np.asarray(fn, float))
    if not waves:
        return np.array([], dtype=float), np.array([], dtype=float)
    wave = np.concatenate(waves)
    flux = np.concatenate(fluxes)
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    if wave_min is not None:
        m = wave >= float(wave_min)
        wave, flux = wave[m], flux[m]
    if wave_max is not None:
        m = wave <= float(wave_max)
        wave, flux = wave[m], flux[m]
    return wave, flux


def load_abs_from_diagnostics(
    diagnostics_glob: str,
    gaia_id: str,
    epoch_indices: list[int],
    *,
    method: str = "mask_ccf",
) -> tuple[np.ndarray, np.ndarray, list[Path | None]]:
    """
    Load exposure-level absolute RVs aligned to ``epoch_indices``.

    Prefers ``exposure_rv_kms`` / ``exposure_rv_err_kms`` on rows with
    ``method == method``. Missing epochs stay NaN.
    """
    gid = str(gaia_id).strip()
    paths = [Path(p) for p in sorted(glob.glob(diagnostics_glob))]
    by_epoch: dict[int, Path] = {}
    for path in paths:
        m = _DIAG_EPOCH_RE.search(path.name)
        if not m or m.group(1) != gid:
            continue
        by_epoch[int(m.group(2))] = path

    n = len(epoch_indices)
    abs_rv = np.full(n, np.nan)
    abs_sig = np.full(n, np.nan)
    used: list[Path | None] = [None] * n
    for k, ep in enumerate(epoch_indices):
        path = by_epoch.get(int(ep))
        if path is None or not path.is_file():
            continue
        df = pd.read_csv(path)
        if "method" in df.columns:
            sub = df.loc[df["method"].astype(str) == str(method)]
        else:
            sub = df
        if sub.empty:
            sub = df
        if "exposure_rv_kms" not in sub.columns:
            continue
        er = sub["exposure_rv_kms"].to_numpy(dtype=float)
        ee = (
            sub["exposure_rv_err_kms"].to_numpy(dtype=float)
            if "exposure_rv_err_kms" in sub.columns
            else np.full(len(sub), np.nan)
        )
        finite = np.isfinite(er) & np.isfinite(ee) & (ee > 0)
        if not np.any(finite):
            finite = np.isfinite(er)
            if not np.any(finite):
                continue
            abs_rv[k] = float(er[finite][0])
            abs_sig[k] = float(ee[finite][0]) if np.isfinite(ee[finite][0]) and ee[finite][0] > 0 else 1.0
        else:
            abs_rv[k] = float(er[finite][0])
            abs_sig[k] = float(ee[finite][0])
        used[k] = path
    return abs_rv, abs_sig, used


def compute_pair_matrix(
    waves: list[np.ndarray],
    fluxes: list[np.ndarray],
    *,
    rv_search_half_width_kms: float = 500.0,
    max_grid_points: int | None = 16384,
) -> tuple[dict[tuple[int, int], EpochPairCcfResult], list[dict]]:
    """
    Compute all pairs with ``i <= j`` (includes auto-correlation diagonal).

    Lower triangle filled later via ``build_relative_matrix_from_pairs`` using
    antisymmetry; diagonal pair results stored explicitly for QC.
    """
    n = len(waves)
    pairs: dict[tuple[int, int], EpochPairCcfResult] = {}
    rows: list[dict] = []
    for i in range(n):
        for j in range(i, n):
            res = epoch_pair_ccf(
                waves[i],
                fluxes[i],
                waves[j],
                fluxes[j],
                rv_search_half_width_kms=float(rv_search_half_width_kms),
                max_grid_points=max_grid_points,
            )
            if i == j:
                # Force auto-corr QC flag when indices match
                qc = dict(res.qc)
                qc["auto_correlation"] = True
                if np.isfinite(res.dv_kms):
                    qc["diag_near_zero"] = bool(
                        abs(res.dv_kms) < max(3.0 * (res.err_kms if np.isfinite(res.err_kms) else 1.0), 1.0)
                    )
                res = EpochPairCcfResult(
                    dv_kms=res.dv_kms,
                    err_kms=res.err_kms,
                    peak=res.peak,
                    width_kms=res.width_kms,
                    peak_snr=res.peak_snr,
                    fit_ok=res.fit_ok,
                    qc=qc,
                )
            pairs[(i, j)] = res
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "dv_kms": res.dv_kms,
                    "err_kms": res.err_kms,
                    "peak": res.peak,
                    "width_kms": res.width_kms,
                    "peak_snr": res.peak_snr,
                    "fit_ok": res.fit_ok,
                    "auto_correlation": bool(res.qc.get("auto_correlation", False)),
                    "qc_ok": bool(res.qc.get("ok", False)),
                }
            )
            if i != j:
                # Explicit antisymmetric long-form row for consumers
                rows.append(
                    {
                        "i": j,
                        "j": i,
                        "dv_kms": -res.dv_kms if np.isfinite(res.dv_kms) else np.nan,
                        "err_kms": res.err_kms,
                        "peak": res.peak,
                        "width_kms": res.width_kms,
                        "peak_snr": res.peak_snr,
                        "fit_ok": res.fit_ok,
                        "auto_correlation": False,
                        "qc_ok": bool(res.qc.get("ok", False)),
                    }
                )
    return pairs, rows


def run_matrix(
    *,
    gaia_id: str,
    data_root: Path,
    out_dir: Path,
    abs_diagnostics_glob: str | None = None,
    max_epochs: int | None = None,
    wave_min: float | None = None,
    wave_max: float | None = None,
    rv_search_half_width_kms: float = 500.0,
    max_grid_points: int | None = 16384,
    abs_method: str = "mask_ccf",
    sigma_ij_scale: float = 1.0,
    discord_n_sigma: float = 3.0,
) -> dict:
    """
    Build and persist epoch CCF matrix (+ optional abs fill) for one star.

    Writes ``epoch_ccf_pairs.csv``, ``epoch_ccf_matrix.npz``, optional
    ``epoch_ccf_abs_fill.csv``, ``epoch_ccf_vs_abs_delta.csv`` (with discordant
    flags when abs anchors exist), and ``epoch_ccf_meta.json``.

    ``sigma_ij_scale`` multiplies off-diagonal formal ``sigma_ij`` (short-pair
    inflation from step 05a; default 1.0 = no change).
    ``discord_n_sigma`` thresholds abs−abs vs CCF ΔRV inconsistency flags.
    """
    epochs = discover_epoch_spectra(data_root, gaia_id)
    if not epochs:
        raise FileNotFoundError(f"No Gaia_DR3_{gaia_id}_epoch_*.txt under {data_root}")
    if max_epochs is not None and max_epochs > 0:
        epochs = epochs[: int(max_epochs)]

    epoch_indices = [e for e, _ in epochs]
    paths = [p for _, p in epochs]
    waves: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    for path in paths:
        w, f = spectrum_to_normalized_1d(path, wave_min=wave_min, wave_max=wave_max)
        if len(w) < 256:
            raise RuntimeError(f"Too few pixels after normalize/cut: {path}")
        waves.append(w)
        fluxes.append(f)

    pairs, long_rows = compute_pair_matrix(
        waves,
        fluxes,
        rv_search_half_width_kms=rv_search_half_width_kms,
        max_grid_points=max_grid_points,
    )
    # Matrix assembly uses off-diagonal upper triangle only
    off_diag = {k: v for k, v in pairs.items() if k[0] != k[1]}
    dv, sig = build_relative_matrix_from_pairs(len(epochs), off_diag)
    sig = inflate_sigma_ij(sig, float(sigma_ij_scale))
    # Overlay measured diagonal (auto-corr) for QC storage
    diag_dv = np.zeros(len(epochs))
    for i in range(len(epochs)):
        res = pairs[(i, i)]
        diag_dv[i] = float(res.dv_kms) if np.isfinite(res.dv_kms) else np.nan
        dv[i, i] = diag_dv[i] if np.isfinite(diag_dv[i]) else 0.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_df = pd.DataFrame(long_rows)
    pairs_df.insert(0, "gaia_id", str(gaia_id))
    pairs_df.insert(1, "epoch_i", [epoch_indices[int(r["i"])] for r in long_rows])
    pairs_df.insert(2, "epoch_j", [epoch_indices[int(r["j"])] for r in long_rows])
    fac = max(1.0, float(sigma_ij_scale))
    if fac != 1.0 and "err_kms" in pairs_df.columns and "i" in pairs_df.columns:
        i_arr = pairs_df["i"].to_numpy(dtype=int)
        j_arr = pairs_df["j"].to_numpy(dtype=int)
        errs = pairs_df["err_kms"].to_numpy(dtype=float).copy()
        ok = (i_arr != j_arr) & np.isfinite(errs) & (errs > 0)
        errs[ok] *= fac
        pairs_df["err_kms"] = errs
    pairs_csv = out_dir / "epoch_ccf_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)

    abs_rv = np.full(len(epochs), np.nan)
    abs_sig = np.full(len(epochs), np.nan)
    abs_paths: list[str | None] = [None] * len(epochs)
    fill_path = None
    compare_rows: list[dict] = []
    fill_result = None
    meta_discord = {"n_abs_rel_pairs": 0, "n_discordant": 0}
    if abs_diagnostics_glob:
        abs_rv, abs_sig, used = load_abs_from_diagnostics(
            abs_diagnostics_glob,
            gaia_id,
            epoch_indices,
            method=abs_method,
        )
        abs_paths = [str(p) if p is not None else None for p in used]
        fill_result = combine_relative_and_absolute(dv, sig, abs_rv, abs_sig)
        fill_df = pd.DataFrame(
            {
                "gaia_id": str(gaia_id),
                "matrix_index": np.arange(len(epochs)),
                "epoch": epoch_indices,
                "spectrum": [str(p) for p in paths],
                "abs_rv_kms": abs_rv,
                "abs_err_kms": abs_sig,
                "v_hat_kms": fill_result.v_hat_kms,
                "sigma_kms": fill_result.sigma_kms,
                "n_abs_anchors": fill_result.n_abs_anchors,
                "float_zeropoint": fill_result.float_zeropoint,
                "relative_only": fill_result.relative_only,
                "epoch_ccf_rel": fill_result.v_hat_kms,  # 11d light product tag
                "epoch_ccf_abs_fill": np.where(
                    fill_result.relative_only,
                    np.nan,
                    fill_result.v_hat_kms,
                ),
            }
        )
        fill_path = out_dir / "epoch_ccf_abs_fill.csv"
        fill_df.to_csv(fill_path, index=False)

        # Compare Δv_ij to A_i - A_j when both abs finite; flag systematics discord
        for i in range(len(epochs)):
            for j in range(i + 1, len(epochs)):
                if not (np.isfinite(abs_rv[i]) and np.isfinite(abs_rv[j])):
                    continue
                if not np.isfinite(dv[i, j]):
                    continue
                dv_abs = float(abs_rv[i] - abs_rv[j])
                disc = abs_rel_delta_discordant(
                    float(dv[i, j]),
                    dv_abs,
                    err_ccf_kms=float(sig[i, j]) if np.isfinite(sig[i, j]) else float("nan"),
                    err_abs_i_kms=float(abs_sig[i]) if np.isfinite(abs_sig[i]) else float("nan"),
                    err_abs_j_kms=float(abs_sig[j]) if np.isfinite(abs_sig[j]) else float("nan"),
                    n_sigma=float(discord_n_sigma),
                )
                compare_rows.append(
                    {
                        "epoch_i": epoch_indices[i],
                        "epoch_j": epoch_indices[j],
                        "dv_ccf_kms": float(dv[i, j]),
                        "dv_abs_kms": dv_abs,
                        "residual_kms": float(disc["residual_kms"]),
                        "err_ccf_kms": float(sig[i, j]) if np.isfinite(sig[i, j]) else np.nan,
                        "sigma_combined_kms": float(disc["sigma_combined_kms"]),
                        "n_sigma_residual": float(disc["n_sigma_residual"]),
                        "epoch_ccf_abs_rel_discordant": bool(disc["discordant"]),
                        "discord_n_sigma": float(discord_n_sigma),
                    }
                )
        if compare_rows:
            pd.DataFrame(compare_rows).to_csv(out_dir / "epoch_ccf_vs_abs_delta.csv", index=False)
            n_disc = sum(1 for r in compare_rows if r["epoch_ccf_abs_rel_discordant"])
            meta_discord = {"n_abs_rel_pairs": len(compare_rows), "n_discordant": n_disc}
        else:
            meta_discord = {"n_abs_rel_pairs": 0, "n_discordant": 0}
    else:
        # Zero-anchor relative-only product (11c wire)
        fill_result = combine_relative_and_absolute(dv, sig, abs_rv, abs_sig)
        fill_df = pd.DataFrame(
            {
                "gaia_id": str(gaia_id),
                "matrix_index": np.arange(len(epochs)),
                "epoch": epoch_indices,
                "spectrum": [str(p) for p in paths],
                "abs_rv_kms": abs_rv,
                "abs_err_kms": abs_sig,
                "v_hat_kms": fill_result.v_hat_kms,
                "sigma_kms": fill_result.sigma_kms,
                "n_abs_anchors": fill_result.n_abs_anchors,
                "float_zeropoint": fill_result.float_zeropoint,
                "relative_only": fill_result.relative_only,
                "epoch_ccf_rel": fill_result.v_hat_kms,
                "epoch_ccf_abs_fill": np.full(len(epochs), np.nan),
            }
        )
        fill_path = out_dir / "epoch_ccf_abs_fill.csv"
        fill_df.to_csv(fill_path, index=False)

    npz_path = out_dir / "epoch_ccf_matrix.npz"
    np.savez_compressed(
        npz_path,
        dv_kms=dv,
        sigma_kms=sig,
        diag_dv_kms=diag_dv,
        epoch_indices=np.asarray(epoch_indices, dtype=int),
        abs_rv_kms=abs_rv,
        abs_sigma_kms=abs_sig,
        spectrum_paths=np.asarray([str(p) for p in paths]),
    )

    diag_abs = [abs(float(x)) for x in diag_dv if np.isfinite(x)]
    meta = {
        "gaia_id": str(gaia_id),
        "n_epochs": len(epochs),
        "epoch_indices": epoch_indices,
        "spectrum_paths": [str(p) for p in paths],
        "pairs_csv": str(pairs_csv),
        "npz_path": str(npz_path),
        "fill_csv": str(fill_path) if fill_path else None,
        "abs_diagnostics_glob": abs_diagnostics_glob,
        "abs_paths": abs_paths,
        "n_abs_anchors": int(fill_result.n_abs_anchors) if fill_result else 0,
        "float_zeropoint": bool(fill_result.float_zeropoint) if fill_result else True,
        "diag_abs_max_kms": float(max(diag_abs)) if diag_abs else None,
        "diag_abs_median_kms": float(np.median(diag_abs)) if diag_abs else None,
        "n_abs_delta_comparisons": len(compare_rows),
        "n_abs_rel_discordant": int(meta_discord["n_discordant"]),
        "discord_n_sigma": float(discord_n_sigma),
        "abs_delta_residual_rms_kms": (
            float(np.sqrt(np.mean(np.square([r["residual_kms"] for r in compare_rows]))))
            if compare_rows
            else None
        ),
        "wave_min": wave_min,
        "wave_max": wave_max,
        "max_grid_points": max_grid_points,
        "rv_search_half_width_kms": rv_search_half_width_kms,
        "sigma_ij_scale": float(max(1.0, float(sigma_ij_scale))),
        "note": (
            "Cascade may adopt epoch_ccf_abs_fill after strong_lines when abs-anchored; "
            "always flag abs vs relative ΔRV discord (systematics test)."
        ),
    }
    meta_path = out_dir / "epoch_ccf_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(
        "Wrote %s (%d epochs); diag|dv| med=%.3f max=%.3f; abs_anchors=%s",
        pairs_csv,
        len(epochs),
        meta["diag_abs_median_kms"] or float("nan"),
        meta["diag_abs_max_kms"] or float("nan"),
        meta["n_abs_anchors"],
    )
    return meta



def load_epoch_ccf_fill_by_epoch(fill_csv: Path) -> dict[int, pd.Series]:
    """
    Load ``epoch_ccf_abs_fill.csv`` keyed by integer epoch index.

    Parameters
    ----------
    fill_csv
        Path written by :func:`run_matrix`.

    Returns
    -------
    dict
        ``epoch ->`` fill row (Series). Empty if file missing or empty.

    Limitations
    -----------
    Expects columns ``epoch``, ``epoch_ccf_rel``, ``epoch_ccf_abs_fill``;
    ``sigma_kms`` optional. Does not alter adopted-RV cascade.
    """
    path = Path(fill_csv)
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if df.empty or "epoch" not in df.columns:
        return {}
    out: dict[int, pd.Series] = {}
    for _, row in df.iterrows():
        try:
            ep = int(row["epoch"])
        except (TypeError, ValueError):
            continue
        out[ep] = row
    return out


def attach_epoch_ccf_fill_columns(
    diagnostics: pd.DataFrame,
    fill_row: pd.Series | None,
) -> pd.DataFrame:
    """
    Return a copy of ``diagnostics`` with epoch-CCF product columns.

    Columns added (NaN when ``fill_row`` is None or value missing):
    ``epoch_ccf_rel``, ``epoch_ccf_abs_fill``, ``epoch_ccf_sigma_kms``,
    ``epoch_ccf_n_abs_anchors``, ``epoch_ccf_relative_only``.

    Abs-anchored ``epoch_ccf_abs_fill`` method rows (via
    :func:`epoch_ccf_product_method_rows`) feed the adopt cascade after
    ``strong_lines``. Relative-only fills are not adopted.
    """
    out = diagnostics.copy()
    rel = float("nan")
    abs_fill = float("nan")
    sig = float("nan")
    n_anch = float("nan")
    rel_only = True
    if fill_row is not None:
        if "epoch_ccf_rel" in fill_row.index:
            rel = float(fill_row["epoch_ccf_rel"])
        elif "v_hat_kms" in fill_row.index:
            rel = float(fill_row["v_hat_kms"])
        if "epoch_ccf_abs_fill" in fill_row.index:
            abs_fill = float(fill_row["epoch_ccf_abs_fill"])
        if "sigma_kms" in fill_row.index:
            sig = float(fill_row["sigma_kms"])
        if "n_abs_anchors" in fill_row.index:
            try:
                n_anch = float(fill_row["n_abs_anchors"])
            except (TypeError, ValueError):
                n_anch = float("nan")
        if "relative_only" in fill_row.index:
            rel_only = bool(fill_row["relative_only"])
    out["epoch_ccf_rel"] = rel
    out["epoch_ccf_abs_fill"] = abs_fill
    out["epoch_ccf_sigma_kms"] = sig
    out["epoch_ccf_n_abs_anchors"] = n_anch
    out["epoch_ccf_relative_only"] = rel_only
    return out


def epoch_ccf_product_method_rows(fill_row: pd.Series) -> list[dict]:
    """
    Build diagnostics-style method rows for epoch-CCF products.

    Methods ``epoch_ccf_rel`` and ``epoch_ccf_abs_fill`` use ``chunk_key=all``.
    ``epoch_ccf_abs_fill`` is consumed by
    :func:`darkhunter_rv.method_evaluation.recommend_adopted_rv` as the tier
    after ``strong_lines`` when abs-anchored. ``epoch_ccf_rel`` is recorded for
    diagnostics only (not in the adopt cascade).

    Parameters
    ----------
    fill_row
        One row from ``epoch_ccf_abs_fill.csv``.

    Returns
    -------
    list of dict
        Zero, one, or two rows (abs fill omitted when relative-only / non-finite).
    """
    rows: list[dict] = []
    rel = float(fill_row["epoch_ccf_rel"]) if "epoch_ccf_rel" in fill_row.index else float(
        fill_row.get("v_hat_kms", float("nan"))
    )
    sig = float(fill_row["sigma_kms"]) if "sigma_kms" in fill_row.index else float("nan")
    qc = bool(np.isfinite(rel) and np.isfinite(sig) and sig > 0)
    base = {
        "chunk_key": "all",
        "rv_kms": rel,
        "rv_err_kms": sig if np.isfinite(sig) else float("nan"),
        "exposure_rv_kms": rel,
        "exposure_rv_err_kms": sig if np.isfinite(sig) else float("nan"),
        "qc_pass": qc,
        "qc_reason": "epoch_ccf_product" if qc else "epoch_ccf_nonfinite",
    }
    rows.append({**base, "method": "epoch_ccf_rel"})
    abs_fill = (
        float(fill_row["epoch_ccf_abs_fill"])
        if "epoch_ccf_abs_fill" in fill_row.index
        else float("nan")
    )
    rel_only = bool(fill_row["relative_only"]) if "relative_only" in fill_row.index else True
    if (not rel_only) and np.isfinite(abs_fill):
        rows.append(
            {
                **base,
                "method": "epoch_ccf_abs_fill",
                "rv_kms": abs_fill,
                "exposure_rv_kms": abs_fill,
                "qc_pass": bool(np.isfinite(abs_fill) and np.isfinite(sig) and sig > 0),
                "qc_reason": "epoch_ccf_abs_fill" if np.isfinite(abs_fill) else "epoch_ccf_nonfinite",
            }
        )
    return rows


def enrich_diagnostics_with_epoch_ccf_fill(
    *,
    diagnostics_glob: str,
    fill_csv: Path,
    out_dir: Path,
    append_method_rows: bool = True,
) -> list[Path]:
    """
    Opt-in post-process: copy diagnostics CSVs and attach epoch-CCF product fields.

    When matrix artifacts exist, matches ``Gaia_DR3_<id>_epoch_<N>_diagnostics.csv``
    to fill rows by epoch index. Writes enriched files under ``out_dir`` (same
    basenames). Does **not** overwrite inputs unless ``out_dir`` equals the
    source directory. Abs-anchored fill method rows enable cascade adoption
    after ``strong_lines`` on the next pipeline / overlap pass.

    Parameters
    ----------
    diagnostics_glob
        Glob of ``*_diagnostics.csv`` files.
    fill_csv
        ``epoch_ccf_abs_fill.csv`` from :func:`run_matrix`.
    out_dir
        Destination directory for enriched copies.
    append_method_rows
        If True, append ``epoch_ccf_rel`` / ``epoch_ccf_abs_fill`` method rows.

    Returns
    -------
    list of Path
        Written enriched diagnostics paths.
    """
    by_ep = load_epoch_ccf_fill_by_epoch(Path(fill_csv))
    if not by_ep:
        raise FileNotFoundError(f"No usable epoch fill table at {fill_csv}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path_str in sorted(glob.glob(str(diagnostics_glob))):
        path = Path(path_str)
        m = _DIAG_EPOCH_RE.search(path.name)
        if not m:
            continue
        ep = int(m.group(2))
        fill_row = by_ep.get(ep)
        df = pd.read_csv(path)
        enriched = attach_epoch_ccf_fill_columns(df, fill_row)
        if append_method_rows and fill_row is not None:
            extra = epoch_ccf_product_method_rows(fill_row)
            if extra:
                enriched = pd.concat([enriched, pd.DataFrame(extra)], ignore_index=True)
        dest = out_dir / path.name
        enriched.to_csv(dest, index=False)
        written.append(dest)
    return written



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gaia-id", required=True, help="Gaia DR3 source id")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/rfoley/darkhunter/rvs/data"),
        help="Directory with Gaia_DR3_*_epoch_*.txt",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for CSV/npz/meta",
    )
    p.add_argument(
        "--abs-diagnostics-glob",
        default=None,
        help="Optional glob of *_diagnostics.csv for absolute anchors / fill",
    )
    p.add_argument(
        "--abs-method",
        default="mask_ccf",
        help="Diagnostics method row used for exposure_rv_kms (default mask_ccf)",
    )
    p.add_argument("--max-epochs", type=int, default=None, help="Use first N epochs only")
    p.add_argument("--wave-min", type=float, default=None, help="Optional Å lower cut")
    p.add_argument("--wave-max", type=float, default=None, help="Optional Å upper cut")
    p.add_argument("--rv-search-half-width-kms", type=float, default=500.0)
    p.add_argument(
        "--max-grid-points",
        type=int,
        default=16384,
        help="Cap log-λ FFT length (power of two). Use 0 for uncapped.",
    )
    p.add_argument(
        "--sigma-ij-scale",
        type=float,
        default=1.0,
        help="Multiply off-diagonal sigma_ij (short-pair inflation; default 1)",
    )
    p.add_argument(
        "--discord-n-sigma",
        type=float,
        default=3.0,
        help="Flag abs vs relative ΔRV when |residual| > N * σ_combined (default 3)",
    )
    p.add_argument(
        "--enrich-diagnostics-glob",
        default=None,
        help=(
            "Glob of *_diagnostics.csv to copy with epoch_ccf_rel / "
            "epoch_ccf_abs_fill columns (and method rows for cascade). Requires fill CSV."
        ),
    )
    p.add_argument(
        "--enrich-out-dir",
        type=Path,
        default=None,
        help="Directory for enriched diagnostics copies (required with --enrich-diagnostics-glob)",
    )
    p.add_argument(
        "--enrich-no-method-rows",
        action="store_true",
        help="With --enrich-diagnostics-glob, attach columns only (no epoch_ccf_* method rows)",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    max_grid = None if int(args.max_grid_points) <= 0 else int(args.max_grid_points)
    meta = run_matrix(
        gaia_id=str(args.gaia_id),
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        abs_diagnostics_glob=args.abs_diagnostics_glob,
        max_epochs=args.max_epochs,
        wave_min=args.wave_min,
        wave_max=args.wave_max,
        rv_search_half_width_kms=float(args.rv_search_half_width_kms),
        max_grid_points=max_grid,
        abs_method=str(args.abs_method),
        sigma_ij_scale=float(args.sigma_ij_scale),
        discord_n_sigma=float(args.discord_n_sigma),
    )
    enrich_paths: list[str] = []
    if args.enrich_diagnostics_glob:
        if args.enrich_out_dir is None:
            raise SystemExit("--enrich-out-dir is required with --enrich-diagnostics-glob")
        fill_csv = meta.get("fill_csv")
        if not fill_csv:
            raise SystemExit("No fill CSV in matrix meta; cannot enrich diagnostics")
        written = enrich_diagnostics_with_epoch_ccf_fill(
            diagnostics_glob=str(args.enrich_diagnostics_glob),
            fill_csv=Path(fill_csv),
            out_dir=Path(args.enrich_out_dir),
            append_method_rows=not bool(args.enrich_no_method_rows),
        )
        enrich_paths = [str(p) for p in written]
        meta["enriched_diagnostics"] = enrich_paths
        logger.info("Enriched %d diagnostics -> %s", len(written), args.enrich_out_dir)

    payload = {k: meta[k] for k in (
        "gaia_id",
        "n_epochs",
        "pairs_csv",
        "npz_path",
        "fill_csv",
        "n_abs_anchors",
        "diag_abs_median_kms",
        "diag_abs_max_kms",
        "n_abs_delta_comparisons",
        "abs_delta_residual_rms_kms",
    )}
    if enrich_paths:
        payload["enriched_diagnostics_n"] = len(enrich_paths)
        payload["enriched_diagnostics_dir"] = str(args.enrich_out_dir)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
