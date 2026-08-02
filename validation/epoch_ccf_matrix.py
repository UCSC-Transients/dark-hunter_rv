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
    build_relative_matrix_from_pairs,
    combine_relative_and_absolute,
    epoch_pair_ccf,
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
) -> dict:
    """
    Build and persist epoch CCF matrix (+ optional abs fill) for one star.

    Writes ``epoch_ccf_pairs.csv``, ``epoch_ccf_matrix.npz``, optional
    ``epoch_ccf_abs_fill.csv``, and ``epoch_ccf_meta.json``.
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
    pairs_csv = out_dir / "epoch_ccf_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)

    abs_rv = np.full(len(epochs), np.nan)
    abs_sig = np.full(len(epochs), np.nan)
    abs_paths: list[str | None] = [None] * len(epochs)
    fill_path = None
    compare_rows: list[dict] = []
    fill_result = None
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

        # Compare Δv_ij to A_i - A_j when both abs finite
        for i in range(len(epochs)):
            for j in range(i + 1, len(epochs)):
                if not (np.isfinite(abs_rv[i]) and np.isfinite(abs_rv[j])):
                    continue
                if not np.isfinite(dv[i, j]):
                    continue
                compare_rows.append(
                    {
                        "epoch_i": epoch_indices[i],
                        "epoch_j": epoch_indices[j],
                        "dv_ccf_kms": float(dv[i, j]),
                        "dv_abs_kms": float(abs_rv[i] - abs_rv[j]),
                        "residual_kms": float(dv[i, j] - (abs_rv[i] - abs_rv[j])),
                        "err_ccf_kms": float(sig[i, j]) if np.isfinite(sig[i, j]) else np.nan,
                    }
                )
        if compare_rows:
            pd.DataFrame(compare_rows).to_csv(out_dir / "epoch_ccf_vs_abs_delta.csv", index=False)
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
        "abs_delta_residual_rms_kms": (
            float(np.sqrt(np.mean(np.square([r["residual_kms"] for r in compare_rows]))))
            if compare_rows
            else None
        ),
        "wave_min": wave_min,
        "wave_max": wave_max,
        "max_grid_points": max_grid_points,
        "rv_search_half_width_kms": rv_search_half_width_kms,
        "note": "Not default adopted RV; relative/fill product only (step 11).",
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
    )
    print(json.dumps({k: meta[k] for k in (
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
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
