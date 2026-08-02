#!/usr/bin/env python3
"""
Find short-Δt epoch pairs and QC absolute + epoch-CCF ΔRV (step 05a).

Closely spaced epochs (default: |Δt| < 1 day ≈ same night) should have
true ΔRV ≈ 0. This tool reports absolute-method and epoch–epoch CCF
differences, flags violations, and recommends a global ``sigma_ij`` inflation
factor for :mod:`validation.epoch_ccf_matrix`.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv/.worktrees/shortpair-05a
  PYTHONPATH=. python -m validation.find_short_pairs \\
    --diagnostics-glob '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/output/Gaia_DR3_*_epoch_*_diagnostics.csv' \\
    --data-root /Users/rfoley/darkhunter/rvs/data \\
    --epoch-ccf-root /Users/rfoley/darkhunter/rvs/dark-hunter_rv/.worktrees/epoch-matrix/validation_output/epoch_ccf \\
    --out-dir validation_output/short_pair_qc \\
    --max-delta-days 1
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv.epoch_ccf import epoch_pair_ccf
from darkhunter_rv.pipeline import _weighted_method_rv_from_rows
from validation.epoch_ccf_matrix import spectrum_to_normalized_1d

logger = logging.getLogger(__name__)

_DIAG_EPOCH_RE = re.compile(
    r"Gaia_DR3_(\d+)_epoch_(\d+)_diagnostics\.csv$",
    re.IGNORECASE,
)

ABS_METHODS: tuple[str, ...] = ("mask_ccf", "template_fft", "strong_lines")


@dataclass(frozen=True)
class EpochRecord:
    """One spectrum epoch with diagnostics path and MJD."""

    gaia_id: str
    epoch: int
    diagnostics_path: Path
    mjd: float
    spectrum_path: str | None


def _mjd_from_diagnostics(path: Path) -> float:
    """Read first finite MJD from a diagnostics CSV."""
    df = pd.read_csv(path, usecols=lambda c: c == "mjd")
    if "mjd" not in df.columns or df.empty:
        return float("nan")
    vals = pd.to_numeric(df["mjd"], errors="coerce").to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    return float(finite[0]) if len(finite) else float("nan")


def _spectrum_path_from_diagnostics(path: Path) -> str | None:
    """Return spectrum ``file`` column from diagnostics when present."""
    df = pd.read_csv(path, nrows=5)
    if "file" not in df.columns or df.empty:
        return None
    val = df["file"].dropna()
    return str(val.iloc[0]) if len(val) else None


def discover_epochs(
    diagnostics_glob: str,
    *,
    gaia_id: str | None = None,
) -> list[EpochRecord]:
    """
    Discover multi-epoch diagnostics and attach MJD / spectrum paths.

    Parameters
    ----------
    diagnostics_glob
        Glob for ``*_diagnostics.csv`` (typically under ``output/``).
    gaia_id
        Optional filter to one Gaia DR3 source id.

    Returns
    -------
    list[EpochRecord]
        Sorted by ``(gaia_id, epoch)``. Epochs missing MJD are omitted.
    """
    out: list[EpochRecord] = []
    for path_s in sorted(glob.glob(diagnostics_glob)):
        path = Path(path_s)
        m = _DIAG_EPOCH_RE.search(path.name)
        if not m:
            continue
        gid, ep = m.group(1), int(m.group(2))
        if gaia_id is not None and str(gaia_id).strip() != gid:
            continue
        mjd = _mjd_from_diagnostics(path)
        if not np.isfinite(mjd):
            logger.debug("Skip %s: no MJD", path)
            continue
        out.append(
            EpochRecord(
                gaia_id=gid,
                epoch=ep,
                diagnostics_path=path,
                mjd=mjd,
                spectrum_path=_spectrum_path_from_diagnostics(path),
            )
        )
    out.sort(key=lambda r: (r.gaia_id, r.epoch))
    return out


def find_pairs(
    epochs: list[EpochRecord],
    *,
    max_delta_days: float = 1.0,
    same_calendar_night: bool = False,
) -> list[tuple[EpochRecord, EpochRecord, float]]:
    """
    Enumerate unique epoch pairs with |Δt| below threshold.

    Parameters
    ----------
    epochs
        Per-epoch records (may include many stars).
    max_delta_days
        Maximum |MJD_i - MJD_j| in days (default 1 ≈ same night window).
    same_calendar_night
        If True, also require ``floor(mjd)`` equality (stricter same-night).

    Returns
    -------
    list[tuple[EpochRecord, EpochRecord, float]]
        ``(earlier, later, delta_days)`` with ``delta_days >= 0``.
    """
    by_star: dict[str, list[EpochRecord]] = defaultdict(list)
    for rec in epochs:
        by_star[rec.gaia_id].append(rec)

    pairs: list[tuple[EpochRecord, EpochRecord, float]] = []
    thresh = float(max_delta_days)
    for gid, recs in by_star.items():
        recs = sorted(recs, key=lambda r: (r.mjd, r.epoch))
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                dt = float(b.mjd - a.mjd)
                if dt > thresh:
                    # Sorted by MJD: further j only increase Δt
                    break
                if same_calendar_night and int(math.floor(a.mjd)) != int(math.floor(b.mjd)):
                    continue
                pairs.append((a, b, dt))
    return pairs


def method_exposure_rv(diagnostics_path: Path, method: str) -> tuple[float, float]:
    """
    Exposure-level absolute RV for one method via pipeline stacker.

    Uses :func:`darkhunter_rv.pipeline._weighted_method_rv_from_rows` so
    mask / template / strong stacks match production offsets tooling.
    """
    df = pd.read_csv(diagnostics_path)
    rows = df.to_dict(orient="records")
    return _weighted_method_rv_from_rows(rows, method)


def load_epoch_ccf_dv(
    epoch_ccf_root: Path | None,
    gaia_id: str,
    epoch_i: int,
    epoch_j: int,
) -> tuple[float, float]:
    """
    Look up ``dv_kms`` / ``err_kms`` from an existing step-11 pairs CSV.

    Returns
    -------
    tuple[float, float]
        ``(dv, err)`` with NaNs when missing. Preference: ``epoch_i < epoch_j``
        row matching ``(epoch_i, epoch_j)``; else antisymmetric of reverse.
    """
    if epoch_ccf_root is None:
        return float("nan"), float("nan")
    pairs_path = Path(epoch_ccf_root) / str(gaia_id) / "epoch_ccf_pairs.csv"
    if not pairs_path.is_file():
        return float("nan"), float("nan")
    df = pd.read_csv(pairs_path)
    if df.empty or "epoch_i" not in df.columns:
        return float("nan"), float("nan")
    sub = df[(df["epoch_i"] == int(epoch_i)) & (df["epoch_j"] == int(epoch_j))]
    if sub.empty:
        sub = df[(df["epoch_i"] == int(epoch_j)) & (df["epoch_j"] == int(epoch_i))]
        if sub.empty:
            return float("nan"), float("nan")
        dv = -float(sub.iloc[0]["dv_kms"])
        err = float(sub.iloc[0]["err_kms"]) if "err_kms" in sub.columns else float("nan")
        return dv, err
    dv = float(sub.iloc[0]["dv_kms"])
    err = float(sub.iloc[0]["err_kms"]) if "err_kms" in sub.columns else float("nan")
    return dv, err


def compute_epoch_ccf_dv(
    spectrum_i: Path,
    spectrum_j: Path,
    *,
    rv_search_half_width_kms: float = 500.0,
    max_grid_points: int | None = 16384,
) -> tuple[float, float]:
    """
    Compute epoch–epoch CCF Δv for one short pair from spectrum files.

    Returns
    -------
    tuple[float, float]
        ``(dv_kms, err_kms)`` with ``dv ≈ v_i - v_j``.
    """
    wi, fi = spectrum_to_normalized_1d(spectrum_i)
    wj, fj = spectrum_to_normalized_1d(spectrum_j)
    res = epoch_pair_ccf(
        wi,
        fi,
        wj,
        fj,
        rv_search_half_width_kms=float(rv_search_half_width_kms),
        max_grid_points=max_grid_points,
    )
    return float(res.dv_kms), float(res.err_kms)


def _flag_violation(
    delta: float,
    err: float,
    *,
    abs_thresh_kms: float,
    n_sigma: float,
) -> bool:
    """True when |ΔRV| exceeds absolute floor or n_sigma × combined error."""
    if not np.isfinite(delta):
        return False
    ad = abs(float(delta))
    if ad >= float(abs_thresh_kms):
        return True
    if np.isfinite(err) and err > 0 and ad >= float(n_sigma) * float(err):
        return True
    return False


def recommend_sigma_ij_scale(
    deltas: np.ndarray,
    formal_errs: np.ndarray,
    *,
    floor: float = 1.0,
    outlier_abs_kms: float = 50.0,
) -> dict[str, float]:
    """
    Recommend global ``sigma_ij`` inflation from short-pair epoch-CCF scatter.

    Default recommendation uses a **robust** scale
    ``max(floor, 1.4826 * MAD / median(σ_formal))`` so a few failed CCFs
    do not dominate. Also reports RMS-based and clipped-RMS scales for ops.

    Parameters
    ----------
    deltas
        Epoch-CCF ΔRV on short pairs (km/s); NaNs ignored.
    formal_errs
        Matching formal ``err_kms``; NaNs / non-positive ignored for median σ.
    floor
        Minimum scale factor (never deflate).
    outlier_abs_kms
        |Δv| clip for ``rms_clipped`` diagnostics (default 50 km/s).

    Returns
    -------
    dict
        ``recommended_sigma_ij_scale`` (MAD-based), plus RMS / clipped helpers.
    """
    d = np.asarray(deltas, float)
    e = np.asarray(formal_errs, float)
    ok_d = np.isfinite(d)
    n = int(np.sum(ok_d))
    empty = {
        "n": 0.0,
        "n_clipped": 0.0,
        "rms_kms": float("nan"),
        "rms_clipped_kms": float("nan"),
        "mad_kms": float("nan"),
        "sigma_robust_kms": float("nan"),
        "median_formal_err_kms": float("nan"),
        "scale_rms": float(floor),
        "scale_rms_clipped": float(floor),
        "recommended_sigma_ij_scale": float(floor),
    }
    if n == 0:
        return empty
    vals = d[ok_d]
    errs = e[ok_d]
    rms = float(np.sqrt(np.mean(vals**2)))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma_rob = float(1.4826 * mad)
    clip = np.abs(vals) <= float(outlier_abs_kms)
    n_clip = int(np.sum(clip))
    rms_clip = float(np.sqrt(np.mean(vals[clip] ** 2))) if n_clip else float("nan")
    ok_e = np.isfinite(errs) & (errs > 0)
    med_e = float(np.median(errs[ok_e])) if np.any(ok_e) else float("nan")

    def _scale(num: float) -> float:
        if not (np.isfinite(num) and np.isfinite(med_e) and med_e > 0):
            return float(floor)
        return float(max(float(floor), num / med_e))

    scale_mad = _scale(sigma_rob)
    scale_rms = _scale(rms)
    scale_clip = _scale(rms_clip) if n_clip else float(floor)
    return {
        "n": float(n),
        "n_clipped": float(n_clip),
        "rms_kms": rms,
        "rms_clipped_kms": rms_clip,
        "mad_kms": mad,
        "sigma_robust_kms": sigma_rob,
        "median_formal_err_kms": med_e,
        "scale_rms": scale_rms,
        "scale_rms_clipped": scale_clip,
        "recommended_sigma_ij_scale": scale_mad,
    }


def build_short_pair_table(
    pairs: list[tuple[EpochRecord, EpochRecord, float]],
    *,
    methods: tuple[str, ...] = ABS_METHODS,
    epoch_ccf_root: Path | None = None,
    compute_epoch_ccf: bool = False,
    data_root: Path | None = None,
    abs_thresh_kms: float = 5.0,
    n_sigma: float = 3.0,
    rv_search_half_width_kms: float = 500.0,
    max_grid_points: int | None = 16384,
) -> pd.DataFrame:
    """
    Build per-pair QC table with abs-method and epoch-CCF ΔRVs.

    Absolute ΔRV uses method stacks (not adopted ``exposure_rv_kms``).
    Epoch-CCF ΔRV prefers ``epoch_ccf_root`` lookup; optionally computes
    pairwise CCF when ``compute_epoch_ccf`` and spectra resolve.
    """
    rows: list[dict] = []
    for a, b, dt in pairs:
        row: dict = {
            "gaia_id": a.gaia_id,
            "epoch_i": a.epoch,
            "epoch_j": b.epoch,
            "mjd_i": a.mjd,
            "mjd_j": b.mjd,
            "delta_t_days": dt,
            "diagnostics_i": str(a.diagnostics_path),
            "diagnostics_j": str(b.diagnostics_path),
        }
        any_abs_violate = False
        for method in methods:
            rvi, eri = method_exposure_rv(a.diagnostics_path, method)
            rvj, erj = method_exposure_rv(b.diagnostics_path, method)
            if np.isfinite(rvi) and np.isfinite(rvj):
                d = float(rvi - rvj)
                # Combined formal error on difference
                if np.isfinite(eri) and np.isfinite(erj) and eri > 0 and erj > 0:
                    e = float(math.sqrt(eri**2 + erj**2))
                else:
                    e = float("nan")
            else:
                d = float("nan")
                e = float("nan")
            row[f"drv_{method}_kms"] = d
            row[f"drv_{method}_err_kms"] = e
            violate = _flag_violation(d, e, abs_thresh_kms=abs_thresh_kms, n_sigma=n_sigma)
            row[f"violate_{method}"] = bool(violate)
            any_abs_violate = any_abs_violate or violate

        dv_ccf, err_ccf = load_epoch_ccf_dv(epoch_ccf_root, a.gaia_id, a.epoch, b.epoch)
        if (not np.isfinite(dv_ccf)) and compute_epoch_ccf:
            sp_i = Path(a.spectrum_path) if a.spectrum_path else None
            sp_j = Path(b.spectrum_path) if b.spectrum_path else None
            if data_root is not None:
                if sp_i is None or not sp_i.is_file():
                    cand = Path(data_root) / f"Gaia_DR3_{a.gaia_id}_epoch_{a.epoch}.txt"
                    sp_i = cand if cand.is_file() else sp_i
                if sp_j is None or not sp_j.is_file():
                    cand = Path(data_root) / f"Gaia_DR3_{b.gaia_id}_epoch_{b.epoch}.txt"
                    sp_j = cand if cand.is_file() else sp_j
            if sp_i is not None and sp_j is not None and sp_i.is_file() and sp_j.is_file():
                try:
                    dv_ccf, err_ccf = compute_epoch_ccf_dv(
                        sp_i,
                        sp_j,
                        rv_search_half_width_kms=rv_search_half_width_kms,
                        max_grid_points=max_grid_points,
                    )
                except Exception as exc:  # noqa: BLE001 — keep QC row
                    logger.warning("epoch CCF failed %s vs %s: %s", sp_i, sp_j, exc)
                    dv_ccf, err_ccf = float("nan"), float("nan")

        row["drv_epoch_ccf_kms"] = dv_ccf
        row["drv_epoch_ccf_err_kms"] = err_ccf
        viol_ccf = _flag_violation(
            dv_ccf, err_ccf, abs_thresh_kms=abs_thresh_kms, n_sigma=n_sigma
        )
        row["violate_epoch_ccf"] = bool(viol_ccf)
        row["violate_any"] = bool(any_abs_violate or viol_ccf)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    table: pd.DataFrame,
    out_dir: Path,
    *,
    max_delta_days: float,
    abs_thresh_kms: float,
    n_sigma: float,
    sigma_scale_info: dict[str, float],
    calibration_json: Path | None = None,
) -> dict[str, Path]:
    """
    Persist short-pair CSV + markdown summary (+ optional calibration JSON).

    Returns
    -------
    dict[str, Path]
        Paths for ``pairs_csv``, ``report_md``, and optional ``scale_json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = out_dir / "short_pairs.csv"
    table.to_csv(pairs_csv, index=False)

    n = len(table)
    n_viol = int(table["violate_any"].sum()) if n and "violate_any" in table.columns else 0
    lines = [
        "# Short-pair QC (step 05a)",
        "",
        f"- Pairs with |Δt| ≤ **{max_delta_days:g}** day(s): **{n}**",
        f"- Violation flags (|ΔRV| ≥ {abs_thresh_kms:g} km/s or ≥ {n_sigma:g}σ): **{n_viol}**",
        f"- Recommended ``sigma_ij`` scale (robust MAD): **{sigma_scale_info.get('recommended_sigma_ij_scale', float('nan')):.4g}**",
        f"  - 1.4826×MAD: {sigma_scale_info.get('sigma_robust_kms', float('nan')):.4g} km/s; MAD: {sigma_scale_info.get('mad_kms', float('nan')):.4g} km/s",
        f"  - RMS (all): {sigma_scale_info.get('rms_kms', float('nan')):.4g} km/s → scale_rms={sigma_scale_info.get('scale_rms', float('nan')):.4g}",
        f"  - RMS (|Δv|≤50): {sigma_scale_info.get('rms_clipped_kms', float('nan')):.4g} km/s → scale_rms_clipped={sigma_scale_info.get('scale_rms_clipped', float('nan')):.4g}",
        f"  - median formal σ: {sigma_scale_info.get('median_formal_err_kms', float('nan')):.4g} km/s",
        f"  - n (epoch-CCF): {int(sigma_scale_info.get('n', 0))} (clipped n={int(sigma_scale_info.get('n_clipped', 0))})",
        "",
        "## Method scatter (short pairs with finite ΔRV)",
        "",
    ]
    for method in ABS_METHODS:
        col = f"drv_{method}_kms"
        if col not in table.columns or table.empty:
            continue
        vals = table[col].to_numpy(dtype=float)
        ok = np.isfinite(vals)
        if not np.any(ok):
            lines.append(f"- `{method}`: no finite pairs")
            continue
        v = vals[ok]
        n_v = int(table[f"violate_{method}"].sum()) if f"violate_{method}" in table.columns else 0
        lines.append(
            f"- `{method}`: n={int(np.sum(ok))}, RMS={float(np.sqrt(np.mean(v**2))):.4g} km/s, "
            f"MAD={float(np.median(np.abs(v - np.median(v)))):.4g} km/s, violations={n_v}"
        )
    if "drv_epoch_ccf_kms" in table.columns and not table.empty:
        vals = table["drv_epoch_ccf_kms"].to_numpy(dtype=float)
        ok = np.isfinite(vals)
        n_v = int(table["violate_epoch_ccf"].sum()) if "violate_epoch_ccf" in table.columns else 0
        if np.any(ok):
            v = vals[ok]
            lines.append(
                f"- `epoch_ccf`: n={int(np.sum(ok))}, RMS={float(np.sqrt(np.mean(v**2))):.4g} km/s, "
                f"MAD={float(np.median(np.abs(v - np.median(v)))):.4g} km/s, violations={n_v}"
            )
        else:
            lines.append("- `epoch_ccf`: no finite pairs (pass `--epoch-ccf-root` or `--compute-epoch-ccf`)")

    lines += [
        "",
        "## Inflation hook",
        "",
        "Pass the recommended scale into the epoch matrix CLI:",
        "",
        "```bash",
        "PYTHONPATH=. python -m validation.epoch_ccf_matrix \\",
        "  --gaia-id <id> --data-root /Users/rfoley/darkhunter/rvs/data \\",
        "  --abs-diagnostics-glob 'output/Gaia_DR3_<id>_epoch_*_diagnostics.csv' \\",
        "  --out-dir validation_output/epoch_ccf/<id> \\",
        f"  --sigma-ij-scale {sigma_scale_info.get('recommended_sigma_ij_scale', 1.0):.6g}",
        "```",
        "",
        "Or call :func:`darkhunter_rv.epoch_ccf.inflate_sigma_ij` before WLS fill.",
        "",
        f"Pairs table: `{pairs_csv}`",
        "",
    ]
    report_md = out_dir / "SHORT_PAIR_QC.md"
    report_md.write_text("\n".join(lines) + "\n")

    paths: dict[str, Path] = {"pairs_csv": pairs_csv, "report_md": report_md}
    payload = {
        "max_delta_days": float(max_delta_days),
        "abs_violation_kms": float(abs_thresh_kms),
        "n_sigma": float(n_sigma),
        "n_pairs": int(n),
        "n_violations": int(n_viol),
        **{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in sigma_scale_info.items()},
        "source_pairs_csv": str(pairs_csv),
        "note": "Multiply off-diagonal sigma_ij by recommended_sigma_ij_scale (never < 1).",
    }
    scale_json = out_dir / "short_pair_sigma_scale.json"
    scale_json.write_text(json.dumps(payload, indent=2) + "\n")
    paths["scale_json"] = scale_json

    if calibration_json is not None:
        calibration_json = Path(calibration_json)
        calibration_json.parent.mkdir(parents=True, exist_ok=True)
        # Compact summary for calibration/ tracking
        cal = {
            "recommended_sigma_ij_scale": payload.get("recommended_sigma_ij_scale"),
            "rms_kms": payload.get("rms_kms"),
            "mad_kms": payload.get("mad_kms"),
            "median_formal_err_kms": payload.get("median_formal_err_kms"),
            "n_epoch_ccf": payload.get("n"),
            "n_pairs": int(n),
            "max_delta_days": float(max_delta_days),
            "report": str(report_md),
            "pairs_csv": str(pairs_csv),
        }
        calibration_json.write_text(json.dumps(cal, indent=2) + "\n")
        paths["calibration_json"] = calibration_json

    # Small tracked summary CSV under calibration/ when path given
    if calibration_json is not None:
        summary_csv = calibration_json.with_suffix(".csv")
        summary_rows = [
            {
                "metric": "recommended_sigma_ij_scale",
                "value": payload.get("recommended_sigma_ij_scale"),
            },
            {"metric": "rms_kms", "value": payload.get("rms_kms")},
            {"metric": "mad_kms", "value": payload.get("mad_kms")},
            {"metric": "median_formal_err_kms", "value": payload.get("median_formal_err_kms")},
            {"metric": "n_pairs", "value": n},
            {"metric": "n_violations", "value": n_viol},
            {"metric": "max_delta_days", "value": max_delta_days},
        ]
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        paths["calibration_summary_csv"] = summary_csv

    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--diagnostics-glob",
        default="/Users/rfoley/darkhunter/rvs/dark-hunter_rv/output/Gaia_DR3_*_epoch_*_diagnostics.csv",
        help="Glob of epoch diagnostics CSVs",
    )
    p.add_argument(
        "--summary-dir",
        default=None,
        help="Deprecated alias: if set, uses <dir>/Gaia_DR3_*_epoch_*_diagnostics.csv",
    )
    p.add_argument("--gaia-id", default=None, help="Optional single-star filter")
    p.add_argument(
        "--max-delta-days",
        type=float,
        default=1.0,
        help="Max |Δt| in days (default 1 = same-night window)",
    )
    p.add_argument(
        "--same-calendar-night",
        action="store_true",
        help="Require floor(MJD) equality in addition to --max-delta-days",
    )
    p.add_argument(
        "--abs-violation-kms",
        type=float,
        default=5.0,
        help="Absolute |ΔRV| floor for violation flag (km/s)",
    )
    p.add_argument(
        "--n-sigma",
        type=float,
        default=3.0,
        help="σ multiplier for violation flag vs combined formal error",
    )
    p.add_argument(
        "--epoch-ccf-root",
        type=Path,
        default=None,
        help="Root with <gaia_id>/epoch_ccf_pairs.csv (step 11 outputs)",
    )
    p.add_argument(
        "--compute-epoch-ccf",
        action="store_true",
        help="Compute pairwise epoch CCF when matrix lookup missing",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/rfoley/darkhunter/rvs/data"),
        help="Spectrum directory for --compute-epoch-ccf",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation_output/short_pair_qc"),
        help="Report output directory",
    )
    p.add_argument(
        "--calibration-json",
        type=Path,
        default=Path("calibration/short_pair_sigma_scale.json"),
        help="Tracked summary JSON under calibration/ (empty string to skip)",
    )
    p.add_argument("--max-pairs", type=int, default=None, help="Cap pairs (debug)")
    p.add_argument("--rv-search-half-width-kms", type=float, default=500.0)
    p.add_argument("--max-grid-points", type=int, default=16384)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    diag_glob = str(args.diagnostics_glob)
    if args.summary_dir:
        diag_glob = str(Path(args.summary_dir) / "Gaia_DR3_*_epoch_*_diagnostics.csv")

    epochs = discover_epochs(diag_glob, gaia_id=args.gaia_id)
    logger.info("Discovered %d epochs from %s", len(epochs), diag_glob)
    pairs = find_pairs(
        epochs,
        max_delta_days=float(args.max_delta_days),
        same_calendar_night=bool(args.same_calendar_night),
    )
    if args.max_pairs is not None and args.max_pairs > 0:
        pairs = pairs[: int(args.max_pairs)]
    logger.info("Short pairs: %d", len(pairs))

    max_grid = None if int(args.max_grid_points) <= 0 else int(args.max_grid_points)
    table = build_short_pair_table(
        pairs,
        epoch_ccf_root=args.epoch_ccf_root,
        compute_epoch_ccf=bool(args.compute_epoch_ccf),
        data_root=Path(args.data_root) if args.data_root else None,
        abs_thresh_kms=float(args.abs_violation_kms),
        n_sigma=float(args.n_sigma),
        rv_search_half_width_kms=float(args.rv_search_half_width_kms),
        max_grid_points=max_grid,
    )

    if table.empty:
        sigma_info = recommend_sigma_ij_scale(np.array([]), np.array([]))
    else:
        sigma_info = recommend_sigma_ij_scale(
            table["drv_epoch_ccf_kms"].to_numpy(dtype=float)
            if "drv_epoch_ccf_kms" in table.columns
            else np.array([]),
            table["drv_epoch_ccf_err_kms"].to_numpy(dtype=float)
            if "drv_epoch_ccf_err_kms" in table.columns
            else np.array([]),
        )

    cal_path: Path | None
    if args.calibration_json is None or str(args.calibration_json).strip() in ("", "none", "None"):
        cal_path = None
    else:
        cal_path = Path(args.calibration_json)

    paths = write_report(
        table,
        Path(args.out_dir),
        max_delta_days=float(args.max_delta_days),
        abs_thresh_kms=float(args.abs_violation_kms),
        n_sigma=float(args.n_sigma),
        sigma_scale_info=sigma_info,
        calibration_json=cal_path,
    )
    summary = {
        "n_epochs": len(epochs),
        "n_pairs": len(pairs),
        "recommended_sigma_ij_scale": sigma_info.get("recommended_sigma_ij_scale"),
        "rms_kms": sigma_info.get("rms_kms"),
        **{k: str(v) for k, v in paths.items()},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
