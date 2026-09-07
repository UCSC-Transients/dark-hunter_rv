#!/usr/bin/env python3
"""Aggregate multi-star epoch-CCF campaign outputs and write diagnostic plots/report.

Expects per-star dirs under --epoch-ccf-root/<gaia_id>/ with:
  epoch_ccf_pairs.csv, epoch_ccf_vs_abs_delta.csv, epoch_ccf_abs_fill.csv,
  epoch_ccf_meta.json (optional).

Adds:
  - MAD-based outlier rejection before residual RMS
  - Per-star residual Δv matrices (diag = auto-correlation)
  - Residual Δv vs S/N (pair CCF peak_snr; optional epoch mask median S/N)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DIAG_EPOCH_RE = re.compile(
    r"Gaia_DR3_(\d+)_epoch_(\d+)_diagnostics\.csv$",
    re.IGNORECASE,
)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))


def mad_clip_mask(
    x: np.ndarray,
    *,
    n_sigma: float = 5.0,
    min_keep: int = 5,
) -> np.ndarray:
    """
    Boolean keep-mask: reject points with |x-median| > n_sigma * 1.4826 * MAD.

    If MAD is zero or too few finite points remain, keep all finite points.
    """
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(x)
    finite = x[keep]
    if finite.size < max(min_keep, 3):
        return keep
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    if not np.isfinite(mad) or mad <= 0:
        return keep
    scale = 1.4826 * mad
    thr = float(n_sigma) * scale
    keep2 = keep & (np.abs(x - med) <= thr)
    if int(np.sum(keep2)) < min_keep:
        return keep
    return keep2


def robust_rms(x: np.ndarray, *, n_sigma: float = 5.0) -> dict[str, float | int]:
    """Return raw and MAD-clipped RMS plus keep counts."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    n_finite = int(np.sum(finite))
    mask = mad_clip_mask(x, n_sigma=n_sigma)
    n_keep = int(np.sum(mask))
    return {
        "n_finite": n_finite,
        "n_kept": n_keep,
        "n_rejected": int(n_finite - n_keep),
        "rms_raw_kms": _rms(x[finite]),
        "rms_clipped_kms": _rms(x[mask]),
        "median_kms": float(np.nanmedian(x)) if n_finite else float("nan"),
        "mad_kms": _mad(x),
        "clip_n_sigma": float(n_sigma),
    }


def _discover_star_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "epoch_ccf_vs_abs_delta.csv").is_file() or (p / "epoch_ccf_pairs.csv").is_file():
            dirs.append(p)
    return dirs


def load_campaign(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (pairs, vs_abs, fill, meta_rows)."""
    pair_frames: list[pd.DataFrame] = []
    vs_frames: list[pd.DataFrame] = []
    fill_frames: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for star_dir in _discover_star_dirs(root):
        gaia_id = star_dir.name
        pairs_path = star_dir / "epoch_ccf_pairs.csv"
        vs_path = star_dir / "epoch_ccf_vs_abs_delta.csv"
        fill_path = star_dir / "epoch_ccf_abs_fill.csv"
        meta_path = star_dir / "epoch_ccf_meta.json"

        if pairs_path.is_file():
            df = pd.read_csv(pairs_path)
            if "gaia_id" not in df.columns:
                df.insert(0, "gaia_id", gaia_id)
            pair_frames.append(df)

        if vs_path.is_file():
            df = pd.read_csv(vs_path)
            if "gaia_id" not in df.columns:
                df.insert(0, "gaia_id", gaia_id)
            vs_frames.append(df)

        if fill_path.is_file():
            df = pd.read_csv(fill_path)
            if "gaia_id" not in df.columns:
                df.insert(0, "gaia_id", gaia_id)
            fill_frames.append(df)

        row: dict = {"gaia_id": gaia_id}
        if meta_path.is_file():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                row.update(
                    {
                        "n_epochs": meta.get("n_epochs"),
                        "n_abs_anchors": meta.get("n_abs_anchors"),
                        "n_abs_delta_comparisons": meta.get("n_abs_delta_comparisons"),
                        "n_abs_rel_discordant": meta.get("n_abs_rel_discordant"),
                        "discord_n_sigma": meta.get("discord_n_sigma"),
                        "abs_delta_residual_rms_kms": meta.get("abs_delta_residual_rms_kms"),
                        "diag_abs_max_kms": meta.get("diag_abs_max_kms"),
                        "diag_abs_median_kms": meta.get("diag_abs_median_kms"),
                        "float_zeropoint": meta.get("float_zeropoint"),
                    }
                )
            except (OSError, json.JSONDecodeError) as exc:
                row["meta_error"] = str(exc)
        meta_rows.append(row)

    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    vs_abs = pd.concat(vs_frames, ignore_index=True) if vs_frames else pd.DataFrame()
    fill = pd.concat(fill_frames, ignore_index=True) if fill_frames else pd.DataFrame()
    meta_df = pd.DataFrame(meta_rows)
    return pairs, vs_abs, fill, meta_df


def attach_pair_snr(vs_abs: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Attach CCF pair ``peak_snr`` onto vs-abs rows (match gaia_id, epoch_i, epoch_j)."""
    out = vs_abs.copy()
    if out.empty or pairs.empty:
        out["pair_peak_snr"] = np.nan
        return out
    need = {"gaia_id", "epoch_i", "epoch_j"}
    if not need.issubset(out.columns) or not need.issubset(pairs.columns):
        out["pair_peak_snr"] = np.nan
        return out
    cols = ["gaia_id", "epoch_i", "epoch_j"]
    if "peak_snr" not in pairs.columns:
        out["pair_peak_snr"] = np.nan
        return out
    out = out.copy()
    out["gaia_id"] = out["gaia_id"].astype(str)
    out["epoch_i"] = out["epoch_i"].astype(int)
    out["epoch_j"] = out["epoch_j"].astype(int)
    # Prefer non-auto rows; take first match
    if "auto_correlation" in pairs.columns:
        src = pairs.loc[~pairs["auto_correlation"].astype(bool)].copy()
    else:
        src = pairs.copy()
    src["gaia_id"] = src["gaia_id"].astype(str)
    src["epoch_i"] = src["epoch_i"].astype(int)
    src["epoch_j"] = src["epoch_j"].astype(int)
    src = src[cols + ["peak_snr"]].drop_duplicates(subset=cols, keep="first")
    merged = out.merge(src, on=cols, how="left", suffixes=("", "_pair"))
    out["pair_peak_snr"] = merged["peak_snr"].to_numpy(dtype=float)
    return out


def load_epoch_mask_snr(diagnostics_root: Path) -> pd.DataFrame:
    """
    Per-exposure median mask ``ccf_peak_snr`` from diagnostics CSVs.

    Returns columns: gaia_id, epoch, median_mask_ccf_peak_snr.
    """
    rows: list[dict] = []
    root = Path(diagnostics_root)
    if not root.is_dir():
        return pd.DataFrame(columns=["gaia_id", "epoch", "median_mask_ccf_peak_snr"])
    for path in sorted(root.glob("Gaia_DR3_*_epoch_*_diagnostics.csv")):
        m = _DIAG_EPOCH_RE.search(path.name)
        if not m:
            continue
        gid, ep = m.group(1), int(m.group(2))
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "method" not in df.columns or "ccf_peak_snr" not in df.columns:
            continue
        chunk = df["chunk_key"].astype(str) if "chunk_key" in df.columns else pd.Series([""] * len(df))
        mask = (df["method"].astype(str) == "mask_ccf") & (chunk != "all")
        snr = pd.to_numeric(df.loc[mask, "ccf_peak_snr"], errors="coerce").to_numpy(dtype=float)
        snr = snr[np.isfinite(snr)]
        rows.append(
            {
                "gaia_id": gid,
                "epoch": ep,
                "median_mask_ccf_peak_snr": float(np.median(snr)) if snr.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def attach_epoch_snr(vs_abs: pd.DataFrame, epoch_snr: pd.DataFrame) -> pd.DataFrame:
    """Attach min(epoch_i, epoch_j) median mask S/N."""
    out = vs_abs.copy()
    if out.empty or epoch_snr.empty:
        out["min_epoch_mask_snr"] = np.nan
        return out
    out["gaia_id"] = out["gaia_id"].astype(str)
    out["epoch_i"] = out["epoch_i"].astype(int)
    out["epoch_j"] = out["epoch_j"].astype(int)
    es = epoch_snr.copy()
    es["gaia_id"] = es["gaia_id"].astype(str)
    es["epoch"] = es["epoch"].astype(int)
    e = es.rename(columns={"epoch": "epoch_i", "median_mask_ccf_peak_snr": "snr_i"})
    f = es.rename(columns={"epoch": "epoch_j", "median_mask_ccf_peak_snr": "snr_j"})
    m = out.merge(e[["gaia_id", "epoch_i", "snr_i"]], on=["gaia_id", "epoch_i"], how="left")
    m = m.merge(f[["gaia_id", "epoch_j", "snr_j"]], on=["gaia_id", "epoch_j"], how="left")
    si = m["snr_i"].to_numpy(dtype=float)
    sj = m["snr_j"].to_numpy(dtype=float)
    out["min_epoch_mask_snr"] = np.fmin(si, sj)
    return out


def compute_summary(
    pairs: pd.DataFrame,
    vs_abs: pd.DataFrame,
    fill: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    clip_n_sigma: float,
) -> dict:
    n_stars = int(meta_df["gaia_id"].nunique()) if len(meta_df) else 0

    if len(pairs) and "auto_correlation" in pairs.columns:
        off = pairs.loc[~pairs["auto_correlation"].astype(bool)].copy()
        diag = pairs.loc[pairs["auto_correlation"].astype(bool)].copy()
    elif len(pairs) and {"epoch_i", "epoch_j"}.issubset(pairs.columns):
        off = pairs.loc[pairs["epoch_i"] != pairs["epoch_j"]].copy()
        diag = pairs.loc[pairs["epoch_i"] == pairs["epoch_j"]].copy()
    else:
        off = pairs.copy()
        diag = pairs.iloc[0:0].copy()

    n_off_pairs = int(len(off))
    n_diag = int(len(diag))
    n_pairs_total = int(len(pairs)) if len(pairs) else 0

    residual = (
        vs_abs["residual_kms"].to_numpy(dtype=float)
        if len(vs_abs) and "residual_kms" in vs_abs.columns
        else np.array([], dtype=float)
    )
    if residual.size == 0 and len(vs_abs) and {"dv_ccf_kms", "dv_abs_kms"}.issubset(vs_abs.columns):
        residual = (
            vs_abs["dv_ccf_kms"].to_numpy(dtype=float) - vs_abs["dv_abs_kms"].to_numpy(dtype=float)
        )

    rob = robust_rms(residual, n_sigma=clip_n_sigma)

    disc_col = "epoch_ccf_abs_rel_discordant"
    if len(vs_abs) and disc_col in vs_abs.columns:
        disc = vs_abs[disc_col].astype(bool).to_numpy()
        n_discordant = int(np.sum(disc))
        n_vs = int(len(vs_abs))
        discordant_frac = float(n_discordant / n_vs) if n_vs else float("nan")
    else:
        n_discordant = 0
        n_vs = int(len(vs_abs))
        discordant_frac = float("nan")

    if len(diag) and "dv_kms" in diag.columns:
        diag_abs_dv = np.abs(diag["dv_kms"].to_numpy(dtype=float))
    else:
        diag_abs_dv = np.array([], dtype=float)

    n_fill_rows = int(len(fill))
    n_relative_only = 0
    n_float_zp = 0
    if len(fill):
        if "n_abs_anchors" in fill.columns:
            g = fill.groupby("gaia_id", as_index=False).first()
            n_abs_anchors_sum = int(np.nansum(g["n_abs_anchors"].to_numpy(dtype=float)))
            median_anchors = float(np.nanmedian(g["n_abs_anchors"].to_numpy(dtype=float)))
        else:
            n_abs_anchors_sum = 0
            median_anchors = float("nan")
        if "relative_only" in fill.columns:
            g = fill.groupby("gaia_id", as_index=False).first()
            n_relative_only = int(np.sum(g["relative_only"].astype(bool)))
        if "float_zeropoint" in fill.columns:
            g = fill.groupby("gaia_id", as_index=False).first()
            n_float_zp = int(np.sum(g["float_zeropoint"].astype(bool)))
    else:
        n_abs_anchors_sum = 0
        median_anchors = float("nan")

    if len(meta_df) and "n_abs_anchors" in meta_df.columns:
        median_anchors_meta = float(np.nanmedian(meta_df["n_abs_anchors"].to_numpy(dtype=float)))
        n_stars_with_anchors = int(np.sum(np.asarray(meta_df["n_abs_anchors"], dtype=float) > 0))
    else:
        median_anchors_meta = median_anchors
        n_stars_with_anchors = 0

    summary = {
        "n_stars": n_stars,
        "n_pairs_csv_rows": n_pairs_total,
        "n_offdiag_pairs": n_off_pairs,
        "n_diag_pairs": n_diag,
        "n_vs_abs_comparisons": n_vs,
        "n_discordant": n_discordant,
        "discordant_fraction": discordant_frac,
        "residual_median_kms": rob["median_kms"],
        "residual_mad_kms": rob["mad_kms"],
        "residual_rms_raw_kms": rob["rms_raw_kms"],
        "residual_rms_clipped_kms": rob["rms_clipped_kms"],
        "residual_n_finite": rob["n_finite"],
        "residual_n_kept_for_rms": rob["n_kept"],
        "residual_n_rejected_for_rms": rob["n_rejected"],
        "residual_clip_n_sigma": rob["clip_n_sigma"],
        "diag_abs_dv_median_kms": float(np.nanmedian(diag_abs_dv)) if diag_abs_dv.size else float("nan"),
        "diag_abs_dv_mad_kms": _mad(diag_abs_dv),
        "diag_abs_dv_p95_kms": (
            float(np.nanpercentile(diag_abs_dv, 95)) if diag_abs_dv.size else float("nan")
        ),
        "diag_abs_dv_max_kms": float(np.nanmax(diag_abs_dv)) if diag_abs_dv.size else float("nan"),
        "n_fill_rows": n_fill_rows,
        "median_n_abs_anchors": median_anchors_meta,
        "n_stars_with_abs_anchors": n_stars_with_anchors,
        "n_stars_relative_only": n_relative_only,
        "n_stars_float_zeropoint": n_float_zp,
        "n_abs_anchors_sum_first_row": n_abs_anchors_sum,
    }
    return summary


def per_star_discordant(vs_abs: pd.DataFrame, *, clip_n_sigma: float) -> pd.DataFrame:
    if vs_abs.empty or "gaia_id" not in vs_abs.columns:
        return pd.DataFrame(
            columns=[
                "gaia_id",
                "n_comparisons",
                "n_discordant",
                "discordant_fraction",
                "residual_rms_raw_kms",
                "residual_rms_clipped_kms",
                "residual_n_rejected",
                "residual_median_kms",
            ]
        )
    disc_col = "epoch_ccf_abs_rel_discordant"
    rows = []
    for gid, g in vs_abs.groupby("gaia_id"):
        n = len(g)
        nd = int(g[disc_col].astype(bool).sum()) if disc_col in g.columns else 0
        res = g["residual_kms"].to_numpy(dtype=float) if "residual_kms" in g.columns else np.array([])
        rob = robust_rms(res, n_sigma=clip_n_sigma)
        rows.append(
            {
                "gaia_id": gid,
                "n_comparisons": n,
                "n_discordant": nd,
                "discordant_fraction": float(nd / n) if n else float("nan"),
                "residual_rms_raw_kms": rob["rms_raw_kms"],
                "residual_rms_clipped_kms": rob["rms_clipped_kms"],
                "residual_n_rejected": rob["n_rejected"],
                "residual_median_kms": rob["median_kms"],
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["discordant_fraction", "n_discordant", "residual_rms_clipped_kms"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _star_epochs(pairs: pd.DataFrame, vs_abs: pd.DataFrame) -> list[int]:
    eps: set[int] = set()
    for df in (pairs, vs_abs):
        if df is None or df.empty:
            continue
        for col in ("epoch_i", "epoch_j", "epoch"):
            if col in df.columns:
                for v in df[col].to_numpy():
                    try:
                        eps.add(int(v))
                    except (TypeError, ValueError):
                        continue
    return sorted(eps)


def build_star_residual_matrix(
    pairs: pd.DataFrame,
    vs_abs: pd.DataFrame,
    epochs: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (value, err, kind) matrices indexed by ``epochs``.

    kind: 0=missing, 1=residual (off-diag), 2=auto-correlation (diag).
    Off-diagonal values are residual = dv_ccf - dv_abs (antisymmetric).
    Diagonal values are auto-correlation dv.
    """
    n = len(epochs)
    idx = {ep: i for i, ep in enumerate(epochs)}
    val = np.full((n, n), np.nan)
    err = np.full((n, n), np.nan)
    kind = np.zeros((n, n), dtype=int)

    if not pairs.empty and {"epoch_i", "epoch_j"}.issubset(pairs.columns):
        for _, row in pairs.iterrows():
            try:
                ei, ej = int(row["epoch_i"]), int(row["epoch_j"])
            except (TypeError, ValueError):
                continue
            if ei not in idx or ej not in idx:
                continue
            i, j = idx[ei], idx[ej]
            is_auto = bool(row["auto_correlation"]) if "auto_correlation" in row.index else (ei == ej)
            if is_auto and ei == ej:
                val[i, j] = float(row.get("dv_kms", np.nan))
                err[i, j] = float(row.get("err_kms", np.nan))
                kind[i, j] = 2

    if not vs_abs.empty and {"epoch_i", "epoch_j", "residual_kms"}.issubset(vs_abs.columns):
        for _, row in vs_abs.iterrows():
            try:
                ei, ej = int(row["epoch_i"]), int(row["epoch_j"])
            except (TypeError, ValueError):
                continue
            if ei not in idx or ej not in idx:
                continue
            i, j = idx[ei], idx[ej]
            r = float(row["residual_kms"])
            e = float(row["sigma_combined_kms"]) if "sigma_combined_kms" in row.index else float(
                row.get("err_ccf_kms", np.nan)
            )
            val[i, j] = r
            err[i, j] = e
            kind[i, j] = 1
            # Antisymmetric counterpart if empty
            if kind[j, i] == 0 or not np.isfinite(val[j, i]):
                val[j, i] = -r if np.isfinite(r) else np.nan
                err[j, i] = e
                kind[j, i] = 1

    return val, err, kind


def plot_star_residual_matrix(
    gaia_id: str,
    epochs: list[int],
    val: np.ndarray,
    err: np.ndarray,
    kind: np.ndarray,
    out_path: Path,
) -> Path:
    """Color-grid of residual Δv; diagonal shows auto-correlation."""
    n = len(epochs)
    fig_w = max(4.5, 1.1 * n + 2.0)
    fig_h = max(4.0, 1.1 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Color scale from finite off-diagonal residuals only
    off = val[kind == 1]
    off = off[np.isfinite(off)]
    if off.size:
        vmax = float(np.nanpercentile(np.abs(off), 95))
        vmax = max(vmax, 1.0)
    else:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # Mask missing for display: use NaN so they appear blank under cmap
    display = np.ma.array(val, mask=~np.isfinite(val))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(display, cmap=cmap, norm=norm, origin="upper", aspect="equal")

    # Annotate cells
    fontsize = 8 if n <= 6 else (6 if n <= 10 else 5)
    for i in range(n):
        for j in range(n):
            if kind[i, j] == 0 or not np.isfinite(val[i, j]):
                continue
            v = val[i, j]
            e = err[i, j]
            # Text color by background lightness
            color = "k"
            if kind[i, j] == 1 and np.isfinite(v):
                # dark text near white/center, light text at saturated ends
                if abs(v) > 0.55 * vmax:
                    color = "w"
            if kind[i, j] == 2:
                label = f"auto\n{v:.3g}"
                if np.isfinite(e):
                    label += f"\n±{e:.2g}"
                color = "k"
            else:
                label = f"{v:.2f}"
                if np.isfinite(e):
                    label += f"\n±{e:.2g}"
            ax.text(j, i, label, ha="center", va="center", fontsize=fontsize, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(e) for e in epochs])
    ax.set_yticklabels([str(e) for e in epochs])
    ax.set_xlabel("epoch j")
    ax.set_ylabel("epoch i")
    ax.set_title(f"Gaia {gaia_id}\nresidual Δv = Δv_CCF − Δv_abs (diag = auto-corr)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("residual Δv (km/s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def make_star_matrices(
    pairs: pd.DataFrame,
    vs_abs: pd.DataFrame,
    out_dir: Path,
    *,
    max_stars: int | None = None,
) -> list[Path]:
    written: list[Path] = []
    mat_dir = out_dir / "star_matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)
    if vs_abs.empty and pairs.empty:
        return written
    gids = sorted(set(vs_abs["gaia_id"].astype(str)) | set(pairs["gaia_id"].astype(str))) if (
        not vs_abs.empty or not pairs.empty
    ) else []
    if "gaia_id" in vs_abs.columns and not vs_abs.empty:
        # Prefer stars that have vs-abs comparisons first
        with_vs = sorted(vs_abs["gaia_id"].astype(str).unique())
        without = [g for g in gids if g not in set(with_vs)]
        gids = with_vs + without
    if max_stars is not None and max_stars > 0:
        gids = gids[: int(max_stars)]

    for gid in gids:
        pstar = pairs.loc[pairs["gaia_id"].astype(str) == str(gid)] if not pairs.empty else pairs
        vstar = vs_abs.loc[vs_abs["gaia_id"].astype(str) == str(gid)] if not vs_abs.empty else vs_abs
        epochs = _star_epochs(pstar, vstar)
        if len(epochs) < 2:
            continue
        val, err, kind = build_star_residual_matrix(pstar, vstar, epochs)
        path = mat_dir / f"{gid}_residual_matrix.png"
        plot_star_residual_matrix(str(gid), epochs, val, err, kind, path)
        written.append(path)
    return written


def make_plots(
    pairs: pd.DataFrame,
    vs_abs: pd.DataFrame,
    per_star: pd.DataFrame,
    out_dir: Path,
    *,
    clip_n_sigma: float,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    residual = (
        vs_abs["residual_kms"].to_numpy(dtype=float)
        if len(vs_abs) and "residual_kms" in vs_abs.columns
        else np.array([], dtype=float)
    )
    disc = (
        vs_abs["epoch_ccf_abs_rel_discordant"].astype(bool).to_numpy()
        if len(vs_abs) and "epoch_ccf_abs_rel_discordant" in vs_abs.columns
        else np.zeros(len(vs_abs), dtype=bool)
    )
    keep = mad_clip_mask(residual, n_sigma=clip_n_sigma) if residual.size else np.array([], dtype=bool)

    # 1. residual histogram (mark clipped outliers)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if residual.size:
        finite = residual[np.isfinite(residual)]
        kept = residual[keep]
        lo, hi = (
            np.nanpercentile(finite, [0.5, 99.5])
            if finite.size > 20
            else (float(np.nanmin(finite)), float(np.nanmax(finite)))
        )
        span = max(hi - lo, 1.0)
        bins = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 60)
        ax.hist(kept[np.isfinite(kept)], bins=bins, color="steelblue", edgecolor="white", alpha=0.9, label="kept")
        rejected = residual[np.isfinite(residual) & ~keep]
        if rejected.size:
            ax.hist(rejected, bins=bins, color="C3", edgecolor="white", alpha=0.55, label="MAD-rejected")
        ax.axvline(0.0, color="k", lw=1.0, ls="--")
        if kept.size:
            ax.axvline(np.median(kept), color="C1", lw=1.2, label=f"median={np.median(kept):.2f}")
        ax.legend(frameon=False)
    ax.set_xlabel(r"residual $dv_{\mathrm{ccf}} - dv_{\mathrm{abs}}$ (km/s)")
    ax.set_ylabel("count")
    ax.set_title("Epoch CCF vs abs ΔRV residual")
    fig.tight_layout()
    p = out_dir / "residual_histogram.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 2. residual vs sigma_combined scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(vs_abs) and "sigma_combined_kms" in vs_abs.columns:
        x = vs_abs["sigma_combined_kms"].to_numpy(dtype=float)
        y = residual
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[ok & keep & ~disc],
            y[ok & keep & ~disc],
            s=12,
            alpha=0.45,
            c="steelblue",
            label="ok kept",
            rasterized=True,
        )
        ax.scatter(
            x[ok & keep & disc],
            y[ok & keep & disc],
            s=18,
            alpha=0.7,
            c="C3",
            label="discordant kept",
            rasterized=True,
        )
        ax.scatter(
            x[ok & ~keep],
            y[ok & ~keep],
            s=10,
            alpha=0.35,
            c="0.5",
            marker="x",
            label="MAD-rejected",
            rasterized=True,
        )
        xs = np.linspace(max(np.nanmin(x[ok]), 1e-3), np.nanmax(x[ok]), 100) if ok.any() else None
        if xs is not None and "discord_n_sigma" in vs_abs.columns:
            thr = float(np.nanmedian(vs_abs["discord_n_sigma"].to_numpy(dtype=float)))
            ax.plot(xs, thr * xs, "k--", lw=1, label=f"±{thr:.0f}σ")
            ax.plot(xs, -thr * xs, "k--", lw=1)
        ax.legend(frameon=False, loc="best")
        ax.set_xlabel(r"$\sigma_{\mathrm{combined}}$ (km/s)")
    ax.set_ylabel(r"residual (km/s)")
    ax.set_title("Residual vs combined uncertainty")
    fig.tight_layout()
    p = out_dir / "residual_vs_sigma_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 3. dv_ccf vs dv_abs
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if len(vs_abs) and {"dv_ccf_kms", "dv_abs_kms"}.issubset(vs_abs.columns):
        x = vs_abs["dv_abs_kms"].to_numpy(dtype=float)
        y = vs_abs["dv_ccf_kms"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok & keep & ~disc], y[ok & keep & ~disc], s=12, alpha=0.45, c="steelblue", label="ok", rasterized=True)
        ax.scatter(x[ok & keep & disc], y[ok & keep & disc], s=18, alpha=0.7, c="C3", label="discordant", rasterized=True)
        ax.scatter(x[ok & ~keep], y[ok & ~keep], s=10, alpha=0.35, c="0.5", marker="x", label="MAD-rejected", rasterized=True)
        if (ok & keep).any():
            sel = ok & keep
            lo = float(np.nanmin([x[sel].min(), y[sel].min()]))
            hi = float(np.nanmax([x[sel].max(), y[sel].max()]))
            pad = 0.05 * max(hi - lo, 1.0)
            lims = [lo - pad, hi + pad]
            ax.plot(lims, lims, "k-", lw=1, label="1:1")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, loc="best")
    ax.set_xlabel(r"$dv_{\mathrm{abs}}$ (km/s)")
    ax.set_ylabel(r"$dv_{\mathrm{ccf}}$ (km/s)")
    ax.set_title("Pair ΔRV: epoch CCF vs absolute")
    fig.tight_layout()
    p = out_dir / "dv_ccf_vs_dv_abs.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 4. auto-correlation |dv| histogram
    if len(pairs) and "auto_correlation" in pairs.columns:
        diag = pairs.loc[pairs["auto_correlation"].astype(bool)]
    elif len(pairs) and {"epoch_i", "epoch_j"}.issubset(pairs.columns):
        diag = pairs.loc[pairs["epoch_i"] == pairs["epoch_j"]]
    else:
        diag = pairs.iloc[0:0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(diag) and "dv_kms" in diag.columns:
        adv = np.abs(diag["dv_kms"].to_numpy(dtype=float))
        adv = adv[np.isfinite(adv)]
        if adv.size:
            hi = max(float(np.nanpercentile(adv, 99)), 1e-4)
            bins = np.linspace(0.0, hi * 1.05, 50)
            ax.hist(adv, bins=bins, color="seagreen", edgecolor="white", alpha=0.9)
            ax.axvline(np.median(adv), color="C3", lw=1.2, label=f"median={np.median(adv):.3g}")
            ax.legend(frameon=False)
    ax.set_xlabel(r"diagonal auto-correlation $|dv|$ (km/s)")
    ax.set_ylabel("count")
    ax.set_title("Self-CCF |dv| (should be near 0)")
    fig.tight_layout()
    p = out_dir / "diag_autocorr_abs_dv_histogram.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 5. per-star discordant fraction bar
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(per_star):
        top = per_star.head(25).iloc[::-1]
        ypos = np.arange(len(top))
        colors = ["C3" if f > 0 else "steelblue" for f in top["discordant_fraction"]]
        ax.barh(ypos, top["discordant_fraction"].to_numpy(dtype=float), color=colors, alpha=0.85)
        ax.set_yticks(ypos)
        labels = [str(g)[-8:] for g in top["gaia_id"]]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("discordant fraction")
        ax.set_title("Top stars by abs↔rel ΔRV discordant fraction (Gaia id suffix)")
        ax.set_xlim(0, 1.05)
    fig.tight_layout()
    p = out_dir / "per_star_discordant_fraction.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 6. residual vs S/N (pair peak_snr)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if len(vs_abs) and "pair_peak_snr" in vs_abs.columns:
        x = vs_abs["pair_peak_snr"].to_numpy(dtype=float)
        y = residual
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        ax.scatter(
            x[ok & keep & ~disc],
            y[ok & keep & ~disc],
            s=14,
            alpha=0.5,
            c="steelblue",
            label="ok kept",
            rasterized=True,
        )
        ax.scatter(
            x[ok & keep & disc],
            y[ok & keep & disc],
            s=18,
            alpha=0.75,
            c="C3",
            label="discordant kept",
            rasterized=True,
        )
        ax.scatter(
            x[ok & ~keep],
            y[ok & ~keep],
            s=12,
            alpha=0.4,
            c="0.45",
            marker="x",
            label="MAD-rejected",
            rasterized=True,
        )
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.legend(frameon=False, loc="best")
    ax.set_xlabel("pair CCF peak S/N")
    ax.set_ylabel(r"residual $\Delta v_{\mathrm{CCF}}-\Delta v_{\mathrm{abs}}$ (km/s)")
    ax.set_title("Residual Δv vs pair CCF S/N")
    fig.tight_layout()
    p = out_dir / "residual_vs_pair_snr.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 7. residual vs min epoch mask S/N (if attached)
    if len(vs_abs) and "min_epoch_mask_snr" in vs_abs.columns:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        x = vs_abs["min_epoch_mask_snr"].to_numpy(dtype=float)
        y = residual
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if ok.any():
            ax.scatter(
                x[ok & keep & ~disc],
                y[ok & keep & ~disc],
                s=14,
                alpha=0.5,
                c="steelblue",
                label="ok kept",
                rasterized=True,
            )
            ax.scatter(
                x[ok & keep & disc],
                y[ok & keep & disc],
                s=18,
                alpha=0.75,
                c="C3",
                label="discordant kept",
                rasterized=True,
            )
            ax.scatter(
                x[ok & ~keep],
                y[ok & ~keep],
                s=12,
                alpha=0.4,
                c="0.45",
                marker="x",
                label="MAD-rejected",
                rasterized=True,
            )
            ax.axhline(0.0, color="k", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.legend(frameon=False, loc="best")
        ax.set_xlabel(r"min(epoch$_i$, epoch$_j$) median mask CCF peak S/N")
        ax.set_ylabel(r"residual $\Delta v_{\mathrm{CCF}}-\Delta v_{\mathrm{abs}}$ (km/s)")
        ax.set_title("Residual Δv vs epoch mask S/N")
        fig.tight_layout()
        p = out_dir / "residual_vs_epoch_mask_snr.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    return written


def write_report(
    summary: dict,
    per_star: pd.DataFrame,
    plot_paths: list[Path],
    matrix_paths: list[Path],
    out_dir: Path,
    epoch_ccf_root: Path,
) -> Path:
    lines = [
        "# Epoch CCF campaign diagnostics",
        "",
        f"Source: `{epoch_ccf_root}`",
        "",
        "## Summary",
        "",
        f"- **n stars:** {summary['n_stars']}",
        f"- **n off-diagonal CCF pairs:** {summary['n_offdiag_pairs']} "
        f"(pairs CSV rows={summary['n_pairs_csv_rows']}, diag={summary['n_diag_pairs']})",
        f"- **n abs ΔRV comparisons:** {summary['n_vs_abs_comparisons']}",
        f"- **residual (dv_ccf − dv_abs):** "
        f"median={summary['residual_median_kms']:.4g} km/s, "
        f"MAD={summary['residual_mad_kms']:.4g} km/s",
        f"- **RMS raw:** {summary['residual_rms_raw_kms']:.4g} km/s "
        f"(n={summary['residual_n_finite']})",
        f"- **RMS after MAD clip ({summary['residual_clip_n_sigma']:.3g}×1.4826·MAD):** "
        f"{summary['residual_rms_clipped_kms']:.4g} km/s "
        f"(kept={summary['residual_n_kept_for_rms']}, "
        f"rejected={summary['residual_n_rejected_for_rms']})",
        f"- **discordant fraction:** {summary['discordant_fraction']:.4f} "
        f"({summary['n_discordant']}/{summary['n_vs_abs_comparisons']})",
        f"- **diag auto-corr |dv|:** "
        f"median={summary['diag_abs_dv_median_kms']:.4g} km/s, "
        f"MAD={summary['diag_abs_dv_mad_kms']:.4g} km/s, "
        f"p95={summary['diag_abs_dv_p95_kms']:.4g} km/s, "
        f"max={summary['diag_abs_dv_max_kms']:.4g} km/s",
        f"- **fill anchors:** median n_abs_anchors={summary['median_n_abs_anchors']:.4g}, "
        f"stars with anchors={summary['n_stars_with_abs_anchors']}, "
        f"relative_only={summary['n_stars_relative_only']}, "
        f"float_zeropoint={summary['n_stars_float_zeropoint']}",
        "",
        "## Plots",
        "",
    ]
    for p in plot_paths:
        lines.append(f"- `{p.name}`")
    lines.append(f"- per-star matrices: `star_matrices/` ({len(matrix_paths)} files)")
    lines.extend(["", "## Top discordant stars", ""])
    if len(per_star):
        top = per_star.head(15)
        lines.append("| gaia_id | n_comp | n_disc | frac | RMS clipped | RMS raw |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in top.iterrows():
            lines.append(
                f"| {r['gaia_id']} | {int(r['n_comparisons'])} | {int(r['n_discordant'])} | "
                f"{r['discordant_fraction']:.3f} | {r['residual_rms_clipped_kms']:.3g} | "
                f"{r['residual_rms_raw_kms']:.3g} |"
            )
    else:
        lines.append("_No vs-abs comparisons found._")
    lines.append("")
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--epoch-ccf-root",
        type=Path,
        required=True,
        help="Directory containing per-star epoch_ccf/<gaia_id>/ outputs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for plots, aggregates, and REPORT.md",
    )
    ap.add_argument(
        "--diagnostics-root",
        type=Path,
        default=None,
        help="Optional root of *_diagnostics.csv for epoch mask S/N (residual vs S/N)",
    )
    ap.add_argument(
        "--mad-clip-sigma",
        type=float,
        default=5.0,
        help="MAD clip threshold in units of 1.4826·MAD before RMS (default 5)",
    )
    ap.add_argument(
        "--max-star-matrices",
        type=int,
        default=None,
        help="Optional cap on per-star matrix plots (default: all)",
    )
    ap.add_argument(
        "--skip-star-matrices",
        action="store_true",
        help="Skip per-star residual matrix grids",
    )
    args = ap.parse_args(argv)

    root = args.epoch_ccf_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: epoch-ccf-root not found: {root}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs, vs_abs, fill, meta_df = load_campaign(root)
    vs_abs = attach_pair_snr(vs_abs, pairs)
    if args.diagnostics_root is not None:
        epoch_snr = load_epoch_mask_snr(Path(args.diagnostics_root))
        vs_abs = attach_epoch_snr(vs_abs, epoch_snr)
        epoch_snr.to_csv(out_dir / "epoch_mask_snr.csv", index=False)

    pairs.to_csv(out_dir / "aggregated_pairs.csv", index=False)
    vs_abs.to_csv(out_dir / "aggregated_vs_abs_delta.csv", index=False)
    fill.to_csv(out_dir / "aggregated_abs_fill.csv", index=False)
    meta_df.to_csv(out_dir / "aggregated_meta.csv", index=False)

    clip = float(args.mad_clip_sigma)
    summary = compute_summary(pairs, vs_abs, fill, meta_df, clip_n_sigma=clip)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    per_star = per_star_discordant(vs_abs, clip_n_sigma=clip)
    per_star.to_csv(out_dir / "per_star_discordant.csv", index=False)

    plot_paths = make_plots(pairs, vs_abs, per_star, out_dir, clip_n_sigma=clip)
    matrix_paths: list[Path] = []
    if not args.skip_star_matrices:
        matrix_paths = make_star_matrices(
            pairs,
            vs_abs,
            out_dir,
            max_stars=args.max_star_matrices,
        )
    report_path = write_report(summary, per_star, plot_paths, matrix_paths, out_dir, root)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote report: {report_path}")
    for p in plot_paths:
        print(f"Wrote plot: {p}")
    print(f"Wrote {len(matrix_paths)} star matrices under {out_dir / 'star_matrices'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
