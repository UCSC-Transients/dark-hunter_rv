#!/usr/bin/env python3
"""Aggregate multi-star epoch-CCF campaign outputs and write diagnostic plots/report.

Expects per-star dirs under --epoch-ccf-root/<gaia_id>/ with:
  epoch_ccf_pairs.csv, epoch_ccf_vs_abs_delta.csv, epoch_ccf_abs_fill.csv,
  epoch_ccf_meta.json (optional).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


def compute_summary(
    pairs: pd.DataFrame,
    vs_abs: pd.DataFrame,
    fill: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> dict:
    n_stars = int(meta_df["gaia_id"].nunique()) if len(meta_df) else 0

    # Off-diagonal CCF pairs (non auto-correlation)
    n_pairs_total = int(len(pairs)) if len(pairs) else 0
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

    residual = (
        vs_abs["residual_kms"].to_numpy(dtype=float)
        if len(vs_abs) and "residual_kms" in vs_abs.columns
        else np.array([], dtype=float)
    )
    if residual.size == 0 and len(vs_abs) and {"dv_ccf_kms", "dv_abs_kms"}.issubset(vs_abs.columns):
        residual = (
            vs_abs["dv_ccf_kms"].to_numpy(dtype=float) - vs_abs["dv_abs_kms"].to_numpy(dtype=float)
        )

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

    # Diagonal auto-correlation |dv|
    if len(diag) and "dv_kms" in diag.columns:
        diag_abs_dv = np.abs(diag["dv_kms"].to_numpy(dtype=float))
    else:
        diag_abs_dv = np.array([], dtype=float)

    # Fill anchors
    n_fill_rows = int(len(fill))
    n_abs_anchors_sum = 0
    n_relative_only = 0
    n_float_zp = 0
    if len(fill):
        if "n_abs_anchors" in fill.columns:
            # per-star unique: take first row per gaia_id
            g = fill.groupby("gaia_id", as_index=False).first()
            n_abs_anchors_sum = int(np.nansum(g["n_abs_anchors"].to_numpy(dtype=float)))
            median_anchors = float(np.nanmedian(g["n_abs_anchors"].to_numpy(dtype=float)))
        else:
            median_anchors = float("nan")
        if "relative_only" in fill.columns:
            g = fill.groupby("gaia_id", as_index=False).first()
            n_relative_only = int(np.sum(g["relative_only"].astype(bool)))
        if "float_zeropoint" in fill.columns:
            g = fill.groupby("gaia_id", as_index=False).first()
            n_float_zp = int(np.sum(g["float_zeropoint"].astype(bool)))
    else:
        median_anchors = float("nan")

    if len(meta_df) and "n_abs_anchors" in meta_df.columns:
        median_anchors_meta = float(np.nanmedian(meta_df["n_abs_anchors"].to_numpy(dtype=float)))
        n_stars_with_anchors = int(
            np.sum(np.asarray(meta_df["n_abs_anchors"], dtype=float) > 0)
        )
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
        "residual_median_kms": float(np.nanmedian(residual)) if residual.size else float("nan"),
        "residual_mad_kms": _mad(residual),
        "residual_rms_kms": _rms(residual),
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


def per_star_discordant(vs_abs: pd.DataFrame) -> pd.DataFrame:
    if vs_abs.empty or "gaia_id" not in vs_abs.columns:
        return pd.DataFrame(
            columns=[
                "gaia_id",
                "n_comparisons",
                "n_discordant",
                "discordant_fraction",
                "residual_rms_kms",
                "residual_median_kms",
            ]
        )
    disc_col = "epoch_ccf_abs_rel_discordant"
    rows = []
    for gid, g in vs_abs.groupby("gaia_id"):
        n = len(g)
        if disc_col in g.columns:
            nd = int(g[disc_col].astype(bool).sum())
        else:
            nd = 0
        res = g["residual_kms"].to_numpy(dtype=float) if "residual_kms" in g.columns else np.array([])
        rows.append(
            {
                "gaia_id": gid,
                "n_comparisons": n,
                "n_discordant": nd,
                "discordant_fraction": float(nd / n) if n else float("nan"),
                "residual_rms_kms": _rms(res),
                "residual_median_kms": float(np.nanmedian(res)) if res.size else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["discordant_fraction", "n_discordant", "residual_rms_kms"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def make_plots(
    pairs: pd.DataFrame,
    vs_abs: pd.DataFrame,
    per_star: pd.DataFrame,
    out_dir: Path,
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

    # 1. residual histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if residual.size:
        finite = residual[np.isfinite(residual)]
        # clip display range for readability while noting outliers
        lo, hi = np.nanpercentile(finite, [0.5, 99.5]) if finite.size > 20 else (finite.min(), finite.max())
        span = max(hi - lo, 1.0)
        bins = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 60)
        ax.hist(finite, bins=bins, color="steelblue", edgecolor="white", alpha=0.9)
        ax.axvline(0.0, color="k", lw=1.0, ls="--")
        ax.axvline(np.median(finite), color="C3", lw=1.2, label=f"median={np.median(finite):.2f}")
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
        ax.scatter(x[ok & ~disc], y[ok & ~disc], s=12, alpha=0.45, c="steelblue", label="ok", rasterized=True)
        ax.scatter(x[ok & disc], y[ok & disc], s=18, alpha=0.7, c="C3", label="discordant", rasterized=True)
        # ±3σ guide if discord threshold known
        xs = np.linspace(max(np.nanmin(x[ok]), 1e-3), np.nanmax(x[ok]), 100) if ok.any() else None
        if xs is not None and "discord_n_sigma" in vs_abs.columns:
            thr = float(np.nanmedian(vs_abs["discord_n_sigma"].to_numpy(dtype=float)))
            ax.plot(xs, thr * xs, "k--", lw=1, label=f"±{thr:.0f}σ")
            ax.plot(xs, -thr * xs, "k--", lw=1)
        ax.legend(frameon=False, loc="best")
        ax.set_xlabel(r"$\sigma_{\mathrm{combined}}$ (km/s)")
    elif len(vs_abs) and "n_sigma_residual" in vs_abs.columns:
        x = vs_abs["n_sigma_residual"].to_numpy(dtype=float)
        y = residual
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok & ~disc], y[ok & ~disc], s=12, alpha=0.45, c="steelblue", label="ok", rasterized=True)
        ax.scatter(x[ok & disc], y[ok & disc], s=18, alpha=0.7, c="C3", label="discordant", rasterized=True)
        ax.legend(frameon=False)
        ax.set_xlabel(r"$n_\sigma$ residual")
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
        ax.scatter(x[ok & ~disc], y[ok & ~disc], s=12, alpha=0.45, c="steelblue", label="ok", rasterized=True)
        ax.scatter(x[ok & disc], y[ok & disc], s=18, alpha=0.7, c="C3", label="discordant", rasterized=True)
        if ok.any():
            lo = float(np.nanmin([x[ok].min(), y[ok].min()]))
            hi = float(np.nanmax([x[ok].max(), y[ok].max()]))
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
            # often near zero; use log-friendly small bins or linear with tight range
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

    # 5. per-star discordant fraction bar (top offenders)
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

    return written


def write_report(
    summary: dict,
    per_star: pd.DataFrame,
    plot_paths: list[Path],
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
        f"MAD={summary['residual_mad_kms']:.4g} km/s, "
        f"RMS={summary['residual_rms_kms']:.4g} km/s",
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
    lines.extend(
        [
            "",
            "## Top discordant stars",
            "",
        ]
    )
    if len(per_star):
        top = per_star.head(15)
        lines.append("| gaia_id | n_comp | n_disc | frac | residual RMS |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in top.iterrows():
            lines.append(
                f"| {r['gaia_id']} | {int(r['n_comparisons'])} | {int(r['n_discordant'])} | "
                f"{r['discordant_fraction']:.3f} | {r['residual_rms_kms']:.3g} |"
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
    args = ap.parse_args(argv)

    root = args.epoch_ccf_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: epoch-ccf-root not found: {root}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs, vs_abs, fill, meta_df = load_campaign(root)

    pairs.to_csv(out_dir / "aggregated_pairs.csv", index=False)
    vs_abs.to_csv(out_dir / "aggregated_vs_abs_delta.csv", index=False)
    fill.to_csv(out_dir / "aggregated_abs_fill.csv", index=False)
    meta_df.to_csv(out_dir / "aggregated_meta.csv", index=False)

    summary = compute_summary(pairs, vs_abs, fill, meta_df)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    per_star = per_star_discordant(vs_abs)
    per_star.to_csv(out_dir / "per_star_discordant.csv", index=False)

    plot_paths = make_plots(pairs, vs_abs, per_star, out_dir)
    report_path = write_report(summary, per_star, plot_paths, out_dir, root)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote report: {report_path}")
    for p in plot_paths:
        print(f"Wrote plot: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
