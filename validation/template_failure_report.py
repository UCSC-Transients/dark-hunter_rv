#!/usr/bin/env python3
"""
Phase 0 — template disagreement stratification (S/N-first).

Reads ``overlap_enriched_per_exposure.csv`` from ``rv_method_overlap_report``, ranks exposures
by ``median_mask_ccf_peak_snr``, and classifies three-method disagreements. Default cohort is the
top S/N percentile with mask + template + strong_lines all valid (the “should be easy” set).

Writes under ``--out-dir``:

* ``phase0_exposure_table.csv`` — full overlap table + failure class and |ΔRV| columns
* ``phase0_cohort_exposures.csv`` — rows in the active S/N cohort
* ``phase0_per_chunk_residuals.csv`` — mask vs template per ``chunk_key`` (cohort only)
* ``phase0_snr_sweep.csv`` — disagreement rates vs S/N floor (lower threshold progressively)
* ``phase0_summary.md`` — human-readable counts and next-step stem lists
* ``cohort_class_B_template_outlier_stems.txt`` — template wrong, mask ≈ strong (triage)
* ``cohort_class_B_template_outlier_paths.txt`` — full spectrum paths when ``--spectrum-list`` set

Example (high-S/N first — top 25% by mask CCF S/N)::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.template_failure_report \\
    --overlap-csv validation_output/template_fft_baseline/overlap/overlap_enriched_per_exposure.csv \\
    --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \\
    --out-dir validation_output/template_fft_baseline/phase0_high_snr \\
    --snr-percentile-floor 75

Lower the bar (median S/N cohort) — pass all required flags (no ``...``)::

  python3 -m validation.template_failure_report \\
    --overlap-csv validation_output/template_fft_baseline/overlap/overlap_enriched_per_exposure.csv \\
    --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \\
    --out-dir validation_output/template_fft_baseline/phase0_median_snr \\
    --snr-percentile-floor 50

Hβ diagnostics (mask + template + Voigt strong-line on one axes) for the cohort::

  python3 -m validation.plot_cohort_hbeta_methods \\
    --cohort-csv validation_output/template_fft_baseline/phase0_high_snr/phase0_cohort_exposures.csv \\
    --out-dir validation_output/template_fft_baseline/phase0_high_snr/hbeta_method_plots \\
    --data-root /Users/rfoley/darkhunter/rvs/data

Or append ``--plot-hbeta-methods --data-root /Users/rfoley/darkhunter/rvs/data`` to the report command.
"""
from __future__ import annotations

import argparse
import logging
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

from darkhunter_rv import config as dh_config

logger = logging.getLogger(__name__)

FAILURE_CLASS_LABELS = {
    "D_agree": "All three agree within threshold",
    "B_template_outlier": "Template off; mask ≈ strong (Hβ anchor)",
    "B_mask_outlier": "Mask off; template ≈ strong",
    "C_mask_template_split": "Template ≈ strong; mask disagrees with both",
    "A_all_disagree": "All pairwise deltas above threshold",
    "E_incomplete": "Not all three methods valid",
    "E_no_strong": "Mask+template valid but strong_lines missing",
}


def _stem_from_row(row: pd.Series) -> str:
    bn = str(row.get("basename", "") or "")
    if bn.endswith(".txt"):
        return Path(bn).stem
    diag = str(row.get("diagnostics_csv", "") or "")
    if diag:
        return Path(diag).stem.replace("_diagnostics", "")
    return ""


def _finite_abs(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return abs(v) if np.isfinite(v) else float("nan")


def classify_three_method_exposure(
    row: pd.Series,
    *,
    threshold_kms: float,
) -> str:
    """
    Classify one exposure when mask, template, and strong_lines validity flags are known.

    Priority: incomplete → agreement → template-outlier (mask≈strong) → other patterns.
    """
    mask_ok = bool(row.get("mask_valid", False))
    tpl_ok = bool(row.get("template_valid", False))
    sl_ok = bool(row.get("strong_lines_valid", False))
    if not (mask_ok and tpl_ok and sl_ok):
        if mask_ok and tpl_ok and not sl_ok:
            return "E_no_strong"
        return "E_incomplete"

    t = float(threshold_kms)
    d_mt = _finite_abs(row.get("delta_mask_minus_template_kms", np.nan))
    d_ms = _finite_abs(row.get("delta_mask_minus_strong_lines_kms", np.nan))
    d_ts = _finite_abs(row.get("delta_template_minus_strong_lines_kms", np.nan))

    if max(d_mt, d_ms, d_ts) <= t:
        return "D_agree"

    if d_ts > t and d_ms <= t:
        return "B_template_outlier"
    if d_ms > t and d_ts <= t:
        return "B_mask_outlier"
    if d_ts <= t and d_mt > t:
        return "C_mask_template_split"
    return "A_all_disagree"


def enrich_overlap_table(tab: pd.DataFrame, *, threshold_kms: float) -> pd.DataFrame:
    out = tab.copy()
    out["stem"] = out.apply(_stem_from_row, axis=1)
    out["failure_class"] = out.apply(
        lambda r: classify_three_method_exposure(r, threshold_kms=threshold_kms),
        axis=1,
    )
    out["abs_delta_mask_template_kms"] = out["delta_mask_minus_template_kms"].apply(_finite_abs)
    out["abs_delta_mask_strong_kms"] = out["delta_mask_minus_strong_lines_kms"].apply(_finite_abs)
    out["abs_delta_template_strong_kms"] = out["delta_template_minus_strong_lines_kms"].apply(_finite_abs)
    out["max_pairwise_abs_delta_kms"] = out[
        ["abs_delta_mask_template_kms", "abs_delta_mask_strong_kms", "abs_delta_template_strong_kms"]
    ].max(axis=1)
    return out


def snr_floor_from_percentile(tab: pd.DataFrame, percentile: float) -> float:
    snr = tab["median_mask_ccf_peak_snr"].astype(float)
    snr = snr[np.isfinite(snr) & (snr > 0)]
    if snr.empty:
        return float("nan")
    q = float(np.clip(percentile, 0.0, 100.0)) / 100.0
    return float(np.quantile(snr, q))


def select_cohort(
    tab: pd.DataFrame,
    *,
    snr_percentile_floor: float | None,
    min_log10_median_mask_ccf_peak_snr: float | None,
    min_median_mask_ccf_peak_snr: float | None,
    require_three_methods: bool,
    max_method_err_kms: float,
) -> tuple[pd.DataFrame, float]:
    """Return cohort subset and the S/N floor (linear median_mask_ccf_peak_snr) used."""
    work = tab.copy()
    if require_three_methods:
        work = work[
            work["mask_valid"].astype(bool)
            & work["template_valid"].astype(bool)
            & work["strong_lines_valid"].astype(bool)
        ]

    if min_median_mask_ccf_peak_snr is not None:
        floor = float(min_median_mask_ccf_peak_snr)
    elif min_log10_median_mask_ccf_peak_snr is not None:
        floor = float(10 ** float(min_log10_median_mask_ccf_peak_snr))
    elif snr_percentile_floor is not None:
        floor = snr_floor_from_percentile(tab, float(snr_percentile_floor))
    else:
        floor = snr_floor_from_percentile(tab, 75.0)

    if not np.isfinite(floor):
        return work.iloc[0:0].copy(), float("nan")

    m_snr = work["median_mask_ccf_peak_snr"].astype(float) >= floor
    work = work.loc[m_snr].copy()

    for col, key in (
        ("mask_err_kms", "mask_valid"),
        ("template_err_kms", "template_valid"),
        ("strong_lines_err_kms", "strong_lines_valid"),
    ):
        if col not in work.columns:
            continue
        err = work[col].astype(float)
        valid = work[key].astype(bool) if key in work.columns else pd.Series(True, index=work.index)
        work = work.loc[valid & (err <= float(max_method_err_kms)) | ~valid]

    return work, floor


def build_per_chunk_residuals(cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, exp in cohort.iterrows():
        diag = Path(str(exp["diagnostics_csv"]))
        if not diag.is_file():
            logger.warning("skip missing diagnostics %s", diag)
            continue
        try:
            df = pd.read_csv(diag)
        except Exception as ex:
            logger.warning("skip %s: %s", diag, ex)
            continue
        mask = df[df["method"].astype(str) == "mask_ccf"].copy()
        mask = mask[mask["chunk_key"].astype(str) != "all"]
        tpl = df[df["method"].astype(str) == "template_fft"].copy()
        tpl = tpl[tpl["chunk_key"].astype(str) != "all"]
        if mask.empty or tpl.empty:
            continue
        m = mask.set_index("chunk_key", drop=False)
        t = tpl.set_index("chunk_key", drop=False)
        keys = sorted(set(m.index.astype(str)) & set(t.index.astype(str)))
        stem = _stem_from_row(exp)
        for ck in keys:
            rm = m.loc[ck]
            rt = t.loc[ck]
            if isinstance(rm, pd.DataFrame):
                rm = rm.iloc[0]
            if isinstance(rt, pd.DataFrame):
                rt = rt.iloc[0]
            rv_m = float(rm.get("rv_kms", np.nan))
            rv_t = float(rt.get("rv_kms", np.nan))
            if not np.isfinite(rv_m) or not np.isfinite(rv_t):
                continue
            rows.append(
                {
                    "stem": stem,
                    "diagnostics_csv": str(diag),
                    "chunk_key": ck,
                    "delta_template_minus_mask_kms": rv_t - rv_m,
                    "abs_delta_template_minus_mask_kms": abs(rv_t - rv_m),
                    "mask_rv_kms": rv_m,
                    "template_rv_kms": rv_t,
                    "mask_ccf_peak_snr": float(rm.get("ccf_peak_snr", np.nan)),
                    "mask_line_count": float(rm.get("mask_line_count", np.nan)),
                    "telluric_fraction": float(rm.get("telluric_fraction", np.nan)),
                    "mask_qc_pass": bool(rm.get("qc_pass", True)),
                    "template_qc_pass": bool(rt.get("qc_pass", True)),
                    "fft_ccf_rss_ratio": float(rt.get("fft_ccf_rss_ratio", np.nan)),
                    "template_key": str(rt.get("template_key", "") or ""),
                    "median_mask_ccf_peak_snr": float(exp.get("median_mask_ccf_peak_snr", np.nan)),
                    "failure_class": str(exp.get("failure_class", "")),
                }
            )
    return pd.DataFrame(rows)


def snr_sweep_table(
    tab: pd.DataFrame,
    *,
    threshold_kms: float,
    percentiles: list[float],
    require_three_methods: bool,
    max_method_err_kms: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for pct in percentiles:
        cohort, floor = select_cohort(
            tab,
            snr_percentile_floor=pct,
            min_log10_median_mask_ccf_peak_snr=None,
            min_median_mask_ccf_peak_snr=None,
            require_three_methods=require_three_methods,
            max_method_err_kms=max_method_err_kms,
        )
        n = len(cohort)
        if n == 0:
            rows.append(
                {
                    "snr_percentile_floor": pct,
                    "median_mask_ccf_peak_snr_floor": floor,
                    "n_exposures": 0,
                    "n_D_agree": 0,
                    "n_B_template_outlier": 0,
                    "frac_B_template_outlier": float("nan"),
                    "n_B_mask_outlier": 0,
                    "n_A_all_disagree": 0,
                    "median_max_pairwise_abs_delta_kms": float("nan"),
                }
            )
            continue
        vc = cohort["failure_class"].value_counts()
        rows.append(
            {
                "snr_percentile_floor": pct,
                "median_mask_ccf_peak_snr_floor": floor,
                "log10_snr_floor": float(np.log10(floor)) if floor > 0 else float("nan"),
                "n_exposures": n,
                "n_D_agree": int(vc.get("D_agree", 0)),
                "n_B_template_outlier": int(vc.get("B_template_outlier", 0)),
                "frac_B_template_outlier": float(vc.get("B_template_outlier", 0)) / n,
                "n_B_mask_outlier": int(vc.get("B_mask_outlier", 0)),
                "n_A_all_disagree": int(vc.get("A_all_disagree", 0)),
                "median_max_pairwise_abs_delta_kms": float(
                    cohort["max_pairwise_abs_delta_kms"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_stem_to_path(spectrum_list: Path | None) -> dict[str, str]:
    if spectrum_list is None or not spectrum_list.is_file():
        return {}
    out: dict[str, str] = {}
    for line in spectrum_list.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out[Path(line).stem] = line
    return out


def _write_stem_and_path_lists(
    cohort: pd.DataFrame,
    out_dir: Path,
    stem_to_path: dict[str, str],
    *,
    class_name: str,
    stem_fname: str,
    path_fname: str,
) -> None:
    sub = cohort[cohort["failure_class"].astype(str) == class_name]
    stems = [s for s in sub["stem"].astype(str).tolist() if s]
    out_dir.joinpath(stem_fname).write_text("\n".join(stems) + ("\n" if stems else ""))
    paths = [stem_to_path[s] for s in stems if s in stem_to_path]
    if paths:
        out_dir.joinpath(path_fname).write_text("\n".join(paths) + "\n")


def write_summary_md(
    path: Path,
    *,
    cohort: pd.DataFrame,
    full: pd.DataFrame,
    snr_floor: float,
    threshold_kms: float,
    snr_percentile_floor: float | None,
    sweep: pd.DataFrame,
    per_chunk: pd.DataFrame,
    out_dir: Path,
) -> None:
    lines: list[str] = [
        "# Template failure Phase 0 (S/N-first)",
        "",
        f"- Discrepancy threshold: **{threshold_kms:.1f} km/s**",
        f"- Cohort S/N floor: **median mask CCF peak S/N ≥ {snr_floor:.3f}**"
        + (
            f" (≥{snr_percentile_floor:.0f}th percentile of full overlap table)"
            if snr_percentile_floor is not None
            else ""
        ),
        f"- Cohort size: **{len(cohort)}** exposures (three methods valid, σ ≤ report cap)",
        f"- Full overlap table: **{len(full)}** exposures",
        "",
        "## Failure classes (cohort)",
        "",
    ]
    for cls, n in cohort["failure_class"].value_counts().items():
        desc = FAILURE_CLASS_LABELS.get(str(cls), "")
        lines.append(f"- `{cls}`: **{int(n)}** — {desc}")
    lines.extend(["", "## Priority triage", ""])
    n_b = int((cohort["failure_class"] == "B_template_outlier").sum())
    lines.append(
        f"- **`B_template_outlier`** ({n_b}): template disagrees with Hβ-strong-line anchor while "
        "mask agrees — see ``hbeta_method_plots/*_h_beta_methods.png`` and "
        "`plot_method_discrep_lines` / `diagnose_template_fft_star` on "
        f"`{out_dir.name}/cohort_class_B_template_outlier_stems.txt`."
    )
    lines.extend(["", "## S/N sweep (lower the floor to see failures appear)", "", "```"])
    lines.append(sweep.to_string(index=False))
    lines.extend(["```", ""])
    if len(per_chunk) > 0:
        b_stems = set(
            cohort.loc[cohort["failure_class"] == "B_template_outlier", "stem"].astype(str).tolist()
        )
        pc_b = per_chunk[per_chunk["stem"].astype(str).isin(b_stems)]
        if len(pc_b) > 0:
            lines.extend(
                [
                    "## Per-chunk (class B template outliers)",
                    "",
                    f"- Chunks: **{len(pc_b)}**",
                    f"- Median |template−mask| per chunk: **{pc_b['abs_delta_template_minus_mask_kms'].median():.2f} km/s**",
                    f"- 90th %ile |template−mask|: **{pc_b['abs_delta_template_minus_mask_kms'].quantile(0.9):.2f} km/s**",
                    "",
                ]
            )
    path.write_text("\n".join(lines) + "\n")


def plot_snr_sweep(sweep: pd.DataFrame, out_path: Path, *, threshold_kms: float) -> None:
    if sweep.empty or "frac_B_template_outlier" not in sweep.columns:
        return
    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    x = sweep["snr_percentile_floor"].astype(float)
    ax1.plot(x, sweep["frac_B_template_outlier"].astype(float) * 100.0, "o-", color="tab:red", label="B template outlier %")
    ax1.plot(x, sweep["n_exposures"].astype(float), "s--", color="0.35", label="n exposures")
    ax1.set_xlabel("S/N percentile floor (median mask CCF peak S/N)")
    ax1.set_ylabel("Fraction / count")
    ax1.set_title(f"Template failure vs S/N floor (|T|>{threshold_kms:.0f} km/s)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_cohort_residuals(cohort: pd.DataFrame, out_path: Path) -> None:
    sub = cohort[
        cohort["mask_valid"].astype(bool)
        & cohort["template_valid"].astype(bool)
        & cohort["strong_lines_valid"].astype(bool)
    ].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    pairs = [
        ("abs_delta_mask_template_kms", "mask − template"),
        ("abs_delta_template_strong_kms", "template − strong"),
        ("abs_delta_mask_strong_kms", "mask − strong"),
    ]
    colors = cohort["failure_class"].astype(str).map(
        {
            "D_agree": "0.55",
            "B_template_outlier": "tab:red",
            "B_mask_outlier": "tab:blue",
            "A_all_disagree": "tab:orange",
            "C_mask_template_split": "tab:purple",
        }
    ).fillna("0.75")
    snr = sub["median_mask_ccf_peak_snr"].astype(float)
    for ax, (col, title) in zip(axes, pairs):
        y = sub[col].astype(float)
        ax.scatter(snr, y, c=colors.loc[sub.index], s=28, alpha=0.85, edgecolors="none")
        ax.set_xlabel("median mask CCF peak S/N")
        ax.set_ylabel(f"|{title}| (km/s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
    fig.suptitle("Three-method |ΔRV| vs S/N (cohort colors by failure class)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 0 template failure report (S/N-first cohort)")
    ap.add_argument("--overlap-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--spectrum-list",
        type=Path,
        default=None,
        help="Campaign spectrum list (full paths) for path stem lists",
    )
    ap.add_argument("--threshold-kms", type=float, default=5.0, help="Pairwise agreement threshold")
    ap.add_argument(
        "--snr-percentile-floor",
        type=float,
        default=75.0,
        help="Cohort: median_mask_ccf_peak_snr >= this percentile of overlap table (default 75 = top quartile)",
    )
    ap.add_argument(
        "--min-log10-median-mask-ccf-peak-snr",
        type=float,
        default=None,
        help="Override percentile: explicit log10(S/N) floor",
    )
    ap.add_argument(
        "--min-median-mask-ccf-peak-snr",
        type=float,
        default=None,
        help="Override percentile: explicit linear S/N floor",
    )
    ap.add_argument(
        "--no-require-three-methods",
        action="store_true",
        help="Include exposures missing template or strong_lines in cohort",
    )
    ap.add_argument(
        "--max-method-err",
        type=float,
        default=float(dh_config.COMPARISON_REPORT_MAX_RV_ERR_KMS),
    )
    ap.add_argument(
        "--sweep-percentiles",
        type=str,
        default="90,75,50,25,10",
        help="Comma-separated percentiles for phase0_snr_sweep.csv",
    )
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--plot-hbeta-methods",
        action="store_true",
        help="After tables, write Hβ mask+template+strong-lines PNGs for the cohort",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Spectrum data root for Hβ plots (e.g. /Users/rfoley/darkhunter/rvs/data)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    if not args.overlap_csv.is_file():
        logging.error("overlap CSV not found: %s", args.overlap_csv)
        return 2

    tab = pd.read_csv(args.overlap_csv)
    full = enrich_overlap_table(tab, threshold_kms=float(args.threshold_kms))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.out_dir / "phase0_exposure_table.csv", index=False)

    cohort, snr_floor = select_cohort(
        full,
        snr_percentile_floor=float(args.snr_percentile_floor)
        if args.min_log10_median_mask_ccf_peak_snr is None
        and args.min_median_mask_ccf_peak_snr is None
        else None,
        min_log10_median_mask_ccf_peak_snr=args.min_log10_median_mask_ccf_peak_snr,
        min_median_mask_ccf_peak_snr=args.min_median_mask_ccf_peak_snr,
        require_three_methods=not bool(args.no_require_three_methods),
        max_method_err_kms=float(args.max_method_err),
    )
    cohort.to_csv(args.out_dir / "phase0_cohort_exposures.csv", index=False)
    logging.info(
        "Cohort: %d exposures (S/N floor=%.3f, three_methods=%s)",
        len(cohort),
        snr_floor,
        not args.no_require_three_methods,
    )

    sweep_pcts = [float(x.strip()) for x in str(args.sweep_percentiles).split(",") if x.strip()]
    sweep = snr_sweep_table(
        full,
        threshold_kms=float(args.threshold_kms),
        percentiles=sweep_pcts,
        require_three_methods=not bool(args.no_require_three_methods),
        max_method_err_kms=float(args.max_method_err),
    )
    sweep.to_csv(args.out_dir / "phase0_snr_sweep.csv", index=False)

    per_chunk = build_per_chunk_residuals(cohort)
    per_chunk.to_csv(args.out_dir / "phase0_per_chunk_residuals.csv", index=False)

    stem_to_path = _load_stem_to_path(args.spectrum_list)
    _write_stem_and_path_lists(
        cohort,
        args.out_dir,
        stem_to_path,
        class_name="B_template_outlier",
        stem_fname="cohort_class_B_template_outlier_stems.txt",
        path_fname="cohort_class_B_template_outlier_paths.txt",
    )
    _write_stem_and_path_lists(
        cohort,
        args.out_dir,
        stem_to_path,
        class_name="D_agree",
        stem_fname="cohort_class_D_agree_stems.txt",
        path_fname="cohort_class_D_agree_paths.txt",
    )

    write_summary_md(
        args.out_dir / "phase0_summary.md",
        cohort=cohort,
        full=full,
        snr_floor=snr_floor,
        threshold_kms=float(args.threshold_kms),
        snr_percentile_floor=float(args.snr_percentile_floor)
        if args.min_log10_median_mask_ccf_peak_snr is None
        and args.min_median_mask_ccf_peak_snr is None
        else None,
        sweep=sweep,
        per_chunk=per_chunk,
        out_dir=args.out_dir,
    )

    if not args.no_plots:
        plot_snr_sweep(sweep, args.out_dir / "phase0_snr_sweep.png", threshold_kms=float(args.threshold_kms))
        plot_cohort_residuals(cohort, args.out_dir / "phase0_cohort_residuals.png")

    if args.plot_hbeta_methods:
        from validation.plot_cohort_hbeta_methods import plot_one_exposure_hbeta

        hbeta_dir = args.out_dir / "hbeta_method_plots"
        hbeta_dir.mkdir(parents=True, exist_ok=True)
        n_hb = 0
        for _, row in cohort.iterrows():
            out_png = hbeta_dir / f"{row['stem']}_h_beta_methods.png"
            try:
                if plot_one_exposure_hbeta(
                    row,
                    out_png,
                    instrument_name="APF",
                    continuum_mode="spline",
                    data_root=args.data_root,
                ):
                    n_hb += 1
            except Exception as ex:
                logging.warning("hbeta plot failed %s: %s", row.get("stem"), ex)
        logging.info("Wrote %d Hβ method PNGs -> %s", n_hb, hbeta_dir.resolve())

    logging.info("Wrote Phase 0 report -> %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
