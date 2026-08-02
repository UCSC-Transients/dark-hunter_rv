#!/usr/bin/env python3
"""
Compare pipeline mask/template (optional strong_lines) RVs to El-Badry literature master.

Lite path (step 08): join on ``gaia_dr3_id`` + nearest BJD/MJD; write per-epoch ΔRV and
per-star bias/RMS tables. Does **not** ingest LAMOST/RAVE (08-full).

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.compare_literature_rvs \\
    --master calibration/literature_rv_master.csv \\
    --diagnostics-glob \\
      '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/validation_output/template_fft_baseline/pipeline_cool_vsini12_mhfix/*_diagnostics.csv' \\
    --report-dir validation_output/literature_crosscheck_lite \\
    --copy-key-table calibration/literature_crosscheck_lite/per_star_bias_rms.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from glob import glob as glob_paths
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv.method_evaluation import exposure_method_flags  # noqa: E402
from validation.rv_overlap_lib import load_literature_epochs, mjd_to_bjd  # noqa: E402

METHOD_ALIASES: dict[str, tuple[str, str, str]] = {
    # cli_name -> (flags_rv_key, flags_err_key, flags_valid_key)
    "mask_ccf": ("mask_rv_kms", "mask_err_kms", "mask_valid"),
    "template_fft": ("template_rv_kms", "template_err_kms", "template_valid"),
    "strong_lines": ("strong_lines_rv_kms", "strong_lines_err_kms", "strong_lines_valid"),
}

DEFAULT_METHODS: tuple[str, ...] = ("mask_ccf", "template_fft")
_GAIA_RE = re.compile(r"Gaia_DR3_(\d{15,25})")


def parse_gaia_id_from_path(path: Path | str) -> str | None:
    """Extract Gaia DR3 source id from a diagnostics path or basename."""
    m = _GAIA_RE.search(Path(path).name)
    return m.group(1) if m else None


def _rows_from_diagnostics_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    chunk = df
    if "chunk_key" in df.columns:
        chunk = df[df["chunk_key"].astype(str) != "all"]
    return chunk.to_dict("records")


def load_pipeline_epochs_from_diagnostics(
    diagnostics_glob: str,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
) -> pd.DataFrame:
    """
    Build one row per (exposure, method) from ``*_diagnostics.csv`` files.

    Uses :func:`darkhunter_rv.method_evaluation.exposure_method_flags` so mask/template
    RVs match pipeline weighting (including dual continuum lanes when both are present).
    """
    wanted = [m.strip() for m in methods if m.strip()]
    unknown = [m for m in wanted if m not in METHOD_ALIASES]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose from {sorted(METHOD_ALIASES)}")

    paths = sorted(glob_paths(diagnostics_glob))
    rows: list[dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str)
        gaia_id = parse_gaia_id_from_path(path)
        if not gaia_id:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        recs = _rows_from_diagnostics_df(df)
        if not recs:
            continue
        flags = exposure_method_flags(recs)
        mjd = float(pd.to_numeric(df["mjd"], errors="coerce").dropna().iloc[0]) if "mjd" in df.columns else float("nan")
        if not np.isfinite(mjd):
            continue
        teff = float(pd.to_numeric(df["teff"], errors="coerce").dropna().iloc[0]) if "teff" in df.columns else float("nan")
        basename = path.name.replace("_diagnostics.csv", "")
        file_col = str(df["file"].iloc[0]) if "file" in df.columns else basename
        for method in wanted:
            rv_k, er_k, ok_k = METHOD_ALIASES[method]
            rv = float(flags.get(rv_k, np.nan))
            err = float(flags.get(er_k, np.nan))
            if not bool(flags.get(ok_k, False)):
                continue
            if not np.isfinite(rv):
                continue
            rows.append(
                {
                    "gaia_dr3_id": gaia_id,
                    "basename": basename,
                    "file": file_col,
                    "diagnostics_path": str(path),
                    "method": method,
                    "mjd": mjd,
                    "bjd": mjd_to_bjd(mjd),
                    "rv_kms": rv,
                    "rv_err_kms": err if np.isfinite(err) else np.nan,
                    "teff": teff,
                    "epoch_id": f"pipeline:{basename}:{method}",
                }
            )
    return pd.DataFrame(rows)


def nearest_literature_join(
    pipeline: pd.DataFrame,
    literature: pd.DataFrame,
    *,
    max_delta_days: float | None = None,
) -> pd.DataFrame:
    """
    For each pipeline epoch, attach the nearest literature epoch by |ΔMJD| on the same Gaia id.

    Parameters
    ----------
    pipeline:
        Rows with ``gaia_dr3_id``, ``mjd``, ``rv_kms``, ``method``.
    literature:
        Master epochs from :func:`load_literature_epochs`.
    max_delta_days:
        If set, drop pairs with |ΔMJD| above this threshold.
    """
    if pipeline.empty or literature.empty:
        return pd.DataFrame()

    lit = literature.copy()
    lit["gaia_dr3_id"] = lit["gaia_dr3_id"].astype(str)
    lit["mjd"] = pd.to_numeric(lit["mjd"], errors="coerce")
    lit["rv_kms"] = pd.to_numeric(lit["rv_kms"], errors="coerce")
    lit["rv_err_kms"] = pd.to_numeric(lit["rv_err_kms"], errors="coerce")

    out_rows: list[dict[str, Any]] = []
    for _, prow in pipeline.iterrows():
        gid = str(prow["gaia_dr3_id"])
        pmjd = float(prow["mjd"])
        if not np.isfinite(pmjd):
            continue
        lg = lit[lit["gaia_dr3_id"] == gid]
        lg = lg[np.isfinite(lg["mjd"].astype(float)) & np.isfinite(lg["rv_kms"].astype(float))]
        if lg.empty:
            continue
        deltas = (lg["mjd"].astype(float) - pmjd).abs()
        j = int(deltas.to_numpy().argmin())
        lit_row = lg.iloc[j]
        delta_days = float(deltas.iloc[j])
        if max_delta_days is not None and delta_days > float(max_delta_days):
            continue
        pipe_rv = float(prow["rv_kms"])
        lit_rv = float(lit_row["rv_kms"])
        pipe_err = float(prow.get("rv_err_kms", np.nan))
        lit_err = float(lit_row["rv_err_kms"]) if pd.notna(lit_row["rv_err_kms"]) else float("nan")
        out_rows.append(
            {
                "gaia_dr3_id": gid,
                "name": str(lit_row.get("name", "")),
                "method": str(prow["method"]),
                "pipeline_basename": str(prow.get("basename", "")),
                "pipeline_mjd": pmjd,
                "pipeline_bjd": float(prow.get("bjd", mjd_to_bjd(pmjd))),
                "pipeline_rv_kms": pipe_rv,
                "pipeline_rv_err_kms": pipe_err,
                "literature_epoch_id": str(lit_row.get("epoch_id", "")),
                "literature_bjd": float(lit_row["bjd"]) if pd.notna(lit_row.get("bjd")) else np.nan,
                "literature_mjd": float(lit_row["mjd"]),
                "literature_rv_kms": lit_rv,
                "literature_rv_err_kms": lit_err,
                "literature_instrument": str(lit_row.get("instrument", "")),
                "reference_key": str(lit_row.get("reference_key", "")),
                "delta_days": delta_days,
                "delta_rv_kms": pipe_rv - lit_rv,
                "abs_delta_rv_kms": abs(pipe_rv - lit_rv),
                "P_orb_days": lit_row.get("P_orb_days", np.nan),
                "M_star_msun": lit_row.get("M_star_msun", np.nan),
                "M2_msun": lit_row.get("M2_msun", np.nan),
                "eccentricity": lit_row.get("eccentricity", np.nan),
                "diagnostics_path": str(prow.get("diagnostics_path", "")),
            }
        )
    return pd.DataFrame(out_rows)


def per_star_bias_rms(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per (gaia_dr3_id, method) mean bias and RMS of pipeline − literature ΔRV."""
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "gaia_dr3_id",
                "name",
                "method",
                "n_epochs",
                "bias_kms",
                "rms_kms",
                "median_abs_delta_kms",
                "median_delta_days",
                "P_orb_days",
                "M_star_msun",
                "M2_msun",
            ]
        )

    rows: list[dict[str, Any]] = []
    group_cols = ["gaia_dr3_id", "method"]
    for (gid, method), g in pairs.groupby(group_cols, sort=True):
        dv = g["delta_rv_kms"].astype(float).to_numpy()
        dv = dv[np.isfinite(dv)]
        if len(dv) == 0:
            continue
        name = str(g["name"].dropna().iloc[0]) if g["name"].notna().any() else ""
        rows.append(
            {
                "gaia_dr3_id": str(gid),
                "name": name,
                "method": str(method),
                "n_epochs": int(len(dv)),
                "bias_kms": float(np.mean(dv)),
                "rms_kms": float(np.sqrt(np.mean(dv**2))),
                "median_abs_delta_kms": float(np.median(np.abs(dv))),
                "median_delta_days": float(np.median(g["delta_days"].astype(float))),
                "P_orb_days": g["P_orb_days"].dropna().iloc[0] if g["P_orb_days"].notna().any() else np.nan,
                "M_star_msun": g["M_star_msun"].dropna().iloc[0] if g["M_star_msun"].notna().any() else np.nan,
                "M2_msun": g["M2_msun"].dropna().iloc[0] if g["M2_msun"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def orbit_qa_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per overlap star with published orbit params from the master CSV."""
    if pairs.empty:
        return pd.DataFrame()
    cols = [
        "gaia_dr3_id",
        "name",
        "P_orb_days",
        "M_star_msun",
        "M2_msun",
        "eccentricity",
        "reference_key",
    ]
    present = [c for c in cols if c in pairs.columns]
    out = pairs[present].drop_duplicates(subset=["gaia_dr3_id"]).sort_values("gaia_dr3_id")
    return out.reset_index(drop=True)


def _write_report_md(
    report_dir: Path,
    *,
    n_pipeline_epochs: int,
    n_stars: int,
    methods: Sequence[str],
    pairs: pd.DataFrame,
    per_star: pd.DataFrame,
    diagnostics_glob: str,
    master_path: Path,
) -> Path:
    lines = [
        "# Literature RV cross-check (lite)",
        "",
        f"- Master: `{master_path}`",
        f"- Diagnostics glob: `{diagnostics_glob}`",
        f"- Methods: {', '.join(methods)}",
        f"- Pipeline exposure×method rows with valid RV: {n_pipeline_epochs}",
        f"- Overlap stars compared: **{n_stars}**",
        f"- Nearest-BJD pairs: {len(pairs)}",
        "",
        "## Per-star bias / RMS",
        "",
    ]
    if per_star.empty:
        lines.append("_No overlapping Gaia IDs with valid pipeline + literature epochs._")
    else:
        hdr = list(per_star.columns)
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("| " + " | ".join("---" for _ in hdr) + " |")
        for _, r in per_star.iterrows():
            cells = []
            for c in hdr:
                v = r[c]
                if c in ("bias_kms", "rms_kms", "median_abs_delta_kms", "median_delta_days") and pd.notna(v):
                    cells.append(f"{float(v):.3f}")
                else:
                    cells.append("" if pd.isna(v) else str(v))
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `epoch_pairs.csv` — nearest literature epoch per pipeline exposure×method",
            "- `per_star_bias_rms.csv` — mean bias and RMS (pipeline − literature)",
            "- `orbit_qa.csv` — published P_orb / M_star / M2 for overlap stars",
            "",
            "Note: lite path skips LAMOST/RAVE (08-full). Epoch separations may be large when",
            "APF and literature campaigns do not overlap in time; nearest-BJD join still runs.",
            "",
        ]
    )
    path = report_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_compare(
    *,
    master_path: Path,
    diagnostics_glob: str,
    report_dir: Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    max_delta_days: float | None = None,
    copy_key_table: Path | None = None,
) -> dict[str, Any]:
    """Load diagnostics + master, nearest-BJD join, write report artifacts."""
    literature = load_literature_epochs(master_path)
    pipeline = load_pipeline_epochs_from_diagnostics(diagnostics_glob, methods=methods)
    pairs = nearest_literature_join(pipeline, literature, max_delta_days=max_delta_days)
    per_star = per_star_bias_rms(pairs)
    orbit = orbit_qa_table(pairs)

    report_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(report_dir / "epoch_pairs.csv", index=False)
    per_star.to_csv(report_dir / "per_star_bias_rms.csv", index=False)
    orbit.to_csv(report_dir / "orbit_qa.csv", index=False)
    _write_report_md(
        report_dir,
        n_pipeline_epochs=int(len(pipeline)),
        n_stars=int(per_star["gaia_dr3_id"].nunique()) if not per_star.empty else 0,
        methods=methods,
        pairs=pairs,
        per_star=per_star,
        diagnostics_glob=diagnostics_glob,
        master_path=master_path,
    )

    if copy_key_table is not None:
        copy_key_table.parent.mkdir(parents=True, exist_ok=True)
        per_star.to_csv(copy_key_table, index=False)

    return {
        "n_pipeline_epochs": int(len(pipeline)),
        "n_stars": int(per_star["gaia_dr3_id"].nunique()) if not per_star.empty else 0,
        "n_pairs": int(len(pairs)),
        "report_dir": str(report_dir),
        "copy_key_table": str(copy_key_table) if copy_key_table else "",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI argument parser for literature cross-check."""
    ap = argparse.ArgumentParser(
        description="Compare pipeline mask/template RVs to El-Badry literature master (lite)."
    )
    ap.add_argument(
        "--master",
        type=Path,
        default=_REPO_ROOT / "calibration" / "literature_rv_master.csv",
        help="Literature master CSV (default: calibration/literature_rv_master.csv)",
    )
    ap.add_argument(
        "--diagnostics-glob",
        required=True,
        help="Glob of *_diagnostics.csv (prefer existing campaign outputs)",
    )
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=_REPO_ROOT / "validation_output" / "literature_crosscheck_lite",
        help="Output directory for REPORT.md and CSVs",
    )
    ap.add_argument(
        "--methods",
        default="mask_ccf,template_fft",
        help="Comma-separated methods (mask_ccf,template_fft[,strong_lines])",
    )
    ap.add_argument(
        "--max-delta-days",
        type=float,
        default=None,
        help="Optional cap on |ΔMJD| for nearest join (default: no cap)",
    )
    ap.add_argument(
        "--copy-key-table",
        type=Path,
        default=_REPO_ROOT / "calibration" / "literature_crosscheck_lite" / "per_star_bias_rms.csv",
        help="Tracked copy of per_star_bias_rms.csv (validation_output is gitignored)",
    )
    ap.add_argument(
        "--no-copy-key-table",
        action="store_true",
        help="Skip writing the tracked key-table copy",
    )
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for ``python -m validation.compare_literature_rvs``."""
    ap = build_arg_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)
    methods = tuple(m.strip() for m in str(args.methods).split(",") if m.strip())
    copy_path = None if args.no_copy_key_table else args.copy_key_table
    summary = run_compare(
        master_path=args.master,
        diagnostics_glob=args.diagnostics_glob,
        report_dir=args.report_dir,
        methods=methods,
        max_delta_days=args.max_delta_days,
        copy_key_table=copy_path,
    )
    print(
        f"literature_crosscheck_lite: n_stars={summary['n_stars']} "
        f"n_pairs={summary['n_pairs']} report_dir={summary['report_dir']}"
    )
    if summary["copy_key_table"]:
        print(f"key_table={summary['copy_key_table']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
