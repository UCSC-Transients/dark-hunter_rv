"""Gaia NSS SB2 cohort fraction vs pipeline ``sb2_candidate`` flags.

Joins exposure-level ``sb2_candidate`` from ``*_diagnostics.csv`` to a local
NSS source-id list (CSV). Live ADQL dump is optional / out of scope — pass a
cached dump of ``gaiadr3.nss_two_body_orbit`` source_ids (or a test stub).

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.sb2_nss_cohort_report \\
    --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \\
    --nss-ids-csv calibration/nss_sb2_source_ids_stub.csv \\
    --out-dir validation_output/sb2_nss_cohort
"""

from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

_GAIA_RE = re.compile(r"Gaia_DR3_(\d{15,25})")


def parse_gaia_id_from_path(path: Path | str) -> str | None:
    m = _GAIA_RE.search(Path(path).name)
    return m.group(1) if m else None


def _truthy(val: object) -> bool:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return False
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"", "nan", "none", "false", "0", "0.0"}:
        return False
    if s in {"true", "1", "1.0", "yes"}:
        return True
    try:
        return bool(int(float(s)))
    except (TypeError, ValueError):
        return False


def load_exposure_sb2_flags(diagnostics_glob: str) -> pd.DataFrame:
    """
    One row per Gaia ID / stem from diagnostics files.

    Uses the first row that carries ``sb2_candidate`` (exposure-level; identical
    across chunk rows when pipeline fuse wrote it).
    """
    rows: list[dict] = []
    for path_s in sorted(glob(diagnostics_glob)):
        path = Path(path_s)
        gaia = parse_gaia_id_from_path(path)
        if gaia is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "sb2_candidate" not in df.columns:
            rows.append(
                {
                    "gaia_dr3_id": gaia,
                    "stem": path.stem.replace("_diagnostics", ""),
                    "sb2_candidate": False,
                    "sb2_col_present": False,
                    "diagnostics_path": str(path),
                }
            )
            continue
        # Prefer chunk_key == all if present; else first row
        sub = df
        if "chunk_key" in df.columns:
            all_rows = df[df["chunk_key"].astype(str) == "all"]
            if not all_rows.empty:
                sub = all_rows
        flag = _truthy(sub.iloc[0]["sb2_candidate"])
        rows.append(
            {
                "gaia_dr3_id": gaia,
                "stem": path.stem.replace("_diagnostics", ""),
                "sb2_candidate": flag,
                "sb2_col_present": True,
                "diagnostics_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_nss_source_ids(path: Path) -> set[str]:
    """
    Load NSS SB2 / two-body source ids.

    Accepts a one-column CSV (``source_id`` / ``gaia_dr3_id``) or whitespace
    list with ``#`` comments.
    """
    text = path.read_text()
    ids: set[str] = set()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, comment="#")
        col = None
        for c in ("source_id", "gaia_dr3_id", "gaia_id", "sourceId"):
            if c in df.columns:
                col = c
                break
        if col is None and len(df.columns) == 1:
            col = df.columns[0]
        if col is None:
            raise ValueError(f"No source_id column in {path}")
        for v in df[col].astype(str):
            s = v.strip()
            if s and s.lower() != "nan":
                ids.add(s)
        return ids
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.add(s.split()[0])
    return ids


def build_fraction_table(
    exposure_flags: pd.DataFrame,
    nss_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (per_star table, summary one-row fraction table).
    """
    if exposure_flags.empty:
        per = pd.DataFrame(
            columns=[
                "gaia_dr3_id",
                "n_exposures",
                "n_flagged",
                "any_flagged",
                "in_nss",
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "n_stars": 0,
                    "n_exposures": 0,
                    "n_flagged_exposures": 0,
                    "n_nss_sb2": 0,
                    "n_flagged_among_nss": 0,
                    "n_flagged_among_non_nss": 0,
                    "frac_flagged_among_nss": float("nan"),
                    "frac_flagged_among_non_nss": float("nan"),
                }
            ]
        )
        return per, summary

    g = exposure_flags.copy()
    g["in_nss"] = g["gaia_dr3_id"].astype(str).isin(nss_ids)
    per = (
        g.groupby("gaia_dr3_id", as_index=False)
        .agg(
            n_exposures=("sb2_candidate", "size"),
            n_flagged=("sb2_candidate", "sum"),
            in_nss=("in_nss", "max"),
        )
        .assign(any_flagged=lambda d: d["n_flagged"] > 0)
    )
    n_stars = int(len(per))
    n_exp = int(len(g))
    n_flagged_exp = int(g["sb2_candidate"].sum())
    nss = per[per["in_nss"]]
    non = per[~per["in_nss"]]
    n_nss = int(len(nss))
    n_flag_nss = int(nss["any_flagged"].sum()) if n_nss else 0
    n_flag_non = int(non["any_flagged"].sum()) if len(non) else 0
    summary = pd.DataFrame(
        [
            {
                "n_stars": n_stars,
                "n_exposures": n_exp,
                "n_flagged_exposures": n_flagged_exp,
                "n_nss_sb2": n_nss,
                "n_flagged_among_nss": n_flag_nss,
                "n_flagged_among_non_nss": n_flag_non,
                "frac_flagged_among_nss": (n_flag_nss / n_nss) if n_nss else float("nan"),
                "frac_flagged_among_non_nss": (n_flag_non / len(non)) if len(non) else float("nan"),
            }
        ]
    )
    return per, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--diagnostics-glob",
        required=True,
        help="Glob of pipeline *_diagnostics.csv (with sb2_candidate when fused)",
    )
    ap.add_argument(
        "--nss-ids-csv",
        type=Path,
        required=True,
        help="CSV or list of Gaia source_ids in NSS two-body / SB2 sample",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation_output/sb2_nss_cohort"),
    )
    args = ap.parse_args(argv)

    flags = load_exposure_sb2_flags(args.diagnostics_glob)
    nss_ids = load_nss_source_ids(args.nss_ids_csv)
    per, summary = build_fraction_table(flags, nss_ids)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    flags.to_csv(out / "exposure_sb2_flags.csv", index=False)
    per.to_csv(out / "per_star.csv", index=False)
    summary.to_csv(out / "fraction_table.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out / 'fraction_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
