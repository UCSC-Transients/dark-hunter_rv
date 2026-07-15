#!/usr/bin/env python3
"""Ingest manual literature RVs into the canonical CSV and/or apply them to summaries."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from darkhunter_rv.gaia_utils import (
    parse_external_rvs_from_star_summary,
    replace_external_rv_section_in_summary,
)
from darkhunter_rv.manual_literature_rvs import (
    default_manual_literature_path,
    load_manual_literature_rows,
    parse_manual_epoch_table,
    upsert_manual_literature_csv,
)
from darkhunter_rv.summary_paths import discover_summary_files, parse_object_id_from_summary


def _log(msg: str) -> None:
    print(msg, flush=True)


def _discover_summaries(out_dir: Path) -> list[Path]:
    flat = sorted(out_dir.glob("Gaia_DR3_*_summary.txt"))
    if flat:
        return flat
    return discover_summary_files(out_dir)


def _cmd_ingest(args: argparse.Namespace) -> int:
    ingest_path = Path(args.ingest)
    if not ingest_path.is_file():
        _log(f"Ingest file not found: {ingest_path}")
        return 2
    text = ingest_path.read_text(encoding="utf-8", errors="replace")
    rows = parse_manual_epoch_table(
        text,
        gaia_id=args.gaia_id,
        default_rv_err=float(args.default_rv_err),
    )
    if not rows:
        _log("No epoch rows parsed from ingest file.")
        return 2
    csv_path = Path(args.csv) if args.csv else default_manual_literature_path()
    out = upsert_manual_literature_csv(rows, path=csv_path)
    teles = sorted({r["telescope"] for r in rows})
    _log(f"Upserted {len(rows)} row(s) for Gaia {args.gaia_id} ({', '.join(teles)}) → {out}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        os.environ.get("DARKHUNTER_OUTPUT_DIR", _REPO / "output")
    )
    csv_path = Path(args.csv) if args.csv else default_manual_literature_path()
    by_id = load_manual_literature_rows(csv_path)
    if not by_id:
        _log(f"No manual literature rows in {csv_path}")
        return 2

    files = _discover_summaries(out_dir)
    if args.star_id:
        sid = str(int(args.star_id))
        files = [p for p in files if parse_object_id_from_summary(p) == sid]
        if sid not in by_id:
            _log(f"No CSV rows for Gaia {sid}")
            return 2
    if not files:
        _log(f"No summary files under {out_dir}")
        return 2

    updated = 0
    for summ in files:
        sid = parse_object_id_from_summary(summ)
        if not sid or sid not in by_id:
            continue
        existing = parse_external_rvs_from_star_summary(summ)
        # replace_external_rv_section_in_summary re-merges LITERATURE_* from CSV.
        replace_external_rv_section_in_summary(summ, existing)
        n_lit = sum(1 for r in by_id[sid] if str(r.get("telescope", "")).startswith("LITERATURE_"))
        _log(f"Gaia_DR3_{sid}: applied {n_lit} literature row(s) → {summ}")
        updated += 1

    if not updated:
        _log("No matching summaries to update.")
        return 2
    _log(f"Done: updated {updated} summary file(s).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Ingest manual literature RVs into calibration/manual_literature_rvs.csv "
            "and/or merge them into star summary [EXTERNAL RV DATA] blocks."
        )
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Canonical CSV path (default: calibration/manual_literature_rvs.csv or env)",
    )
    ap.add_argument(
        "--ingest",
        metavar="FILE",
        default=None,
        help="Whitespace epoch table (lit_N TELESCOPE MJD RV ERR RMS)",
    )
    ap.add_argument(
        "--gaia-id",
        default=None,
        help="Gaia DR3 source id (required with --ingest)",
    )
    ap.add_argument(
        "--default-rv-err",
        type=float,
        default=0.6,
        help="Fill non-finite RV errors on ingest (default: 0.6 km/s)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Patch summaries from the canonical CSV",
    )
    ap.add_argument("--star-id", default=None, help="Limit --apply to one Gaia id")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Pipeline output root for --apply (default: output/ or DARKHUNTER_OUTPUT_DIR)",
    )
    args = ap.parse_args()

    if not args.ingest and not args.apply:
        ap.error("Specify --ingest and/or --apply")

    rc = 0
    if args.ingest:
        if not args.gaia_id:
            ap.error("--ingest requires --gaia-id")
        rc = _cmd_ingest(args) or rc
    if args.apply:
        rc = _cmd_apply(args) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
