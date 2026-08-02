#!/usr/bin/env python3
"""
SB2 search and two-template spectral separation for a Gaia DR3 star.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.sb2_search \\
    --gaia-id 77413727493690112 \\
    --spec-root /Users/rfoley/darkhunter/rvs/data \\
    --out-dir validation_output/sb2_77413727493690112

Loads Gaia metadata and pipeline RV seeds from ``output/Gaia_DR3_<id>_summary.txt``
disk-first; queries Gaia only when the summary lacks complete metadata (use ``--force-gaia``
to refresh).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config
from darkhunter_rv.instruments import get_instrument_profile
from darkhunter_rv.sb2 import run_sb2_star


def main() -> int:
    ap = argparse.ArgumentParser(description="SB2 mask-CCF search and template separation")
    ap.add_argument("--gaia-id", required=True, help="Gaia DR3 source_id")
    ap.add_argument(
        "--spec-root",
        type=Path,
        default=Path("/Users/rfoley/darkhunter/rvs/data"),
        help="Root directory with Gaia_DR3_*_epoch_*.txt spectra",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Star summary (default: output/Gaia_DR3_<id>_summary.txt)",
    )
    ap.add_argument("--instrument", default="APF", choices=["APF", "GHOST", "MAROON-X"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--force-gaia", action="store_true", help="Re-query Gaia even if summary parses")
    ap.add_argument("--force-fit", action="store_true", help="Run template fit even if SB2 gate fails")
    ap.add_argument("--no-bias", action="store_true", help="Skip per-chunk b0 debias on CCF velocities")
    ap.add_argument("--teff1-window", type=float, default=1200.0, help="Teff1 search half-width (K)")
    ap.add_argument(
        "--delta-chi2-min",
        type=float,
        default=None,
        help="SB2 gate: minimum Δχ² on median CCF (default 2.5)",
    )
    ap.add_argument("--refine-fit", action="store_true", help="Run continuous least-squares refine after coarse grid")
    ap.add_argument("--vsini1-init-kms", type=float, default=None, help="Primary template vsini init (km/s)")
    ap.add_argument("--vsini1-max-kms", type=float, default=None, help="Primary template vsini upper bound (km/s)")
    ap.add_argument(
        "--continuum-mode",
        choices=["sinc_blaze", "sinc_blaze_only", "spline"],
        default="sinc_blaze",
        help="Continuum normalization for SB2 template fit (default sinc_blaze)",
    )
    ap.add_argument(
        "--order-edge-trim-frac",
        type=float,
        default=0.0,
        help="Exclude first/last fraction of pixels per order from chi2 (e.g. 0.08)",
    )
    ap.add_argument(
        "--rv-prior-sigma-kms",
        type=float,
        default=1.0,
        help="Gaussian prior width on CCF velocity seeds (km/s)",
    )
    ap.add_argument("--plots", action="store_true", help="Write median-CCF PNGs")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    gid = str(args.gaia_id).strip()
    summary_path = args.summary or (config.OUTPUT_DIR / f"Gaia_DR3_{gid}_summary.txt")
    instrument = get_instrument_profile(args.instrument)

    report = run_sb2_star(
        gid,
        Path(args.spec_root),
        Path(summary_path),
        instrument,
        Path(args.out_dir),
        force_gaia=bool(args.force_gaia),
        force_fit=bool(args.force_fit),
        apply_bias=not bool(args.no_bias),
        teff1_window=float(args.teff1_window),
        delta_chi2_min=args.delta_chi2_min,
        write_plots=bool(args.plots),
        refine_fit=bool(args.refine_fit),
        vsini1_init_kms=args.vsini1_init_kms,
        vsini1_max_kms=args.vsini1_max_kms,
        continuum_mode=str(args.continuum_mode),
        order_edge_trim_frac=float(args.order_edge_trim_frac),
        rv_prior_sigma_kms=float(args.rv_prior_sigma_kms),
    )
    n_sb2 = sum(1 for e in report.get("epochs", []) if e.get("sb2_candidate"))
    logging.info(
        "SB2 search done: %s epochs, %s sb2_candidate, median Δχ²=%.2f → %s",
        report.get("n_epochs"),
        n_sb2,
        float(report.get("median_delta_chi2", float("nan"))),
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
