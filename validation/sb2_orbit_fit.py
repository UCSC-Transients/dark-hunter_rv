#!/usr/bin/env python3
"""
SB2 orbit RV fits: independent and joint double-lined Keplerian models.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.sb2_orbit_fit \\
    --gaia-id 77413727493690112 \\
    --sb2-dir validation_output/sb2_77413727493690112 \\
    --summary /Users/rfoley/darkhunter/rvs/output/Gaia_DR3_77413727493690112_summary.txt \\
    --out-dir validation_output/sb2_77413727493690112/orbit_fits
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
from darkhunter_rv.sb2_rv_fit import GaussianPrior, OrbitPriors, run_sb2_orbit_fits


def _optional_gaussian_prior(value: float | None, sigma: float | None, name: str) -> GaussianPrior | None:
    if value is None and sigma is None:
        return None
    if value is None or sigma is None:
        raise ValueError(f"{name} prior requires both mean and sigma")
    if float(sigma) <= 0:
        raise ValueError(f"{name} prior sigma must be > 0")
    return GaussianPrior(value=float(value), sigma=float(sigma))


def main() -> int:
    ap = argparse.ArgumentParser(description="SB2 independent and joint Keplerian orbit fits")
    ap.add_argument("--gaia-id", required=True)
    ap.add_argument("--sb2-dir", type=Path, required=True, help="Directory with sb2_epochs.csv and sb2_fit.json")
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--spec-root", type=Path, default=Path("/Users/rfoley/darkhunter/rvs/data"))
    ap.add_argument("--instrument", default="APF", choices=["APF", "GHOST", "MAROON-X"])
    ap.add_argument("--mass-prior-sigma-frac", type=float, default=0.2)
    ap.add_argument("--prior-period-days", type=float, default=None, help="Gaussian prior mean for P (days)")
    ap.add_argument("--prior-period-sigma-days", type=float, default=None, help="Gaussian prior sigma for P (days)")
    ap.add_argument("--prior-ecc", type=float, default=None, help="Gaussian prior mean for eccentricity")
    ap.add_argument("--prior-ecc-sigma", type=float, default=None, help="Gaussian prior sigma for eccentricity")
    ap.add_argument("--prior-gamma-kms", type=float, default=None, help="Gaussian prior mean for primary gamma (km/s)")
    ap.add_argument("--prior-gamma-sigma-kms", type=float, default=None, help="Gaussian prior sigma for primary gamma (km/s)")
    ap.add_argument("--prior-k-kms", type=float, default=None, help="Gaussian prior mean for primary K (km/s)")
    ap.add_argument("--prior-k-sigma-kms", type=float, default=None, help="Gaussian prior sigma for primary K (km/s)")
    ap.add_argument("--prior-omega-deg", type=float, default=None, help="Gaussian prior mean for omega (deg)")
    ap.add_argument("--prior-omega-sigma-deg", type=float, default=None, help="Gaussian prior sigma for omega (deg)")
    ap.add_argument("--rerun-ccf", action="store_true", help="Regenerate sb2_epochs.csv before fitting")
    ap.add_argument("--delta-chi2-min", type=float, default=None)
    ap.add_argument("--vsini1-init-kms", type=float, default=None, help="Primary template vsini init (km/s)")
    ap.add_argument("--vsini1-max-kms", type=float, default=None, help="Primary template vsini upper bound (km/s)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    gid = str(args.gaia_id).strip()
    sb2_dir = Path(args.sb2_dir)
    summary_path = args.summary or (config.OUTPUT_DIR / f"Gaia_DR3_{gid}_summary.txt")
    epoch_analyses = None
    orbit_priors = OrbitPriors(
        period_days=_optional_gaussian_prior(args.prior_period_days, args.prior_period_sigma_days, "period"),
        eccentricity=_optional_gaussian_prior(args.prior_ecc, args.prior_ecc_sigma, "eccentricity"),
        gamma1_kms=_optional_gaussian_prior(args.prior_gamma_kms, args.prior_gamma_sigma_kms, "gamma"),
        k1_kms=_optional_gaussian_prior(args.prior_k_kms, args.prior_k_sigma_kms, "K"),
        omega_deg=_optional_gaussian_prior(args.prior_omega_deg, args.prior_omega_sigma_deg, "omega"),
    )

    if args.rerun_ccf:
        instrument = get_instrument_profile(args.instrument)
        report = run_sb2_star(
            gid,
            Path(args.spec_root),
            Path(summary_path),
            instrument,
            sb2_dir,
            delta_chi2_min=args.delta_chi2_min,
            write_plots=False,
            refine_fit=False,
            vsini1_init_kms=args.vsini1_init_kms,
            vsini1_max_kms=args.vsini1_max_kms,
            return_epoch_analyses=True,
        )
        epoch_analyses = report.get("epoch_analyses")

    run_sb2_orbit_fits(
        sb2_dir,
        Path(summary_path),
        Path(args.out_dir),
        gaia_id=gid,
        mass_sigma_frac=float(args.mass_prior_sigma_frac),
        epoch_analyses=epoch_analyses,
        orbit_priors=orbit_priors,
    )
    logging.info("SB2 orbit fits written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
