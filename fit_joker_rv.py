#!/usr/bin/env python3
"""Fit APF RV summaries with The Joker (default orbit path).

Use ``--rvchi2`` to run the least-squares Keplerian fitter instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from astropy.time import Time

from darkhunter_rv.joker_rv_fit import (
    JOKER_VARIANT_ORDER,
    DEFAULT_PRIOR_SIZE,
    envelope_report,
    masses_from_report_variants,
    random_param_rows,
    run_joker_variant,
    should_skip_refit,
    skipped_variant_block,
    summarize_sample_arrays,
)
from darkhunter_rv.rv_keplerian_plots import (
    our_telescope_points,
    plot_joker_corners,
    plot_joker_fit_residuals,
    plot_joker_multi_fit,
    plot_rv_data_only,
)
from fit_apf_rv_keplerian import (
    _sanitize_rv_arrays,
    count_pipeline_rows,
    fetch_gaia_nss_orbit,
    load_gaia_nss_from_cache,
    load_mass_priors_from_summary,
    load_nss_priors_from_summary,
    load_table_m1_map_from_csv,
    parse_m1_from_summary,
    parse_object_id_from_summary,
    parse_summary,
    prefetch_gaia_nss_bulk,
    report_stem,
    resolve_inclination_deg_for_rv_mass,
    resolve_m1_msun_for_rv_mass,
    resolve_observability_window,
    resolve_summary_files,
)


def run_one_joker(
    summary_path: Path,
    reports_dir: Path,
    *,
    min_points: int,
    max_points: Optional[int],
    variants: List[str],
    use_gaia_nss: bool,
    gaia_cache_path: Optional[Path],
    observability_cache_path: Optional[Path],
    query_gaia_online: bool,
    table_m1_msun: Optional[float],
    prior_size: int,
    force: bool,
    mcmc_tune: int,
    mcmc_draws: int,
    mcmc_chains: int,
    plots_root: Optional[Path] = None,
    seed: int = 0,
) -> Optional[dict]:
    points = parse_summary(summary_path)
    n_epochs = len(points)
    if n_epochs < min_points:
        print(
            f"[SKIP] {summary_path}: n_epochs={n_epochs} < min_points={min_points} "
            f"(file rows in [PIPELINE RESULTS]={count_pipeline_rows(summary_path)})"
        )
        return None
    if max_points is not None and n_epochs > max_points:
        print(f"[SKIP] {summary_path}: n_epochs={n_epochs} > max_points={max_points}")
        return None

    t_all = np.array([p.mjd for p in points], dtype=float)
    y_all = np.array([p.rv for p in points], dtype=float)
    yerr_all = np.array([p.rv_err for p in points], dtype=float)
    rms_arr = np.array([p.rms for p in points], dtype=float)
    n_before = len(points)
    ok = np.isfinite(t_all) & np.isfinite(y_all)
    points_fit = [p for p, keep in zip(points, ok) if keep]
    t, y, yerr = _sanitize_rv_arrays(t_all, y_all, yerr_all, rms_fallback=rms_arr)
    if len(t) < min_points:
        print(
            f"[SKIP] {summary_path.name}: n_finite_rv={len(t)} < min_points={min_points} "
            f"(parsed {n_before} rows)"
        )
        return None
    if max_points is not None and len(t) > max_points:
        print(f"[SKIP] {summary_path.name}: n_finite_rv={len(t)} > max_points={max_points}")
        return None

    gaia_source_id = parse_object_id_from_summary(summary_path)
    stem = report_stem(summary_path, gaia_source_id)
    out_json = reports_dir / f"{stem}_joker_fit.json"
    if should_skip_refit(out_json, summary_path, len(t), force=force):
        print(f"[SKIP] {summary_path.name}: joker JSON up to date (n_rv={len(t)})")
        try:
            return json.loads(out_json.read_text())
        except Exception:
            return None

    gaia_nss = None
    if use_gaia_nss:
        gaia_nss = load_nss_priors_from_summary(summary_path)
        if gaia_nss is None and gaia_source_id is not None:
            gaia_nss = load_gaia_nss_from_cache(gaia_source_id, gaia_cache_path)
            if gaia_nss is None and query_gaia_online:
                gaia_nss = fetch_gaia_nss_orbit(gaia_source_id, cache_path=gaia_cache_path)

    t_ref = float(np.median(t))
    now_mjd = float(Time.now().mjd)
    table_m1 = table_m1_msun
    dummy = {
        "gaia_source_id": gaia_source_id,
        "gaia_nss": gaia_nss,
    }
    m1 = resolve_m1_msun_for_rv_mass(dummy, summary_path=summary_path, table_m1_msun=table_m1)
    if m1 is None:
        m1 = parse_m1_from_summary(summary_path)
    incl = resolve_inclination_deg_for_rv_mass(dummy, summary_path=summary_path)
    used_m2 = None
    if gaia_nss:
        used_m2 = gaia_nss.get("m2_msun")
    if used_m2 is None:
        used_m2 = load_mass_priors_from_summary(summary_path).get("m2_msun")

    rng = np.random.default_rng(seed)
    variant_blocks: Dict[str, dict] = {}
    sample_params: Dict[str, List[np.ndarray]] = {}
    median_params: Dict[str, np.ndarray] = {}
    corner_arrays: Dict[str, Dict[str, np.ndarray]] = {}

    for variant in variants:
        chain_path = reports_dir / f"{stem}_joker_{variant}.hdf5"
        arr, spec, sampler = run_joker_variant(
            variant,
            t,
            y,
            yerr,
            gaia_nss,
            t_ref_mjd=t_ref,
            prior_size=prior_size,
            chain_path=chain_path,
            old_chain_path=chain_path if chain_path.is_file() else None,
            rng=rng,
            mcmc_tune=mcmc_tune,
            mcmc_draws=mcmc_draws,
            mcmc_chains=mcmc_chains,
        )
        if arr is None:
            reason = str(spec.get("skip_reason") or "skipped")
            variant_blocks[variant] = skipped_variant_block(variant, reason, len(t))
            continue
        block = summarize_sample_arrays(
            p_days=arr["P_days"],
            k_kms=arr["K_kms"],
            e=arr["e"],
            omega_rad=arr["omega_rad"],
            m0_rad=arr["M0_rad"],
            gamma_kms=arr["gamma_kms"],
            t=t,
            y=y,
            yerr=yerr,
            t_ref_mjd=t_ref,
            sampler=sampler,
            variant=variant,
            m1_msun=m1,
            inclination_deg=incl,
        )
        variant_blocks[variant] = block
        median_params[variant] = np.asarray(block["params_raw"], dtype=float)
        sample_params[variant] = random_param_rows(
            arr["P_days"],
            arr["K_kms"],
            arr["e"],
            arr["omega_rad"],
            arr["M0_rad"],
            arr["gamma_kms"],
            n_draw=10,
            rng=rng,
        )
        corner_arrays[variant] = {
            "P_days": arr["P_days"],
            "K_kms": arr["K_kms"],
            "e": arr["e"],
            "omega_deg": np.degrees(arr["omega_rad"]),
            "gamma_kms": arr["gamma_kms"],
        }

    masses = masses_from_report_variants(
        variant_blocks,
        m1_msun=m1,
        inclination_deg=incl,
        used_m2_msun=None if used_m2 is None else float(used_m2),
    )
    obs = resolve_observability_window(summary_path, gaia_source_id, observability_cache_path)
    report = envelope_report(
        gaia_source_id=gaia_source_id,
        summary_file=str(summary_path),
        n_points=len(t),
        t_ref_mjd=t_ref,
        now_mjd=now_mjd,
        gaia_nss=gaia_nss,
        variants=variant_blocks,
        masses=masses,
        observability_window=obs,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_png = reports_dir / f"{stem}_keplerian_fit.png"
    resid_png = reports_dir / f"{stem}_keplerian_residuals.png"
    data_png = reports_dir / f"{stem}_rv_data.png"
    corner_png = reports_dir / f"{stem}_joker_corner.png"
    if median_params:
        plot_joker_multi_fit(
            summary_path, points_fit, sample_params, median_params, report, out_png
        )
        plot_joker_fit_residuals(
            summary_path, points_fit, sample_params, median_params, report, resid_png
        )
    if corner_arrays:
        plot_joker_corners(corner_arrays, corner_png, gaia_id=gaia_source_id)
    ours = our_telescope_points(points_fit)
    if len(ours) >= 1:
        plot_rv_data_only(summary_path, ours, report, data_png)

    if plots_root is not None and gaia_source_id:
        star_dir = plots_root / f"Gaia_DR3_{gaia_source_id}"
        star_dir.mkdir(parents=True, exist_ok=True)
        if data_png.is_file():
            shutil.copy2(data_png, star_dir / f"Gaia_DR3_{gaia_source_id}_rv_plot.png")
        if out_png.is_file():
            shutil.copy2(out_png, star_dir / f"Gaia_DR3_{gaia_source_id}_keplerian_fit.png")
        if resid_png.is_file():
            shutil.copy2(resid_png, star_dir / f"Gaia_DR3_{gaia_source_id}_keplerian_residuals.png")
        if corner_png.is_file():
            shutil.copy2(corner_png, star_dir / f"Gaia_DR3_{gaia_source_id}_joker_corner.png")

    out_json.write_text(json.dumps(report, indent=2))
    return report


def main(argv: Optional[List[str]] = None) -> int:
    try:
        from erfa import ErfaWarning  # type: ignore

        warnings.filterwarnings("ignore", category=ErfaWarning)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Fit RV summaries with The Joker (or --rvchi2).")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=Path("rv_fit_reports"))
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--use-gaia-nss", action="store_true")
    parser.add_argument("--query-gaia-online", action="store_true")
    parser.add_argument("--gaia-cache", type=Path, default=None)
    parser.add_argument("--data-csv", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prior-size", type=int, default=DEFAULT_PRIOR_SIZE)
    parser.add_argument("--mcmc-tune", type=int, default=500)
    parser.add_argument("--mcmc-draws", type=int, default=500)
    parser.add_argument("--mcmc-chains", type=int, default=2)
    parser.add_argument(
        "--variants",
        default="rv_only,period,ecc,full",
        help="Comma-separated variant ids.",
    )
    parser.add_argument(
        "--rvchi2",
        action="store_true",
        help="Run chi2 least-squares Keplerian fitter instead of The Joker.",
    )
    parser.add_argument("--plots-root", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.rvchi2:
        from fit_apf_rv_keplerian import main as chi2_main

        forwarded: List[str] = [
            "--output-dir",
            str(args.output_dir),
            "--reports-dir",
            str(args.reports_dir),
            "--min-points",
            str(args.min_points),
        ]
        if args.summary is not None:
            forwarded.extend(["--summary", str(args.summary)])
        if args.all:
            forwarded.append("--all")
        if args.use_gaia_nss:
            forwarded.append("--use-gaia-nss")
        if args.query_gaia_online:
            forwarded.append("--query-gaia-online")
        if args.force:
            forwarded.append("--force")
        if args.data_csv is not None:
            forwarded.extend(["--data-csv", str(args.data_csv)])
        sys.argv = ["fit_apf_rv_keplerian.py", *forwarded]
        chi2_main()
        return 0

    out_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    gaia_cache = args.gaia_cache if args.gaia_cache else reports_dir / "gaia_nss_cache.json"
    obs_cache = reports_dir / "observability_windows_cache.json"
    m1_map: Dict[str, float] = {}
    if args.data_csv is not None:
        m1_map = load_table_m1_map_from_csv(args.data_csv)

    files = resolve_summary_files(out_dir, args.summary, args.all)
    if not files:
        print(f"No summary files under {out_dir}", file=sys.stderr)
        return 2

    if args.use_gaia_nss and args.query_gaia_online:
        ids = [parse_object_id_from_summary(p) for p in files]
        prefetch_gaia_nss_bulk([i for i in ids if i], gaia_cache)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in JOKER_VARIANT_ORDER:
            print(f"Unknown variant {v}", file=sys.stderr)
            return 2

    combined: List[dict] = []
    plots_root = args.plots_root if args.plots_root is not None else out_dir
    for summ in files:
        gid = parse_object_id_from_summary(summ)
        table_m1 = m1_map.get(gid) if gid else None
        rep = run_one_joker(
            summ,
            reports_dir,
            min_points=args.min_points,
            max_points=args.max_points,
            variants=variants,
            use_gaia_nss=args.use_gaia_nss,
            gaia_cache_path=gaia_cache,
            observability_cache_path=obs_cache,
            query_gaia_online=args.query_gaia_online,
            table_m1_msun=table_m1,
            prior_size=args.prior_size,
            force=args.force,
            mcmc_tune=args.mcmc_tune,
            mcmc_draws=args.mcmc_draws,
            mcmc_chains=args.mcmc_chains,
            plots_root=plots_root,
        )
        if rep is not None:
            combined.append(rep)
            print(f"[OK] {summ.name} n={rep.get('n_points')}")

    (reports_dir / "apf_joker_fit_summary.json").write_text(json.dumps(combined, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
