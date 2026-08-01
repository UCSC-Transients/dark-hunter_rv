#!/usr/bin/env python3
"""
Test recommended / secondary strong-line candidates vs mask RV on APF spectra.

Detection QC rejects non-detections and red telluric/fringe-contaminated fits.
Ca H&K are scored separately (activity caveat). Hβ is the reference.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.strong_line_candidate_sweep \\
    --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \\
    --overlap-csv validation_output/template_fft_baseline/pipeline_cool_vsini12_mhfix/overlap/overlap_enriched_per_exposure.csv \\
    --data-root /Users/rfoley/darkhunter/rvs/data \\
    --out-dir validation_output/strong_line_candidate_sweep \\
    --continuum-mode sinc_blaze
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config, continuum, instruments, io_utils, qc, rv_core
from darkhunter_rv.blaze import BlazeCalibration
from darkhunter_rv.pipeline import _rv_kms_from_hb_joint_line_center
from validation.strong_line_teff_sweep import (
    _continuum_kw,
    _load_stems,
    _order_covering_rest,
    _resolve_spectrum,
    teff_bin_label,
)

logger = logging.getLogger(__name__)

# Redward of this, CCD fringe / water is a known APF concern (Vogt+2014).
RED_FRINGE_WARN_A = 8000.0
MIN_CORE_DEPTH = 0.05
MAX_TELLURIC_FRAC = 0.08
MAX_ABS_RV_KMS = 400.0
MAX_ERR_KMS = 40.0
HELPFUL_ABS_DELTA_KMS = 15.0


@dataclass(frozen=True)
class LineCandidate:
    name: str
    rest_a: float
    tier: str  # reference | recommended | secondary
    broad_lines: bool
    red_risk: bool
    activity: bool


CANDIDATES: tuple[LineCandidate, ...] = (
    LineCandidate("Hbeta", 4861.32, "reference", True, False, False),
    # Recommended
    LineCandidate("MgIb2", 5172.68, "recommended", False, False, False),
    LineCandidate("MgIb3", 5183.60, "recommended", False, False, False),
    LineCandidate("CaI4227", 4226.73, "recommended", False, False, False),
    LineCandidate("CaIIK", 3933.66, "recommended", True, False, True),
    LineCandidate("CaII8498", 8498.02, "recommended", False, True, False),
    LineCandidate("CaII8662", 8662.14, "recommended", False, True, False),
    LineCandidate("CaI6162", 6162.17, "recommended", False, False, False),
    # Secondary
    LineCandidate("MgIb1", 5167.32, "secondary", False, False, False),
    LineCandidate("CaI6122", 6122.22, "secondary", False, False, False),
    LineCandidate("FeI5269", 5269.54, "secondary", False, False, False),
    LineCandidate("FeI5328", 5328.04, "secondary", False, False, False),
    LineCandidate("CaIIH", 3968.47, "secondary", True, False, True),
    LineCandidate("MgI8807", 8806.76, "secondary", False, True, False),
)


def local_core_depth(wave: np.ndarray, flux_norm: np.ndarray, rest: float) -> float:
    """1 - min(flux/local_cont) in ±1.5 Å; nan if insufficient pixels."""
    w = np.asarray(wave, float)
    f = np.asarray(flux_norm, float)
    m = (w >= rest - 25.0) & (w <= rest + 25.0) & np.isfinite(f)
    if int(np.sum(m)) < 30:
        return float("nan")
    ww, ff = w[m], f[m]
    core = (ww >= rest - 4.5) & (ww <= rest + 4.5)
    wing = ~core
    if int(np.sum(wing)) < 15 or int(np.sum(core)) < 5:
        return float("nan")
    cont = float(np.nanmedian(ff[wing]))
    if not np.isfinite(cont) or cont <= 0.05:
        return float("nan")
    return float(1.0 - np.nanmin(ff[core]) / cont)


def telluric_fraction_near(wave: np.ndarray, rest: float, half_a: float = 40.0) -> float:
    w = np.asarray(wave, float)
    m = (w >= rest - half_a) & (w <= rest + half_a) & np.isfinite(w)
    if int(np.sum(m)) < 5:
        return float("nan")
    bad = qc.wavelength_band_mask(w[m], qc.rv_contamination_bands())
    return float(np.mean(bad))


def detection_ok(
    *,
    fit_ok: bool,
    depth: float,
    err_kms: float,
    rv_kms: float,
    telluric_frac: float,
    red_risk: bool,
    rest_a: float,
) -> tuple[bool, str]:
    if not fit_ok:
        return False, "fit_failed"
    if not np.isfinite(depth) or depth < MIN_CORE_DEPTH:
        return False, "shallow_or_undetected"
    if not np.isfinite(err_kms) or err_kms <= 0 or err_kms > MAX_ERR_KMS:
        return False, "bad_err"
    if not np.isfinite(rv_kms) or abs(rv_kms) > MAX_ABS_RV_KMS:
        return False, "bad_rv"
    if np.isfinite(telluric_frac) and telluric_frac > MAX_TELLURIC_FRAC:
        return False, "telluric_contaminated"
    if red_risk and float(rest_a) >= RED_FRINGE_WARN_A:
        # Keep detection if telluric OK, but caller flags fringe risk separately.
        pass
    return True, ""


def fit_candidates(
    spectrum_path: Path,
    *,
    teff: float,
    continuum_mode: str,
    blaze_cal: BlazeCalibration | None,
    instrument_name: str = "APF",
) -> list[dict]:
    instrument = instruments.get_instrument_profile(instrument_name)
    _, spec_data = io_utils.read_spectrum(str(spectrum_path))
    valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
    hot = float(teff) >= float(config.METHOD_REGION_STRONG_LINES_MIN_TEFF_K)
    R_inst = float(getattr(instrument, "resolving_power", 60_000.0))
    out: list[dict] = []
    for cand in CANDIDATES:
        rest = float(cand.rest_a)
        o = _order_covering_rest(spec_data, valid_orders, rest)
        base = {
            "line": cand.name,
            "rest_a": rest,
            "tier": cand.tier,
            "red_risk": bool(cand.red_risk),
            "activity": bool(cand.activity),
            "broad_lines": bool(cand.broad_lines),
        }
        if o is None:
            out.append({**base, "ok": False, "detected": False, "reason": "no_order_coverage"})
            continue
        w = np.array(spec_data[o]["wavelength"], float)
        f = np.array(spec_data[o]["flux"], float)
        e = np.array(spec_data[o]["eflux"], float)
        try:
            kw = _continuum_kw(
                continuum_mode=continuum_mode,
                echelle_order=o,
                hot=hot,
                blaze_cal=blaze_cal,
            )
            nw, nf, _ = continuum.fit_continuum(w, f, e, **kw)
            nw, nf, _ = continuum.despike_normalized_pre_ccf(nw, nf, np.ones_like(nf))
        except Exception as ex:
            out.append(
                {
                    **base,
                    "ok": False,
                    "detected": False,
                    "reason": f"continuum:{ex}",
                    "order": int(o),
                }
            )
            continue
        depth = local_core_depth(nw, nf, rest)
        tfrac = telluric_fraction_near(nw, rest)
        # Metals: always narrow window. Ca H&K / Hβ: broad when hot OR activity lines.
        use_broad = bool(cand.broad_lines) and (hot or cand.activity or cand.name == "Hbeta")
        bundle = rv_core.measure_strong_line_voigt_lorentz(
            nw,
            nf,
            rest=rest,
            broad_lines=use_broad,
            resolving_power=R_inst,
        )
        if bundle is None:
            out.append(
                {
                    **base,
                    "ok": False,
                    "detected": False,
                    "reason": "fit_failed",
                    "order": int(o),
                    "depth": depth,
                    "telluric_frac": tfrac,
                }
            )
            continue
        rv = float(_rv_kms_from_hb_joint_line_center(bundle))
        err = float(bundle.get("err_voigt_kms", np.nan))
        fit_ok = bool(np.isfinite(rv))
        det, reason = detection_ok(
            fit_ok=fit_ok,
            depth=depth,
            err_kms=err,
            rv_kms=rv,
            telluric_frac=tfrac,
            red_risk=cand.red_risk,
            rest_a=rest,
        )
        fringe_flag = bool(cand.red_risk and rest >= RED_FRINGE_WARN_A)
        out.append(
            {
                **base,
                "ok": fit_ok,
                "detected": det,
                "reason": reason if not det else "",
                "order": int(o),
                "depth": depth,
                "telluric_frac": tfrac,
                "rv_kms": rv,
                "err_kms": err,
                "fringe_warn": fringe_flag,
                "use_broad": use_broad,
            }
        )
    return out


def run_sweep(
    *,
    spectrum_list: Path,
    overlap_csv: Path,
    data_root: Path,
    out_dir: Path,
    continuum_mode: str,
    blaze_path: Path | None,
    instrument_name: str = "APF",
    limit: int | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    ov = pd.read_csv(overlap_csv)
    ov["stem"] = ov["basename"].astype(str).str.replace(r"\.txt$", "", regex=True)
    mask_by_stem = {
        str(r["stem"]): (
            float(r["mask_rv_kms"]) if pd.notna(r["mask_rv_kms"]) else float("nan"),
            bool(r["mask_valid"]) if pd.notna(r["mask_valid"]) else False,
            float(r["teff"]) if pd.notna(r["teff"]) else float("nan"),
            float(r["gaia_source_id"]) if pd.notna(r.get("gaia_source_id")) else float("nan"),
        )
        for _, r in ov.iterrows()
    }
    blaze_cal = BlazeCalibration.load(blaze_path) if blaze_path and Path(blaze_path).is_file() else None

    stems = _load_stems(spectrum_list)
    if limit is not None:
        stems = stems[: int(limit)]

    rows: list[dict] = []
    for i, stem_raw in enumerate(stems, start=1):
        path = _resolve_spectrum(stem_raw, data_root)
        stem = Path(stem_raw).name.replace(".txt", "")
        logger.info("[%d/%d] %s", i, len(stems), stem)
        if path is None:
            continue
        mask_rv, mask_ok, teff_ov, gaia = mask_by_stem.get(
            stem, (float("nan"), False, float("nan"), float("nan"))
        )
        teff = float(teff_ov) if np.isfinite(teff_ov) else float(config.DEFAULT_TEFF)
        fits = fit_candidates(
            path,
            teff=teff,
            continuum_mode=continuum_mode,
            blaze_cal=blaze_cal,
            instrument_name=instrument_name,
        )
        hb = next((c for c in fits if c["line"] == "Hbeta" and c.get("detected")), None)
        hb_rv = float(hb["rv_kms"]) if hb is not None else float("nan")
        for c in fits:
            abs_m = (
                abs(float(c["rv_kms"]) - float(mask_rv))
                if c.get("detected") and mask_ok and np.isfinite(mask_rv)
                else float("nan")
            )
            abs_hb = (
                abs(float(c["rv_kms"]) - hb_rv)
                if c.get("detected") and np.isfinite(hb_rv)
                else float("nan")
            )
            helpful = bool(
                c.get("detected")
                and mask_ok
                and np.isfinite(abs_m)
                and abs_m <= HELPFUL_ABS_DELTA_KMS
            )
            rows.append(
                {
                    "stem": stem,
                    "gaia_source_id": gaia,
                    "teff": teff,
                    "teff_bin": teff_bin_label(teff),
                    "mask_rv_kms": mask_rv,
                    "mask_valid": mask_ok,
                    "hbeta_rv_kms": hb_rv,
                    "hbeta_abs_delta_mask_kms": (
                        abs(hb_rv - mask_rv) if np.isfinite(hb_rv) and mask_ok else float("nan")
                    ),
                    **{k: c.get(k) for k in (
                        "line", "rest_a", "tier", "red_risk", "activity", "broad_lines",
                        "ok", "detected", "reason", "order", "depth", "telluric_frac",
                        "rv_kms", "err_kms", "fringe_warn", "use_broad",
                    )},
                    "abs_delta_mask_kms": abs_m,
                    "abs_delta_hbeta_kms": abs_hb,
                    "helpful_vs_mask": helpful,
                    "beats_hbeta_vs_mask": bool(
                        helpful
                        and np.isfinite(abs_m)
                        and np.isfinite(hb_rv)
                        and mask_ok
                        and abs_m + 0.5 < abs(hb_rv - mask_rv)
                    ),
                }
            )

    per = pd.DataFrame(rows)
    per_path = out_dir / "per_line_fits.csv"
    per.to_csv(per_path, index=False)
    summary = _summarize(per)
    sum_path = out_dir / "line_summary.csv"
    summary.to_csv(sum_path, index=False)
    md = _write_report(per, summary, out_dir / "CANDIDATE_SWEEP_SUMMARY.md")
    logger.info("Wrote %s %s %s", per_path, sum_path, md)
    return summary


def _summarize(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line, g in per.groupby("line", sort=False):
        n = len(g)
        det = g["detected"].fillna(False)
        n_det = int(det.sum())
        g_det = g[det]
        g_mask = g_det[g_det["mask_valid"].fillna(False)]
        tier = str(g["tier"].iloc[0])
        red = bool(g["red_risk"].iloc[0])
        act = bool(g["activity"].iloc[0])
        reasons = g.loc[~det, "reason"].fillna("").value_counts().head(3)
        reason_str = "; ".join(f"{k}:{v}" for k, v in reasons.items() if k)
        med_d = float(g_mask["abs_delta_mask_kms"].median()) if len(g_mask) else float("nan")
        med_hb = float(g_mask["hbeta_abs_delta_mask_kms"].median()) if len(g_mask) else float("nan")
        helpful_n = int(g_mask["helpful_vs_mask"].fillna(False).sum()) if len(g_mask) else 0
        beat_n = int(g_mask["beats_hbeta_vs_mask"].fillna(False).sum()) if len(g_mask) else 0
        med_depth = float(g_det["depth"].median()) if len(g_det) else float("nan")
        med_t = float(g_det["telluric_frac"].median()) if len(g_det) else float("nan")
        # Multi-epoch activity proxy: per-star std of line RV vs mask residual
        epoch_std = float("nan")
        if act and "gaia_source_id" in g.columns and len(g_det) >= 4:
            stds = []
            for _, sg in g_det.groupby("gaia_source_id"):
                if len(sg) >= 3 and sg["mask_valid"].all():
                    stds.append(float(np.nanstd(sg["rv_kms"].astype(float))))
            if stds:
                epoch_std = float(np.median(stds))
        # Verdict rules
        det_rate = n_det / max(n, 1)
        helpful_rate = helpful_n / max(len(g_mask), 1) if len(g_mask) else 0.0
        if det_rate < 0.40:
            verdict = "exclude_undetected"
        elif red and (not np.isfinite(med_t) or med_t > 0.02 or med_d > 25.0):
            verdict = "exclude_red_risk"
        elif act and helpful_rate < 0.35:
            verdict = "exclude_activity_unstable"
        elif helpful_rate >= 0.50 and (not np.isfinite(med_d) or med_d <= 12.0):
            verdict = "keep_candidate"
        elif helpful_rate >= 0.25 and med_d <= 20.0:
            verdict = "marginal"
        else:
            verdict = "exclude_unhelpful"
        rows.append(
            {
                "line": line,
                "tier": tier,
                "rest_a": float(g["rest_a"].iloc[0]),
                "red_risk": red,
                "activity": act,
                "n": n,
                "n_detected": n_det,
                "detect_frac": det_rate,
                "n_mask_paired": len(g_mask),
                "helpful_n": helpful_n,
                "helpful_frac": helpful_rate,
                "beats_hbeta_n": beat_n,
                "median_abs_delta_mask_kms": med_d,
                "median_hbeta_abs_delta_mask_kms": med_hb,
                "median_depth": med_depth,
                "median_telluric_frac": med_t,
                "median_epoch_rv_std_kms": epoch_std,
                "top_fail_reasons": reason_str,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def _write_report(per: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    lines = [
        "# Strong-line candidate sweep",
        "",
        f"Exposures: **{per['stem'].nunique()}**; lines tested: **{per['line'].nunique()}**.",
        "",
        f"Detection: core depth ≥ {MIN_CORE_DEPTH}, σ ≤ {MAX_ERR_KMS} km/s, "
        f"|RV| ≤ {MAX_ABS_RV_KMS} km/s, telluric fraction ≤ {MAX_TELLURIC_FRAC}.",
        f"Helpful: detected and |RV−mask| ≤ {HELPFUL_ABS_DELTA_KMS} km/s.",
        "",
        "## Per-line summary",
        "",
    ]
    show = summary[
        [
            "line",
            "tier",
            "detect_frac",
            "helpful_frac",
            "median_abs_delta_mask_kms",
            "median_hbeta_abs_delta_mask_kms",
            "median_telluric_frac",
            "median_epoch_rv_std_kms",
            "verdict",
            "top_fail_reasons",
        ]
    ].copy()
    for c in ("detect_frac", "helpful_frac"):
        show[c] = show[c].map(lambda x: f"{100*x:.0f}%")
    for c in (
        "median_abs_delta_mask_kms",
        "median_hbeta_abs_delta_mask_kms",
        "median_telluric_frac",
        "median_epoch_rv_std_kms",
    ):
        show[c] = show[c].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—")
    lines.append(show.to_markdown(index=False))
    lines.extend(["", "## Verdicts", ""])
    for verd, g in summary.groupby("verdict"):
        names = ", ".join(f"`{x}`" for x in g["line"].tolist())
        lines.append(f"- **{verd}**: {names}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Ca H&K (`activity=True`): multi-epoch RV scatter reported when ≥3 epochs/star.",
            "- Red lines (`red_risk`): Ca II IR / Mg I 8807 — fail closed if telluric window contaminated.",
            "- Do not promote lines with `exclude_*` into `strong_line_rests_for_teff`.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spectrum-list", type=Path, required=True)
    p.add_argument("--overlap-csv", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/Users/rfoley/darkhunter/rvs/data"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--continuum-mode", default="sinc_blaze")
    p.add_argument("--blaze-calibration", type=Path, default=config.BLAZE_CALIBRATION_FILE)
    p.add_argument("--instrument", default="APF")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    run_sweep(
        spectrum_list=args.spectrum_list,
        overlap_csv=args.overlap_csv,
        data_root=args.data_root,
        out_dir=args.out_dir,
        continuum_mode=str(args.continuum_mode),
        blaze_path=args.blaze_calibration,
        instrument_name=str(args.instrument),
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
