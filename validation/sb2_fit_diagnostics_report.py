#!/usr/bin/env python3
"""
SB2 template-fit diagnostics for one ``--sb2-dir``.

Reads ``sb2_report.json``, ``sb2_epochs.csv``, and ``sb2_fit.json`` (when present),
compares continuum modes, RV consistency, residual structure, and SB2 vs single-star χ².

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.sb2_fit_diagnostics_report \\
    --sb2-dir validation_output/sb2_77413727493690112 \\
    --spec-root /Users/rfoley/darkhunter/rvs/data
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config, io_utils
from darkhunter_rv.instruments import get_instrument_profile
from darkhunter_rv.sb2 import (
    Sb2FitParams,
    Sb2FitSettings,
    _fit_pixel_mask,
    _load_phoenix_component,
    _normalize_blend_order,
    _order_edge_mask,
    _order_obs_norm,
    build_sb2_fit_settings,
    discover_epoch_paths,
    load_star_context,
)

logger = logging.getLogger(__name__)

_CORE_DEPTH_MAX = 0.985
_EDGE_FRAC = 0.08


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _region_rms(residual: np.ndarray, mask: np.ndarray) -> float | None:
    m = mask & np.isfinite(residual)
    if int(np.sum(m)) < 5:
        return None
    return float(np.sqrt(np.mean(residual[m] ** 2)))


def _norm_mode_comparison(
    spec_data: dict,
    order: int,
    hot: bool,
    blaze_path: Path | None,
) -> list[dict[str, Any]]:
    """Edge vs center RMS after continuum normalize (no model)."""
    d = spec_data[order]
    wave = np.asarray(d["wavelength"], float)
    flux = np.asarray(d["flux"], float)
    err = np.asarray(d["eflux"], float)
    n = len(wave)
    if n < 30:
        return []
    center = _order_edge_mask(n, _EDGE_FRAC)
    edge = ~center
    rows: list[dict[str, Any]] = []
    for mode in ("spline", "sinc_blaze", "sinc_blaze_only"):
        settings = build_sb2_fit_settings(continuum_mode=mode, blaze_calibration_path=blaze_path)
        pack = _order_obs_norm(spec_data, order, hot, settings=settings)
        if pack is None:
            continue
        _, nf = pack
        med = float(np.nanmedian(nf[center])) if int(np.sum(center)) else float("nan")
        with np.errstate(invalid="ignore"):
            edge_rms = _region_rms(nf - med, edge)
            center_rms = _region_rms(nf - med, center)
        rows.append(
            {
                "continuum_mode": settings.continuum_mode,
                "order": int(order),
                "edge_rms": edge_rms,
                "center_rms": center_rms,
                "edge_over_center": (edge_rms / center_rms) if edge_rms and center_rms else None,
            }
        )
    return rows


def _epoch_residual_breakdown(
    params: Sb2FitParams,
    spec_data: dict,
    instrument,
    hot: bool,
    settings: Sb2FitSettings,
    vel1: float,
    vel2: float,
) -> dict[str, Any]:
    rp = float(instrument.resolving_power)
    res_all: list[float] = []
    core_mask_all: list[bool] = []
    edge_mask_all: list[bool] = []
    valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
    for order in valid_orders[::3]:
        obs = _order_obs_norm(spec_data, order, hot, settings=settings)
        if obs is None:
            continue
        wave, obs_nf = obs
        c1 = _load_phoenix_component(params.teff1, params.logg1, params.feh, params.vsini1, vel1, wave, rp)
        c2 = _load_phoenix_component(params.teff2, params.logg2, params.feh, params.vsini2, vel2, wave, rp)
        if c1 is None or c2 is None:
            continue
        _, f1, _ = c1
        _, f2, _ = c2
        norm_pack = _normalize_blend_order(wave, f1 + f2, hot, order, settings=settings)
        if norm_pack is None:
            continue
        nw, model_nf = norm_pack
        n = min(len(wave), len(obs_nf), len(nw), len(model_nf))
        obs_nf = np.asarray(obs_nf[:n], float)
        model_nf = np.asarray(model_nf[:n], float)
        fit_mask = _fit_pixel_mask(nw[:n], obs_nf, model_nf, settings)
        if int(np.sum(fit_mask)) < 20:
            continue
        resid = obs_nf - model_nf
        core = (obs_nf < _CORE_DEPTH_MAX) & fit_mask
        edge = (~_order_edge_mask(n, _EDGE_FRAC)) & fit_mask
        center = _order_edge_mask(n, _EDGE_FRAC) & fit_mask
        res_all.extend(resid[fit_mask].tolist())
        core_mask_all.extend(core[fit_mask].tolist())
        edge_mask_all.extend(edge[fit_mask].tolist())

    if not res_all:
        return {}
    res = np.asarray(res_all, float)
    core_m = np.asarray(core_mask_all, bool)
    edge_m = np.asarray(edge_mask_all, bool)
    center_m = ~edge_m
    return {
        "rms_all": _region_rms(res, np.ones(len(res), bool)),
        "rms_core": _region_rms(res, core_m),
        "rms_continuum": _region_rms(res, ~core_m),
        "rms_edge": _region_rms(res, edge_m),
        "rms_center_trimmed": _region_rms(res, center_m),
        "n_pixels": int(len(res)),
    }


def _template_chi2_sb2_vs_single(
    params: Sb2FitParams,
    spec_data: dict,
    instrument,
    hot: bool,
    settings: Sb2FitSettings,
    vel_sb2_1: float,
    vel_sb2_2: float,
    vel_single: float,
) -> dict[str, float]:
    rp = float(instrument.resolving_power)
    chi2_sb2 = 0.0
    chi2_single = 0.0
    n_pix = 0
    valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
    for order in valid_orders[::3]:
        obs = _order_obs_norm(spec_data, order, hot, settings=settings)
        if obs is None:
            continue
        wave, obs_nf = obs
        c1 = _load_phoenix_component(params.teff1, params.logg1, params.feh, params.vsini1, vel_sb2_1, wave, rp)
        c2 = _load_phoenix_component(params.teff2, params.logg2, params.feh, params.vsini2, vel_sb2_2, wave, rp)
        c1s = _load_phoenix_component(params.teff1, params.logg1, params.feh, params.vsini1, vel_single, wave, rp)
        if c1 is None or c2 is None or c1s is None:
            continue
        _, f1, _ = c1
        _, f2, _ = c2
        _, f1s, _ = c1s
        norm_sb2 = _normalize_blend_order(wave, f1 + f2, hot, order, settings=settings)
        norm_single = _normalize_blend_order(wave, f1s, hot, order, settings=settings)
        if norm_sb2 is None or norm_single is None:
            continue
        nw_sb2, model_sb2 = norm_sb2
        nw_single, model_single = norm_single
        n = min(len(wave), len(obs_nf), len(nw_sb2), len(model_sb2), len(model_single))
        obs_nf = np.asarray(obs_nf[:n], float)
        model_sb2 = np.asarray(model_sb2[:n], float)
        model_single = np.asarray(model_single[:n], float)
        mask = _fit_pixel_mask(nw_sb2[:n], obs_nf, model_sb2, settings)
        if int(np.sum(mask)) < 20:
            continue
        n_pix += int(np.sum(mask))
        chi2_sb2 += float(np.sum((obs_nf[mask] - model_sb2[mask]) ** 2))
        chi2_single += float(np.sum((obs_nf[mask] - model_single[mask]) ** 2))
    delta = chi2_single - chi2_sb2
    return {
        "chi2_sb2": chi2_sb2,
        "chi2_single_star": chi2_single,
        "delta_chi2_single_minus_sb2": delta,
        "n_pixels": float(n_pix),
        "sb2_preferred": bool(delta > 0.0 and n_pix > 0),
    }


def build_diagnostics_report(
    sb2_dir: Path,
    *,
    spec_root: Path,
    instrument_name: str = "APF",
    blaze_path: Path | None = None,
) -> dict[str, Any]:
    sb2_dir = Path(sb2_dir)
    report = _load_json(sb2_dir / "sb2_report.json")
    fit_json = _load_json(sb2_dir / "sb2_fit.json")
    epochs_csv = sb2_dir / "sb2_epochs.csv"
    gaia_id = str(report.get("gaia_id", ""))
    instrument = get_instrument_profile(instrument_name)
    summary_path = Path(report.get("summary_path", config.OUTPUT_DIR / f"Gaia_DR3_{gaia_id}_summary.txt"))
    ctx = load_star_context(gaia_id, summary_path, force_gaia=False) if gaia_id else None
    hot = bool(ctx and ctx.teff > config.HOT_STAR_TEFF_THRESHOLD)

    fit_settings_data = fit_json.get("fit_settings", {})
    settings = build_sb2_fit_settings(
        continuum_mode=str(fit_settings_data.get("continuum_mode", "sinc_blaze")),
        blaze_calibration_path=blaze_path,
        order_edge_trim_frac=float(fit_settings_data.get("order_edge_trim_frac", 0.0)),
        rv_prior_sigma_kms=float(fit_settings_data.get("rv_prior_sigma_kms", 1.0)),
    )

    out: dict[str, Any] = {
        "sb2_dir": str(sb2_dir),
        "gaia_id": gaia_id,
        "teff_gaia": report.get("teff_gaia"),
        "logg_gaia": report.get("logg_gaia"),
        "mh_gaia": report.get("mh_gaia"),
        "median_delta_chi2_ccf": report.get("median_delta_chi2"),
        "fit_settings_used": fit_settings_data or {
            "continuum_mode": settings.continuum_mode,
            "order_edge_trim_frac": settings.order_edge_trim_frac,
            "rv_prior_sigma_kms": settings.rv_prior_sigma_kms,
        },
    }

    if epochs_csv.is_file():
        ep_df = pd.read_csv(epochs_csv)
    else:
        ep_df = pd.DataFrame()

    fit_epochs = {e["basename"]: e for e in fit_json.get("epochs", [])}

    if not ep_df.empty:
        rv_rows = []
        for _, row in ep_df.iterrows():
            bn = str(row["basename"])
            fe = fit_epochs.get(bn, {})
            rv_rows.append(
                {
                    "basename": bn,
                    "ccf_rv1_kms": float(row.get("rv1_kms", np.nan)),
                    "ccf_rv2_kms": float(row.get("rv2_kms", np.nan)),
                    "fit_vel1_kms": fe.get("vel1_kms"),
                    "fit_vel2_kms": fe.get("vel2_kms"),
                    "delta_vel1_kms": (
                        float(fe["vel1_kms"]) - float(row["rv1_kms"])
                        if fe.get("vel1_kms") is not None and np.isfinite(row.get("rv1_kms", np.nan))
                        else None
                    ),
                    "delta_vel2_kms": (
                        float(fe["vel2_kms"]) - float(row["rv2_kms"])
                        if fe.get("vel2_kms") is not None and np.isfinite(row.get("rv2_kms", np.nan))
                        else None
                    ),
                    "ccf_delta_chi2": float(row.get("delta_chi2", np.nan)),
                    "sb2_candidate": bool(row.get("sb2_candidate", False)),
                }
            )
        out["rv_table"] = rv_rows
        out["rv_max_abs_delta_vel1"] = max(
            (abs(r["delta_vel1_kms"]) for r in rv_rows if r.get("delta_vel1_kms") is not None),
            default=None,
        )
        out["rv_max_abs_delta_vel2"] = max(
            (abs(r["delta_vel2_kms"]) for r in rv_rows if r.get("delta_vel2_kms") is not None),
            default=None,
        )

    if fit_json.get("params") and gaia_id:
        p = Sb2FitParams(**fit_json["params"])
        out["stellar_params"] = {
            "teff1": p.teff1,
            "teff2": p.teff2,
            "logg1": p.logg1,
            "logg2": p.logg2,
            "feh": p.feh,
            "vsini1": p.vsini1,
            "vsini2": p.vsini2,
            "delta_teff1_from_gaia": p.teff1 - float(report.get("teff_gaia", np.nan)),
            "chi2_red": fit_json.get("chi2_red"),
            "n_data": fit_json.get("n_data"),
            "n_dof": fit_json.get("n_dof"),
        }
        teff2_grid_warn = abs(p.teff2 - round(p.teff2 / 300.0) * 300.0) < 1.0 and p.teff2 >= 3000
        out["stellar_params"]["teff2_on_300k_grid"] = bool(teff2_grid_warn)

        spec_paths = discover_epoch_paths(spec_root, gaia_id)
        path_by_bn = {p.name: p for p in spec_paths}
        norm_rows: list[dict[str, Any]] = []
        residual_rows: list[dict[str, Any]] = []
        chi2_rows: list[dict[str, Any]] = []
        fit_epoch_by_bn = {e["basename"]: e for e in fit_json.get("epochs", [])}

        for spec_path in spec_paths[:3]:
            _, spec_data = io_utils.read_spectrum(str(spec_path))
            valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
            if not valid_orders:
                continue
            ref_order = valid_orders[len(valid_orders) // 2]
            norm_rows.extend(_norm_mode_comparison(spec_data, ref_order, hot, blaze_path))

            ep_fit = fit_epoch_by_bn.get(spec_path.name, {})
            if not ep_fit:
                continue
            v1 = float(ep_fit.get("vel1_kms", 0.0))
            v2 = float(ep_fit.get("vel2_kms", 0.0))
            rb = _epoch_residual_breakdown(p, spec_data, instrument, hot, settings, v1, v2)
            if rb:
                rb["basename"] = spec_path.name
                residual_rows.append(rb)
            ep_df_row = ep_df[ep_df["basename"] == spec_path.name] if not ep_df.empty else pd.DataFrame()
            rv1_ccf = float(ep_df_row["rv1_kms"].iloc[0]) if len(ep_df_row) else v1
            chi2_cmp = _template_chi2_sb2_vs_single(p, spec_data, instrument, hot, settings, v1, v2, rv1_ccf)
            if chi2_cmp.get("n_pixels", 0) > 0:
                chi2_cmp["basename"] = spec_path.name
                chi2_rows.append(chi2_cmp)

        out["normalization_comparison"] = norm_rows
        out["residual_breakdown"] = residual_rows
        out["sb2_vs_single_star"] = chi2_rows
        if chi2_rows:
            out["median_delta_chi2_single_minus_sb2"] = float(
                np.median([r["delta_chi2_single_minus_sb2"] for r in chi2_rows])
            )

    return out


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SB2 fit diagnostics",
        "",
        f"- **Gaia ID:** {report.get('gaia_id')}",
        f"- **Directory:** `{report.get('sb2_dir')}`",
        f"- **Median CCF Δχ²:** {report.get('median_delta_chi2_ccf')}",
        "",
    ]
    sp = report.get("stellar_params")
    if sp:
        lines.extend(
            [
                "## Fitted stellar parameters",
                "",
                f"| Param | Value |",
                f"|-------|-------|",
                f"| Teff1 | {sp.get('teff1'):.0f} K (ΔGaia {sp.get('delta_teff1_from_gaia'):+.0f}) |",
                f"| Teff2 | {sp.get('teff2'):.0f} K |",
                f"| logg2 | {sp.get('logg2'):.2f} |",
                f"| vsini1 / vsini2 | {sp.get('vsini1'):.2f} / {sp.get('vsini2'):.2f} km/s |",
                f"| χ²_red | {sp.get('chi2_red')} (n_data={sp.get('n_data')}, dof={sp.get('n_dof')}) |",
                "",
            ]
        )
    if report.get("rv_max_abs_delta_vel1") is not None:
        lines.append(
            f"- **Max |fit_vel − CCF RV|:** primary {report['rv_max_abs_delta_vel1']:.2f} km/s, "
            f"secondary {report.get('rv_max_abs_delta_vel2', float('nan')):.2f} km/s"
        )
    if report.get("median_delta_chi2_single_minus_sb2") is not None:
        d = report["median_delta_chi2_single_minus_sb2"]
        pref = "SB2 preferred" if d > 0 else "single-star competitive"
        lines.append(f"- **SB2 vs single-star:** median Δχ²(single−SB2) = {d:.1f} ({pref})")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SB2 template-fit diagnostics report")
    ap.add_argument("--sb2-dir", type=Path, required=True)
    ap.add_argument("--spec-root", type=Path, default=Path("/Users/rfoley/darkhunter/rvs/data"))
    ap.add_argument("--instrument", default="APF", choices=["APF", "GHOST", "MAROON-X"])
    ap.add_argument("--blaze-calibration", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    blaze_path = args.blaze_calibration or config.BLAZE_CALIBRATION_FILE

    report = build_diagnostics_report(
        Path(args.sb2_dir),
        spec_root=Path(args.spec_root),
        instrument_name=str(args.instrument),
        blaze_path=blaze_path,
    )
    out_dir = Path(args.sb2_dir)
    json_path = out_dir / "sb2_diagnostics_summary.json"
    md_path = out_dir / "sb2_diagnostics_summary.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_summary_markdown(report), encoding="utf-8")
    logger.info("Wrote %s and %s", json_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
