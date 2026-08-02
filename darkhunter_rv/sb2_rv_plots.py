"""RV and CCF figures for SB2 orbit fits."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from darkhunter_rv.rv_keplerian_plots import (
    FIT_VARIANT_LABEL,
    FIT_VARIANT_ORDER,
    FIT_VARIANT_STYLE,
    _plot_fit_curves,
    _plot_points,
    _style_rv_axes,
    _variant_param_lines,
)
from darkhunter_rv.sb2_rv_fit import Sb2EpochRow, Sb2TemplateMasses, predict_joint_rvs
from fit_apf_rv_keplerian import RVPoint, mass_function_msun, solve_m2sini_msun

_PREFERRED_VARIANT_ORDER = ("fix_period_ecc", "fix_period", "fix_ecc", "free")


def _preferred_variant_key(fit_variants: dict[str, tuple[np.ndarray, dict]]) -> str | None:
    for key in _PREFERRED_VARIANT_ORDER:
        if key in fit_variants:
            return key
    return next(iter(fit_variants), None)


def _annotate_orbit_params(
    ax,
    fit_variants: dict[str, tuple[np.ndarray, dict]],
    m1_msun: float,
    *,
    companion_label: str = "M₂ sin(i)",
    primary_label: str = "M₁",
) -> None:
    """P, e, and mass annotations matching ``build_plot`` in fit_apf_rv_keplerian."""
    key = _preferred_variant_key(fit_variants)
    if key is None:
        return
    _, rep = fit_variants[key]
    if rep.get("P_days") is None or rep.get("e") is None:
        return
    p_day = int(round(float(rep["P_days"])))
    ecc = float(rep["e"])
    ax.text(
        0.015,
        0.985,
        f"P={p_day} d   e={ecc:.3f}",
        transform=ax.transAxes,
        fontsize=9.5,
        color="tab:red",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="tab:red", alpha=0.9, boxstyle="round,pad=0.2"),
    )
    m2sini = None
    if m1_msun > 0 and np.isfinite(m1_msun):
        fm = rep.get("mass_function_msun")
        if fm is None or not np.isfinite(fm):
            k = rep.get("K_kms")
            if k is not None and np.isfinite(k):
                fm = mass_function_msun(float(rep["P_days"]), float(k), ecc)
        if fm is not None and np.isfinite(fm) and float(fm) > 0:
            m2sini = float(solve_m2sini_msun(float(fm), float(m1_msun)))
    if m2sini is not None:
        ax.text(
            0.015,
            0.935,
            f"{companion_label}={m2sini:.4f} M⊙   {primary_label}={m1_msun:.4f} M⊙",
            transform=ax.transAxes,
            fontsize=9.2,
            color="black",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.95, boxstyle="round,pad=0.2"),
        )


def _t_span(points: Sequence[RVPoint]) -> tuple[float, float]:
    if not points:
        return 0.0, 1.0
    t = np.array([p.mjd for p in points], float)
    pad = max(0.03 * float(np.ptp(t)), 5.0)
    return float(np.min(t) - pad), float(np.max(t) + pad)


def plot_sb2_independent_fit(
    gaia_id: str,
    pts1: Sequence[RVPoint],
    pts2: Sequence[RVPoint],
    variants: dict[str, dict[str, tuple[np.ndarray, dict]]],
    out_png: Path,
    *,
    masses: Sb2TemplateMasses,
) -> None:
    """Two-panel independent Keplerian fits (star1 all epochs, star2 detected only)."""
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.6), sharex=True)
    panel_cfg = (
        (axes[0], "Star 1 (all epochs)", pts1, "star1", masses.m1_prior_msun, "M₂ sin(i)", "M₁"),
        (axes[1], "Star 2 (sb2_candidate)", pts2, "star2", masses.m2_prior_msun, "M₁ sin(i)", "M₂"),
    )
    for ax, label, pts, var_key, m_primary, comp_lbl, prim_lbl in panel_cfg:
        if len(pts) < 2:
            ax.set_title(f"{label}: insufficient data")
            continue
        t = np.array([p.mjd for p in pts], float)
        t_lo, t_hi = _t_span(pts)
        t_dense = np.linspace(t_lo, t_hi, 1500)
        fit_vars = variants.get(var_key, {})
        t_ref = float(fit_vars.get("free", (None, {"t_ref_mjd": float(np.median(t))}))[1].get("t_ref_mjd", np.median(t)))
        _plot_points(ax, pts, include_literature=False)
        if fit_vars:
            _plot_fit_curves(ax, t_dense, fit_vars, t_ref)
            _annotate_orbit_params(
                ax,
                fit_vars,
                float(m_primary),
                companion_label=comp_lbl,
                primary_label=prim_lbl,
            )
            param_lines = _variant_param_lines(fit_vars, float(m_primary))
            if param_lines:
                ax.text(
                    0.99,
                    0.02,
                    "\n".join(param_lines),
                    transform=ax.transAxes,
                    fontsize=7.0,
                    va="bottom",
                    ha="right",
                    family="monospace",
                    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92, boxstyle="round,pad=0.2"),
                )
            handles = [
                plt.Line2D([0], [0], color=FIT_VARIANT_STYLE[k][0], ls=FIT_VARIANT_STYLE[k][1], lw=1.8, label=FIT_VARIANT_LABEL[k])
                for k in FIT_VARIANT_ORDER
                if k in fit_vars
            ]
            ax.legend(handles=handles, loc="upper center", fontsize=7.5, ncol=2)
        ax.set_ylabel("RV (km/s)")
        ax.set_title(label)
        ax.set_xlim(t_lo, t_hi)
        _style_rv_axes(ax)
    axes[1].set_xlabel("MJD")
    fig.suptitle(f"SB2 independent fits: Gaia DR3 {gaia_id}", y=1.01)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_joint_sb2_fit(
    gaia_id: str,
    epochs: list[Sb2EpochRow],
    joint_variants: dict[str, tuple[np.ndarray, dict]],
    predictions: list[dict[str, Any]],
    out_png: Path,
    *,
    variant_key: str | None = None,
) -> None:
    """Both components on one plot; missing star2 epochs show predicted open markers."""
    key = variant_key or ("fix_period_ecc" if "fix_period_ecc" in joint_variants else "free")
    if key not in joint_variants:
        return
    params, rep = joint_variants[key]
    pred_by_bn = {p["basename"]: p for p in predictions}
    t1, rv1, e1 = [], [], []
    t2_det, rv2_det, e2 = [], [], []
    t2_pred, rv2_pred = [], []
    for row in epochs:
        if not np.isfinite(row.mjd):
            continue
        rv1_use = row.rv1_kms if np.isfinite(row.rv1_kms) else row.summary_rv_kms
        if np.isfinite(rv1_use):
            t1.append(row.mjd)
            rv1.append(rv1_use)
            e1.append(row.rv1_err_kms if np.isfinite(row.rv1_err_kms) else row.summary_rv_err_kms)
        if row.sb2_candidate and np.isfinite(row.rv2_kms):
            t2_det.append(row.mjd)
            rv2_det.append(row.rv2_kms)
            e2.append(row.rv2_err_kms if np.isfinite(row.rv2_err_kms) else row.summary_rv_err_kms)
        elif not row.sb2_candidate:
            pr = pred_by_bn.get(row.basename)
            if pr is not None and np.isfinite(pr.get("rv2_pred_kms", float("nan"))):
                t2_pred.append(row.mjd)
                rv2_pred.append(pr["rv2_pred_kms"])
    t_all = np.array(t1 + t2_det + t2_pred, float)
    if t_all.size < 2:
        return
    t_lo, t_hi = float(np.min(t_all) - 5), float(np.max(t_all) + 5)
    t_dense = np.linspace(t_lo, t_hi, 2000)
    rv1_curve, rv2_curve = predict_joint_rvs(rep, params, t_dense)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.errorbar(t1, rv1, yerr=e1, fmt="o", ms=5, capsize=2, color="tab:blue", label="Star 1")
    if t2_det:
        ax.errorbar(t2_det, rv2_det, yerr=e2, fmt="s", ms=5, capsize=2, color="tab:orange", label="Star 2 (detected)")
    if t2_pred:
        ax.plot(t2_pred, rv2_pred, "s", ms=7, mfc="none", mec="tab:orange", mew=1.5, label="Star 2 (predicted)")
    ax.plot(t_dense, rv1_curve, "-", color="tab:blue", lw=1.8, alpha=0.85, label=f"Star 1 model ({key})")
    ax.plot(t_dense, rv2_curve, "--", color="tab:orange", lw=1.8, alpha=0.85, label=f"Star 2 model ({key})")
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    ax.set_title(f"SB2 joint fit: Gaia DR3 {gaia_id}  P={rep.get('P_days', float('nan')):.1f} d  e={rep.get('e', float('nan')):.3f}")
    ax.set_xlim(t_lo, t_hi)
    ax.legend(loc="best", fontsize=8)
    _style_rv_axes(ax)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_missing_epoch_ccf(
    vel_kms: np.ndarray,
    ccf: np.ndarray,
    *,
    rv1_pred: float,
    rv2_pred: float,
    out_path: Path,
    title: str = "",
) -> None:
    """Median CCF with joint-fit RV predictions for epochs without star2 detection."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(vel_kms, ccf, "-", color="0.2", lw=1.2, label="median CCF")
    ax.axvline(rv1_pred, color="tab:blue", ls="-", lw=1.5, label=f"rv1 pred {rv1_pred:.1f} km/s")
    ax.axvline(rv2_pred, color="tab:orange", ls="--", lw=1.5, label=f"rv2 pred {rv2_pred:.1f} km/s")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("CCF")
    ax.set_title(title or "CCF with joint-fit RV lines")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
