"""Tests for method fusion v1: tiered pick, bias, σ inflation, discordance reject."""
from __future__ import annotations

import numpy as np
import pytest

from darkhunter_rv.method_fusion import (
    BiasSurface,
    FusionConfig,
    calibrated_rv_kms,
    discordance_reject,
    fuse_exposure,
    inflate_sigma_kms,
    inter_method_spread_kms,
    select_method_tiered,
)
from validation.rv_method_diagnostics_report import _binned_method_coverage_table


def _flags(
    *,
    mask_rv: float = 0.0,
    mask_err: float = 0.2,
    tpl_rv: float = 0.1,
    tpl_err: float = 0.2,
    sl_rv: float = 0.0,
    sl_err: float = 0.3,
    mask_valid: bool = True,
    tpl_valid: bool = True,
    sl_valid: bool = True,
    snr: float = 20.0,
) -> dict:
    return {
        "mask_valid": mask_valid,
        "template_valid": tpl_valid,
        "strong_lines_valid": sl_valid,
        "mask_rv_kms": mask_rv,
        "mask_err_kms": mask_err,
        "template_rv_kms": tpl_rv,
        "template_err_kms": tpl_err,
        "strong_lines_rv_kms": sl_rv,
        "strong_lines_err_kms": sl_err,
        "median_mask_ccf_peak_snr": snr,
    }


def test_bias_surface_constant_and_interp():
    const = BiasSurface(constant_kms=1.5)
    assert const.delta(5000.0) == pytest.approx(1.5)
    surf = BiasSurface(teff_knots_k=(4000.0, 6000.0), delta_kms=(0.0, 2.0))
    assert surf.delta(5000.0) == pytest.approx(1.0)
    cfg = FusionConfig(bias_mask=BiasSurface(constant_kms=3.0))
    assert calibrated_rv_kms(10.0, "mask_ccf", 5500.0, cfg) == pytest.approx(7.0)


def test_tiered_hot_prefers_template():
    fl = _flags()
    cfg = FusionConfig(hot_teff_k=6500.0, mask_snr_min=5.0)
    assert select_method_tiered(fl, teff=7000.0, cfg=cfg) == "template_fft"


def test_tiered_cool_high_snr_prefers_mask():
    fl = _flags(snr=20.0)
    cfg = FusionConfig(hot_teff_k=6500.0, mask_snr_min=5.0)
    assert select_method_tiered(fl, teff=5000.0, cfg=cfg) == "mask_ccf"


def test_tiered_cool_low_snr_falls_to_template():
    fl = _flags(snr=1.0)
    cfg = FusionConfig(hot_teff_k=6500.0, mask_snr_min=5.0)
    assert select_method_tiered(fl, teff=5000.0, cfg=cfg) == "template_fft"


def test_discordance_rejects_large_delta_despite_small_sigma():
    cal = {"mask_ccf": 0.0, "template_fft": 50.0, "strong_lines": float("nan")}
    sig = {"mask_ccf": 0.2, "template_fft": 0.2, "strong_lines": float("nan")}
    valid = {"mask_ccf": True, "template_fft": True, "strong_lines": False}
    cfg = FusionConfig(discord_eta_kms=10.0, discord_sigma_mult=5.0)
    reject, reason = discordance_reject(cal, sig, valid, cfg)
    assert reject is True
    assert reason == "discordance"


def test_discordance_accepts_agreement():
    cal = {"mask_ccf": 0.0, "template_fft": 0.3, "strong_lines": float("nan")}
    sig = {"mask_ccf": 0.2, "template_fft": 0.2, "strong_lines": float("nan")}
    valid = {"mask_ccf": True, "template_fft": True, "strong_lines": False}
    cfg = FusionConfig(discord_eta_kms=10.0, discord_sigma_mult=5.0)
    reject, reason = discordance_reject(cal, sig, valid, cfg)
    assert reject is False
    assert reason == ""


def test_fuse_exposure_rejects_discordance():
    fl = _flags(mask_rv=0.0, tpl_rv=40.0, mask_err=0.15, tpl_err=0.15)
    out = fuse_exposure(fl, teff=5500.0)
    assert out["rv_accepted"] is False
    assert out["reject_reason"] == "discordance"
    assert not np.isfinite(out["rv_calibrated_kms"])


def test_fuse_exposure_accepts_and_inflates_sigma():
    fl = _flags(mask_rv=1.0, tpl_rv=1.2, sl_valid=False)
    cfg = FusionConfig(
        inflation_k=1.0,
        sigma_floor_mask_kms=0.1,
        bias_mask=BiasSurface(constant_kms=0.5),
    )
    out = fuse_exposure(fl, teff=5500.0, cfg=cfg)
    assert out["rv_accepted"] is True
    assert out["adopted_method_v2"] == "mask_ccf"
    assert out["rv_calibrated_kms"] == pytest.approx(0.5)
    # spread = 0.2 (calibrated mask=0.5, template=1.2 → wait bias only on mask)
    # calibrated mask = 1.0 - 0.5 = 0.5; template = 1.2 - 0 = 1.2; spread = 0.7
    spread = out["inter_method_spread_kms"]
    assert spread == pytest.approx(0.7)
    expected = float(np.sqrt(0.2**2 + 0.1**2 + (1.0 * 0.7) ** 2))
    assert out["sigma_eff_kms"] == pytest.approx(expected)


def test_fuse_no_valid_method():
    fl = _flags(mask_valid=False, tpl_valid=False, sl_valid=False)
    out = fuse_exposure(fl, teff=5500.0)
    assert out["rv_accepted"] is False
    assert out["reject_reason"] == "no_valid_method"


def test_inflate_sigma_no_spread_when_single_method():
    cfg = FusionConfig(inflation_k=2.0, sigma_floor_template_kms=0.3)
    s = inflate_sigma_kms(0.4, "template_fft", 5.0, cfg, n_methods_valid=1)
    assert s == pytest.approx(float(np.sqrt(0.4**2 + 0.3**2)))


def test_inter_method_spread_nan_if_one_valid():
    assert not np.isfinite(
        inter_method_spread_kms(
            {"mask_ccf": 1.0, "template_fft": float("nan"), "strong_lines": float("nan")},
            {"mask_ccf": True, "template_fft": False, "strong_lines": False},
        )
    )


def test_coverage_table_counts_nan_in_denominator():
    teff = np.array([4500.0, 5500.0, 6500.0, float("nan")])
    rv_m = np.array([1.0, float("nan"), 2.0, 3.0])
    rv_t = np.array([1.0, 1.5, float("nan"), 3.0])
    rv_s = np.array([float("nan"), float("nan"), float("nan"), float("nan")])
    tab = _binned_method_coverage_table(
        teff,
        rv_m,
        rv_t,
        rv_s,
        2,
        teff_bin_lo=4000.0,
        teff_bin_hi=7000.0,
    )
    assert len(tab) == 2
    # bins: [4000,5500), [5500,7000]
    row0 = tab.iloc[0]
    assert row0["n_total"] == 1  # 4500 only
    assert row0["n_finite_mask"] == 1
    assert row0["frac_finite_mask"] == pytest.approx(1.0)
    row1 = tab.iloc[1]
    assert row1["n_total"] == 2  # 5500, 6500
    assert row1["n_finite_mask"] == 1  # NaN at 5500 excluded from finite
    assert row1["frac_finite_mask"] == pytest.approx(0.5)
    assert row1["n_finite_template"] == 1
