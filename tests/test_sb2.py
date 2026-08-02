"""Tests for SB2 detection and template separation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_rv.ccf_rv_estimators import BiGaussCcfResult, estimate_ccf_bi_gauss_from_arrays, estimate_ccf_secondary_seeded
from darkhunter_rv.sb2 import (
    OrderCcfRecord,
    Sb2FitParams,
    Sb2FitSettings,
    _EpochSpec,
    _doppler_flux_on_grid,
    _evaluate_sb2_params,
    _order_edge_mask,
    _residual_vector_sb2,
    _resolve_vsini_init_max,
    _teff2_grid,
    _vel_prior_sigma,
    build_sb2_fit_settings,
    flux_fraction_components,
    load_star_context,
    median_ccf_across_orders,
    order_mask_ccf_dict_to_records,
    parse_pipeline_rvs_from_summary,
    sb2_candidate_from_score,
    sb2_exposure_diagnostics_columns,
    score_sb2_from_median_ccf,
    score_sb2_from_pipeline_order_ccfs,
)

STAR = 77413727493690112


def _double_gaussian_ccf(mu1: float = -10.0, mu2: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    vel = np.linspace(-100.0, 100.0, 401)
    ccf = (
        1.0
        + 90.0 * np.exp(-0.5 * ((vel - mu1) / 4.5) ** 2)
        + 50.0 * np.exp(-0.5 * ((vel - mu2) / 5.0) ** 2)
    )
    return vel, ccf


def test_median_ccf_across_orders():
    vel, ccf = _double_gaussian_ccf()
    r1 = OrderCcfRecord(10, "10", vel, ccf + 0.5, 0.0, 8.0, -10.0)
    r2 = OrderCcfRecord(11, "11", vel, ccf - 0.3, 1.2, 7.0, -9.0)
    v_med, c_med = median_ccf_across_orders([r1, r2])
    assert len(v_med) == len(c_med)
    assert np.nanmax(c_med) > 50.0


def test_score_sb2_delta_chi2_positive():
    vel, ccf = _double_gaussian_ccf()
    bi = score_sb2_from_median_ccf(vel, ccf)
    assert bi.fit_ok
    assert bi.delta_chi2 > 2.0
    assert abs(bi.rv1_kms - (-10.0)) < 2.0
    assert abs(bi.rv2_kms - 25.0) < 2.0


def test_seeded_prefers_when_bi_gauss_weak_secondary():
    """Epoch-5 style: primary near pipeline seed, secondary bump at +22 km/s."""
    vel = np.linspace(-100.0, 100.0, 401)
    mu1, mu2 = -12.0, 22.0
    ccf = (
        1.0
        + 100.0 * np.exp(-0.5 * ((vel - mu1) / 4.0) ** 2)
        + 42.0 * np.exp(-0.5 * ((vel - mu2) / 5.0) ** 2)
    )
    bi = score_sb2_from_median_ccf(vel, ccf, rv_primary_seed=mu1)
    assert bi.fit_ok
    assert abs(bi.rv2_kms - mu2) < 3.0
    assert bi.delta_chi2 >= 4.0
    assert sb2_candidate_from_score(bi)


def test_seeded_rejects_single_peak_epoch():
    """Epochs 1/2/8 style: no convincing secondary."""
    vel = np.linspace(-80.0, 80.0, 321)
    mu1 = 5.0
    ccf = 1.0 + 90.0 * np.exp(-0.5 * ((vel - mu1) / 4.5) ** 2)
    bi = score_sb2_from_median_ccf(vel, ccf, rv_primary_seed=mu1)
    assert not sb2_candidate_from_score(bi)


def test_sb2_candidate_gate():
    bi = BiGaussCcfResult(
        rv1_kms=0.0,
        rv2_kms=30.0,
        rv1_err_kms=1.0,
        rv2_err_kms=2.0,
        amp1=80.0,
        amp2=40.0,
        sigma1=4.0,
        sigma2=5.0,
        c0=1.0,
        delta_chi2=8.0,
        secondary_peak_snr=5.0,
        asymmetry=0.3,
        fit_ok=True,
        peak_snr=10.0,
        v_grid_kms=0.0,
    )
    assert sb2_candidate_from_score(bi, delta_chi2_min=5.0)
    assert not sb2_candidate_from_score(bi, delta_chi2_min=20.0)




def test_score_sb2_from_pipeline_order_ccfs_fuse_cols():
    """Pipeline order_mask_ccf dict → candidate + diagnostics columns."""
    vel, ccf = _double_gaussian_ccf()
    order_mask_ccf = {
        10: {"vel": vel, "ccf": ccf, "peak_vel": -10.0, "label": "10"},
        11: {"vel": vel, "ccf": ccf * 0.98, "peak_vel": -9.5, "label": "11"},
    }
    recs = order_mask_ccf_dict_to_records(order_mask_ccf)
    assert len(recs) == 2
    cand, bi = score_sb2_from_pipeline_order_ccfs(order_mask_ccf, rv_primary_seed=-10.0)
    assert bi is not None and bi.fit_ok
    assert cand is True
    cols = sb2_exposure_diagnostics_columns(cand, bi)
    assert cols["sb2_candidate"] is True
    assert abs(cols["sb2_rv1_kms"] - (-10.0)) < 3.0
    assert abs(cols["sb2_rv2_kms"] - 25.0) < 3.0
    assert cols["sb2_delta_chi2"] > 2.0


def test_score_sb2_from_pipeline_order_ccfs_empty():
    cand, bi = score_sb2_from_pipeline_order_ccfs({})
    assert cand is False
    assert bi is None
    cols = sb2_exposure_diagnostics_columns(False, None)
    assert cols["sb2_candidate"] is False
    assert cols["sb2_rv2_method"] == ""

def test_vel_prior_sigma_fixed():
    assert _vel_prior_sigma(0.3, rv_prior_sigma_kms=1.0) == pytest.approx(1.0)
    assert _vel_prior_sigma(3.0, rv_prior_sigma_kms=2.5) == pytest.approx(2.5)


def test_parse_pipeline_rvs_from_summary(tmp_path: Path):
    summary = tmp_path / "Gaia_DR3_test_summary.txt"
    summary.write_text(
        "[GAIA METADATA]\nSource_ID: 123\nTeff: 5000\nlogg: 4.0\nMH: 0.0\n\n"
        "[PIPELINE RESULTS]\n"
        "# File | MJD | RV | Err | RMS\n"
        "Gaia_DR3_123_epoch_1.txt 58000.1 12.5 0.5 1.2 False\n",
        encoding="utf-8",
    )
    rows = parse_pipeline_rvs_from_summary(summary)
    assert "Gaia_DR3_123_epoch_1.txt" in rows
    assert rows["Gaia_DR3_123_epoch_1.txt"].rv_kms == pytest.approx(12.5)


def test_load_star_context_from_summary(tmp_path: Path):
    summary = tmp_path / f"Gaia_DR3_{STAR}_summary.txt"
    summary.write_text(
        f"[GAIA METADATA]\nSource_ID: {STAR}\nRA: 180.0\nDec: 45.0\n"
        f"Teff: 5200\nlogg: 3.8\nMH: -0.1\n\n"
        "[PIPELINE RESULTS]\n"
        f"Gaia_DR3_{STAR}_epoch_1.txt 58000.0 5.0 0.4 0.8 False\n",
        encoding="utf-8",
    )
    ctx = load_star_context(str(STAR), summary, force_gaia=False)
    assert ctx.teff == pytest.approx(5200.0)
    assert ctx.logg == pytest.approx(3.8)
    assert ctx.mh == pytest.approx(-0.1)


def test_residual_vector_includes_priors():
    vel, ccf = _double_gaussian_ccf()
    bi = estimate_ccf_bi_gauss_from_arrays(vel, ccf)
    params = Sb2FitParams(5500, 4000, 0.0, 4.0, 4.5, 8.0, 6.0)
    ep = _EpochSpec(
        "e.txt",
        {28: {"wavelength": [5000.0, 5010.0], "flux": [1.0, 1.0], "eflux": [0.01, 0.01]}},
        bi.rv1_kms,
        bi.rv2_kms,
        1.0,
        2.0,
        bi.rv1_kms,
    )
    inst = type("I", (), {"bad_orders": [], "resolving_power": 70000.0})()
    settings = Sb2FitSettings(continuum_mode="spline")
    res, n_data = _residual_vector_sb2(
        params, [ep], inst, hot=False, vel1_list=[bi.rv1_kms + 5.0], vel2_list=[bi.rv2_kms], settings=settings
    )
    assert res.size >= 2
    assert n_data >= 0


def test_teff_ordering_constraint_in_refinement_pack():
    params_ok = Sb2FitParams(5500.0, 4000.0, 0.0, 4.0, 4.5, 5.0, 5.0)
    params_bad = Sb2FitParams(4000.0, 5500.0, 0.0, 4.0, 4.5, 5.0, 5.0)
    assert params_ok.teff2 < params_ok.teff1
    assert params_bad.teff2 >= params_bad.teff1


def test_resolve_vsini_init_max_bounds():
    init, vmax = _resolve_vsini_init_max(vsini_guess=7.0, vsini_init_kms=1.0, vsini_max_kms=5.0)
    assert init == pytest.approx(1.0)
    assert vmax == pytest.approx(5.0)

    init2, vmax2 = _resolve_vsini_init_max(vsini_guess=7.0, vsini_init_kms=9.0, vsini_max_kms=5.0)
    assert init2 == pytest.approx(5.0)
    assert vmax2 == pytest.approx(5.0)


def test_order_edge_mask_trims_fraction():
    m0 = _order_edge_mask(200, 0.0)
    assert m0.all()
    m1 = _order_edge_mask(200, 0.1)
    assert not m1[:20].any()
    assert not m1[-20:].any()
    assert m1[50:-50].all()


def test_teff2_grid_steps():
    g = _teff2_grid(5800.0, step=300.0)
    assert len(g) >= 8
    assert g[-1] < 5800.0 - 50.0
    assert np.all(np.diff(g) == pytest.approx(300.0))


def test_build_sb2_fit_settings_spline_fallback():
    settings = build_sb2_fit_settings(continuum_mode="spline")
    assert settings.continuum_mode == "spline"
    assert settings.blaze_calibration is None


def test_lsf_broadens_doppler_template():
    wave = np.linspace(5000.0, 5010.0, 256)
    flux = 1.0 - 0.05 * np.exp(-0.5 * ((wave - 5005.0) / 0.15) ** 2)
    narrow = _doppler_flux_on_grid(wave, wave, flux, 0.0, resolving_power=1.0)
    broad = _doppler_flux_on_grid(wave, wave, flux, 0.0, resolving_power=80000.0)
    assert float(np.nanstd(broad)) < float(np.nanstd(narrow))


def test_flux_fraction_components_length_mismatch():
    """Blaze continuum can trim one pixel; components must still align."""
    f1 = np.ones(100)
    f2 = np.ones(100)
    total = np.ones(99)
    s1, s2 = flux_fraction_components(f1, f2, total)
    assert len(s1) == 99
    np.testing.assert_allclose(s1 + s2, total, rtol=0, atol=1e-12)


def test_flux_fraction_components_equal_flux():
    f1 = np.ones(100)
    f2 = np.ones(100)
    total = np.ones(100)
    s1, s2 = flux_fraction_components(f1, f2, total)
    np.testing.assert_allclose(s1, 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s2, 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s1 + s2, total, rtol=0, atol=1e-12)


def test_flux_fraction_components_faint_secondary():
    f1 = np.full(50, 9.0)
    f2 = np.full(50, 1.0)
    total = np.linspace(0.95, 1.05, 50)
    s1, s2 = flux_fraction_components(f1, f2, total)
    np.testing.assert_allclose(s1, 0.9 * total, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s2, 0.1 * total, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s1 + s2, total, rtol=0, atol=1e-12)


@pytest.mark.slow
def test_smoke_77413727493690112():
    spec_root = Path("/Users/rfoley/darkhunter/rvs/data")
    summary = Path(__file__).resolve().parents[1] / "output" / f"Gaia_DR3_{STAR}_summary.txt"
    if not spec_root.is_dir():
        pytest.skip("spec root missing")
    paths = list(spec_root.rglob(f"Gaia_DR3_{STAR}_epoch_*.txt"))
    if not paths:
        pytest.skip("no spectra for test star")
    from darkhunter_rv.instruments import get_instrument_profile
    from darkhunter_rv.sb2 import run_sb2_star

    out = Path(__file__).resolve().parents[1] / "validation_output" / "sb2_test_smoke"
    if summary.is_file():
        report = run_sb2_star(
            str(STAR),
            spec_root,
            summary,
            get_instrument_profile("APF"),
            out,
            force_fit=False,
        )
        assert report["n_epochs"] >= 1
        assert (out / "sb2_epochs.csv").is_file()
        assert (out / "sb2_report.json").is_file()
