"""Unit tests for Joker prior specs, masses, and JSON helpers (no live NUTS)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_rv.joker_rv_fit import (
    envelope_report,
    is_unimodal_enough,
    masses_from_report_variants,
    mean_anomaly_rad,
    median_params_from_arrays,
    period_bounds_days,
    prior_spec_for_variant,
    should_skip_refit,
    sigma_k0_kms,
    sigma_v_kms,
    summarize_sample_arrays,
    t_periastron_gaia_to_mjd,
)
from fit_apf_rv_keplerian import enrich_nss_keplerian_fields, website_table_masses_from_report


def test_sigma_k0_and_sigma_v() -> None:
    rvs = np.array([10.0, 20.0, 30.0])
    assert sigma_k0_kms(rvs) == 30.0
    assert sigma_v_kms(rvs) == 100.0
    rvs2 = np.array([80.0, 90.0, 100.0])
    assert sigma_k0_kms(rvs2) == 90.0
    assert sigma_v_kms(rvs2) == 100.0
    rvs3 = np.array([-120.0, 10.0])
    assert sigma_v_kms(rvs3) == 120.0


def test_period_bounds_expand_for_long_nss() -> None:
    lo, hi = period_bounds_days({"period_days": 2500.0, "period_days_error": 10.0})
    assert lo == 20.0
    assert hi >= 2550.0


def test_ecc_prior_is_truncated() -> None:
    spec = prior_spec_for_variant(
        "ecc",
        {"eccentricity": 0.2, "eccentricity_error": 0.05},
        t_ref_mjd=60000.0,
    )
    assert spec["skip_reason"] is None
    assert spec["fields"]["e"]["truncated"] is True


def test_full_prior_needs_omega() -> None:
    spec = prior_spec_for_variant(
        "full",
        {
            "period_days": 100.0,
            "period_days_error": 1.0,
            "eccentricity": 0.1,
            "eccentricity_error": 0.02,
        },
        t_ref_mjd=60000.0,
    )
    assert spec["skip_reason"] == "missing_omega_prior"


def test_t0_and_m0() -> None:
    mjd = t_periastron_gaia_to_mjd(0.0)
    assert mjd > 57000.0
    m0 = mean_anomaly_rad(mjd, 100.0, mjd)
    assert m0 == 0.0


def test_summarize_and_masses() -> None:
    n = 40
    p = np.full(n, 100.0)
    k = np.full(n, 20.0)
    e = np.full(n, 0.1)
    om = np.zeros(n)
    m0 = np.zeros(n)
    g = np.full(n, 5.0)
    t = np.linspace(60000.0, 60100.0, 8)
    y = np.zeros(8)
    yerr = np.ones(8)
    block = summarize_sample_arrays(
        p_days=p,
        k_kms=k,
        e=e,
        omega_rad=om,
        m0_rad=m0,
        gamma_kms=g,
        t=t,
        y=y,
        yerr=yerr,
        t_ref_mjd=float(np.median(t)),
        sampler="rejection",
        variant="rv_only",
        m1_msun=1.0,
        inclination_deg=60.0,
    )
    assert block["P_days"] == pytest.approx(100.0)
    assert block["n_samples"] == 40
    masses = masses_from_report_variants(
        {"rv_only": block, "full": block},
        m1_msun=1.0,
        inclination_deg=60.0,
        used_m2_msun=0.4,
    )
    assert masses["used_m2_msun"] == 0.4
    assert masses["m2sini_msun"] is not None
    assert masses["m2_rv_astrometry_msun"] is not None


def test_skip_refit(tmp_path: Path) -> None:
    summ = tmp_path / "s.txt"
    js = tmp_path / "j.json"
    summ.write_text("x")
    js.write_text('{"n_points": 5}')
    assert should_skip_refit(js, summ, 5, force=False)
    assert not should_skip_refit(js, summ, 6, force=False)
    assert not should_skip_refit(js, summ, 5, force=True)


def test_unimodal_gate() -> None:
    tight = np.full(20, 100.0) + np.linspace(-0.1, 0.1, 20)
    wide = np.concatenate([np.full(20, 50.0), np.full(20, 400.0)])
    assert is_unimodal_enough(tight)
    assert not is_unimodal_enough(wide)


def test_envelope_aliases_free() -> None:
    variants = {"rv_only": {"P_days": 10.0, "K_kms": 1.0, "e": 0.0, "params_raw": [1, 1, 0, 0, 0, 0]}}
    env = envelope_report(
        gaia_source_id="1",
        summary_file="s",
        n_points=4,
        t_ref_mjd=1.0,
        now_mjd=2.0,
        gaia_nss=None,
        variants=variants,
        masses={"m2sini_msun": 0.1},
        observability_window=None,
    )
    assert env["fit_variants"]["free"]["P_days"] == 10.0


def test_enrich_nss_from_thiele_innes() -> None:
    from darkhunter_rv.thiele_innes_inclination import campbell_to_thiele_innes

    A, B, F, G = campbell_to_thiele_innes(2.0, 60.0, 40.0, 100.0)
    out: dict = {"period_days": 80.0, "eccentricity": 0.1}
    enrich_nss_keplerian_fields(
        {
            "period_error": 1.5,
            "eccentricity_error": 0.02,
            "a_thiele_innes": A,
            "b_thiele_innes": B,
            "f_thiele_innes": F,
            "g_thiele_innes": G,
        },
        out,
    )
    assert out["period_days_error"] == 1.5
    assert "omega_deg" in out
    assert "inclination_deg" in out


def test_website_masses_joker_full_column() -> None:
    p_days = 100.0
    k = 30.0
    e = 0.1
    from fit_apf_rv_keplerian import mass_function_msun

    fm = mass_function_msun(p_days, k, e)
    rep = {
        "used_m2_msun": 0.55,
        "used_m1_msun": 1.0,
        "gaia_nss": {"inclination_deg": 60.0},
        "fit_variants": {
            "rv_only": {"P_days": p_days, "K_kms": k, "e": e, "mass_function_msun": fm},
            "full": {"P_days": p_days, "K_kms": k, "e": e, "mass_function_msun": fm},
        },
    }
    cols = website_table_masses_from_report(rep)
    assert cols["m2_msun"] == pytest.approx(0.55)
    assert cols["m2sin_i_msun"] is not None
    assert cols["m2_at_i_msun"] is not None
    assert cols["m2_rv_astrometry_msun"] is not None


def test_median_params_picks_nearest_p() -> None:
    p = np.array([90.0, 100.0, 110.0])
    k = np.array([1.0, 2.0, 3.0])
    e = np.array([0.1, 0.2, 0.3])
    om = np.zeros(3)
    m0 = np.zeros(3)
    g = np.zeros(3)
    params = median_params_from_arrays(p, k, e, om, m0, g)
    assert params[1] == 2.0
