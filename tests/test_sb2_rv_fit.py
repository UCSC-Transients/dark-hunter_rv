"""Tests for SB2 orbit RV fitting."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from darkhunter_rv.sb2_rv_fit import (
    GaussianPrior,
    OrbitPriors,
    Sb2EpochRow,
    Sb2TemplateMasses,
    build_rv_points_star1,
    fit_independent_sb2,
    fit_joint_sb2_all_variants,
    load_template_masses,
    orbital_unit_amplitude,
    predict_joint_rvs,
)
from fit_apf_rv_keplerian import RVPoint, rv_model


def _synthetic_orbit_rvs(
    mjds: np.ndarray,
    *,
    p_days: float = 100.0,
    e: float = 0.2,
    k1: float = 15.0,
    k2: float = 10.0,
    g1: float = -5.0,
    g2: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    t_ref = float(np.median(mjds))
    omega = 0.4
    h = e * math.cos(omega)
    k = e * math.sin(omega)
    m0 = 0.1
    u = orbital_unit_amplitude(math.log(p_days), h, k, m0, mjds, t_ref)
    return g1 + k1 * u, g2 + k2 * u


def test_joint_fit_recovers_mass_ratio(tmp_path: Path):
    mjds = np.linspace(60000.0, 60250.0, 16)
    rv1_true, rv2_true = _synthetic_orbit_rvs(mjds, p_days=100.0, k1=20.0, k2=10.0)
    epochs = []
    for i, mjd in enumerate(mjds):
        epochs.append(
            Sb2EpochRow(
                basename=f"epoch_{i}.txt",
                mjd=float(mjd),
                rv1_kms=float(rv1_true[i]),
                rv1_err_kms=0.3,
                rv2_kms=float(rv2_true[i]),
                rv2_err_kms=0.3,
                summary_rv_kms=float(rv1_true[i]),
                summary_rv_err_kms=0.3,
                sb2_candidate=True,
            )
        )
    masses = Sb2TemplateMasses(5500, 4500, 4.5, 4.5, 1.0, 2.0)
    joint = fit_joint_sb2_all_variants(epochs, masses, gaia_nss=None, mass_sigma_frac=2.0)
    assert "free" in joint
    rep = joint["free"][1]
    assert rep["converged"]
    k_ratio = rep["K1_kms"] / rep["K2_kms"]
    assert 1.4 < k_ratio < 2.6


def test_predict_joint_rvs_matches_rv_model():
    mjds = np.array([60000.0, 60100.0, 60200.0])
    rv1, rv2 = _synthetic_orbit_rvs(mjds)
    epochs = [
        Sb2EpochRow(
            basename=f"e{i}.txt",
            mjd=float(m),
            rv1_kms=float(rv1[i]),
            rv1_err_kms=0.5,
            rv2_kms=float(rv2[i]),
            rv2_err_kms=0.5,
            summary_rv_kms=float(rv1[i]),
            summary_rv_err_kms=0.5,
            sb2_candidate=True,
        )
        for i, m in enumerate(mjds)
    ]
    masses = Sb2TemplateMasses(5500, 4500, 4.5, 4.5, 1.0, 0.5)
    joint = fit_joint_sb2_all_variants(epochs, masses, gaia_nss=None, mass_sigma_frac=0.5)
    params, rep = joint["free"]
    rv1_p, rv2_p = predict_joint_rvs(rep, params, mjds)
    assert np.allclose(rv1_p, rv1, atol=2.0)
    assert np.allclose(rv2_p, rv2, atol=2.0)


def test_independent_fit_report_schema(tmp_path: Path):
    summary = tmp_path / "summary.txt"
    summary.write_text(
        "\n".join(
            [
                "[GAIA METADATA]",
                "Period: 1036.46",
                "Eccentricity: 0.276",
                "[PIPELINE RESULTS]",
                "Gaia_DR3_test_epoch_1.txt 60000.0 -5.0 0.5 10.0",
                "Gaia_DR3_test_epoch_2.txt 60100.0 -3.0 0.5 10.0",
                "Gaia_DR3_test_epoch_3.txt 60200.0 -1.0 0.5 10.0",
            ]
        ),
        encoding="utf-8",
    )
    mjds = np.linspace(60000, 60200, 3)
    rv1, _ = _synthetic_orbit_rvs(mjds)
    epochs = [
        Sb2EpochRow(
            basename=f"Gaia_DR3_test_epoch_{i+1}.txt",
            mjd=float(m),
            rv1_kms=float(rv1[i]),
            rv1_err_kms=0.5,
            rv2_kms=float("nan"),
            rv2_err_kms=float("nan"),
            summary_rv_kms=float(rv1[i]),
            summary_rv_err_kms=0.5,
            sb2_candidate=False,
        )
        for i, m in enumerate(mjds)
    ]
    _, reports = fit_independent_sb2(epochs, summary)
    assert "star1" in reports
    free = reports["star1"].get("free")
    assert free is not None
    for key in ("P_days", "K_kms", "e", "gamma_kms", "chi2_red", "t_ref_mjd"):
        assert key in free


def test_load_template_masses_from_json(tmp_path: Path):
    p = tmp_path / "sb2_fit.json"
    p.write_text(
        json.dumps(
            {
                "params": {
                    "teff1": 5500,
                    "teff2": 4200,
                    "logg1": 4.5,
                    "logg2": 4.0,
                    "feh": 0.0,
                    "vsini1": 2.0,
                    "vsini2": 2.0,
                }
            }
        ),
        encoding="utf-8",
    )
    m = load_template_masses(p)
    assert m.m1_prior_msun > 0
    assert m.m2_prior_msun > 0


def test_build_rv_points_star1_uses_summary_fallback():
    rows = [
        Sb2EpochRow(
            basename="a.txt",
            mjd=60000.0,
            rv1_kms=float("nan"),
            rv1_err_kms=float("nan"),
            rv2_kms=float("nan"),
            rv2_err_kms=float("nan"),
            summary_rv_kms=-12.0,
            summary_rv_err_kms=0.4,
            sb2_candidate=False,
        )
    ]
    pts = build_rv_points_star1(rows)
    assert len(pts) == 1
    assert pts[0].rv == pytest.approx(-12.0)


def test_build_rv_points_star1_non_candidate_prefers_summary_over_rv1():
    rows = [
        Sb2EpochRow(
            basename="b.txt",
            mjd=60100.0,
            rv1_kms=-5.0,
            rv1_err_kms=0.2,
            rv2_kms=float("nan"),
            rv2_err_kms=float("nan"),
            summary_rv_kms=-3.0,
            summary_rv_err_kms=0.5,
            sb2_candidate=False,
        )
    ]
    pts = build_rv_points_star1(rows)
    assert len(pts) == 1
    assert pts[0].rv == pytest.approx(-3.0)
    assert pts[0].rv_err == pytest.approx(0.5)


def test_build_rv_points_star1_candidate_uses_rv1():
    rows = [
        Sb2EpochRow(
            basename="c.txt",
            mjd=60200.0,
            rv1_kms=-7.5,
            rv1_err_kms=0.3,
            rv2_kms=-20.0,
            rv2_err_kms=0.4,
            summary_rv_kms=-6.0,
            summary_rv_err_kms=0.5,
            sb2_candidate=True,
        )
    ]
    pts = build_rv_points_star1(rows)
    assert len(pts) == 1
    assert pts[0].rv == pytest.approx(-7.5)
    assert pts[0].rv_err == pytest.approx(0.3)


def test_joint_fit_orbit_priors_are_applied():
    mjds = np.linspace(60000.0, 60400.0, 5)
    rv1_true, rv2_true = _synthetic_orbit_rvs(mjds, p_days=100.0, e=0.25, k1=18.0, k2=10.0, g1=-6.0)
    epochs = []
    for i, mjd in enumerate(mjds):
        epochs.append(
            Sb2EpochRow(
                basename=f"prior_{i}.txt",
                mjd=float(mjd),
                rv1_kms=float(rv1_true[i]),
                rv1_err_kms=8.0,
                rv2_kms=float(rv2_true[i]),
                rv2_err_kms=8.0,
                summary_rv_kms=float(rv1_true[i]),
                summary_rv_err_kms=8.0,
                sb2_candidate=True,
            )
        )
    masses = Sb2TemplateMasses(5500, 4500, 4.5, 4.5, 1.0, 2.0)
    priors = OrbitPriors(
        period_days=GaussianPrior(1370.6, 1.5),
        eccentricity=GaussianPrior(0.197, 0.013),
        gamma1_kms=GaussianPrior(-4.50, 0.12),
        k1_kms=GaussianPrior(14.01, 0.16),
        omega_deg=GaussianPrior(263.0, 3.0),
    )
    joint = fit_joint_sb2_all_variants(epochs, masses, gaia_nss=None, mass_sigma_frac=1.0, orbit_priors=priors)
    rep = joint["free"][1]
    assert abs(rep["P_days"] - 1370.6) < 10.0
    assert abs(rep["e"] - 0.197) < 0.08
    assert abs(rep["gamma1_kms"] - (-4.5)) < 0.8
    assert abs(rep["K1_kms"] - 14.01) < 1.0
