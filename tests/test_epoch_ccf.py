"""Synthetic tests for epoch–epoch CCF pair API and WLS absolute fill."""
from __future__ import annotations

import numpy as np
import pytest

from darkhunter_rv import config
from darkhunter_rv.epoch_ccf import (
    combine_relative_and_absolute,
    epoch_pair_ccf,
)


def _synthetic_spectrum(
    wave: np.ndarray,
    *,
    rv_kms: float = 0.0,
    line_centers_rest: np.ndarray | None = None,
    depth: float = 0.55,
    width_kms: float = 8.0,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Continuum-normalized flux with Gaussian absorption lines at rest + Doppler."""
    if line_centers_rest is None:
        line_centers_rest = np.array(
            [5050.0, 5080.0, 5110.0, 5135.0, 5160.0, 5185.0],
            dtype=float,
        )
    z = float(rv_kms) / float(config.C_KMS)
    flux = np.ones_like(wave, dtype=float)
    for lam0 in line_centers_rest:
        lam_obs = float(lam0) * (1.0 + z)
        sigma_wave = lam_obs * (width_kms / float(config.C_KMS))
        flux -= depth * np.exp(-0.5 * ((wave - lam_obs) / (sigma_wave + 1e-12)) ** 2)
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        flux = flux + rng.normal(0.0, noise, size=len(flux))
    return flux


def test_epoch_pair_ccf_recovers_injected_doppler():
    wave = np.linspace(5000.0, 5200.0, 5000)
    true_dv = 37.5
    flux_j = _synthetic_spectrum(wave, rv_kms=0.0)
    flux_i = _synthetic_spectrum(wave, rv_kms=true_dv)
    res = epoch_pair_ccf(wave, flux_i, wave, flux_j, rv_search_half_width_kms=200.0)
    assert res.fit_ok
    assert np.isfinite(res.dv_kms)
    assert np.isfinite(res.err_kms) and res.err_kms > 0
    # Recover within formal error (allow 3σ or 1 km/s floor for grid discreteness)
    tol = max(3.0 * res.err_kms, 1.0)
    assert abs(res.dv_kms - true_dv) < tol


def test_epoch_pair_ccf_antisymmetric():
    wave = np.linspace(5000.0, 5200.0, 4000)
    flux_a = _synthetic_spectrum(wave, rv_kms=12.0)
    flux_b = _synthetic_spectrum(wave, rv_kms=-8.0)
    ij = epoch_pair_ccf(wave, flux_a, wave, flux_b, rv_search_half_width_kms=200.0)
    ji = epoch_pair_ccf(wave, flux_b, wave, flux_a, rv_search_half_width_kms=200.0)
    assert ij.fit_ok and ji.fit_ok
    assert abs(ji.dv_kms + ij.dv_kms) < max(0.5, 0.1 * abs(ij.dv_kms))


def test_combine_one_absolute_fills_all():
    # True velocities; one absolute anchor at epoch 0
    v_true = np.array([10.0, 25.0, -5.0, 40.0])
    n = len(v_true)
    dv = np.zeros((n, n))
    sig = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            dv[i, j] = v_true[i] - v_true[j]
    abs_rv = np.full(n, np.nan)
    abs_sig = np.full(n, np.nan)
    abs_rv[0] = v_true[0]
    abs_sig[0] = 0.2
    out = combine_relative_and_absolute(dv, sig, abs_rv, abs_sig)
    assert out.n_abs_anchors == 1
    assert not out.float_zeropoint
    assert not out.relative_only
    assert np.allclose(out.v_hat_kms, v_true, atol=0.05)


def test_combine_zero_absolute_relative_only():
    v_true = np.array([0.0, 15.0, -20.0])  # relatives vs v0=0
    n = len(v_true)
    dv = np.zeros((n, n))
    sig = np.full((n, n), 0.3)
    for i in range(n):
        for j in range(n):
            dv[i, j] = v_true[i] - v_true[j]
    abs_rv = np.full(n, np.nan)
    abs_sig = np.full(n, np.nan)
    out = combine_relative_and_absolute(dv, sig, abs_rv, abs_sig)
    assert out.n_abs_anchors == 0
    assert out.float_zeropoint
    assert out.relative_only
    assert out.v_hat_kms[0] == pytest.approx(0.0)
    # Relative differences preserved
    for i in range(n):
        for j in range(i + 1, n):
            assert (out.v_hat_kms[i] - out.v_hat_kms[j]) == pytest.approx(
                v_true[i] - v_true[j], abs=0.05
            )


def test_auto_correlation_near_zero():
    wave = np.linspace(5000.0, 5200.0, 3000)
    flux = _synthetic_spectrum(wave, rv_kms=5.0)
    res = epoch_pair_ccf(wave, flux, wave, flux, rv_search_half_width_kms=100.0)
    assert abs(res.dv_kms) < 1.0
    assert res.qc.get("auto_correlation") is True
