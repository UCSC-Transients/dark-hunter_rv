"""Tests for spectrum-as-mask epoch CCF helpers."""
from __future__ import annotations

import numpy as np

from darkhunter_rv.epoch_mask_ccf import (
    ChunkNorm,
    pair_spectrum_mask_ccf,
    spectrum_mask_feature_count,
    spectrum_mask_from_norm,
)


def test_spectrum_mask_from_norm_strength():
    wave = np.linspace(5000.0, 5010.0, 101)
    flux = np.ones_like(wave)
    flux[40:45] = 0.5
    mw, ms = spectrum_mask_from_norm(wave, flux)
    assert mw.shape == wave.shape
    assert float(ms[42]) == 0.5
    assert float(ms[0]) == 0.0


def test_spectrum_mask_feature_count_finds_cores():
    s = np.zeros(100)
    s[20] = 0.3
    s[50] = 0.4
    s[51] = 0.2  # adjacent — thinned
    s[80] = 0.1
    n = spectrum_mask_feature_count(s, min_depth=0.05, min_sep_pix=5)
    assert n == 3


def test_spectrum_mask_smooth_differs_from_raw():
    rng = np.random.default_rng(0)
    wave = np.linspace(5000.0, 5020.0, 401)
    flux = np.ones_like(wave)
    flux[150:160] = 0.4
    flux = flux + 0.05 * rng.normal(size=flux.size)
    _, ms_raw = spectrum_mask_from_norm(wave, flux, smooth_sigma=None)
    _, ms_sm = spectrum_mask_from_norm(wave, flux, smooth_sigma=3.0)
    assert not np.allclose(ms_raw, ms_sm)


def _synth_chunks(rv_kms: float, *, noise: float = 0.01, seed: int = 1) -> list[ChunkNorm]:
    """Two chunks with a common absorption line Doppler-shifted by ``rv_kms``."""
    rng = np.random.default_rng(seed)
    c = 299792.458
    rest = 5500.0
    out: list[ChunkNorm] = []
    for si, w0 in enumerate((5480.0, 5520.0)):
        wave = np.linspace(w0, w0 + 30.0, 300)
        # line at rest Doppler-shifted
        lam = rest * (1.0 + rv_kms / c)
        flux = 1.0 - 0.6 * np.exp(-0.5 * ((wave - lam) / 0.15) ** 2)
        flux = flux + noise * rng.normal(size=flux.size)
        eflux = np.full_like(flux, noise)
        out.append(
            ChunkNorm(
                chunk_key=f"10_{si}",
                order=10,
                wave=wave,
                flux_norm=flux,
                eflux_norm=eflux,
            )
        )
    return out


def test_pair_auto_corr_smoothed_near_zero():
    chunks = _synth_chunks(12.0, noise=0.02, seed=2)
    # Empty bias → no b0 shift
    res = pair_spectrum_mask_ccf(
        chunks,
        chunks,
        epoch_x=1,
        epoch_y=1,
        bias={},
        auto_smooth_sigma=3.0,
        max_chunk_err_kms=200.0,
        min_chunks_for_stack=1,
        qc_thresholds={
            "min_mask_line_count": 1,
            "min_ccf_peak_snr": 0.5,
            "min_ccf_peak": 0.0,
            "max_telluric_fraction": 1.0,
            "max_ccf_asymmetry": 1.0,
        },
    )
    assert res.auto_correlation
    assert res.n_chunks >= 1
    assert np.isfinite(res.dv_kms)
    assert abs(res.dv_kms) < 5.0  # near zero relative; lag sampling ~few km/s


def test_pair_swap_opposite_sign():
    a = _synth_chunks(0.0, noise=0.01, seed=3)
    b = _synth_chunks(25.0, noise=0.01, seed=4)
    qc_loose = {
        "min_mask_line_count": 1,
        "min_ccf_peak_snr": 0.5,
        "min_ccf_peak": 0.0,
        "max_telluric_fraction": 1.0,
        "max_ccf_asymmetry": 1.0,
    }
    ab = pair_spectrum_mask_ccf(
        a,
        b,
        epoch_x=1,
        epoch_y=2,
        bias={},
        max_chunk_err_kms=200.0,
        min_chunks_for_stack=1,
        qc_thresholds=qc_loose,
    )
    ba = pair_spectrum_mask_ccf(
        b,
        a,
        epoch_x=2,
        epoch_y=1,
        bias={},
        max_chunk_err_kms=200.0,
        min_chunks_for_stack=1,
        qc_thresholds=qc_loose,
    )
    assert ab.n_chunks >= 1 and ba.n_chunks >= 1
    assert np.isfinite(ab.dv_kms) and np.isfinite(ba.dv_kms)
    # Independent recomputation should be opposite within a few lag samples
    assert abs(ab.dv_kms + ba.dv_kms) < 8.0
    assert ab.dv_kms * ba.dv_kms < 0
