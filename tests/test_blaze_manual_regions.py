"""Manual region constraints in blaze continuum masks."""

from __future__ import annotations

import numpy as np

from darkhunter_rv.blaze import (
    FIT_MODEL_SINC_POLY,
    OrderBlazeModel,
    blaze_fit_continuum_mask,
    eval_blaze_sinc2,
    eval_poly_multiplier,
    fit_order_blaze_iterative,
    shared_blaze_order_envelope,
)


def _flat_flux(n: int, rng: np.random.Generator) -> np.ndarray:
    return 100.0 + rng.normal(0, 0.5, size=n)


def test_fixed_line_stays_off_after_iteration() -> None:
    rng = np.random.default_rng(1)
    w = np.linspace(5000.0, 5100.0, 101)
    f = _flat_flux(w.size, rng)
    line = np.zeros(w.size, bool)
    line[50:55] = True
    cont = np.zeros(w.size, bool)
    cont[10:90] = True
    base = cont.copy()
    result = fit_order_blaze_iterative(
        w,
        f,
        initial_mask=base,
        fixed_line_mask=line,
        fixed_cont_mask=cont & ~line,
        thresholds=(0.9, 0.95),
    )
    assert result is not None
    assert not result.continuum_mask[line].any()
    assert result.continuum_mask[20]


def test_fixed_cont_stays_on_after_expansion() -> None:
    rng = np.random.default_rng(2)
    w = np.linspace(5000.0, 5100.0, 101)
    f = _flat_flux(w.size, rng)
    f[40:45] = f[40:45] * 0.5  # absorption pocket
    cont = np.zeros(w.size, bool)
    cont[30:70] = True
    mask = blaze_fit_continuum_mask(
        w,
        f,
        initial_mask=cont,
        fixed_cont_mask=cont,
        thresholds=(0.85, 0.9, 0.95),
    )
    assert mask[35]
    assert mask[50]


def test_cr_excluded_inside_continuum() -> None:
    rng = np.random.default_rng(3)
    w = np.linspace(5000.0, 5100.0, 51)
    f = _flat_flux(w.size, rng)
    f[25] = 1e6
    cont = np.ones(w.size, bool)
    cr = np.ones(w.size, bool)
    cr[25] = False
    mask = blaze_fit_continuum_mask(
        w,
        f,
        initial_mask=cont,
        fixed_cont_mask=cont & cr,
        cr_mask=cr,
    )
    assert not mask[25]
    assert mask[10]


def test_blaze_fit_forwards_initial_mask() -> None:
    rng = np.random.default_rng(4)
    w = np.linspace(5000.0, 5100.0, 101)
    f = 100.0 + rng.normal(0, 0.2, size=w.size)
    narrow = np.zeros(w.size, bool)
    narrow[30:70] = True
    mask = blaze_fit_continuum_mask(
        w,
        f,
        initial_mask=narrow,
        thresholds=(0.98,),
    )
    assert mask[40]
    assert not mask[10]
    assert not mask[90]


def test_shared_blaze_envelope_scale_only() -> None:
    rng = np.random.default_rng(8)
    w = np.linspace(5180.0, 5220.0, 81)
    blaze_model = OrderBlazeModel(
        echelle_order=35,
        model="sinc2",
        center_angstrom=5200.0,
        width_angstrom=120.0,
        power=2.0,
        n_spectra_fit=1,
        wavelength_min=5180.0,
        wavelength_max=5220.0,
    )
    shape = blaze_model.blaze_on_grid(w)
    scale_true = 150.0
    f = scale_true * shape + rng.normal(0, 0.5, size=w.size)
    result = shared_blaze_order_envelope(w, f, blaze_model)
    assert result is not None
    assert abs(result.scale - scale_true) < 3.0
    norm = f / result.envelope
    ok = result.continuum_mask & np.isfinite(norm) & (norm > 0)
    assert abs(float(np.nanmedian(norm[ok])) - 1.0) < 0.06


def test_sinc_poly_shared_shape_per_epoch_poly() -> None:
    rng = np.random.default_rng(5)
    w = np.linspace(5000.0, 5100.0, 101)
    blaze_model = OrderBlazeModel(
        echelle_order=0,
        model="sinc2",
        center_angstrom=5050.0,
        width_angstrom=80.0,
        power=2.0,
        n_spectra_fit=1,
        wavelength_min=5000.0,
        wavelength_max=5100.0,
    )
    sinc = blaze_model.blaze_on_grid(w)
    x = (w - 5050.0) / 25.0
    poly_true = np.exp(0.05 + 0.30 * x + 0.10 * x**2)
    scale = 100.0
    f = scale * sinc * poly_true + rng.normal(0, 0.25, size=w.size)
    base = np.ones(w.size, bool)
    base[48:53] = False
    result = shared_blaze_order_envelope(
        w,
        f,
        blaze_model,
        base_mask=base,
        fit_model=FIT_MODEL_SINC_POLY,
        poly_order=2,
    )
    assert result is not None
    assert result.fit_model == FIT_MODEL_SINC_POLY
    norm = f / result.envelope
    ok = result.continuum_mask & np.isfinite(norm)
    assert abs(float(np.nanmedian(norm[ok])) - 1.0) < 0.12


def test_sinc_poly_fit_skewed_envelope_iterative_tooling() -> None:
    rng = np.random.default_rng(5)
    w = np.linspace(5000.0, 5100.0, 101)
    sinc = eval_blaze_sinc2(w, 5050.0, 80.0, power=2.0, amplitude=100.0)
    x = (w - 5050.0) / 25.0
    poly_true = np.exp(0.05 + 0.30 * x + 0.10 * x**2)
    f = sinc * poly_true + rng.normal(0, 0.25, size=w.size)
    base = np.ones(w.size, bool)
    base[48:53] = False
    result = fit_order_blaze_iterative(
        w,
        f,
        initial_mask=base,
        fit_model=FIT_MODEL_SINC_POLY,
        poly_order=2,
        thresholds=(0.9, 0.95),
    )
    assert result is not None
    assert result.fit_model == FIT_MODEL_SINC_POLY
    from darkhunter_rv.blaze import eval_fit_envelope

    env = eval_fit_envelope(result, w)
    assert np.all(env > 0)
    norm = f / env
    ok = result.continuum_mask & np.isfinite(norm)
    assert abs(float(np.nanmedian(norm[ok])) - 1.0) < 0.12


def test_eval_blaze_sinc2_always_positive() -> None:
    w = np.linspace(5000.0, 5100.0, 201)
    blaze = eval_blaze_sinc2(w, 5050.0, 40.0, power=3.0, amplitude=2.5)
    assert np.all(blaze > 0)


def test_eval_poly_multiplier_always_positive() -> None:
    w = np.linspace(5000.0, 5100.0, 51)
    poly = eval_poly_multiplier(w, (0.1, -0.5, 0.2), 5050.0, 20.0)
    assert np.all(poly > 0)
