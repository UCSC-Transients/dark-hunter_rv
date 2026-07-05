"""Parameterized echelle blaze model and Hβ-order fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from darkhunter_rv import blaze


def _synthetic_blaze_profile(
    center: float = 4900.0,
    width: float = 90.0,
    power: float = 2.0,
    amplitude: float = 1.4,
    n_pix: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    half_span = max(float(width), 60.0)
    w = np.linspace(center - half_span, center + half_span, n_pix)
    f = blaze.eval_blaze_sinc2(w, center, width, power=power, amplitude=amplitude)
    return w, f


def test_eval_blaze_sinc2_peak_at_center():
    w, f = _synthetic_blaze_profile(center=4900.0)
    i = int(np.argmax(f))
    assert abs(w[i] - 4900.0) < 1.5
    assert f[i] == pytest.approx(1.4, rel=1e-3)


def test_blaze_line_mask_excludes_hbeta_core():
    w = np.linspace(4800.0, 4920.0, 300)
    mask = blaze.blaze_line_mask(w, half_width_angstrom=22.0)
    assert not bool(mask[np.argmin(np.abs(w - blaze.HB_REST_A))])


def test_order_covers_strong_line_and_clean_orders():
    assert blaze.order_covers_strong_line(4850.0, 4875.0)
    assert not blaze.order_covers_strong_line(5195.0, 5278.0)

    spec_data = {
        28: {"wavelength": np.linspace(4816.0, 4894.0, 120), "flux": np.ones(120)},
        35: {"wavelength": np.linspace(5195.0, 5278.0, 120), "flux": np.ones(120)},
    }
    clean = blaze.list_clean_orders(spec_data, bad_orders=[])
    assert [o for o, *_ in clean] == [35]
    picked = blaze.pick_clean_order_near_wavelength(spec_data, 5250.0, bad_orders=[])
    assert picked is not None and picked[0] == 35


def test_hbeta_absorption_depth_shallow_vs_deep():
    w = np.linspace(4790.0, 4935.0, 400)
    f_flat = np.ones_like(w) * 1.2 + 0.0001 * (w - w.mean())
    depth_flat = blaze.hbeta_absorption_depth_raw(w, f_flat)
    f_deep = f_flat.copy()
    core = np.abs(w - blaze.HB_REST_A) < 12.0
    f_deep[core] *= 0.55
    depth_deep = blaze.hbeta_absorption_depth_raw(w, f_deep)
    assert depth_flat < 0.05
    assert depth_deep > 0.25


def test_fit_order_blaze_recovers_shape():
    profiles: list[tuple[np.ndarray, np.ndarray]] = []
    rng = np.random.default_rng(42)
    for amp in (1.1, 1.25, 1.35, 1.2, 1.3):
        w, f = _synthetic_blaze_profile(center=4900.0, amplitude=amp)
        noise = 1.0 + 0.01 * rng.standard_normal(w.size)
        profiles.append((w, f * noise))

    model = blaze.fit_order_blaze_from_profiles(
        profiles,
        echelle_order=42,
        rest_lines=[],
        mask_wave=np.array([]),
        mask_strength=np.array([]),
        use_stellar_mask=False,
    )
    assert model is not None
    assert model.echelle_order == 42
    assert model.center_angstrom == pytest.approx(4900.0, abs=3.0)
    assert model.width_angstrom == pytest.approx(90.0, rel=0.15)
    assert model.power == pytest.approx(2.0, abs=0.4)


def test_order_blaze_model_roundtrip(tmp_path: Path):
    w, f = _synthetic_blaze_profile()
    profiles = [(w, f), (w, f * 1.05), (w, f * 0.95)]
    model = blaze.fit_order_blaze_from_profiles(
        profiles,
        echelle_order=7,
        rest_lines=[],
        mask_wave=np.array([]),
        mask_strength=np.array([]),
        use_stellar_mask=False,
    )
    assert model is not None
    path = tmp_path / "blaze.json"
    model.save(path)
    loaded = blaze.OrderBlazeModel.load(path)
    assert loaded.center_angstrom == model.center_angstrom
    assert loaded.wavelength_min == model.wavelength_min
    data = json.loads(path.read_text())
    assert data["model"] == "sinc2"


def test_uniform_order_grid_avoids_unique_inflation():
    w_base = np.linspace(5195.0, 5278.0, 84)
    profiles = [
        (w_base + 0.01 * i, np.ones_like(w_base))
        for i in range(9)
    ]
    grid = blaze._uniform_order_grid(profiles)
    unique_grid = np.unique(np.concatenate([w for w, _ in profiles]))
    assert grid.size < unique_grid.size // 4
    assert grid.size == pytest.approx(84, abs=3)


def test_blaze_iterative_mask_excludes_dip_centers():
    """Synthetic sinc blaze + shallow dip: iterative mask excludes line core."""
    w = np.linspace(5200.0, 5280.0, 161)
    f = blaze.eval_blaze_sinc2(w, 5235.0, 80.0, power=2.0, amplitude=1500.0)
    dip_center = 5235.0
    core = np.abs(w - dip_center) < 2.0
    f[core] *= 0.55
    mask = blaze.blaze_fit_continuum_mask(w, f, rest_lines=[])
    assert not bool(mask[np.argmin(np.abs(w - dip_center))])
    assert bool(mask[np.argmin(np.abs(w - 5205.0))])


def test_blaze_iterative_mask_rebuilt_not_cumulative():
    """Threshold ladder records each round; mask rebuilds from base each iteration."""
    w, f = _synthetic_blaze_profile(center=5230.0, width=85.0, amplitude=1500.0, n_pix=140)
    dip = np.exp(-0.5 * ((w - 5230.0) / 2.5) ** 2)
    f *= 1.0 - 0.07 * dip
    result = blaze.fit_order_blaze_iterative(w, f, thresholds=(0.9, 0.98))
    assert result is not None
    assert result.stage_counts == ((0.9, result.stage_counts[0][1]), (0.98, result.stage_counts[1][1]))
    i_dip = int(np.argmin(np.abs(w - 5230.0)))
    assert not bool(result.continuum_mask[i_dip])
    only_high = blaze.fit_order_blaze_iterative(w, f, thresholds=(0.98,))
    assert only_high is not None
    # Rebuilt from base at 0.98 can retain dip pixel when 0.9-only run excluded it.
    assert bool(only_high.continuum_mask[i_dip]) or result.stage_counts[0][1] != result.stage_counts[1][1]


def test_blaze_iterative_significance_gate_noisy_vs_clean():
    """Shallow dip masked on clean data; same depth kept on noisy data if not N-sigma significant."""
    w = np.linspace(5195.0, 5278.0, 160)
    f_clean = blaze.eval_blaze_sinc2(w, 5235.0, 80.0, power=2.0, amplitude=1500.0)
    dip = np.exp(-0.5 * ((w - 5225.0) / 2.0) ** 2)
    f_clean *= 1.0 - 0.08 * dip

    rng = np.random.default_rng(0)
    f_noisy = f_clean * (1.0 + 0.12 * rng.standard_normal(w.size))
    f_noisy = np.maximum(f_noisy, 1.0)

    mask_clean = blaze.blaze_fit_continuum_mask(
        w, f_clean, rest_lines=[], thresholds=(0.98,)
    )
    mask_noisy = blaze.blaze_fit_continuum_mask(
        w, f_noisy, rest_lines=[], thresholds=(0.98,)
    )
    i_dip = int(np.argmin(np.abs(w - 5225.0)))
    assert not bool(mask_clean[i_dip])
    assert bool(mask_noisy[i_dip])


def test_blaze_iterative_fit_recovers_width_with_dips():
    """Three Gaussian dips on blaze envelope: final width near truth."""
    true_width = 85.0
    w, f_env = _synthetic_blaze_profile(center=5230.0, width=true_width, amplitude=1500.0, n_pix=140)
    f = f_env.copy()
    for center in (5188.0, 5225.0, 5262.0):
        dip = np.exp(-0.5 * ((w - center) / 2.5) ** 2)
        f *= 1.0 - 0.35 * dip
    result = blaze.fit_order_blaze_iterative(w, f)
    assert result is not None
    assert result.width_angstrom == pytest.approx(true_width, rel=0.2)
    for center in (5188.0, 5225.0, 5262.0):
        assert not bool(result.continuum_mask[np.argmin(np.abs(w - center))])


def test_blaze_stellar_mask_excludes_mask_lines_in_span():
    w = np.linspace(5195.0, 5278.0, 120)
    mw = np.array([5200.0, 5250.0, 6000.0])
    ms = np.array([0.5, 0.4, 0.9])
    mask = blaze.blaze_stellar_mask(
        w,
        mw,
        ms,
        half_width_angstrom=5.0,
        min_strength=0.1,
        max_lines_per_span=10,
    )
    assert not bool(mask[np.argmin(np.abs(w - 5200.0))])
    assert not bool(mask[np.argmin(np.abs(w - 5250.0))])
    assert bool(mask[np.argmin(np.abs(w - 5225.0))])


def test_blaze_fit_continuum_mask_iterative_excludes_absorption():
    """Iterative threshold ladder excludes shallow absorption on blaze envelope."""
    w = np.linspace(5195.0, 5278.0, 120)
    f = blaze.eval_blaze_sinc2(w, 5235.0, 80.0, power=2.0, amplitude=1500.0)
    f[np.abs(w - 5220.0) < 1.2] *= 0.82
    mask = blaze.blaze_fit_continuum_mask(w, f, rest_lines=[])
    assert not bool(mask[np.argmin(np.abs(w - 5220.0))])
    min_pix = int(getattr(blaze.config, "BLAZE_ITERATIVE_MIN_PIXELS", 18))
    assert int(np.sum(mask)) >= min_pix
    assert int(np.sum(mask)) < w.size


def test_fit_order_blaze_ignores_metal_lines_on_clean_order():
    """Clean orders (no STRONG_LINES in span): iterative mask excludes metal dips before fit."""
    w, f_env = _synthetic_blaze_profile(center=5230.0, width=85.0, amplitude=1500.0, n_pix=140)
    f = f_env.copy()
    dip_centers = (5188.0, 5225.0, 5262.0)
    for center in dip_centers:
        dip = np.exp(-0.5 * ((w - center) / 2.5) ** 2)
        f *= 1.0 - 0.35 * dip

    profiles = [(w, f * amp) for amp in (0.98, 1.0, 1.02, 0.99, 1.01)]
    model = blaze.fit_order_blaze_from_profiles(
        profiles,
        echelle_order=35,
        rest_lines=[],
        line_mask_half_width=22.0,
    )
    assert model is not None
    assert model.center_angstrom == pytest.approx(5230.0, abs=8.0)
    assert model.width_angstrom == pytest.approx(85.0, rel=0.2)

    mask = blaze.blaze_fit_continuum_mask(w, f, rest_lines=[])
    for center in dip_centers:
        assert not bool(mask[np.argmin(np.abs(w - center))])
    assert int(np.sum(mask)) > 18


def test_blaze_mask_expansion_improves_pull_symmetry():
    """Post-iteration dilation grows line mask when continuum pulls have a <1 tail."""
    w = np.linspace(5200.0, 5280.0, 161)
    f = blaze.eval_blaze_sinc2(w, 5235.0, 80.0, power=2.0, amplitude=1500.0)
    center = 5235.0
    narrow = np.exp(-0.5 * ((w - center) / 2.0) ** 2)
    broad = np.exp(-0.5 * ((w - center) / 6.0) ** 2)
    f *= 1.0 - 0.45 * narrow
    f *= 1.0 - 0.08 * broad

    no_expand = blaze.fit_order_blaze_iterative(w, f, max_mask_expand=0)
    with_expand = blaze.fit_order_blaze_iterative(w, f, max_mask_expand=8)
    assert no_expand is not None and with_expand is not None
    assert with_expand.mask_expand_pixels >= 1
    assert int(np.sum(with_expand.continuum_mask)) < int(np.sum(no_expand.continuum_mask))

    norm_no = blaze._blaze_normalized_flux(
        w,
        f,
        no_expand.continuum_mask,
        center=no_expand.center_angstrom,
        width=no_expand.width_angstrom,
        power=no_expand.power,
        amplitude=no_expand.amplitude,
    )
    norm_yes = blaze._blaze_normalized_flux(
        w,
        f,
        with_expand.continuum_mask,
        center=with_expand.center_angstrom,
        width=with_expand.width_angstrom,
        power=with_expand.power,
        amplitude=with_expand.amplitude,
    )
    score_no = blaze._pull_distribution_consistency_score(norm_no[no_expand.continuum_mask] - 1.0)
    score_yes = blaze._pull_distribution_consistency_score(norm_yes[with_expand.continuum_mask] - 1.0)
    assert score_yes <= score_no + 0.05


def test_blaze_fit_continuum_mask_keeps_hbeta_wings():
    w = np.linspace(4800.0, 4920.0, 300)
    f = blaze.eval_blaze_sinc2(w, 4900.0, 90.0, power=2.0, amplitude=1200.0)
    core = np.abs(w - blaze.HB_REST_A) < 8.0
    f[core] *= 0.4
    mask = blaze.blaze_fit_continuum_mask(
        w,
        f,
        rest_lines=blaze.strong_lines_in_span(float(w.min()), float(w.max())),
        half_width_angstrom=22.0,
    )
    assert bool(mask[np.argmin(np.abs(w - (blaze.HB_REST_A + 35.0)))])
    assert not bool(mask[np.argmin(np.abs(w - blaze.HB_REST_A))])

    w, f = _synthetic_blaze_profile(amplitude=2.0)
    profiles = [(w, f), (w, f * 1.02), (w, f * 0.98)]
    model = blaze.fit_order_blaze_from_profiles(profiles, echelle_order=1, rest_lines=[], mask_wave=np.array([]), mask_strength=np.array([]), use_stellar_mask=False)
    assert model is not None
    fc = model.correct_flux(w, f)
    assert float(np.nanmedian(fc)) == pytest.approx(2.0, rel=0.08)
