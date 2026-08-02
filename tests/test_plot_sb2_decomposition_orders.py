"""Unit tests for SB2 decomposition order plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from validation.plot_sb2_decomposition_orders import (
    PlotScope,
    build_order_series,
    compute_decomposition_series,
    default_output_path,
    run_plot_job,
    resolve_plot_scope,
)


def test_resolve_plot_scope_default():
    scope = resolve_plot_scope(
        epoch_basename="Gaia_DR3_1_epoch_5.txt",
        epoch_index=None,
        order=None,
        all_epochs=False,
    )
    assert scope == PlotScope.ONE_EPOCH_ALL_ORDERS


def test_resolve_plot_scope_single_order():
    scope = resolve_plot_scope(
        epoch_basename="Gaia_DR3_1_epoch_5.txt",
        epoch_index=None,
        order=35,
        all_epochs=False,
    )
    assert scope == PlotScope.ONE_EPOCH_ONE_ORDER


def test_resolve_plot_scope_all_epochs_one_order():
    scope = resolve_plot_scope(
        epoch_basename=None,
        epoch_index=None,
        order=35,
        all_epochs=True,
    )
    assert scope == PlotScope.ALL_EPOCHS_ONE_ORDER


def test_resolve_plot_scope_errors():
    with pytest.raises(ValueError, match="--all-epochs requires --order"):
        resolve_plot_scope(epoch_basename=None, epoch_index=None, order=None, all_epochs=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_plot_scope(
            epoch_basename="x.txt",
            epoch_index=None,
            order=35,
            all_epochs=True,
        )
    with pytest.raises(ValueError, match="requires --epoch-basename"):
        resolve_plot_scope(epoch_basename=None, epoch_index=None, order=35, all_epochs=False)
    with pytest.raises(ValueError, match="specify --epoch-basename"):
        resolve_plot_scope(epoch_basename=None, epoch_index=None, order=None, all_epochs=False)


def test_default_output_path_naming():
    sb2_dir = Path("validation_output/sb2_1")
    assert default_output_path(
        sb2_dir,
        PlotScope.ONE_EPOCH_ALL_ORDERS,
        epoch_basename="Gaia_DR3_1_epoch_5.txt",
        order=None,
    ) == sb2_dir / "plots" / "Gaia_DR3_1_epoch_5_sb2_decomp_orders.pdf"
    assert default_output_path(
        sb2_dir,
        PlotScope.ONE_EPOCH_ONE_ORDER,
        epoch_basename="Gaia_DR3_1_epoch_5.txt",
        order=35,
    ) == sb2_dir / "plots" / "Gaia_DR3_1_epoch_5_order35_sb2_decomp.pdf"
    assert default_output_path(
        sb2_dir,
        PlotScope.ALL_EPOCHS_ONE_ORDER,
        epoch_basename=None,
        order=35,
    ) == sb2_dir / "plots" / "order35_all_epochs_sb2_decomp.pdf"


def test_compute_decomposition_series():
    data = np.array([1.0, 1.1, 0.9])
    s1 = np.array([0.6, 0.65, 0.55])
    s2 = np.array([0.4, 0.42, 0.38])
    total, dms1, resid = compute_decomposition_series(data, s1, s2)
    np.testing.assert_allclose(total, s1 + s2)
    np.testing.assert_allclose(dms1, data - s1)
    np.testing.assert_allclose(resid, data - total)


def test_build_order_series_predetermined_sum():
    wave = np.linspace(5000.0, 5100.0, 50)
    s1 = 0.35 + 0.01 * np.sin((wave - 5000.0) / 7.0)
    s2 = 0.15 + 0.005 * np.cos((wave - 5000.0) / 9.0)
    data_norm = s1 + s2
    wave_sep = np.linspace(4990.0, 5110.0, 80)
    flux1 = np.interp(wave_sep, wave, s1, left=s1[0], right=s1[-1])
    flux2 = np.interp(wave_sep, wave, s2, left=s2[0], right=s2[-1])
    series = build_order_series(wave, data_norm, wave_sep, flux1, flux2)
    assert series is not None
    np.testing.assert_allclose(series.total_model, series.star1 + series.star2, rtol=0, atol=1e-12)
    assert float(np.nanmedian(series.total_model)) < 1.2
    np.testing.assert_allclose(series.total_model, data_norm, rtol=0, atol=1e-2)


@pytest.mark.slow
def test_smoke_generate_all_decomposition_plot_modes(tmp_path: Path):
    sb2_dir = Path(__file__).resolve().parents[1] / "validation_output" / "sb2_77413727493690112"
    if not (sb2_dir / "sb2_epochs.csv").is_file():
        pytest.skip("sb2_77413727493690112 test data missing")

    run_plot_job(
        sb2_dir=sb2_dir,
        out_path=tmp_path / "one_epoch_all_orders.pdf",
        scope=PlotScope.ONE_EPOCH_ALL_ORDERS,
        epoch_basename="Gaia_DR3_77413727493690112_epoch_5.txt",
        epoch_index=None,
        order=None,
        instrument_name="APF",
        gaia_id="77413727493690112",
        continuum_mode="spline",
    )
    assert (tmp_path / "one_epoch_all_orders.pdf").is_file()

    run_plot_job(
        sb2_dir=sb2_dir,
        out_path=tmp_path / "one_epoch_one_order.pdf",
        scope=PlotScope.ONE_EPOCH_ONE_ORDER,
        epoch_basename="Gaia_DR3_77413727493690112_epoch_5.txt",
        epoch_index=None,
        order=35,
        instrument_name="APF",
        gaia_id="77413727493690112",
        continuum_mode="spline",
    )
    assert (tmp_path / "one_epoch_one_order.pdf").is_file()

    run_plot_job(
        sb2_dir=sb2_dir,
        out_path=tmp_path / "one_order_all_epochs.pdf",
        scope=PlotScope.ALL_EPOCHS_ONE_ORDER,
        epoch_basename=None,
        epoch_index=None,
        order=35,
        instrument_name="APF",
        gaia_id="77413727493690112",
        continuum_mode="spline",
    )
    assert (tmp_path / "one_order_all_epochs.pdf").is_file()
