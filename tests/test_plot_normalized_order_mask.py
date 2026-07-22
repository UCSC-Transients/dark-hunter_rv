"""Per-order norm plot can overlay stellar mask shifted by order RV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_rv.plotting import (
    _mask_transmission_shifted_full_depth,
    plot_normalized_order,
)


def test_mask_transmission_full_depth_reaches_zero() -> None:
    w = np.linspace(5000.0, 5020.0, 2000)
    mw = np.array([5010.0])
    ms = np.array([1.0])
    trans = _mask_transmission_shifted_full_depth(w, mw, ms, rv_mask_kms=0.0)
    assert trans is not None
    assert float(np.min(trans)) == pytest.approx(0.0, abs=0.02)
    assert float(np.min(trans)) < 0.2


def test_plot_normalized_order_with_shifted_mask(tmp_path: Path) -> None:
    w = np.linspace(5000.0, 5020.0, 400)
    flux = np.full_like(w, 80.0)
    flux[180:220] = 40.0
    mw = np.array([5005.0, 5010.0, 5015.0])
    ms = np.array([1.0, 0.8, 0.6])
    out = tmp_path / "norm_mask.png"
    plot_normalized_order(
        w,
        flux,
        None,
        out,
        title="test",
        mask_wave=mw,
        mask_strength=ms,
        rv_mask_kms=10.0,
    )
    assert out.is_file()
    assert out.stat().st_size > 1000
