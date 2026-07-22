"""Per-order norm plot can overlay stellar mask shifted by order RV."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from darkhunter_rv.plotting import plot_normalized_order


def test_plot_normalized_order_with_shifted_mask(tmp_path: Path) -> None:
    w = np.linspace(5000.0, 5020.0, 400)
    flux = np.ones_like(w)
    flux[180:220] = 0.7
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
