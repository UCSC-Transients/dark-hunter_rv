"""Smoke test for adopted-RV match diagnostic plot (#41 / step 04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from darkhunter_rv.pipeline import _panels_for_adopted_rv_match
from darkhunter_rv.plotting import plot_adopted_rv_match


def test_plot_adopted_rv_match_writes_png(tmp_path: Path) -> None:
    w = np.linspace(4850.0, 4880.0, 600)
    flux = np.ones_like(w)
    flux[280:320] = 0.55
    mw = np.array([4861.3, 4865.0])
    ms = np.array([0.5, 0.3])
    out = tmp_path / "stem_adopted_rv_match.png"
    plot_adopted_rv_match(
        [("order35", w, flux)],
        out,
        adopted_rv_kms=12.5,
        mask_wave=mw,
        mask_strength=ms,
        strong_lines=[("Hbeta", 4861.3)],
        title="stem",
        rv_source_label="cascade adopted (debiased)",
    )
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_panels_for_adopted_rv_match_prefers_strong_line_orders() -> None:
    preps = [
        {
            "chunk_key": "10",
            "nw": np.linspace(5000.0, 5100.0, 100),
            "nf": np.ones(100),
        },
        {
            "chunk_key": "35",
            "nw": np.linspace(4840.0, 4900.0, 100),
            "nf": np.ones(100),
        },
        {
            "chunk_key": "35_1",
            "nw": np.linspace(4840.0, 4870.0, 50),
            "nf": np.ones(50),
        },
        {
            "chunk_key": "40",
            "nw": np.linspace(5200.0, 5300.0, 100),
            "nf": np.ones(100),
        },
    ]
    panels = _panels_for_adopted_rv_match(preps, [("Hbeta", 4861.3)], max_panels=2)
    assert len(panels) == 2
    keys = [p[0] for p in panels]
    assert "35" in keys
