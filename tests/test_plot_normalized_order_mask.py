"""Per-order norm plot: SED continuum scale + native-strength mask at continuum=1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from darkhunter_rv.continuum import (
    build_fixed_cont_mask,
    fit_continuum,
    load_sed_continuum_spans,
    load_sed_region_spans,
)
from darkhunter_rv.plotting import (
    _mask_transmission_shifted_native,
    plot_normalized_order,
)


def test_mask_transmission_native_strength_continuum_one() -> None:
    w = np.linspace(5000.0, 5020.0, 2000)
    mw = np.array([5010.0])
    ms = np.array([0.4])
    trans = _mask_transmission_shifted_native(w, mw, ms, rv_mask_kms=0.0)
    assert trans is not None
    assert float(np.median(trans)) == pytest.approx(1.0, abs=0.02)
    assert float(np.min(trans)) == pytest.approx(0.6, abs=0.05)


def test_continuum_none_sed_span_median_scale() -> None:
    w = np.linspace(5000.0, 5100.0, 201)
    flux = np.full(201, 50.0)
    # Plateau at 100 between 5050–5070 Å
    flux[100:141] = 100.0
    flux[110:116] = 40.0  # absorption in continuum region
    eflux = np.full(201, 2.0)
    spans = [(5050.0, 5070.0)]
    _, nf, ne = fit_continuum(w, flux, eflux, continuum_mode="none", continuum_spans=spans)
    cont_pix = (w >= 5050.0) & (w <= 5070.0) & (flux > 50.0)
    assert np.median(nf[cont_pix]) == pytest.approx(1.0, abs=0.05)
    assert np.min(nf[110:116]) < 0.55
    assert ne[0] == pytest.approx(2.0 / 100.0, rel=0.01)


def test_fixed_cont_mask_excludes_line_spans() -> None:
    w = np.linspace(5000.0, 5100.0, 201)
    flux = np.full(201, 100.0)
    cont = [(5050.0, 5070.0)]
    lines = [(5055.0, 5060.0)]
    m = build_fixed_cont_mask(w, flux, continuum_spans=cont, line_spans=lines, edge_pixels=0)
    assert m[(w >= 5050.0) & (w <= 5054.0)].all()
    assert not m[(w >= 5055.0) & (w <= 5060.0)].any()
    # Median scale ignores line-span pixels
    flux[(w >= 5055.0) & (w <= 5060.0)] = 10.0
    _, nf, _ = fit_continuum(
        w, flux, np.ones_like(flux), continuum_mode="none", continuum_spans=cont, line_spans=lines
    )
    assert np.median(nf[m]) == pytest.approx(1.0, abs=0.05)


def test_load_sed_continuum_spans(tmp_path: Path) -> None:
    doc = {
        "orders": {
            "10": {
                "continuum_regions": [[5000.0, 5001.0], [5010.0, 5012.0]],
                "line_regions": [[5005.0, 5006.0]],
            },
            "11": {"continuum_regions": [[5200.0, 5205.0]]},
        }
    }
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(doc))
    spans = load_sed_continuum_spans(path)
    assert len(spans) == 3
    assert spans[0] == (5000.0, 5001.0)
    cont, lines = load_sed_region_spans(path)
    assert len(cont) == 3
    assert len(lines) == 1
    assert lines[0] == (5005.0, 5006.0)


def test_plot_normalized_order_with_shifted_mask(tmp_path: Path) -> None:
    w = np.linspace(5000.0, 5020.0, 400)
    flux = np.ones_like(w)
    flux[180:220] = 0.5
    mw = np.array([5005.0, 5010.0, 5015.0])
    ms = np.array([0.5, 0.4, 0.3])
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
