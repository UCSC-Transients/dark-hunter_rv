"""Joker plot helpers emit PNG files from fake posterior arrays."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from darkhunter_rv.joker_rv_fit import params_from_keplerian
from darkhunter_rv.rv_keplerian_plots import plot_joker_corners, plot_joker_multi_fit
from fit_apf_rv_keplerian import RVPoint


def test_joker_fit_and_corner_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 30
    p = 80.0 + rng.normal(0, 0.5, n)
    k = 15.0 + rng.normal(0, 0.2, n)
    e = np.clip(0.1 + rng.normal(0, 0.01, n), 0.0, 0.9)
    om = rng.normal(0.5, 0.05, n)
    m0 = rng.normal(0.2, 0.05, n)
    g = 10.0 + rng.normal(0, 0.1, n)
    med = params_from_keplerian(80.0, 15.0, 0.1, 0.5, 0.2, 10.0)
    samples = [params_from_keplerian(p[i], k[i], e[i], om[i], m0[i], g[i]) for i in range(10)]
    points = [
        RVPoint(file="a", mjd=60000.0 + i * 10.0, rv=10.0, rv_err=0.2, rms=0.2, telescope="APF")
        for i in range(5)
    ]
    report = {"t_ref_mjd": 60020.0, "now_mjd": 60100.0, "fit_variants": {"rv_only": {"P_days": 80.0, "e": 0.1}}}
    fit_png = tmp_path / "fit.png"
    corner_png = tmp_path / "corner.png"
    summary = tmp_path / "Gaia_DR3_1_summary.txt"
    summary.write_text("Gaia Source ID: 1\n")
    plot_joker_multi_fit(
        summary,
        points,
        {"rv_only": samples},
        {"rv_only": med},
        report,
        fit_png,
    )
    plot_joker_corners(
        {
            "rv_only": {
                "P_days": p,
                "K_kms": k,
                "e": e,
                "omega_deg": np.degrees(om),
                "gamma_kms": g,
            }
        },
        corner_png,
        gaia_id="1",
    )
    assert fit_png.is_file()
    assert corner_png.is_file()
    assert fit_png.stat().st_size > 1000
    assert corner_png.stat().st_size > 1000
