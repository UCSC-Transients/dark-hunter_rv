"""Tests for strong-line product list, inclusion gates, and IVW debias (#91–#93)."""

from pathlib import Path

import numpy as np

from darkhunter_rv import config, rv_core
from darkhunter_rv.strong_lines import (
    StrongLineInclusionConfig,
    combine_strong_line_rvs,
    line_uses_broad_profile,
    product_strong_line_rests,
    read_strong_line_offsets,
    strong_line_fit_metrics,
    strong_line_passes_inclusion,
)


def test_product_list_wires_keep_metals():
    rests = product_strong_line_rests()
    names = [n for n, _ in rests]
    assert names[0] == "Hbeta"
    assert "MgIb2" in names
    assert "CaI6122" in names
    assert "CaI6162" in names
    assert "MgIb3" in names
    assert "CaI4227" in names
    assert "CaIIK" not in names
    assert "CaII8662" not in names
    assert names.index("MgIb2") < names.index("Hgamma")


def test_strong_line_rests_for_teff_matches_product():
    hot = rv_core.strong_line_rests_for_teff(6500.0)
    cool = rv_core.strong_line_rests_for_teff(4500.0)
    assert hot == cool == product_strong_line_rests()
    assert hot[0][0] == "Hbeta"
    assert cool[-1][0] == "Halpha"


def test_line_uses_broad_profile():
    assert line_uses_broad_profile("Hbeta", hot_spectrum=False) is True
    assert line_uses_broad_profile("Hgamma", hot_spectrum=True) is True
    assert line_uses_broad_profile("Hgamma", hot_spectrum=False) is False
    assert line_uses_broad_profile("MgIb2", hot_spectrum=True) is False


def test_inclusion_rejects_shallow_low_snr_narrow():
    cfg = StrongLineInclusionConfig(min_depth=0.05, min_snr=3.0, min_width_kms=3.0)
    ok, reason = strong_line_passes_inclusion(
        {
            "depth": 0.01,
            "snr": 10.0,
            "width_kms": 10.0,
            "err_kms": 2.0,
            "rv_kms": 5.0,
            "telluric_frac": 0.0,
        },
        cfg,
    )
    assert not ok and reason == "shallow_depth"
    ok2, reason2 = strong_line_passes_inclusion(
        {
            "depth": 0.4,
            "snr": 1.0,
            "width_kms": 10.0,
            "err_kms": 2.0,
            "rv_kms": 5.0,
            "telluric_frac": 0.0,
        },
        cfg,
    )
    assert not ok2 and reason2 == "low_snr"
    ok3, reason3 = strong_line_passes_inclusion(
        {
            "depth": 0.4,
            "snr": 10.0,
            "width_kms": 0.5,
            "err_kms": 2.0,
            "rv_kms": 5.0,
            "telluric_frac": 0.0,
        },
        cfg,
    )
    assert not ok3 and reason3 == "too_narrow"
    ok4, _ = strong_line_passes_inclusion(
        {
            "depth": 0.4,
            "snr": 10.0,
            "width_kms": 12.0,
            "err_kms": 2.0,
            "rv_kms": 5.0,
            "telluric_frac": 0.0,
        },
        cfg,
    )
    assert ok4


def test_fit_metrics_depth_and_snr():
    rest = 5183.6
    w = np.linspace(rest - 30, rest + 30, 400)
    f = np.ones_like(w) - 0.35 * np.exp(-0.5 * ((w - rest) / 0.9) ** 2)
    f += np.random.default_rng(0).normal(0.0, 0.01, size=f.shape)
    m = strong_line_fit_metrics(
        wave=w,
        flux_norm=f,
        rest=rest,
        rv_kms=1.0,
        err_kms=2.0,
        bundle={"hb_joint_fit_params": [0.3, 0.1, rest, 0.4, 0.2, 0.5, 1.0, 0.0]},
    )
    assert m["depth"] > 0.2
    assert m["snr"] > 3.0
    assert m["width_kms"] > 1.0


def test_combine_strong_line_rvs_debias_and_ivw():
    offsets = {"Hbeta": 1.0, "MgIb2": 2.0}
    out = combine_strong_line_rvs(
        [
            {"line": "Hbeta", "rv_kms": 11.0, "err_kms": 1.0, "depth": 1.0, "included": True},
            {"line": "MgIb2", "rv_kms": 12.0, "err_kms": 1.0, "depth": 1.0, "included": True},
        ],
        offsets,
        depth_weight_power=0.0,
    )
    # debiased 10 and 10 → IVW 10
    assert out["n_lines"] == 2
    assert abs(out["rv_kms"] - 10.0) < 1e-6
    assert out["err_kms"] < 1.0


def test_combine_skips_excluded_and_empty():
    out = combine_strong_line_rvs(
        [{"line": "Hbeta", "rv_kms": 1.0, "err_kms": 1.0, "included": False}],
        {},
    )
    assert out["n_lines"] == 0
    assert not np.isfinite(out["rv_kms"])


def test_read_strong_line_offsets_file():
    path = Path(config.REPO_ROOT) / "calibration" / "strong_line_offsets.txt"
    offs = read_strong_line_offsets(path)
    assert offs["Hbeta"] == 0.494
    assert offs["CaI6122"] == 1.274
    assert read_strong_line_offsets(None) == {}
    assert read_strong_line_offsets(Path("/no/such/file.txt")) == {}
