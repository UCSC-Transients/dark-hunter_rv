"""Tests for strong-line product list, inclusion gates, and IVW debias (#91–#93)."""

from pathlib import Path

import numpy as np

from darkhunter_rv import config, rv_core
from darkhunter_rv.strong_lines import (
    StrongLineInclusionConfig,
    combine_strong_line_rvs,
    line_uses_broad_profile,
    product_strong_line_rests,
    read_strong_line_calibration,
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
        qualities={"Hbeta": 1.0, "MgIb2": 1.0},
        depth_weight_power=0.0,
    )
    # debiased 10 and 10 → IVW 10
    assert out["n_lines"] == 2
    assert abs(out["rv_kms"] - 10.0) < 1e-6
    assert out["err_kms"] < 1.0


def test_combine_uses_species_quality_not_only_snr():
    """Equal formal errors: higher Q_line must dominate the stack."""
    out = combine_strong_line_rvs(
        [
            {"line": "CaI4227", "rv_kms": 0.0, "err_kms": 1.0, "included": True},
            {"line": "CaI6122", "rv_kms": 10.0, "err_kms": 1.0, "included": True},
        ],
        offsets={},
        qualities={"CaI4227": 0.118, "CaI6122": 1.0},
        depth_weight_power=0.0,
    )
    # weights 0.118 and 1.0 → RV ≈ 10*1/(1.118) ≈ 8.94
    assert out["n_lines"] == 2
    assert out["rv_kms"] > 8.0
    w_by = {d["line"]: d["weight"] for d in out["details"]}
    assert w_by["CaI6122"] > 5.0 * w_by["CaI4227"]


def test_combine_skips_excluded_and_empty():
    out = combine_strong_line_rvs(
        [{"line": "Hbeta", "rv_kms": 1.0, "err_kms": 1.0, "included": False}],
        {},
    )
    assert out["n_lines"] == 0
    assert not np.isfinite(out["rv_kms"])


def test_read_strong_line_calibration_quality():
    path = Path(config.REPO_ROOT) / "calibration" / "strong_line_offsets.txt"
    offs, quals = read_strong_line_calibration(path)
    assert offs["Hbeta"] == 0.494
    assert quals["CaI6122"] == 1.0
    assert quals["CaI4227"] < quals["MgIb2"] < quals["CaI6122"]
    assert read_strong_line_offsets(path)["CaI6122"] == 1.274


def test_inclusion_approx_pass_rate_on_candidate_sweep_csv():
    """
    Empirically check depth+err inclusion rates on the 114-stem candidate sweep.

    Full width/S/N gates need continuum arrays; this validates the depth/err subset
    that drove the keep/exclude decisions.
    """
    csv_path = (
        Path(config.REPO_ROOT)
        / "validation_output"
        / "strong_line_candidate_sweep"
        / "per_line_fits.csv"
    )
    if not csv_path.is_file():
        import pytest

        pytest.skip("candidate sweep CSV not present")
    import pandas as pd

    per = pd.read_csv(csv_path)
    cfg = StrongLineInclusionConfig()
    # Approximate: depth + err only (width/snr unavailable in CSV) — expect high pass for keep lines.
    keep = ["Hbeta", "MgIb2", "MgIb3", "CaI6122", "CaI6162"]
    for line in keep:
        g = per[per.line == line]
        assert len(g) >= 50
        n_ok = 0
        for _, row in g.iterrows():
            metrics = {
                "depth": float(row["depth"]) if pd.notna(row["depth"]) else float("nan"),
                "snr": 10.0,  # assume continuum S/N OK when depth recorded
                "width_kms": 12.0,
                "err_kms": float(row["err_kms"]) if pd.notna(row["err_kms"]) else float("nan"),
                "rv_kms": float(row["rv_kms"]) if pd.notna(row["rv_kms"]) else float("nan"),
                "telluric_frac": float(row["telluric_frac"])
                if pd.notna(row.get("telluric_frac"))
                else 0.0,
            }
            ok, _ = strong_line_passes_inclusion(metrics, cfg)
            n_ok += int(ok)
        rate = n_ok / len(g)
        assert rate >= 0.75, f"{line} inclusion rate {rate:.2f} too low"
