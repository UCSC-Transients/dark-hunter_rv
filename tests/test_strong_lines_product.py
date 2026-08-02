"""Tests for strong-line product list, inclusion gates, and IVW debias (#91–#93)."""

from pathlib import Path

import numpy as np
import pandas as pd

from darkhunter_rv import config, rv_core
from darkhunter_rv.strong_lines import (
    StrongLineInclusionConfig,
    combine_strong_line_rvs,
    estimate_strong_line_offsets_and_qualities,
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
    assert names.index("MgIb2") < names.index("Hgamma")


def test_strong_line_rests_for_teff_matches_product():
    hot = rv_core.strong_line_rests_for_teff(6500.0)
    cool = rv_core.strong_line_rests_for_teff(4500.0)
    assert hot == cool == product_strong_line_rests()


def test_line_uses_broad_profile():
    assert line_uses_broad_profile("Hbeta", hot_spectrum=False) is True
    assert line_uses_broad_profile("MgIb2", hot_spectrum=True) is False


def test_inclusion_rejects_shallow_low_snr_narrow():
    cfg = StrongLineInclusionConfig(min_depth=0.05, min_snr=8.0, min_width_kms=3.0)
    ok, reason = strong_line_passes_inclusion(
        {"depth": 0.01, "snr": 20.0, "width_kms": 10.0, "err_kms": 2.0, "rv_kms": 5.0, "telluric_frac": 0.0},
        cfg,
    )
    assert not ok and reason == "shallow_depth"
    ok_snr, reason_snr = strong_line_passes_inclusion(
        {"depth": 0.4, "snr": 2.0, "width_kms": 12.0, "err_kms": 2.0, "rv_kms": 5.0, "telluric_frac": 0.0},
        cfg,
    )
    assert not ok_snr and reason_snr == "low_snr"
    ok4, _ = strong_line_passes_inclusion(
        {"depth": 0.4, "snr": 20.0, "width_kms": 12.0, "err_kms": 2.0, "rv_kms": 5.0, "telluric_frac": 0.0},
        cfg,
    )
    assert ok4


def test_local_snr_near_line_uses_flux_eflux_and_varies_with_level():
    from darkhunter_rv.strong_lines import local_snr_near_line

    rest = 5183.6
    w = np.linspace(rest - 30, rest + 30, 400)
    # Bright continuum → high S/N; faint → low (same relative noise fraction would cancel —
    # here eflux fixed so S/N tracks flux level / color / sensitivity).
    f_bright = np.full_like(w, 1000.0)
    f_faint = np.full_like(w, 100.0)
    e = np.full_like(w, 10.0)
    snr_b = local_snr_near_line(w, f_bright, e, rest)
    snr_f = local_snr_near_line(w, f_faint, e, rest)
    assert snr_b == 100.0
    assert snr_f == 10.0
    assert snr_b > snr_f


def test_fit_metrics_prefer_flux_snr_near_line():
    rest = 5183.6
    w = np.linspace(rest - 30, rest + 30, 400)
    f_norm = np.ones_like(w) - 0.35 * np.exp(-0.5 * ((w - rest) / 0.9) ** 2)
    f_raw = np.full_like(w, 500.0)
    e_raw = np.full_like(w, 10.0)
    m = strong_line_fit_metrics(
        wave=w,
        flux_norm=f_norm,
        rest=rest,
        rv_kms=1.0,
        err_kms=2.0,
        bundle={"hb_joint_fit_params": [0.3, 0.1, rest, 0.4, 0.2, 0.5, 1.0, 0.0]},
        flux=f_raw,
        eflux=e_raw,
        wave_native=w,
    )
    assert m["depth"] > 0.2
    assert abs(m["snr"] - 50.0) < 1e-6
    assert abs(m["snr_near_line"] - 50.0) < 1e-6


def test_quality_estimated_only_after_debias():
    """Biased lines: raw medians differ; quality uses debiased residuals."""
    rows = []
    rng = np.random.default_rng(1)
    # Line A: bias +5, tight after debias; Line B: bias -3, loose after debias
    for _ in range(80):
        mask = float(rng.normal(0.0, 0.2))
        rows.append({"line": "A", "rv_kms": mask + 5.0 + rng.normal(0.0, 0.3), "mask_rv_kms": mask})
        rows.append({"line": "B", "rv_kms": mask - 3.0 + rng.normal(0.0, 2.0), "mask_rv_kms": mask})
    offs, quals = estimate_strong_line_offsets_and_qualities(rows)
    assert abs(offs["A"] - 5.0) < 0.2
    assert abs(offs["B"] + 3.0) < 0.6
    assert quals["A"] > quals["B"]
    assert abs(quals["A"] - 1.0) < 1e-6  # best MAD → quality 1


def test_combine_debias_then_weight_by_quality_times_snr2():
    out = combine_strong_line_rvs(
        [
            {"line": "Hbeta", "rv_kms": 11.0, "snr": 2.0, "included": True},
            {"line": "MgIb2", "rv_kms": 12.0, "snr": 2.0, "included": True},
        ],
        offsets={"Hbeta": 1.0, "MgIb2": 2.0},
        qualities={"Hbeta": 1.0, "MgIb2": 1.0},
    )
    # debiased both 10; equal Q and snr → 10
    assert abs(out["rv_kms"] - 10.0) < 1e-6


def test_combine_separates_quality_from_snr():
    # Same S/N: higher Q wins
    out_q = combine_strong_line_rvs(
        [
            {"line": "CaI4227", "rv_kms": 0.0, "snr": 5.0, "included": True},
            {"line": "CaI6122", "rv_kms": 10.0, "snr": 5.0, "included": True},
        ],
        offsets={},
        qualities={"CaI4227": 0.118, "CaI6122": 1.0},
    )
    assert out_q["rv_kms"] > 8.0
    # Same Q: higher S/N wins
    out_s = combine_strong_line_rvs(
        [
            {"line": "A", "rv_kms": 0.0, "snr": 1.0, "included": True},
            {"line": "B", "rv_kms": 10.0, "snr": 4.0, "included": True},
        ],
        offsets={},
        qualities={"A": 1.0, "B": 1.0},
    )
    # weights 1 and 16 → RV = 160/17 ≈ 9.41
    assert out_s["rv_kms"] > 9.0


def test_combine_skips_excluded_and_empty():
    out = combine_strong_line_rvs(
        [{"line": "Hbeta", "rv_kms": 1.0, "snr": 5.0, "included": False}],
        {},
    )
    assert out["n_lines"] == 0


def test_read_strong_line_calibration_quality():
    path = Path(config.REPO_ROOT) / "calibration" / "strong_line_offsets.txt"
    offs, quals = read_strong_line_calibration(path)
    assert offs["Hbeta"] == 0.494
    assert quals["CaI6122"] == 1.0
    assert quals["CaI4227"] < quals["MgIb2"] < quals["CaI6122"]


def test_campaign_qualities_match_debiased_mad_estimator():
    csv_path = (
        Path(config.REPO_ROOT)
        / "validation_output"
        / "strong_line_candidate_sweep"
        / "per_line_fits.csv"
    )
    if not csv_path.is_file():
        import pytest

        pytest.skip("candidate sweep CSV not present")
    per = pd.read_csv(csv_path)
    keep = ["Hbeta", "MgIb2", "MgIb3", "CaI6122", "CaI6162", "CaI4227"]
    rows = []
    for _, r in per.iterrows():
        if r.line not in keep or not bool(r.detected) or not bool(r.mask_valid):
            continue
        rows.append(
            {"line": str(r.line), "rv_kms": float(r.rv_kms), "mask_rv_kms": float(r.mask_rv_kms)}
        )
    offs, quals = estimate_strong_line_offsets_and_qualities(rows)
    file_offs, file_quals = read_strong_line_calibration(
        Path(config.REPO_ROOT) / "calibration" / "strong_line_offsets.txt"
    )
    for line in keep:
        assert abs(offs[line] - file_offs[line]) < 0.05
        assert abs(quals[line] - file_quals[line]) < 0.05
