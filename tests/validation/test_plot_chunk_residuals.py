"""Tests for chunk residual plotting helpers."""
from __future__ import annotations

import numpy as np
import pytest

from validation.plot_chunk_residuals import (
    _chunk_sort_key,
    _gaia_dr3_label,
    _ordered_chunks,
    _summarize_chunks_per_object,
    _weighted_mean_and_errors,
    apply_sample_object_bias_clip,
    apply_spectrum_chunk_outlier_clip,
    build_order_relative_residuals,
    iterative_spectrum_chunk_clip_mask,
    summarize_order_relative_residuals,
)


def test_chunk_sort_key() -> None:
    assert _chunk_sort_key("42") < _chunk_sort_key("43")
    assert _chunk_sort_key("42_1") < _chunk_sort_key("42_2")


def test_ordered_chunks() -> None:
    assert _ordered_chunks(["10", "2", "2_1"]) == ["2", "2_1", "10"]


def test_weighted_mean_and_errors() -> None:
    mu, stat, intrinsic = _weighted_mean_and_errors(
        np.array([0.1, 0.2, 0.0]),
        np.array([0.05, 0.05, 0.05]),
    )
    assert mu == pytest.approx(0.1, abs=0.05)
    assert stat > 0
    assert intrinsic >= 0


def test_weighted_mean_requires_errors() -> None:
    mu, stat, intrinsic = _weighted_mean_and_errors(
        np.array([0.1, 0.2]),
        np.array([np.nan, np.nan]),
    )
    assert not np.isfinite(mu)
    assert not np.isfinite(stat)
    assert not np.isfinite(intrinsic)

    mu2, stat2, _ = _weighted_mean_and_errors(
        np.array([0.1, 0.9]),
        np.array([0.05, np.nan]),
    )
    assert mu2 == pytest.approx(0.1)
    assert stat2 == pytest.approx(0.05)


def test_summarize_skips_chunks_without_errors() -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {"chunk_key": "1", "file": "a", "residual_kms": 0.1, "rv_err_kms": 0.05},
            {"chunk_key": "2", "file": "a", "residual_kms": 5.0, "rv_err_kms": np.nan},
            {"chunk_key": "3", "file": "a", "residual_kms": -0.2, "rv_err_kms": 0.0},
        ]
    )
    s = _summarize_chunks_per_object(df, min_measurements=1)
    assert list(s["chunk_key"].astype(str)) == ["1"]
    assert np.isfinite(float(s.iloc[0]["statistical_err_kms"]))


def test_apply_clip_drops_missing_err_from_kept() -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {"file": "a", "rv_kms": 10.0, "rv_err_kms": 0.05, "exposure_rv_kms": 10.0, "residual_kms": 0.0},
            {"file": "a", "rv_kms": 10.1, "rv_err_kms": 0.05, "exposure_rv_kms": 10.0, "residual_kms": 0.1},
            {"file": "a", "rv_kms": 50.0, "rv_err_kms": np.nan, "exposure_rv_kms": 10.0, "residual_kms": 40.0},
        ]
    )
    out = apply_spectrum_chunk_outlier_clip(df, nsigma=10.0, max_delta_kms=20.0)
    assert not bool(out.loc[2, "chunk_kept"])
    assert bool(out.loc[0, "chunk_kept"]) and bool(out.loc[1, "chunk_kept"])



def test_iterative_clip_removes_sigma_outlier() -> None:
    rv = np.array([10.0, 10.1, 10.05, 50.0])
    err = np.array([0.05, 0.05, 0.05, 0.05])
    keep = iterative_spectrum_chunk_clip_mask(rv, err, nsigma=10.0, max_delta_kms=0.0)
    assert not keep[3]
    assert keep[:3].all()


def test_iterative_clip_removes_far_from_weighted_mean() -> None:
    rv = np.array([10.0, 10.1, 80.0])
    err = np.array([0.05, 0.05, 0.05])
    keep = iterative_spectrum_chunk_clip_mask(rv, err, nsigma=0.0, max_delta_kms=30.0)
    assert not keep[2]
    assert keep[:2].all()


def test_apply_spectrum_chunk_outlier_clip() -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {"file": "a", "rv_kms": 10.0, "rv_err_kms": 0.05, "exposure_rv_kms": 10.0, "residual_kms": 0.0},
            {"file": "a", "rv_kms": 10.1, "rv_err_kms": 0.05, "exposure_rv_kms": 10.0, "residual_kms": 0.1},
            {"file": "a", "rv_kms": 50.0, "rv_err_kms": 0.05, "exposure_rv_kms": 10.0, "residual_kms": 40.0},
        ]
    )
    out = apply_spectrum_chunk_outlier_clip(df, nsigma=10.0)
    assert not bool(out.loc[2, "chunk_kept"])
    kept = out[out["chunk_kept"]]
    assert len(kept) == 2
    assert kept["residual_kms"].abs().max() < 0.2


def test_apply_sample_object_bias_clip() -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "chunk_key": "10",
                "weighted_mean_residual_kms": 0.1,
                "statistical_err_kms": 0.05,
                "intrinsic_scatter_kms": 0.02,
            },
            {
                "gaia_dr3_id": "2",
                "chunk_key": "10",
                "weighted_mean_residual_kms": 0.15,
                "statistical_err_kms": 0.05,
                "intrinsic_scatter_kms": 0.02,
            },
            {
                "gaia_dr3_id": "3",
                "chunk_key": "10",
                "weighted_mean_residual_kms": 25.0,
                "statistical_err_kms": 0.05,
                "intrinsic_scatter_kms": 0.02,
            },
        ]
    )
    out = apply_sample_object_bias_clip(df, nsigma=5.0, max_delta_kms=10.0)
    assert bool(out.loc[out["gaia_dr3_id"] == "3", "sample_kept"].iloc[0]) is False
    assert out.loc[out["gaia_dr3_id"].isin(["1", "2"]), "sample_kept"].all()


def test_summarize_chunks_per_object_min_measurements() -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {"chunk_key": "1", "file": "a", "residual_kms": 0.1, "rv_err_kms": 0.05},
            {"chunk_key": "1", "file": "b", "residual_kms": 0.3, "rv_err_kms": 0.05},
            {"chunk_key": "1", "file": "c", "residual_kms": 0.2, "rv_err_kms": 0.05},
            {"chunk_key": "2", "file": "a", "residual_kms": -0.2, "rv_err_kms": 0.05},
            {"chunk_key": "2", "file": "b", "residual_kms": -0.1, "rv_err_kms": 0.05},
        ]
    )
    s = _summarize_chunks_per_object(df, min_measurements=3)
    assert len(s) == 1
    assert s.iloc[0]["chunk_key"] == "1"
    assert int(s.iloc[0]["n_measurements"]) == 3


def test_equal_weight_demean_avoids_weight_dependent_object_offset() -> None:
    """
    Structured bias + different IVW weights must not imprint object offsets after demean.

    Equal-weight demean keeps residuals ≈ bias - mean(bias) for every object.
    """
    import pandas as pd

    orders = np.arange(10, dtype=float)
    bias = np.where(orders < 5, -1.0, 1.0)
    # Object A: small errors on blue (low orders); B: small errors on red.
    rows = []
    for gid, err_blue, err_red in (("A", 0.05, 0.5), ("B", 0.5, 0.05)):
        for o, b in zip(orders, bias):
            err = err_blue if o < 5 else err_red
            rows.append(
                {
                    "gaia_dr3_id": gid,
                    "file": f"{gid}.fits",
                    "rv_kms": 20.0 + float(b),
                    "rv_err_kms": float(err),
                    "exposure_rv_kms": 20.0,
                    "residual_kms": float(b),
                    "chunk_key": str(int(o)),
                    "chunk_order": int(o),
                    "mjd": 0.0,
                    "teff": 5000.0,
                    "log10_median_mask_ccf_peak_snr": 1.0,
                    "used_in_exposure_stack": True,
                    "qc_pass": True,
                    "diagnostics_path": "",
                }
            )
    tab = pd.DataFrame(rows)
    out = apply_spectrum_chunk_outlier_clip(tab, nsigma=0.0, max_delta_kms=0.0)
    # Per-object equal-weight mean residual must be ~0; curves must match.
    for gid, g in out.groupby("gaia_dr3_id"):
        r = g["residual_kms"].astype(float).values
        e = g["rv_err_kms"].astype(float).values
        mu, _, _ = _weighted_mean_and_errors(r, e)
        # IVW of residuals need not be 0 under equal-weight demean; equal-weight mean must.
        assert float(np.mean(r)) == pytest.approx(0.0, abs=1e-12)
        assert list(np.round(r, 6)) == list(np.round(bias - np.mean(bias), 6))

    # Relative to shared order mean, object IVW offsets stay near zero (same curve).
    summaries = {}
    for gid, g in out.groupby("gaia_dr3_id"):
        summaries[gid] = _summarize_chunks_per_object(g, min_measurements=1)
    bias_rows = []
    for gid, sdf in summaries.items():
        for _, r in sdf.iterrows():
            bias_rows.append({"gaia_dr3_id": gid, "sample_kept": True, **r.to_dict()})
    bias_df = pd.DataFrame(bias_rows)
    rel = build_order_relative_residuals(bias_df)
    summary = summarize_order_relative_residuals(rel)
    for _, row in summary.iterrows():
        assert abs(float(row["ivw_residual_to_order_mean_kms"])) < 0.05
        assert abs(int(row["n_positive"]) - int(row["n_negative"])) <= 1



def test_order_relative_residuals_summary() -> None:
    import pandas as pd

    bias = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "chunk_key": "10",
                "chunk_order": 10,
                "weighted_mean_residual_kms": 1.0,
                "statistical_err_kms": 0.1,
                "intrinsic_scatter_kms": 0.0,
                "sample_kept": True,
            },
            {
                "gaia_dr3_id": "2",
                "chunk_key": "10",
                "chunk_order": 10,
                "weighted_mean_residual_kms": -1.0,
                "statistical_err_kms": 0.1,
                "intrinsic_scatter_kms": 0.0,
                "sample_kept": True,
            },
            {
                "gaia_dr3_id": "1",
                "chunk_key": "20",
                "chunk_order": 20,
                "weighted_mean_residual_kms": 0.5,
                "statistical_err_kms": 0.1,
                "intrinsic_scatter_kms": 0.0,
                "sample_kept": True,
            },
            {
                "gaia_dr3_id": "2",
                "chunk_key": "20",
                "chunk_order": 20,
                "weighted_mean_residual_kms": -0.5,
                "statistical_err_kms": 0.1,
                "intrinsic_scatter_kms": 0.0,
                "sample_kept": True,
            },
        ]
    )
    rel = build_order_relative_residuals(bias)
    assert "residual_to_order_mean_kms" in rel.columns
    # Equal-weight IVW sample mean at each chunk is 0 → deltas equal raw biases.
    g1 = rel[rel["gaia_dr3_id"] == "1"]
    assert float(g1.loc[g1["chunk_key"] == "10", "residual_to_order_mean_kms"].iloc[0]) == pytest.approx(1.0)
    summary = summarize_order_relative_residuals(rel)
    assert set(summary["gaia_dr3_id"].astype(str)) == {"1", "2"}
    row1 = summary[summary["gaia_dr3_id"] == "1"].iloc[0]
    assert int(row1["n_positive"]) == 2
    assert int(row1["n_negative"]) == 0
    assert float(row1["ivw_residual_to_order_mean_kms"]) == pytest.approx(0.75)
    row2 = summary[summary["gaia_dr3_id"] == "2"].iloc[0]
    assert int(row2["n_negative"]) == 2
    assert float(row2["ivw_residual_to_order_mean_kms"]) == pytest.approx(-0.75)
