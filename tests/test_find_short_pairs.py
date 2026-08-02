"""Unit tests for short-pair discovery and sigma inflation recommendation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from darkhunter_rv.epoch_ccf import inflate_sigma_ij
from validation.find_short_pairs import (
    EpochRecord,
    discover_epochs,
    find_pairs,
    recommend_sigma_ij_scale,
    write_report,
)


def _write_diag(path: Path, *, mjd: float, spectrum: str = "spec.txt") -> None:
    pd.DataFrame(
        {
            "file": [spectrum],
            "chunk_key": ["0_a"],
            "mjd": [mjd],
            "method": ["mask_ccf"],
            "rv_kms": [10.0],
            "rv_err_kms": [0.5],
            "qc_pass": [True],
            "exposure_rv_kms": [10.0],
            "exposure_rv_err_kms": [0.5],
            "used_in_exposure_stack": [True],
        }
    ).to_csv(path, index=False)


def test_find_pairs_respects_delta_and_same_night(tmp_path: Path):
    gid = "111"
    recs = [
        EpochRecord(gid, 1, tmp_path / "a.csv", 60000.1, None),
        EpochRecord(gid, 2, tmp_path / "b.csv", 60000.9, None),  # same night, Δt=0.8
        EpochRecord(gid, 3, tmp_path / "c.csv", 60001.2, None),  # next calendar night
        EpochRecord(gid, 4, tmp_path / "d.csv", 60010.0, None),  # far
    ]
    pairs = find_pairs(recs, max_delta_days=1.0, same_calendar_night=False)
    keys = {(a.epoch, b.epoch) for a, b, _ in pairs}
    assert (1, 2) in keys
    assert (2, 3) in keys  # Δt=0.3
    assert (1, 3) not in keys  # Δt=1.1 > 1
    assert (1, 4) not in keys

    same = find_pairs(recs, max_delta_days=1.0, same_calendar_night=True)
    same_keys = {(a.epoch, b.epoch) for a, b, _ in same}
    assert same_keys == {(1, 2)}


def test_discover_epochs_from_glob(tmp_path: Path):
    gid = "2223334445556667778"
    _write_diag(tmp_path / f"Gaia_DR3_{gid}_epoch_1_diagnostics.csv", mjd=60100.0)
    _write_diag(tmp_path / f"Gaia_DR3_{gid}_epoch_2_diagnostics.csv", mjd=60100.5)
    found = discover_epochs(str(tmp_path / "Gaia_DR3_*_diagnostics.csv"), gaia_id=gid)
    assert [r.epoch for r in found] == [1, 2]
    assert found[0].mjd == pytest.approx(60100.0)


def test_recommend_sigma_ij_scale_never_deflates():
    deltas = np.array([0.1, -0.2, 0.15])
    errs = np.array([1.0, 1.0, 1.0])  # formal errs >> scatter
    info = recommend_sigma_ij_scale(deltas, errs)
    assert info["recommended_sigma_ij_scale"] >= 1.0

    # Underestimated formal errors → inflate (MAD path)
    errs2 = np.array([0.01, 0.01, 0.01])
    info2 = recommend_sigma_ij_scale(deltas, errs2)
    assert info2["recommended_sigma_ij_scale"] > 1.0
    # One huge outlier must not dominate MAD recommendation vs RMS
    deltas3 = np.array([0.1, -0.1, 0.05, 300.0])
    errs3 = np.array([1.0, 1.0, 1.0, 1.0])
    info3 = recommend_sigma_ij_scale(deltas3, errs3)
    assert info3["recommended_sigma_ij_scale"] < info3["scale_rms"]


def test_inflate_sigma_ij_scales_off_diagonal_only():
    sig = np.array([[1.0, 2.0], [2.0, 1.5]])
    out = inflate_sigma_ij(sig, 3.0)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 1] == pytest.approx(1.5)
    assert out[0, 1] == pytest.approx(6.0)
    assert out[1, 0] == pytest.approx(6.0)
    assert inflate_sigma_ij(sig, 0.5)[0, 1] == pytest.approx(2.0)  # floor 1.0


def test_write_report_creates_artifacts(tmp_path: Path):
    df = pd.DataFrame(
        {
            "gaia_id": ["1"],
            "epoch_i": [1],
            "epoch_j": [2],
            "delta_t_days": [0.01],
            "drv_mask_ccf_kms": [0.2],
            "drv_template_fft_kms": [np.nan],
            "drv_strong_lines_kms": [np.nan],
            "violate_mask_ccf": [False],
            "violate_template_fft": [False],
            "violate_strong_lines": [False],
            "drv_epoch_ccf_kms": [0.3],
            "drv_epoch_ccf_err_kms": [0.1],
            "violate_epoch_ccf": [False],
            "violate_any": [False],
        }
    )
    info = recommend_sigma_ij_scale(
        df["drv_epoch_ccf_kms"].to_numpy(),
        df["drv_epoch_ccf_err_kms"].to_numpy(),
    )
    paths = write_report(
        df,
        tmp_path / "qc",
        max_delta_days=1.0,
        abs_thresh_kms=5.0,
        n_sigma=3.0,
        sigma_scale_info=info,
        calibration_json=tmp_path / "calibration" / "short_pair_sigma_scale.json",
    )
    assert paths["pairs_csv"].is_file()
    assert paths["report_md"].is_file()
    assert paths["calibration_json"].is_file()
    assert paths["calibration_summary_csv"].is_file()
