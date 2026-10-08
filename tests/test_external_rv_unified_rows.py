"""Tests for LAMOST/RAVE external RV normalization in gaia_utils."""

import pytest

from darkhunter_rv.gaia_utils import (
    _external_rvs_from_unified_rows,
    lamost_lmjm_to_mjd,
    lamost_obsdate_to_mjd,
)

RA, DEC = 71.96109727, 44.90186273


def _mrs(rv, err, coadd, lmjm, band_dupes=2, obsid="851214080"):
    row = {
        "ext_cat": "LAMOST_MRS",
        "obs_str": "2020-11-04",
        "rv_z": rv,
        "err_z": err,
        "flag_raw": "0",
        "ext_id": obsid,
        "ext_coadd": coadd,
        "ext_lmjm": lmjm,
    }
    return [dict(row) for _ in range(band_dupes)]


def _mrs_night_rows():
    """Gaia DR3 204912477375300224, LAMOST MRS obsid 851214080 (B and R rows each)."""
    return (
        _mrs(-11.21, 1.07, 1, 0)
        + _mrs(14.25, 1.22, 0, 85187608)
        + _mrs(3.95, 1.22, 0, 85187631)
        + _mrs(12.09, 1.18, 0, 85187657)
    )


def test_lamost_times_are_beijing_local() -> None:
    assert lamost_lmjm_to_mjd(85187608) == pytest.approx(59157.72778, abs=1e-5)
    assert lamost_lmjm_to_mjd(0) is None
    assert lamost_obsdate_to_mjd("2020-11-04") == pytest.approx(59157.0 + 16.0 / 24.0)


def test_lamost_mrs_keeps_one_coadd_per_night_at_exposure_time() -> None:
    rows = _external_rvs_from_unified_rows(_mrs_night_rows(), ra_deg=RA, dec_deg=DEC)
    assert len(rows) == 1
    assert rows[0]["telescope"] == "LAMOST_MRS"
    assert rows[0]["rv"] == pytest.approx(-11.21)
    expected = sum(lamost_lmjm_to_mjd(m) for m in (85187608, 85187631, 85187657)) / 3
    assert rows[0]["mjd"] == pytest.approx(expected)


def test_lamost_mrs_without_coadd_uses_single_exposures_once() -> None:
    raw = [r for r in _mrs_night_rows() if r["ext_coadd"] == 0]
    rows = _external_rvs_from_unified_rows(raw, ra_deg=RA, dec_deg=DEC)
    assert [r["rv"] for r in rows] == pytest.approx([14.25, 3.95, 12.09])
    assert rows[0]["mjd"] == pytest.approx(lamost_lmjm_to_mjd(85187608))


def test_lamost_rows_without_ids_drop_exact_duplicates() -> None:
    raw = [
        {"ext_cat": "LAMOST_MRS", "obs_str": "2020-11-04", "rv_z": 14.25, "err_z": 1.22, "flag_raw": "0"},
        {"ext_cat": "LAMOST_MRS", "obs_str": "2020-11-04", "rv_z": 14.25, "err_z": 1.22, "flag_raw": "0"},
        {"ext_cat": "LAMOST_MRS", "obs_str": "2020-11-04", "rv_z": -11.21, "err_z": 1.07, "flag_raw": "0"},
    ]
    rows = _external_rvs_from_unified_rows(raw, ra_deg=RA, dec_deg=DEC)
    assert sorted(r["rv"] for r in rows) == pytest.approx([-11.21, 14.25])
    assert all(r["mjd"] == pytest.approx(59157.0 + 16.0 / 24.0) for r in rows)


def test_rave_row_gets_mjd_from_obs_id() -> None:
    rows = _external_rvs_from_unified_rows(
        [
            {
                "ext_cat": "RAVE_DR6",
                "obs_str": "",
                "rv_z": 12.5,
                "err_z": 1.0,
                "flag_raw": "101021_0941_2",
            }
        ],
        ra_deg=180.0,
        dec_deg=-30.0,
    )
    assert len(rows) == 1
    assert rows[0]["telescope"] == "RAVE_DR6"
    assert rows[0]["mjd"] > 40000
    assert "conv=helio→bary" in rows[0]["flag"]
