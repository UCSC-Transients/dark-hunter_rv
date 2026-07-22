"""Parallax coalesce: nss_two_body_orbit → nss_acceleration_astro → gaia_source."""

import math

import pytest

from darkhunter_rv import gaia_utils

STAR = 77413727493690112
ACC_PLX = 11.409495987268548
ACC_ERR = 0.031981006
SRC_PLX = 15.04137660
SRC_ERR = 0.61232984


def test_coalesce_uses_acceleration_when_two_body_missing():
    row = {
        "final_parallax": ACC_PLX,
        "final_parallax_error": ACC_ERR,
        "plx_source": SRC_PLX,
        "plx_err_source": SRC_ERR,
    }
    plx, err = gaia_utils._coalesce_parallax_from_row(row)
    assert plx == pytest.approx(ACC_PLX)
    assert err == pytest.approx(ACC_ERR)


def test_coalesce_python_fallback_acceleration_fields():
    row = {
        "acc_parallax": ACC_PLX,
        "acc_parallax_error": ACC_ERR,
        "plx_source": SRC_PLX,
        "plx_err_source": SRC_ERR,
    }
    plx, err = gaia_utils._coalesce_parallax_from_row(row)
    assert plx == pytest.approx(ACC_PLX)
    assert err == pytest.approx(ACC_ERR)


def test_coalesce_falls_back_to_gaia_source():
    row = {"plx_source": SRC_PLX, "plx_err_source": SRC_ERR}
    plx, err = gaia_utils._coalesce_parallax_from_row(row)
    assert plx == pytest.approx(SRC_PLX)
    assert err == pytest.approx(SRC_ERR)


def test_coalesce_skips_nss_parallax_without_error():
    """NSS π with null σ must not stick; fall through to gaia_source pair."""
    row = {
        "parallax": 12.0,
        "parallax_error": float("nan"),
        "plx_source": SRC_PLX,
        "plx_err_source": SRC_ERR,
    }
    plx, err = gaia_utils._coalesce_parallax_from_row(row)
    assert plx == pytest.approx(SRC_PLX)
    assert err == pytest.approx(SRC_ERR)


def test_coalesce_skips_masked_nss_without_warning():
    import numpy as np
    import warnings

    row = {
        "parallax": np.ma.masked,
        "parallax_error": np.ma.masked,
        "plx_source": SRC_PLX,
        "plx_err_source": SRC_ERR,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plx, err = gaia_utils._coalesce_parallax_from_row(row)
    assert plx == pytest.approx(SRC_PLX)
    assert err == pytest.approx(SRC_ERR)
    assert not any("masked element" in str(w.message) for w in caught)


def test_process_query_results_keeps_large_gaia_source_error():
    meta = gaia_utils.process_query_results(
        [{"plx_source": SRC_PLX, "plx_err_source": SRC_ERR, "ra_source": 1.0, "dec_source": 2.0, "source_id": STAR}],
        [],
    )["metadata"]
    assert meta["Parallax"] == pytest.approx(SRC_PLX)
    assert meta["Parallax_Error"] == pytest.approx(SRC_ERR)


def test_process_query_results_includes_flame_fields():
    meta = gaia_utils.process_query_results(
        [
            {
                "plx_source": SRC_PLX,
                "plx_err_source": SRC_ERR,
                "ra_source": 1.0,
                "dec_source": 2.0,
                "source_id": STAR,
                "mass_flame": 1.05,
                "age_flame": 4.2,
                "flags_flame": "123",
            }
        ],
        [],
    )["metadata"]
    assert meta["Mass_FLAME"] == pytest.approx(1.05)
    assert meta["Age_FLAME"] == pytest.approx(4.2)
    assert meta["Flags_FLAME"] == "123"


def test_process_query_results_omits_empty_flags_flame():
    meta = gaia_utils.process_query_results(
        [
            {
                "plx_source": SRC_PLX,
                "plx_err_source": SRC_ERR,
                "ra_source": 1.0,
                "dec_source": 2.0,
                "source_id": STAR,
                "flags_flame": None,
            }
        ],
        [],
    )["metadata"]
    assert "Flags_FLAME" not in meta
    assert math.isnan(meta["Mass_FLAME"])


@pytest.mark.integration
def test_live_gaia_query_acceleration_parallax_for_77413727493690112():
    """Requires network; values from nss_acceleration_astro for this SB candidate."""
    data = gaia_utils.query_gaia_data(STAR)
    assert data is not None
    meta = data["metadata"]
    assert meta["Parallax"] == pytest.approx(ACC_PLX, rel=0, abs=1e-6)
    assert meta["Parallax_Error"] == pytest.approx(ACC_ERR, rel=0, abs=1e-6)
