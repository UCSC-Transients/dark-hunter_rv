"""Tests for literature RV cross-check CLI join logic (step 08 lite)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from validation.compare_literature_rvs import (
    build_arg_parser,
    nearest_literature_join,
    parse_gaia_id_from_path,
    per_star_bias_rms,
)
from validation.rv_overlap_lib import bjd_to_mjd, load_literature_epochs

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase_a"


def test_parse_gaia_id_from_path() -> None:
    p = Path("Gaia_DR3_1028887114002082432_epoch_1_diagnostics.csv")
    assert parse_gaia_id_from_path(p) == "1028887114002082432"
    assert parse_gaia_id_from_path("nope.csv") is None


def test_cli_parse_methods_defaults() -> None:
    ap = build_arg_parser()
    args = ap.parse_args(["--diagnostics-glob", "x/*_diagnostics.csv"])
    assert args.methods == "mask_ccf,template_fft"
    assert args.report_dir.name == "literature_crosscheck_lite"


def test_nearest_literature_join_picks_closest() -> None:
    lit = load_literature_epochs(_FIXTURES / "literature_mini.csv")
    # Literature for STAR_A at BJD 2459900 and 2459903.
    # Pipeline at BJD 2459902.5 should match 2459903 (Δ=1.0 d vs Δ=2.5 d).
    pipe = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1000000000000000001",
                "basename": "Gaia_DR3_1000000000000000001_epoch_1",
                "method": "mask_ccf",
                "mjd": bjd_to_mjd(2459902.5),
                "bjd": 2459902.5,
                "rv_kms": 10.4,
                "rv_err_kms": 0.05,
                "diagnostics_path": "",
            }
        ]
    )
    pairs = nearest_literature_join(pipe, lit)
    assert len(pairs) == 1
    assert pairs.iloc[0]["literature_bjd"] == pytest.approx(2459903.0)
    assert pairs.iloc[0]["delta_days"] == pytest.approx(0.5)
    assert pairs.iloc[0]["delta_rv_kms"] == pytest.approx(10.4 - 10.5)


def test_nearest_literature_join_max_delta_days() -> None:
    lit = load_literature_epochs(_FIXTURES / "literature_mini.csv")
    pipe = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1000000000000000001",
                "basename": "e1",
                "method": "template_fft",
                "mjd": bjd_to_mjd(2459950.0),
                "bjd": 2459950.0,
                "rv_kms": 11.0,
                "rv_err_kms": 0.1,
                "diagnostics_path": "",
            }
        ]
    )
    assert nearest_literature_join(pipe, lit, max_delta_days=10.0).empty
    kept = nearest_literature_join(pipe, lit, max_delta_days=100.0)
    assert len(kept) == 1


def test_per_star_bias_rms() -> None:
    pairs = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "name": "A",
                "method": "mask_ccf",
                "delta_rv_kms": 1.0,
                "delta_days": 10.0,
                "P_orb_days": 100.0,
                "M_star_msun": 1.0,
                "M2_msun": 1.2,
            },
            {
                "gaia_dr3_id": "1",
                "name": "A",
                "method": "mask_ccf",
                "delta_rv_kms": -1.0,
                "delta_days": 20.0,
                "P_orb_days": 100.0,
                "M_star_msun": 1.0,
                "M2_msun": 1.2,
            },
        ]
    )
    tab = per_star_bias_rms(pairs)
    assert len(tab) == 1
    assert tab.iloc[0]["bias_kms"] == pytest.approx(0.0)
    assert tab.iloc[0]["rms_kms"] == pytest.approx(1.0)
    assert tab.iloc[0]["n_epochs"] == 2
