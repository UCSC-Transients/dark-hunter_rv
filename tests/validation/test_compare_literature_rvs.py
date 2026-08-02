"""Tests for literature RV cross-check CLI join logic (step 08)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from validation.compare_literature_rvs import (
    build_arg_parser,
    load_external_epochs_from_summaries,
    nearest_literature_join,
    nearest_pipeline_join,
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
    assert args.summaries_glob is None
    assert "LAMOST_LRS" in args.external_sources


def test_nearest_literature_join_picks_closest() -> None:
    lit = load_literature_epochs(_FIXTURES / "literature_mini.csv")
    # Literature for STAR_A at BJD 2459900 and 2459903.
    # Pipeline at BJD 2459902.5 should match 2459903 (Δ=0.5 d vs Δ=2.5 d).
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


def test_load_external_epochs_from_summaries_filters_sources(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_1000000000000000001_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\nSource_ID: 1000000000000000001\nRA: 1.0\nDec: 2.0\n"
        "\n[EXTERNAL RV DATA]\n"
        "LAMOST_LRS 59000.0 -20.0 1.2 z_meas\n"
        "GALAH_AAT 59001.0 -21.0 0.2 frame=bary-native\n"
        "RAVE_DR6 59002.0 -19.5 1.0 conv=helio→bary\n"
        "\n[PIPELINE RESULTS]\n"
        "Gaia_DR3_1000000000000000001_epoch_1.txt 60000.0 -10.0 0.2 0.3 False\n"
    )
    df = load_external_epochs_from_summaries(
        str(tmp_path / "Gaia_DR3_*_summary.txt"),
        sources=("LAMOST_LRS", "RAVE_DR6"),
    )
    assert len(df) == 2
    assert set(df["method"]) == {"LAMOST_LRS", "RAVE_DR6"}
    assert df["gaia_dr3_id"].iloc[0] == "1000000000000000001"


def test_nearest_pipeline_join_external_minus_pipeline() -> None:
    external = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "method": "LAMOST_LRS",
                "mjd": 59000.0,
                "rv_kms": -20.0,
                "rv_err_kms": 1.0,
                "summary_path": "x",
            }
        ]
    )
    pipeline = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "method": "mask_ccf",
                "basename": "e1",
                "mjd": 59010.0,
                "rv_kms": -18.0,
                "rv_err_kms": 0.1,
            },
            {
                "gaia_dr3_id": "1",
                "method": "template_fft",
                "basename": "e1",
                "mjd": 59100.0,
                "rv_kms": 0.0,
                "rv_err_kms": 0.1,
            },
        ]
    )
    pairs = nearest_pipeline_join(external, pipeline)
    assert len(pairs) == 1
    assert pairs.iloc[0]["pipeline_method"] == "mask_ccf"
    assert pairs.iloc[0]["delta_days"] == pytest.approx(10.0)
    assert pairs.iloc[0]["delta_rv_kms"] == pytest.approx(-2.0)
