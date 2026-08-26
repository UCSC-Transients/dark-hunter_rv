"""Tests for machine-readable RV summary JSON (pop CandidateRecord.rv_summary)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from darkhunter_rv import config, io_utils
from darkhunter_rv.rv_summary_json import (
    RV_SUMMARY_SCHEMA_VERSION,
    attach_joker_fit_to_summary_json,
    build_rv_summary_dict,
    load_rv_summary_json,
    refresh_rv_summary_json_from_txt,
    rv_summary_from_star_summary_txt,
)


def _gaia_data(*, with_orbit: bool = True) -> dict:
    meta = {
        "Source_ID": 1234567890123456789,
        "RA": 10.0,
        "Dec": -20.0,
        "NSS_Solution_Type": "Orbital" if with_orbit else "None",
        "Period": 100.0,
        "Period_Error": 1.0,
        "Eccentricity": 0.1,
        "Eccentricity_Error": 0.01,
        "A_Thiele_Innes": 0.5,
        "A_Thiele_Innes_Error": 0.05,
        "B_Thiele_Innes": -0.4,
        "B_Thiele_Innes_Error": 0.04,
        "F_Thiele_Innes": -0.3,
        "F_Thiele_Innes_Error": 0.03,
        "G_Thiele_Innes": -0.6,
        "G_Thiele_Innes_Error": 0.06,
    }
    return {
        "metadata": meta,
        "external_rvs": [
            {
                "telescope": "LAMOST_LRS",
                "mjd": 56316.0,
                "rv": -42.0,
                "rv_err": 5.0,
                "flag": "lit",
            }
        ],
    }


def test_build_rv_summary_dict_fields() -> None:
    merged = {
        "epoch_1.txt": "epoch_1.txt 60000.0 1.0 0.1 0.5 False",
        "epoch_2.txt": "epoch_2.txt 60001.0 2.0 0.2 0.6 False",
    }
    payload = build_rv_summary_dict(
        1234567890123456789,
        gaia_data=_gaia_data(),
        merged_pipeline_lines=merged,
    )
    assert payload["schema_version"] == RV_SUMMARY_SCHEMA_VERSION
    assert payload["source_id"] == 1234567890123456789
    assert payload["instruments"] == ["APF", "LAMOST_LRS"]
    assert payload["n_epochs"] == 3
    assert payload["n_pipeline_epochs"] == 2
    assert payload["n_external_epochs"] == 1
    assert payload["nss_solution_type"] == "Orbital"
    assert payload["nss_orbital"]["period_day"] == pytest.approx(100.0)
    assert payload["thiele_innes"]["A"] == pytest.approx(0.5)
    assert len(payload["pipeline_epochs"]) == 2
    assert payload["pipeline_epochs"][0]["telescope"] == "APF"


def test_write_star_summary_emits_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    oid = 1234567890123456789
    io_utils.write_star_summary(
        oid,
        _gaia_data(),
        [
            {
                "file": "/fake/Gaia_DR3_x_epoch_1.txt",
                "mjd": 60000.0,
                "rv": 1.0,
                "rv_err": 0.1,
                "rv_rms": 0.5,
                "fallback": False,
            }
        ],
    )
    json_path = tmp_path / f"Gaia_DR3_{oid}_summary.json"
    txt_path = tmp_path / f"Gaia_DR3_{oid}_summary.txt"
    assert json_path.is_file()
    payload = load_rv_summary_json(json_path)
    assert payload["n_epochs"] == 2
    assert payload["summary_txt"] == txt_path.name


def test_refresh_from_txt_and_attach_joker(tmp_path: Path) -> None:
    oid = 987654321098765432
    txt = tmp_path / f"Gaia_DR3_{oid}_summary.txt"
    txt.write_text(
        f"### STAR SUMMARY: {oid} ###\n\n"
        "[GAIA METADATA]\n"
        f"Source_ID: {oid}\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "NSS_Solution_Type: Orbital\n"
        "Period: 50.0\n"
        "Eccentricity: 0.2\n"
        "\n[EXTERNAL RV DATA]\n"
        "# Telescope | MJD | RV (km/s) | Err (km/s) | Flag/ID\n"
        "# No external data found.\n"
        "\n[PIPELINE RESULTS]\n"
        "# File | MJD | RV (km/s) | Err (km/s) | wRMS (km/s) | Fallback?\n"
        "Gaia_DR3_x_epoch_1.txt 60000.0 1.0 0.1 0.5 False\n",
        encoding="utf-8",
    )
    refresh_rv_summary_json_from_txt(txt)
    joker_path = tmp_path / f"Gaia_DR3_{oid}_joker_fit.json"
    joker_path.write_text(
        json.dumps({"gaia_source_id": str(oid), "fit_engine": "thejoker", "n_points": 1}),
        encoding="utf-8",
    )
    attach_joker_fit_to_summary_json(
        txt, json.loads(joker_path.read_text()), joker_fit_path=joker_path
    )
    payload = rv_summary_from_star_summary_txt(txt, joker_fit_path=joker_path)
    assert payload["joker_fit"]["fit_engine"] == "thejoker"
    assert payload["joker_fit_path"].endswith("_joker_fit.json")
