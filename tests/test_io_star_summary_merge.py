"""Star summary merges pipeline rows across per-spectrum runs."""
from pathlib import Path

import pytest

from darkhunter_rv import config, io_utils


def test_write_star_summary_merges_epochs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    oid = 9999999999999999999
    gaia_data = {"metadata": {"Teff": 5800.0}, "external_rvs": []}

    io_utils.write_star_summary(
        oid,
        gaia_data,
        [
            {
                "file": "/fake/a_epoch_1.txt",
                "mjd": 60000.0,
                "rv": 1.0,
                "rv_err": 0.1,
                "rv_rms": 0.5,
                "fallback": False,
            }
        ],
    )
    io_utils.write_star_summary(
        oid,
        gaia_data,
        [
            {
                "file": "/other/b_epoch_2.txt",
                "mjd": 60001.0,
                "rv": 2.0,
                "rv_err": 0.2,
                "rv_rms": 0.6,
                "fallback": True,
            }
        ],
    )

    text = (tmp_path / f"Gaia_DR3_{oid}_summary.txt").read_text()
    assert "a_epoch_1.txt" in text
    assert "b_epoch_2.txt" in text
    assert text.index("epoch_1") < text.index("epoch_2")


def test_write_star_summary_preserves_sed_m1(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    oid = 1234567890123456789
    path = tmp_path / f"Gaia_DR3_{oid}_summary.txt"
    path.write_text(
        f"### STAR SUMMARY: {oid} ###\n\n"
        "[GAIA METADATA]\n"
        f"Source_ID: {oid}\n"
        "Teff: 5800.0\n"
        "M1: 0.91000000\n"
        "m1_msun: 0.91000000\n"
        "M1_p16: 0.88000000\n"
        "M1_p84: 0.95000000\n"
        "\n[EXTERNAL RV DATA]\n"
        "# Telescope | MJD | RV (km/s) | Err (km/s) | Flag/ID\n"
        "# No external data found.\n"
        "\n[PIPELINE RESULTS]\n"
        "# File | MJD | RV (km/s) | Err (km/s) | wRMS (km/s) | Fallback?\n"
        "old_epoch_1.txt 60000.00000 1.000 0.100 0.500 False\n",
        encoding="utf-8",
    )

    io_utils.write_star_summary(
        oid,
        {"metadata": {"Source_ID": oid, "Teff": 5750.0}, "external_rvs": []},
        [
            {
                "file": "/fake/new_epoch_2.txt",
                "mjd": 60002.0,
                "rv": 3.0,
                "rv_err": 0.1,
                "rv_rms": 0.4,
                "fallback": False,
            }
        ],
    )
    text = path.read_text(encoding="utf-8")
    assert "M1: 0.91000000" in text
    assert "m1_msun: 0.91000000" in text
    assert "M1_p16: 0.88000000" in text
    assert "M1_p84: 0.95000000" in text
    assert "Teff: 5750.00000000" in text
    assert "new_epoch_2.txt" in text


def test_write_star_summary_preserves_meta_when_query_fails(tmp_path, monkeypatch):
    """Live Gaia None must not wipe existing [GAIA METADATA] / EXTERNAL rows."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    oid = 1125337614021152256
    path = tmp_path / f"Gaia_DR3_{oid}_summary.txt"
    path.write_text(
        f"### STAR SUMMARY: {oid} ###\n\n"
        "[GAIA METADATA]\n"
        f"Source_ID: {oid}\n"
        "RA: 10.0\n"
        "Dec: 20.0\n"
        "Parallax: 2.26000000\n"
        "Parallax_Error: 0.02200000\n"
        "Teff: 5500.00000000\n"
        "\n[EXTERNAL RV DATA]\n"
        "# Telescope | MJD | RV (km/s) | Err (km/s) | Flag/ID\n"
        "DESI_SV 60000.00000 1.000 0.500 flag\n"
        "\n[PIPELINE RESULTS]\n"
        "# File | MJD | RV (km/s) | Err (km/s) | wRMS (km/s) | Fallback?\n"
        "epoch_1.txt 60000.00000 16.000 0.100 0.200 False\n",
        encoding="utf-8",
    )

    io_utils.write_star_summary(oid, None, [])

    text = path.read_text(encoding="utf-8")
    assert "Not Found or Query Failed" not in text
    assert "Parallax_Error: 0.02200000" in text
    assert "DESI_SV" in text
    assert "epoch_1.txt" in text
