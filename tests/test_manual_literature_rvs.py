"""Tests for manual literature RV CSV merge surviving summary overwrite."""

from __future__ import annotations

from pathlib import Path

from darkhunter_rv import gaia_utils, io_utils
from darkhunter_rv.manual_literature_rvs import (
    merge_manual_literature,
    parse_manual_epoch_table,
    upsert_manual_literature_csv,
)


def _seed_csv(tmp_path: Path, gaia_id: int = 77413727493690112) -> Path:
    csv_path = tmp_path / "manual_literature_rvs.csv"
    rows = parse_manual_epoch_table(
        "lit_1 LITERATURE_Griffin1994 40494.09000000 -17.70000000 nan nan\n"
        "lit_2 LITERATURE_Griffin1994 46335.07000000 8.60000000 nan nan\n",
        gaia_id=gaia_id,
        default_rv_err=0.6,
    )
    upsert_manual_literature_csv(rows, path=csv_path)
    return csv_path


def test_parse_manual_epoch_table_fills_nan_err() -> None:
    rows = parse_manual_epoch_table(
        "# hdr\nlit_1 LITERATURE_Griffin1994 40494.09 -17.7 nan nan\n",
        gaia_id=99,
        default_rv_err=0.6,
    )
    assert len(rows) == 1
    assert rows[0]["rv_err_kms"] == "0.60000000"
    assert rows[0]["telescope"] == "LITERATURE_Griffin1994"
    assert rows[0]["flag"] == "Griffin1994"


def test_merge_manual_literature_replaces_only_literature(tmp_path, monkeypatch) -> None:
    csv_path = _seed_csv(tmp_path, gaia_id=42)
    monkeypatch.setenv("DARKHUNTER_MANUAL_LITERATURE_RVS", str(csv_path))
    existing = [
        {"telescope": "DESI_DR1", "mjd": 59200.0, "rv": -1.0, "rv_err": 1.0, "flag": "x"},
        {"telescope": "LITERATURE_Old", "mjd": 50000.0, "rv": 0.0, "rv_err": 1.0, "flag": "old"},
    ]
    merged = merge_manual_literature(42, existing, path=csv_path)
    teles = [r["telescope"] for r in merged]
    assert "DESI_DR1" in teles
    assert "LITERATURE_Old" not in teles
    assert teles.count("LITERATURE_Griffin1994") == 2
    assert all(r["rv_err"] == 0.6 for r in merged if r["telescope"].startswith("LITERATURE_"))


def test_merge_manual_literature_noop_without_csv_rows(tmp_path) -> None:
    csv_path = _seed_csv(tmp_path, gaia_id=42)
    existing = [
        {"telescope": "LITERATURE_Hand", "mjd": 50000.0, "rv": 1.0, "rv_err": 0.5, "flag": "h"},
    ]
    merged = merge_manual_literature(999, existing, path=csv_path)
    assert merged == existing


def test_write_star_summary_injects_literature(tmp_path, monkeypatch) -> None:
    csv_path = _seed_csv(tmp_path, gaia_id=77413727493690112)
    monkeypatch.setenv("DARKHUNTER_MANUAL_LITERATURE_RVS", str(csv_path))
    monkeypatch.setattr(io_utils.config, "OUTPUT_DIR", tmp_path)
    io_utils.write_star_summary(
        77413727493690112,
        {
            "metadata": {"Source_ID": 77413727493690112, "RA": 1.0, "Dec": 2.0},
            "external_rvs": [],
        },
        [
            {
                "file": "Gaia_DR3_77413727493690112_epoch_1.txt",
                "mjd": 60000.0,
                "rv": -3.0,
                "rv_err": 0.5,
                "rv_rms": 0.4,
                "fallback": False,
            }
        ],
    )
    summ = tmp_path / "Gaia_DR3_77413727493690112_summary.txt"
    ext = gaia_utils.parse_external_rvs_from_star_summary(summ)
    assert len(ext) == 2
    assert all(r["telescope"] == "LITERATURE_Griffin1994" for r in ext)


def test_replace_external_readds_literature_from_csv(tmp_path, monkeypatch) -> None:
    gid = 77413727493690112
    csv_path = _seed_csv(tmp_path, gaia_id=gid)
    monkeypatch.setenv("DARKHUNTER_MANUAL_LITERATURE_RVS", str(csv_path))
    summ = tmp_path / f"Gaia_DR3_{gid}_summary.txt"
    summ.write_text(
        f"[GAIA METADATA]\nSource_ID: {gid}\nRA: 1.0\nDec: 2.0\n"
        "\n[EXTERNAL RV DATA]\n# No external data found.\n"
        "\n[PIPELINE RESULTS]\n# hdr\n",
        encoding="utf-8",
    )
    gaia_utils.replace_external_rv_section_in_summary(
        summ,
        [{"telescope": "DESI_DR1", "mjd": 59230.0, "rv": -12.0, "rv_err": 1.0, "flag": "x"}],
    )
    ext = gaia_utils.parse_external_rvs_from_star_summary(summ)
    teles = [r["telescope"] for r in ext]
    assert "DESI_DR1" in teles
    assert teles.count("LITERATURE_Griffin1994") == 2
