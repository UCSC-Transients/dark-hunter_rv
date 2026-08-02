"""Tests for Gaia NSS SB2 cohort fraction report."""

from pathlib import Path

import pandas as pd

from validation.sb2_nss_cohort_report import (
    build_fraction_table,
    load_exposure_sb2_flags,
    load_nss_source_ids,
)


def test_load_nss_and_fraction_table(tmp_path: Path):
    id_nss = "1111111111111111111"
    id_other = "3333333333333333333"
    nss = tmp_path / "nss.csv"
    nss.write_text(f"source_id\n{id_nss}\n2222222222222222222\n")
    ids = load_nss_source_ids(nss)
    assert id_nss in ids

    d1 = tmp_path / f"Gaia_DR3_{id_nss}_epoch_1_diagnostics.csv"
    d2 = tmp_path / f"Gaia_DR3_{id_other}_epoch_1_diagnostics.csv"
    pd.DataFrame(
        {"chunk_key": ["all", "10_0"], "sb2_candidate": [True, True]}
    ).to_csv(d1, index=False)
    pd.DataFrame(
        {"chunk_key": ["all"], "sb2_candidate": [False]}
    ).to_csv(d2, index=False)

    flags = load_exposure_sb2_flags(str(tmp_path / "Gaia_DR3_*_diagnostics.csv"))
    assert len(flags) == 2
    per, summary = build_fraction_table(flags, ids)
    assert int(summary.iloc[0]["n_stars"]) == 2
    assert int(summary.iloc[0]["n_nss_sb2"]) == 1
    assert int(summary.iloc[0]["n_flagged_among_nss"]) == 1
    assert abs(float(summary.iloc[0]["frac_flagged_among_nss"]) - 1.0) < 1e-9
    assert int(summary.iloc[0]["n_flagged_among_non_nss"]) == 0
