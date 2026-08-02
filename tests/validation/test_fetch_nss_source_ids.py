"""Tests for fetch_nss_source_ids helpers (no live Gaia)."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from validation.fetch_nss_source_ids import (
    collect_gaia_ids_from_diagnostics,
    query_nss_two_body_ids,
)


def test_collect_gaia_ids(tmp_path: Path):
    (tmp_path / "Gaia_DR3_1111111111111111111_epoch_1_diagnostics.csv").write_text("a\n")
    (tmp_path / "Gaia_DR3_2222222222222222222_epoch_2_diagnostics.csv").write_text("a\n")
    ids = collect_gaia_ids_from_diagnostics(str(tmp_path / "*_diagnostics.csv"))
    assert ids == ["1111111111111111111", "2222222222222222222"]


def test_query_nss_chunks_mocked():
    with patch(
        "validation.fetch_nss_source_ids.execute_gaia_adql",
        return_value=[{"source_id": 1111111111111111111, "nss_solution_type": "SB2"}],
    ) as mock:
        rows = query_nss_two_body_ids(["1111111111111111111", "2222222222222222222"], chunk_size=1)
    assert mock.call_count == 2
    assert rows == [{"source_id": "1111111111111111111", "nss_solution_type": "SB2"}]
