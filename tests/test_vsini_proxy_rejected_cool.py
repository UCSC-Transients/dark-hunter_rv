"""Cool vs hot rejected/nonfinite vsini proxy grid defaults (step 10 / #87)."""

from __future__ import annotations

from darkhunter_rv import config


def test_cool_rejected_vsini_grid_narrower_than_hot():
    assert float(config.VSINI_PROXY_REJECTED_GRID_KMS_COOL) < float(
        config.VSINI_PROXY_REJECTED_GRID_KMS
    )
    assert float(config.VSINI_PROXY_REJECTED_GRID_KMS_COOL) <= 25.0


def test_cool_nonfinite_vsini_grid_narrower_than_hot():
    assert float(config.VSINI_PROXY_NONFINITE_GRID_KMS_COOL) < float(
        config.VSINI_PROXY_NONFINITE_GRID_KMS
    )
    assert float(config.VSINI_PROXY_NONFINITE_GRID_KMS_COOL) <= 25.0
