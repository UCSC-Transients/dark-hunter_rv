"""Tests for offline trust-weight A/B report."""

from pathlib import Path

import pandas as pd

from validation.trust_weight_ab_report import run_ab, stack_exposure
from darkhunter_rv import qc


def test_stack_trust_changes_weights(tmp_path: Path):
    cfg = dict(qc.DEFAULT_TRUST_WEIGHTS)
    df = pd.DataFrame(
        {
            "rv_kms": [10.0, 10.2, 30.0],
            "rv_err_kms": [1.0, 1.0, 1.0],
            "telluric_fraction": [0.0, 0.0, 0.0],
            "ccf_peak": [20.0, 20.0, 20.0],
            "ccf_asymmetry": [0.0, 0.0, 0.0],
            "qc_pass": [True, True, True],
        }
    )
    off = stack_exposure(df, trust_cfg=cfg, enabled=False)
    on = stack_exposure(df, trust_cfg=cfg, enabled=True)
    assert on["n_chunks"] == 3
    # Outlier at 30 should pull off stack more than on stack
    assert abs(on["rv_kms"] - 10.1) < abs(off["rv_kms"] - 10.1)


def test_run_ab_writes_summary(tmp_path: Path):
    d = tmp_path / "Gaia_DR3_1111111111111111111_epoch_1_diagnostics.csv"
    pd.DataFrame(
        {
            "chunk_key": ["10_0", "10_1", "all"],
            "method": ["mask_ccf", "mask_ccf", "mask_ccf"],
            "rv_kms": [5.0, 5.1, 5.05],
            "rv_err_kms": [0.5, 0.5, 0.1],
            "telluric_fraction": [0.0, 0.0, 0.0],
            "ccf_peak": [15.0, 15.0, 15.0],
            "ccf_asymmetry": [0.0, 0.0, 0.0],
            "qc_pass": [True, True, True],
        }
    ).to_csv(d, index=False)
    qc_path = tmp_path / "order_chunk_qc.yaml"
    qc_path.write_text("trust_weights:\n  enabled: false\n")
    per, summary = run_ab(str(tmp_path / "*_diagnostics.csv"), qc_config=qc_path)
    assert summary["n_exposures"] == 1
    assert len(per) == 1
