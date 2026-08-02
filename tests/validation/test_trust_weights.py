"""Trust-weighted IVW stack (step 02b)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_rv import qc
from darkhunter_rv.pipeline import _weighted_method_rv_from_rows


@pytest.mark.validation
def test_load_trust_weights_defaults_opt_in_off():
    cfg = qc.load_trust_weights_config(Path("order_chunk_qc.yaml"))
    assert cfg["enabled"] is False
    assert cfg["residual_scale_kms"] > 0
    assert cfg["telluric_hard_max"] > cfg["telluric_soft_max"]


@pytest.mark.validation
def test_trust_factors_residual_telluric_ccf():
    cfg = dict(qc.DEFAULT_TRUST_WEIGHTS)
    good = qc.chunk_trust_components(
        rv_kms=10.0,
        robust_mean_kms=10.0,
        telluric_fraction=0.0,
        ccf_peak_snr=20.0,
        ccf_asymmetry=0.0,
        cfg=cfg,
    )
    assert good["trust_weight"] == pytest.approx(1.0)

    far = qc.chunk_trust_components(
        rv_kms=20.0,
        robust_mean_kms=10.0,
        telluric_fraction=0.0,
        ccf_peak_snr=20.0,
        ccf_asymmetry=0.0,
        cfg=cfg,
    )
    assert far["trust_residual"] < 1.0
    assert far["trust_weight"] < good["trust_weight"]

    tel = qc.chunk_trust_components(
        rv_kms=10.0,
        robust_mean_kms=10.0,
        telluric_fraction=0.5,
        ccf_peak_snr=20.0,
        ccf_asymmetry=0.0,
        cfg=cfg,
    )
    assert tel["trust_telluric"] == 0.0
    assert tel["trust_weight"] == 0.0

    # Missing CCF metrics must not kill template-like rows
    miss = qc.chunk_trust_components(
        rv_kms=10.0,
        robust_mean_kms=10.0,
        telluric_fraction=0.0,
        ccf_peak_snr=float("nan"),
        ccf_asymmetry=float("nan"),
        cfg=cfg,
    )
    assert miss["trust_ccf"] == pytest.approx(1.0)


@pytest.mark.validation
def test_ivw_weights_with_trust_opt_in():
    errs = np.array([1.0, 1.0, 1.0])
    trust = np.array([1.0, 0.5, 0.0])
    plain = qc.ivw_weights_with_trust(errs, trust, enabled=False)
    assert np.allclose(plain, 1.0)
    scaled = qc.ivw_weights_with_trust(errs, trust, enabled=True)
    assert scaled[0] == pytest.approx(1.0)
    assert scaled[1] == pytest.approx(0.5)
    assert scaled[2] == pytest.approx(0.0)


@pytest.mark.validation
def test_weighted_method_rv_trust_scales_stack():
    rows = []
    # Five good chunks near 10 km/s + one telluric-heavy outlier at 40
    for i, rv in enumerate([10.0, 10.1, 9.9, 10.05, 9.95, 40.0]):
        rows.append(
            {
                "method": "mask_ccf",
                "chunk_key": f"30_{i}",
                "rv_kms": rv,
                "rv_err_kms": 1.0,
                "qc_pass": True,
                "telluric_fraction": 0.5 if rv > 30 else 0.0,
                "ccf_peak_snr": 20.0,
                "ccf_asymmetry": 0.0,
            }
        )
    off = dict(qc.DEFAULT_TRUST_WEIGHTS)
    off["enabled"] = False
    mu_plain, _ = _weighted_method_rv_from_rows(rows, "mask_ccf", trust_cfg=off)
    on = dict(off)
    on["enabled"] = True
    mu_trust, _ = _weighted_method_rv_from_rows(rows, "mask_ccf", trust_cfg=on)
    assert mu_plain > 12.0  # outlier pulls plain IVW
    assert abs(mu_trust - 10.0) < abs(mu_plain - 10.0)
    assert abs(mu_trust - 10.0) < 0.2
