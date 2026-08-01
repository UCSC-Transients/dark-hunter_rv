"""Unit tests for strong-line candidate detection QC."""

import numpy as np

from validation.strong_line_candidate_sweep import (
    CANDIDATES,
    detection_ok,
    local_core_depth,
)


def test_candidates_include_hbeta_and_mg():
    names = {c.name for c in CANDIDATES}
    assert "Hbeta" in names
    assert "MgIb3" in names
    assert "CaIIK" in names
    assert any(c.red_risk for c in CANDIDATES)
    assert any(c.activity for c in CANDIDATES)


def test_local_core_depth_detects_absorption():
    rest = 5183.6
    w = np.linspace(rest - 30, rest + 30, 400)
    f = np.ones_like(w) - 0.4 * np.exp(-0.5 * ((w - rest) / 0.8) ** 2)
    d = local_core_depth(w, f, rest)
    assert d > 0.2


def test_detection_ok_rejects_shallow_and_telluric():
    ok, reason = detection_ok(
        fit_ok=True,
        depth=0.01,
        err_kms=2.0,
        rv_kms=10.0,
        telluric_frac=0.0,
        red_risk=False,
        rest_a=5183.6,
    )
    assert not ok and reason == "shallow_or_undetected"
    ok2, reason2 = detection_ok(
        fit_ok=True,
        depth=0.3,
        err_kms=2.0,
        rv_kms=10.0,
        telluric_frac=0.2,
        red_risk=True,
        rest_a=8662.0,
    )
    assert not ok2 and reason2 == "telluric_contaminated"
    ok3, _ = detection_ok(
        fit_ok=True,
        depth=0.3,
        err_kms=2.0,
        rv_kms=10.0,
        telluric_frac=0.01,
        red_risk=False,
        rest_a=5183.6,
    )
    assert ok3
