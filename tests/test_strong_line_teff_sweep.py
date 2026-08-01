"""Unit tests for strong-line Teff sweep selection helpers."""

import numpy as np

from validation.strong_line_teff_sweep import (
    pipeline_score,
    select_best_line,
    teff_bin_label,
)


def test_teff_bin_label():
    assert teff_bin_label(4200.0) == "<4500"
    assert teff_bin_label(5000.0) == "4500-5500"
    assert teff_bin_label(6000.0) == "5500-6500"
    assert teff_bin_label(7000.0) == ">=6500"


def test_pipeline_score_prefers_earlier_candidate_on_tie():
    # Same err → earlier cand_index wins (lower score)
    assert pipeline_score(5.0, 0) < pipeline_score(5.0, 1)


def test_select_best_line_short_circuit_keeps_hbeta():
    cands = [
        {"line": "Hbeta", "cand_index": 0, "err_kms": 5.0, "rv_kms": 10.0},
        {"line": "Hgamma", "cand_index": 1, "err_kms": 1.0, "rv_kms": 50.0},
    ]
    best = select_best_line(cands, short_circuit_hbeta=True)
    assert best is not None
    assert best["line"] == "Hbeta"


def test_select_best_line_free_can_pick_lower_err():
    cands = [
        {"line": "Hbeta", "cand_index": 0, "err_kms": 8.0, "rv_kms": 10.0},
        {"line": "Hgamma", "cand_index": 1, "err_kms": 1.0, "rv_kms": 11.0},
    ]
    # Without short-circuit, lower total score wins (1.0+0.5=1.5 vs 8.0)
    best = select_best_line(cands, short_circuit_hbeta=False)
    assert best is not None
    assert best["line"] == "Hgamma"


def test_select_best_line_empty():
    assert select_best_line([]) is None
    assert select_best_line([], short_circuit_hbeta=False) is None


def test_select_best_line_nonfinite_err_penalized():
    cands = [
        {"line": "Hbeta", "cand_index": 0, "err_kms": float("nan"), "rv_kms": 1.0},
        {"line": "Halpha", "cand_index": 1, "err_kms": 3.0, "rv_kms": 2.0},
    ]
    best = select_best_line(cands, short_circuit_hbeta=False)
    assert best is not None
    assert best["line"] == "Halpha"
    assert np.isfinite(best["score"])
