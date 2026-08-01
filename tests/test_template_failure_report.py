"""Tests for validation.template_failure_report classification and cohort selection."""

from __future__ import annotations

import pandas as pd

from validation.template_failure_report import (
    classify_three_method_exposure,
    enrich_overlap_table,
    select_cohort,
    snr_floor_from_percentile,
)


def _row(**kw) -> dict:
    base = {
        "mask_valid": True,
        "template_valid": True,
        "strong_lines_valid": True,
        "delta_mask_minus_template_kms": 0.0,
        "delta_mask_minus_strong_lines_kms": 0.0,
        "delta_template_minus_strong_lines_kms": 0.0,
        "median_mask_ccf_peak_snr": 5.0,
        "mask_err_kms": 0.5,
        "template_err_kms": 0.5,
        "strong_lines_err_kms": 1.0,
    }
    base.update(kw)
    return base


def test_classify_D_agree():
    r = pd.Series(_row())
    assert classify_three_method_exposure(r, threshold_kms=5.0) == "D_agree"


def test_classify_B_template_outlier():
    r = pd.Series(
        _row(
            delta_mask_minus_template_kms=-50.0,
            delta_mask_minus_strong_lines_kms=1.0,
            delta_template_minus_strong_lines_kms=51.0,
        )
    )
    assert classify_three_method_exposure(r, threshold_kms=5.0) == "B_template_outlier"


def test_classify_E_incomplete():
    r = pd.Series(_row(template_valid=False))
    assert classify_three_method_exposure(r, threshold_kms=5.0) == "E_incomplete"


def test_snr_percentile_floor():
    tab = pd.DataFrame({"median_mask_ccf_peak_snr": [1.0, 2.0, 3.0, 4.0]})
    assert snr_floor_from_percentile(tab, 75.0) == 3.25


def test_select_cohort_top_quartile():
    rows = [
        _row(median_mask_ccf_peak_snr=1.0, basename="low.txt"),
        _row(median_mask_ccf_peak_snr=2.0, basename="mid.txt"),
        _row(median_mask_ccf_peak_snr=3.0, basename="hi.txt"),
        _row(median_mask_ccf_peak_snr=4.0, basename="top.txt"),
    ]
    tab = enrich_overlap_table(pd.DataFrame(rows), threshold_kms=5.0)
    cohort, floor = select_cohort(
        tab,
        snr_percentile_floor=75.0,
        min_log10_median_mask_ccf_peak_snr=None,
        min_median_mask_ccf_peak_snr=None,
        require_three_methods=True,
        max_method_err_kms=100.0,
    )
    assert floor == 3.25
    assert len(cohort) == 1
    assert cohort.iloc[0]["basename"] == "top.txt"
