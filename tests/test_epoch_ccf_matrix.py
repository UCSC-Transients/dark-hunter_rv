"""Tests for epoch CCF matrix builder CLI / I/O (synthetic)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from darkhunter_rv import config
from darkhunter_rv.epoch_ccf import build_relative_matrix_from_pairs
from validation.epoch_ccf_matrix import (
    compute_pair_matrix,
    discover_epoch_spectra,
    load_abs_from_diagnostics,
    main,
    run_matrix,
    spectrum_to_normalized_1d,
)


def _write_fake_spectrum(path: Path, *, rv_kms: float, n_orders: int = 2) -> None:
    """Minimal APF-like text spectrum with Doppler-shifted Gaussian lines."""
    lines = [
        "# TEST SPECTRUM\n",
        f"# RV_INJECT_KMS = {rv_kms}\n",
    ]
    z = float(rv_kms) / float(config.C_KMS)
    centers0 = [5050.0, 5080.0, 5110.0]
    for o in range(n_orders):
        lines.append(f"# Order {o}\n")
        lo = 5000.0 + 40.0 * o
        hi = lo + 80.0
        wave = np.linspace(lo, hi, 800)
        flux = np.ones_like(wave)
        for c0 in centers0:
            if c0 < lo or c0 > hi:
                continue
            lam = c0 * (1.0 + z)
            sig = lam * (8.0 / float(config.C_KMS))
            flux -= 0.5 * np.exp(-0.5 * ((wave - lam) / (sig + 1e-12)) ** 2)
        for w, f in zip(wave, flux):
            lines.append(f"{w:.6f} {f:.8f}\n")
    path.write_text("".join(lines))


def test_discover_epoch_spectra(tmp_path: Path):
    gid = "1234567890123456789"
    _write_fake_spectrum(tmp_path / f"Gaia_DR3_{gid}_epoch_2.txt", rv_kms=0.0)
    _write_fake_spectrum(tmp_path / f"Gaia_DR3_{gid}_epoch_1.txt", rv_kms=10.0)
    (tmp_path / f"Gaia_DR3_{gid}_other.txt").write_text("x")
    found = discover_epoch_spectra(tmp_path, gid)
    assert [e for e, _ in found] == [1, 2]


def test_spectrum_to_normalized_1d(tmp_path: Path):
    path = tmp_path / "Gaia_DR3_1_epoch_1.txt"
    _write_fake_spectrum(path, rv_kms=5.0)
    w, f = spectrum_to_normalized_1d(path)
    assert len(w) > 100
    assert np.all(np.diff(w) >= 0)
    assert np.nanmedian(f) == pytest.approx(1.0, abs=0.2)


def _synth_flux(wave: np.ndarray, rv_kms: float) -> np.ndarray:
    z = float(rv_kms) / float(config.C_KMS)
    flux = np.ones_like(wave, dtype=float)
    for lam0 in (5050.0, 5080.0, 5110.0, 5135.0, 5160.0):
        lam = lam0 * (1.0 + z)
        sig = lam * (8.0 / float(config.C_KMS))
        flux -= 0.5 * np.exp(-0.5 * ((wave - lam) / (sig + 1e-12)) ** 2)
    return flux


def test_compute_pair_matrix_diagonal_and_antisym():
    wave = np.linspace(5000.0, 5200.0, 3000)
    f0 = _synth_flux(wave, 0.0)
    f1 = _synth_flux(wave, 20.0)
    f2 = _synth_flux(wave, -10.0)
    pairs, rows = compute_pair_matrix([wave, wave, wave], [f0, f1, f2], rv_search_half_width_kms=200.0)
    assert (0, 0) in pairs and pairs[(0, 0)].qc.get("auto_correlation") is True
    assert abs(pairs[(0, 0)].dv_kms) < 1.0
    off = {k: v for k, v in pairs.items() if k[0] != k[1]}
    dv, _sig = build_relative_matrix_from_pairs(3, off)
    assert abs(dv[1, 2] + dv[2, 1]) < 1e-9
    assert any(r["i"] == 1 and r["j"] == 0 for r in rows)


def test_run_matrix_writes_artifacts(tmp_path: Path):
    gid = "9998887776665554444"
    rvs = [0.0, 25.0, -15.0]
    for i, rv in enumerate(rvs, start=1):
        _write_fake_spectrum(tmp_path / f"Gaia_DR3_{gid}_epoch_{i}.txt", rv_kms=rv)

    # Fake diagnostics with abs = injected (zeropoint free but differences match)
    abs_vals = [100.0, 125.0, 85.0]  # same deltas as rvs
    for i, a in enumerate(abs_vals, start=1):
        pd.DataFrame(
            {
                "method": ["mask_ccf"],
                "exposure_rv_kms": [a],
                "exposure_rv_err_kms": [0.2],
            }
        ).to_csv(tmp_path / f"Gaia_DR3_{gid}_epoch_{i}_diagnostics.csv", index=False)

    out = tmp_path / "out"
    meta = run_matrix(
        gaia_id=gid,
        data_root=tmp_path,
        out_dir=out,
        abs_diagnostics_glob=str(tmp_path / f"Gaia_DR3_{gid}_epoch_*_diagnostics.csv"),
        rv_search_half_width_kms=200.0,
        max_grid_points=4096,
    )
    assert (out / "epoch_ccf_pairs.csv").is_file()
    assert (out / "epoch_ccf_matrix.npz").is_file()
    assert (out / "epoch_ccf_abs_fill.csv").is_file()
    assert (out / "epoch_ccf_meta.json").is_file()
    assert (out / "epoch_ccf_vs_abs_delta.csv").is_file()
    assert meta["n_epochs"] == 3
    assert meta["n_abs_anchors"] == 3
    assert meta["diag_abs_max_kms"] is not None
    assert meta["diag_abs_max_kms"] < 2.0

    z = np.load(out / "epoch_ccf_matrix.npz")
    assert z["dv_kms"].shape == (3, 3)
    # Diagonal ~0
    assert np.nanmax(np.abs(np.diag(z["dv_kms"]))) < 2.0

    fill = pd.read_csv(out / "epoch_ccf_abs_fill.csv")
    assert "epoch_ccf_rel" in fill.columns
    assert "epoch_ccf_abs_fill" in fill.columns
    assert fill["n_abs_anchors"].iloc[0] == 3

    cmp_df = pd.read_csv(out / "epoch_ccf_vs_abs_delta.csv")
    assert len(cmp_df) >= 1
    # CCF Δv should track abs Δv within a few km/s on clean synthetics
    assert float(np.nanmedian(np.abs(cmp_df["residual_kms"]))) < 5.0


def test_load_abs_from_diagnostics(tmp_path: Path):
    gid = "111"
    pd.DataFrame(
        {
            "method": ["mask_ccf", "template_fft"],
            "exposure_rv_kms": [12.5, 99.0],
            "exposure_rv_err_kms": [0.3, 1.0],
        }
    ).to_csv(tmp_path / f"Gaia_DR3_{gid}_epoch_1_diagnostics.csv", index=False)
    abs_rv, abs_sig, used = load_abs_from_diagnostics(
        str(tmp_path / f"Gaia_DR3_{gid}_epoch_*_diagnostics.csv"),
        gid,
        [1, 2],
    )
    assert abs_rv[0] == pytest.approx(12.5)
    assert abs_sig[0] == pytest.approx(0.3)
    assert used[0] is not None
    assert np.isnan(abs_rv[1])


def test_cli_main_synthetic(tmp_path: Path):
    gid = "5554443332221110009"
    for i, rv in enumerate([0.0, 12.0], start=1):
        _write_fake_spectrum(tmp_path / f"Gaia_DR3_{gid}_epoch_{i}.txt", rv_kms=rv)
    out = tmp_path / "cli_out"
    rc = main(
        [
            "--gaia-id",
            gid,
            "--data-root",
            str(tmp_path),
            "--out-dir",
            str(out),
            "--rv-search-half-width-kms",
            "200",
            "--max-grid-points",
            "4096",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    meta = json.loads((out / "epoch_ccf_meta.json").read_text())
    assert meta["n_epochs"] == 2
    assert meta["float_zeropoint"] is True
    assert meta["n_abs_anchors"] == 0
