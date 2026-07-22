"""HIRES Makee reader, continuum none, and rename helpers."""

from __future__ import annotations

from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import pytest

from darkhunter_rv.continuum import fit_continuum
from darkhunter_rv.io_utils import (
    _hires_eflux_from_unc,
    apply_barycentric_wavelength_shift,
    compute_hires_bjd,
    extract_mjd_from_header,
    hires_berv_kms,
    hires_photon_weighted_time,
    read_spectrum_hires,
)
from darkhunter_rv.instruments import get_instrument_profile
from darkhunter_rv import config
from scripts.rename_hires_by_gaia import (
    crosscheck_targname,
    parse_makee_chip_name,
    pick_gaia_match,
    target_filename,
    targname_matches_source_id,
    write_headers_and_rename,
    HiresChipFile,
    HiresObservation,
)


def _write_hires_chip(
    path: Path,
    *,
    n_orders: int,
    n_pix: int,
    wave0: float,
    dlam: float,
    relative_unc: bool = True,
) -> None:
    flux = np.full((n_orders, n_pix), 100.0, dtype=np.float32)
    for i in range(n_orders):
        flux[i] += 10.0 * i
    if relative_unc:
        unc = np.full_like(flux, 0.05)
    else:
        unc = np.full_like(flux, 5.0)
    wave = np.zeros_like(flux, dtype=np.float64)
    for i in range(n_orders):
        wave[i] = wave0 + i * 50.0 + dlam * np.arange(n_pix)
    fits.HDUList(
        [
            fits.PrimaryHDU(flux),
            fits.ImageHDU(unc),
            fits.ImageHDU(wave),
        ]
    ).writeto(path, overwrite=True)


def test_hires_instrument_profile() -> None:
    inst = get_instrument_profile("HIRES")
    assert inst.num_orders == 49
    assert inst.resolving_power == 45000.0
    assert inst.header_keywords["mjd"] == "MJD"
    assert inst.bias_file is None


def test_hires_eflux_relative_vs_absolute() -> None:
    flux = np.full(100, 120.0)
    rel = np.full(100, 0.07)
    e = _hires_eflux_from_unc(flux, rel)
    assert np.allclose(e, rel * np.abs(flux))
    abs_unc = np.full(100, 8.0)
    e2 = _hires_eflux_from_unc(flux, abs_unc)
    assert np.allclose(e2, abs_unc)


def test_read_spectrum_hires_triplet(tmp_path: Path) -> None:
    gid = 1551542027851147904
    for chip, n_ord, wave0 in (("b", 3, 4000.0), ("r", 2, 5000.0), ("i", 2, 6500.0)):
        path = tmp_path / f"Gaia_DR3_{gid}_{chip}_epoch_1.fits"
        _write_hires_chip(path, n_orders=n_ord, n_pix=20, wave0=wave0, dlam=0.1)
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["BJD"] = 2460000.5
            hdul[0].header["MJD"] = 60000.0
            hdul[0].header["RA"] = "13:36:27.00"
            hdul[0].header["DEC"] = "+45:52:23.0"
            hdul.flush()

    blue = tmp_path / f"Gaia_DR3_{gid}_b_epoch_1.fits"
    with fits.open(blue) as hdul:
        wave0_raw = float(hdul[2].data[0, 0])
        berv = hires_berv_kms(hdul[0].header)
    header, spec = read_spectrum_hires(blue)
    assert len(spec) == 7
    assert float(header["BERV"]) == pytest.approx(berv, rel=0, abs=1e-6)
    # Wavelengths shifted by BERV; time prefers MJD over BJD
    assert spec[0]["wavelength"][0] == pytest.approx(
        wave0_raw * (1.0 + berv / config.C_KMS), rel=0, abs=1e-8
    )
    assert float(hires_photon_weighted_time(header).mjd) == pytest.approx(60000.0)
    # Wavelength order: first order from blue, last from intermediate
    assert spec[0]["wavelength"][0] < spec[6]["wavelength"][0]
    med_f = np.median(spec[0]["flux"])
    med_e = np.median(spec[0]["eflux"])
    assert med_e == pytest.approx(0.05 * med_f, rel=1e-3)

    inst = get_instrument_profile("HIRES")
    mjd = extract_mjd_from_header(header, inst)
    assert mjd == pytest.approx(60000.0, rel=0, abs=1e-9)


def test_photon_weighted_time_prefers_mjd() -> None:
    hdr = fits.Header()
    hdr["MJD"] = 61170.445934
    hdr["BJD"] = 2461170.99
    t = hires_photon_weighted_time(hdr)
    assert float(t.mjd) == pytest.approx(61170.445934)


def test_barycentric_wavelength_shift() -> None:
    wave = np.array([5000.0, 5001.0])
    shifted = apply_barycentric_wavelength_shift(wave, -14.712206)
    assert shifted[0] < wave[0]
    assert shifted[0] == pytest.approx(wave[0] * (1.0 - 14.712206 / config.C_KMS))


def test_continuum_none_passthrough() -> None:
    w = np.linspace(5000.0, 5100.0, 200)
    flux = np.full(200, 80.0)
    flux[80:90] = 40.0
    eflux = np.full(200, 4.0)
    nw, nf, ne = fit_continuum(w, flux, eflux, continuum_mode="none")
    assert np.allclose(nw, w)
    assert np.allclose(nf, flux)
    assert np.allclose(ne, eflux)


def test_parse_makee_and_targname() -> None:
    parsed = parse_makee_chip_name(Path("bj560.64.fits"))
    assert parsed is not None
    assert parsed.chip == "b"
    assert parsed.outfile == "j560"
    assert parsed.frameno == 64
    assert targname_matches_source_id("155154202785114", 1551542027851147904)
    assert not targname_matches_source_id("999", 1551542027851147904)
    ok, _ = crosscheck_targname("155154202785114", 1551542027851147904)
    assert ok
    bad, msg = crosscheck_targname("999", 1551542027851147904)
    assert not bad
    forced, _ = crosscheck_targname("999", 1551542027851147904, force=True)
    assert forced


def test_pick_gaia_match_tiebreak() -> None:
    rows = [
        {"source_id": 1, "sep": 0.001, "phot_g_mean_mag": 12.0},
        {"source_id": 2, "sep": 0.001, "phot_g_mean_mag": 10.0},
        {"source_id": 3, "sep": 0.002, "phot_g_mean_mag": 9.0},
    ]
    pick = pick_gaia_match(rows)
    assert pick is not None
    assert pick["source_id"] == 2


def test_write_headers_and_rename_dry_and_apply(tmp_path: Path) -> None:
    chips = {}
    for chip in ("b", "i", "r"):
        path = tmp_path / f"{chip}j560.64.fits"
        _write_hires_chip(path, n_orders=1, n_pix=8, wave0=4500.0, dlam=0.2)
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["RA"] = "13:36:27.00"
            hdul[0].header["DEC"] = "+45:52:23.0"
            hdul[0].header["MJD"] = 61170.445934
            hdul[0].header["TARGNAME"] = "155154202785114"
            hdul.flush()
        chips[chip] = HiresChipFile(path=path, chip=chip, outfile="j560", frameno=64)

    with fits.open(chips["b"].path) as hdul:
        header = hdul[0].header.copy()
    obs = HiresObservation(
        outfile="j560",
        frameno=64,
        chips=chips,
        ra_deg=204.1125,
        dec_deg=45.8730555556,
        mjd=61170.445934,
        targname="155154202785114",
        header_sample=header,
    )
    sid = 1551542027851147904
    planned = write_headers_and_rename(obs, sid, 1, apply=False)
    assert planned[0][1].name == target_filename(sid, "b", 1)
    assert chips["b"].path.is_file()

    write_headers_and_rename(obs, sid, 1, apply=True)
    dest_b = tmp_path / target_filename(sid, "b", 1)
    assert dest_b.is_file()
    with fits.open(dest_b) as hdul:
        assert hdul[0].header["GAIADR3ID"] == f"Gaia DR3 {sid}"
        assert "BJD" in hdul[0].header
        assert float(hdul[0].header["BJD"]) == pytest.approx(
            compute_hires_bjd(hdul[0].header), rel=0, abs=1e-8
        )
        berv = hires_berv_kms(hdul[0].header)
        assert berv == pytest.approx(-14.6986, abs=0.01)
