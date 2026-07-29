#!/usr/bin/env python3
"""Rename HIRES Makee chip FITS files by Gaia DR3 ID from coordinates.

Groups ``[bir]<outfile>.<frameno>.fits`` triplets and cone-searches Gaia DR3.
The coordinate match is always the adopted source_id. ``TARGNAME`` is only a
confirmation when it equals a truncated form of that id (observatories truncate
target names); a mismatch does not override coordinates. Optional SIMBAD
resolution of non-Gaia ``TARGNAME`` values is logged the same way.

Writes ``GAIADR3ID`` and ``BJD`` headers, then renames to
``Gaia_DR3_<id>_{b|i|r}_epoch_<N>.fits``.

Default is dry-run; pass ``--apply`` to write headers and rename.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from darkhunter_rv.gaia_utils import execute_gaia_adql
from darkhunter_rv.io_utils import compute_hires_bjd, hires_photon_weighted_time

LOGGER = logging.getLogger("rename_hires_by_gaia")

MAKEE_CHIP_RE = re.compile(
    r"^(?P<chip>[bir])(?P<outfile>.+)\.(?P<frameno>\d+)\.fits$",
    re.IGNORECASE,
)
DIGIT_TARG_RE = re.compile(r"^(\d{8,19})$")
# Truncated Makee labels like GaiaDR3_2200363 (digits are a source_id prefix).
GAIA_DR3_PREFIX_TARG_RE = re.compile(
    r"^(?:Gaia[_\s]?DR3[_\s]?|GaiaDR3_)(\d{7,19})$",
    re.IGNORECASE,
)
GAIA_DR3_IN_IDS_RE = re.compile(r"Gaia\s*DR3\s+(\d{16,19})", re.IGNORECASE)


@dataclass(frozen=True)
class HiresChipFile:
    """One Makee chip FITS in a science triplet."""

    path: Path
    chip: str  # b | i | r
    outfile: str
    frameno: int


@dataclass
class HiresObservation:
    """One exposure = three chips sharing OUTFILE + FRAMENO."""

    outfile: str
    frameno: int
    chips: dict[str, HiresChipFile]
    ra_deg: float
    dec_deg: float
    mjd: float
    targname: str
    header_sample: fits.Header


def parse_makee_chip_name(path: Path) -> HiresChipFile | None:
    """Parse ``bj560.64.fits`` → chip/outfile/frameno, or None if not Makee-style."""
    m = MAKEE_CHIP_RE.match(path.name)
    if m is None:
        return None
    return HiresChipFile(
        path=path,
        chip=m.group("chip").lower(),
        outfile=m.group("outfile"),
        frameno=int(m.group("frameno")),
    )


def sexagesimal_ra_dec_to_deg(ra: str | float, dec: str | float) -> tuple[float, float]:
    """Convert FITS ``RA``/``DEC`` (sexagesimal or degrees) to ICRS degrees."""
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="icrs")
    return float(coord.ra.deg), float(coord.dec.deg)


def photon_weighted_mjd(header: fits.Header) -> float:
    """Photon-weighted MJD from header ``MJD``/``BJD`` (not DATE mid)."""
    return float(hires_photon_weighted_time(header).mjd)


def targname_is_digit_prefix(targname: str) -> str | None:
    """Return digit string if TARGNAME is a Gaia source_id prefix, else None.

    Accepts bare digit strings and truncated Makee labels ``GaiaDR3_<digits>``.
    """
    s = str(targname).strip()
    m = DIGIT_TARG_RE.match(s)
    if m:
        return m.group(1)
    m = GAIA_DR3_PREFIX_TARG_RE.match(s)
    return m.group(1) if m else None


def targname_matches_source_id(targname: str, source_id: int) -> bool:
    """True if TARGNAME digit prefix matches the start of ``source_id``."""
    prefix = targname_is_digit_prefix(targname)
    if prefix is None:
        return False
    return str(int(source_id)).startswith(prefix)


def simbad_gaia_dr3_id(targname: str) -> int | None:
    """
    Resolve ``TARGNAME`` via SIMBAD; return Gaia DR3 source_id if listed in IDS.

    Returns None if SIMBAD unavailable, object unknown, or no Gaia DR3 id.
    """
    try:
        from astroquery.simbad import Simbad
    except ImportError:
        LOGGER.warning("astroquery.simbad not installed; cannot resolve %r", targname)
        return None
    try:
        simbad = Simbad()
        simbad.add_votable_fields("ids")
        table = simbad.query_object(str(targname).strip())
    except Exception as exc:
        LOGGER.warning("SIMBAD query failed for %r: %s", targname, exc)
        return None
    if table is None or len(table) == 0:
        return None
    ids_val = table["ids"][0] if "ids" in table.colnames else table["IDS"][0]
    ids_str = " ".join(ids_val) if isinstance(ids_val, (list, tuple, np.ndarray)) else str(ids_val)
    m = GAIA_DR3_IN_IDS_RE.search(ids_str.replace("\n", " ").replace("|", " "))
    return int(m.group(1)) if m else None


def gaia_cone_search(
    ra_deg: float,
    dec_deg: float,
    *,
    radius_arcsec: float = 3.0,
) -> list[dict[str, Any]]:
    """Cone-search ``gaiadr3.gaia_source``; rows include source_id, ra, dec, phot_g_mean_mag, sep."""
    radius_deg = float(radius_arcsec) / 3600.0
    query = f"""
SELECT source_id, ra, dec, phot_g_mean_mag,
       DISTANCE(
         POINT('ICRS', {ra_deg}, {dec_deg}),
         POINT('ICRS', ra, dec)
       ) AS sep
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
  POINT('ICRS', ra, dec),
  CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
)
ORDER BY sep ASC
"""
    rows = execute_gaia_adql(query, "HIRES Gaia cone")
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            out.append(
                {
                    "source_id": int(row["source_id"]),
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "phot_g_mean_mag": (
                        float(row["phot_g_mean_mag"])
                        if row.get("phot_g_mean_mag") is not None
                        and np.isfinite(float(row["phot_g_mean_mag"]))
                        else np.inf
                    ),
                    "sep": float(row["sep"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def pick_gaia_match(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Nearest neighbor; tie-break by brightest G mag."""
    if not rows:
        return None
    min_sep = min(float(r["sep"]) for r in rows)
    near = [r for r in rows if abs(float(r["sep"]) - min_sep) < 1e-12]
    near.sort(key=lambda r: float(r.get("phot_g_mean_mag", np.inf)))
    return near[0]


def crosscheck_targname(
    targname: str,
    source_id: int,
    *,
    force: bool = False,
) -> tuple[bool, str]:
    """
    Annotate a coordinate-matched Gaia ID using optional ``TARGNAME`` confirmation.

    Coordinates always win: this always returns ``ok=True``. A digit / ``GaiaDR3_*``
    ``TARGNAME`` that is a prefix of ``source_id`` is confirmation (names are often
    length-limited). A mismatch is logged as unconfirmed, not a rejection.
    Non-prefix names may be checked via SIMBAD for the same confirm/unconfirmed
    messaging. ``force`` is retained for CLI compatibility and is unused.
    """
    del force  # CLI compatibility; TARGNAME never overrides coordinates.
    name = str(targname).strip()
    if not name:
        return True, "empty TARGNAME; coordinates only"
    if targname_is_digit_prefix(name) is not None:
        if targname_matches_source_id(name, source_id):
            return True, f"TARGNAME confirms truncated Gaia DR3 {source_id}"
        return True, (
            f"TARGNAME {name!r} does not match truncated Gaia DR3 {source_id} "
            "(using coordinates; TARGNAME may be truncated/wrong)"
        )
    sim_id = simbad_gaia_dr3_id(name)
    if sim_id is None:
        return True, (
            f"TARGNAME={name!r} not confirmed via SIMBAD "
            f"(using coordinates → Gaia DR3 {source_id})"
        )
    if int(sim_id) == int(source_id):
        return True, f"SIMBAD {name!r} confirms Gaia DR3 {source_id}"
    return True, (
        f"SIMBAD {name!r} → Gaia DR3 {sim_id}, coordinates → {source_id} "
        "(using coordinates)"
    )


def discover_makee_observations(directory: Path) -> list[HiresObservation]:
    """Scan directory for Makee chip FITS and group complete b/i/r triplets."""
    by_key: dict[tuple[str, int], dict[str, HiresChipFile]] = defaultdict(dict)
    for path in sorted(directory.glob("*.fits")):
        parsed = parse_makee_chip_name(path)
        if parsed is None:
            continue
        by_key[(parsed.outfile, parsed.frameno)][parsed.chip] = parsed

    observations: list[HiresObservation] = []
    for (outfile, frameno), chips in sorted(by_key.items()):
        missing = [c for c in ("b", "i", "r") if c not in chips]
        if missing:
            LOGGER.warning(
                "Incomplete triplet outfile=%s frameno=%s missing=%s; skip",
                outfile,
                frameno,
                missing,
            )
            continue
        with fits.open(chips["b"].path) as hdul:
            header = hdul[0].header.copy()
        ra_deg, dec_deg = sexagesimal_ra_dec_to_deg(header["RA"], header["DEC"])
        mjd = photon_weighted_mjd(header)
        observations.append(
            HiresObservation(
                outfile=outfile,
                frameno=frameno,
                chips=chips,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                mjd=mjd,
                targname=str(header.get("TARGNAME", "") or "").strip(),
                header_sample=header,
            )
        )
    return observations


def target_filename(source_id: int, chip: str, epoch: int) -> str:
    """``Gaia_DR3_<id>_{b|i|r}_epoch_<N>.fits``."""
    return f"Gaia_DR3_{int(source_id)}_{chip}_epoch_{int(epoch)}.fits"


def write_headers_and_rename(
    obs: HiresObservation,
    source_id: int,
    epoch: int,
    *,
    apply: bool,
) -> list[tuple[Path, Path]]:
    """
    Write ``GAIADR3ID`` + ``BJD`` on each chip and rename to Gaia epoch names.

    Returns list of (src, dest) pairs. With ``apply=False``, only reports planned moves.
    """
    gaia_name = f"Gaia DR3 {int(source_id)}"
    planned: list[tuple[Path, Path]] = []
    for chip in ("b", "i", "r"):
        src = obs.chips[chip].path
        dest = src.parent / target_filename(source_id, chip, epoch)
        planned.append((src, dest))
        if not apply:
            continue
        with fits.open(src, mode="update") as hdul:
            hdr = hdul[0].header
            hdr["GAIADR3ID"] = (gaia_name, "Gaia DR3 designation")
            # Always recompute from photon-weighted MJD when present (not DATE mid).
            bjd = compute_hires_bjd(hdr, obs.ra_deg, obs.dec_deg)
            hdr["BJD"] = (bjd, "Barycentric JD (TDB) photon-weighted")
            hdul.flush()
        if dest.exists() and dest.resolve() != src.resolve():
            raise FileExistsError(f"Refusing to overwrite existing {dest}")
        src.rename(dest)
    return planned


def resolve_observation(
    obs: HiresObservation,
    *,
    radius_arcsec: float,
    force: bool,
) -> tuple[int | None, str]:
    """Return (source_id, status message); source_id None if skipped."""
    rows = gaia_cone_search(obs.ra_deg, obs.dec_deg, radius_arcsec=radius_arcsec)
    match = pick_gaia_match(rows)
    if match is None:
        return None, (
            f"no Gaia match within {radius_arcsec}\" for "
            f"{obs.outfile}.{obs.frameno} at ({obs.ra_deg:.6f},{obs.dec_deg:.6f})"
        )
    source_id = int(match["source_id"])
    _, msg = crosscheck_targname(obs.targname, source_id, force=force)
    return source_id, (
        f"Gaia DR3 {source_id} sep={float(match['sep']) * 3600:.2f}\" "
        f"G={match['phot_g_mean_mag']}; {msg}"
    )


def run_rename(
    directory: Path,
    *,
    apply: bool,
    radius_arcsec: float,
    force: bool,
) -> int:
    """Process all Makee triplets under ``directory``. Return number of observations handled."""
    observations = discover_makee_observations(directory)
    if not observations:
        LOGGER.warning("No complete Makee triplets in %s", directory)
        return 0

    # Resolve Gaia ids first, then assign epochs per source_id by MJD order.
    resolved: list[tuple[HiresObservation, int, str]] = []
    for obs in observations:
        source_id, msg = resolve_observation(
            obs, radius_arcsec=radius_arcsec, force=force
        )
        if source_id is None:
            LOGGER.error("SKIP %s.%s: %s", obs.outfile, obs.frameno, msg)
            continue
        LOGGER.info("MATCH %s.%s: %s", obs.outfile, obs.frameno, msg)
        resolved.append((obs, source_id, msg))

    by_source: dict[int, list[HiresObservation]] = defaultdict(list)
    for obs, source_id, _ in resolved:
        by_source[source_id].append(obs)

    n_done = 0
    for source_id, obs_list in sorted(by_source.items()):
        obs_list.sort(key=lambda o: o.mjd)
        for epoch, obs in enumerate(obs_list, start=1):
            planned = write_headers_and_rename(
                obs, source_id, epoch, apply=apply
            )
            mode = "APPLY" if apply else "DRY-RUN"
            for src, dest in planned:
                LOGGER.info("%s: %s -> %s", mode, src.name, dest.name)
            n_done += 1
    return n_done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("/Users/rfoley/darkhunter/rvs/data/hires"),
        help="Directory containing Makee HIRES chip FITS",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write headers and rename (default is dry-run)",
    )
    parser.add_argument(
        "--radius-arcsec",
        type=float,
        default=3.0,
        help="Gaia cone-search radius in arcseconds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deprecated no-op: coordinates always define the Gaia ID",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    directory = args.dir.expanduser().resolve()
    if not directory.is_dir():
        LOGGER.error("Not a directory: %s", directory)
        return 2
    n = run_rename(
        directory,
        apply=bool(args.apply),
        radius_arcsec=float(args.radius_arcsec),
        force=bool(args.force),
    )
    LOGGER.info("Handled %d observation(s); apply=%s", n, bool(args.apply))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
