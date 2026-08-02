"""Fetch Gaia DR3 NSS two-body ``source_id`` values for a diagnostics cohort.

Queries ``gaiadr3.nss_two_body_orbit`` via astroquery for IDs present in
``*_diagnostics.csv`` paths. Writes a CSV usable by
``validation.sb2_nss_cohort_report``.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.fetch_nss_source_ids \\
    --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \\
    --out-csv calibration/nss_two_body_source_ids.csv
"""

from __future__ import annotations

import argparse
import logging
import re
from glob import glob
from pathlib import Path

import pandas as pd

from darkhunter_rv.gaia_utils import execute_gaia_adql

_GAIA_RE = re.compile(r"Gaia_DR3_(\d{15,25})")
_LOG = logging.getLogger(__name__)


def collect_gaia_ids_from_diagnostics(diagnostics_glob: str) -> list[str]:
    ids: set[str] = set()
    for path_s in glob(diagnostics_glob):
        m = _GAIA_RE.search(Path(path_s).name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def query_nss_two_body_ids(source_ids: list[str], *, chunk_size: int = 100) -> list[dict]:
    """Return rows with source_id + nss_solution_type for matches in NSS two-body."""
    out: list[dict] = []
    for i in range(0, len(source_ids), chunk_size):
        chunk = source_ids[i : i + chunk_size]
        if not chunk:
            continue
        id_list = ", ".join(chunk)
        q = f"""
        SELECT source_id, nss_solution_type
        FROM gaiadr3.nss_two_body_orbit
        WHERE source_id IN ({id_list})
        """
        rows = execute_gaia_adql(q.strip(), f"NSS two-body chunk {i // chunk_size + 1}")
        for r in rows:
            sid = r.get("source_id")
            if sid is None:
                continue
            out.append(
                {
                    "source_id": str(int(sid)) if not isinstance(sid, str) else str(sid),
                    "nss_solution_type": str(r.get("nss_solution_type", "") or ""),
                }
            )
    # de-dupe by source_id (keep first type)
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in out:
        if r["source_id"] in seen:
            continue
        seen.add(r["source_id"])
        uniq.append(r)
    return uniq


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics-glob", required=True)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("calibration/nss_two_body_source_ids.csv"),
    )
    ap.add_argument("--chunk-size", type=int, default=100)
    args = ap.parse_args(argv)

    ids = collect_gaia_ids_from_diagnostics(args.diagnostics_glob)
    _LOG.info("Found %d unique Gaia IDs in diagnostics glob", len(ids))
    if not ids:
        _LOG.error("No Gaia IDs found")
        return 1
    rows = query_nss_two_body_ids(ids, chunk_size=max(1, int(args.chunk_size)))
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["source_id", "nss_solution_type"])
        _LOG.warning("No NSS two-body matches (empty catalog or query failure)")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"n_cohort={len(ids)} n_nss_match={len(df)} → {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
