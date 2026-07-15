"""Manual literature RVs: canonical CSV merged into summary [EXTERNAL RV DATA]."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

from . import config

LITERATURE_TELESCOPE_PREFIX = "LITERATURE_"

CSV_FIELDS = (
    "gaia_dr3_id",
    "telescope",
    "mjd",
    "rv_kms",
    "rv_err_kms",
    "flag",
)

_DEFAULT_PATH = config.REPO_ROOT / "calibration" / "manual_literature_rvs.csv"


def default_manual_literature_path() -> Path:
    env = os.environ.get("DARKHUNTER_MANUAL_LITERATURE_RVS")
    if env:
        return Path(env)
    return _DEFAULT_PATH


def _sid_key(gaia_id: int | str | None) -> str | None:
    if gaia_id is None:
        return None
    try:
        return str(int(gaia_id))
    except (TypeError, ValueError):
        s = str(gaia_id).strip()
        return s or None


def _finite(val: Any) -> float | None:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _row_to_external(row: dict[str, str]) -> dict[str, Any] | None:
    tele = str(row.get("telescope", "") or "").strip()
    mjd = _finite(row.get("mjd"))
    rv = _finite(row.get("rv_kms"))
    rv_err = _finite(row.get("rv_err_kms"))
    if not tele or mjd is None or rv is None:
        return None
    if rv_err is None or rv_err <= 0:
        rv_err = 1.0
    flag = str(row.get("flag", "") or "").strip()
    return {
        "telescope": tele,
        "mjd": mjd,
        "rv": rv,
        "rv_err": rv_err,
        "flag": flag,
    }


def load_manual_literature_rows(
    path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load CSV keyed by Gaia DR3 id → EXTERNAL-shaped rows."""
    p = Path(path) if path is not None else default_manual_literature_path()
    out: dict[str, list[dict[str, Any]]] = {}
    if not p.is_file():
        return out
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            sid = _sid_key(raw.get("gaia_dr3_id"))
            if sid is None:
                continue
            ext = _row_to_external(raw)
            if ext is None:
                continue
            out.setdefault(sid, []).append(ext)
    for sid in out:
        out[sid].sort(key=lambda r: (float(r["mjd"]), str(r["telescope"])))
    return out


def rows_for_gaia_id(
    gaia_id: int | str | None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    sid = _sid_key(gaia_id)
    if sid is None:
        return []
    return list(load_manual_literature_rows(path).get(sid, []))


def merge_manual_literature(
    gaia_id: int | str | None,
    existing_external: list | None,
    *,
    path: Path | None = None,
) -> list:
    """
    If the CSV has rows for ``gaia_id``, replace all ``LITERATURE_*`` EXTERNAL
    rows with the CSV set and keep other catalogs. If none, return existing unchanged.
    """
    existing = list(existing_external or [])
    lit = rows_for_gaia_id(gaia_id, path=path)
    if not lit:
        return existing
    # Lazy import avoids circular dependency with gaia_utils.
    from .gaia_utils import merge_external_rv_lists

    return merge_external_rv_lists(
        existing,
        lit,
        replace_prefixes=(LITERATURE_TELESCOPE_PREFIX,),
    )


def parse_manual_epoch_table(
    text: str,
    *,
    gaia_id: int | str,
    default_rv_err: float = 0.6,
) -> list[dict[str, str]]:
    """
    Parse whitespace epoch lines like::

        lit_1 LITERATURE_Griffin1994 39382.12000000 3.20000000 nan nan

    Columns: label, telescope, mjd, rv, rv_err, rms (rms ignored).
    Non-finite ``rv_err`` is replaced with ``default_rv_err``.
    """
    sid = _sid_key(gaia_id)
    if sid is None:
        raise ValueError(f"invalid gaia_id: {gaia_id!r}")
    rows: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        # Skip header-like lines
        if parts[0].lower() in ("input", "#") or "instrument" in line.lower() and "mjd" in line.lower():
            continue
        tele = parts[1] if len(parts) >= 5 else parts[0]
        # lit_N TELE MJD RV [err] [rms]  OR  TELE MJD RV ERR [flag...]
        if str(parts[0]).lower().startswith("lit_") or (
            len(parts) >= 5 and _finite(parts[2]) is not None and _finite(parts[3]) is not None
        ):
            if len(parts) < 4:
                continue
            tele = parts[1]
            mjd_i, rv_i, err_i = 2, 3, 4
        else:
            tele = parts[0]
            mjd_i, rv_i, err_i = 1, 2, 3
        mjd = _finite(parts[mjd_i])
        rv = _finite(parts[rv_i])
        if mjd is None or rv is None:
            continue
        rv_err = _finite(parts[err_i]) if len(parts) > err_i else None
        if rv_err is None or rv_err <= 0:
            rv_err = float(default_rv_err)
        flag = tele.replace(LITERATURE_TELESCOPE_PREFIX, "") if tele.startswith(LITERATURE_TELESCOPE_PREFIX) else tele
        rows.append(
            {
                "gaia_dr3_id": sid,
                "telescope": tele,
                "mjd": f"{mjd:.8f}",
                "rv_kms": f"{rv:.8f}",
                "rv_err_kms": f"{rv_err:.8f}",
                "flag": flag,
            }
        )
    return rows


def upsert_manual_literature_csv(
    new_rows: list[dict[str, str]],
    path: Path | None = None,
    *,
    replace_gaia_telescopes: bool = True,
) -> Path:
    """
    Write/update the canonical CSV. When ``replace_gaia_telescopes`` is True,
    drop existing rows matching any (gaia_dr3_id, telescope) present in ``new_rows``.
    """
    p = Path(path) if path is not None else default_manual_literature_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if p.is_file():
        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                existing.append({k: str(raw.get(k, "") or "") for k in CSV_FIELDS})

    drop_keys: set[tuple[str, str]] = set()
    if replace_gaia_telescopes:
        for r in new_rows:
            sid = _sid_key(r.get("gaia_dr3_id"))
            tele = str(r.get("telescope", "") or "").strip()
            if sid and tele:
                drop_keys.add((sid, tele))

    kept = [
        r
        for r in existing
        if (_sid_key(r.get("gaia_dr3_id")), str(r.get("telescope", "") or "").strip()) not in drop_keys
    ]
    normalized: list[dict[str, str]] = []
    for r in new_rows:
        sid = _sid_key(r.get("gaia_dr3_id"))
        if sid is None:
            continue
        normalized.append(
            {
                "gaia_dr3_id": sid,
                "telescope": str(r.get("telescope", "") or "").strip(),
                "mjd": str(r.get("mjd", "")),
                "rv_kms": str(r.get("rv_kms", "")),
                "rv_err_kms": str(r.get("rv_err_kms", "")),
                "flag": str(r.get("flag", "") or ""),
            }
        )
    kept.extend(normalized)
    kept.sort(
        key=lambda r: (
            r["gaia_dr3_id"],
            float(r["mjd"]) if _finite(r["mjd"]) is not None else 0.0,
            r["telescope"],
        )
    )
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    return p
