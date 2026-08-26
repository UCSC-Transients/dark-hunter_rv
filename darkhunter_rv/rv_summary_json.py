"""Machine-readable star summary JSON for dark-hunter_pop CandidateRecord.rv_summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from darkhunter_rv import config
from darkhunter_rv.gaia_utils import (
    merge_manual_literature,
    normalize_parsed_star_metadata,
    parse_external_rvs_from_star_summary,
    parse_gaia_metadata_from_star_summary,
)
from darkhunter_rv.thiele_innes_inclination import thiele_innes_from_metadata

logger = logging.getLogger(__name__)

RV_SUMMARY_SCHEMA_VERSION = 1

_NSS_ORBITAL_KEYS: tuple[tuple[str, str], ...] = (
    ("period_day", "Period"),
    ("period_day_err", "Period_Error"),
    ("eccentricity", "Eccentricity"),
    ("eccentricity_err", "Eccentricity_Error"),
    ("t_periastron_day", "T_Periastron"),
    ("t_periastron_day_err", "T_Periastron_Error"),
    ("mass_ratio", "Mass_Ratio"),
    ("mass_ratio_err", "Mass_Ratio_Error"),
    ("center_mass_velocity_kms", "Center_Mass_Velocity"),
    ("center_mass_velocity_kms_err", "Center_Mass_Velocity_Error"),
    ("semi_amp_primary_kms", "Semi_Amp_Primary"),
    ("semi_amp_primary_kms_err", "Semi_Amp_Primary_Error"),
    ("semi_amp_secondary_kms", "Semi_Amp_Secondary"),
    ("semi_amp_secondary_kms_err", "Semi_Amp_Secondary_Error"),
    ("inclination_deg", "Inclination"),
    ("inclination_deg_err", "Inclination_Error"),
    ("arg_periastron_deg", "Arg_Periastron"),
    ("arg_periastron_deg_err", "Arg_Periastron_Error"),
)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        xf = float(value)
        return xf if np.isfinite(xf) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _meta_float(meta: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in meta:
            continue
        val = meta[key]
        if val is None:
            continue
        try:
            xf = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(xf):
            return xf
    return None


def _pipeline_telescope_from_filename(name: str) -> str:
    s = name.lower()
    if "kpf" in s:
        return "KPF"
    if "ghost" in s:
        return "GHOST"
    if "maroon" in s:
        return "MAROON-X"
    return "APF"


def _nss_solution_type(meta: Mapping[str, Any] | None) -> str | None:
    if not meta:
        return None
    raw = meta.get("NSS_Solution_Type", meta.get("nss_solution_type"))
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in ("none", "nan", "null"):
        return None
    return text


def _nss_orbital_block(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    out: dict[str, Any] = {}
    for out_key, meta_key in _NSS_ORBITAL_KEYS:
        val = _meta_float(meta, meta_key, out_key)
        if val is not None:
            out[out_key] = val
    return out


def _thiele_innes_block(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    ti = thiele_innes_from_metadata(meta)
    if ti is None:
        return {}
    return {
        "A": float(ti.A) if np.isfinite(ti.A) else None,
        "B": float(ti.B) if np.isfinite(ti.B) else None,
        "F": float(ti.F) if np.isfinite(ti.F) else None,
        "G": float(ti.G) if np.isfinite(ti.G) else None,
        "A_err": float(ti.A_err) if np.isfinite(ti.A_err) else None,
        "B_err": float(ti.B_err) if np.isfinite(ti.B_err) else None,
        "F_err": float(ti.F_err) if np.isfinite(ti.F_err) else None,
        "G_err": float(ti.G_err) if np.isfinite(ti.G_err) else None,
    }


def _format_external_rv_rows(external_rvs: list) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ext in external_rvs:
        try:
            rows.append(
                {
                    "telescope": str(ext.get("telescope", "")),
                    "mjd": float(ext["mjd"]),
                    "rv_kms": float(ext["rv"]),
                    "rv_err_kms": float(ext["rv_err"]),
                    "flag": str(ext.get("flag", "") or ""),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _format_pipeline_epochs(merged_lines: Mapping[str, str]) -> list[dict[str, Any]]:
    epochs: list[dict[str, Any]] = []
    for basename, line in merged_lines.items():
        parts = line.split()
        if len(parts) < 5:
            continue
        fallback = False
        if len(parts) >= 6 and parts[-1] in ("True", "False"):
            fallback = parts[-1] == "True"
            parts = parts[:-1]
        if len(parts) < 5:
            continue
        try:
            epochs.append(
                {
                    "file": parts[0],
                    "basename": basename,
                    "mjd": float(parts[1]),
                    "rv_kms": float(parts[2]),
                    "rv_err_kms": float(parts[3]),
                    "wrms_kms": float(parts[4]),
                    "fallback": fallback,
                    "telescope": _pipeline_telescope_from_filename(parts[0]),
                }
            )
        except ValueError:
            continue
    return epochs


def _collect_instruments(
    pipeline_epochs: list[dict[str, Any]],
    external_rvs: list[dict[str, Any]],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in pipeline_epochs + external_rvs:
        tel = str(row.get("telescope", "") or "").strip()
        if not tel or tel in seen:
            continue
        seen.add(tel)
        ordered.append(tel)
    return ordered


def build_rv_summary_dict(
    obj_id: str | int,
    *,
    gaia_data: Mapping[str, Any] | None,
    merged_pipeline_lines: Mapping[str, str],
    summary_txt_name: str | None = None,
    joker_fit: Mapping[str, Any] | None = None,
    joker_fit_path: str | None = None,
) -> dict[str, Any]:
    """Build the pop-facing JSON summary for one Gaia source."""
    sid = int(obj_id)
    meta_raw = gaia_data.get("metadata") if gaia_data else None
    meta = normalize_parsed_star_metadata(meta_raw) if meta_raw else None
    external_raw: list = []
    if gaia_data and gaia_data.get("external_rvs"):
        external_raw = list(gaia_data["external_rvs"])
    external_raw = merge_manual_literature(sid, external_raw)
    external_rows = _format_external_rv_rows(external_raw)
    pipeline_epochs = _format_pipeline_epochs(merged_pipeline_lines)
    instruments = _collect_instruments(pipeline_epochs, external_rows)
    n_pipeline = len(pipeline_epochs)
    n_external = len(external_rows)

    out: dict[str, Any] = {
        "schema_version": RV_SUMMARY_SCHEMA_VERSION,
        "source_id": sid,
        "summary_txt": summary_txt_name or f"Gaia_DR3_{sid}_summary.txt",
        "instruments": instruments,
        "n_epochs": n_pipeline + n_external,
        "n_pipeline_epochs": n_pipeline,
        "n_external_epochs": n_external,
        "gaia_metadata": _json_safe(meta) if meta else {},
        "nss_solution_type": _nss_solution_type(meta),
        "nss_orbital": _json_safe(_nss_orbital_block(meta)),
        "thiele_innes": _json_safe(_thiele_innes_block(meta)),
        "external_rvs": external_rows,
        "pipeline_epochs": pipeline_epochs,
    }
    if joker_fit is not None:
        out["joker_fit"] = _json_safe(dict(joker_fit))
    if joker_fit_path:
        out["joker_fit_path"] = joker_fit_path
    return out


def rv_summary_json_path(obj_id: str | int, output_dir: Path | None = None) -> Path:
    root = output_dir or config.OUTPUT_DIR
    return Path(root) / f"Gaia_DR3_{int(obj_id)}_summary.json"


def write_rv_summary_json(
    obj_id: str | int,
    gaia_data: Mapping[str, Any] | None,
    merged_pipeline_lines: Mapping[str, str],
    *,
    output_dir: Path | None = None,
    joker_fit: Mapping[str, Any] | None = None,
    joker_fit_path: str | None = None,
) -> Path:
    """Write Gaia_DR3_<id>_summary.json next to the sectioned summary.txt."""
    out_path = rv_summary_json_path(obj_id, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_rv_summary_dict(
        obj_id,
        gaia_data=gaia_data,
        merged_pipeline_lines=merged_pipeline_lines,
        summary_txt_name=out_path.with_suffix(".txt").name,
        joker_fit=joker_fit,
        joker_fit_path=joker_fit_path,
    )
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    logger.info("RV summary JSON written to %s (n_epochs=%s)", out_path, payload["n_epochs"])
    return out_path


def load_rv_summary_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def rv_summary_from_star_summary_txt(
    summary_path: Path,
    *,
    joker_fit_path: Path | None = None,
) -> dict[str, Any]:
    """Rebuild JSON from an on-disk sectioned summary (backfill / joker refresh)."""
    from darkhunter_rv.io_utils import _parse_star_summary_pipeline_lines

    summary_path = Path(summary_path)
    obj_id = summary_path.stem.replace("Gaia_DR3_", "").replace("_summary", "")
    merged = _parse_star_summary_pipeline_lines(summary_path.read_text(encoding="utf-8"))
    meta = parse_gaia_metadata_from_star_summary(summary_path)
    external = parse_external_rvs_from_star_summary(summary_path)
    external = merge_manual_literature(int(obj_id), external)
    gaia_data = {"metadata": meta, "external_rvs": external} if meta or external else None

    joker_fit: dict[str, Any] | None = None
    joker_fit_path_str: str | None = None
    if joker_fit_path is not None and Path(joker_fit_path).is_file():
        joker_fit_path_str = str(joker_fit_path)
        try:
            loaded = json.loads(Path(joker_fit_path).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                joker_fit = loaded
        except json.JSONDecodeError:
            joker_fit = None

    return build_rv_summary_dict(
        obj_id,
        gaia_data=gaia_data,
        merged_pipeline_lines=merged,
        summary_txt_name=summary_path.name,
        joker_fit=joker_fit,
        joker_fit_path=joker_fit_path_str,
    )


def refresh_rv_summary_json_from_txt(
    summary_path: Path,
    *,
    joker_fit_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Rewrite JSON alongside an existing summary.txt."""
    summary_path = Path(summary_path)
    obj_id = summary_path.stem.replace("Gaia_DR3_", "").replace("_summary", "")
    payload = rv_summary_from_star_summary_txt(summary_path, joker_fit_path=joker_fit_path)
    out_path = rv_summary_json_path(obj_id, output_dir or summary_path.parent)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out_path


def attach_joker_fit_to_summary_json(
    summary_path: Path,
    joker_report: Mapping[str, Any],
    *,
    joker_fit_path: Path | None = None,
) -> Path:
    """Merge a Joker fit report into the star's summary JSON."""
    summary_path = Path(summary_path)
    json_path = rv_summary_json_path(
        summary_path.stem.replace("Gaia_DR3_", "").replace("_summary", ""),
        summary_path.parent,
    )
    if json_path.is_file():
        payload = load_rv_summary_json(json_path)
    else:
        payload = rv_summary_from_star_summary_txt(summary_path, joker_fit_path=joker_fit_path)
    payload["joker_fit"] = _json_safe(dict(joker_report))
    if joker_fit_path is not None:
        payload["joker_fit_path"] = str(joker_fit_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return json_path
