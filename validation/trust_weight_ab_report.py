"""Offline trust-weight A/B: re-stack mask_ccf chunks from diagnostics.

Compares IVW-only vs trust-scaled IVW (same formulas as ``darkhunter_rv.qc``)
without re-running the pipeline. Uses existing ``*_diagnostics.csv`` chunk rows.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.trust_weight_ab_report \
    --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \
    --out-dir validation_output/trust_ab_post103 \
    --teff-max 5000
"""

from __future__ import annotations

import argparse
import json
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from darkhunter_rv import qc
from validation.chunk_calibration import relative_pair_table, summarize_relative_gate


def _finite_mask(rv: np.ndarray, err: np.ndarray) -> np.ndarray:
    return np.isfinite(rv) & np.isfinite(err) & (err > 0)


def _gaia_id_from_stem(stem: str) -> str:
    m = re.search(r"Gaia_DR3_(\d+)", stem)
    return m.group(1) if m else stem


def _ccf_snr_series(chunk_df: pd.DataFrame) -> pd.Series:
    """Prefer ``ccf_peak_snr`` (pipeline); fall back to ``ccf_peak`` for synthetic tests."""
    if "ccf_peak_snr" in chunk_df.columns:
        return pd.to_numeric(chunk_df["ccf_peak_snr"], errors="coerce")
    if "ccf_peak" in chunk_df.columns:
        return pd.to_numeric(chunk_df["ccf_peak"], errors="coerce")
    return pd.Series(np.nan, index=chunk_df.index)


def stack_exposure(
    chunk_df: pd.DataFrame,
    *,
    trust_cfg: dict,
    enabled: bool,
) -> dict:
    """IVW stack one exposure's mask_ccf chunk rows."""
    work = chunk_df.reset_index(drop=True)
    rv = pd.to_numeric(work["rv_kms"], errors="coerce").to_numpy(float)
    err = pd.to_numeric(work["rv_err_kms"], errors="coerce").to_numpy(float)
    ok = _finite_mask(rv, err)
    if "qc_pass" in work.columns:
        qc_ok = work["qc_pass"].map(
            lambda x: True if x is True or str(x).lower() in {"true", "1"} else False
        ).to_numpy(bool)
        ok = ok & qc_ok
    rv = rv[ok]
    err = err[ok]
    if rv.size < 1:
        return {
            "rv_kms": float("nan"),
            "err_kms": float("nan"),
            "n_chunks": 0,
            "median_abs_resid": float("nan"),
        }
    robust = float(np.median(rv))
    work_ok = work.loc[ok].reset_index(drop=True)
    tel = (
        pd.to_numeric(work_ok["telluric_fraction"], errors="coerce").to_numpy(float)
        if "telluric_fraction" in work_ok.columns
        else np.full(rv.size, np.nan)
    )
    peak = _ccf_snr_series(work_ok).to_numpy(float)
    asym = (
        pd.to_numeric(work_ok["ccf_asymmetry"], errors="coerce").to_numpy(float)
        if "ccf_asymmetry" in work_ok.columns
        else np.full(rv.size, np.nan)
    )
    trusts = []
    for i in range(rv.size):
        comps = qc.chunk_trust_components(
            rv_kms=float(rv[i]),
            robust_mean_kms=robust,
            telluric_fraction=float(tel[i]) if i < tel.size else float("nan"),
            ccf_peak_snr=float(peak[i]) if i < peak.size else float("nan"),
            ccf_asymmetry=float(asym[i]) if i < asym.size else float("nan"),
            cfg=trust_cfg,
        )
        trusts.append(comps["trust_weight"])
    tw = np.asarray(trusts, float)
    w = qc.ivw_weights_with_trust(err, tw, enabled=enabled)
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0:
        return {
            "rv_kms": float("nan"),
            "err_kms": float("nan"),
            "n_chunks": int(rv.size),
            "median_abs_resid": float(np.median(np.abs(rv - robust))),
        }
    rv_hat = float(np.sum(w * rv) / wsum)
    err_hat = float(1.0 / np.sqrt(wsum))
    return {
        "rv_kms": rv_hat,
        "err_kms": err_hat,
        "n_chunks": int(rv.size),
        "median_abs_resid": float(np.median(np.abs(rv - robust))),
    }


def _exposure_teff(df: pd.DataFrame) -> float:
    if "teff" not in df.columns:
        return float("nan")
    series = pd.to_numeric(df["teff"], errors="coerce").dropna()
    return float(series.iloc[0]) if not series.empty else float("nan")


def _exposure_mjd(df: pd.DataFrame) -> float:
    if "mjd" not in df.columns:
        return float("nan")
    series = pd.to_numeric(df["mjd"], errors="coerce").dropna()
    return float(series.iloc[0]) if not series.empty else float("nan")


def run_ab(
    diagnostics_glob: str,
    *,
    qc_config: Path,
    max_files: int | None = None,
    teff_max: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    trust_cfg = qc.load_trust_weights_config(qc_config)
    paths = sorted(glob(diagnostics_glob))
    if max_files is not None and max_files > 0:
        paths = paths[: int(max_files)]
    rows: list[dict] = []
    for path_s in paths:
        path = Path(path_s)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "method" not in df.columns:
            continue
        teff = _exposure_teff(df)
        if teff_max is not None and np.isfinite(teff_max):
            if not np.isfinite(teff) or teff >= float(teff_max):
                continue
        mask = df[df["method"].astype(str) == "mask_ccf"].copy()
        if "chunk_key" in mask.columns:
            mask = mask[mask["chunk_key"].astype(str) != "all"]
        if mask.empty:
            continue
        off = stack_exposure(mask, trust_cfg=trust_cfg, enabled=False)
        on = stack_exposure(mask, trust_cfg=trust_cfg, enabled=True)
        stem = path.name.replace("_diagnostics.csv", "")
        rows.append(
            {
                "diagnostics_path": str(path),
                "stem": stem,
                "gaia_dr3_id": _gaia_id_from_stem(stem),
                "mjd": _exposure_mjd(df),
                "teff": teff,
                "n_chunks": off["n_chunks"],
                "rv_off_kms": off["rv_kms"],
                "err_off_kms": off["err_kms"],
                "rv_on_kms": on["rv_kms"],
                "err_on_kms": on["err_kms"],
                "delta_rv_kms": (
                    on["rv_kms"] - off["rv_kms"]
                    if np.isfinite(on["rv_kms"]) and np.isfinite(off["rv_kms"])
                    else float("nan")
                ),
                "err_ratio_on_over_off": (
                    on["err_kms"] / off["err_kms"]
                    if np.isfinite(on["err_kms"])
                    and np.isfinite(off["err_kms"])
                    and off["err_kms"] > 0
                    else float("nan")
                ),
            }
        )
    per = pd.DataFrame(rows)
    summary: dict = {
        "n_exposures": int(len(per)),
        "teff_max": teff_max,
        "median_err_off_kms": float(per["err_off_kms"].median()) if len(per) else float("nan"),
        "median_err_on_kms": float(per["err_on_kms"].median()) if len(per) else float("nan"),
        "median_err_ratio_on_over_off": float(per["err_ratio_on_over_off"].median())
        if len(per)
        else float("nan"),
        "median_abs_delta_rv_kms": float(per["delta_rv_kms"].abs().median())
        if len(per)
        else float("nan"),
        "qc_config": str(qc_config),
    }
    if len(per) >= 2 and "gaia_dr3_id" in per.columns:
        for label, rv_col, err_col in (
            ("off", "rv_off_kms", "err_off_kms"),
            ("on", "rv_on_kms", "err_on_kms"),
        ):
            epochs = per.rename(columns={rv_col: "rv_stack_kms", err_col: "err_stack_kms"})[
                ["gaia_dr3_id", "stem", "mjd", "rv_stack_kms", "err_stack_kms"]
            ].copy()
            epochs["file"] = epochs["stem"].astype(str)
            pairs = relative_pair_table(
                epochs, rv_col="rv_stack_kms", err_col="err_stack_kms", pair_window_days=7.0
            )
            gate = summarize_relative_gate(pairs, goal_kms=0.1)
            summary[f"relative_gate_{label}"] = gate
    return per, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics-glob", required=True)
    ap.add_argument("--qc-config", type=Path, default=Path("order_chunk_qc.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("validation_output/trust_ab"))
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--teff-max",
        type=float,
        default=None,
        help="Keep exposures with Teff < this (cool subset); default = no cut",
    )
    args = ap.parse_args(argv)

    per, summary = run_ab(
        args.diagnostics_glob,
        qc_config=args.qc_config,
        max_files=args.max_files or None,
        teff_max=args.teff_max,
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_exposure_ab.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
