#!/usr/bin/env python3
"""
Diagnostic CLI: spectrum-as-mask epoch CCF pages (PDF + pairs CSV).

Production matrix / abs fill / cascade enrich live in ``validation.epoch_ccf_matrix``
(default ``--engine mask``). This module keeps the multi-page CCF PDF viewer.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.epoch_mask_ccf_matrix \\
    --gaia-id 1125337614021152256 \\
    --data-root /Users/rfoley/darkhunter/rvs/data \\
    --abs-fill-csv validation_output/epoch_ccf/1125337614021152256/epoch_ccf_abs_fill.csv \\
    --out-dir validation_output/epoch_mask_ccf/1125337614021152256
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config, io_utils, qc
from darkhunter_rv.blaze import BlazeCalibration
from darkhunter_rv.epoch_mask_ccf import (
    DEFAULT_AUTO_SMOOTH_SIGMA,
    ChunkNorm,
    PairResult,
    pair_spectrum_mask_ccf,
    prepare_epoch_chunks,
    spectrum_mask_feature_count,
    spectrum_mask_from_norm,
)
from darkhunter_rv.instruments import get_instrument_profile

logger = logging.getLogger(__name__)

_EPOCH_RE = re.compile(r"^Gaia_DR3_(\d+)_epoch_(\d+)\.txt$")

# Re-export library helpers for older tests / notebooks.
__all__ = [
    "ChunkNorm",
    "PairResult",
    "DEFAULT_AUTO_SMOOTH_SIGMA",
    "discover_epoch_spectra",
    "pair_spectrum_mask_ccf",
    "prepare_epoch_chunks",
    "spectrum_mask_feature_count",
    "spectrum_mask_from_norm",
    "run_matrix",
]


def discover_epoch_spectra(data_root: Path, gaia_id: str) -> list[tuple[int, Path]]:
    """Find ``Gaia_DR3_<id>_epoch_*.txt`` under ``data_root``, sorted by epoch index."""
    gid = str(gaia_id).strip()
    out: list[tuple[int, Path]] = []
    for path in sorted(data_root.glob(f"Gaia_DR3_{gid}_epoch_*.txt")):
        m = _EPOCH_RE.match(path.name)
        if not m or m.group(1) != gid:
            continue
        out.append((int(m.group(2)), path))
    out.sort(key=lambda t: t[0])
    return out


def run_matrix(
    *,
    gaia_id: str,
    data_root: Path,
    out_dir: Path,
    abs_fill_csv: Path | None = None,
    instrument_name: str = "APF",
    continuum_mode: str | None = None,
    blaze_calibration: Path | None = None,
    qc_config: Path | None = None,
    auto_smooth_sigma: float = DEFAULT_AUTO_SMOOTH_SIGMA,
    max_chunk_err_kms: float = 50.0,
    no_bias: bool = False,
    write_pdf: bool = True,
) -> pd.DataFrame:
    """Compute all ordered pairs; write CSV (+ optional multi-page PDF)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = discover_epoch_spectra(Path(data_root), gaia_id)
    if not epochs:
        raise FileNotFoundError(f"No epochs for Gaia {gaia_id} under {data_root}")

    instrument = get_instrument_profile(instrument_name)
    bias: dict = {}
    if not no_bias and instrument.bias_file:
        bias = io_utils.read_bias(instrument.bias_file)
        logger.info("Loaded bias table with %d keys from %s", len(bias), instrument.bias_file)

    mode = str(continuum_mode) if continuum_mode is not None else str(config.MASK_CONTINUUM_MODE)
    blaze_path = (
        Path(blaze_calibration)
        if blaze_calibration is not None
        else Path(config.BLAZE_CALIBRATION_FILE)
    )
    blaze_cal: BlazeCalibration | None = None
    if mode in ("sinc_blaze", "sinc_blaze_only") and blaze_path.is_file():
        blaze_cal = BlazeCalibration.load(blaze_path)
        logger.info("Loaded blaze calibration from %s (continuum_mode=%s)", blaze_path, mode)
    elif mode in ("sinc_blaze", "sinc_blaze_only"):
        logger.warning("Blaze file missing (%s); continuum will fall back to spline per order", blaze_path)

    qc_path = Path(qc_config) if qc_config is not None else Path("order_chunk_qc.yaml")
    qc_thresholds = qc.load_qc_config(qc_path, instrument.name)

    abs_by_ep: dict[int, float] = {}
    if abs_fill_csv is not None and Path(abs_fill_csv).is_file():
        fill = pd.read_csv(abs_fill_csv)
        if "epoch" in fill.columns and "abs_rv_kms" in fill.columns:
            abs_by_ep = {
                int(r.epoch): float(r.abs_rv_kms)
                for _, r in fill.iterrows()
                if np.isfinite(float(r.abs_rv_kms))
            }

    prepared: dict[int, list[ChunkNorm]] = {}
    for ep, path in epochs:
        t0 = time.time()
        prepared[ep] = prepare_epoch_chunks(
            path,
            instrument_name=instrument_name,
            continuum_mode=mode,
            blaze_cal=blaze_cal,
        )
        logger.info("  epoch %s: %d chunks (%.1fs)", ep, len(prepared[ep]), time.time() - t0)

    rows: list[dict] = []
    results: list[PairResult] = []
    ep_list = [e for e, _ in epochs]
    n_pairs = len(ep_list) ** 2
    done = 0
    t_all = time.time()
    for ex in ep_list:
        for ey in ep_list:
            done += 1
            t0 = time.time()
            res = pair_spectrum_mask_ccf(
                prepared[ex],
                prepared[ey],
                epoch_x=ex,
                epoch_y=ey,
                bias=bias,
                auto_smooth_sigma=auto_smooth_sigma,
                max_chunk_err_kms=float(max_chunk_err_kms),
                qc_thresholds=qc_thresholds,
            )
            results.append(res)
            dv_abs = float("nan")
            resid = float("nan")
            if ex in abs_by_ep and ey in abs_by_ep:
                dv_abs = abs_by_ep[ex] - abs_by_ep[ey]
                if np.isfinite(res.dv_kms):
                    resid = float(res.dv_kms) - dv_abs
            rows.append(
                {
                    "gaia_id": str(gaia_id),
                    "epoch_x": res.epoch_x,
                    "epoch_y": res.epoch_y,
                    "dv_kms": res.dv_kms,
                    "err_kms": res.err_kms,
                    "n_chunks": res.n_chunks,
                    "n_chunks_raw": res.n_chunks_raw,
                    "n_chunks_clipped": res.n_chunks_clipped,
                    "auto_correlation": res.auto_correlation,
                    "auto_smooth_sigma": float(auto_smooth_sigma) if res.auto_correlation else 0.0,
                    "lag_sample_kms": res.lag_sample_kms,
                    "peak_from_stack_kms": res.peak_from_stack_kms,
                    "dv_abs_kms": dv_abs,
                    "residual_kms": resid,
                }
            )
            logger.info(
                "  [%d/%d] %d vs %d  dv=%+.3f  n=%d  (%.1fs)",
                done,
                n_pairs,
                ex,
                ey,
                res.dv_kms,
                res.n_chunks,
                time.time() - t0,
            )

    pairs_df = pd.DataFrame(rows)
    pairs_csv = out_dir / "epoch_mask_ccf_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)

    meta = {
        "gaia_id": str(gaia_id),
        "n_epochs": len(ep_list),
        "epoch_indices": ep_list,
        "continuum_mode": mode,
        "chunk_layout": str(config.DEFAULT_CHUNK_LAYOUT) if config.DEFAULT_CHUNK_LAYOUT else None,
        "blaze_calibration": str(blaze_path) if blaze_cal is not None else None,
        "qc_config": str(qc_path),
        "auto_smooth_sigma": float(auto_smooth_sigma),
        "no_bias": bool(no_bias),
        "elapsed_s": float(time.time() - t_all),
        "pairs_csv": str(pairs_csv),
        "note": "Diagnostic PDF CLI; production fill uses validation.epoch_ccf_matrix --engine mask",
    }
    (out_dir / "epoch_mask_ccf_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if write_pdf:
        pdf_path = out_dir / "epoch_mask_ccf_pages.pdf"
        _write_ccf_pdf(results, abs_by_ep, pdf_path, gaia_id=str(gaia_id))
        meta["pdf"] = str(pdf_path)
        (out_dir / "epoch_mask_ccf_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        logger.info("Wrote PDF %s", pdf_path)

    logger.info("Wrote %s (%.1fs total)", pairs_csv, time.time() - t_all)
    return pairs_df


def _write_ccf_pdf(
    results: list[PairResult],
    abs_by_ep: dict[int, float],
    pdf_path: Path,
    *,
    gaia_id: str,
) -> None:
    with PdfPages(pdf_path) as pdf:
        for res in results:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            if len(res.vel_stack) and len(res.ccf_stack):
                ax.plot(res.vel_stack, res.ccf_stack, color="C0", lw=0.9, label="median chunk CCF")
            ax.axvline(0.0, color="0.6", lw=0.8, zorder=0)
            if np.isfinite(res.dv_kms):
                ax.axvline(
                    res.dv_kms,
                    color="C3",
                    lw=1.6,
                    label=rf"$\Delta v_{{\star-\star}}$ (chunk IVW) = {res.dv_kms:+.3f} km/s",
                )
            if np.isfinite(res.peak_from_stack_kms):
                ax.axvline(
                    res.peak_from_stack_kms,
                    color="0.4",
                    ls=":",
                    lw=1.0,
                    label=rf"median-stack argmax = {res.peak_from_stack_kms:+.3f} km/s",
                )
            dv_abs = float("nan")
            if res.epoch_x in abs_by_ep and res.epoch_y in abs_by_ep:
                dv_abs = abs_by_ep[res.epoch_x] - abs_by_ep[res.epoch_y]
                ax.axvline(
                    dv_abs,
                    color="C1",
                    ls="--",
                    lw=1.3,
                    label=rf"$A_1-A_2$ = {dv_abs:+.3f} km/s",
                )
            if len(res.vel_stack):
                ax.set_xlim(float(np.min(res.vel_stack)), float(np.max(res.vel_stack)))
            ax.set_xlabel("lag (km/s)")
            ax.set_ylabel("CCF")
            title = f"{gaia_id}  mask-style CCF(epoch {res.epoch_x} vs {res.epoch_y})"
            if res.auto_correlation:
                title += "  [auto; smoothed mask]"
            else:
                resid = (
                    float(res.dv_kms) - dv_abs
                    if np.isfinite(res.dv_kms) and np.isfinite(dv_abs)
                    else float("nan")
                )
                title += (
                    f"  residual Δv★★−(A₁−A₂)={resid:+.3f} km/s  n_chunks={res.n_chunks}"
                )
            ax.set_title(title)
            ax.legend(loc="best", fontsize=8)
            ax.text(
                0.01,
                0.02,
                f"lag sampling Δv≈{res.lag_sample_kms:.3f} km/s  err={res.err_kms:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaia-id", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--abs-fill-csv",
        type=Path,
        default=None,
        help="Optional epoch_ccf_abs_fill.csv for abs Δv overlay / residual column",
    )
    ap.add_argument("--instrument", default="APF")
    ap.add_argument(
        "--continuum-mode",
        default=None,
        help=f"Continuum for both epochs (default: mask lane {config.MASK_CONTINUUM_MODE})",
    )
    ap.add_argument(
        "--blaze-calibration",
        type=Path,
        default=None,
        help=f"Blaze JSON (default: {config.BLAZE_CALIBRATION_FILE})",
    )
    ap.add_argument(
        "--qc-config",
        type=Path,
        default=None,
        help="QC YAML (default: order_chunk_qc.yaml)",
    )
    ap.add_argument(
        "--max-chunk-err",
        type=float,
        default=50.0,
        help="Hard err ceiling (km/s); also min'd with instrument QC max_chunk_err_kms",
    )
    ap.add_argument(
        "--auto-smooth-sigma",
        type=float,
        default=DEFAULT_AUTO_SMOOTH_SIGMA,
        help="Gaussian σ (pixels) for auto-corr mask smoothing",
    )
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    run_matrix(
        gaia_id=str(args.gaia_id),
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        abs_fill_csv=args.abs_fill_csv,
        instrument_name=str(args.instrument),
        continuum_mode=args.continuum_mode,
        blaze_calibration=args.blaze_calibration,
        qc_config=args.qc_config,
        max_chunk_err_kms=float(args.max_chunk_err),
        auto_smooth_sigma=float(args.auto_smooth_sigma),
        no_bias=bool(args.no_bias),
        write_pdf=not bool(args.no_pdf),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
