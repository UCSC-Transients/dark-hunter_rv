#!/usr/bin/env python3
"""
Per-order SB2 decomposition plots from separated component spectra.

Default: one epoch, all orders, multipage PDF (one order per page).

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.plot_sb2_decomposition_orders \\
    --sb2-dir validation_output/sb2_77413727493690112 \\
    --epoch-basename Gaia_DR3_77413727493690112_epoch_5.txt \\
    --out validation_output/sb2_77413727493690112/plots/epoch5_decomp.pdf
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config, io_utils
from darkhunter_rv.instruments import get_instrument_profile
from darkhunter_rv.plotting import _hb_tight_ylim_local_cont
from darkhunter_rv.sb2 import Sb2FitSettings, _order_obs_norm, build_sb2_fit_settings


class PlotScope(str, Enum):
    ONE_EPOCH_ALL_ORDERS = "one_epoch_all_orders"
    ONE_EPOCH_ONE_ORDER = "one_epoch_one_order"
    ALL_EPOCHS_ONE_ORDER = "all_epochs_one_order"


@dataclass
class EpochDecompInfo:
    basename: str
    spectrum_path: Path
    primary_path: Path
    secondary_path: Path


@dataclass
class OrderSeries:
    wavelength: np.ndarray
    data_norm: np.ndarray
    star1: np.ndarray
    star2: np.ndarray
    total_model: np.ndarray
    data_minus_star1: np.ndarray
    residual: np.ndarray


def resolve_plot_scope(
    *,
    epoch_basename: str | None,
    epoch_index: int | None,
    order: int | None,
    all_epochs: bool,
) -> PlotScope:
    if all_epochs:
        if order is None:
            raise ValueError("--all-epochs requires --order")
        if epoch_basename is not None or epoch_index is not None:
            raise ValueError("--all-epochs cannot be combined with --epoch-basename or --epoch-index")
        return PlotScope.ALL_EPOCHS_ONE_ORDER
    if order is not None:
        if epoch_basename is None and epoch_index is None:
            raise ValueError("--order without --all-epochs requires --epoch-basename or --epoch-index")
        return PlotScope.ONE_EPOCH_ONE_ORDER
    if epoch_basename is None and epoch_index is None:
        raise ValueError("specify --epoch-basename or --epoch-index")
    return PlotScope.ONE_EPOCH_ALL_ORDERS


def default_output_path(
    sb2_dir: Path,
    scope: PlotScope,
    *,
    epoch_basename: str | None,
    order: int | None,
) -> Path:
    plots_dir = sb2_dir / "plots"
    if scope == PlotScope.ONE_EPOCH_ALL_ORDERS:
        stem = Path(epoch_basename or "epoch").stem
        return plots_dir / f"{stem}_sb2_decomp_orders.pdf"
    if scope == PlotScope.ONE_EPOCH_ONE_ORDER:
        stem = Path(epoch_basename or "epoch").stem
        return plots_dir / f"{stem}_order{order}_sb2_decomp.pdf"
    return plots_dir / f"order{order}_all_epochs_sb2_decomp.pdf"


def compute_decomposition_series(
    data_norm: np.ndarray,
    star1: np.ndarray,
    star2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total_model, data_minus_star1, residual."""
    total = star1 + star2
    data_minus_star1 = data_norm - star1
    residual = data_norm - total
    return total, data_minus_star1, residual


def read_separated_spectrum_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        if data.size < 2:
            raise ValueError(f"empty separated spectrum: {path}")
        return np.array([data[0]]), np.array([data[1]])
    return np.asarray(data[:, 0], float), np.asarray(data[:, 1], float)


def load_sb2_epochs_table(sb2_dir: Path) -> pd.DataFrame:
    path = sb2_dir / "sb2_epochs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)


def _row_to_epoch_info(sb2_dir: Path, row: pd.Series) -> EpochDecompInfo:
    basename = str(row["basename"])
    stem = Path(basename).stem
    primary = sb2_dir / f"{stem}_primary_v0.txt"
    secondary = sb2_dir / f"{stem}_secondary_v0.txt"
    if not primary.is_file() or not secondary.is_file():
        raise FileNotFoundError(
            f"missing separated spectra for {basename}: need {primary.name} and {secondary.name}"
        )
    spec_path = Path(str(row.get("spectrum_path", "")))
    if not spec_path.is_file():
        raise FileNotFoundError(f"spectrum not found for {basename}: {spec_path}")
    return EpochDecompInfo(
        basename=basename,
        spectrum_path=spec_path,
        primary_path=primary,
        secondary_path=secondary,
    )


def select_epoch_row(df: pd.DataFrame, *, basename: str | None, index: int | None) -> pd.Series:
    if basename is not None:
        hit = df[df["basename"] == basename]
        if hit.empty:
            raise ValueError(f"epoch basename not in sb2_epochs.csv: {basename}")
        return hit.iloc[0]
    if index is None:
        raise ValueError("epoch index required")
    if index < 0 or index >= len(df):
        raise ValueError(f"epoch index out of range: {index} (n={len(df)})")
    return df.iloc[int(index)]


def _hot_from_sb2_dir(sb2_dir: Path) -> bool:
    report = sb2_dir / "sb2_report.json"
    if report.is_file():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            teff = float(data.get("teff_gaia", 0.0))
            if np.isfinite(teff) and teff > 0:
                return teff > config.HOT_STAR_TEFF_THRESHOLD
        except Exception:
            pass
    return False


def normalize_order_flux(
    spec_data: dict,
    order: int,
    *,
    hot: bool,
    settings: Sb2FitSettings,
) -> tuple[np.ndarray, np.ndarray] | None:
    return _order_obs_norm(spec_data, int(order), hot, settings=settings)


def build_order_series(
    wave: np.ndarray,
    data_norm: np.ndarray,
    wave_sep: np.ndarray,
    flux1_sep: np.ndarray,
    flux2_sep: np.ndarray,
    *,
    epoch_basename: str | None = None,
    order: int | None = None,
) -> OrderSeries | None:
    star1 = np.interp(wave, wave_sep, flux1_sep, left=np.nan, right=np.nan)
    star2 = np.interp(wave, wave_sep, flux2_sep, left=np.nan, right=np.nan)
    valid = np.isfinite(data_norm) & np.isfinite(star1) & np.isfinite(star2)
    total, data_minus_star1, residual = compute_decomposition_series(data_norm, star1, star2)
    if int(np.sum(valid)) < 10:
        return None
    if np.any(valid):
        med_total = float(np.nanmedian(total[valid]))
        if med_total > 1.35:
            ctx = f"order {order} " if order is not None else ""
            epoch = epoch_basename or "epoch"
            logging.warning(
                "%s%s: median(star1+star2)=%.2f (>1.35); separated files may be stale "
                "(re-run sb2_search to regenerate flux-fraction export)",
                ctx,
                epoch,
                med_total,
            )
    return OrderSeries(
        wavelength=wave,
        data_norm=data_norm,
        star1=star1,
        star2=star2,
        total_model=total,
        data_minus_star1=data_minus_star1,
        residual=residual,
    )


def plot_order_decomposition_page(
    fig_title: str,
    series: OrderSeries,
    *,
    order: int,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    w = series.wavelength

    ax0, ax1, ax2 = axes
    ax0.step(w, series.data_norm, where="mid", color="0.35", lw=0.8, label="data (norm)", zorder=2)
    ax0.plot(w, series.total_model, "k-", lw=1.0, alpha=0.9, label="total model", zorder=3)
    ax0.plot(w, series.star1, color="tab:blue", lw=0.9, alpha=0.85, label="star 1", zorder=4)
    ax0.plot(w, series.star2, color="tab:orange", ls="--", lw=0.9, alpha=0.85, label="star 2", zorder=5)
    _hb_tight_ylim_local_cont(ax0, w, [series.data_norm, series.total_model, series.star1, series.star2])
    ax0.set_ylabel("Norm flux")
    ax0.set_title(f"{fig_title}  order {order}")
    ax0.legend(loc="best", fontsize=7)
    ax0.grid(True, alpha=0.2)

    ax1.step(w, series.data_minus_star1, where="mid", color="tab:blue", lw=0.8, label="data − star1", zorder=2)
    ax1.plot(w, series.star2, color="tab:orange", ls="--", lw=0.9, alpha=0.9, label="star 2", zorder=3)
    _hb_tight_ylim_local_cont(ax1, w, [series.data_minus_star1, series.star2])
    ax1.set_ylabel("Norm flux")
    ax1.legend(loc="best", fontsize=7)
    ax1.grid(True, alpha=0.2)

    ax2.step(w, series.residual, where="mid", color="0.25", lw=0.8, label="data − total", zorder=2)
    ax2.axhline(0.0, color="0.5", ls="--", lw=1.0, zorder=1)
    resid_spread = float(np.nanmax(np.abs(series.residual[np.isfinite(series.residual)]))) if np.any(np.isfinite(series.residual)) else 0.1
    pad = max(0.02, 0.15 * resid_spread)
    ax2.set_ylim(-resid_spread - pad, resid_spread + pad)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Wavelength (Å)")
    ax2.legend(loc="best", fontsize=7)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


def _valid_orders(spec_data: dict, instrument) -> list[int]:
    return sorted(o for o in spec_data if o not in instrument.bad_orders)


def render_epoch_orders_pdf(
    pdf: PdfPages,
    epoch: EpochDecompInfo,
    *,
    instrument,
    hot: bool,
    fit_settings: Sb2FitSettings,
    orders: list[int] | None,
    gaia_id: str | None,
) -> int:
    _, spec_data = io_utils.read_spectrum(str(epoch.spectrum_path))
    wave_sep, flux1 = read_separated_spectrum_txt(epoch.primary_path)
    _, flux2 = read_separated_spectrum_txt(epoch.secondary_path)
    order_list = orders if orders is not None else _valid_orders(spec_data, instrument)
    n_pages = 0
    gid = gaia_id or "Gaia"
    for order in order_list:
        norm = normalize_order_flux(spec_data, order, hot=hot, settings=fit_settings)
        if norm is None:
            logging.warning("skip order %s for %s: normalization failed", order, epoch.basename)
            continue
        wave, data_norm = norm
        series = build_order_series(
            wave, data_norm, wave_sep, flux1, flux2, epoch_basename=epoch.basename, order=order
        )
        if series is None:
            logging.warning("skip order %s for %s: insufficient valid decomposition points", order, epoch.basename)
            continue
        title = f"{gid}  {epoch.basename}"
        fig = plot_order_decomposition_page(title, series, order=order)
        pdf.savefig(fig)
        plt.close(fig)
        n_pages += 1
    return n_pages


def run_plot_job(
    sb2_dir: Path,
    out_path: Path,
    scope: PlotScope,
    *,
    epoch_basename: str | None,
    epoch_index: int | None,
    order: int | None,
    instrument_name: str,
    gaia_id: str | None,
    continuum_mode: str = "sinc_blaze",
    blaze_calibration_path: Path | None = None,
) -> int:
    sb2_dir = Path(sb2_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    instrument = get_instrument_profile(instrument_name)
    hot = _hot_from_sb2_dir(sb2_dir)
    fit_settings = build_sb2_fit_settings(
        continuum_mode=str(continuum_mode),
        blaze_calibration_path=blaze_calibration_path,
    )
    df = load_sb2_epochs_table(sb2_dir)

    with PdfPages(out_path) as pdf:
        if scope == PlotScope.ALL_EPOCHS_ONE_ORDER:
            assert order is not None
            n_total = 0
            for _, row in df.iterrows():
                try:
                    epoch = _row_to_epoch_info(sb2_dir, row)
                except FileNotFoundError as ex:
                    logging.warning("skip %s: %s", row.get("basename"), ex)
                    continue
                n = render_epoch_orders_pdf(
                    pdf,
                    epoch,
                    instrument=instrument,
                    hot=hot,
                    fit_settings=fit_settings,
                    orders=[int(order)],
                    gaia_id=gaia_id,
                )
                n_total += n
            if n_total == 0:
                raise RuntimeError(f"no pages written for order {order}")
            logging.info("Wrote %d page(s) -> %s", n_total, out_path.resolve())
            return 0

        row = select_epoch_row(df, basename=epoch_basename, index=epoch_index)
        epoch = _row_to_epoch_info(sb2_dir, row)
        orders = [int(order)] if order is not None else None
        n = render_epoch_orders_pdf(
            pdf,
            epoch,
            instrument=instrument,
            hot=hot,
            fit_settings=fit_settings,
            orders=orders,
            gaia_id=gaia_id,
        )
        if n == 0:
            raise RuntimeError(f"no pages written for {epoch.basename}")
        logging.info("Wrote %d page(s) -> %s", n, out_path.resolve())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="SB2 per-order decomposition plots (multipage PDF)")
    ap.add_argument("--sb2-dir", type=Path, required=True, help="SB2 output dir with separated *_primary_v0.txt files")
    ap.add_argument("--epoch-basename", default=None, help="Epoch basename from sb2_epochs.csv")
    ap.add_argument("--epoch-index", type=int, default=None, help="0-based epoch index in sb2_epochs.csv")
    ap.add_argument("--order", type=int, default=None, help="Plot only this echelle order")
    ap.add_argument("--all-epochs", action="store_true", help="With --order: one page per epoch for that order")
    ap.add_argument("--out", type=Path, default=None, help="Output PDF path (default under sb2-dir/plots/)")
    ap.add_argument("--gaia-id", default=None, help="Optional Gaia ID for plot titles")
    ap.add_argument("--instrument", default="APF", choices=["APF", "GHOST", "MAROON-X"])
    ap.add_argument(
        "--continuum-mode",
        choices=["sinc_blaze", "sinc_blaze_only", "spline"],
        default="sinc_blaze",
        help="Continuum mode for plotted observed spectrum normalization",
    )
    ap.add_argument(
        "--blaze-calibration",
        type=Path,
        default=config.BLAZE_CALIBRATION_FILE,
        help="Per-order blaze calibration JSON (default: config.BLAZE_CALIBRATION_FILE)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    scope = resolve_plot_scope(
        epoch_basename=args.epoch_basename,
        epoch_index=args.epoch_index,
        order=args.order,
        all_epochs=bool(args.all_epochs),
    )
    out = args.out or default_output_path(
        args.sb2_dir,
        scope,
        epoch_basename=args.epoch_basename,
        order=args.order,
    )
    return run_plot_job(
        args.sb2_dir,
        out,
        scope,
        epoch_basename=args.epoch_basename,
        epoch_index=args.epoch_index,
        order=args.order,
        instrument_name=args.instrument,
        gaia_id=args.gaia_id,
        continuum_mode=args.continuum_mode,
        blaze_calibration_path=args.blaze_calibration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
