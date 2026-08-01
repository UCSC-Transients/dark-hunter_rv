#!/usr/bin/env python3
"""
Force-fit Balmer strong lines per exposure and compare to mask RV by Teff bin.

Answers step 06 / #43: on real APF spectra, do non-Hβ lines ever beat Hβ vs mask,
or does the pipeline Hβ short-circuit leave residual wins on the table?

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.strong_line_teff_sweep \\
    --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \\
    --overlap-csv validation_output/template_fft_baseline/pipeline_cool_vsini12_mhfix/overlap/overlap_enriched_per_exposure.csv \\
    --data-root /Users/rfoley/darkhunter/rvs/data \\
    --out-dir validation_output/strong_line_teff_sweep \\
    --continuum-mode sinc_blaze
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config, continuum, instruments, io_utils, rv_core
from darkhunter_rv.blaze import BlazeCalibration
from darkhunter_rv.pipeline import _rv_kms_from_hb_joint_line_center

logger = logging.getLogger(__name__)

LINE_REST: list[tuple[str, float]] = [
    ("Halpha", float(rv_core.HA_REST_A)),
    ("Hbeta", float(rv_core.HB_REST_A)),
    ("Hgamma", float(rv_core.HG_REST_A)),
    ("Hdelta", float(rv_core.HD_REST_A)),
]

TEFF_EDGES = [0.0, 4500.0, 5500.0, 6500.0, 1.0e5]
TEFF_LABELS = ["<4500", "4500-5500", "5500-6500", ">=6500"]


def teff_bin_label(teff: float) -> str:
    t = float(teff) if teff == teff else float("nan")
    if not np.isfinite(t):
        return "unknown"
    for lo, hi, lab in zip(TEFF_EDGES[:-1], TEFF_EDGES[1:], TEFF_LABELS):
        if lo <= t < hi:
            return lab
    return TEFF_LABELS[-1]


def pipeline_score(err_kms: float, cand_index: int) -> float:
    """Match pipeline.strong_lines candidate scoring (err + 0.5 * preference index)."""
    score = float(err_kms) if np.isfinite(err_kms) and err_kms > 0 else 50.0
    return score + 0.5 * float(cand_index)


def select_best_line(
    candidates: list[dict],
    *,
    short_circuit_hbeta: bool = True,
    hbeta_err_cut: float = 20.0,
) -> dict | None:
    """
    Pick one candidate using Teff preference order already encoded in ``cand_index``.

    When ``short_circuit_hbeta`` is True (production), stop after a successful Hβ with
    score < ``hbeta_err_cut`` (pipeline uses best_score after Hβ loop).
    """
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda c: (int(c["cand_index"]), str(c["line"])))
    best: dict | None = None
    best_score = float("inf")
    for c in ordered:
        score = pipeline_score(float(c.get("err_kms", np.nan)), int(c["cand_index"]))
        if score < best_score:
            best_score = score
            best = {**c, "score": score}
        if (
            short_circuit_hbeta
            and str(c["line"]) == "Hbeta"
            and best is not None
            and best_score < float(hbeta_err_cut)
        ):
            break
    return best


def _order_covering_rest(spec_data: dict, valid_orders: list[int], rest: float) -> int | None:
    for o in valid_orders:
        w = np.array(spec_data[o]["wavelength"], float)
        lo, hi = float(np.min(w)), float(np.max(w))
        if lo <= float(rest) <= hi:
            return int(o)
    return None


def _continuum_kw(
    *,
    continuum_mode: str,
    echelle_order: int,
    hot: bool,
    blaze_cal: BlazeCalibration | None,
) -> dict:
    mode = str(continuum_mode)
    kw: dict = {"continuum_mode": mode, "echelle_order": int(echelle_order)}
    if mode in ("sinc_blaze", "sinc_blaze_only"):
        if blaze_cal is not None:
            model = blaze_cal.model_for_order(int(echelle_order))
            if model is not None:
                kw["blaze_model"] = model
            else:
                kw["continuum_mode"] = "spline"
                mode = "spline"
        else:
            kw["continuum_mode"] = "spline"
            mode = "spline"
    if mode in ("spline", "sinc_blaze"):
        kw["exclude_near_lines_width"] = float(
            config.HOT_SPLINE_EXCLUDE_NEAR_LINES_WIDTH
            if hot
            else config.COOL_SPLINE_EXCLUDE_NEAR_LINES_WIDTH
        )
    return kw


def fit_all_balmer_lines(
    spectrum_path: Path,
    *,
    teff: float,
    continuum_mode: str,
    blaze_cal: BlazeCalibration | None,
) -> list[dict]:
    """Fit every Balmer candidate that falls on an echelle order; no short-circuit."""
    instrument = instruments.guess_instrument(str(spectrum_path))
    spec_data = io_utils.load_spectrum(str(spectrum_path), instrument)
    valid_orders = instruments.valid_orders(instrument, spec_data)
    hot = float(teff) >= float(config.METHOD_REGION_STRONG_LINES_MIN_TEFF_K)
    pref = {name: i for i, (name, _) in enumerate(rv_core.strong_line_rests_for_teff(teff))}
    R_inst = float(getattr(instrument, "resolving_power", 60_000.0))
    out: list[dict] = []
    for line_name, rest in LINE_REST:
        o = _order_covering_rest(spec_data, valid_orders, rest)
        if o is None:
            out.append(
                {
                    "line": line_name,
                    "rest_a": float(rest),
                    "ok": False,
                    "reason": "no_order_coverage",
                    "cand_index": int(pref.get(line_name, 99)),
                }
            )
            continue
        w = np.array(spec_data[o]["wavelength"], float)
        f = np.array(spec_data[o]["flux"], float)
        e = np.array(spec_data[o]["eflux"], float)
        try:
            kw = _continuum_kw(
                continuum_mode=continuum_mode,
                echelle_order=o,
                hot=hot,
                blaze_cal=blaze_cal,
            )
            nw, nf, _ = continuum.fit_continuum(w, f, e, **kw)
            nw, nf, _ = continuum.despike_normalized_pre_ccf(nw, nf, np.ones_like(nf))
        except Exception as ex:
            out.append(
                {
                    "line": line_name,
                    "rest_a": float(rest),
                    "ok": False,
                    "reason": f"continuum:{ex}",
                    "cand_index": int(pref.get(line_name, 99)),
                    "order": int(o),
                }
            )
            continue
        bundle = rv_core.measure_strong_line_voigt_lorentz(
            nw,
            nf,
            rest=float(rest),
            broad_lines=bool(hot),
            resolving_power=R_inst,
        )
        if bundle is None:
            out.append(
                {
                    "line": line_name,
                    "rest_a": float(rest),
                    "ok": False,
                    "reason": "fit_failed",
                    "cand_index": int(pref.get(line_name, 99)),
                    "order": int(o),
                }
            )
            continue
        rv = float(_rv_kms_from_hb_joint_line_center(bundle))
        err = float(bundle.get("err_voigt_kms", np.nan))
        out.append(
            {
                "line": line_name,
                "rest_a": float(rest),
                "ok": bool(np.isfinite(rv)),
                "reason": "" if np.isfinite(rv) else "nonfinite_rv",
                "cand_index": int(pref.get(line_name, 99)),
                "order": int(o),
                "rv_kms": rv,
                "err_kms": err,
                "score": pipeline_score(err, int(pref.get(line_name, 99))),
            }
        )
    return out


def _resolve_spectrum(stem_or_path: str, data_root: Path) -> Path | None:
    p = Path(stem_or_path)
    if p.is_file():
        return p
    name = p.name
    if not name.endswith(".txt"):
        name = f"{name}.txt"
    cand = data_root / name
    return cand if cand.is_file() else None


def _load_stems(spectrum_list: Path) -> list[str]:
    lines = spectrum_list.read_text().splitlines()
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def run_sweep(
    *,
    spectrum_list: Path,
    overlap_csv: Path,
    data_root: Path,
    out_dir: Path,
    continuum_mode: str,
    blaze_path: Path | None,
    limit: int | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    ov = pd.read_csv(overlap_csv)
    ov["stem"] = ov["basename"].astype(str).str.replace(r"\.txt$", "", regex=True)
    mask_by_stem = {
        str(r["stem"]): (
            float(r["mask_rv_kms"]) if pd.notna(r["mask_rv_kms"]) else float("nan"),
            bool(r["mask_valid"]) if pd.notna(r["mask_valid"]) else False,
            float(r["teff"]) if pd.notna(r["teff"]) else float("nan"),
        )
        for _, r in ov.iterrows()
    }
    blaze_cal = BlazeCalibration.load(blaze_path) if blaze_path and blaze_path.is_file() else None
    if blaze_cal is None and continuum_mode in ("sinc_blaze", "sinc_blaze_only"):
        logger.warning("No blaze calibration loaded; continuum will fall back to spline")

    stems = _load_stems(spectrum_list)
    if limit is not None:
        stems = stems[: int(limit)]

    per_line_rows: list[dict] = []
    summary_rows: list[dict] = []
    for i, stem_raw in enumerate(stems, start=1):
        path = _resolve_spectrum(stem_raw, data_root)
        stem = Path(stem_raw).name.replace(".txt", "")
        logger.info("[%d/%d] %s", i, len(stems), stem)
        if path is None:
            summary_rows.append({"stem": stem, "ok": False, "reason": "missing_spectrum"})
            continue
        mask_rv, mask_ok, teff_ov = mask_by_stem.get(stem, (float("nan"), False, float("nan")))
        teff = float(teff_ov) if np.isfinite(teff_ov) else float(config.DEFAULT_TEFF)
        fits = fit_all_balmer_lines(
            path,
            teff=teff,
            continuum_mode=continuum_mode,
            blaze_cal=blaze_cal,
        )
        ok_fits = [c for c in fits if c.get("ok")]
        for c in fits:
            abs_d = (
                abs(float(c["rv_kms"]) - float(mask_rv))
                if c.get("ok") and mask_ok and np.isfinite(mask_rv)
                else float("nan")
            )
            per_line_rows.append(
                {
                    "stem": stem,
                    "teff": teff,
                    "teff_bin": teff_bin_label(teff),
                    "mask_rv_kms": mask_rv,
                    "mask_valid": mask_ok,
                    "line": c["line"],
                    "rest_a": c["rest_a"],
                    "ok": bool(c.get("ok")),
                    "reason": c.get("reason", ""),
                    "order": c.get("order", np.nan),
                    "rv_kms": c.get("rv_kms", np.nan),
                    "err_kms": c.get("err_kms", np.nan),
                    "cand_index": c.get("cand_index", np.nan),
                    "score": c.get("score", np.nan),
                    "abs_delta_mask_kms": abs_d,
                }
            )

        prod = select_best_line(ok_fits, short_circuit_hbeta=True)
        free = select_best_line(ok_fits, short_circuit_hbeta=False)
        by_abs = (
            min(ok_fits, key=lambda c: abs(float(c["rv_kms"]) - float(mask_rv)))
            if ok_fits and mask_ok and np.isfinite(mask_rv)
            else None
        )
        hb = next((c for c in ok_fits if c["line"] == "Hbeta"), None)

        def _abs(c: dict | None) -> float:
            if c is None or not mask_ok or not np.isfinite(mask_rv):
                return float("nan")
            return abs(float(c["rv_kms"]) - float(mask_rv))

        summary_rows.append(
            {
                "stem": stem,
                "teff": teff,
                "teff_bin": teff_bin_label(teff),
                "mask_rv_kms": mask_rv,
                "mask_valid": mask_ok,
                "n_lines_ok": len(ok_fits),
                "prod_line": None if prod is None else prod["line"],
                "prod_abs_delta_mask_kms": _abs(prod),
                "free_line": None if free is None else free["line"],
                "free_abs_delta_mask_kms": _abs(free),
                "oracle_line": None if by_abs is None else by_abs["line"],
                "oracle_abs_delta_mask_kms": _abs(by_abs),
                "hbeta_abs_delta_mask_kms": _abs(hb),
                "free_differs_from_prod": bool(
                    free is not None and prod is not None and free["line"] != prod["line"]
                ),
                "oracle_beats_hbeta": bool(
                    by_abs is not None
                    and hb is not None
                    and _abs(by_abs) + 1e-9 < _abs(hb)
                    and by_abs["line"] != "Hbeta"
                ),
            }
        )

    per_line = pd.DataFrame(per_line_rows)
    summary = pd.DataFrame(summary_rows)
    per_line_path = out_dir / "per_line_fits.csv"
    summary_path = out_dir / "per_exposure_summary.csv"
    per_line.to_csv(per_line_path, index=False)
    summary.to_csv(summary_path, index=False)

    md = _write_report(summary, per_line, out_dir / "SWEEP_SUMMARY.md")
    logger.info("Wrote %s, %s, %s", per_line_path, summary_path, md)
    return summary


def _write_report(summary: pd.DataFrame, per_line: pd.DataFrame, path: Path) -> Path:
    s = summary.copy()
    ok = s[s.get("mask_valid", False) == True] if "mask_valid" in s.columns else s  # noqa: E712
    lines = [
        "# Strong-line Teff sweep (#43)",
        "",
        f"Exposures: **{len(s)}**; with valid mask: **{len(ok)}**.",
        "",
        "## Pipeline product line (with Hβ short-circuit)",
        "",
    ]
    if "prod_line" in s.columns:
        vc = s["prod_line"].fillna("none").value_counts()
        lines.append("| line | n |")
        lines.append("|------|--:|")
        for k, v in vc.items():
            lines.append(f"| `{k}` | {int(v)} |")
        lines.append("")

    lines.extend(
        [
            "## Free selection (no Hβ short-circuit)",
            "",
        ]
    )
    if "free_line" in s.columns:
        vc = s["free_line"].fillna("none").value_counts()
        lines.append("| line | n |")
        lines.append("|------|--:|")
        for k, v in vc.items():
            lines.append(f"| `{k}` | {int(v)} |")
        lines.append("")
        n_diff = int(s["free_differs_from_prod"].fillna(False).sum()) if len(s) else 0
        lines.append(f"Free selection differs from product: **{n_diff}/{len(s)}**.")
        lines.append("")

    lines.extend(
        [
            "## Oracle (lowest |RV−mask| among successful fits)",
            "",
        ]
    )
    if "oracle_line" in ok.columns and len(ok):
        vc = ok["oracle_line"].fillna("none").value_counts()
        lines.append("| line | n |")
        lines.append("|------|--:|")
        for k, v in vc.items():
            lines.append(f"| `{k}` | {int(v)} |")
        lines.append("")
        n_beat = int(ok["oracle_beats_hbeta"].fillna(False).sum())
        lines.append(
            f"Non-Hβ oracle beats Hβ vs mask: **{n_beat}/{len(ok)}** "
            f"({100.0 * n_beat / max(len(ok), 1):.1f}%)."
        )
        lines.append("")

    lines.extend(["## Median |RV−mask| by Teff bin × line", "",])
    if len(per_line) and "abs_delta_mask_kms" in per_line.columns:
        pl = per_line[per_line["ok"] & per_line["mask_valid"]].copy()
        if len(pl):
            tab = (
                pl.groupby(["teff_bin", "line"], observed=True)["abs_delta_mask_kms"]
                .median()
                .unstack()
            )
            lines.append(tab.to_markdown())
            lines.append("")
            succ = pl.groupby(["teff_bin", "line"], observed=True).size().unstack(fill_value=0)
            lines.extend(["## Successful fits by Teff bin × line", "", succ.to_markdown(), ""])

    if len(ok):
        lines.extend(
            [
                "## Product vs oracle residual (mask-valid)",
                "",
                f"- median |prod−mask|: **{ok['prod_abs_delta_mask_kms'].median():.2f}** km/s",
                f"- median |free−mask|: **{ok['free_abs_delta_mask_kms'].median():.2f}** km/s",
                f"- median |oracle−mask|: **{ok['oracle_abs_delta_mask_kms'].median():.2f}** km/s",
                f"- median |Hβ−mask|: **{ok['hbeta_abs_delta_mask_kms'].median():.2f}** km/s",
                "",
            ]
        )

    lines.extend(
        [
            "## Verdict",
            "",
            "If product is almost always Hβ and oracle rarely prefers another Balmer line with a",
            "material residual gain, keep Teff preference + Hβ short-circuit; document Hα/Hγ/Hδ as",
            "fallback-only on APF.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spectrum-list", type=Path, required=True)
    p.add_argument("--overlap-csv", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/Users/rfoley/darkhunter/rvs/data"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--continuum-mode",
        default="sinc_blaze",
        help="Strong-lane continuum (campaign default TEMPLATE_CONTINUUM_MODE)",
    )
    p.add_argument(
        "--blaze-calibration",
        type=Path,
        default=config.BLAZE_CALIBRATION_FILE,
    )
    p.add_argument("--limit", type=int, default=None, help="Optional cap for smoke runs")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    run_sweep(
        spectrum_list=args.spectrum_list,
        overlap_csv=args.overlap_csv,
        data_root=args.data_root,
        out_dir=args.out_dir,
        continuum_mode=str(args.continuum_mode),
        blaze_path=args.blaze_calibration,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
