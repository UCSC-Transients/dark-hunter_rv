#!/usr/bin/env python3
"""
Always-run epoch–epoch CCF for every multi-epoch Gaia ID under a data root.

Campaign / systematics hook (HUMAN_GATES #1): for each star with ≥2 epochs,
build the relative matrix, abs fill (when diagnostics exist), discordant flags,
and optionally enrich diagnostics so the adopt cascade can use
``epoch_ccf_abs_fill``.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.run_epoch_ccf_multi_epoch \\
    --data-root /Users/rfoley/darkhunter/rvs/data \\
    --out-root validation_output/epoch_ccf \\
    --abs-diagnostics-root output \\
    --enrich-diagnostics-root output \\
    --max-stars 20
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

from validation.epoch_ccf_matrix import enrich_diagnostics_with_epoch_ccf_fill, run_matrix

logger = logging.getLogger(__name__)

_EPOCH_FILE_RE = re.compile(r"^Gaia_DR3_(\d+)_epoch_(\d+)\.txt$")


def discover_multi_epoch_gaia_ids(data_root: Path, *, min_epochs: int = 2) -> list[str]:
    """Return Gaia IDs with at least ``min_epochs`` spectrum files under ``data_root``."""
    counts: dict[str, set[int]] = defaultdict(set)
    root = Path(data_root)
    for path in root.glob("Gaia_DR3_*_epoch_*.txt"):
        m = _EPOCH_FILE_RE.match(path.name)
        if not m:
            continue
        counts[m.group(1)].add(int(m.group(2)))
    out = [gid for gid, eps in counts.items() if len(eps) >= int(min_epochs)]
    out.sort()
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/rfoley/darkhunter/rvs/data"),
        help="Directory with Gaia_DR3_*_epoch_*.txt",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Per-star outputs under out-root/<gaia_id>/",
    )
    p.add_argument(
        "--abs-diagnostics-root",
        type=Path,
        default=None,
        help="Root with Gaia_DR3_<id>_epoch_*_diagnostics.csv (abs anchors)",
    )
    p.add_argument(
        "--enrich-diagnostics-root",
        type=Path,
        default=None,
        help="If set, enrich diagnostics in-place under this root (same basenames)",
    )
    p.add_argument("--min-epochs", type=int, default=2)
    p.add_argument("--max-stars", type=int, default=None)
    p.add_argument("--gaia-id", action="append", default=None, help="Limit to these IDs (repeatable)")
    p.add_argument("--discord-n-sigma", type=float, default=3.0)
    p.add_argument("--abs-method", default="mask_ccf")
    p.add_argument(
        "--engine",
        choices=("mask", "fft"),
        default="mask",
        help="Pair engine (default mask = spectrum-as-mask production CCF)",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.gaia_id:
        ids = [str(g) for g in args.gaia_id]
    else:
        ids = discover_multi_epoch_gaia_ids(data_root, min_epochs=int(args.min_epochs))
    if args.max_stars is not None and int(args.max_stars) > 0:
        ids = ids[: int(args.max_stars)]

    logger.info("Running epoch CCF (engine=%s) for %d multi-epoch star(s)", args.engine, len(ids))
    n_ok = 0
    n_fail = 0
    for gid in ids:
        out_dir = out_root / gid
        abs_glob = None
        if args.abs_diagnostics_root is not None:
            abs_glob = str(Path(args.abs_diagnostics_root) / f"Gaia_DR3_{gid}_epoch_*_diagnostics.csv")
        try:
            meta = run_matrix(
                gaia_id=gid,
                data_root=data_root,
                out_dir=out_dir,
                abs_diagnostics_glob=abs_glob,
                abs_method=str(args.abs_method),
                discord_n_sigma=float(args.discord_n_sigma),
                engine=str(args.engine),
            )
            fill_csv = meta.get("fill_csv")
            if args.enrich_diagnostics_root is not None and fill_csv:
                diag_glob = str(
                    Path(args.enrich_diagnostics_root) / f"Gaia_DR3_{gid}_epoch_*_diagnostics.csv"
                )
                written = enrich_diagnostics_with_epoch_ccf_fill(
                    diagnostics_glob=diag_glob,
                    fill_csv=Path(fill_csv),
                    out_dir=Path(args.enrich_diagnostics_root),
                    append_method_rows=True,
                )
                logger.info("%s: enriched %d diagnostics", gid, len(written))
            logger.info(
                "%s: anchors=%s discordant=%s",
                gid,
                meta.get("n_abs_anchors"),
                meta.get("n_abs_rel_discordant"),
            )
            n_ok += 1
        except Exception:
            logger.exception("Failed epoch CCF for %s", gid)
            n_fail += 1
    logger.info("Done: ok=%d fail=%d", n_ok, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
