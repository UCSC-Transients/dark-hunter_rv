# Human decision checklist (post soft-residuals)

Program code lanes for mask / template / strong / fusion / epoch-CCF / SB2 tooling are on `main` through [#106](https://github.com/UCSC-Transients/dark-hunter_rv/pull/106). Docs closeout: [#107](https://github.com/UCSC-Transients/dark-hunter_rv/pull/107).

**Applied defaults (2026-08-02, post-#106 continue):**

| # | Decision | Applied | Issue |
|---|----------|---------|-------|
| 1 | `epoch_ccf_*` default adopted RV | **No** (opt-in enrich only) | [#94](https://github.com/UCSC-Transients/dark-hunter_rv/issues/94) **closed** |
| 2 | Trust weights default on | **No** (`--trust-weights` only) | — |
| 3 | Pre-final bias rebuild + ziggy | **Still deferred** — say go to run | [#57](https://github.com/UCSC-Transients/dark-hunter_rv/issues/57), [#39](https://github.com/UCSC-Transients/dark-hunter_rv/issues/39) **open** |
| 4 | Lit n≥10 | **Waived** at n_stars=8 | [#45](https://github.com/UCSC-Transients/dark-hunter_rv/issues/45) **closed** |
| 5 | Full `output/` SB2 refit for flag rates | **Optional** — not blocking | [#44](https://github.com/UCSC-Transients/dark-hunter_rv/issues/44) **closed** |

## Still needs explicit “go”

1. **Bias rebuild + ziggy catalog refit** (`bash scripts/rebuild_mask_bias.sh` then ziggy `refit_all_per_object_parallel.sh`) — see `calibration/mask_lane_deploy.md` / ORCHESTRATOR §12.1.
2. Reopen any closed issue above if policy changes (e.g. enable epoch_ccf adopt, expand lit sample).

## Orchestrator stop condition

ORCHESTRATOR §9 code DoD **met**. Active residual: HUMAN_GATES #3 only.

## Quick verify

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python -m pytest tests/test_sb2.py tests/test_sb2_nss_cohort_report.py tests/validation/test_fetch_nss_source_ids.py tests/validation/test_trust_weight_ab_report.py -q
```
