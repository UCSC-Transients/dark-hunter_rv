# Human decision checklist (post soft-residuals)

Program code lanes for mask / template / strong / fusion / epoch-CCF / SB2 tooling are on `main` through [#105](https://github.com/UCSC-Transients/dark-hunter_rv/pull/105). Remaining items need **human decisions or data**, not more code scaffolding.

| # | Decision | Default if no answer | Notes |
|---|----------|----------------------|-------|
| 1 | Enable **`epoch_ccf_*` as default adopted RV**? | **No** (opt-in enrich only) | Issue [#94](https://github.com/UCSC-Transients/dark-hunter_rv/issues/94) |
| 2 | Enable **trust weights by default** (`order_chunk_qc.yaml`)? | **No** (CLI `--trust-weights` only) | A/B: formal σ rises ~1.6–2.4× ([#104](https://github.com/UCSC-Transients/dark-hunter_rv/pull/104)) |
| 3 | **Pre-final** `bias_statistics.txt` rebuild + ziggy catalog refit? | Deferred until you say go | §12.1 ziggy alert; issue [#57](https://github.com/UCSC-Transients/dark-hunter_rv/issues/57) |
| 4 | Lit cross-check **n≥10**: ingest ≥2 missing El-Badry APF spectra, or **waive** at n=8? | Leave [#45](https://github.com/UCSC-Transients/dark-hunter_rv/issues/45) open | 15 master IDs lack local spectra |
| 5 | SB2 cohort: full **`output/` refit** so `sb2_candidate` populates? | Optional; ziggy or local campaign | NSS dump 147/155 already on main; flag rates need fused diags |

## Orchestrator stop condition

Treat the ORCHESTRATOR §9 DoD as **met for code** except items that explicitly require the table above. Soft residuals after #105 are gate-bound.

## Quick verify (no Gaia)

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python -m pytest tests/test_sb2.py tests/test_sb2_nss_cohort_report.py tests/validation/test_fetch_nss_source_ids.py tests/validation/test_trust_weight_ab_report.py -q
```
