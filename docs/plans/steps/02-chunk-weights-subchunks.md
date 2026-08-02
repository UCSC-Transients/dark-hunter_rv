---
step_id: 02-chunk-weights-subchunks
phase: C
status: in_progress
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/39
branches:
  - step/02a-subchunk-study
  - step/02b-trust-weights-stack
depends_on: [01-benchmark-cool-precision]
blocks: [10-template-fft-precision]
master_todo_id: chunk-weights-subchunks
related_legacy_plans:
  - rv_pipeline_roadmap_3a7b3787.plan.md
  - rv_mismatch_diagnosis_24a11615.plan.md
  - pr_repo_organization_326c519c.plan.md
  - rv_precision_framework_8bc25b55.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
  - validation/chunk_optimization_advice.md
---

# Step 02: Wavelength chunks and trust-weighted stack

## Goal / science outcome

Replace arbitrary whole-order chunks with validated sub-chunking and post-debias weights that improve exposure RV precision and robustness. **Shared chunk layout** applies to mask_ccf and template_fft; trust-weight tuning (02b) is mask-stack focused but should not break template stacks.

## Scope (in) / non-goals (out)

**In:** Chunk layout campaign on APF; telluric/mask-line/consistency weights; persist in diagnostics; update `order_chunk_qc.yaml`.

**Out:** Template measurement knobs (step 10); full method fusion (step 03); exhaustive mixed per-order tilings (ruled out — uniform layout wins).

## Prerequisites

- Step 01 baseline metrics
- `validation/chunk_campaign.py`, `validation/chunk_calibration.py`

## Implementation tasks

### 02a (`step/02a-subchunk-study`) — **largely complete**

- [x] Run APF campaign with subchunks 2,3,4 (+ merge layouts) on 114-exposure list
- [x] Phase 1b: `subchunks_8` campaign (114/114 diagnostics, cache ingested)
- [x] `per_order_chunk_baseline`, `chunk_adaptive_stack`, `spectrum_tiling_search` tooling
- [x] **Decision:** uniform **`subchunks_8`** beats `subchunks_4` (median σ_RV 0.0189 vs 0.0223 km/s); adaptive mix adds no gain over pure s8
- [x] Ruled out: N=5,6,7,>8; per-order n=2/3/4 greedy mix (worse than uniform s8 under production stack)
- [x] **Production defaults:** `subchunks_8.yaml` in config + refit scripts
- [x] `calibration/bias_train.txt`, `scripts/rebuild_mask_bias.sh`, `calibration/mask_lane_deploy.md`
- [x] **Debias table verify (2026-08):** committed `bias_statistics.txt` is Jun-16 `subchunks_8` closeout (`a312993`); 364 `order_sub` keys, sub index 0–7; `tests/validation/test_build_bias_set.py` 5 passed. **Do not rebuild now.**
- [ ] ~~Rebuild + commit `bias_statistics.txt` for subchunks_8~~ → **DEFERRED pre-final product** (keep current Jun-16 committed table)
- [ ] Refit catalog on ziggy → **BLOCKED pending human** (do not run ziggy from agent)

### 02a deferred / human TODO

**Rebuild bias (pre-final only):** keep current committed Jun-16 `bias_statistics.txt`. Fresh rebuild is deferred until immediately before final product ship — not urgent for Phase 0c / 1A thin card.

**Refit catalog (ziggy) — TODO for human** (exact commands from `calibration/mask_lane_deploy.md`):

```bash
# On ziggy: rebuild mask bias (when human approves pre-final rebuild)
cd /data2/darkhunter/dark-hunter_rv
git pull
PY=/home/marley/anaconda2/envs/gaia-env/bin/python \
  OUT=/data2/darkhunter/dark-hunter_rv/output \
  bash scripts/rebuild_mask_bias.sh

# Then refit catalog
cd /data2/darkhunter/dark-hunter_rv
bash scripts/refit_all_per_object_parallel.sh

# Or single star:
STAR_ID=1702370142434513152 bash scripts/refit_star_rvs.sh
```

Local equivalents (not ziggy; for reference only):

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
bash scripts/rebuild_mask_bias.sh
bash scripts/refit_all_per_object_parallel.sh
```

### 02b (`step/02b-trust-weights-stack`) — **implemented (opt-in)**

- [x] Implement trust weights (residual vs robust mean, telluric fraction, CCF quality) scaling IVW in `pipeline.py`
- [x] Add weight columns to diagnostics CSV
- [x] Version `order_chunk_qc.yaml` thresholds
- [x] Offline A/B vs IVW-only: `validation/trust_weight_ab_report.py` (re-stack from diagnostics; supports `--teff-max` + relative gate)
  - Cool Teff<5000 (`output/`, n=16): median formal σ **0.022 → 0.035 km/s** (ratio ~1.59); relative pairs=0
  - First 200 `output/` diags: median formal σ **0.160 → 0.425 km/s** (ratio ~2.20); relative median |ΔRV| **9.67 → 7.52 km/s** (46 pairs)
  - **Keep default off** — formal σ rises under trust; human review `validation_output/trust_ab_post103/`
- [ ] Optional: full pipeline campaign with `--trust-weights` (heavy; not blocking if offline A/B accepted)

**Note:** Implementation ships **opt-in** (`trust_weights.enabled: false`; CLI `--trust-weights`). Offline σ_RV + relative-gate A/B captured post-#103; default remains off.

**Defer 02b** until template lane baseline (step 10) is captured — avoids retuning weights twice.

## Key files

- `calibration/chunk_layouts/subchunks_8.yaml`
- `validation/chunk_campaign.py`, `validation/chunk_adaptive_stack.py`
- `validation/per_order_chunk_baseline.py`, `validation/chunk_optimization_advice.md`
- `darkhunter_rv/pipeline.py`, `validation/build_bias_set.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
CAMPAIGN=validation_output/chunk_campaign

# Offline re-eval after campaign
PYTHONPATH=. python3 -m validation.chunk_adaptive_stack --campaign-dir "$CAMPAIGN"
PYTHONPATH=. python3 -m validation.per_order_chunk_baseline \
  --campaign-dir "$CAMPAIGN" --split-ns 1,2,3,4,8

# Deploy (production)
CHUNK_LAYOUT=calibration/chunk_layouts/subchunks_8.yaml bash scripts/refit_star_rvs.sh
```

## Acceptance criteria

- [x] Subchunk study shows improved median σ_RV vs subchunks_4 on common cohort
- [x] Production layout + committed Jun-16 `subchunks_8` bias verified (rebuild deferred pre-final)
- [ ] Ziggy catalog refit — human TODO (commands above); agent must not run ziggy
- [x] Trust-weighted stack (02b) — opt-in code + tests; campaign validation open; not blocking step 10

## Tests / validation

- `tests/validation/test_chunk_adaptive_stack.py`
- `tests/validation/test_per_order_chunk_baseline.py`
- `tests/validation/test_spectrum_tiling_search.py`

## Propagation checklist (on merge)

- [ ] Close 02a when deploy lands; keep issue #39 open for 02b
- [ ] Update `chunk_optimization_advice.md` deploy section (subchunks_8 winner)

## Open decisions

- **Resolved:** global uniform `subchunks_8`, not per-order mix.
- **Open:** trust weights before or after template lane? → **after** template baseline.
