---
step_id: 05-short-pair-epoch-ccf
phase: E
status: complete
github_issue: https://github.com/UCSC-Transients/dark-hunter_rv/issues/42
branches:
  - step/05a-short-pair-calibration
merged: 2026-08-02 via #98 → #101
depends_on: [11-epoch-ccf-matrix]
blocks: []
master_todo_id: short-pair-epoch-ccf
related_legacy_plans:
  - rv_pipeline_roadmap_3a7b3787.plan.md
  - pr_repo_organization_326c519c.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
  - docs/operations.md
---

# Step 05: Short-pair calibration (QC for relative / absolute RVs)

## Goal / science outcome

Use closely spaced epoch pairs (~0 true ΔRV) to calibrate and stress-test absolute methods and the **epoch–epoch CCF matrix** (step 11).

## Scope (in) / non-goals (out)

**In:** `validation/find_short_pairs.py`; reports of ΔRV(abs methods) and ΔRV(epoch CCF) on short pairs; hooks in calibration docs.

**Out:** Primary relative-RV measurement (→ **step 11**). Legacy 05b “epoch CCF consistency only” is **superseded** by step 11.

## Prerequisites

- Multi-epoch stars in summaries / diagnostics
- Step 11 matrix (or at least pairwise CCF) available for relative checks
- Absolute baselines from mask / template / strong (Phase 1)

## Implementation tasks

### 05a (`step/05a-short-pair-calibration`)

- [x] Find pairs with Δt &lt; configurable threshold (default: same night)
- [x] Report ΔRV per absolute method; flag pairs violating ~0 km/s assumption
- [x] Report ΔRV from step 11 matrix vs 0; use to inflate \(\sigma_{ij}\)
- [x] Integrate into `run_calibration_setup` docs

### 05b — superseded

- ~~Epoch CCF as QC-only monitor~~ → see [11-epoch-ccf-matrix.md](11-epoch-ccf-matrix.md)

## Key files

- `validation/find_short_pairs.py` (new)
- `validation/run_calibration_setup.py`
- Step 11 outputs under `validation_output/epoch_ccf/`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m validation.find_short_pairs --summary-dir output --max-delta-days 1
```

## Acceptance criteria

- Short-pair report runs on survey output
- Quantifies abs and epoch-CCF scatter on Δt≈0 pairs
- Playbook documents Δt and thresholds

## Tests / validation

- Unit tests with synthetic pair timestamps
- Manual run on a known multi-epoch star

## Propagation checklist (on merge)

- [ ] Master todo `short-pair-epoch-ccf` → completed
- [ ] Point issue #42 at 05a-only scope + link step 11

## Open decisions

- Same night only vs &lt;24 h?
- Phase-gate known binaries?
