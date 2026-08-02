# RV pipeline step tracker (GitHub)

**Step plans (rendered preview in IDE):** [steps/](steps/) — open any `.md` and use **Markdown: Open Preview** (Cmd+Shift+V).

**Order of attack (baselines first + parallel subagents):** [ATTACK_ORDER.md](ATTACK_ORDER.md)

**Orchestrator brief (launch subagents from this):** [ORCHESTRATOR.md](ORCHESTRATOR.md)

Orchestrator (local legacy): `.cursor/plans/rv_pipeline_master_plan_8447f2cd.plan.md`

Workflow: [WORKFLOW.md](WORKFLOW.md)

## Per-method precision lanes (current focus)

| Lane | Steps | Status |
|------|-------|--------|
| **mask_ccf** | 01, 02a, 09 | Defaults → `subchunks_8`; bias keep Jun-16 (rebuild pre-final); **01 complete (waivers)** |
| **template_fft** | 10 | **Complete** (#88 + 10c offsets on main via #95/#101) |
| **strong_lines** | 06 | **Complete** ([#90](https://github.com/UCSC-Transients/dark-hunter_rv/pull/90)) — Hβ-primary Balmer; metals + Q×SNR² IVW |
| **epoch_ccf** (relative) | 11 | **11a–c + 05a on main**; 11d product tags / default adopt still open ([#94](https://github.com/UCSC-Transients/dark-hunter_rv/issues/94)) |
| **fusion / adoption** | 03, 04 | **Complete on main** (#97 → #101) |

| Step | Plan | Status | Issue | Branch(es) | Merged |
|------|------|--------|-------|------------|--------|
| 00 Literature RV master | [steps/00-literature-rv-master.md](steps/00-literature-rv-master.md) | completed | [#37](https://github.com/astrofoley/dark-hunter_rv/issues/37) | `step/00-literature-rv-master` | 2026-06-07 ([#46](https://github.com/astrofoley/dark-hunter_rv/pull/46)) |
| 01 Benchmark cool precision | [steps/01-benchmark-cool-precision.md](steps/01-benchmark-cool-precision.md) | **complete** (waivers) | [#38](https://github.com/UCSC-Transients/dark-hunter_rv/issues/38) | `step/01-cool-closeout` | 2026-08-02 ([#96](https://github.com/UCSC-Transients/dark-hunter_rv/pull/96) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 02 Chunk weights / subchunks | [steps/02-chunk-weights-subchunks.md](steps/02-chunk-weights-subchunks.md) | in_progress (**02b opt-in on main**; campaign A/B open; 02a bias rebuild deferred) | [#39](https://github.com/UCSC-Transients/dark-hunter_rv/issues/39) | `step/02a-bias-defer-verify`, `step/02b-trust-weights-stack` | 2026-08-02 ([#95](https://github.com/UCSC-Transients/dark-hunter_rv/pull/95)/[#99](https://github.com/UCSC-Transients/dark-hunter_rv/pull/99) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 03 Method fusion / coverage | [steps/03-method-fusion-coverage.md](steps/03-method-fusion-coverage.md) | **complete** | [#40](https://github.com/UCSC-Transients/dark-hunter_rv/issues/40) | `step/03-method-fusion-coverage` | 2026-08-02 ([#97](https://github.com/UCSC-Transients/dark-hunter_rv/pull/97) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 04 Adopted-RV match plots | [steps/04-adopted-rv-match-plots.md](steps/04-adopted-rv-match-plots.md) | **complete** (visual residual optional) | [#41](https://github.com/UCSC-Transients/dark-hunter_rv/issues/41) | `step/04-adopted-rv-match-plots` | 2026-08-02 ([#97](https://github.com/UCSC-Transients/dark-hunter_rv/pull/97) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 05 Short-pair QC | [steps/05-short-pair-epoch-ccf.md](steps/05-short-pair-epoch-ccf.md) | **complete** (05a) | [#42](https://github.com/UCSC-Transients/dark-hunter_rv/issues/42) | `step/05a-short-pair-calibration` | 2026-08-02 ([#98](https://github.com/UCSC-Transients/dark-hunter_rv/pull/98) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 06 Strong-line line list | [steps/06-strong-line-line-list.md](steps/06-strong-line-line-list.md) | **complete** | [#43](https://github.com/UCSC-Transients/dark-hunter_rv/issues/43), [#91](https://github.com/UCSC-Transients/dark-hunter_rv/issues/91)–[#93](https://github.com/UCSC-Transients/dark-hunter_rv/issues/93) | `step/06-strong-line-teff-sweep` | 2026-08-02 ([#90](https://github.com/UCSC-Transients/dark-hunter_rv/pull/90)) |
| 07 SB2 search | [steps/07-sb2-search.md](steps/07-sb2-search.md) | in_progress (07a/b on main; NSS table + pipeline fuse residual) | [#44](https://github.com/UCSC-Transients/dark-hunter_rv/issues/44) | `step/07-sb2-search` | 2026-08-02 ([#99](https://github.com/UCSC-Transients/dark-hunter_rv/pull/99) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 08 External RV cross-check | [steps/08-external-rv-crosscheck.md](steps/08-external-rv-crosscheck.md) | **complete PARTIAL** (CLI on main; n_lit=4 need ≥10) | [#45](https://github.com/UCSC-Transients/dark-hunter_rv/issues/45) | `step/08-external-rv-crosscheck`, `step/08-external-rv-full` | 2026-08-02 ([#96](https://github.com/UCSC-Transients/dark-hunter_rv/pull/96)/[#99](https://github.com/UCSC-Transients/dark-hunter_rv/pull/99) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 09 CCF RV estimator (mask) | [steps/09-ccf-rv-estimator.md](steps/09-ccf-rv-estimator.md) | complete (`gauss_offset`) | — | — | — |
| 10 Template FFT precision | [steps/10-template-fft-precision.md](steps/10-template-fft-precision.md) | **complete** (10c offsets) | [#87](https://github.com/UCSC-Transients/dark-hunter_rv/issues/87) | `step/10-template-fft-precision`, `step/10c-method-offsets` | 2026-08-01 ([#88](https://github.com/UCSC-Transients/dark-hunter_rv/pull/88)); 10c via [#95](https://github.com/UCSC-Transients/dark-hunter_rv/pull/95)/[#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101) |
| 11 Epoch–epoch CCF matrix | [steps/11-epoch-ccf-matrix.md](steps/11-epoch-ccf-matrix.md) | in_progress (**11a–c on main**; 11d open) | [#94](https://github.com/UCSC-Transients/dark-hunter_rv/issues/94) | `step/11-epoch-ccf-matrix`, `step/11-epoch-ccf-matrix-cli` | 2026-08-02 ([#95](https://github.com/UCSC-Transients/dark-hunter_rv/pull/95)/[#98](https://github.com/UCSC-Transients/dark-hunter_rv/pull/98) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |

Update this file when an issue closes or a step status changes. Keep in sync with `.cursor/plans/rv-pipeline/INDEX.md`.
