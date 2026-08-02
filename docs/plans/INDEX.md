# RV pipeline step tracker (GitHub)

**Step plans (rendered preview in IDE):** [steps/](steps/) — open any `.md` and use **Markdown: Open Preview** (Cmd+Shift+V).

**Order of attack (baselines first + parallel subagents):** [ATTACK_ORDER.md](ATTACK_ORDER.md)

**Orchestrator brief (launch subagents from this):** [ORCHESTRATOR.md](ORCHESTRATOR.md)

**Human gates (remaining decisions):** [HUMAN_GATES.md](HUMAN_GATES.md)

Orchestrator (local legacy): `.cursor/plans/rv_pipeline_master_plan_8447f2cd.plan.md`

Workflow: [WORKFLOW.md](WORKFLOW.md)

## Per-method precision lanes (current focus)

| Lane | Steps | Status |
|------|-------|--------|
| **mask_ccf** | 01, 02a, 09 | `subchunks_8`; **trust weights default on**; bias rebuild deferred ([#57](https://github.com/UCSC-Transients/dark-hunter_rv/issues/57)) |
| **template_fft** | 10 | **Complete** (#88 + 10c via #95/#101) |
| **strong_lines** | 06 | **Complete** ([#90](https://github.com/UCSC-Transients/dark-hunter_rv/pull/90)) |
| **epoch_ccf** (relative) | 11 | Cascade after strong_lines; always-run multi-epoch + abs/rel discord flags |
| **fusion / adoption** | 03, 04 | **Complete** (#97 → #101) |

| Step | Plan | Status | Issue | Branch(es) | Merged |
|------|------|--------|-------|------------|--------|
| 00 Literature RV master | [steps/00-literature-rv-master.md](steps/00-literature-rv-master.md) | completed | [#37](https://github.com/astrofoley/dark-hunter_rv/issues/37) | `step/00-literature-rv-master` | 2026-06-07 ([#46](https://github.com/astrofoley/dark-hunter_rv/pull/46)) |
| 01 Benchmark cool precision | [steps/01-benchmark-cool-precision.md](steps/01-benchmark-cool-precision.md) | **complete** (waivers) | [#38](https://github.com/UCSC-Transients/dark-hunter_rv/issues/38) closed | `step/01-cool-closeout` | 2026-08-02 ([#96](https://github.com/UCSC-Transients/dark-hunter_rv/pull/96) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 02 Chunk weights / subchunks | [steps/02-chunk-weights-subchunks.md](steps/02-chunk-weights-subchunks.md) | **02b default on**; **bias rebuild open** | [#39](https://github.com/UCSC-Transients/dark-hunter_rv/issues/39), [#57](https://github.com/UCSC-Transients/dark-hunter_rv/issues/57) | `step/02b-trust-weights-stack` | 2026-08-02 ([#95](https://github.com/UCSC-Transients/dark-hunter_rv/pull/95)/[#99](https://github.com/UCSC-Transients/dark-hunter_rv/pull/99) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)); trust default-on via gate-flips |
| 03 Method fusion / coverage | [steps/03-method-fusion-coverage.md](steps/03-method-fusion-coverage.md) | **complete** | [#40](https://github.com/UCSC-Transients/dark-hunter_rv/issues/40) closed | `step/03-method-fusion-coverage` | 2026-08-02 ([#97](https://github.com/UCSC-Transients/dark-hunter_rv/pull/97) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 04 Adopted-RV match plots | [steps/04-adopted-rv-match-plots.md](steps/04-adopted-rv-match-plots.md) | **complete** | [#41](https://github.com/UCSC-Transients/dark-hunter_rv/issues/41) closed | `step/04-adopted-rv-match-plots` | 2026-08-02 ([#97](https://github.com/UCSC-Transients/dark-hunter_rv/pull/97) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 05 Short-pair QC | [steps/05-short-pair-epoch-ccf.md](steps/05-short-pair-epoch-ccf.md) | **complete** | [#42](https://github.com/UCSC-Transients/dark-hunter_rv/issues/42) closed | `step/05a-short-pair-calibration` | 2026-08-02 ([#98](https://github.com/UCSC-Transients/dark-hunter_rv/pull/98) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)) |
| 06 Strong-line line list | [steps/06-strong-line-line-list.md](steps/06-strong-line-line-list.md) | **complete** | [#43](https://github.com/UCSC-Transients/dark-hunter_rv/issues/43) closed | `step/06-strong-line-teff-sweep` | 2026-08-02 ([#90](https://github.com/UCSC-Transients/dark-hunter_rv/pull/90)) |
| 07 SB2 search | [steps/07-sb2-search.md](steps/07-sb2-search.md) | **complete** (optional full-refit rates) | [#44](https://github.com/UCSC-Transients/dark-hunter_rv/issues/44) closed | — | 2026-08-02 ([#99](https://github.com/UCSC-Transients/dark-hunter_rv/pull/99)/[#103](https://github.com/UCSC-Transients/dark-hunter_rv/pull/103)/[#105](https://github.com/UCSC-Transients/dark-hunter_rv/pull/105)) |
| 08 External RV cross-check | [steps/08-external-rv-crosscheck.md](steps/08-external-rv-crosscheck.md) | **waiting** (n=8; want ≥10) | [#45](https://github.com/UCSC-Transients/dark-hunter_rv/issues/45) | — | 2026-08-02 ([#96](https://github.com/UCSC-Transients/dark-hunter_rv/pull/96)/[#104](https://github.com/UCSC-Transients/dark-hunter_rv/pull/104)) |
| 09 CCF RV estimator (mask) | [steps/09-ccf-rv-estimator.md](steps/09-ccf-rv-estimator.md) | complete (`gauss_offset`) | — | — | — |
| 10 Template FFT precision | [steps/10-template-fft-precision.md](steps/10-template-fft-precision.md) | **complete** | [#87](https://github.com/UCSC-Transients/dark-hunter_rv/issues/87) closed | — | 2026-08-01 ([#88](https://github.com/UCSC-Transients/dark-hunter_rv/pull/88)); 10c via [#95](https://github.com/UCSC-Transients/dark-hunter_rv/pull/95) |
| 11 Epoch–epoch CCF matrix | [steps/11-epoch-ccf-matrix.md](steps/11-epoch-ccf-matrix.md) | **in_progress** (cascade + always-run + discord) | [#94](https://github.com/UCSC-Transients/dark-hunter_rv/issues/94) | `phase/gate-flips-epoch-trust` | 2026-08-02 ([#95](https://github.com/UCSC-Transients/dark-hunter_rv/pull/95)/[#98](https://github.com/UCSC-Transients/dark-hunter_rv/pull/98)/[#103](https://github.com/UCSC-Transients/dark-hunter_rv/pull/103)); gate flips PR |

Update this file when an issue closes or a step status changes. Keep in sync with `.cursor/plans/rv-pipeline/INDEX.md`.
