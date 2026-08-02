# RV pipeline step tracker (GitHub)

**Step plans (rendered preview in IDE):** [steps/](steps/) — open any `.md` and use **Markdown: Open Preview** (Cmd+Shift+V).

**Order of attack (baselines first + parallel subagents):** [ATTACK_ORDER.md](ATTACK_ORDER.md)

**Orchestrator brief (launch subagents from this):** [ORCHESTRATOR.md](ORCHESTRATOR.md)

Orchestrator (local legacy): `.cursor/plans/rv_pipeline_master_plan_8447f2cd.plan.md`

Workflow: [WORKFLOW.md](WORKFLOW.md)

## Per-method precision lanes (current focus)

| Lane | Steps | Status |
|------|-------|--------|
| **mask_ccf** | 01, 02a, 09 | Defaults → `subchunks_8`; bias keep Jun-16; **step 01 complete (waivers)** |
| **template_fft** | 10 | **Complete** (#88 + 10c offsets `method_rv_offsets.txt`) |
| **strong_lines** | 06 | **Complete** ([#90](https://github.com/UCSC-Transients/dark-hunter_rv/pull/90)) — Hβ-primary Balmer; metals + Q×SNR² IVW |
| **epoch_ccf** (relative) | 11 | **matrix CLI done** (local `be9b0a3`); short-pair QC next ([#94](https://github.com/UCSC-Transients/dark-hunter_rv/issues/94)) |
| **fusion / adoption** | 03, 04 | **03+04 complete locally** (`9de6c68`, `fa0578c`); PRs pending |

| Step | Plan | Status | Issue | Branch(es) | Merged |
|------|------|--------|-------|------------|--------|
| 00 Literature RV master | [steps/00-literature-rv-master.md](steps/00-literature-rv-master.md) | completed | [#37](https://github.com/astrofoley/dark-hunter_rv/issues/37) | `step/00-literature-rv-master` | 2026-06-07 ([#46](https://github.com/astrofoley/dark-hunter_rv/pull/46)) |
| 01 Benchmark cool precision | [steps/01-benchmark-cool-precision.md](steps/01-benchmark-cool-precision.md) | **complete** (waivers) | [#38](https://github.com/UCSC-Transients/dark-hunter_rv/issues/38) | `step/01-benchmark-cool-precision`, `step/01-cool-closeout` | local `5c352e4` |
| 02 Chunk weights / subchunks | [steps/02-chunk-weights-subchunks.md](steps/02-chunk-weights-subchunks.md) | in_progress (02a verified; rebuild deferred pre-final; 02b later) | [#39](https://github.com/astrofoley/dark-hunter_rv/issues/39) | `step/02a-subchunk-study`, `step/02a-bias-defer-verify`, `step/02b-trust-weights-stack` | — |
| 03 Method fusion / coverage | [steps/03-method-fusion-coverage.md](steps/03-method-fusion-coverage.md) | **complete** (local) | [#40](https://github.com/UCSC-Transients/dark-hunter_rv/issues/40) | `step/03-method-fusion-coverage` | local `9de6c68` |
| 04 Adopted-RV match plots | [steps/04-adopted-rv-match-plots.md](steps/04-adopted-rv-match-plots.md) | **complete** (local; visual residual) | [#41](https://github.com/UCSC-Transients/dark-hunter_rv/issues/41) | `step/04-adopted-rv-match-plots` | local `fa0578c` |
| 05 Short-pair QC | [steps/05-short-pair-epoch-ccf.md](steps/05-short-pair-epoch-ccf.md) | in_progress (05a) | [#42](https://github.com/UCSC-Transients/dark-hunter_rv/issues/42) | `step/05a-short-pair-calibration` | — |
| 06 Strong-line line list | [steps/06-strong-line-line-list.md](steps/06-strong-line-line-list.md) | **complete** | [#43](https://github.com/UCSC-Transients/dark-hunter_rv/issues/43), [#91](https://github.com/UCSC-Transients/dark-hunter_rv/issues/91)–[#93](https://github.com/UCSC-Transients/dark-hunter_rv/issues/93) | `step/06-strong-line-teff-sweep` | 2026-08-02 ([#90](https://github.com/UCSC-Transients/dark-hunter_rv/pull/90)) |
| 07 SB2 search | [steps/07-sb2-search.md](steps/07-sb2-search.md) | pending | [#44](https://github.com/astrofoley/dark-hunter_rv/issues/44) | `step/07a-sb2-detection`, `step/07b-sb2-reporting` | — |
| 08 External RV cross-check | [steps/08-external-rv-crosscheck.md](steps/08-external-rv-crosscheck.md) | in_progress (**lite done**; full later) | [#45](https://github.com/UCSC-Transients/dark-hunter_rv/issues/45) | `step/08-external-rv-crosscheck` | lite local `a6ffe75` |
| 09 CCF RV estimator (mask) | [steps/09-ccf-rv-estimator.md](steps/09-ccf-rv-estimator.md) | complete (`gauss_offset`) | — | — | — |
| 10 Template FFT precision | [steps/10-template-fft-precision.md](steps/10-template-fft-precision.md) | **complete** (10c offsets) | [#87](https://github.com/UCSC-Transients/dark-hunter_rv/issues/87) | `step/10-template-fft-precision`, `step/10c-method-offsets` | 2026-08-01 ([#88](https://github.com/UCSC-Transients/dark-hunter_rv/pull/88)); 10c local `18d6cec` |
| 11 Epoch–epoch CCF matrix | [steps/11-epoch-ccf-matrix.md](steps/11-epoch-ccf-matrix.md) | in_progress (11a–c local) | [#94](https://github.com/UCSC-Transients/dark-hunter_rv/issues/94) | `step/11-epoch-ccf-matrix`, `step/11-epoch-ccf-matrix-cli` | local `be9b0a3` |

Update this file when an issue closes or a step status changes. Keep in sync with `.cursor/plans/rv-pipeline/INDEX.md`.
