# RV pipeline orchestrator brief

**Audience:** A separate orchestrator agent that launches and coordinates **subagents** until the program in this file is complete.  
**Repo root:** `/Users/rfoley/darkhunter/rvs/dark-hunter_rv`  
**Data root:** `/Users/rfoley/darkhunter/rvs/data`  
**GitHub:** `UCSC-Transients/dark-hunter_rv`  
**Last updated:** 2026-08-01

This file is the **single entrypoint**. It links every step plan, command, constraint, and subagent contract. If something is detailed elsewhere, the link is authoritative; if something is only here, treat it as binding.

**Always-on skills:** orchestrator and every subagent must read and use **strict-workflow** and **caveman** (§0.3) for the whole run.

---

## 0. Orchestrator mandate

### 0.1 Mission (north star)

1. Deliver **reliable per-method absolute RVs** (mask_ccf, template_fft, strong_lines) on the campaign / bias-train cohort.
2. Deliver **honest method fusion** (adopted RV + σ + reject reasons).
3. Deliver **epoch–epoch CCF matrix** relative RVs that fill missing absolutes when ≥1 absolute anchor exists, and remain useful with zero anchors.
4. Finish **open closeouts** before speculative polish (trust weights, SB2, full catalogs).

### 0.2 How to operate

1. **Before any other work:** read and follow **both** required skills (full text):
   - `/Users/rfoley/darkhunter/rvs/dark-hunter_rv/.cursor/skills/strict-workflow/SKILL.md`
   - `/Users/rfoley/darkhunter/rvs/dark-hunter_rv/.cursor/skills/caveman/SKILL.md`  
   Also acceptable mirrors: `.cursor/skills/strict-workflow/SKILL.md`, `.cursor/skills/caveman/SKILL.md` under repo root; user-global copies under `~/.cursor/skills/` if present — **repo copies win** when both exist.
2. Read this file top-to-bottom once per session; then refresh [INDEX.md](INDEX.md) and live GitHub state.
3. Launch **only** subagents whose **Start when** gate is true.
4. Give each subagent a **copy of its launch card** (below) plus the linked step `.md` — do not invent scope. **Every subagent prompt must instruct that agent to read and use the same two skills** (§0.3) before coding or reporting.
5. After each subagent returns: update checkboxes here / INDEX / step frontmatter; run the listed verification; commit per §0.4.
6. **Never** merge PRs while CI is red or still running unless the human explicitly overrides.
7. **Never** push unless the human asks (local `[AI Checkpoint]` commits are the default).
8. User domain knowledge wins; contradict only with empirical evidence from live files/logs.

### 0.3 Required skills (orchestrator **and every subagent**)

**Mandatory.** Orchestrator and **all** subagents always use:

| Skill | Absolute path (prefer) | What it enforces |
|-------|------------------------|------------------|
| **strict-workflow** | `/Users/rfoley/darkhunter/rvs/dark-hunter_rv/.cursor/skills/strict-workflow/SKILL.md` | Domain authority; verify live files; exact scope; Plan&Halt for *out-of-scope* new work; issues/PRs; pytest + `[AI Checkpoint]` micro-commits; no push unless asked; typing/docs standards |
| **caveman** | `/Users/rfoley/darkhunter/rvs/dark-hunter_rv/.cursor/skills/caveman/SKILL.md` | Terse communication (default **full**); keep full technical accuracy; no fluff; code/errors verbatim |

**Subagent launch requirement:** First lines of every subagent prompt must be:

```text
REQUIRED SKILLS (read full files now, then obey for entire task):
1. /Users/rfoley/darkhunter/rvs/dark-hunter_rv/.cursor/skills/strict-workflow/SKILL.md
2. /Users/rfoley/darkhunter/rvs/dark-hunter_rv/.cursor/skills/caveman/SKILL.md
Reply in caveman (full). Checkpoint commits after pytest. Do not push unless told.
```

Do not launch a subagent without that block. If a subagent ignores skills, re-prompt with the block and the failing constraint.

Additional rules (also follow):

| Rule / doc | Path |
|------------|------|
| Executable commands use concrete paths (no `/path/to/...`) | `.cursor/rules/executable-commands.mdc` |
| No inline imports | cursor-team-kit `no-inline-imports` |
| Step workflow (branch/PR/propagation) | [WORKFLOW.md](WORKFLOW.md) |
| Attack order (science sequencing) | [ATTACK_ORDER.md](ATTACK_ORDER.md) |

**Note on Plan & Halt:** This orchestrator brief **is** the approved plan for the work listed here. Subagents implementing listed tasks do **not** need to re-plan-and-halt for in-scope work. They **must** halt and ask before expanding scope, retuning frozen continuum, or changing chunk layout.

### 0.4 Git / PR protocol

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
# Prefer feature branches: step/NN-<slug> from up-to-date main after #90 merges
git checkout main && git pull
git checkout -b step/NN-<slug>   # or continue existing step branch

# After pytest green:
git add <relevant files>
git commit -m "$(cat <<'EOF'
[AI Checkpoint] <precise description>

EOF
)"
# Push / PR only when human requests:
# git push -u origin HEAD
# gh pr create ...  # see WORKFLOW.md
```

- PR title: `step/NN: <title>`
- PR body: `Closes #<issue>` + link to `docs/plans/steps/NN-….md`
- Do **not** include untracked SB2 WIP, logs, `bias_statistics*.bak*`, or data dumps unless asked.

### 0.5 Frozen / anti-goals (absolute)

Do **not** prioritize or change without explicit human OK:

- Continuum product: **`split`** frozen (mask: `sinc_blaze_only`; template/strong: `sinc_blaze`).
- Mask chunk layout: **`calibration/chunk_layouts/subchunks_8.yaml`** — no retile.
- ML fusion scorer.
- Making epoch CCF the sole absolute zeropoint (it is **relative-first**).
- Force-push / `--no-verify` / amend rules (see user git rules).

---

## 1. Document and artifact catalog

### 1.1 Plans (read these)

| Doc | Role |
|-----|------|
| **This file** | Orchestrator entrypoint + subagent cards |
| [ATTACK_ORDER.md](ATTACK_ORDER.md) | Phase sequencing + parallel map |
| [INDEX.md](INDEX.md) | Live step status table |
| [WORKFLOW.md](WORKFLOW.md) | Start/finish step, branch, propagation |
| [steps/00-literature-rv-master.md](steps/00-literature-rv-master.md) | Lit master (**done**) |
| [steps/01-benchmark-cool-precision.md](steps/01-benchmark-cool-precision.md) | Mask cool precision / Phase A |
| [steps/02-chunk-weights-subchunks.md](steps/02-chunk-weights-subchunks.md) | 02a done; 02b deferred |
| [steps/03-method-fusion-coverage.md](steps/03-method-fusion-coverage.md) | Fusion |
| [steps/04-adopted-rv-match-plots.md](steps/04-adopted-rv-match-plots.md) | Adopted RV plots |
| [steps/05-short-pair-epoch-ccf.md](steps/05-short-pair-epoch-ccf.md) | Short-pair **QC** (05b → 11) |
| [steps/06-strong-line-line-list.md](steps/06-strong-line-line-list.md) | Strong lines IVW |
| [steps/07-sb2-search.md](steps/07-sb2-search.md) | SB2 (Phase 5) |
| [steps/08-external-rv-crosscheck.md](steps/08-external-rv-crosscheck.md) | Lit/catalog check (08 lite early) |
| [steps/09-ccf-rv-estimator.md](steps/09-ccf-rv-estimator.md) | Mask estimator (**done**, `gauss_offset`) |
| [steps/10-template-fft-precision.md](steps/10-template-fft-precision.md) | Template lane; **10c open** |
| [steps/11-epoch-ccf-matrix.md](steps/11-epoch-ccf-matrix.md) | Epoch×epoch CCF + abs fill |

### 1.2 Science / ops docs

| Doc | Role |
|-----|------|
| [docs/operations.md](../operations.md) | Calibration + production ops |
| [docs/validation_playbook.md](../validation_playbook.md) | How to run validation |
| [docs/rv_methods_evaluation.md](../rv_methods_evaluation.md) | Method evaluation / overlap |
| [docs/broad_line_method.md](../broad_line_method.md) | Strong-line product |
| [docs/external_rv_sources.md](../external_rv_sources.md) | External catalogs |
| [docs/contributing.md](../contributing.md) | Dev conventions |
| [calibration/mask_lane_deploy.md](../../calibration/mask_lane_deploy.md) | Mask deploy + bias rebuild |

### 1.3 Key calibration / config files

| Path | Purpose |
|------|---------|
| `calibration/chunk_layouts/subchunks_8.yaml` | Production chunk layout |
| `calibration/bias_train.txt` | Bias training spectrum list (114) |
| `bias_statistics.txt` (repo root) | Per-chunk mask debias (rebuild via script) |
| `calibration/strong_line_offsets.txt` | Strong-line offset + **quality Q** |
| `method_rv_offsets.txt` (repo root, optional) | Global template/strong offsets vs mask |
| `calibration/literature_rv_master.csv` | Literature RVs (step 00) |
| `calibration/blaze_orders_apf.json` | Blaze for continuum split |
| `validation_output/chunk_campaign/spectrum_list.txt` | 114-stem campaign list |

### 1.4 Important validation outputs (reference, do not delete)

| Path | Meaning |
|------|---------|
| `validation_output/template_fft_baseline/pipeline_blaze_split/` | Pre-#88 template baseline (**do not use refreshed overlap_blaze_split as pre-fix**) |
| `validation_output/template_fft_baseline/pipeline_cool_vsini12_mhfix/` | Post-#88 confirm + FULL_COMPARISON.md |
| `validation_output/strong_line_teff_sweep/` | Balmer Teff sweep |
| `validation_output/strong_line_candidate_sweep/` | Metal keep/exclude + Q calibration source |

### 1.5 GitHub issues / PRs (as of 2026-08-01)

| ID | Topic | State note |
|----|-------|------------|
| [#90](https://github.com/UCSC-Transients/dark-hunter_rv/pull/90) | Strong-line product IVW | **MERGED** 2026-08-02 |
| [#43](https://github.com/UCSC-Transients/dark-hunter_rv/issues/43) | Step 06 | **CLOSED** via #90 |
| [#91](https://github.com/UCSC-Transients/dark-hunter_rv/issues/91)–[#93](https://github.com/UCSC-Transients/dark-hunter_rv/issues/93) | Wire / inclusion / Q×SNR² | **CLOSED** via #90 |
| [#38](https://github.com/astrofoley/dark-hunter_rv/issues/38) | Step 01 | in_progress |
| [#39](https://github.com/astrofoley/dark-hunter_rv/issues/39) | Step 02 | 02a closeout + 02b later |
| [#40](https://github.com/astrofoley/dark-hunter_rv/issues/40) | Step 03 fusion | pending |
| [#41](https://github.com/astrofoley/dark-hunter_rv/issues/41) | Step 04 plots | pending |
| [#42](https://github.com/astrofoley/dark-hunter_rv/issues/42) | Step 05 short-pair | pending (scope → 05a) |
| [#44](https://github.com/astrofoley/dark-hunter_rv/issues/44) | Step 07 SB2 | Phase 5 |
| [#45](https://github.com/astrofoley/dark-hunter_rv/issues/45) | Step 08 external | 08 lite early OK |
| [#87](https://github.com/UCSC-Transients/dark-hunter_rv/issues/87) | Step 10 template | 10c offsets open |
| Step 11 | Epoch CCF | **Create issue when starting** |

Refresh with:

```bash
gh pr view 90 --json state,statusCheckRollup,url
gh issue list --repo UCSC-Transients/dark-hunter_rv --limit 30
```

---

## 2. Method recipes (product truth)

### 2.1 mask_ccf

- Layout: `subchunks_8`
- Continuum: `--continuum-mode split` → mask uses `sinc_blaze_only`
- Estimator: `gauss_offset`
- Debias: `bias_statistics.txt` per `chunk_key`
- Deploy doc: `calibration/mask_lane_deploy.md`

### 2.2 template_fft

- Same chunk YAML; continuum `sinc_blaze` under split
- Cool vsini rejected/nonfinite → 12 km/s; MH parse fixed (#88)
- Offsets: `method_rv_offsets.txt` with **mask as truth** (10c)
- Confirm metrics: median \|mask−template\| ~1.86 km/s, MAD ~0.89 (FULL_COMPARISON.md)

### 2.3 strong_lines

- Product list: Hβ → MgIb2 → CaI6122 → CaI6162 → MgIb3 → CaI4227 → Hγ → Hδ → Hα
- Inclusion: depth, width, err, telluric, continuum `median(flux/eflux)` near line
- Weight: \(w = Q_{\mathrm{line}} \times (\mathrm{S/N}_{\mathrm{near\,line}})^2\)
- File: `calibration/strong_line_offsets.txt`
- Pipeline: `read_strong_line_calibration` → `combine_strong_line_rvs` → one diagnostics row `method=strong_lines`, `qc_reason=ivw_n=…`

### 2.4 epoch_ccf (step 11 — to build)

- All epochs × all epochs CCF + auto-correlation diagonal
- Matrix \(\Delta v_{ij},\sigma_{ij}\); combine with sparse absolutes \(A_i\) via WLS
- Zero anchors → relative-only (fixable zeropoint)
- Details: [steps/11-epoch-ccf-matrix.md](steps/11-epoch-ccf-matrix.md)

---

## 3. Phase machine (execute in order; parallelize per gates)

```text
Phase 0  Closeouts (#90, bias, 10c)
Phase 1  Absolute baselines (mask / template / strong) + step11 spike
Phase 2  Absolute validation (08 lite, 01 writeup)
Phase 3  Fusion 03 → plots 04
Phase 4  Epoch CCF matrix product + 05a short-pair QC
Phase 5  02b trust weights, 07 SB2, 08 full
```

### Progress checklist (orchestrator maintains)

- [x] **P0a** PR #90 CI green + merged; step 06 → complete; INDEX updated
- [x] **P0b** Orchestrator plans on main via [#95](https://github.com/UCSC-Transients/dark-hunter_rv/pull/95) / [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101)
- [~] **P0c** Keep committed Jun-16 `bias_statistics.txt` (s8). **Fresh rebuild deferred to pre-final product.** Ziggy: alert human only (see §12.1)
- [x] **P0d** `method_rv_offsets.txt` on main
- [x] **P1** Absolute baselines on main (#95)
- [x] **P2** 01 + 08 lite on main (#96 → #101)
- [x] **P3** Fusion + adopted plots on main (#97 → #101)
- [x] **P4** Matrix CLI + short-pair QC on main (#98 → #101)
- [x] **P5** Trust / SB2 / 08-full on main (#99 → #101) — residuals remain (below)

### NEXT (orchestrator — 2026-08-02 post-#105)

1. ~~#103 / #104 / #105~~ **MERGED** (soft residuals through NSS dump `a7f2b8d`).
2. **Code DoD largely met.** Remaining = human/data gates — see [HUMAN_GATES.md](HUMAN_GATES.md).
3. Closed issues #40/#41/#42 after stack land; #44/#45/#94 stay open with residual notes.
4. Do **not** enable default epoch_ccf adopt, trust-on, or ziggy bias rebuild without table answers.

### Branch map (historical; stack landed)

| Phase | PRs | Landed on main |
|-------|-----|----------------|
| 1 | [#95](https://github.com/UCSC-Transients/dark-hunter_rv/pull/95) | 2026-08-02 |
| 2–5 stack | [#96](https://github.com/UCSC-Transients/dark-hunter_rv/pull/96)–[#99](https://github.com/UCSC-Transients/dark-hunter_rv/pull/99) → [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101) | 2026-08-02 `c6801f7` |
| Soft residuals | [#103](https://github.com/UCSC-Transients/dark-hunter_rv/pull/103)–[#105](https://github.com/UCSC-Transients/dark-hunter_rv/pull/105) | 2026-08-02 |

### 0.6 Human session policy (2026-08-02)

Binding for this orchestrator run (resume-safe):

| Topic | Decision |
|-------|----------|
| P0b docs | **DONE** — on main via #95/#101 |
| Bias rebuild | Keep current file; rebuild before final product |
| Ziggy | Never run; chat alert + log in this file (§10 / §12.1). Pause card if ziggy output required |
| Issue #87 | Leave closed; finish 10c; comment results; mark step 10 complete when verified |
| Step 11 issue | Create on start (`UCSC-Transients/dark-hunter_rv`) |
| Agent control | Orchestrator chooses launch/kill/parallel |
| Run length | Through program DoD including Phase 5 (02b, SB2, 08-full) |
| Git | `[AI Checkpoint]` local commits only — no PR for checkpoints |
| PRs | No Phase 0 PR. **PR required per Phase 1–5.** Significant cards also get PRs (multi-PR per card OK if sandboxed). Ask human before push/PR |
| SB2 | Inventory + **reuse** existing untracked `sb2*` code |
| INDEX/ORCHESTRATOR | Orchestrator owns after card returns; subagents update **their step md only** |
| Worktrees | **Each subagent MUST use an isolated `git worktree`** — never `git checkout` the primary orchestrator worktree |

---

## 4. Absolute baseline exit criteria

| Method | Finite product | Calibration | Sanity gate |
|--------|----------------|-------------|-------------|
| mask_ccf | chunk IVW + bias | `bias_statistics.txt` current for s8 | Phase A / cool benchmark (step 01) |
| template_fft | bank + vsini path | `method_rv_offsets.txt` | mask−template MAD ≲ few km/s on cool 114 |
| strong_lines | ≥1 included line IVW | `strong_line_offsets.txt` Q+offset | `ivw_n=` in diagnostics; unit tests green |
| epoch_ccf | relative matrix | abs fill when anchors | diagonal ~0; short-pair Δ≈0 |

---

## 5. File ownership (avoid parallel collisions)

| Owner subagent | May write | Must not write |
|----------------|-----------|----------------|
| MASK-DEPLOY | `bias_statistics.txt`, `scripts/rebuild_mask_bias.sh` notes, `calibration/mask_lane_deploy.md`, step 02 plan | `method_rv_offsets.txt`, `strong_lines.py` |
| TEMPLATE-OFFSETS | `method_rv_offsets.txt`, step 10 plan, ops playbook template section | `bias_statistics.txt` |
| STRONG-QA | step 06 / INDEX after merge, optional validation_output strong reports | fusion module |
| EPOCH-CCF-* | `darkhunter_rv/epoch_ccf.py`, `validation/epoch_ccf_matrix.py`, tests, step 11 | production adoption defaults until Phase 4 |
| FUSION | `darkhunter_rv/method_fusion.py`, diagnostics report coverage | bias rebuild |
| PLOTS | `plotting.py` adopted match, pipeline plot hooks | fusion math |
| LIT-08 | `validation/compare_literature_rvs.py` (or equiv), playbook | SB2 detection |

If two subagents need the same file: **serialize** or split into sequential PRs.

---

## 6. Subagent launch cards

Copy the card into the subagent prompt. Fill `BRANCH` and `ISSUE`.

**Every launch** must begin with the §0.3 required-skills block (strict-workflow + caveman). Then paste the card. Subagent must read both `SKILL.md` files before doing work.

---

### CARD P0-MERGE-90 — Human / orchestrator gate

**Start when:** `gh pr checks 90` all green (or pytest conclusion success).  
**Do:**

```bash
gh pr checks 90
gh pr merge 90 --merge   # only if human approved merge policy; else ask human
```

**Then update:**

- `docs/plans/steps/06-strong-line-line-list.md` → `status: complete`
- `docs/plans/INDEX.md` Merged column + strong_lines lane
- Confirm issues #43/#91/#92/#93 closed by PR

**Must not:** merge on red/in-progress CI.

---

### CARD MASK-DEPLOY (Phase 0c / 1A)

**Start when:** Anytime (independent of #90); prefer after main is clean.  
**Step plan:** [02-chunk-weights-subchunks.md](steps/02-chunk-weights-subchunks.md) · [mask_lane_deploy.md](../../calibration/mask_lane_deploy.md)  
**Issue:** #39  
**Branch:** `step/02a-subchunk-study` or `step/02a-bias-commit`

**Tasks:**

1. Rebuild mask bias for `subchunks_8`:

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
bash scripts/rebuild_mask_bias.sh
# If orders already exist for bias_train:
# SKIP_PIPELINE=1 bash scripts/rebuild_mask_bias.sh
```

2. Commit `bias_statistics.txt` if changed and sane (`tests/validation/test_build_bias_set.py`).
3. Document ziggy refit:

```bash
# On ziggy (paths from mask_lane_deploy.md):
# cd /data2/darkhunter/dark-hunter_rv && git pull
# bash scripts/refit_all_per_object_parallel.sh
```

4. Check off 02a “Rebuild + commit bias” / “Refit catalog” in step 02 plan.
5. Pytest: `PYTHONPATH=. python -m pytest tests/validation/test_build_bias_set.py -q`

**Done when:** bias file committed; deploy doc still accurate; step 02a closeout boxes checked.  
**Must not:** implement 02b trust weights; touch template offsets.

---

### CARD TEMPLATE-OFFSETS (Phase 0d / 1B)

**Start when:** Post-#88 code on main (already merged). Needs diagnostics with mask+template.  
**Step plan:** [10-template-fft-precision.md](steps/10-template-fft-precision.md) §10c  
**Issue:** #87  
**Branch:** `step/10-template-fft-precision` or `step/10c-method-offsets`

**Tasks:**

1. Compute offsets (mask = truth), e.g.:

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python -m validation.compute_method_rv_offsets \
  --instrument APF \
  --diagnostics-glob 'validation_output/template_fft_baseline/pipeline_cool_vsini12_mhfix/**/*_diagnostics.csv' \
  --out method_rv_offsets.txt
# Adjust glob to whichever post-#88 diagnostics exist; see compute_method_rv_offsets --help
```

2. Ensure pipeline reads offsets (`config` / `io_utils.read_method_rv_offsets`).
3. Update `docs/operations.md` + step 10c checkboxes; record residual improvement if any.
4. Tests: `PYTHONPATH=. python -m pytest tests/test_method_rv_offsets.py -q`

**Done when:** `method_rv_offsets.txt` in repo (or documented install path); 10c tasks checked; step 10 can be marked complete if human agrees.  
**Must not:** rebuild `bias_statistics.txt`; change vsini/MH knobs without A/B.

---

### CARD STRONG-QA (Phase 1C)

**Start when:** #90 **merged**.  
**Step plan:** [06-strong-line-line-list.md](steps/06-strong-line-line-list.md) · [broad_line_method.md](../broad_line_method.md)  
**Branch:** docs on main or short `step/06-post-merge-qa`

**Tasks:**

1. `PYTHONPATH=. python -m pytest tests/test_strong_lines_product.py tests/test_h_beta_rv.py -q`
2. Spot-check one spectrum with `--run-all-methods`; confirm diagnostics `method=strong_lines` and `qc_reason` starts with `ivw_n=`.
3. Confirm `calibration/strong_line_offsets.txt` Q table matches plan.
4. Mark step 06 propagation checklist; INDEX complete.

**Done when:** smoke OK; INDEX/step 06 complete post-merge.  
**Must not:** redesign Q formula; expand line list without issue.

---

### CARD EPOCH-SPIKE (Phase 1D — can start **now**)

**Start when:** Immediately (no dependency on #90).  
**Step plan:** [11-epoch-ccf-matrix.md](steps/11-epoch-ccf-matrix.md)  
**Issue:** Create GitHub issue “RV step 11: Epoch–epoch CCF matrix” under `UCSC-Transients/dark-hunter_rv`; set frontmatter `github_issue`.  
**Branch:** `step/11-epoch-ccf-matrix`

**Tasks (spike = 11a engine + 11c combiner tests, not full pipeline default):**

1. Implement `darkhunter_rv/epoch_ccf.py`: pair CCF API + WLS abs/rel combiner (synthetic first).
2. Tests: Doppler recovery; \(\Delta v_{ji}=-\Delta v_{ij}\); one-abs fill; zero-abs relative-only.
3. Do **not** enable as default adopted RV yet.

**Done when:** pytest green for synthetic suite; issue filed; step 11 status `in_progress`.  
**Must not:** depend on fusion; change mask/template defaults.

---

### CARD EPOCH-MATRIX (Phase 4 — after spike)

**Start when:** EPOCH-SPIKE merged or solid on branch; multi-epoch spectra available.  
**Tasks:** `validation/epoch_ccf_matrix.py` all×all + auto-corr; persist long-form CSV/npz; pair-parallel OK.  
**Validate:** short-pair Δ≈0; compare to \(A_i-A_j\) when both abs finite.  
**Sub-split OK:** one agent I/O+CLI, one agent CCF physics (if not done).

---

### CARD LIT-08-LITE (Phase 2)

**Start when:** Mask (and ideally template) baselines usable; **do not wait for SB2**.  
**Step plan:** [08-external-rv-crosscheck.md](steps/08-external-rv-crosscheck.md) · lit CSV `calibration/literature_rv_master.csv`  
**Issue:** #45  

**Tasks:** Compare pipeline mask/template (strong optional) to El-Badry master; per-star bias/RMS tables; playbook recipe. Soften hard `depends_on: 07` in step 08 frontmatter for lite path.

**Done when:** CLI report runs on overlapping Gaia IDs; documented output path under `validation_output/`.

---

### CARD BENCH-01 (Phase 2)

**Start when:** Deployed mask bias in place (MASK-DEPLOY done or verified current).  
**Step plan:** [01-benchmark-cool-precision.md](steps/01-benchmark-cool-precision.md)  
**Issue:** #38  

**Tasks:** Close remaining writeup / Phase A gate summary against `subchunks_8` + current bias; mark step complete if acceptance met.

---

### CARD FUSION-03 (Phase 3)

**Start when:** P1 absolute exit criteria met for methods that feed adoption (mask required; template+strong strongly preferred).  
**Step plan:** [03-method-fusion-coverage.md](steps/03-method-fusion-coverage.md)  
**Issue:** #40  
**Branch:** `step/03-method-fusion-coverage`

**Tasks:**

1. Add `darkhunter_rv/method_fusion.py` (bias surfaces, σ inflation, discordance gates).
2. Coverage CSV in `rv_method_diagnostics_report` (`binned_method_coverage_vs_teff.csv`).
3. Tests `tests/test_method_fusion.py`; document in `docs/rv_methods_evaluation.md`.

**Done when:** acceptance criteria in step 03 checked.  
**Must not:** change raw per-method RVs without calibrated columns.

---

### CARD PLOTS-04 (Phase 3)

**Start when:** Fusion column / adopted RV schema stable (can scaffold earlier).  
**Step plan:** [04-adopted-rv-match-plots.md](steps/04-adopted-rv-match-plots.md)  
**Issue:** #41  

**Tasks:** `plot_adopted_rv_match`; pipeline `--plots` / `--plots-focus` hook; smoke test.

---

### CARD SHORTPAIR-05A (Phase 4/5)

**Start when:** Step 11 matrix exists (or pairwise CCF).  
**Step plan:** [05-short-pair-epoch-ccf.md](steps/05-short-pair-epoch-ccf.md)  
**Issue:** #42  

**Tasks:** `validation/find_short_pairs.py`; report abs ΔRV and epoch-CCF ΔRV on Δt≈0 pairs; inflate \(\sigma_{ij}\) from scatter.

---

### CARD TRUST-02B (Phase 5)

**Start when:** Fusion metrics stable.  
**Step plan:** step 02 §02b  
**Must not:** start before Phase 3.

---

### CARD SB2-07 (Phase 5)

**Start when:** Clean abs/rel residuals; step 06 complete.  
**Step plan:** [07-sb2-search.md](steps/07-sb2-search.md)  
**Note:** Untracked `darkhunter_rv/sb2*.py` / `validation/sb2*.py` may exist — inventory before rewriting; leave out of unrelated PRs.

---

### CARD LIT-08-FULL (Phase 5)

**Start when:** Adopted RV stable; optional after SB2.  
**Tasks:** LAMOST/RAVE; orbit overlay from master CSV (step 08 remaining tasks).

---

## 7. Parallel launch schedules

### Schedule NOW (post-#90 merge)

| Parallel slot | Card | Notes |
|---------------|------|-------|
| 1 | **STRONG-QA** | Smoke + confirm INDEX/step 06 complete |
| 2 | **EPOCH-SPIKE** | Safe anytime |
| 3 | **MASK-DEPLOY** | Long-running bias rebuild OK |
| 4 | **TEMPLATE-OFFSETS** | If post-#88 diagnostics available |

### Schedule AFTER #90 merge (same as NOW)

| Parallel slot | Card |
|---------------|------|
| 1 | STRONG-QA |
| 2 | Continue MASK / TEMPLATE if not done |
| 3 | LIT-08-LITE (if mask ready) |
| 4 | BENCH-01 (if bias ready) |

### Schedule AFTER P1 exit criteria

| Serial | Card |
|--------|------|
| 1 | FUSION-03 |
| 2 | PLOTS-04 |
| then parallel | EPOCH-MATRIX completion, SHORTPAIR-05A |

### Schedule AFTER P3–P4

TRUST-02B · SB2-07 · LIT-08-FULL

---

## 8. Standard commands cheat sheet

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
export PYTHONPATH=.

# Tests (narrow → widen)
python -m pytest tests/test_strong_lines_product.py -q
python -m pytest tests/test_method_rv_offsets.py tests/validation/test_build_bias_set.py -q
python -m pytest tests/ -q --tb=line   # before PR if feasible

# Mask bias
bash scripts/rebuild_mask_bias.sh

# Method offsets
python -m validation.compute_method_rv_offsets --help

# Overlap / Teff reports (after --run-all-methods diagnostics exist)
python -m validation.rv_method_overlap_report \
  --diagnostics-glob 'OUTPUT_DIR/*_diagnostics.csv' \
  --out-dir validation_output/ORCH_overlap
python -m validation.rv_method_diagnostics_report \
  --diagnostics-glob 'OUTPUT_DIR/*_diagnostics.csv' \
  --out-dir validation_output/ORCH_teff

# Campaign list
# validation_output/chunk_campaign/spectrum_list.txt
# Spectra: /Users/rfoley/darkhunter/rvs/data/Gaia_DR3_*_epoch_*.txt
```

Replace `OUTPUT_DIR` with a real diagnostics directory (e.g. `output` or a `validation_output/...` arm).

---

## 9. Definition of done (entire program)

Orchestrator may stop when **all** are true:

1. Steps **06, 09, 02a, 10** complete in INDEX (09 already; others closed).
2. Steps **01** acceptance documented or explicitly waived by human.
3. Step **03** fusion merged; coverage tables exist.
4. Step **04** adopted plots available under `--plots-focus`.
5. Step **11** ships matrix + abs-fill CLI; synthetic + ≥1 real multi-epoch demo.
6. Step **05a** short-pair report exists using abs + epoch CCF.
7. Step **08 lite** literature comparison artifact exists.
8. Phase 5 items either done or explicitly deferred with INDEX notes.
9. [ATTACK_ORDER.md](ATTACK_ORDER.md) / this file checkboxes reflect reality.

---

## 10. Orchestrator status log (append-only)

Subagents / orchestrator: append a line when a card finishes.

| When (UTC) | Card | Result | PR/commit | Notes |
|------------|------|--------|-----------|-------|
| 2026-08-01 | (bootstrap) | plans written | local | #90 CI in progress; ORCHESTRATOR.md created |
| 2026-08-02 | P0-MERGE-90 | **done** | main `dbdcdc0` / [#90](https://github.com/UCSC-Transients/dark-hunter_rv/pull/90) | step 06 complete; #43/#91–#93 closed |
| 2026-08-02 | (policy) | human grill locked | local §0.6 | Shared-worktree clash → kill; relaunch in isolated worktrees |
| 2026-08-02 | (launch) | 4 cards parallel (wt) | — | STRONG-QA, MASK-DEPLOY(thin), TEMPLATE-OFFSETS, EPOCH-SPIKE |
| 2026-08-02 | MASK-DEPLOY | **done** (thin) | `5fd36fa` on `step/02a-bias-defer-verify` | Jun-16 bias verified (364 keys); rebuild deferred; ziggy §12.1 |
| 2026-08-02 | EPOCH-SPIKE | **done** | `dbe7e8d` on `step/11-epoch-ccf-matrix` | [#94](https://github.com/UCSC-Transients/dark-hunter_rv/issues/94); pytest 5/5; matrix CLI = later card |
| 2026-08-02 | TEMPLATE-OFFSETS | **done** | `18d6cec` on `step/10c-method-offsets` | APF tmpl −1.048 / strong −2.803; #87 commented; step 10 → complete |
| 2026-08-02 | STRONG-QA | hung→relaunch | — | First wt agent stalled; relaunched |
| 2026-08-02 | STRONG-QA | **done** | `767d68b` on `step/06-post-merge-qa` | pytest 19p/1s; Q match; archived smoke lacks `ivw_n=` (code emits) |
| 2026-08-02 | BENCH-01 | **done** (waivers) | `5c352e4` on `step/01-cool-closeout` | σ_RV north star MET; chunk-scatter/Phase A waived; #38 commented |
| 2026-08-02 | LIT-08-LITE | **done** | `a6ffe75` on `step/08-external-rv-crosscheck` | n_stars=4; compare_literature_rvs CLI; #45 commented |
| 2026-08-02 | FUSION-03 | **done** | `9de6c68` on `step/03-method-fusion-coverage` | method_fusion v1 + coverage; 25 tests; #40 commented |
| 2026-08-02 | PLOTS-04 | launched | `step/04-adopted-rv-match-plots` @ fusion tip | — |
| 2026-08-02 | PLOTS-04 | **done** | `fa0578c` on `step/04-adopted-rv-match-plots` | adopted_rv_match plot + pipeline hook; visual residual |
| 2026-08-02 | EPOCH-MATRIX | launched | `step/11-epoch-ccf-matrix-cli` | Phase 4 |
| 2026-08-02 | EPOCH-MATRIX | **done** | `be9b0a3` on `step/11-epoch-ccf-matrix-cli` | real star 468391…; diag~0; SHORTPAIR next |
| 2026-08-02 | SHORTPAIR-05A | launched | `step/05a-short-pair-calibration` | — |
| 2026-08-02 | SHORTPAIR-05A | **done** | `47caa15` on `step/05a-short-pair-calibration` | σ-scale≈1.28 MAD; #42 commented |
| 2026-08-02 | (P5 launch) | TRUST-02B + SB2-07 + LIT-08-FULL | — | parallel worktrees |
| 2026-08-02 | TRUST-02B | **done** | `f0400f9` on `step/02b-trust-weights-stack` | opt-in trust IVW; default off; campaign A/B residual |
| 2026-08-02 | SB2-07 | **done** (partial) | `7129124` on `step/07-sb2-search` | WIP tracked; BiGauss APIs fixed; NSS/pipeline fuse residual |
| 2026-08-02 | LIT-08-FULL | **done PARTIAL** | `fec76dc` on `step/08-external-rv-full` | LAMOST/RAVE+overlay; n_lit=4 blocker |
| 2026-08-03 | PR wave | opened stacked | #95–#99 | Merge Phase 1→5 in order; bases are prior phase branches |
| 2026-08-02 | (post-#101) | stack on main | `c6801f7` / [#101](https://github.com/UCSC-Transients/dark-hunter_rv/pull/101) | soft residuals remain |
| 2026-08-02 | soft-residuals | **PR** | `phase/soft-residuals` | docs post-#101 + 11d enrich + SB2 pipeline fuse; supersedes #102 |
| 2026-08-02 | #103 merged | **done** | main `a174732` | continue soft residuals-2 |
| 2026-08-02 | TRUST-AB offline | **done** | `phase/soft-residuals-2` | median σ↑~2.4×; keep opt-in |
| 2026-08-02 | LIT n=8 | **PARTIAL** | same branch | max spectra∩master; teff empty-row fix |
| 2026-08-02 | #104 merged | **done** | main `d3e1043` | soft-residuals-3 next |
| 2026-08-02 | NSS fetch | **done** | `phase/soft-residuals-3` | 147/155 NSS two-body; flag rates need refit |
| 2026-08-02 | #105 merged | **done** | main `a7f2b8d` | code DoD; human gates next |
| 2026-08-02 | issues | closed #40/#41/#42 | — | #44/#45/#94 remain open |
| 2026-08-02 | HUMAN_GATES | **PR** | `phase/soft-residuals-4` | decision checklist |
| 2026-08-02 | #95–#99 → #101 | **merged to main** | `c6801f7` | Full stack on `main` |
| 2026-08-02 | docs status | in flight | `docs/post-101-status` | INDEX/ORCHESTRATOR/ATTACK_ORDER post-merge |
| 2026-08-02 | 11d wiring | **done** (local) | `f14f0a5` on `step/11d-product-wiring` | enrich hook + fusion docs; default-adopt still human |

---

## 11. Quick “what do I launch next?” decision tree

```text
Is #90 merged?
  YES → STRONG-QA + MASK-DEPLOY + TEMPLATE-OFFSETS + EPOCH-SPIKE (parallel)
  NO / CI running → do not merge; launch EPOCH-SPIKE and/or MASK-DEPLOY and/or TEMPLATE-OFFSETS only

Is bias_statistics committed for subchunks_8?
  NO → MASK-DEPLOY
  YES → BENCH-01 eligible

Is method_rv_offsets done?
  NO → TEMPLATE-OFFSETS
  YES → mark 10c / consider step 10 complete

Are mask+template(+strong) baseline exits met?
  NO → finish P1 cards
  YES → FUSION-03 then PLOTS-04; finish EPOCH-MATRIX

Phase 5: do 02b + SB2 (reuse WIP) + 08-full after P3–P4.
```

---

## 12. Human touchpoints (orchestrator must ask)

- Merge approval for PRs to `main`
- Push to origin
- Ziggy production refit (credentials / machine access)
- Any change to frozen continuum or `subchunks_8`
- Expanding strong-line list / Q recalibration campaign
- Enabling `epoch_ccf` as default adopted input
- Pre-final `bias_statistics.txt` rebuild (P0c deferred)

### 12.1 Ziggy alert log (append when a card would run ziggy)

| When (UTC) | Card | Commands (do not run without human) | Need output to continue? | Status |
|------------|------|--------------------------------------|--------------------------|--------|
| 2026-08-02 | MASK-DEPLOY | See step 02 plan “ziggy TODO”; `refit_all_per_object_parallel.sh` on ziggy | No (defer OK) | **ALERT — waiting human** |
