# RV summary JSON (pop integration)

Machine-readable per-star summaries for `dark-hunter_pop` `CandidateRecord.rv_summary`.

Parent: [UCSC-Transients/dark-hunter_pop#31](https://github.com/UCSC-Transients/dark-hunter_pop/issues/31).

## Files

| Artifact | Role |
|---|---|
| `output/Gaia_DR3_<source_id>_summary.txt` | Legacy sectioned text (website, humans, existing fitters) |
| `output/Gaia_DR3_<source_id>_summary.json` | Pop-facing JSON contract (written with every `write_star_summary`) |
| `rv_fit_reports/<stem>_joker_fit.json` | Full Joker fit report (also referenced from summary JSON) |

## Schema (v1)

Top-level keys:

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | Currently `1` |
| `source_id` | int | Gaia DR3 source id |
| `summary_txt` | str | Companion text summary basename |
| `instruments` | list[str] | Unique telescopes across pipeline + external RV rows |
| `n_epochs` | int | Total usable RV epochs (pipeline + external) |
| `n_pipeline_epochs` | int | APF/KPF/… pipeline rows |
| `n_external_epochs` | int | Literature / archive RV rows |
| `gaia_metadata` | object | Normalized `[GAIA METADATA]` block |
| `nss_solution_type` | str \| null | NSS solution type when present |
| `nss_orbital` | object | Period, eccentricity, T_periastron, semi-amplitudes, inclination, … |
| `thiele_innes` | object | A/B/F/G (+ errors) for pop `ThieleInnesElements` |
| `external_rvs` | list | `{telescope, mjd, rv_kms, rv_err_kms, flag}` |
| `pipeline_epochs` | list | `{file, basename, mjd, rv_kms, rv_err_kms, wrms_kms, fallback, telescope}` |
| `joker_fit` | object \| absent | Joker envelope report when `fit_joker_rv.py` has run |
| `joker_fit_path` | str \| absent | Path to standalone Joker JSON |

## Pop mapping

`dark-hunter_pop` ingests the JSON block into `CandidateRecord` as:

- `rv_summary` ← full JSON (or subset during adapter phase)
- NSS / Thiele–Innes optional blocks can be filled from `nss_solution_type`, `nss_orbital`, `thiele_innes`
- Coordinates / parallax can be taken from `gaia_metadata` (`RA`, `Dec`, `Parallax`)

Breaking changes to field names require a docs PR in `dark-hunter_pop` first
(`FOUNDATION_INTERFACE_FREEZE.md`).

## Writers

- `darkhunter_rv.io_utils.write_star_summary` — always writes JSON alongside `.txt`
- `fit_joker_rv.run_one_joker` — refreshes `joker_fit` / `joker_fit_path` in summary JSON
- `darkhunter_rv.rv_summary_json.refresh_rv_summary_json_from_txt` — backfill helper

## Example

```bash
python fit_joker_rv.py --summary output/Gaia_DR3_<id>_summary.txt --use-gaia-nss
python -c "from pathlib import Path; from darkhunter_rv.rv_summary_json import load_rv_summary_json as L; print(L(Path('output/Gaia_DR3_<id>_summary.json'))['n_epochs'])"
```
