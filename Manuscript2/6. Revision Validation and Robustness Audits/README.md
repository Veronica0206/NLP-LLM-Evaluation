# Revision validation and robustness audits

This stage contains the compact R2 analyses added to document validation,
routing, uncertainty, and protocol sensitivity. These are supporting audits,
not additional primary modeling tasks.

Existing notebook filenames are retained to preserve earlier public URLs.

## Contents

| Path | Purpose | Input availability |
|---|---|---|
| `MentalHealth_R1_existing_data_analyses.ipynb` | Existing-data revision-analysis code. | Uses external analysis-ready inputs; notebook outputs are cleared. |
| `MentalHealth_R1_human_validation_and_tableS2.ipynb` | Human-audit summaries and Supplementary Table S2. | Rerun requires external audit inputs; raw workbooks are excluded. |
| `01_human_validation/compute_human_aspect_kappas.py` | Canonical chance-corrected human aspect agreement for Table 5 and Table S10. | External human-audit inputs. |
| `04_entropy_routing/run_entropy_routing.py` | Entropy, maximum-score, top-two-margin, out-of-fold category, and random routing across review budgets. | External analysis-ready full corpus. |
| `05_multillm_documentation/analyze_multillm_documentation.py` | Sample characterization, factor summaries, and cross-LLM hard-label, entropy, aspect-count, co-occurrence, prevalence, and strength robustness. | External full corpus and crossed multi-LLM output. |
| `06_table_ci_reporting/compute_aspect_mean_f1_ci.py` | Post-cluster bootstrap interval for the Table 4 mean F1 across six aspect heads. | External saved aspect predictions. |

## Running the scripts

Install the tested CPU dependencies from `../requirements-cpu.txt`, then run
the scripts from the `Manuscript2/` directory. Exact commands and input
boundaries are documented in each analysis subfolder.

Write regenerated files under an untracked `outputs/` directory. Do not commit
generated result files, corpus rows, post text, human-audit workbooks, or
row-level outputs.

## Analysis boundaries

- The entropy-routing script is CPU-only and does not train a model or call an
  API. Released-category routing is estimated out of fold; random routing is a
  finite-corpus expectation.
- The multi-LLM script analyzes an archived 100-item, 21,600-output crossed
  design (four model families, three prompts, six temperatures, and three
  seeds). It does not regenerate those API outputs.
- Binary aspect presence is used for presence/co-occurrence structure;
  three-level NONE/WEAK/CLEAR strength is evaluated separately for ordinal
  reproducibility.
- Human-agreement analyses report a deliberately enriched, non-representative
  audit sample. They do not create an expert-adjudicated reference standard.
- The canonical human aspect script uses seed 2025.

See `../REPRODUCIBILITY_INDEX.md` for the table- and figure-level mapping.
