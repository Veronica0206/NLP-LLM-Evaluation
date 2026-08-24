# R2 table and figure reproduction index

Paths below are relative to `Manuscript2/`. This index distinguishes public
code/configuration and aggregate evidence from the row-level inputs supplied
separately through the external data package.

## Main manuscript

| Artifact | Public code/configuration | External input or boundary | Reproduction scope |
|---|---|---|---|
| Figure 1, study pipeline | No statistical generator; the figure is a manually authored diagram of the stages documented in this directory. | Current figure artwork is part of the manuscript submission package, not this code repository. | Manual layout/export. |
| Table 1, disagreement regression | `2. Code for Label Comparison and Entropy/Original_AI_Labels_Exploration.ipynb` | `0. Dataset/analysis_ready/mental_health_unified_labels_final.csv`. | CPU notebook; the published value is recreated by running the relevant cells. |
| Table 2, hard-label Panels A-C | `3. Hard Label Modeling/MentalHealth_OriginalLabel_multiclass.ipynb`; `3. Hard Label Modeling/MentalHealth_mini4oLabel_multiclass.ipynb`; `3. Hard Label Modeling/MentalHealth_SameLabel_multiclass.ipynb` | `0. Dataset/analysis_ready/mental_health_unified_labels_final.csv`; saved splits/predictions are external when used for metric-only reconstruction. | Model retraining or metric reconstruction. Selected settings are in `3. Hard Label Modeling/selected_hyperparameters/`. |
| Table 3, soft-label models | `4. Soft Label Modeling/MentalHealth_4omini_SoftTrain.ipynb`; `4. Soft Label Modeling/MentalHealth_4omini_SoftAll.ipynb` | `0. Dataset/analysis_ready/mental_health_unified_labels_final.csv`; saved predictions are external. | Transformer retraining or metric reconstruction. Selected settings are in `4. Soft Label Modeling/selected_hyperparameters/`. |
| Table 4, six aspect heads | `5. Multi-Task Aspect Modeling/MentalHealth_4omini_aspects.ipynb`; `6. Revision Validation and Robustness Audits/06_table_ci_reporting/compute_aspect_mean_f1_ci.py` | Full corpus for training; `0. Dataset/modeling_outputs/06_AspectLabel/test_predictions_aspect.csv` for CPU interval reconstruction. | Model retraining for base predictions; CPU post-cluster bootstrap for the mean-row intervals. |
| Table 5, human agreement audit | `6. Revision Validation and Robustness Audits/MentalHealth_R1_human_validation_and_tableS2.ipynb`; `6. Revision Validation and Robustness Audits/01_human_validation/compute_human_aspect_kappas.py` | Files under `0. Dataset/human_audit/`; raw returned workbooks/post text are not in GitHub. | CPU audit summaries and canonical seed-2025 chance-corrected analysis. Safe reference outputs are under `6. Revision Validation and Robustness Audits/01_human_validation/expected_outputs/`. |

The Table 1 logistic-model AUC and the Table S5 entropy-routing AUROC are
different estimands produced by different analyses; they should not be used
interchangeably.

## Supplementary material

| Artifact | Public code/configuration | External input or boundary | Reproduction scope |
|---|---|---|---|
| Table S1, upstream sources and mappings | Documentation-only table in the manuscript supplement. | Source-dataset documentation. | Manually curated; no executable generator. |
| Table S2, released-versus-AI confusion matrix | `6. Revision Validation and Robustness Audits/MentalHealth_R1_human_validation_and_tableS2.ipynb` | External human-audit input under `0. Dataset/human_audit/`. | CPU notebook. |
| Table S3, hard-label per-class metrics | The three Table 2 notebooks. | External saved hard-label predictions or model-training input. | Metric reconstruction or retraining. |
| Table S4, aspect per-class metrics | The Table 4 aspect notebook. | External saved aspect predictions or model-training input. | Metric reconstruction or retraining. |
| Table S5, routing baselines and review budgets | `6. Revision Validation and Robustness Audits/04_entropy_routing/run_entropy_routing.py` | `0. Dataset/analysis_ready/mental_health_unified_labels_final.csv`. | CPU script; aggregate reference CSVs and manifest are under `6. Revision Validation and Robustness Audits/04_entropy_routing/expected_outputs/`. |
| Figure S1, review-budget curves | Same routing script as Table S5. | Analysis-ready full corpus. | CPU-generated from the budget-curve CSV. |
| Table S6, multi-LLM audit sample characteristics | `6. Revision Validation and Robustness Audits/05_multillm_documentation/analyze_multillm_documentation.py` | Full corpus plus `0. Dataset/multi_llm/mh_labeling_final.csv`. | CPU reconstruction/verification of the historical sampling strata and sample summaries. |
| Table S7, multi-LLM factor summaries | Same multi-LLM script. | `0. Dataset/multi_llm/mh_labeling_final.csv`; API generation is not rerun by this script. | CPU descriptive summaries and fixed-condition agreement. |
| Table S8, cross-LLM robustness | Same multi-LLM script. | `0. Dataset/multi_llm/mh_labeling_final.csv`. | CPU entropy, aspect-count, co-occurrence, prevalence, binary-presence, and three-level-strength analyses. Aggregate reference files are under `6. Revision Validation and Robustness Audits/05_multillm_documentation/expected_outputs/`. |
| Table S9, weighted precision and recall | The three Table 2 notebooks. | External saved hard-label predictions or model-training input. | Metric reconstruction or retraining. |
| Table S10, chance-corrected human aspect agreement | `6. Revision Validation and Robustness Audits/01_human_validation/compute_human_aspect_kappas.py` | Files under `0. Dataset/human_audit/`; raw text/workbooks are excluded from GitHub. | CPU script using the canonical seed-2025, post-cluster bootstrap analysis. |
| Figure S2, entropy distributions | `2. Code for Label Comparison and Entropy/Original_AI_Labels_Exploration.ipynb` | `0. Dataset/analysis_ready/mental_health_unified_labels_final.csv`. | CPU notebook. |

## Public aggregate evidence

The `expected_outputs/` directories contain small, statement-free tables and
manifests used to check a rerun. They do not contain corpus rows or row-level
predictions. In particular, the routing release excludes row-level routing
scores, and the human-audit release excludes post text and returned workbooks.

## Known reproduction boundaries

- GitHub must be paired with the synchronized external data package to rerun
  analyses that require row-level inputs.
- The exact original Colab/GPU software image was not retained; see
  `ENVIRONMENT.md`.
- Figure 1 and Table S1 are manually maintained documentation artifacts.
- Multi-LLM API generation is not repeated by the CPU analysis script. The
  script analyzes and validates the archived crossed output.
- Human-audit reruns require authorized external audit inputs. The raw
  workbooks are intentionally not public, so GitHub alone cannot rerun this
  component.
- An immutable commit/tag identifies the exact public code version; the
  external archive must be versioned separately so the two identifiers can be
  cited together.
