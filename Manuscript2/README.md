# Rubric-Guided Annotation for Affective Text Classification: Label-Space Overlap, Multi-View Supervision, and Entropy Diagnostics

This repository contains the Google Colab notebooks supporting the experiments
described in the manuscript. Folders follow the analysis steps in Fig. 1 of the
manuscript, with one repository-level subdivision: Fig. 1 step 3B, downstream
supervision and modeling, is split into `3b1_Hard_Label_Modeling` for
hard-label panels and `3b2_Soft_and_Aspect_Modeling` for soft-label and
aspect-modeling analyses. All notebooks retain executed cell outputs for
transparency.

## Structure

| Folder | Fig. 1 step | Contents |
|---|---|---|
| `1_Dataset` | 1. Data Source: Sentiment Analysis for Mental Health | OSF data-package pointer and local data-layout guidance |
| `2_LLM_Annotation` | 2. Rubric-Guided LLM Annotation | Sanitized rubric-guided `gpt-4o-mini` annotation notebook (prompt, JSON schema, settings, retry/checkpoint logic) |
| `3a_Label_Comparison_and_Entropy` | 3A. Annotation-output Characterization | Released-label vs. LLM-label concordance, entropy and score-dispersion diagnostics, co-occurrence checks, review-budget routing, duplicate-text cluster sensitivity |
| `3b1_Hard_Label_Modeling` | 3B. Downstream Supervision and Modeling Branches | Multiclass modeling under released-label, LLM-label, and agreement-subset supervision |
| `3b2_Soft_and_Aspect_Modeling` | 3B. Downstream Supervision and Modeling Branches | Soft-label transformer training (both evaluation regimes), six-head aspect modeling, aspect-mean bootstrap intervals |
| `4a_Human_Agreement_Audit` | 4A. Human Agreement Audit | 300-post non-clinician agreement audit: primary labels and chance-corrected aspect agreement |
| `4b_MultiLLM_Protocol_Audit` | 4B. Multi-LLM Protocol Sensitivity Audit | 100-item crossed protocol-sensitivity audit (four LLMs, three prompts, six temperatures, three seeds) |
| `prompts` | 4B (support) | Exact system-prompt strings (minimal, rubric, chain-of-thought) and crossing constants used to construct the multi-LLM audit grid |

## Reproduction Map

| Manuscript item | Notebook |
|---|---|
| Agreement, confusion matrix (Table S3), entropy distribution (Fig. S1), ECE, duplicate-post analysis, co-occurrence results | `3a_Label_Comparison_and_Entropy/Label_Comparison_and_Entropy.ipynb` |
| Logistic regressions (Table I), zero-entropy block diagnostics | `3a_Label_Comparison_and_Entropy/Label_Comparison_and_Entropy.ipynb` |
| Canonical tie-aware AUROC, tie-preserving strata, review-budget routing (Fig. S2, Table S5) | `3a_Label_Comparison_and_Entropy/Entropy_Routing.ipynb` |
| Cluster-aware sensitivity analyses (Table S4) | `3a_Label_Comparison_and_Entropy/Cluster_Sensitivity.ipynb` |
| Hard-label panels (Table II; Tables S6--S8) | `3b1_Hard_Label_Modeling/Hard_OriginalLabel.ipynb`, `3b1_Hard_Label_Modeling/Hard_AILabel.ipynb`, `3b1_Hard_Label_Modeling/Hard_AgreementSubset.ipynb` |
| Soft-label results (Table III) | `3b2_Soft_and_Aspect_Modeling/Soft_TrainOnly.ipynb`, `3b2_Soft_and_Aspect_Modeling/Soft_TrainAll.ipynb` |
| Aspect-head results (Table IV; Table S9) | `3b2_Soft_and_Aspect_Modeling/Aspect_MultiTask.ipynb`, `3b2_Soft_and_Aspect_Modeling/Aspect_Mean_F1_Bootstrap.ipynb` |
| Human agreement audit (Table V; Table S10) | `4a_Human_Agreement_Audit/Human_Agreement_Audit.ipynb`, `4a_Human_Agreement_Audit/Human_Aspect_Agreement.ipynb` |
| Multi-LLM protocol sensitivity (Tables S11--S13) | `4b_MultiLLM_Protocol_Audit/MultiLLM_Protocol_Sensitivity.ipynb` |

Hyperparameter search spaces and fixed settings (Table S2) are documented in
the configuration cells of the `3b1` and `3b2` notebooks.

## Data

The analysis-ready data package is shared separately through OSF.

- DOI: [10.17605/OSF.IO/YQ9TZ](https://doi.org/10.17605/OSF.IO/YQ9TZ)
- During peer review, the package is accessible through the anonymized
  view-only link:
  <https://osf.io/yq9tz/overview?view_only=c3805375590e4873b907671df16fc4b5>
  (the DOI resolves to the project's public page, which remains private
  until acceptance).

The OSF package provides `mental_health_unified_labels_final.csv` (the
unified-labels corpus consumed by the comparison, entropy, and modeling
notebooks), the two 300-post human-audit annotator workbooks, and the
multi-LLM audit files `mh_labeling_final.csv` (the 21,600-record crossed
evaluation output consumed by `4b_MultiLLM_Protocol_Audit`) and
`mh_sample_100.csv` (the 100-post entropy-stratified audit sample).

This repository does not store raw or derived data files, generated outputs,
model checkpoints, API keys, or private credentials. Notebook path
configuration cells reference the authors' working layout; update them to
point at the downloaded OSF files before running.

The API-based annotation notebook is included as a sanitized source-protocol
copy. The pre-generated LLM annotation outputs needed for downstream analyses
are provided in the OSF data package, so readers do not need to re-query the
API to reproduce the manuscript analyses.

The two audits follow the same pattern: both the data and the analysis
code are available. For the 300-post human agreement audit, the OSF
package provides the audit sample and both completed annotator workbooks,
and `4a_Human_Agreement_Audit` contains the full agreement analysis. For
the multi-LLM protocol audit, `prompts/` contains the exact system-prompt
strings and crossing constants that defined the 21,600-record evaluation
grid, the OSF package provides the complete crossed outputs
(`mh_labeling_final.csv`) and the 100-post stratified sample
(`mh_sample_100.csv`), and `4b_MultiLLM_Protocol_Audit` contains the full
sensitivity analysis. The provider-calling and sample-drawing steps are
represented by these exact inputs and archived outputs rather than by
rerunnable scripts; re-querying hosted endpoints would not regenerate the
archived outputs in any case, because hosted model versions change over
time. Every reported audit number is computed by the audit notebooks from
the archived files.

## Computational Environment

All notebooks were developed in Google Colab and local Python environments.
CPU runtime was used for classical ML/statistical analyses; GPU runtime is
recommended for transformer fine-tuning and soft-label modeling.

Core dependencies include `pandas`, `numpy`, `scikit-learn`, `scipy`,
`statsmodels`, `matplotlib`, `seaborn`, `PyTorch`, `transformers`,
`lightgbm`, `nltk`, and `openpyxl`. Version provenance differs by
notebook family. The routing, cluster-sensitivity, and multi-LLM audit
notebooks record executed package versions in their output manifests; the
other CPU-side notebooks retain executed outputs and kernel metadata.
The modeling notebooks ran on the Colab GPU image current at execution
time and are not version-pinned; their commented install lines reference
`transformers==4.41.1` as the tested reference version, and their
retained outputs record the executed configurations, seeds, search
spaces, and results against which any re-run can be checked. For new
runs, installing the referenced `transformers` version on a current Colab
image is the closest starting point to the original environment.

## License

Released under the [GNU General Public License v3.0](LICENSE).
