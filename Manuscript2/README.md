# LLM-Based Annotation for Affective Text Classification

This repository contains the Google Colab notebooks supporting the experiments
described in the manuscript. Folders follow the analysis steps in Fig. 1 of the
manuscript, and all notebooks retain executed cell outputs for transparency.

## Structure

| Folder | Fig. 1 step | Contents |
|---|---|---|
| `1_Dataset` | (1) Corpus | OSF data-package pointer and local data-layout guidance |
| `2_LLM_Annotation` | (2) Annotation protocol | Sanitized rubric-guided `gpt-4o-mini` annotation notebook (prompt, JSON schema, settings, retry/checkpoint logic) |
| `3a_Label_Comparison_and_Entropy` | (3) Diagnostics | Released-label vs. LLM-label concordance, entropy and score-dispersion diagnostics, co-occurrence checks, review-budget routing, duplicate-text cluster sensitivity |
| `3b_Hard_Label_Modeling` | (3) Hard-label learnability | Multiclass modeling under released-label, LLM-label, and agreement-subset supervision |
| `3c_Soft_and_Aspect_Modeling` | (3) Soft and aspect supervision | Soft-label transformer training (both evaluation regimes), six-head aspect modeling, aspect-mean bootstrap intervals |
| `4a_Human_Agreement_Audit` | (4) Human audit | 300-post non-clinician agreement audit: primary labels and chance-corrected aspect agreement |
| `4b_MultiLLM_Protocol_Audit` | (4) Multi-LLM audit | 100-item crossed protocol-sensitivity audit (four LLMs, three prompts, six temperatures, three seeds) |

## Reproduction Map

| Manuscript item | Notebook |
|---|---|
| Agreement, confusion matrix (Table S3), entropy distribution (Fig. S1), ECE, duplicate-post analysis, co-occurrence results | `3a/Label_Comparison_and_Entropy.ipynb` |
| Logistic regressions (Table I), zero-entropy block diagnostics | `3a/Label_Comparison_and_Entropy.ipynb` |
| Canonical tie-aware AUROC, tie-preserving strata, review-budget routing (Fig. S2, Table S5) | `3a/Entropy_Routing.ipynb` |
| Cluster-aware sensitivity analyses (Table S4) | `3a/Cluster_Sensitivity.ipynb` |
| Hard-label panels (Table II; Tables S6--S8) | `3b/Hard_OriginalLabel.ipynb`, `3b/Hard_AILabel.ipynb`, `3b/Hard_AgreementSubset.ipynb` |
| Soft-label results (Table III) | `3c/Soft_TrainOnly.ipynb`, `3c/Soft_TrainAll.ipynb` |
| Aspect-head results (Table IV; Table S9) | `3c/Aspect_MultiTask.ipynb`, `3c/Aspect_Mean_F1_Bootstrap.ipynb` |
| Human agreement audit (Table V; Table S10) | `4a/Human_Agreement_Audit.ipynb`, `4a/Human_Aspect_Agreement.ipynb` |
| Multi-LLM protocol sensitivity (Tables S11--S13) | `4b/MultiLLM_Protocol_Sensitivity.ipynb` |

Hyperparameter search spaces and fixed settings (Table S2) are documented in
the configuration cells of the `3b` and `3c` notebooks.

## Data

The analysis-ready data package is shared separately through OSF.

- DOI: [10.17605/OSF.IO/YQ9TZ](https://doi.org/10.17605/OSF.IO/YQ9TZ)

This repository does not store raw or derived data files, generated outputs,
model checkpoints, API keys, or private credentials. Notebook path
configuration cells reference the authors' working layout; update them to
point at the downloaded OSF files before running.

The API-based annotation notebook is included as a sanitized source-protocol
copy. The pre-generated LLM annotation outputs needed for downstream analyses
are provided in the OSF data package, so readers do not need to re-query the
API to reproduce the manuscript analyses.

## Computational Environment

All notebooks were developed in Google Colab and local Python environments.
CPU runtime was used for classical ML/statistical analyses; GPU runtime is
recommended for transformer fine-tuning and soft-label modeling.

Core dependencies include `pandas`, `numpy`, `scikit-learn`, `scipy`,
`statsmodels`, `matplotlib`, `seaborn`, `PyTorch`, `transformers`,
`lightgbm`, `nltk`, and `openpyxl`.

## License

Released under the [GNU General Public License v3.0](LICENSE).
