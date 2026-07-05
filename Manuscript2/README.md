# Manuscript 2: LLM-Based Annotation for Affective Text Classification

This folder contains the Google Colab notebooks supporting the experiments described in the manuscript. The notebooks are organized by analysis stage and retain executed cell outputs for transparency.

## Structure

### 0. Dataset
OSF data-package notes and local data-layout guidance. The analysis-ready data files are shared separately through OSF and are not stored in this repository.

### 1. Code for LLM Annotation
Sanitized rubric-guided LLM annotation protocol using `gpt-4o-mini`, including the prompt, JSON output schema, model settings, retry/checkpoint logic, and post-processing steps. Private credential-check lines and local metadata were removed; cell outputs are retained.

### 2. Code for Label Comparison and Entropy
Released-label versus LLM-label concordance analysis, score-vector entropy diagnostics, score dispersion summaries, and affective co-occurrence checks.

### 3. Hard Label Modeling
Primary hard-label downstream modeling comparing released-label, LLM-label, and agreement-subset supervision regimes across multiclass classifiers.

### 4. Soft Label Modeling
Soft-label transformer training and evaluation using LLM score-vector targets, including soft-train/hard-test and full soft-label evaluation workflows.

### 5. Multi-Task Aspect Modeling
Six-head aspect-level modeling for co-occurring affective evidence and aspect-level supervision diagnostics.

### 6. Revision Validation and Robustness Audits
R1 revision analyses, paired tests, entropy routing summaries, class-wise aspect metrics, human-validation audit tables, Supplementary Table S2 generation, and multi-LLM robustness summaries.

## Naming Convention

Notebook names follow the analysis stage and task:

```text
MentalHealth_[Analysis_or_Modeling_Task].ipynb
```

- `MentalHealth_4omini_Labeling.ipynb`: LLM annotation protocol and output-schema documentation
- `Original_AI_Labels_Exploration.ipynb`: released-label versus LLM-label comparison and entropy diagnostics
- `MentalHealth_*_multiclass.ipynb`: hard-label multiclass modeling notebooks
- `MentalHealth_4omini_Soft*.ipynb`: soft-label modeling notebooks
- `MentalHealth_R1_*.ipynb`: revision validation and robustness audit notebooks

## Data

The analysis-ready data package is shared separately through OSF:

https://osf.io/yq9tz/overview?view_only=c3805375590e4873b907671df16fc4b5

Place the downloaded files under `0. Dataset/` or update notebook path configuration cells before running the notebooks. This repository does not store raw or derived data files, standalone generated outputs, model checkpoints, API keys, or private credentials.

The API-based annotation notebook is included as a sanitized source-protocol copy. The pre-generated LLM annotation outputs needed for downstream analyses are provided in the OSF data package, so readers do not need to re-query the API to reproduce the manuscript analyses.

## Computational Environment

All notebooks were developed in Google Colab and local Python environments. CPU runtime was used for classical ML/statistical analyses; GPU runtime is recommended for transformer fine-tuning and soft-label modeling.

Core dependencies include `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `PyTorch`, `transformers`, `lightgbm`, `nltk`, and `openpyxl`.

## License

Released under the repository's [GNU General Public License v3.0](../LICENSE).
