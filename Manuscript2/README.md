# Manuscript 2: LLM-Based Annotation for Affective Text Classification

Supporting code and notebooks for the manuscript:

**LLM-Based Annotation for Affective Text Classification: Label-Space Overlap, Multi-View Supervision, and Entropy Diagnostics**

## Overview

This folder contains a sanitized source copy of the LLM annotation notebook and Jupyter analysis notebooks with cell outputs retained for transparency and reproducibility. The study evaluates a rubric-conditioned LLM annotation protocol for a post hoc-harmonized seven-class affective-state text corpus. The analyses examine label concordance, entropy and score-dispersion diagnostics, affective co-occurrence, hard-label and soft-label supervision, multi-task aspect modeling, human-validation audits, and multi-LLM robustness checks.

Standalone data files, generated output files, model checkpoints, figures, API keys, local paths, and private working notes are not included. Data materials for review and reproduction are provided separately through OSF:

https://osf.io/yq9tz/overview?view_only=c3805375590e4873b907671df16fc4b5

## Repository Structure

```text
.
├── 0. Dataset/
│   └── README.md
│
├── 1. Code for LLM Annotation/
│   ├── MentalHealth_4omini_Labeling.ipynb
│   └── README.md
│
├── 2. Code for Label Comparison and Entropy/
│   └── Original_AI_Labels_Exploration.ipynb
│
├── 3. Hard Label Modeling/
│   ├── MentalHealth_OriginalLabel_multiclass.ipynb
│   ├── MentalHealth_mini4oLabel_multiclass.ipynb
│   └── MentalHealth_SameLabel_multiclass.ipynb
│
├── 4. Soft Label Modeling/
│   ├── MentalHealth_4omini_SoftTrain.ipynb
│   └── MentalHealth_4omini_SoftAll.ipynb
│
├── 5. Multi-Task Aspect Modeling/
│   └── MentalHealth_4omini_aspects.ipynb
│
└── 6. Revision Validation and Robustness Audits/
    ├── MentalHealth_R1_existing_data_analyses.ipynb
    └── MentalHealth_R1_human_validation_and_tableS2.ipynb
```

## Dataset

The analysis-ready data package is shared separately through OSF:

https://osf.io/yq9tz/overview?view_only=c3805375590e4873b907671df16fc4b5

Place the downloaded files under `0. Dataset/` or update the path configuration cells in each notebook to point to the downloaded data location.

The GitHub repository intentionally does not store:

- raw or derived data files;
- generated output tables or figures;
- model checkpoints;
- API keys or private credentials.

## Analysis Pipeline

The notebooks are designed to be reviewed or run in the following order:

| Step | Folder | Purpose |
|---|---|---|
| 0 | `0. Dataset/` | OSF data package location and data-layout notes |
| 1 | `1. Code for LLM Annotation/` | Sanitized annotation notebook with full prompt, output schema, model settings, and post-processing logic |
| 2 | `2. Code for Label Comparison and Entropy/` | Released-label versus LLM-label concordance, score-vector entropy, and co-occurrence diagnostics |
| 3 | `3. Hard Label Modeling/` | Supervised models trained under released-label, LLM-label, and agreement-subset panels |
| 4 | `4. Soft Label Modeling/` | Soft-label transformer training using LLM score-vector targets |
| 5 | `5. Multi-Task Aspect Modeling/` | Six-head aspect-level modeling for co-occurring affective evidence |
| 6 | `6. Revision Validation and Robustness Audits/` | R1 paired tests, entropy routing summaries, class-wise aspect metrics, human-validation audit, and multi-LLM robustness summaries |

## Requirements

The notebooks were developed in Google Colab and local Python environments. Core dependencies include:

- Python 3.10+
- pandas
- numpy
- scikit-learn
- scipy
- statsmodels
- matplotlib
- seaborn
- PyTorch
- transformers
- lightgbm
- nltk
- openpyxl

GPU runtime is recommended for transformer modeling notebooks.

## Reproducing Results

1. Download the OSF data package: https://osf.io/yq9tz/overview?view_only=c3805375590e4873b907671df16fc4b5
2. Place the data under `0. Dataset/` or update notebook path configuration cells.
3. Open notebooks in the order listed in the analysis pipeline.
4. Run each notebook after confirming the input paths.
5. Generated files should be written to local output folders and are excluded from version control.

The API-based LLM annotation notebook is included as a sanitized source-protocol copy with cell outputs retained. Credential-check lines, private credentials, and local metadata were removed from that notebook. The pre-generated LLM annotation outputs needed for downstream analyses are provided in the OSF data package, so readers do not need to re-query the API to reproduce the manuscript analyses.

## Repository Note

Shared notebooks retain their cell outputs for transparency, but standalone generated files are not versioned. Local machine paths and private metadata have been removed from the shared copies.
