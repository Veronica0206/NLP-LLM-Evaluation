# Rubric-Conditioned Large Language Model Labeling: Agreement, Uncertainty, and Label Consistency in Subjective Text Annotation

**Jin Liu**

*Accepted for publication in [Computers in Human Behavior](https://www.sciencedirect.com/journal/computers-in-human-behavior)*

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.chb.2026.108988-blue)](https://doi.org/10.1016/j.chb.2026.108988)

## Overview

This repository contains the data and code for reproducing all analyses in the paper. Using the [HateXplain](https://huggingface.co/datasets/hatexplain) corpus (20,148 social-media posts, each annotated by three crowdworkers on a three-level scale: *normal*, *offensive*, *hatespeech*), we evaluate GPT-4o-mini as a rubric-conditioned labeling protocol. The study examines (a) concordance with human reference constructions, (b) run-to-run protocol stability, (c) entropy-based uncertainty as a disagreement signal, and (d) downstream classifier performance under hard-label and soft-label supervision regimes.

## Key Findings

- **Human-level agreement**: LLM-unanimous concordance (Cohen's kappa = 0.434) approaches mean human-human agreement (kappa = 0.460).
- **Protocol stability**: Two identical labeling runs agree on 93.0% of items (kappa = 0.882).
- **More learnable supervision**: LLM-generated labels yield higher downstream performance than majority-vote human labels, interpreted as greater internal consistency.
- **Entropy as triage signal**: Normalized Shannon entropy of LLM probability vectors predicts human-AI disagreement (OR = 4.79), supporting entropy-guided human review in hybrid workflows.
- **Soft-label advantage**: Training ALBERT on AI probability vectors outperforms training on human vote distributions under both hard-evaluation and soft-evaluation protocols.

## Repository Structure

```
.
├── 0. Dataset with 4omini Labels (run1)/
│   └── hatexplain_model_ready_with_ai_labels.csv        # Model-ready dataset (20,148 × 13)
│
├── 1. Code for 4ominiLabelling/
│   └── HateSpeech 4omini_Labeling.ipynb                 # GPT-4o-mini batch annotation pipeline
│
├── 2. Code for Labels Comparision_human vs 4omini/
│   ├── TwoSetsLabels.ipynb                              # Run-to-run AI consistency (Run 1 vs Run 2)
│   └── HateSpeech overall_evaluation.ipynb              # Human-AI agreement, confusion matrices, entropy analysis
│
├── 3. Hard Label Modeling_Primary Analysis/
│   ├── T31_HateSpeech_human_pure_multiclass.ipynb       # Models trained on human-unanimous labels (n=9,845)
│   ├── T32_HateSpeech_human_majority_multiclass.ipynb   # Models trained on human-majority labels (n=19,229)
│   └── T33_HateSpeech_4o_mini_multiclass.ipynb          # Models trained on AI labels (n=20,148)
│
├── 4. Hard Label Modeling_Supportive Analysis/
│   ├── T41_HateSpeech_4o_mini_pure_multiclass.ipynb     # AI labels on human-unanimous subset (controlled)
│   └── T42_HateSpeech_4o_mini_majority_multiclass.ipynb # AI labels on human-majority subset (controlled)
│
└── 5. Soft Label Modeling/
    ├── T51_HateSpeech_human_softlabels.ipynb            # ALBERT with human vote distributions (Panel A)
    ├── T51_HateSpeech_4omini_softlabels.ipynb           # ALBERT with AI probability vectors (Panel A)
    ├── T52_HateSpeech_human_softlabels.ipynb            # Full soft-train/soft-eval with human labels (Panel B)
    └── T52_HateSpeech_4omini_softlabels.ipynb           # Full soft-train/soft-eval with AI labels (Panel B)
```

## Dataset

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `hatexplain_model_ready_with_ai_labels.csv` | 20,148 | 13 | Streamlined version for modeling: post text, three human labels, human-majority, human-unanimous, AI hard label, and AI class probabilities |

**Label classes**: `normal`, `offensive`, `hatespeech`

**Label variants**:
- `human_pure` -- unanimous agreement across all three raters (n = 9,845)
- `human_majority` -- at least two of three raters agree (n = 19,229)
- `ai_label_mini` -- GPT-4o-mini hard label (full coverage, n = 20,148)
- `ai_p_normal_mini`, `ai_p_offensive_mini`, `ai_p_hate_mini` -- class probability vector

**Original data source**: [HateXplain on Hugging Face](https://huggingface.co/datasets/hatexplain) (Mathew et al., 2021)

## Analysis Pipeline

The notebooks are designed to be run in the following order:

| Step | Notebook(s) | Section in Paper |
|------|-------------|------------------|
| 1. AI annotation | `1. Code for 4ominiLabelling/` | Section 2.3 |
| 2. Agreement & entropy analysis | `2. Code for Labels Comparision_human vs 4omini/` | Sections 3.2--3.4 |
| 3. Hard-label modeling (primary) | `3. Hard Label Modeling_Primary Analysis/` (T31--T33) | Section 3.5.1 |
| 4. Hard-label modeling (controlled) | `4. Hard Label Modeling_Supportive Analysis/` (T41--T42) | Section 3.5.2 |
| 5. Soft-label modeling | `5. Soft Label Modeling/` (T51--T52) | Section 3.5 (soft-label) |

## Requirements

All notebooks were developed and executed in [Google Colab](https://colab.research.google.com/) with GPU runtime (NVIDIA T4 or L4).

**Core dependencies**:

- Python 3.12+
- scikit-learn
- PyTorch
- transformers (Hugging Face)
- lightgbm
- nltk
- statsmodels
- openai (for GPT-4o-mini annotation in Step 1)
- pandas, numpy, matplotlib, seaborn

## Reproducing Results

1. Clone or download this repository.
2. Upload the datasets from `0. Dataset with 4omini Labels (run1)/` to your Google Drive.
3. Open each notebook in Google Colab and update the file paths to match your Drive mount.
4. Run notebooks in the order listed in the [Analysis Pipeline](#analysis-pipeline) table above.
5. Each notebook is self-contained and includes its own train/validation/test split using a fixed random seed for reproducibility.

> **Note**: Step 1 (AI annotation) requires an OpenAI API key. The pre-generated AI labels are already included in the datasets, so this step can be skipped if you only wish to reproduce the modeling and evaluation results.

> **Reproducibility note**: Soft-label Panel B hard metrics (weighted F1, AUC) may exhibit minor numerical differences due to stochastic bootstrap resampling across independent runs. All soft metrics (Soft CE, Brier) and all Panel A results reproduce exactly.

## Citation

If you use this code or data in your work, please cite:

```bibtex
@article{liu2025rubric,
  title={Rubric-Conditioned Large Language Model Labeling: Agreement, Uncertainty, and Label Consistency in Subjective Text Annotation},
  author={Liu, Jin},
  journal={Computers in Human Behavior},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is released under the [MIT License](LICENSE).

The HateXplain dataset is subject to its own license terms; see the [original repository](https://github.com/hate-alert/HateXplain) for details.
