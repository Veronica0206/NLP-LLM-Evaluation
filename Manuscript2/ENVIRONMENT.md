# R2 computational environment

This record separates retained evidence about the reported runs from the
environment used to validate the public CPU analysis scripts.

## Tested CPU release environment

The R2 CPU analysis utilities were validated with the following environment:

| Component | Version |
|---|---:|
| Python | 3.9.6 |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.5.2 |
| SciPy | 1.13.1 |
| Matplotlib | 3.9.4 |
| openpyxl | 3.1.5 |

Install the numerical stack with `requirements-cpu.txt`. That file covers the
tested CPU analyses only; it is not an original GPU/Colab lock file.

## Retained configuration and run provenance

| Component | What is documented |
|---|---|
| Six R2 modeling notebooks | Public code records training seed 2025, grouped-split constants, model settings, and selected output paths. Notebook outputs are cleared. |
| Modeling devices | Internal run provenance recorded CPU for hard-label Panels A and C, and CUDA for Panel B, both soft-label workflows, and the six-head aspect workflow. These are provenance notes, not public executed output. |
| Transformer checkpoints | Public code identifies `albert-base-v2` and `dmis-lab/biobert-v1.1`. |
| Selected model settings | Public JSON files under the three modeling stages preserve the chosen hyperparameters. |
| Full-corpus annotation | Public code records the `gpt-4o-mini` model name; the external archived data retain system fingerprints and timestamps. |
| Revision CPU analyses | Public `expected_outputs/` files preserve statement-free aggregates and manifests; the multi-LLM manifest records Python and core numerical-library versions. |

## Original GPU environment boundary

The following details were not retained from the original modeling runs:

- exact Python, PyTorch, Transformers, PEFT, LightGBM, CUDA, cuDNN, and driver
  versions for all six notebooks;
- GPU model for runs whose output records only `cuda`;
- Hugging Face model commit revisions;
- exact API-client versions used for annotation generation; and
- a complete `pip freeze`, Colab base-image identifier, or container digest.

Commented installation lines in a notebook are not treated as proof of the
installed runtime. Accordingly, this repository does not claim bitwise replay
of the original transformer-training environment. It supports inspection of
the exact R2 code/configuration and CPU re-analysis from permitted saved inputs.

## Hardware guidance

- The revision-audit scripts are CPU-only.
- Classical models can run on CPU.
- GPU access is recommended for ALBERT/BioBERT and soft-label or multi-task
  transformer training.

## Credentials

API credentials are not part of the reproduction package. Annotation code reads
credentials from environment variables or notebook-secret stores, including
`OPENAI_API_KEY`, `GOOGLE_API_KEY`, and `TOGETHER_API_KEY` where applicable.
Never place credential values in source, output cells, archives, or Git history.
