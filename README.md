# NLP-LLM-Evaluation

[![GitHub repo](https://img.shields.io/badge/GitHub-NLP--LLM--Evaluation-black)](https://github.com/Veronica0206/NLP-LLM-Evaluation)
[![Python](https://img.shields.io/badge/Python-Jupyter%20notebooks-blue)](https://www.python.org/)

Research manuscripts and reproducible code evaluating LLM performance on natural language tasks.

This repository collects studies on LLM annotation, NLP supervision quality, uncertainty, and downstream modeling. Each manuscript directory is self-contained with code and documentation for reproducibility (Manuscript 1 additionally includes its model-ready dataset), while this top-level README is organized as a quick portfolio map for AI evaluation, model reliability, and NLP research roles.

## AI evaluation lens

The repo is organized around a practical evaluation question: when LLMs are used to label, score, or structure subjective text, how should their behavior be measured before those labels are trusted in downstream models? The included manuscript folders cover rubric-conditioned LLM annotation, hard-label and soft-label supervision, entropy/uncertainty diagnostics, calibration-oriented checks, failure-mode analysis through disagreement and ablation workflows, and downstream NLP modeling.

## For hiring reviewers

This repo demonstrates hands-on LLM evaluation and reproducible NLP experimentation rather than only manuscript archiving. It includes rubric-conditioned annotation, human-AI agreement analysis, run-to-run consistency checks, entropy and uncertainty diagnostics, hard-label and soft-label supervision, transformer/classical baselines, and downstream modeling pipelines.

## Repository map

| Manuscript folder | Research question | Data/task | Evaluation methods | Modeling methods | What it demonstrates for AI evaluation roles |
|---|---|---|---|---|---|
| [`Manuscript1/`](Manuscript1/) | How reliable is rubric-conditioned GPT-4o-mini labeling for subjective hate-speech annotation, and how useful are AI labels for downstream learning? | HateXplain social-media posts with three-level labels: `normal`, `offensive`, `hatespeech`. | Human-AI agreement, Cohen's kappa, run-to-run LLM consistency, confusion matrices, entropy as a disagreement signal, hard-label and soft-label evaluation. | Classical and transformer-based downstream modeling, including ALBERT soft-label experiments and hard-label supervision comparisons. | LLM-as-annotator evaluation, rubric-conditioned prompting, uncertainty triage, label-quality analysis, and reproducible annotation-to-modeling workflow design. |
| [`Manuscript2/`](Manuscript2/) | How can LLM-based annotation be evaluated as multi-view supervision for affective text classification? | Post hoc-harmonized seven-class affective-state text corpus with released labels, LLM hard labels, score vectors, and aspect-level annotations. | Label-space overlap, entropy and score-dispersion diagnostics, affective co-occurrence analysis, human-validation audit, paired revision tests, and multi-LLM robustness summaries. | Hard-label classifiers, soft-label transformer training, and multi-task aspect modeling for co-occurring affective evidence. | Multi-view LLM supervision evaluation, uncertainty-aware routing, score-vector diagnostics, human-audit validation, and reproducible revision robustness analysis. |
| [`Manuscript3/`](Manuscript3/) | Can aspect-level LLM annotations be fused into better ordinal sentiment supervision for patient medication reviews? | Drug Review corpus with review text, drug/condition metadata, holistic labels, and aspect-level annotations for efficacy, safety, burden, and cost. | Entropy analysis, class distributions, Jensen-Shannon distances, ablations, leave-one-aspect-out diagnostics, and hard/soft supervision comparisons. | Aspect-weight learning with ALBERT and BioBERT, plus downstream probes including LR, RF, LightGBM, GRU, CNN, ALBERT, and BioBERT under text-only and text-plus-metadata inputs. | Aspect-level LLM annotation design, label-fusion evaluation, soft-label learning, ablation-based failure analysis, and patient-text NLP modeling. |
| [`DrugReview/`](DrugReview/) | How well does rubric-guided `gpt-4o-mini` annotation of patient medication reviews agree with noisy rating-derived labels, and how learnable and uncertainty-aware are the resulting labels? | Drugs.com/UCI Drug Review corpus in two size-matched cohorts (clinically coherent Mood/Anxiety and heterogeneous multi-indication); five-class ordinal sentiment plus efficacy/safety aspects. | Ordinal agreement (accuracy, quadratic-weighted kappa, MAE), normalized-entropy uncertainty, a 177-review human-validation audit, and duplicate-aware sensitivity analysis. | Downstream probes (LR, SVM, RF, LightGBM, GRU, CNN, ALBERT) under text-only and text-plus-metadata inputs, plus an exploratory aspect-level multi-task ALBERT for efficacy and safety. | Rubric-conditioned annotation evaluation under imperfect labels, ordinal-agreement and uncertainty diagnostics, human-audit validation, learnability probing, and a reproducible notebook pipeline. |

## Core capabilities demonstrated

- Rubric-conditioned LLM annotation
- Human-AI agreement and run-to-run consistency analysis
- Hard-label and soft-label learning
- Entropy and uncertainty diagnostics
- Calibration and failure-mode analysis through confusion, disagreement, and ablation workflows
- Transformer and classical ML baselines
- Reproducible evaluation workflows using notebook-based pipelines

## Suggested reviewer path

1. Start with this README overview to see the portfolio structure.
2. Review [`Manuscript1/`](Manuscript1/) for hate-speech LLM annotation evaluation, agreement, entropy, and hard/soft supervision.
3. Review [`Manuscript2/`](Manuscript2/) for affective-state annotation, entropy diagnostics, multi-view supervision, and revision robustness analyses.
4. Review [`Manuscript3/`](Manuscript3/) for patient medication review annotation, aspect-level fusion, and downstream model comparisons.
5. Review [`DrugReview/`](DrugReview/) for a patient-medication-review pilot on rubric-guided annotation quality, ordinal agreement, uncertainty, and downstream learnability.
6. Open the notebooks in each manuscript folder for implementation details, model training, and analysis outputs.

## Manuscripts

| # | Title | Domain | Status | Directory |
|---|---|---|---|---|
| 1 | Rubric-Conditioned Large Language Model Labeling: Agreement, Uncertainty, and Label Consistency in Subjective Text Annotation | Hate Speech | Accepted, *Computers in Human Behavior* | [`Manuscript1/`](Manuscript1/) |
| 2 | LLM-Based Annotation for Affective Text Classification: Label-Space Overlap, Multi-View Supervision, and Entropy Diagnostics | Affective-state text classification | Revision-stage manuscript folder with reproducible notebooks | [`Manuscript2/`](Manuscript2/) |
| 3 | Learning to Fuse Aspect-Level LLM Annotations for Low-Quality Ordinal Sentiment Supervision | Patient medication reviews | Experimental manuscript folder with reproducible notebooks | [`Manuscript3/`](Manuscript3/) |
| — | Rubric-Guided LLM Annotation of Patient Medication Reviews Under Imperfect Labels: Ordinal Agreement, Uncertainty, and Learnability | Patient medication reviews (pilot) | Pilot study folder with reproducible notebooks | [`DrugReview/`](DrugReview/) |

Manuscript numbering follows the author's internal research series; manuscripts not listed here are not yet public.

## How to use

Each manuscript folder contains its own `README.md` with detailed instructions on data, analysis pipeline, and reproduction steps. Navigate to the relevant directory for paper-specific documentation and run notebooks in the documented order.

## Citation

Please cite individual papers using the BibTeX entries provided in each manuscript's README.

## License

This repository is released under the GNU General Public License v3.0. Dataset sources may carry their own terms; see each manuscript directory for data-specific notes.
