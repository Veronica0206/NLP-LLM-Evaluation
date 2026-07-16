# DrugReview: Rubric-Guided LLM Annotation of Patient Medication Reviews Beyond Overall Ratings — Ordinal Agreement, Uncertainty, and Learnability

This folder contains the Google Colab notebooks supporting the experiments in the
manuscript. The study evaluates rubric-guided `gpt-4o-mini` annotation of patient
medication reviews against rating-derived labels, through ordinal agreement,
uncertainty diagnostics, calibration, a duplicate-aware sensitivity check, and
downstream learnability probes across two size-matched cohorts: a clinically
coherent **Mood/Anxiety** cohort and a **Heterogeneous**-indication cohort.

## Structure

### Cohort construction (A0)

| Notebook | Purpose |
|---|---|
| `A0_Cohort_Construction` | Builds the two study cohorts from the raw UCI Drug Review corpus: concatenates the original train/test files, cleans and bins the 10-point rating into five ordinal classes, selects the Mood/Anxiety cohort by the prespecified condition list, and constructs the size-matched Heterogeneous cohort. |

### Label diagnostics: agreement, uncertainty, duplicates (A1–A4)

Rating-derived vs. rubric-guided `gpt-4o-mini` label analysis for each cohort ×
annotation regime: exact/ordinal agreement (accuracy, QWK, macro/weighted F1, JS
distance), soft-label diagnostics (E[p(y)], NLL, Brier, E[|Δ_ord|]), normalized-entropy
stratification, the disagreement logistic regressions, and the deduplicated
(unique-review) sensitivity check. Calibration (ECE/MCE, Fig. S1) is reproduced separately
in `A5_Calibration`.

| Notebook | Cohort | Annotation regime |
|---|---|---|
| `A1_Label_Diagnostics_MoodAnxiety_TextOnly` | Mood/Anxiety | Text-only |
| `A2_Label_Diagnostics_MoodAnxiety_TextMetadata` | Mood/Anxiety | Text+metadata |
| `A3_Label_Diagnostics_Heterogeneous_TextOnly` | Heterogeneous | Text-only |
| `A4_Label_Diagnostics_Heterogeneous_TextMetadata` | Heterogeneous | Text+metadata |

### Calibration (A5)

| Notebook | Purpose |
|---|---|
| `A5_Calibration` | Reproduces the calibration results (Supplementary Table S4 and Figure S1): expected and maximum calibration error (ECE/MCE) over **15 equal-width bins** of `p_max`, for (A) the `gpt-4o-mini` annotator scored against rating-derived agreement across the four cohort × annotation regimes, and (B) the downstream ALBERT probe scored against its panel target across the six label panels. Emits the two-panel reliability diagram (`fig_calibration.pdf`). |

### Downstream learnability probes (T11–T43)

Five-class ordinal sentiment probes comparing three label sources — rating-derived,
`gpt-4o-mini` text-only annotation, and `gpt-4o-mini` text+metadata annotation — under
text-only and text+metadata prediction inputs. Each notebook trains the full probe
suite: LR, SVM, RF, LightGBM, GRU, CNN, and ALBERT, with weighted precision/recall/F1,
macro-AUC, and ordinal QWK/MAE on the held-out test set.

| Notebook | Cohort | Prediction input | Label source |
|---|---|---|---|
| `T11_Human_Text_Only` | Mood/Anxiety | Text-only | Rating-derived |
| `T12_mini4o_Text_Only` | Mood/Anxiety | Text-only | LLM (text-only annotation) |
| `T13_mini4o_Text_Only` | Mood/Anxiety | Text-only | LLM (text+metadata annotation) |
| `T21_Human_Text_Only` | Heterogeneous | Text-only | Rating-derived |
| `T22_mini4o_Text_Only` | Heterogeneous | Text-only | LLM (text-only annotation) |
| `T23_mini4o_Text_Only` | Heterogeneous | Text-only | LLM (text+metadata annotation) |
| `T31_Human_Text_Metadata` | Mood/Anxiety | Text+metadata | Rating-derived |
| `T32_mini4o_Text_Metadata` | Mood/Anxiety | Text+metadata | LLM (text-only annotation) |
| `T33_mini4o_Text_Metadata` | Mood/Anxiety | Text+metadata | LLM (text+metadata annotation) |
| `T41_Human_Text_Metadata` | Heterogeneous | Text+metadata | Rating-derived |
| `T42_mini4o_Text_Metadata` | Heterogeneous | Text+metadata | LLM (text-only annotation) |
| `T43_mini4o_Text_Metadata` | Heterogeneous | Text+metadata | LLM (text+metadata annotation) |

### Exploratory aspect-level multi-task probes (T51–T54)

Shared-encoder ALBERT models with two three-class heads jointly predicting efficacy and
safety aspects, across the two cohorts with and without metadata inputs.

| Notebook | Cohort | Prediction input |
|---|---|---|
| `T51_mood_anxiety_without_metadata` | Mood/Anxiety | Text-only |
| `T52_mood_anxiety_with_metadata` | Mood/Anxiety | Text+metadata |
| `T53_heterogeneous_without_metadata` | Heterogeneous | Text-only |
| `T54_heterogeneous_with_metadata` | Heterogeneous | Text+metadata |

## Notebook → manuscript map

| Manuscript element | Notebook(s) |
|---|---|
| Dataset & cohort construction; duplicate counts (Dataset Characteristics) | `A0` |
| Table I — hard-label agreement & ordinal diagnostics | `A1`–`A4` (one column each) |
| Table II — agreement by normalized entropy | `A1`–`A4` |
| Soft-label diagnostics; disagreement logistic regressions | `A1`–`A4` |
| Calibration ECE/MCE (Suppl. Table S4, Fig. S1) | `A5` |
| Table VI — deduplicated (unique-review) sensitivity | `A1`–`A4` |
| Table III — downstream probes, Mood/Anxiety, text-only prediction, Panels A/B/C | `T11` / `T12` / `T13` |
| Table IV — downstream probes, Heterogeneous, text-only prediction, Panels A/B/C | `T21` / `T22` / `T23` |
| Suppl. Tables S2–S3 — text+metadata prediction inputs | `T31`–`T33`, `T41`–`T43` |
| Suppl. Table S1 — exploratory efficacy/safety multi-task probes | `T51`–`T54` |

> **Not included as runnable code.** Two manuscript components are intentionally not
> shipped here: (1) the `gpt-4o-mini` **annotation step** that produces the rubric-guided
> labels — it requires an OpenAI API key, and the operative rubric, decision rules, and
> output schema are given in the Supplementary Document; and (2) the **pilot
> human-agreement audit** (manuscript Table V), which is computed from separately
> collected two-annotator labels not distributed with this repository.

## Data

The experiments use the [Drug Review Dataset](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com)
from the UCI Machine Learning Repository (originally scraped from Drugs.com; also
mirrored on Kaggle). Two data layers are involved:

1. **Raw corpus** — download `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` from the
   UCI/Kaggle link and place them under `./data/`. `A0_Cohort_Construction` consumes these.
2. **Derived, rubric-guided labeled datasets** — the cohort files carrying the
   rating-derived labels, `gpt-4o-mini` rubric labels, and model-reported class-probability
   vectors are archived on OSF: **https://doi.org/10.17605/OSF.IO/7V9KP**
   (subject to the source dataset's licensing and terms of use). These are the inputs the
   diagnostics (`A1`–`A4`) and probe (`T*`) notebooks load; the raw derivative is *not*
   regenerated here because the annotation step requires an API key.

Each notebook sets a relative `DrugReview_ROOT` (default `.`) near the top and reads its
input from a path under `./data/`. Expected input filenames:

| Notebook(s) | Expected input file(s) (under `./data/`) |
|---|---|
| `A0` | `drugsComTrain_raw.csv`, `drugsComTest_raw.csv` |
| `A1` / `A3` | `PureText/drugsCom_{mood_anxiety,generalized}_with_ai_labels_mini.csv` |
| `A2` / `A4` | `Text_Metadata/drugsCom_{mood_anxiety,generalized}_with_ai_labels_mini.csv` |
| `A5` (Panel A) | `PureText/` and `Text_Metadata/drugsCom_{mood_anxiety,generalized}_with_ai_labels_mini.csv` |
| `A5` (Panel B) | `predictions/{mood,het}_{rating,ai_to,ai_tm}_dl_models_predictions.csv` (ALBERT held-out test predictions exported by `T11`–`T23`) |
| `T11`–`T43` | `drugreview_{mood_anxiety,heterogeneous}_labeled.csv` |
| `T51`–`T54` | `drugsCom_{mood_anxiety,generalized}_with_ai_labels_mini.csv` |

The notebooks were authored in Google Colab and include an optional, guarded Drive-mount
cell; set `DrugReview_ROOT` (or the `DRUGREVIEW_ROOT` environment variable where present)
to run them locally.

## Naming Convention

Analysis notebooks are prefixed `A` (cohort construction, then label diagnostics by
cohort × annotation regime). Downstream-probe notebooks are prefixed `T` and named
`T{cohort}{source}_{ProbeInput}`, where the cohort/input block encodes Mood/Anxiety vs.
Heterogeneous and text-only vs. text+metadata prediction inputs, and the source suffix
(`Human` / `mini4o`) encodes the rating-derived or `gpt-4o-mini` label source. Aspect
notebooks (`T51`–`T54`) are named by cohort and metadata regime.

## Computational Environment

All notebooks were executed in Google Colab: CPU runtime for classical ML models, GPU
runtime for the deep-learning and transformer probes. Core dependencies (scikit-learn,
LightGBM, PyTorch, Hugging Face Transformers, statsmodels, NLTK) are listed in the
repository-root [`requirements.txt`](../requirements.txt); analyses were implemented in
Python 3.

## License

Released under the repository's [GNU General Public License v3.0](../LICENSE).
