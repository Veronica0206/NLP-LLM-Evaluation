# Multivariate Coding of Psychological Text With Large Language Models: Co-Occurring Evidence, Label Disagreement, and Entropy

This folder contains the Google Colab notebooks supporting the manuscript. A
rubric-guided `gpt-4o-mini` protocol coded 53,043 mental-health social-media
posts with a forced-choice label, a seven-class score vector, and a
none/weak/clear evidence rating for each category other than Normal.
The folders follow the manuscript: the dataset and the annotation protocol,
the notebook that builds the main tables, the notebooks behind the three
analyses of the ratings, and the notebooks behind the supplementary analyses.
All notebooks retain executed cell outputs for transparency.

## Structure

| Folder | Manuscript part | Contents |
|---|---|---|
| `1_Dataset` | Method: Corpus | OSF data-package pointer and local data-layout guidance |
| `2_Annotation_Protocol` | Method: Annotation Protocol and Rubric | Sanitized rubric-guided `gpt-4o-mini` annotation notebook (prompt, rubric, JSON schema, settings, retry/checkpoint logic) |
| `3_Multivariate_Evidence` | Results: Tables 1 to 4 | One notebook that reads the annotated corpus and the archived audit and model outputs and computes the main-text results and most supplementary tables |
| `4_MultiLLM_Audit` | First analysis: agreement across four LLMs | 100-post crossed audit (four LLMs, three prompts, six temperatures, three seeds; 21,600 outputs) |
| `5_Human_Audit` | Second analysis: agreement with human annotators | 300-post audit with two non-clinician annotators: forced-choice labels and aspect ratings |
| `6_MultiTask_Models` | Third analysis: learnability of the ratings | Six-head ALBERT and BioBERT models trained on the aspect ratings, and bootstrap intervals for the six-head mean F1 |
| `7_Supplementary_Analyses/Label_Comparison_and_Entropy` | Supplemental Material | Comparison of the forced-choice label with the released labels, entropy and calibration of the class-score vector, review-budget routing, and cluster-aware intervals |
| `7_Supplementary_Analyses/Label_and_Score_Models` | Supplemental Material | Models trained on the released label, the AI label, the agreement subset, and the class-score vector |
| `prompts` | First analysis (support) | Exact system-prompt strings (minimal, rubric-based, and checklist-augmented; `cot` in the code) and crossing constants of the four-LLM audit grid |

## Reproduction Map

| Manuscript item | Notebook |
|---|---|
| Table 1: agreement on the aspect ratings | `3_Multivariate_Evidence/Multivariate_Evidence.ipynb`, including the agreement on each level of the three-level rating and its value expected by chance (Panel B); intervals for the four LLMs over all conditions in Panel A from `4_MultiLLM_Audit/MultiLLM_Protocol_Sensitivity.ipynb` |
| Table 2: multi-task models | `6_MultiTask_Models/Aspect_MultiTask.ipynb` (training and test predictions); `3_Multivariate_Evidence/Multivariate_Evidence.ipynb` (Panel A, agreement with the AI ratings, and the F1 by dimension in Panel C); `6_MultiTask_Models/Aspect_Mean_F1_Bootstrap.ipynb` (intervals for the six-head mean) |
| Tables 3 and 4: the forced-choice label and entropy | `3_Multivariate_Evidence/Multivariate_Evidence.ipynb` |
| Supplementary Table S1: upstream sources | Documentation; no computation |
| Supplementary Tables S2, S6, S7 | `4_MultiLLM_Audit/MultiLLM_Protocol_Sensitivity.ipynb`; the agreement on each level in S6, Panel D, and the agreement within each prompt, temperature, and seed in S7, Panel C, from `3_Multivariate_Evidence/Multivariate_Evidence.ipynb` |
| Supplementary Table S3: composition of the 300-post audit sample | Archived output of the sample-drawing step; not computed by a notebook in this repository |
| Supplementary Tables S4, S5: hyperparameter search and selected configurations | Configuration cells and retained outputs of `6_MultiTask_Models/Aspect_MultiTask.ipynb` and the notebooks in `7_Supplementary_Analyses/Label_and_Score_Models` |
| Supplementary Tables S8, S10, S11, S14 to S21, S23, S26 to S28, S30, S31 | `3_Multivariate_Evidence/Multivariate_Evidence.ipynb` |
| Supplementary Table S9: agreement on each aspect rating | `5_Human_Audit/Human_Aspect_Agreement.ipynb` |
| Supplementary Table S12: F1 intervals of the multi-task models and their agreement with the AI ratings | `6_MultiTask_Models/Aspect_MultiTask.ipynb` (each head); `6_MultiTask_Models/Aspect_Mean_F1_Bootstrap.ipynb` (mean of six); `7_Supplementary_Analyses/Label_Comparison_and_Entropy/Cluster_Sensitivity.ipynb` (cluster bootstrap); `3_Multivariate_Evidence/Multivariate_Evidence.ipynb` (Panel C, agreement by dimension) |
| Supplementary Table S13: per-class metrics of the multi-task models | Test predictions from `6_MultiTask_Models/Aspect_MultiTask.ipynb` |
| Supplementary Tables S22, S29, S32; Supplementary Figures S1 and S2 | `7_Supplementary_Analyses/Label_Comparison_and_Entropy/Entropy_Routing.ipynb` |
| Supplementary Tables S24, S25: confusion matrix and three-way patterns | `5_Human_Audit/Human_Agreement_Audit.ipynb` |
| Supplementary Tables S33, S34: models trained on the forced-choice label | `7_Supplementary_Analyses/Label_and_Score_Models/Hard_OriginalLabel.ipynb`, `Hard_AILabel.ipynb`, `Hard_AgreementSubset.ipynb` |
| Supplementary Table S35: models trained on the class-score vector | `7_Supplementary_Analyses/Label_and_Score_Models/Soft_TrainOnly.ipynb`, `Soft_TrainAll.ipynb`; the argmax agreement in its note from `7_Supplementary_Analyses/Label_Comparison_and_Entropy/Label_Comparison_and_Entropy.ipynb` |
| Supplementary Table S36: cluster-aware intervals | `7_Supplementary_Analyses/Label_Comparison_and_Entropy/Cluster_Sensitivity.ipynb` |

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
evaluation output consumed by `4_MultiLLM_Audit`) and
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
and `5_Human_Audit` contains the full agreement analysis. For
the multi-LLM protocol audit, `prompts/` contains the exact system-prompt
strings and crossing constants that defined the 21,600-record evaluation
grid, the OSF package provides the complete crossed outputs
(`mh_labeling_final.csv`) and the 100-post stratified sample
(`mh_sample_100.csv`), and `4_MultiLLM_Audit` contains the full
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
