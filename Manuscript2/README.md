# Manuscript 2: LLM-Based Annotation for Affective Text Classification

This directory contains the R2 code and sanitized public notebooks for the
manuscript's annotation, characterization, downstream-modeling, human-audit,
entropy-routing, and multi-LLM robustness analyses.

The existing public notebook names have been retained so earlier GitHub links
continue to work. The modeling notebooks now reflect the R2 analysis settings.

## What is included

- Sanitized notebooks documenting the annotation and analysis workflows.
- The six R2 downstream-modeling notebooks for hard labels, soft labels, and
  six aspect heads, including their safe executed result outputs.
- CPU scripts for entropy routing, multi-LLM robustness, aspect mean-F1
  intervals, and chance-corrected human aspect agreement.
- An [environment record](ENVIRONMENT.md) and a
  [table/figure reproduction index](REPRODUCIBILITY_INDEX.md).

## What is not included

The GitHub repository does not contain corpus rows, post text, row-level model
predictions, human-audit workbooks, the crossed multi-LLM output, model
checkpoints, API credentials, or private local metadata. These materials are
either supplied separately through the external data package or are restricted
from redistribution.

The accompanying data package is available through OSF:

- DOI: [10.17605/OSF.IO/YQ9TZ](https://doi.org/10.17605/OSF.IO/YQ9TZ)

GitHub therefore provides code, configurations, and safe executed-notebook
evidence; the external package is the source for inputs that can be redistributed.
Restricted inputs remain outside the public release, so GitHub alone is not a
complete data bundle. See the
[dataset instructions](0.%20Dataset/README.md) for the expected local layout.

## Directory map

| Stage | Public contents |
|---|---|
| `0. Dataset/` | External-package DOI and local input-layout guidance. |
| `1. Code for LLM Annotation/` | Sanitized `gpt-4o-mini` prompt, JSON schema, parsing, retry, and checkpoint protocol. |
| `2. Code for Label Comparison and Entropy/` | Released-versus-AI agreement, entropy, score dispersion, and affective co-occurrence analyses. |
| `3. Hard Label Modeling/` | R2 released-label, AI-hard-label, and agreement-subset models. |
| `4. Soft Label Modeling/` | R2 soft-train and full soft-label transformer workflows. |
| `5. Multi-Task Aspect Modeling/` | R2 six-head aspect models. |
| `6. Revision Validation and Robustness Audits/` | Human audit, routing baselines and review-budget curves, multi-LLM analyses, and bootstrap intervals. |

## Reproduction workflow

1. Download any permitted external inputs and place them under `0. Dataset/`
   following its README, or pass equivalent authorized paths to the
   scripts/notebooks.
2. Create the CPU environment documented in `ENVIRONMENT.md` for the
   revision-audit scripts.
3. Use the public notebooks for the annotation-characterization and modeling
   stages. GPU access is recommended for transformer training.
4. Use the scripts in `6. Revision Validation and Robustness Audits/` for the
   CPU-only R2 analyses.
5. Keep regenerated standalone files under an untracked local `outputs/`
   directory. Consult `REPRODUCIBILITY_INDEX.md` for the artifact-to-code map.

The API annotation notebook is a source-protocol record. Pre-generated,
permitted annotation inputs are used by the downstream analyses, so reproducing
the reported tables does not require re-querying an API. A new API run may
differ because hosted model endpoints can change.

## Evidence boundary

The six R2 modeling notebooks retain the executed training logs, metrics,
tables, and figures from the reported reruns. Only cells exposing post-text
previews or private path setup were cleared; credentials, Colab account
metadata, and private paths were removed. The annotation, label-exploration,
and archived revision-reader notebooks remain output-cleared. Public
path/configuration edits are not presented as a separate re-execution of the
modeling notebooks.

The exact original Colab/GPU software image was not retained. The tested CPU
release environment is documented separately and is not presented as the
original transformer-training environment.

## License

Code is released under the repository's
[GNU General Public License v3.0](../LICENSE). External datasets remain subject
to their source licenses and redistribution terms.
