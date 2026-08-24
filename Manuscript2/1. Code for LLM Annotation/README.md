# Code for LLM Annotation

This folder contains a sanitized source copy of the original LLM annotation notebook:

- `MentalHealth_4omini_Labeling.ipynb`

The notebook documents the rubric-conditioned annotation protocol used to generate the hard labels, score vectors, and aspect ratings used in the manuscript. The complete prompt and JSON output schema are in the cell defining `BATCH_ANNOTATION_INSTRUCTIONS_V2`.

## Sanitization

The shared notebook excludes private credentials and data files. The API credential-check lines were removed and Colab/user metadata were stripped. Cell outputs are retained for transparency. This repository does not store raw or derived CSV data, generated output files, API keys, or model checkpoints.

## Protocol Summary

The annotation protocol uses `gpt-4o-mini` with `temperature=0`. The runtime configuration in the notebook specifies batch size, concurrency, retry, and checkpoint settings. The parser records the returned label, score vector, aspect flags, model version, system fingerprint, timestamp, and raw JSON in the intermediate output; the final cleaned annotation file removes `u_raw_json`.

The pre-generated annotation outputs needed for downstream analyses are provided through the OSF data package. Readers can inspect the full prompt/schema here without re-querying the API.
