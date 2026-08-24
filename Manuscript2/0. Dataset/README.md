# External data inputs

Data inputs are distributed separately from GitHub.

- OSF DOI: [10.17605/OSF.IO/YQ9TZ](https://doi.org/10.17605/OSF.IO/YQ9TZ)

Download the permitted files and place them in this directory, or provide their
paths through the public notebook configuration cells and script arguments.
The external package manifest is authoritative for exact filenames and
redistribution status.

Suggested local layout:

```text
0. Dataset/
├── analysis_ready/          # full-corpus analysis input
├── modeling_outputs/        # saved predictions/splits, when distributed
├── human_audit/             # human_validation_aspect_matrix_300.csv
├── multi_llm/               # crossed multi-LLM analysis input
└── labeling/                # permitted annotation-stage files
```

The main full-corpus analyses expect an analysis-ready file containing the
released label, AI hard label, seven score columns, six aspect fields, and the
grouping text/key used to create leakage-controlled splits. The multi-LLM and
human-audit analyses require their corresponding external inputs.

Raw human-audit workbooks and other files containing restricted post text are
not stored in GitHub. Availability of any statement-free derivative is governed
by the external package manifest. The repository's aggregate
`expected_outputs/` files are verification targets, not substitutes for the
row-level inputs required to rerun an analysis.
