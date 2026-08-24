# Code for LLM annotation

`MentalHealth_4omini_Labeling.ipynb` is the sanitized source-protocol notebook
for the full-corpus rubric-guided annotation.

It documents the `gpt-4o-mini` prompt, `temperature=0` setting, structured JSON
schema, parser, retry/checkpoint behavior, and post-processing used to produce:

- one seven-class hard label;
- one seven-class score vector; and
- six aspect ratings.

The public notebook does not contain credentials, post text, row-level API
responses, or Colab execution-user metadata. All notebook outputs were cleared
during release sanitization. Credentials must be supplied at run time through
the documented environment/secret mechanism and must never be committed.

The manuscript analyses use the permitted pre-generated annotation files from
the external data package. Re-querying the API is not required to reproduce the
reported downstream analyses, and a new API run should not be assumed to be
byte-identical to the archived annotation output.
