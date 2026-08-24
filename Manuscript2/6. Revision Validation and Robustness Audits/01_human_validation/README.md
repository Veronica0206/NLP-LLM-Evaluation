# Human-agreement analysis

`compute_human_aspect_kappas.py` produces the canonical seed-2025,
post-cluster-bootstrap agreement results used in Table 5 and Table S10.

From `Manuscript2/`:

```bash
python3 "6. Revision Validation and Robustness Audits/01_human_validation/compute_human_aspect_kappas.py" \
  --audit-matrix "0. Dataset/human_audit/human_validation_aspect_matrix_300.csv" \
  --output-dir "6. Revision Validation and Robustness Audits/01_human_validation/outputs"
```

The external audit matrix contains only sample IDs and released, AI, and human
labels; it contains no post text. Raw returned workbooks remain restricted and
must not be committed. The CSVs and manifest in `expected_outputs/` are
aggregate reference outputs for verification; they are not the audit input.

The audit used two independent, label-blinded non-expert annotators and a
deliberately enriched 300-post sample. Its results characterize agreement and
task difficulty; they are not corpus-representative estimates and do not form
an expert-adjudicated reference standard.
