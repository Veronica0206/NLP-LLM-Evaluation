# Table 4 mean-F1 confidence intervals

`compute_aspect_mean_f1_ci.py` resamples held-out posts once per bootstrap
replicate, applies the same sampled rows to all six aspect heads, and then
averages head-level F1 values. This is the CPU calculation for the Table 4 mean
weighted- and macro-F1 intervals.

From `Manuscript2/`:

```bash
python3 "6. Revision Validation and Robustness Audits/06_table_ci_reporting/compute_aspect_mean_f1_ci.py" \
  --predictions "0. Dataset/modeling_outputs/06_AspectLabel/test_predictions_aspect.csv" \
  --output "6. Revision Validation and Robustness Audits/06_table_ci_reporting/outputs/aspect_mean_f1_bootstrap.csv"
```

The row-level prediction file is external and is not stored in GitHub. The
script uses 1,000 percentile-bootstrap resamples and seed 2025. The
statement-free result in `expected_outputs/` is the aggregate verification
target.
