# Multi-LLM protocol-sensitivity analysis

`analyze_multillm_documentation.py` analyzes the archived 100-item crossed
audit: four model families by three prompts by six temperatures by three seeds,
for 21,600 model outputs. It performs no API calls.

From `Manuscript2/`:

```bash
python3 "6. Revision Validation and Robustness Audits/05_multillm_documentation/analyze_multillm_documentation.py" \
  --full-corpus "0. Dataset/analysis_ready/mental_health_unified_labels_final.csv" \
  --multillm-outputs "0. Dataset/multi_llm/mh_labeling_final.csv" \
  --output-dir "6. Revision Validation and Robustness Audits/05_multillm_documentation/outputs" \
  --bootstrap-resamples 2000
```

Both input files are external and are not bundled in GitHub. The script
reconstructs and checks the historical entropy-stratified sample, validates the
crossed design and schema, and exports the sample, factor, agreement, entropy,
aspect-count, co-occurrence, prevalence, and strength summaries used in Tables
S6-S8.

Generated aggregate CSVs and the validation manifest should remain in the
untracked local `outputs/` directory. Binary aspect presence supports
presence/co-occurrence analyses; the original NONE/WEAK/CLEAR ratings are
analyzed separately for ordinal reproducibility.
