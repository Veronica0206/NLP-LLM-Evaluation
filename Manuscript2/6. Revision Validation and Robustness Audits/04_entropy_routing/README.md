# Entropy routing and review-budget baselines

`run_entropy_routing.py` compares normalized entropy with maximum score,
top-two score margin, out-of-fold released-category routing, and random routing
over review budgets. It is CPU-only and makes no API calls.

From `Manuscript2/`:

```bash
python3 "6. Revision Validation and Robustness Audits/04_entropy_routing/run_entropy_routing.py" \
  --input "0. Dataset/analysis_ready/mental_health_unified_labels_final.csv" \
  --output-dir "6. Revision Validation and Robustness Audits/04_entropy_routing/outputs"
```

The analysis-ready corpus is external and is not stored in GitHub. The public
`expected_outputs/` directory contains aggregate budget curves, key operating
points, AUROCs, category rates, and a statement-free manifest. Row-level routing
scores are intentionally excluded.

The primary settings are five grouped folds for the category baseline, a
0.5%-to-100% review-budget grid, and the script's fixed outcome-independent tie
handling. See the manifest for exact definitions and seeds.
