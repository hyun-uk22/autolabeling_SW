# Presentation Evaluation Fixture

This folder contains a small deterministic fixture for presentation/demo metrics.
It is not a substitute for real model API results.

Use it to verify that the evaluation pipeline produces the expected table:

```powershell
python evaluate_experiments.py --runs low=data\presentation_eval\runs\low cascade=data\presentation_eval\runs\cascade --gt_dir data\presentation_eval\ground_truth --out_dir data\presentation_eval\reports
```

Metrics to present:
- detection quality: precision, recall, F1, mean IoU
- efficiency: average elapsed time, estimated time saved
- cascade behavior: escalation rate, high model attempts, relative cost
