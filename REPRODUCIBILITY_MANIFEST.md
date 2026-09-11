# Reproducibility manifest

This repository consolidates the versioned analysis components used to produce the manuscript results into a portable, repository-relative workflow.

## Scientific components

| Component | Role in the study | Public implementation |
|---|---|---|
| AARS method configuration (v4) | Anchor-relative model and frozen hyperparameters | `src/aars/`, `config/METHOD_FREEZE_v4.json` |
| External evaluation workflow (v5) | Held-out climate evaluation and target-weather use | `src/aars/evaluate_frozen_v4_external.py` |
| Baseline benchmark (v6) | Zero-shot, global-bias, and raw anchor-relative comparisons | `src/baselines/`, `src/strict/external_baseline_benchmark_strict.py` |
| Climate-level statistics | Bootstrap confidence intervals and paired sign-flip tests | `src/strict/statistical_robustness_strict.py` |
| Strict holdout evaluation (v8) | Exclusion of calibration anchors from all reported test metrics | `src/strict/evaluate_aars_strict.py`, `data/results/strict/` |

## Authoritative publication artifacts

Use the following files as the publication reference set:

- `data/processed/energy_results_classguard_v2_6000_physics.csv`
- `data/processed/energy_results_external_8000_physics.csv`
- `data/anchors/external_selected_anchor_bank.csv`
- `data/anchors/external_source_only_anchor_scores.csv`
- all files under `data/results/strict/`
- `config/METHOD_FREEZE_v4.json`

The strict evaluation excludes the calibration designs from the test universe: 999 designs per climate for 1-shot adaptation and 997 for 3-shot adaptation.

## Portability

The scripts under `src/strict/` use repository-relative paths and command-line arguments. The scientific formulas, feature definitions, seeds, hyperparameters, strict masks, and metrics are preserved from the study workflow.

## Validation

Repository validation is summarized in `VALIDATION_REPORT.md`. The fast reference check is:

```bash
python src/run_reproduction.py --quick-reference-check
```

A one-climate model-refit smoke check is:

```bash
python src/strict/evaluate_aars_strict.py --cities Ardahan --output-dir validation/smoke_ardahan
```
