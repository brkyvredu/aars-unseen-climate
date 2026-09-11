# Artifact provenance

This package was consolidated from the original AARS/CLASS-Guard artifacts recovered from the project history.

## Exact recovered components

- CLASS-Guard v2 paired EnergyPlus simulation code: `src/simulation/`
- Adaptive Anchor v4 development code: `src/aars/adaptive_anchor_loco.py`
- Frozen external evaluator v5: `src/aars/evaluate_frozen_v4_external.py`
- External baseline benchmark v6: `src/baselines/external_baseline_benchmark.py`
- Strict holdout v8 result tables: `data/results/strict/`
- Original strict v8 scripts are preserved under `provenance/recovered_scripts/`.

## Portable strict scripts

The scripts under `src/strict/` are portability-cleaned entrypoints assembled from the exact recovered v5/v6/v8 code. Their scientific logic, seeds, features, hyperparameters, calibration masks, and metrics are preserved; only hard-coded temporary paths/imports were replaced by repository-relative CLI arguments.

A one-climate reproduction check (Ardahan) was run in the pinned analysis environment and matched the recovered v8 strict metrics exactly for MAE, RMSE, R², Spearman, Kendall, regret, and top-5% recovery.

## Authoritative result set

The strict calibration-excluded v8 results are the public reference results. Older all-1000-row archived fits are not used as publication truth because refitting HistGradientBoosting under the recorded environment produced small numerical differences. See `data/results/strict/STRICT_EVALUATION_AUDIT.md`.
