# Recovery inventory

This file records the research artifacts recovered from the project/conversation history and how they were used in the consolidated package.

## Recovered versioned research archives

| Historical artifact | Role | Public-package use |
|---|---|---|
| `classguard_adaptive_anchor_v4.zip` | Adaptive-anchor/AARS development code | exact `adaptive_anchor_loco.py` recovered into `src/aars/` |
| `classguard_external_validation_v5.zip` | Frozen external-validation method | exact external evaluator + climate-selection/design-space scripts + freeze config recovered |
| `classguard_external_baselines_v6.zip` | Zero-shot, global-bias and raw-relative baselines | exact baseline helper recovered into `src/baselines/` |
| `classguard_statistical_robustness_v7.zip` | Pre-strict statistical/figure package | preserved privately; superseded by strict-v8 publication results |
| `classguard_strict_holdout_v8.zip` | Calibration-excluded final evaluation | authoritative strict result tables; original scripts preserved under `provenance/` |

## Recovered CLASS-Guard v2 simulation pipeline

The consolidated package contains:

- `generate_paired_design_space_v2.py`
- `run_energy_paired_v2.py`
- `add_epw_physics_features_v2.py`
- `audit_paired_results_v2.py`
- `validate_smoke_test_v2.py`
- the `classguard_v2/` support package
- external-climate design-space and selection scripts

These reproduce the paired EnergyPlus workflow represented in the processed source/external tables, subject to availability of EnergyPlus 26.1.0 and the original EPW files.

## Authoritative recovered data

- source physics table: 6,000 rows
- external physics table: 8,000 rows
- selected external anchor bank
- source-only external anchor scores
- strict-v8 city-level AARS metrics
- strict-v8 all-method city-level metrics
- strict-v8 paired comparison table
- strict-v8 bootstrap confidence intervals
- strict-v8 common-997 1-shot/3-shot comparison
- strict-v8 accuracy/simulation-budget table

## Models

No original serialized `.pkl`, `.pickle`, or `.joblib` models were found in the preserved artifacts. This is consistent with the recovered analysis code, which refits the frozen source models from the processed source data. The public package therefore does not fabricate model binaries.

## Portability cleanup

The original strict-v8 scripts used temporary `/mnt/data/strict_eval_work` paths and a temporary import layout. They remain unchanged under `provenance/recovered_scripts/`. Repository-relative portable equivalents are provided under `src/strict/`.

The portable strict AARS runner was smoke-tested against the recovered v8 Ardahan rows and reproduced all reported metrics exactly.
