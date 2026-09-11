# Strict evaluation audit

The frozen AARS method, selected anchors, source/external climates, and hyperparameters were not changed. The only protocol change is exclusion of the calibration anchors from the evaluation universe: 999 designs/climate for 1-shot and 997 for 3-shot.

## Important reproducibility note

Re-fitting the frozen HistGradientBoosting models in the current environment (scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0) did not reproduce the archived all-1000-row metrics bit-for-bit. The differences are small for aggregate MAE/R² but can be larger for single-design regret because the selected optimum can change when near-optimal alternatives are close. Therefore, the strict results in this package are treated as the authoritative reproducible rerun and the software versions are recorded. No external result was used to tune the method.

### Total-energy comparison

| Budget | Archived all-1000 MAE | Frozen rerun all-1000 MAE | Strict holdout MAE | Mask-only MAE change | Archived R² | Strict R² |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.350886 | 3.328891 | 3.330520 | +0.001629 | 0.969911 | 0.970948 |
| 3 | 2.941874 | 2.953947 | 2.955668 | +0.001720 | 0.979362 | 0.979393 |

The calibration-row exclusion itself changes mean total-energy MAE by less than 0.002 kWh/m²-year in both budgets. The larger difference between archived and rerun values arises from refitting rather than from the holdout mask.

## Primary strict numbers

- 1-shot total: MAE 3.331 kWh/m²-year, R² 0.971, top-5% recovery 0.890, n=999/climate.
- 3-shot total: MAE 2.956 kWh/m²-year, R² 0.979, top-5% recovery 0.908, n=997/climate.
- 3-shot vs aligned zero-shot: 87.63% lower MAE; exact paired sign-flip p=0.007812.
- 3-shot vs 3-shot global bias: 48.75% lower MAE; p=0.007812.
- 3-shot vs 3-shot raw relative: 38.73% lower MAE; p=0.007812.
- 3-shot vs 1-shot on common 997-design holdout: 11.26% lower mean MAE; p=0.117188.
