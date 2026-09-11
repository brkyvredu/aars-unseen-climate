# Figure reproduction

The repository contains data-driven figure scripts for the numerical figures.

- `make_fig02_climate_space.py`: physical source/external climate coverage.
- `make_fig03_zero_shot_recovery.py`: aligned zero-shot to 3-shot AARS recovery.
- `make_fig04_total_energy_mae_with_relative_change.py`: strict city/method MAE heatmap with aligned zero-shot changes.
- `make_fig05_decision_reliability_by_city.py`: 1-shot/3-shot decision-reliability plot with collision-free city labels.
- `make_fig06_budget_tradeoff.py`: creates the prediction-error and design-recovery panels as separate publication files.

Figure 1 is a conceptual workflow schematic rather than a numerical result and is therefore not generated from the result tables in this repository.

The scripts are supplied to reproduce the scientific data shown in the figures. Minor manuscript-specific layout operations (for example combining two exported panels into one multi-panel figure) are presentation-only and do not alter the underlying values.
