# Code and result provenance

This repository consolidates the versioned simulation, AARS, baseline, statistical, and strict-holdout components used in the study into a portable, repository-relative workflow.

## Scientific lineage

- Paired EnergyPlus simulation workflow: `src/simulation/`
- Frozen AARS configuration and model functions: `src/aars/`, `config/METHOD_FREEZE_v4.json`
- External baseline benchmark: `src/baselines/`
- Strict calibration-excluded evaluation: `src/strict/`
- Authoritative result tables: `data/results/strict/`

The public `src/strict/` scripts use repository-relative paths and command-line arguments. This portability cleanup does not intentionally change the scientific formulas, feature definitions, random seeds, hyperparameters, calibration masks, or metrics used in the reported evaluation.

## Authoritative result set

The calibration-excluded strict result set under `data/results/strict/` is the publication reference. One target anchor is excluded from the 1-shot test universe (999 evaluated designs/climate), and three anchors are excluded from the 3-shot universe (997 evaluated designs/climate).

Older exploratory or superseded all-1000-row outputs are not distributed as publication results.

## Reproducibility checks

`VALIDATION_REPORT.md` records the static checks, unit tests, reference-table verification, and one-climate model-refit smoke test performed on the public package.
