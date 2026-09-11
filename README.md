# AARS — Reproducibility Package

This repository contains the data, research code, strict external-validation results, and portable reproduction entrypoints associated with the manuscript:

**Simulation-Budgeted Few-Shot Adaptation of Building Energy Surrogates in Unseen Climates: A Physics-Guided Anchor-Relative Approach**

The repository consolidates the versioned study components into repository-relative workflows while preserving the frozen method configuration, evaluation masks, seeds, hyperparameters, and publication result tables.

## 1. What this package reproduces

The study uses a paired simulation design:

- **1,000 architectural designs** shared across all climates.
- **6 source climates**: Ankara, Antalya, Bursa, Erzurum, Istanbul, Izmir.
- **6,000 source EnergyPlus simulations**.
- **8 held-out climates**: Ardahan, Bartin, Bingol, Canakkale, Hatay, Kars, Konya, Sanliurfa.
- **8,000 external EnergyPlus simulations**.
- **1-shot adaptation**: 1 selected target simulation; the calibration anchor is excluded from evaluation, leaving **999 test designs/climate**.
- **3-shot adaptation**: 3 selected target simulations; all three anchors are excluded, leaving **997 test designs/climate**.

No external target energy label is used for anchor ranking. The full external simulations are retained as benchmarking ground truth, while adaptation sees only the selected target anchor simulation(s).

### Locked publication results

For total annual site energy across the eight held-out climates:

| Protocol | Mean MAE (kWh/m²-year) | Reported R² | Top-5% recovery |
|---|---:|---:|---:|
| AARS 1-shot | 3.330520 | 0.971 | 89.0% |
| AARS 3-shot | 2.955668 | 0.979 | 90.8% |

Strict comparison means:

| Protocol | Mean total-energy MAE |
|---|---:|
| Zero-shot absolute | 23.898214 |
| 1-shot global-bias calibration | 5.822442 |
| 1-shot raw anchor-relative transfer | 4.020386 |
| AARS 1-shot | 3.330520 |
| 3-shot global-bias calibration | 5.767655 |
| 3-shot raw anchor-relative transfer | 4.823735 |
| AARS 3-shot | 2.955668 |

On the aligned 3-shot holdout, AARS reduced MAE by **87.63%** relative to zero-shot prediction, **48.75%** relative to global-bias calibration, and **38.73%** relative to raw anchor-relative transfer. Each of these three paired climate-level comparisons favored AARS in 8/8 climates with exact two-sided sign-flip `p = 0.0078125`.

The 3-shot versus 1-shot comparison on the common 997-design universe produced an **11.26%** mean MAE reduction, but only 6/8 climates favored 3-shot and the exact sign-flip result was `p = 0.1171875`; it should not be interpreted as a statistically significant uniform improvement.

See `data/results/strict/STRICT_EVALUATION_AUDIT.md` and `PROVENANCE.md` for the authoritative reproducibility notes.

## 2. Repository structure

```text
.
├── README.md
├── CITATION.cff
├── DATA_DICTIONARY.md
├── PROVENANCE.md
├── REPRODUCIBILITY_MANIFEST.md
├── VALIDATION_REPORT.md
├── LICENSE
├── DATA_LICENSE.md
├── requirements-analysis.txt
├── requirements-simulation.txt
│
├── config/
│   ├── METHOD_FREEZE_v4.json
│   └── EXTERNAL_CLIMATE_SELECTION_TEMPLATE.json
│
├── data/
│   ├── design_space/
│   │   ├── design_space_classguard_v2_6000.csv
│   │   └── design_space_external_8000.csv
│   ├── processed/
│   │   ├── energy_results_classguard_v2_6000.csv
│   │   ├── energy_results_classguard_v2_6000_physics.csv
│   │   ├── energy_results_external_8000.csv
│   │   └── energy_results_external_8000_physics.csv
│   ├── anchors/
│   │   ├── external_selected_anchor_bank.csv
│   │   └── external_source_only_anchor_scores.csv
│   ├── results/strict/
│   │   ├── external_metrics_by_city_strict.csv
│   │   ├── external_all_methods_by_city_strict.csv
│   │   ├── paired_external_comparisons_strict.csv
│   │   ├── bootstrap_climate_ci_strict.csv
│   │   ├── aars_1shot_vs_3shot_common997_by_city.csv
│   │   ├── accuracy_budget_tradeoff_strict.csv
│   │   ├── external_summary_strict.csv
│   │   ├── external_all_methods_summary_strict.csv
│   │   ├── worst_climate_summary_strict.csv
│   │   └── statistical_metadata_strict.json
│   └── metadata/
│       ├── climate_descriptor_summary.csv
│       ├── source_run_manifest_sanitized.json
│       └── external_run_manifest_sanitized.json
│
├── src/
│   ├── simulation/       # paired design / EnergyPlus / EPW physics pipeline
│   ├── aars/             # AARS v4 + frozen external evaluator v5
│   ├── baselines/        # baseline benchmark v6
│   ├── strict/           # portable strict-holdout reproduction entrypoints
│   ├── run_reproduction.py
│   ├── verify_reference_results.py
│   └── verify_reproduced_outputs.py
│
├── figures/
│   ├── scripts/
│   └── generated/
│
├── models/
│   └── README.md
│
├── tests/
└── validation/
    └── smoke_ardahan/
```

## 3. Which files are authoritative?

Use these files as the publication truth:

- `data/processed/energy_results_classguard_v2_6000_physics.csv`
- `data/processed/energy_results_external_8000_physics.csv`
- `data/anchors/external_selected_anchor_bank.csv`
- `data/anchors/external_source_only_anchor_scores.csv`
- everything under `data/results/strict/`
- `config/METHOD_FREEZE_v4.json`

Older all-1000-row external metrics and pre-strict result variants are deliberately not part of the public result set.

## 4. Frozen AARS method

The frozen method uses:

- a **10-design maximin candidate anchor bank** in standardized design-only space,
- source-only nested leave-one-climate-out anchor scoring,
- target-weather-aware weighting without target energy labels,
- **heating**: paired physics-delta Ridge baseline plus a nonlinear HistGradientBoosting residual,
- **cooling**: nonlinear raw anchor-relative HistGradientBoosting transfer,
- **total energy**: predicted heating + predicted cooling + source-domain mean non-HVAC intensity.

Frozen outer HGB configuration:

```text
max_iter = 220
learning_rate = 0.06
max_leaf_nodes = 31
min_samples_leaf = 20
l2_regularization = 2.0
random_state = 42
```

Nested source-only anchor scoring uses the same structure with `max_iter = 100` for the inner scoring models. The heating physics baseline uses Ridge with `alpha = 1` and `fit_intercept = False`.

The final external anchor order is stable across the eight targets:

```text
base_0948 → base_0945 → base_0724
```

The exact scores are distributed in `data/anchors/external_source_only_anchor_scores.csv`.

## 5. Models: why there are no `.pkl` / `.joblib` files

No serialized model binary is required to reproduce the paper. The recovered research code fits the frozen Ridge and HistGradientBoosting estimators from the processed **source** simulation data at evaluation time. No model binary was found in the preserved project artifacts, and distributing newly serialized estimators would create a new artifact that was not used for the reported study.

The model definition is therefore represented by the source data, frozen configuration, exact analysis code, and pinned software environment. See `models/README.md`.

## 6. Quick verification

Create an analysis environment and install:

```bash
python -m pip install -r requirements-analysis.txt
```

Then run the fast reference-table check:

```bash
python src/run_reproduction.py --quick-reference-check
```

This validates the packaged strict tables against the locked publication values without fitting models.

A one-climate strict model-refit smoke check can be run with:

```bash
python src/strict/evaluate_aars_strict.py \
  --cities Ardahan \
  --output-dir outputs/smoke_ardahan
```

During consolidation, the Ardahan smoke run matched the recovered strict-v8 city metrics **exactly** for MAE, RMSE, R², Spearman, Kendall, design regret, and top-5% recovery under both 1-shot and 3-shot budgets.

## 7. Full strict reproduction

To refit AARS for all eight external climates, reproduce the strict baselines, rerun the climate-level statistics, and compare generated outputs with the packaged v8 references:

```bash
python src/run_reproduction.py
```

The full run is compute-intensive because the frozen HGB models are repeatedly refit for each target climate and anchor. Generated files are written under `outputs/` and are not required for using the already-distributed reference tables.

## 8. Reproducing the EnergyPlus simulations

Analysis reproduction does **not** require rerunning EnergyPlus because both the raw simulation outputs and the physics-enriched processed tables are distributed.

To regenerate the simulation data from scratch, the recovered pipeline is under `src/simulation/`. It requires:

- EnergyPlus **26.1.0**,
- the relevant TMYx.2009-2023 EPW files,
- the dependencies in `requirements-simulation.txt`.

The EPW files were obtained from Climate.OneBuilding and are **not redistributed** in this repository. Users should obtain the corresponding weather files from the original provider and respect the provider's applicable terms.

The recorded simulation assumptions include:

- weekday occupancy 08:00–17:00,
- occupied heating/cooling setpoints 20/26 °C,
- unoccupied heating/cooling setpoints 16/30 °C,
- occupancy density 0.536 person/m²,
- activity 120 W/person,
- lighting 9 W/m²,
- equipment 5 W/m²,
- outdoor air 5 L/s-person + 0.6 L/s-m²,
- infiltration 0.30 ACH,
- four timesteps per hour,
- IdealLoadsAirSystem,
- annual weather simulation,
- DDY objects detected when available but design-day sizing periods disabled for the annual study.

The sanitized run manifests are in `data/metadata/`.

## 9. Software environment

Authoritative strict analysis environment:

```text
NumPy         2.3.5
pandas        2.2.3
SciPy         1.17.0
scikit-learn  1.8.0
matplotlib    3.10.8
```

The strict rerun matters because refitting HistGradientBoosting in a later software environment did not reproduce older archived all-1000-row fits bit-for-bit. The calibration-excluded strict-v8 rerun distributed here is the authoritative public result set. No external result was used to tune the final method.

## 10. Statistical analysis

The inferential unit is the **external climate** (`n = 8`), not the 8,000 individual target designs.

The strict statistical pipeline uses:

- 20,000 climate-level bootstrap resamples,
- exact paired sign-flip tests across all `2^8 = 256` sign patterns,
- a common 997-design universe for the direct 1-shot versus 3-shot comparison.

The portable implementation is `src/strict/statistical_robustness_strict.py`.

## 11. Figures

Data-driven figure scripts are supplied under `figures/scripts/` and use the packaged authoritative data. Figure 1 is a conceptual workflow schematic and is not a numerical result generated by the analysis pipeline.

Example:

```bash
python figures/scripts/make_fig03_zero_shot_recovery.py
python figures/scripts/make_fig04_total_energy_mae_with_relative_change.py
python figures/scripts/make_fig05_decision_reliability_by_city.py
python figures/scripts/make_fig06_budget_tradeoff.py
```

Generated PDF/PNG outputs are written to `figures/generated/`.

## 12. Code and result provenance

The repository uses portable, repository-relative analysis entrypoints while retaining the frozen scientific configuration used for the reported study. Details are in `PROVENANCE.md` and `REPRODUCIBILITY_MANIFEST.md`.

## 13. Data dictionary

See `DATA_DICTIONARY.md` for identifiers, targets, design features, climate descriptors, physics-derived features, and strict metric columns.

## 14. Citation

`CITATION.cff` includes the public GitHub repository URL. After the Zenodo archival release, add the Zenodo DOI (and the article DOI when available) to the citation metadata and manuscript Data Availability statement.

## 15. License

Source code is released under the MIT License (`LICENSE`). Unless otherwise indicated, author-generated processed simulation outputs and derived research tables under `data/` are released under CC BY 4.0 (`DATA_LICENSE.md`). Third-party EPW weather files are not included.

## 16. Authors

- Özge Güneş Yavru — Bursa Uludağ University
- İsmail Burak Yavru — Bursa Uludağ University

Corresponding author: **ozgeyavru@uludag.edu.tr**
