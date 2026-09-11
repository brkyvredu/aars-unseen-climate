# Data dictionary

This document summarizes the public data tables used by the AARS reproducibility workflow. Units and definitions follow the manuscript and the EnergyPlus post-processing pipeline.

## 1. Design-space tables

### `data/design_space/design_space_classguard_v2_6000.csv`
### `data/design_space/design_space_external_8000.csv`

Key fields:

| Field | Meaning |
|---|---|
| `base_design_id` | Stable identifier for one of the 1,000 paired architectural designs. |
| `city` | Climate/location assigned to the paired design. |
| `design_id` | City-specific design identifier. |
| `energy_design_id` | Unique EnergyPlus simulation identifier. |
| `room_width_m`, `room_depth_m`, `room_height_m` | Classroom geometry in metres. |
| `floor_area_m2`, `volume_m3` | Derived floor area and volume. |
| `orientation` | Principal window orientation (`N`, `E`, `S`, `W`). |
| `orientation_deg` | Orientation in degrees. |
| `window_wall_ratio` | Window-to-wall ratio used in the paired design space. |
| `glazing_type` | Glazing configuration category. |
| `wall_u_category` | Envelope insulation category. |
| `wall_u_value_w_m2k`, `roof_u_value_w_m2k`, `floor_u_value_w_m2k` | Opaque-envelope U-values (W/m²K). |
| `glazing_u_value` | Glazing U-value (W/m²K). |
| `shgc` | Solar heat-gain coefficient. |
| `vlt` | Visible light transmittance. |
| `shading_depth_m` | External shading depth; fixed at 0 in the reported study. |

## 2. Processed EnergyPlus tables

### `data/processed/energy_results_classguard_v2_6000_physics.csv`
### `data/processed/energy_results_external_8000_physics.csv`

These are the source and held-out physics-enriched simulation tables. The source table has 6,000 rows (1,000 designs × 6 climates) and the external table has 8,000 rows (1,000 designs × 8 climates).

### Simulation outputs

| Field | Meaning / unit |
|---|---|
| `total_site_energy_kwh_m2` | Annual total site energy intensity (kWh/m²-year). |
| `heating_kwh_m2` | Annual heating load intensity (kWh/m²-year). |
| `cooling_kwh_m2` | Annual cooling load intensity (kWh/m²-year). |
| `lighting_kwh_m2` | Annual lighting energy intensity (kWh/m²-year). |
| `equipment_kwh_m2` | Annual equipment energy intensity (kWh/m²-year). |
| `status` | Simulation status; publication analysis retains successful rows. |
| `warning_count`, `severe_count` | EnergyPlus warning/severe-error counts. |

The corresponding GJ fields (`*_gj`) retain the EnergyPlus aggregate values before floor-area normalization.

### Climate descriptors

| Field family | Meaning |
|---|---|
| `climate_annual_mean_temp_c`, `climate_annual_min_temp_c`, `climate_annual_max_temp_c` | Annual dry-bulb temperature statistics. |
| `climate_temp_p01_c`, `climate_temp_p05_c`, `climate_temp_p95_c`, `climate_temp_p99_c` | Dry-bulb percentiles. |
| `climate_hours_below_0c`, `climate_hours_above_30c` | Annual threshold-hour counts. |
| `climate_heating_degree_days_18c` | HDD with 18 °C base. |
| `climate_cooling_degree_days_22c`, `climate_cooling_degree_days_24c` | CDD with 22/24 °C bases. |
| `climate_annual_ghi_kwh_m2`, `climate_annual_dni_kwh_m2`, `climate_annual_dhi_kwh_m2` | Annual solar-radiation descriptors. |
| `climate_mean_relative_humidity_pct` | Mean relative humidity (%). |
| `climate_mean_wind_speed_m_s` | Mean wind speed (m/s). |
| `ground_temperature_month_01_c` … `ground_temperature_month_12_c` | Monthly ground temperatures used by the pipeline. |
| `climate_vertical_n_kwh_m2`, `climate_vertical_e_kwh_m2`, `climate_vertical_s_kwh_m2`, `climate_vertical_w_kwh_m2` | Annual vertical-plane solar irradiation by orientation. |

Latitude, longitude, and elevation are retained for metadata/audit purposes; latitude and longitude are excluded from the predictive feature set described in the manuscript.

### Physics-derived features

| Field | Definition / interpretation |
|---|---|
| `envelope_area_to_volume_ratio` | Envelope area-to-volume ratio. |
| `ua_envelope_w_k` | Aggregate envelope heat-transfer coefficient UA (W/K). |
| `ua_intensity_w_m2k` | UA normalized by floor area (W/m²K). |
| `solar_aperture_ratio` | Window area × SHGC normalized by floor area. |
| `heating_transmission_proxy_kwh_m2` | UA-intensity × HDD18 × 24 / 1000. |
| `cooling_transmission_proxy_kwh_m2` | UA-intensity × CDD22 × 24 / 1000. |
| `orientation_solar_gain_proxy_kwh_m2` | Solar aperture × orientation-specific annual vertical irradiation. |

## 3. Anchor-selection tables

### `data/anchors/external_selected_anchor_bank.csv`

| Field | Meaning |
|---|---|
| `target_city` | Held-out target climate. |
| `anchor_budget` | Allowed number of target simulations (1 or 3). |
| `rank` | Rank within the selected anchor order. |
| `anchor_id` | `base_design_id` of the selected calibration design. |

### `data/anchors/external_source_only_anchor_scores.csv`

Contains source-only, target-weather-aware scores for each candidate anchor and external target climate. Target energy labels are not used to compute these ranking scores.

## 4. Strict evaluation tables

### `data/results/strict/external_metrics_by_city_strict.csv`

Key fields:

| Field | Meaning |
|---|---|
| `target_city` | Held-out target climate. |
| `anchor_budget` | 1-shot or 3-shot adaptation budget. |
| `target` | `heating`, `cooling`, or `total`. |
| `universe` | Evaluation universe; reported rows are `strict_holdout`. |
| `n_calibration` | Number of target designs used for adaptation. |
| `n_eval` | Number of non-calibration test designs (999 or 997). |
| `calibration_anchor_ids` | Anchor IDs excluded from the test metrics. |
| `mae` | Mean absolute error. |
| `rmse` | Root mean squared error. |
| `r2` | Coefficient of determination. |
| `spearman` | Spearman rank correlation. |
| `kendall` | Kendall rank correlation. |
| `design_regret_kwh_m2` | Energy regret of the design selected by predicted minimum. |
| `top_recovery` | Fraction of the true top-5% designs recovered by the predicted top-5%. |
| `top_k` | Number of designs represented by the top-5% set. |

### `data/results/strict/external_all_methods_by_city_strict.csv`

Contains the same predictive/decision metrics for AARS and the aligned comparison protocols. `comparison_budget` identifies the holdout universe used for fair comparison.

### `data/results/strict/paired_external_comparisons_strict.csv`

Contains climate-level paired comparisons. Important fields include `mean_a`, `mean_b`, `relative_improvement_pct`, `exact_signflip_p_two_sided`, and the numbers of climates favoring each method.

### `data/results/strict/bootstrap_climate_ci_strict.csv`

Contains climate-level bootstrap means and 95% confidence intervals. The independent unit is the external climate (`n_climates = 8`).

### Other strict tables

- `aars_1shot_vs_3shot_common997_by_city.csv`: direct 1-shot/3-shot comparison on the common 997-design universe.
- `accuracy_budget_tradeoff_strict.csv`: 0/1/3 target-simulation accuracy and design-recovery summary.
- `external_summary_strict.csv`: AARS aggregate summaries.
- `external_all_methods_summary_strict.csv`: aggregate method summaries.
- `worst_climate_summary_strict.csv`: worst-climate summaries for selected metrics.
- `statistical_metadata_strict.json`: seeds, bootstrap count, independent unit, and holdout sizes.
