# Strict external evaluation: statistical findings

Primary evaluation excludes the selected target-climate calibration designs from every metric and from the decision candidate universe. Thus, one-shot AARS is evaluated on 999 designs per climate and three-shot AARS on 997 designs per climate. No method component, anchor, hyperparameter, source climate, or external climate was changed.

## Three-shot AARS, total energy
- MAE: 2.956 kWh/m²-year (95% climate-bootstrap CI 2.393–3.538)
- R²: 0.979 (95% CI 0.968–0.990)
- Top-5% recovery: 0.907 (95% CI 0.838–0.960)
- Design regret: 1.292 kWh/m²-year (95% CI 0.538–2.336)

## Paired comparisons
- AARS_v4_3_shot vs zero_shot_absolute_aligned_3shot: mean MAE 2.956 vs 23.901; relative improvement 87.63%; improvement CI [8.253, 34.834]; exact sign-flip p=0.007812; A better in 8/8 climates.
- AARS_v4_3_shot vs 3_shot_global_bias: mean MAE 2.956 vs 5.768; relative improvement 48.75%; improvement CI [0.948, 5.232]; exact sign-flip p=0.007812; A better in 8/8 climates.
- AARS_v4_3_shot vs 3_shot_raw_relative: mean MAE 2.956 vs 4.824; relative improvement 38.73%; improvement CI [1.203, 2.492]; exact sign-flip p=0.007812; A better in 8/8 climates.
- AARS_v4_1_shot vs zero_shot_absolute_aligned_1shot: mean MAE 3.331 vs 23.899; relative improvement 86.06%; improvement CI [8.230, 34.045]; exact sign-flip p=0.007812; A better in 8/8 climates.
- AARS_v4_1_shot vs 1_shot_global_bias: mean MAE 3.331 vs 5.822; relative improvement 42.80%; improvement CI [0.926, 4.535]; exact sign-flip p=0.007812; A better in 8/8 climates.
- AARS_v4_1_shot vs 1_shot_raw_relative: mean MAE 3.331 vs 4.020; relative improvement 17.16%; improvement CI [0.240, 1.110]; exact sign-flip p=0.039062; A better in 7/8 climates.
- AARS_v4_3_shot_common997 vs AARS_v4_1_shot_common997: mean MAE 2.956 vs 3.331; relative improvement 11.26%; improvement CI [-0.003, 0.816]; exact sign-flip p=0.117188; A better in 6/8 climates.
