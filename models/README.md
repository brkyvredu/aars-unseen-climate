# Model artifacts

No serialized `.pkl`/`.joblib` model is required for the reported AARS results.
The study fits the frozen Ridge and HistGradientBoosting models from the processed
source simulation table at evaluation time. This matches the recovered research
code and avoids distributing stale environment-specific binary estimators.

The frozen model settings are recorded in `config/METHOD_FREEZE_v4.json`,
`src/aars/evaluate_frozen_v4_external.py`, and the pinned analysis environment.
