# Validation report for the consolidated public package

Validation performed during consolidation:

## Static validation

- All Python files under `src/`, `tests/`, `figures/scripts/`, and recovered strict provenance scripts compile successfully with `py_compile`.
- No personal user-directory paths remain in the public data/manifests. Historical hard-coded `/mnt/data/...` paths are retained only in `provenance/recovered_scripts/` to preserve the original strict-v8 artifacts.

## Unit tests

Command:

```bash
PYTHONPATH=src/simulation pytest tests -q
```

Result:

```text
3 passed
```

## Authoritative result-table verification

Command:

```bash
python src/verify_reference_results.py
```

Verified values include:

- AARS 1-shot total MAE = 3.330520267
- AARS 3-shot total MAE = 2.955667640
- 1-shot top-5 recovery = 0.890000
- 3-shot top-5 recovery = 0.907500
- zero-shot total MAE = 23.898213826
- 1-shot global-bias MAE = 5.822441794
- 3-shot global-bias MAE = 5.767654694
- 1-shot raw-relative MAE = 4.020385849
- 3-shot raw-relative MAE = 4.823735099
- 3-shot vs aligned zero-shot relative improvement = 87.633731912%
- exact paired sign-flip p = 0.0078125

All packaged reference checks passed.

## Strict model-refit smoke test

A portable strict rerun was executed for **Ardahan** using the recovered frozen AARS method and packaged data/anchors.

The regenerated 1-shot and 3-shot metrics matched the recovered strict-v8 reference rows **bit-for-bit** for:

- MAE,
- RMSE,
- R²,
- Spearman correlation,
- Kendall correlation,
- design regret,
- top-5% recovery.

This smoke test validates the portable path cleanup and the linkage between the recovered v5 method code and v8 strict evaluation logic.

## Figure scripts

All distributed data-driven figure scripts execute successfully against the packaged reference data and write PDF + PNG outputs under `figures/generated/`.

The exact smoke-test outputs used for this check are retained under `validation/smoke_ardahan/`.
