# Validation report

Validation was performed on the public repository package before the archival release.

## Static validation

- All Python files under `src/`, `tests/`, and `figures/scripts/` compile successfully with `py_compile`.
- No personal user-directory paths or credential-like values were found in the public source/configuration/metadata files checked during packaging.
- Third-party EPW weather files are not distributed.

## Unit tests

Command:

```bash
PYTHONPATH=src/simulation pytest tests -q
```

Result:

```text
3 passed
```

## Authoritative reference-table verification

Command:

```bash
python src/run_reproduction.py --quick-reference-check
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

A portable strict rerun was executed for **Ardahan** using the frozen AARS method and packaged data/anchor files:

```bash
python src/strict/evaluate_aars_strict.py --cities Ardahan --output-dir validation/smoke_ardahan
```

The regenerated 1-shot and 3-shot rows matched the packaged strict reference rows exactly for:

- MAE,
- RMSE,
- R²,
- Spearman correlation,
- Kendall correlation,
- design regret,
- top-5% recovery.

The smoke-test output tables are retained under `validation/smoke_ardahan/`.

## Figure scripts

All scripts under `figures/scripts/` execute against the packaged data and generate their PDF/PNG outputs under `figures/generated/`.
