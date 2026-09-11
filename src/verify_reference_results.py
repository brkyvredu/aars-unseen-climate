from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results/strict"


def close(name, value, expected, tol=1e-6):
    if not np.isclose(value, expected, atol=tol, rtol=0):
        raise AssertionError(f"{name}: {value} != {expected} (tol={tol})")
    print(f"PASS {name}: {value:.9f}")


def main():
    aars = pd.read_csv(RESULTS / "external_metrics_by_city_strict.csv")
    allm = pd.read_csv(RESULTS / "external_all_methods_by_city_strict.csv")
    pairs = pd.read_csv(RESULTS / "paired_external_comparisons_strict.csv")

    assert len(aars) == 48
    assert aars.query("anchor_budget == 1")["n_eval"].eq(999).all()
    assert aars.query("anchor_budget == 3")["n_eval"].eq(997).all()

    total1 = aars.query("anchor_budget == 1 and target == 'total'")
    total3 = aars.query("anchor_budget == 3 and target == 'total'")
    close("AARS 1-shot mean MAE", total1["mae"].mean(), 3.330520, 1e-6)
    close("AARS 3-shot mean MAE", total3["mae"].mean(), 2.955668, 1e-6)
    close("AARS 1-shot top-5 recovery", total1["top_recovery"].mean(), 0.890000, 1e-12)
    close("AARS 3-shot top-5 recovery", total3["top_recovery"].mean(), 0.907500, 1e-12)

    total = allm[allm["target"].eq("total")]
    expected = {
        "zero_shot_absolute": 23.898214,
        "1_shot_global_bias": 5.822442,
        "3_shot_global_bias": 5.767655,
        "1_shot_raw_relative": 4.020386,
        "3_shot_raw_relative": 4.823735,
        "AARS_v4_1_shot": 3.330520,
        "AARS_v4_3_shot": 2.955668,
    }
    for protocol, exp in expected.items():
        close(protocol, total.loc[total["protocol"].eq(protocol), "mae"].mean(), exp, 1e-6)

    p3 = pairs[
        (pairs["method_a"] == "AARS_v4_3_shot")
        & (pairs["method_b"] == "zero_shot_absolute_aligned_3shot")
    ].iloc[0]
    close("3-shot vs aligned zero-shot relative improvement (%)", p3["relative_improvement_pct"], 87.63373191159802, 1e-9)
    close("3-shot vs aligned zero-shot sign-flip p", p3["exact_signflip_p_two_sided"], 0.0078125, 1e-12)

    print("\nAll packaged reference checks passed.")


if __name__ == "__main__":
    main()
