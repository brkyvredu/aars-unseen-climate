from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data/results/strict"
GEN_AARS = ROOT / "outputs/strict_aars"
GEN_BASE = ROOT / "outputs/strict_baselines"
GEN_STATS = ROOT / "outputs/strict_statistics"

KEY_METRICS = ["mae", "rmse", "r2", "spearman", "kendall", "design_regret_kwh_m2", "top_recovery"]


def compare_csv(ref_path, gen_path, keys, metrics, tol=1e-10):
    r = pd.read_csv(ref_path)
    g = pd.read_csv(gen_path)
    m = r.merge(g, on=keys, suffixes=("_ref", "_gen"), how="outer", indicator=True)
    if not m["_merge"].eq("both").all():
        raise AssertionError(f"Key mismatch: {ref_path.name} vs {gen_path.name}")
    for c in metrics:
        diff = np.nanmax(np.abs(m[f"{c}_ref"].to_numpy(float) - m[f"{c}_gen"].to_numpy(float)))
        if diff > tol:
            raise AssertionError(f"{gen_path.name}: {c} max abs diff {diff} > {tol}")
        print(f"PASS {gen_path.name} {c}: max abs diff={diff:.3g}")


def main():
    compare_csv(
        REF / "external_metrics_by_city_strict.csv",
        GEN_AARS / "external_metrics_by_city_strict.csv",
        ["target_city", "anchor_budget", "target"],
        KEY_METRICS,
    )
    compare_csv(
        REF / "external_all_methods_by_city_strict.csv",
        GEN_BASE / "external_all_methods_by_city_strict.csv",
        ["protocol", "target_city", "target", "comparison_budget"],
        KEY_METRICS,
    )
    compare_csv(
        REF / "paired_external_comparisons_strict.csv",
        GEN_STATS / "paired_external_comparisons_strict.csv",
        ["method_a", "method_b", "comparison_budget", "metric"],
        ["mean_a", "mean_b", "relative_improvement_pct", "exact_signflip_p_two_sided"],
        tol=1e-9,
    )
    print("\nFull reproduced outputs match the packaged strict reference results.")


if __name__ == "__main__":
    main()
