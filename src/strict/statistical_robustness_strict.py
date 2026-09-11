from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_COLS = ["mae", "rmse", "r2", "spearman", "kendall", "design_regret_kwh_m2", "top_recovery"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Climate-level strict AARS statistical analysis.")
    p.add_argument("--all-methods", type=Path, default=REPO_ROOT / "outputs/strict_baselines/external_all_methods_by_city_strict.csv")
    p.add_argument("--aars-metrics", type=Path, default=REPO_ROOT / "outputs/strict_aars/external_metrics_by_city_strict.csv")
    p.add_argument("--aars-predictions", type=Path, default=REPO_ROOT / "outputs/strict_aars/external_predictions_strict.csv")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/strict_statistics")
    p.add_argument("--bootstrap-resamples", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20260820)
    p.add_argument("--paired-bootstrap-seed", type=int, default=20260821)
    return p.parse_args()


def metric(y_true, y_pred, top_frac=0.05):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    k = max(1, int(np.ceil(top_frac * len(y_true))))
    true_top = set(np.argsort(y_true)[:k])
    pred_top = set(np.argsort(y_pred)[:k])
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "kendall": float(kendalltau(y_true, y_pred).statistic),
        "design_regret_kwh_m2": float(y_true[int(np.argmin(y_pred))] - y_true.min()),
        "top_recovery": float(len(true_top & pred_top) / k),
    }


def signflip(a, b, lower_better=True):
    diff = (b - a) if lower_better else (a - b)
    observed = abs(diff.mean())
    values = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(diff)):
        values.append(abs(np.mean(diff * np.asarray(signs))))
    values = np.asarray(values)
    p = float(np.mean(values >= observed - 1e-15))
    return diff, p


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.read_csv(args.all_methods)
    proposed = pd.read_csv(args.aars_metrics)
    predictions = pd.read_csv(args.aars_predictions)

    # Common 997-design universe for direct AARS 3-shot vs 1-shot comparison.
    common_rows = []
    for city in sorted(predictions["target_city"].unique()):
        p1 = predictions[(predictions["target_city"] == city) & (predictions["anchor_budget"] == 1)].copy().sort_values("base_design_id")
        p3 = predictions[(predictions["target_city"] == city) & (predictions["anchor_budget"] == 3)].copy().sort_values("base_design_id")
        cal3 = set(p3.loc[p3["is_calibration_anchor"], "base_design_id"].astype(str))
        p1 = p1[~p1["base_design_id"].astype(str).isin(cal3)].copy()
        p3 = p3[~p3["base_design_id"].astype(str).isin(cal3)].copy()
        assert len(p1) == 997 and len(p3) == 997
        for budget, data in ((1, p1), (3, p3)):
            for target in ("heating", "cooling", "total"):
                mm = metric(data[f"true_{target}"], data[f"pred_{target}"])
                common_rows.append(
                    {
                        "protocol": f"AARS_v4_{budget}_shot_common997",
                        "target_city": city,
                        "target": target,
                        "n_eval": 997,
                        **mm,
                    }
                )
    common = pd.DataFrame(common_rows)
    common.to_csv(args.output_dir / "aars_1shot_vs_3shot_common997_by_city.csv", index=False)

    rng = np.random.default_rng(args.bootstrap_seed)
    B = args.bootstrap_resamples
    bootstrap_rows = []
    for budget in (1, 3):
        for target in ("heating", "cooling", "total"):
            d = proposed[(proposed["anchor_budget"] == budget) & (proposed["target"] == target)].sort_values("target_city")
            for met in METRIC_COLS:
                vals = d[met].to_numpy(float)
                draws = vals[rng.integers(0, len(vals), size=(B, len(vals)))].mean(axis=1)
                bootstrap_rows.append(
                    {
                        "protocol": f"AARS_v4_{budget}_shot",
                        "target": target,
                        "metric": met,
                        "mean": float(vals.mean()),
                        "ci_low": float(np.quantile(draws, 0.025)),
                        "ci_high": float(np.quantile(draws, 0.975)),
                        "n_climates": len(vals),
                        "bootstrap_resamples": B,
                        "seed": args.bootstrap_seed,
                    }
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output_dir / "bootstrap_climate_ci_strict.csv", index=False)

    comparisons = [
        ("AARS_v4_3_shot", "zero_shot_absolute_aligned_3shot", 3),
        ("AARS_v4_3_shot", "3_shot_global_bias", 3),
        ("AARS_v4_3_shot", "3_shot_raw_relative", 3),
        ("AARS_v4_1_shot", "zero_shot_absolute_aligned_1shot", 1),
        ("AARS_v4_1_shot", "1_shot_global_bias", 1),
        ("AARS_v4_1_shot", "1_shot_raw_relative", 1),
    ]
    pair_rows = []
    rng2 = np.random.default_rng(args.paired_bootstrap_seed)
    for method_a, method_b, budget in comparisons:
        da = combined[(combined["protocol"] == method_a) & (combined["target"] == "total")].sort_values("target_city")
        db = combined[(combined["protocol"] == method_b) & (combined["target"] == "total")].sort_values("target_city")
        assert da["target_city"].tolist() == db["target_city"].tolist()
        av = da["mae"].to_numpy(float)
        bv = db["mae"].to_numpy(float)
        diff, p = signflip(av, bv, True)
        bootdiff = diff[rng2.integers(0, len(diff), size=(B, len(diff)))].mean(axis=1)
        pair_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "comparison_budget": budget,
                "metric": "total_mae",
                "mean_a": float(av.mean()),
                "mean_b": float(bv.mean()),
                "absolute_improvement_b_minus_a": float(diff.mean()),
                "relative_improvement_pct": float(100 * (bv.mean() - av.mean()) / bv.mean()),
                "ci_low_improvement": float(np.quantile(bootdiff, 0.025)),
                "ci_high_improvement": float(np.quantile(bootdiff, 0.975)),
                "exact_signflip_p_two_sided": p,
                "a_better_climates": int((diff > 0).sum()),
                "b_better_climates": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "n_climates": 8,
            }
        )

    d1 = common[(common["protocol"] == "AARS_v4_1_shot_common997") & (common["target"] == "total")].sort_values("target_city")
    d3 = common[(common["protocol"] == "AARS_v4_3_shot_common997") & (common["target"] == "total")].sort_values("target_city")
    av = d3["mae"].to_numpy(float)
    bv = d1["mae"].to_numpy(float)
    diff, p = signflip(av, bv, True)
    bootdiff = diff[rng2.integers(0, 8, size=(B, 8))].mean(axis=1)
    pair_rows.append(
        {
            "method_a": "AARS_v4_3_shot_common997",
            "method_b": "AARS_v4_1_shot_common997",
            "comparison_budget": 3,
            "metric": "total_mae",
            "mean_a": float(av.mean()),
            "mean_b": float(bv.mean()),
            "absolute_improvement_b_minus_a": float(diff.mean()),
            "relative_improvement_pct": float(100 * (bv.mean() - av.mean()) / bv.mean()),
            "ci_low_improvement": float(np.quantile(bootdiff, 0.025)),
            "ci_high_improvement": float(np.quantile(bootdiff, 0.975)),
            "exact_signflip_p_two_sided": p,
            "a_better_climates": int((diff > 0).sum()),
            "b_better_climates": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "n_climates": 8,
        }
    )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(args.output_dir / "paired_external_comparisons_strict.csv", index=False)

    worst_rows = []
    for budget in (1, 3):
        for target in ("heating", "cooling", "total"):
            d = proposed[(proposed["anchor_budget"] == budget) & (proposed["target"] == target)].copy()
            r = d.loc[d["r2"].idxmin()]
            e = d.loc[d["mae"].idxmax()]
            worst_rows.append(
                {
                    "anchor_budget": budget,
                    "target": target,
                    "worst_r2_city": r["target_city"],
                    "worst_r2": r["r2"],
                    "mae_at_worst_r2": r["mae"],
                    "highest_mae_city": e["target_city"],
                    "highest_mae": e["mae"],
                    "r2_at_highest_mae": e["r2"],
                }
            )
    pd.DataFrame(worst_rows).to_csv(args.output_dir / "worst_climate_summary_strict.csv", index=False)

    grouped = combined.groupby(["protocol", "target", "comparison_budget"], as_index=False)[METRIC_COLS].mean()
    z = grouped[(grouped["protocol"] == "zero_shot_absolute") & (grouped["target"] == "total")].iloc[0]
    a1 = grouped[(grouped["protocol"] == "AARS_v4_1_shot") & (grouped["target"] == "total")].iloc[0]
    a3 = grouped[(grouped["protocol"] == "AARS_v4_3_shot") & (grouped["target"] == "total")].iloc[0]
    budget = pd.DataFrame(
        [
            {"target_simulations": 0, "protocol": "zero_shot_absolute", "evaluation_n_per_climate": 1000, **{m: float(z[m]) for m in METRIC_COLS}},
            {"target_simulations": 1, "protocol": "AARS_v4_1_shot_strict", "evaluation_n_per_climate": 999, **{m: float(a1[m]) for m in METRIC_COLS}},
            {"target_simulations": 3, "protocol": "AARS_v4_3_shot_strict", "evaluation_n_per_climate": 997, **{m: float(a3[m]) for m in METRIC_COLS}},
        ]
    )
    budget.to_csv(args.output_dir / "accuracy_budget_tradeoff_strict.csv", index=False)

    metadata = {
        "bootstrap_resamples": B,
        "bootstrap_seed": args.bootstrap_seed,
        "paired_bootstrap_seed": args.paired_bootstrap_seed,
        "independent_unit": "external climate",
        "exact_signflip_patterns": 256,
        "strict_holdout": True,
        "one_shot_n_eval": 999,
        "three_shot_n_eval": 997,
        "aars1_vs_aars3_common_holdout_n": 997,
    }
    (args.output_dir / "statistical_metadata_strict.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(budget.to_string(index=False))
    print()
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
