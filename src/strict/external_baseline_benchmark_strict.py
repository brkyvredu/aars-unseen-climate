from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "baselines"))
import external_baseline_benchmark as b  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce strict external baseline comparisons.")
    p.add_argument("--source-physics", type=Path, default=REPO_ROOT / "data/processed/energy_results_classguard_v2_6000_physics.csv")
    p.add_argument("--external-physics", type=Path, default=REPO_ROOT / "data/processed/energy_results_external_8000_physics.csv")
    p.add_argument("--selected-anchor-bank", type=Path, default=REPO_ROOT / "data/anchors/external_selected_anchor_bank.csv")
    p.add_argument("--source-only-anchor-scores", type=Path, default=REPO_ROOT / "data/anchors/external_source_only_anchor_scores.csv")
    p.add_argument("--proposed-strict-metrics", type=Path, default=REPO_ROOT / "outputs/strict_aars/external_metrics_by_city_strict.csv")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/strict_baselines")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-frac", type=float, default=0.05)
    return p.parse_args()


def filter_success(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["status"].astype(str).str.lower().eq("ok")
        & pd.to_numeric(df["severe_count"], errors="coerce").fillna(0).eq(0)
    ].copy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = filter_success(pd.read_csv(args.source_physics))
    external = filter_success(pd.read_csv(args.external_physics))
    bank = pd.read_csv(args.selected_anchor_bank)
    scores = pd.read_csv(args.source_only_anchor_scores)
    bank["anchor_id"] = bank["anchor_id"].astype(str)
    scores["anchor_id"] = scores["anchor_id"].astype(str)

    features, numeric, categorical = b.feature_lists(source)
    missing = sorted(set(features) - set(external.columns))
    if missing:
        raise ValueError(f"External dataset missing features: {missing}")

    for target, key in (("heating_kwh_m2", "heating"), ("cooling_kwh_m2", "cooling")):
        model = b.make_hgb(numeric, categorical, args.seed)
        model.fit(source[features], source[target])
        external[f"_zero_{key}"] = np.maximum(model.predict(external[features]), 0.0)

    anchor_ids = (
        bank.loc[bank["anchor_budget"].eq(3)]
        .sort_values(["target_city", "rank"])["anchor_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    raw_models = {}
    for aid in anchor_ids:
        raw_models[(aid, "heating")] = b.fit_raw_relative_model(
            source, aid, "heating_kwh_m2", features, numeric, categorical, args.seed
        )
        raw_models[(aid, "cooling")] = b.fit_raw_relative_model(
            source, aid, "cooling_kwh_m2", features, numeric, categorical, args.seed
        )

    non_hvac = float(
        (
            source["total_site_energy_kwh_m2"]
            - source["heating_kwh_m2"]
            - source["cooling_kwh_m2"]
        ).mean()
    )

    rows, prediction_rows = [], []
    for city, g0 in external.groupby("city", sort=True):
        g = g0.copy().reset_index(drop=True)
        top3 = (
            bank.loc[bank["target_city"].eq(city) & bank["anchor_budget"].eq(3)]
            .sort_values("rank")["anchor_id"]
            .astype(str)
            .tolist()
        )
        top1 = top3[:1]
        city_scores = scores.loc[scores["target_city"].eq(city)].set_index("anchor_id")
        zero = {
            "heating": g["_zero_heating"].to_numpy(),
            "cooling": g["_zero_cooling"].to_numpy(),
        }

        # Standalone operational zero-shot baseline on all 1,000 designs.
        for target, truth, pred in (
            ("heating", g["heating_kwh_m2"].to_numpy(), zero["heating"]),
            ("cooling", g["cooling_kwh_m2"].to_numpy(), zero["cooling"]),
            ("total", g["total_site_energy_kwh_m2"].to_numpy(), zero["heating"] + zero["cooling"] + non_hvac),
        ):
            rows.append(
                {
                    "protocol": "zero_shot_absolute",
                    "target_city": city,
                    "target": target,
                    "comparison_budget": 0,
                    "n_calibration": 0,
                    "n_eval": 1000,
                    "calibration_anchor_ids": "",
                    **b.metric(truth, pred, args.top_frac),
                }
            )

        for budget, ids in ((1, top1), (3, top3)):
            is_calibration = g["base_design_id"].astype(str).isin(ids).to_numpy()
            mask = ~is_calibration

            # Fair zero-shot rows on the exact same non-calibration universe.
            for target, truth, pred in (
                ("heating", g["heating_kwh_m2"].to_numpy(), zero["heating"]),
                ("cooling", g["cooling_kwh_m2"].to_numpy(), zero["cooling"]),
                ("total", g["total_site_energy_kwh_m2"].to_numpy(), zero["heating"] + zero["cooling"] + non_hvac),
            ):
                rows.append(
                    {
                        "protocol": f"zero_shot_absolute_aligned_{budget}shot",
                        "target_city": city,
                        "target": target,
                        "comparison_budget": budget,
                        "n_calibration": 0,
                        "n_eval": int(mask.sum()),
                        "calibration_anchor_ids": "|".join(ids),
                        **b.metric(truth[mask], pred[mask], args.top_frac),
                    }
                )

            bias = {}
            for key, target_col, zero_col in (
                ("heating", "heating_kwh_m2", "_zero_heating"),
                ("cooling", "cooling_kwh_m2", "_zero_cooling"),
            ):
                corrections = []
                for aid in ids:
                    row = g.loc[g["base_design_id"].astype(str).eq(aid)].iloc[0]
                    corrections.append(float(row[target_col] - row[zero_col]))
                bias[key] = np.maximum(g[zero_col].to_numpy() + np.mean(corrections), 0.0)

            relative = {}
            for key, target_col, score_col in (
                ("heating", "heating_kwh_m2", "heating_score"),
                ("cooling", "cooling_kwh_m2", "cooling_score"),
            ):
                preds, weights = [], []
                for aid in ids:
                    anchor_truth = float(
                        g.loc[g["base_design_id"].astype(str).eq(aid), target_col].iloc[0]
                    )
                    delta = raw_models[(aid, key)].predict(g[features])
                    preds.append(np.maximum(anchor_truth + delta, 0.0))
                    weights.append(1.0 / max(float(city_scores.loc[aid, score_col]), 1e-9))
                weights = np.asarray(weights, dtype=float)
                weights /= weights.sum()
                relative[key] = sum(w * pred for w, pred in zip(weights, preds))

            for protocol, pred in (
                (f"{budget}_shot_global_bias", bias),
                (f"{budget}_shot_raw_relative", relative),
            ):
                pred_total = pred["heating"] + pred["cooling"] + non_hvac
                for target, truth, pred_value in (
                    ("heating", g["heating_kwh_m2"].to_numpy(), pred["heating"]),
                    ("cooling", g["cooling_kwh_m2"].to_numpy(), pred["cooling"]),
                    ("total", g["total_site_energy_kwh_m2"].to_numpy(), pred_total),
                ):
                    rows.append(
                        {
                            "protocol": protocol,
                            "target_city": city,
                            "target": target,
                            "comparison_budget": budget,
                            "n_calibration": budget,
                            "n_eval": int(mask.sum()),
                            "calibration_anchor_ids": "|".join(ids),
                            **b.metric(truth[mask], pred_value[mask], args.top_frac),
                        }
                    )

            for i, row in g.iterrows():
                prediction_rows.append(
                    {
                        "target_city": city,
                        "budget": budget,
                        "base_design_id": str(row["base_design_id"]),
                        "is_calibration_anchor": bool(is_calibration[i]),
                        "zero_heating": float(zero["heating"][i]),
                        "zero_cooling": float(zero["cooling"][i]),
                        "bias_heating": float(bias["heating"][i]),
                        "bias_cooling": float(bias["cooling"][i]),
                        "raw_heating": float(relative["heating"][i]),
                        "raw_cooling": float(relative["cooling"][i]),
                    }
                )
        print(f"Done: {city}", flush=True)

    base = pd.DataFrame(rows)
    base.to_csv(args.output_dir / "external_baseline_metrics_by_city_strict.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "external_baseline_predictions_strict.csv", index=False)

    metric_cols = ["mae", "rmse", "r2", "spearman", "kendall", "design_regret_kwh_m2", "top_recovery"]
    summary = base.groupby(["protocol", "target", "comparison_budget"], as_index=False)[metric_cols].mean()
    summary.to_csv(args.output_dir / "external_baseline_summary_strict.csv", index=False)

    proposed = pd.read_csv(args.proposed_strict_metrics).copy()
    proposed["protocol"] = proposed["anchor_budget"].map({1: "AARS_v4_1_shot", 3: "AARS_v4_3_shot"})
    proposed["comparison_budget"] = proposed["anchor_budget"]
    proposed = proposed[
        [
            "protocol",
            "target_city",
            "target",
            "comparison_budget",
            "n_calibration",
            "n_eval",
            "calibration_anchor_ids",
            *metric_cols,
        ]
    ]

    combined = pd.concat([base, proposed], ignore_index=True)
    combined.to_csv(args.output_dir / "external_all_methods_by_city_strict.csv", index=False)
    combined_summary = combined.groupby(["protocol", "target", "comparison_budget"], as_index=False)[metric_cols].mean()
    combined_summary.to_csv(args.output_dir / "external_all_methods_summary_strict.csv", index=False)

    metadata = {
        "strict_holdout": True,
        "calibration_rows_excluded": True,
        "fair_comparison_masks": True,
        "zero_shot_standalone_n": 1000,
        "one_shot_aligned_n": 999,
        "three_shot_aligned_n": 997,
    }
    (args.output_dir / "strict_baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print()
    print(combined_summary[combined_summary["target"].eq("total")].sort_values(["comparison_budget", "mae"]).to_string(index=False))


if __name__ == "__main__":
    main()
