from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "aars"))

from evaluate_frozen_v4_external import (  # noqa: E402
    heating_physics_residual_fit_predict,
    metric,
    model_feature_list,
    raw_relative_fit_predict,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reproduce the strict calibration-excluded AARS external evaluation. "
            "This is a portable version of the recovered v8 strict runner."
        )
    )
    p.add_argument(
        "--source-physics",
        type=Path,
        default=REPO_ROOT / "data/processed/energy_results_classguard_v2_6000_physics.csv",
    )
    p.add_argument(
        "--external-physics",
        type=Path,
        default=REPO_ROOT / "data/processed/energy_results_external_8000_physics.csv",
    )
    p.add_argument(
        "--selected-anchor-bank",
        type=Path,
        default=REPO_ROOT / "data/anchors/external_selected_anchor_bank.csv",
    )
    p.add_argument(
        "--source-only-anchor-scores",
        type=Path,
        default=REPO_ROOT / "data/anchors/external_source_only_anchor_scores.csv",
    )
    p.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs/strict_aars"
    )
    p.add_argument("--outer-max-iter", type=int, default=220)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional subset for a smoke check. Seed indexing still follows the full sorted target-city list.",
    )
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

    if set(source["city"]) & set(external["city"]):
        raise ValueError("Source and external city sets are not disjoint.")

    features, numeric, categorical = model_feature_list(source)
    missing = sorted(set(features) - set(external.columns))
    if missing:
        raise ValueError(f"External dataset missing model features: {missing}")

    non_hvac = float(
        (
            source["total_site_energy_kwh_m2"]
            - source["heating_kwh_m2"]
            - source["cooling_kwh_m2"]
        ).mean()
    )

    all_cities = sorted(external["city"].unique().tolist())
    requested = set(args.cities) if args.cities else set(all_cities)
    unknown = requested - set(all_cities)
    if unknown:
        raise ValueError(f"Unknown target cities: {sorted(unknown)}")

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    verification_rows: list[dict] = []

    for target_i, city in enumerate(all_cities):
        if city not in requested:
            continue

        g = external.loc[external["city"].eq(city)].copy().reset_index(drop=True)
        if g["base_design_id"].nunique() != 1000:
            raise ValueError(f"{city}: expected 1,000 base designs.")

        top3 = (
            bank.loc[
                bank["target_city"].eq(city) & bank["anchor_budget"].eq(3)
            ]
            .sort_values("rank")["anchor_id"]
            .astype(str)
            .tolist()
        )
        if len(top3) != 3:
            raise ValueError(f"{city}: expected exactly three ranked anchors, found {top3}")

        city_scores = scores.loc[scores["target_city"].eq(city)].set_index("anchor_id")

        predictions: dict[str, dict[str, np.ndarray]] = {}
        for anchor_id in top3:
            pred_h = heating_physics_residual_fit_predict(
                source,
                g,
                anchor_id,
                features,
                numeric,
                categorical,
                args.outer_max_iter,
                args.seed + target_i,
            )
            pred_c = raw_relative_fit_predict(
                source,
                g,
                anchor_id,
                "cooling_kwh_m2",
                features,
                numeric,
                categorical,
                args.outer_max_iter,
                args.seed + target_i,
            )
            predictions[anchor_id] = {"heating": pred_h, "cooling": pred_c}

        for budget in (1, 3):
            ids = top3[:budget]

            wh = np.asarray(
                [
                    1.0 / max(float(city_scores.loc[a, "heating_score"]), 1e-9)
                    for a in ids
                ],
                dtype=float,
            )
            wc = np.asarray(
                [
                    1.0 / max(float(city_scores.loc[a, "cooling_score"]), 1e-9)
                    for a in ids
                ],
                dtype=float,
            )
            wh /= wh.sum()
            wc /= wc.sum()

            pred_h = sum(w * predictions[a]["heating"] for w, a in zip(wh, ids))
            pred_c = sum(w * predictions[a]["cooling"] for w, a in zip(wc, ids))
            pred_t = pred_h + pred_c + non_hvac

            is_calibration = g["base_design_id"].astype(str).isin(ids).to_numpy()
            mask = ~is_calibration
            expected_n = 1000 - budget
            if int(mask.sum()) != expected_n:
                raise AssertionError(
                    f"{city}, {budget}-shot: expected {expected_n} evaluation designs, "
                    f"found {int(mask.sum())}."
                )

            output_map = {
                "heating": (g["heating_kwh_m2"].to_numpy(float), pred_h),
                "cooling": (g["cooling_kwh_m2"].to_numpy(float), pred_c),
                "total": (g["total_site_energy_kwh_m2"].to_numpy(float), pred_t),
            }

            for target, (truth, pred) in output_map.items():
                # All-row metric is retained only as an audit trace; publication truth is strict_holdout.
                verification_rows.append(
                    {
                        "target_city": city,
                        "anchor_budget": budget,
                        "target": target,
                        "universe": "all1000_audit_only",
                        "n_calibration": budget,
                        "n_eval": 1000,
                        "calibration_anchor_ids": "|".join(ids),
                        **metric(truth, pred, args.top_frac),
                        "top_k": max(1, int(np.ceil(args.top_frac * len(truth)))),
                    }
                )
                strict_metric = metric(truth[mask], pred[mask], args.top_frac)
                metric_rows.append(
                    {
                        "target_city": city,
                        "anchor_budget": budget,
                        "target": target,
                        "universe": "strict_holdout",
                        "n_calibration": budget,
                        "n_eval": int(mask.sum()),
                        "calibration_anchor_ids": "|".join(ids),
                        **strict_metric,
                        "top_k": max(1, int(np.ceil(args.top_frac * int(mask.sum())))),
                    }
                )

            for i, row in g.iterrows():
                prediction_rows.append(
                    {
                        "target_city": city,
                        "anchor_budget": budget,
                        "base_design_id": str(row["base_design_id"]),
                        "is_calibration_anchor": bool(is_calibration[i]),
                        "true_heating": float(row["heating_kwh_m2"]),
                        "pred_heating": float(pred_h[i]),
                        "true_cooling": float(row["cooling_kwh_m2"]),
                        "pred_cooling": float(pred_c[i]),
                        "true_total": float(row["total_site_energy_kwh_m2"]),
                        "pred_total": float(pred_t[i]),
                    }
                )

        print(f"Done: {city} | ranked anchors={top3}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    verification_df = pd.DataFrame(verification_rows)

    metrics.to_csv(args.output_dir / "external_metrics_by_city_strict.csv", index=False)
    predictions_df.to_csv(args.output_dir / "external_predictions_strict.csv", index=False)
    verification_df.to_csv(args.output_dir / "external_metrics_all1000_audit.csv", index=False)

    metric_cols = [
        "mae",
        "rmse",
        "r2",
        "spearman",
        "kendall",
        "design_regret_kwh_m2",
        "top_recovery",
    ]
    summary = metrics.groupby(["anchor_budget", "target"], as_index=False)[metric_cols].mean()
    summary.to_csv(args.output_dir / "external_summary_strict.csv", index=False)

    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
