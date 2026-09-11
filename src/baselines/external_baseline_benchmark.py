from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DESIGN_FEATURES = [
    "room_width_m",
    "room_depth_m",
    "room_height_m",
    "floor_area_m2",
    "volume_m3",
    "actual_wwr",
    "window_area_m2",
    "actual_wall_u_no_film_w_m2k",
    "actual_roof_u_no_film_w_m2k",
    "actual_floor_u_no_film_w_m2k",
    "actual_glazing_u_w_m2k",
    "actual_glazing_shgc",
    "vlt",
    "envelope_area_to_volume_ratio",
    "ua_intensity_w_m2k",
    "solar_aperture_ratio",
]

PHYSICS_FEATURES = [
    "heating_transmission_proxy_kwh_m2",
    "cooling_transmission_proxy_kwh_m2",
    "orientation_solar_gain_proxy_kwh_m2",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Frozen external baselines for CLASS-Guard/AARS. "
            "Uses the same source/external data and source-only selected anchors."
        )
    )
    p.add_argument("--source-physics", type=Path, required=True)
    p.add_argument("--external-physics", type=Path, required=True)
    p.add_argument("--selected-anchor-bank", type=Path, required=True)
    p.add_argument("--source-only-anchor-scores", type=Path, required=True)
    p.add_argument("--proposed-metrics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-frac", type=float, default=0.05)
    return p.parse_args()


def feature_lists(df: pd.DataFrame):
    climate = [
        c for c in df.columns
        if c.startswith("climate_")
        and c not in {"climate_latitude", "climate_longitude"}
    ]
    ground = [
        c for c in df.columns
        if c.startswith("ground_temperature_month_")
    ]
    features = list(dict.fromkeys(
        [*DESIGN_FEATURES, *climate, *ground, *PHYSICS_FEATURES, "orientation"]
    ))
    categorical = ["orientation"]
    numeric = [c for c in features if c not in categorical]
    return features, numeric, categorical


def make_hgb(numeric, categorical, seed):
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
    reg = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("reg", reg)])


def metric(y_true, y_pred, top_frac):
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
        "design_regret_kwh_m2": float(
            y_true[int(np.argmin(y_pred))] - y_true.min()
        ),
        "top_recovery": float(len(true_top & pred_top) / k),
    }


def source_anchor_map(source, anchor_id, target):
    return (
        source.loc[source["base_design_id"].astype(str).eq(str(anchor_id))]
        .set_index("city")[target]
        .to_dict()
    )


def fit_raw_relative_model(
    source,
    anchor_id,
    target,
    features,
    numeric,
    categorical,
    seed,
):
    amap = source_anchor_map(source, anchor_id, target)
    y_delta = source[target] - source["city"].map(amap)
    model = make_hgb(numeric, categorical, seed)
    model.fit(source[features], y_delta)
    return model


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.source_physics)
    external = pd.read_csv(args.external_physics)
    bank = pd.read_csv(args.selected_anchor_bank)
    scores = pd.read_csv(args.source_only_anchor_scores)
    proposed = pd.read_csv(args.proposed_metrics).copy()

    source = source[
        source["status"].astype(str).str.lower().eq("ok")
        & pd.to_numeric(source["severe_count"], errors="coerce").fillna(0).eq(0)
    ].copy()
    external = external[
        external["status"].astype(str).str.lower().eq("ok")
        & pd.to_numeric(external["severe_count"], errors="coerce").fillna(0).eq(0)
    ].copy()

    features, numeric, categorical = feature_lists(source)
    missing = sorted(set(features) - set(external.columns))
    if missing:
        raise ValueError(f"External dataset missing features: {missing}")

    # Standard absolute zero-shot baselines.
    abs_models = {}
    abs_pred = {}
    for target in ["heating_kwh_m2", "cooling_kwh_m2"]:
        model = make_hgb(numeric, categorical, args.seed)
        model.fit(source[features], source[target])
        abs_models[target] = model
        abs_pred[target] = model.predict(external[features])

    external = external.copy()
    external["_zero_heating"] = np.maximum(abs_pred["heating_kwh_m2"], 0.0)
    external["_zero_cooling"] = np.maximum(abs_pred["cooling_kwh_m2"], 0.0)

    # The top-three bank is source-only selected in the frozen external experiment.
    anchor_ids = (
        bank.loc[bank["anchor_budget"].eq(3)]
        .sort_values(["target_city", "rank"])
        ["anchor_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    raw_relative_models = {}
    for anchor_id in anchor_ids:
        raw_relative_models[(anchor_id, "heating")] = fit_raw_relative_model(
            source, anchor_id, "heating_kwh_m2",
            features, numeric, categorical, args.seed,
        )
        raw_relative_models[(anchor_id, "cooling")] = fit_raw_relative_model(
            source, anchor_id, "cooling_kwh_m2",
            features, numeric, categorical, args.seed,
        )

    non_hvac = float(
        (
            source["total_site_energy_kwh_m2"]
            - source["heating_kwh_m2"]
            - source["cooling_kwh_m2"]
        ).mean()
    )

    rows = []

    for city, g0 in external.groupby("city", sort=True):
        g = g0.copy().reset_index(drop=True)

        city_bank = (
            bank.loc[
                bank["target_city"].eq(city)
                & bank["anchor_budget"].eq(3)
            ]
            .sort_values("rank")
        )
        top3 = city_bank["anchor_id"].astype(str).tolist()
        top1 = top3[:1]

        city_scores = scores.loc[scores["target_city"].eq(city)].set_index("anchor_id")

        protocols = {}

        # 0-shot absolute.
        protocols["zero_shot_absolute"] = {
            "heating": g["_zero_heating"].to_numpy(),
            "cooling": g["_zero_cooling"].to_numpy(),
        }

        for budget, ids in [(1, top1), (3, top3)]:
            # Additive bias correction of absolute predictor.
            bias_pred = {}
            for key, target_col, zero_col in [
                ("heating", "heating_kwh_m2", "_zero_heating"),
                ("cooling", "cooling_kwh_m2", "_zero_cooling"),
            ]:
                corrections = []
                for aid in ids:
                    row = g.loc[g["base_design_id"].astype(str).eq(aid)].iloc[0]
                    corrections.append(float(row[target_col] - row[zero_col]))
                bias_pred[key] = g[zero_col].to_numpy() + np.mean(corrections)
                bias_pred[key] = np.maximum(bias_pred[key], 0.0)
            protocols[f"{budget}_shot_global_bias"] = bias_pred

            # Raw anchor-relative HGB ablation (no heating physics residual).
            rel_pred = {}
            for key, target_col, score_col in [
                ("heating", "heating_kwh_m2", "heating_score"),
                ("cooling", "cooling_kwh_m2", "cooling_score"),
            ]:
                preds = []
                ws = []
                for aid in ids:
                    anchor_truth = float(
                        g.loc[g["base_design_id"].astype(str).eq(aid), target_col].iloc[0]
                    )
                    delta = raw_relative_models[(aid, key)].predict(g[features])
                    preds.append(np.maximum(anchor_truth + delta, 0.0))
                    ws.append(1.0 / max(float(city_scores.loc[aid, score_col]), 1e-9))
                ws = np.asarray(ws, float)
                ws /= ws.sum()
                rel_pred[key] = sum(w * p for w, p in zip(ws, preds))
            protocols[f"{budget}_shot_raw_relative"] = rel_pred

        for protocol, pred in protocols.items():
            total_pred = pred["heating"] + pred["cooling"] + non_hvac
            outputs = {
                "heating": (g["heating_kwh_m2"], pred["heating"]),
                "cooling": (g["cooling_kwh_m2"], pred["cooling"]),
                "total": (g["total_site_energy_kwh_m2"], total_pred),
            }
            for target, (truth, prediction) in outputs.items():
                rows.append(
                    {
                        "protocol": protocol,
                        "target_city": city,
                        "target": target,
                        **metric(truth, prediction, args.top_frac),
                    }
                )

    baseline = pd.DataFrame(rows)
    baseline.to_csv(args.output_dir / "external_baseline_metrics_by_city.csv", index=False)

    baseline_summary = (
        baseline.groupby(["protocol", "target"], as_index=False)[
            [
                "mae", "rmse", "r2", "spearman", "kendall",
                "design_regret_kwh_m2", "top_recovery",
            ]
        ].mean()
    )
    baseline_summary.to_csv(args.output_dir / "external_baseline_summary.csv", index=False)

    # Add the frozen proposed method for direct comparison.
    proposed = proposed.rename(columns={"anchor_budget": "budget"})
    proposed["protocol"] = proposed["budget"].map(
        {1: "AARS_v4_1_shot", 3: "AARS_v4_3_shot"}
    )
    proposed_cmp = proposed[
        [
            "protocol", "target_city", "target",
            "mae", "rmse", "r2", "spearman", "kendall",
            "design_regret_kwh_m2", "top_recovery",
        ]
    ].copy()

    combined = pd.concat([baseline, proposed_cmp], ignore_index=True)
    combined.to_csv(args.output_dir / "external_all_methods_by_city.csv", index=False)

    combined_summary = (
        combined.groupby(["protocol", "target"], as_index=False)[
            [
                "mae", "rmse", "r2", "spearman", "kendall",
                "design_regret_kwh_m2", "top_recovery",
            ]
        ].mean()
    )
    combined_summary.to_csv(args.output_dir / "external_all_methods_summary.csv", index=False)

    total_table = (
        combined_summary.loc[combined_summary["target"].eq("total")]
        .sort_values(["mae", "r2"], ascending=[True, False])
    )
    print(total_table.to_string(index=False))


if __name__ == "__main__":
    main()
