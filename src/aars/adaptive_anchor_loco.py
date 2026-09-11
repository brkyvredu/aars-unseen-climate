from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = ["heating_kwh_m2", "cooling_kwh_m2"]

PHYSICS_DELTA_FEATURES = [
    "heating_transmission_proxy_kwh_m2",
    "cooling_transmission_proxy_kwh_m2",
    "orientation_solar_gain_proxy_kwh_m2",
]

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

ANCHOR_SELECTION_FEATURES = [
    "room_width_m",
    "room_depth_m",
    "room_height_m",
    "actual_wwr",
    "actual_wall_u_no_film_w_m2k",
    "actual_roof_u_no_film_w_m2k",
    "actual_floor_u_no_film_w_m2k",
    "actual_glazing_u_w_m2k",
    "actual_glazing_shgc",
    "vlt",
    "envelope_area_to_volume_ratio",
    "ua_intensity_w_m2k",
    "solar_aperture_ratio",
    "orientation_cos",
    "orientation_sin",
]

CLIMATE_DISTANCE_FEATURES = [
    "climate_annual_mean_temp_c",
    "climate_heating_degree_days_18c",
    "climate_cooling_degree_days_22c",
    "climate_annual_ghi_kwh_m2",
    "climate_mean_relative_humidity_pct",
    "climate_mean_wind_speed_m_s",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Adaptive Anchor Relative Surrogate (AARS) with source-only "
            "nested anchor selection."
        )
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidate-pool-size", type=int, default=10)
    p.add_argument("--anchor-budgets", type=int, nargs="+", default=[1, 3])
    p.add_argument("--outer-max-iter", type=int, default=220)
    p.add_argument("--inner-max-iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-cities", nargs="*", default=None)
    p.add_argument("--top-frac", type=float, default=0.05)
    return p.parse_args()


def add_orientation(df):
    out = df.copy()
    angles = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
    theta = np.deg2rad(out["orientation"].map(angles).astype(float))
    out["orientation_cos"] = np.cos(theta)
    out["orientation_sin"] = np.sin(theta)
    return out


def validate(df):
    required = {
        "base_design_id",
        "energy_design_id",
        "city",
        "orientation",
        "total_site_energy_kwh_m2",
        *TARGETS,
        *DESIGN_FEATURES,
        *PHYSICS_DELTA_FEATURES,
        *CLIMATE_DISTANCE_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n_cities = df["city"].nunique()
    coverage = df.groupby("base_design_id")["city"].nunique()
    if not (coverage == n_cities).all():
        raise ValueError("Dataset is not fully paired across climates.")


def model_feature_list(df):
    climate = [
        c
        for c in df.columns
        if c.startswith("climate_")
        and c not in {"climate_latitude", "climate_longitude"}
    ]
    ground = [
        c for c in df.columns if c.startswith("ground_temperature_month_")
    ]
    features = list(
        dict.fromkeys(
            [
                *DESIGN_FEATURES,
                *climate,
                *ground,
                *PHYSICS_DELTA_FEATURES,
                "orientation",
            ]
        )
    )
    categorical = ["orientation"]
    numeric = [c for c in features if c not in categorical]
    return features, numeric, categorical


def make_hgb(numeric, categorical, max_iter, seed):
    pre = ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric,
            ),
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
        max_iter=max_iter,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("reg", reg)])


def maximin_anchor_pool(df, pool_size):
    first_city = sorted(df["city"].unique())[0]
    base = (
        df.loc[df["city"].eq(first_city)]
        .sort_values("base_design_id")
        .drop_duplicates("base_design_id")
        .reset_index(drop=True)
    )

    x = StandardScaler().fit_transform(
        base[ANCHOR_SELECTION_FEATURES].to_numpy(float)
    )
    selected = [int(np.argmin(np.square(x).sum(axis=1)))]
    dmin = np.square(x - x[selected[0]]).sum(axis=1)
    dmin[selected] = -np.inf

    while len(selected) < pool_size:
        idx = int(np.argmax(dmin))
        selected.append(idx)
        dnew = np.square(x - x[idx]).sum(axis=1)
        dmin = np.minimum(dmin, dnew)
        dmin[selected] = -np.inf

    return base.iloc[selected]["base_design_id"].astype(str).tolist()


def climate_weights(df, source_cities, target_city):
    city_climate = (
        df.groupby("city")[CLIMATE_DISTANCE_FEATURES]
        .first()
        .loc[source_cities + [target_city]]
    )
    scaler = StandardScaler().fit(city_climate)
    z = pd.DataFrame(
        scaler.transform(city_climate),
        index=city_climate.index,
    )
    target = z.loc[target_city].to_numpy()
    distance = np.asarray(
        [
            np.linalg.norm(z.loc[c].to_numpy() - target)
            for c in source_cities
        ],
        dtype=float,
    )
    tau = max(float(np.median(distance)), 1e-6)
    weight = np.exp(-distance / tau)
    weight = weight / weight.sum()
    return dict(zip(source_cities, weight))


def anchor_physics_delta(df, anchor_id):
    anchor = (
        df.loc[df["base_design_id"].astype(str).eq(str(anchor_id))]
        .set_index("city")[PHYSICS_DELTA_FEATURES]
    )
    matrix = np.column_stack(
        [
            df[c].to_numpy(float)
            - df["city"].map(anchor[c].to_dict()).to_numpy(float)
            for c in PHYSICS_DELTA_FEATURES
        ]
    )
    return matrix


def raw_relative_fit_predict(
    train_df,
    test_df,
    anchor_id,
    target,
    features,
    numeric,
    categorical,
    max_iter,
    seed,
):
    train_anchor = (
        train_df.loc[
            train_df["base_design_id"].astype(str).eq(str(anchor_id))
        ]
        .set_index("city")[target]
        .to_dict()
    )
    target_anchor = float(
        test_df.loc[
            test_df["base_design_id"].astype(str).eq(str(anchor_id)),
            target,
        ].iloc[0]
    )

    train_target = (
        train_df[target] - train_df["city"].map(train_anchor)
    )
    model = make_hgb(numeric, categorical, max_iter, seed)
    model.fit(train_df[features], train_target)
    pred_delta = model.predict(test_df[features])
    return np.maximum(target_anchor + pred_delta, 0.0)


def heating_physics_residual_fit_predict(
    train_df,
    test_df,
    anchor_id,
    features,
    numeric,
    categorical,
    max_iter,
    seed,
):
    target = "heating_kwh_m2"

    train_anchor_y = (
        train_df.loc[
            train_df["base_design_id"].astype(str).eq(str(anchor_id))
        ]
        .set_index("city")[target]
        .to_dict()
    )
    target_anchor_y = float(
        test_df.loc[
            test_df["base_design_id"].astype(str).eq(str(anchor_id)),
            target,
        ].iloc[0]
    )

    x_delta_train = anchor_physics_delta(train_df, anchor_id)
    y_delta_train = (
        train_df[target].to_numpy(float)
        - train_df["city"].map(train_anchor_y).to_numpy(float)
    )

    physics = Ridge(alpha=1.0, fit_intercept=False)
    physics.fit(x_delta_train, y_delta_train)
    physics_train = physics.predict(x_delta_train)
    residual = y_delta_train - physics_train

    model = make_hgb(numeric, categorical, max_iter, seed)
    model.fit(train_df[features], residual)

    target_anchor_physics = (
        test_df.loc[
            test_df["base_design_id"].astype(str).eq(str(anchor_id)),
            PHYSICS_DELTA_FEATURES,
        ]
        .iloc[0]
        .to_numpy(float)
    )
    x_delta_test = (
        test_df[PHYSICS_DELTA_FEATURES].to_numpy(float)
        - target_anchor_physics.reshape(1, -1)
    )

    pred_delta = (
        physics.predict(x_delta_test)
        + model.predict(test_df[features])
    )
    return np.maximum(target_anchor_y + pred_delta, 0.0)


def inner_score_anchor(
    outer_train,
    target_city,
    anchor_id,
    features,
    numeric,
    categorical,
    inner_max_iter,
    seed,
    city_weights,
):
    source_cities = sorted(outer_train["city"].unique())
    per_target = {"heating": [], "cooling": []}

    for inner_city in source_cities:
        inner_train = outer_train.loc[
            outer_train["city"] != inner_city
        ].copy()
        inner_val = outer_train.loc[
            outer_train["city"] == inner_city
        ].copy()

        pred_h = heating_physics_residual_fit_predict(
            inner_train,
            inner_val,
            anchor_id,
            features,
            numeric,
            categorical,
            inner_max_iter,
            seed,
        )
        pred_c = raw_relative_fit_predict(
            inner_train,
            inner_val,
            anchor_id,
            "cooling_kwh_m2",
            features,
            numeric,
            categorical,
            inner_max_iter,
            seed,
        )

        per_target["heating"].append(
            (
                inner_city,
                mean_absolute_error(
                    inner_val["heating_kwh_m2"],
                    pred_h,
                ),
            )
        )
        per_target["cooling"].append(
            (
                inner_city,
                mean_absolute_error(
                    inner_val["cooling_kwh_m2"],
                    pred_c,
                ),
            )
        )

    scores = {}
    for target, values in per_target.items():
        scores[target] = float(
            sum(
                city_weights[city] * mae
                for city, mae in values
            )
        )
    return scores


def metric(y, pred, top_frac):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    k = max(1, int(np.ceil(top_frac * len(y))))
    true_top = set(np.argsort(y)[:k])
    pred_top = set(np.argsort(pred)[:k])
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
        "kendall": float(kendalltau(y, pred).statistic),
        "design_regret_kwh_m2": float(
            y[int(np.argmin(pred))] - y.min()
        ),
        "top_recovery": float(len(true_top & pred_top) / k),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df[
        df["status"].astype(str).str.lower().eq("ok")
        & pd.to_numeric(df["severe_count"], errors="coerce")
        .fillna(0)
        .eq(0)
    ].copy()
    df = add_orientation(df).reset_index(drop=True)
    validate(df)

    features, numeric, categorical = model_feature_list(df)
    all_cities = sorted(df["city"].unique())
    test_cities = args.test_cities or all_cities

    pool = maximin_anchor_pool(
        df,
        args.candidate_pool_size,
    )

    metric_rows = []
    score_rows = []
    selected_rows = []

    for outer_fold, test_city in enumerate(test_cities):
        outer_train = df.loc[df["city"] != test_city].copy()
        outer_test = df.loc[df["city"] == test_city].copy()
        source_cities = sorted(outer_train["city"].unique())

        cweights = climate_weights(
            df,
            source_cities,
            test_city,
        )

        anchor_scores = []
        for anchor_id in pool:
            scores = inner_score_anchor(
                outer_train,
                test_city,
                anchor_id,
                features,
                numeric,
                categorical,
                args.inner_max_iter,
                args.seed + outer_fold,
                cweights,
            )
            anchor_scores.append(
                {
                    "anchor_id": anchor_id,
                    "heating_score": scores["heating"],
                    "cooling_score": scores["cooling"],
                }
            )

        score_df = pd.DataFrame(anchor_scores)

        # Common anchor score: dimensionless source-only inner-CV error.
        h_scale = max(float(score_df["heating_score"].median()), 1e-9)
        c_scale = max(float(score_df["cooling_score"].median()), 1e-9)
        score_df["combined_score"] = (
            score_df["heating_score"] / h_scale
            + score_df["cooling_score"] / c_scale
        )
        score_df = score_df.sort_values("combined_score").reset_index(drop=True)

        for _, row in score_df.iterrows():
            score_rows.append(
                {
                    "test_city": test_city,
                    **row.to_dict(),
                }
            )

        max_budget = max(args.anchor_budgets)
        selected_bank = score_df.head(max_budget).copy()

        # Fit each selected anchor once on all source climates.
        anchor_predictions = {}
        for _, arow in selected_bank.iterrows():
            anchor_id = str(arow["anchor_id"])
            pred_h = heating_physics_residual_fit_predict(
                outer_train,
                outer_test,
                anchor_id,
                features,
                numeric,
                categorical,
                args.outer_max_iter,
                args.seed + outer_fold,
            )
            pred_c = raw_relative_fit_predict(
                outer_train,
                outer_test,
                anchor_id,
                "cooling_kwh_m2",
                features,
                numeric,
                categorical,
                args.outer_max_iter,
                args.seed + outer_fold,
            )
            anchor_predictions[anchor_id] = {
                "heating": pred_h,
                "cooling": pred_c,
            }

        non_hvac = float(
            (
                outer_train["total_site_energy_kwh_m2"]
                - outer_train["heating_kwh_m2"]
                - outer_train["cooling_kwh_m2"]
            ).mean()
        )

        for budget in args.anchor_budgets:
            chosen = selected_bank.head(budget).copy()

            # Per-output inverse inner-CV weights.
            wh = 1.0 / np.maximum(
                chosen["heating_score"].to_numpy(float),
                1e-9,
            )
            wc = 1.0 / np.maximum(
                chosen["cooling_score"].to_numpy(float),
                1e-9,
            )
            wh /= wh.sum()
            wc /= wc.sum()

            chosen_ids = chosen["anchor_id"].astype(str).tolist()

            pred_h = sum(
                w * anchor_predictions[aid]["heating"]
                for w, aid in zip(wh, chosen_ids)
            )
            pred_c = sum(
                w * anchor_predictions[aid]["cooling"]
                for w, aid in zip(wc, chosen_ids)
            )
            pred_total = pred_h + pred_c + non_hvac

            for rank, aid in enumerate(chosen_ids, start=1):
                selected_rows.append(
                    {
                        "test_city": test_city,
                        "anchor_budget": budget,
                        "rank": rank,
                        "anchor_id": aid,
                    }
                )

            outputs = {
                "heating": (
                    outer_test["heating_kwh_m2"].to_numpy(),
                    pred_h,
                ),
                "cooling": (
                    outer_test["cooling_kwh_m2"].to_numpy(),
                    pred_c,
                ),
                "total": (
                    outer_test["total_site_energy_kwh_m2"].to_numpy(),
                    pred_total,
                ),
            }

            for target, (truth, pred) in outputs.items():
                metric_rows.append(
                    {
                        "test_city": test_city,
                        "anchor_budget": budget,
                        "target": target,
                        **metric(truth, pred, args.top_frac),
                    }
                )

        print(
            f"Done: {test_city} | selected="
            f"{selected_bank['anchor_id'].astype(str).tolist()}"
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        args.output_dir / "adaptive_anchor_metrics_by_city.csv",
        index=False,
    )

    summary = (
        metrics_df.groupby(
            ["anchor_budget", "target"],
            as_index=False,
        )[
            [
                "mae",
                "rmse",
                "r2",
                "spearman",
                "kendall",
                "design_regret_kwh_m2",
                "top_recovery",
            ]
        ]
        .mean()
    )
    summary.to_csv(
        args.output_dir / "adaptive_anchor_summary.csv",
        index=False,
    )

    pd.DataFrame(score_rows).to_csv(
        args.output_dir / "source_only_anchor_scores.csv",
        index=False,
    )
    pd.DataFrame(selected_rows).to_csv(
        args.output_dir / "selected_anchor_bank.csv",
        index=False,
    )

    metadata = {
        "candidate_pool_size": args.candidate_pool_size,
        "candidate_pool": pool,
        "anchor_budgets": args.anchor_budgets,
        "selection": (
            "Maximin candidate bank; source-only nested LOCO scores weighted "
            "by weather-distance to target climate; combined normalized "
            "heating/cooling score."
        ),
        "heating_model": (
            "paired physics-delta Ridge baseline + HGB residual"
        ),
        "cooling_model": "raw paired anchor-relative HGB",
        "target_energy_labels_used_for_anchor_selection": False,
        "target_weather_descriptors_used_for_anchor_selection": True,
    }
    with (args.output_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
