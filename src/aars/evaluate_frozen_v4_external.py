from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Adaptive Anchor Relative Surrogate v4 on "
            "previously untouched external climates."
        )
    )
    p.add_argument("--source-physics", type=Path, required=True)
    p.add_argument("--external-physics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidate-pool-size", type=int, default=10)
    p.add_argument("--anchor-budgets", type=int, nargs="+", default=[1, 3])
    p.add_argument("--inner-max-iter", type=int, default=100)
    p.add_argument("--outer-max-iter", type=int, default=220)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-frac", type=float, default=0.05)
    return p.parse_args()


def add_orientation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    angles = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
    theta = np.deg2rad(out["orientation"].map(angles).astype(float))
    out["orientation_cos"] = np.cos(theta)
    out["orientation_sin"] = np.sin(theta)
    return out


def model_feature_list(df: pd.DataFrame):
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
        max_iter=max_iter,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("reg", reg)])


def maximin_anchor_pool(source: pd.DataFrame, pool_size: int) -> list[str]:
    first_city = sorted(source["city"].unique())[0]
    base = (
        source.loc[source["city"].eq(first_city)]
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


def climate_weights(source: pd.DataFrame, target: pd.DataFrame) -> dict[str, float]:
    source_cities = sorted(source["city"].unique())
    src = source.groupby("city")[CLIMATE_DISTANCE_FEATURES].first()
    target_vec = target[CLIMATE_DISTANCE_FEATURES].iloc[0]

    combined = pd.concat(
        [
            src,
            pd.DataFrame([target_vec], index=["__target__"]),
        ]
    )
    scaler = StandardScaler().fit(combined)
    z = pd.DataFrame(
        scaler.transform(combined),
        index=combined.index,
    )
    t = z.loc["__target__"].to_numpy()

    dist = np.asarray(
        [np.linalg.norm(z.loc[c].to_numpy() - t) for c in source_cities],
        dtype=float,
    )
    tau = max(float(np.median(dist)), 1e-6)
    w = np.exp(-dist / tau)
    w /= w.sum()
    return dict(zip(source_cities, w))


def anchor_physics_delta(df: pd.DataFrame, anchor_id: str) -> np.ndarray:
    anchor = (
        df.loc[df["base_design_id"].astype(str).eq(str(anchor_id))]
        .set_index("city")[PHYSICS_DELTA_FEATURES]
    )
    return np.column_stack(
        [
            df[c].to_numpy(float)
            - df["city"].map(anchor[c].to_dict()).to_numpy(float)
            for c in PHYSICS_DELTA_FEATURES
        ]
    )


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
    train_target = train_df[target] - train_df["city"].map(train_anchor)
    model = make_hgb(numeric, categorical, max_iter, seed)
    model.fit(train_df[features], train_target)
    return np.maximum(
        target_anchor + model.predict(test_df[features]),
        0.0,
    )


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
    residual = y_delta_train - physics.predict(x_delta_train)

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
    source,
    anchor_id,
    features,
    numeric,
    categorical,
    max_iter,
    seed,
    weights,
):
    source_cities = sorted(source["city"].unique())
    values = {"heating": [], "cooling": []}

    for inner_city in source_cities:
        tr = source.loc[source["city"] != inner_city].copy()
        va = source.loc[source["city"] == inner_city].copy()

        ph = heating_physics_residual_fit_predict(
            tr, va, anchor_id,
            features, numeric, categorical, max_iter, seed,
        )
        pc = raw_relative_fit_predict(
            tr, va, anchor_id, "cooling_kwh_m2",
            features, numeric, categorical, max_iter, seed,
        )
        values["heating"].append(
            weights[inner_city]
            * mean_absolute_error(va["heating_kwh_m2"], ph)
        )
        values["cooling"].append(
            weights[inner_city]
            * mean_absolute_error(va["cooling_kwh_m2"], pc)
        )

    return {
        "heating_score": float(sum(values["heating"])),
        "cooling_score": float(sum(values["cooling"])),
    }


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = add_orientation(pd.read_csv(args.source_physics))
    external = add_orientation(pd.read_csv(args.external_physics))

    source = source[
        source["status"].astype(str).str.lower().eq("ok")
        & pd.to_numeric(source["severe_count"], errors="coerce").fillna(0).eq(0)
    ].copy()
    external = external[
        external["status"].astype(str).str.lower().eq("ok")
        & pd.to_numeric(external["severe_count"], errors="coerce").fillna(0).eq(0)
    ].copy()

    if set(source["city"]) & set(external["city"]):
        overlap = sorted(set(source["city"]) & set(external["city"]))
        raise ValueError(f"External validation contains source cities: {overlap}")

    features, numeric, categorical = model_feature_list(source)
    missing_external_features = sorted(set(features) - set(external.columns))
    if missing_external_features:
        raise ValueError(
            f"External physics CSV missing model features: {missing_external_features}"
        )

    pool = maximin_anchor_pool(source, args.candidate_pool_size)

    metric_rows = []
    selected_rows = []
    score_rows = []

    for target_i, (target_city, test_df) in enumerate(
        external.groupby("city", sort=True)
    ):
        test_df = test_df.copy()
        if test_df["base_design_id"].nunique() != 1000:
            raise ValueError(
                f"External city {target_city} does not contain 1,000 base designs."
            )

        weights = climate_weights(source, test_df)

        scored = []
        for anchor_id in pool:
            s = inner_score_anchor(
                source,
                anchor_id,
                features,
                numeric,
                categorical,
                args.inner_max_iter,
                args.seed + target_i,
                weights,
            )
            scored.append({"anchor_id": anchor_id, **s})

        score_df = pd.DataFrame(scored)
        hscale = max(float(score_df["heating_score"].median()), 1e-9)
        cscale = max(float(score_df["cooling_score"].median()), 1e-9)
        score_df["combined_score"] = (
            score_df["heating_score"] / hscale
            + score_df["cooling_score"] / cscale
        )
        score_df = score_df.sort_values("combined_score").reset_index(drop=True)

        for _, row in score_df.iterrows():
            score_rows.append({"target_city": target_city, **row.to_dict()})

        bank = score_df.head(max(args.anchor_budgets))
        predictions = {}
        for anchor_id in bank["anchor_id"].astype(str):
            ph = heating_physics_residual_fit_predict(
                source,
                test_df,
                anchor_id,
                features,
                numeric,
                categorical,
                args.outer_max_iter,
                args.seed + target_i,
            )
            pc = raw_relative_fit_predict(
                source,
                test_df,
                anchor_id,
                "cooling_kwh_m2",
                features,
                numeric,
                categorical,
                args.outer_max_iter,
                args.seed + target_i,
            )
            predictions[anchor_id] = {"heating": ph, "cooling": pc}

        non_hvac = float(
            (
                source["total_site_energy_kwh_m2"]
                - source["heating_kwh_m2"]
                - source["cooling_kwh_m2"]
            ).mean()
        )

        for budget in args.anchor_budgets:
            chosen = bank.head(budget)
            ids = chosen["anchor_id"].astype(str).tolist()

            wh = 1.0 / np.maximum(chosen["heating_score"].to_numpy(float), 1e-9)
            wc = 1.0 / np.maximum(chosen["cooling_score"].to_numpy(float), 1e-9)
            wh /= wh.sum()
            wc /= wc.sum()

            pred_h = sum(w * predictions[a]["heating"] for w, a in zip(wh, ids))
            pred_c = sum(w * predictions[a]["cooling"] for w, a in zip(wc, ids))
            pred_total = pred_h + pred_c + non_hvac

            for rank, anchor_id in enumerate(ids, start=1):
                selected_rows.append(
                    {
                        "target_city": target_city,
                        "anchor_budget": budget,
                        "rank": rank,
                        "anchor_id": anchor_id,
                    }
                )

            outputs = {
                "heating": (test_df["heating_kwh_m2"], pred_h),
                "cooling": (test_df["cooling_kwh_m2"], pred_c),
                "total": (test_df["total_site_energy_kwh_m2"], pred_total),
            }
            for target, (truth, pred) in outputs.items():
                metric_rows.append(
                    {
                        "target_city": target_city,
                        "anchor_budget": budget,
                        "target": target,
                        **metric(truth, pred, args.top_frac),
                    }
                )

        print(
            f"Done: {target_city} | ranked anchors="
            f"{bank['anchor_id'].astype(str).tolist()}"
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        args.output_dir / "external_metrics_by_city.csv",
        index=False,
    )

    summary = (
        metrics.groupby(["anchor_budget", "target"], as_index=False)[
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
        args.output_dir / "external_summary.csv",
        index=False,
    )

    pd.DataFrame(selected_rows).to_csv(
        args.output_dir / "external_selected_anchor_bank.csv",
        index=False,
    )
    pd.DataFrame(score_rows).to_csv(
        args.output_dir / "external_source_only_anchor_scores.csv",
        index=False,
    )

    metadata = {
        "method_status": "FROZEN BEFORE EXTERNAL EVALUATION",
        "source_cities": sorted(source["city"].unique().tolist()),
        "external_cities": sorted(external["city"].unique().tolist()),
        "candidate_pool_size": args.candidate_pool_size,
        "anchor_budgets": args.anchor_budgets,
        "target_energy_labels_used_for_anchor_selection": False,
        "target_weather_descriptors_used_for_anchor_selection": True,
        "target_energy_labels_used_after_selection": (
            "Only the selected anchor simulation values are used for calibration; "
            "the remaining target labels are used exclusively for final evaluation."
        ),
    }
    with (args.output_dir / "external_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
