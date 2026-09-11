from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from classguard_v2.weather import summarize_epw


CLIMATE_FEATURES = [
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
            "Select external EPW climates objectively using source-climate "
            "distance and greedy maximin diversity."
        )
    )
    p.add_argument("--source-physics", type=Path, required=True)
    p.add_argument("--candidate-weather-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-select", type=int, default=8)
    p.add_argument(
        "--exclude-city",
        nargs="*",
        default=["Ankara", "Antalya", "Bursa", "Erzurum", "Istanbul", "Izmir"],
    )
    p.add_argument(
        "--include-vertical-solar",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p.parse_args()


def scan_epws(root: Path, exclude: set[str], include_vertical_solar: bool) -> pd.DataFrame:
    rows = []
    for epw in sorted(root.rglob("*.epw")):
        city = epw.parent.name
        if city.lower() in {x.lower() for x in exclude}:
            continue
        try:
            row = summarize_epw(
                city,
                epw,
                include_vertical_solar=include_vertical_solar,
            )
            row["candidate_city"] = city
            row["candidate_epw"] = str(epw.resolve())
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "candidate_city": city,
                    "candidate_epw": str(epw.resolve()),
                    "scan_error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not rows:
        raise RuntimeError(f"No EPW candidates found under: {root}")
    return pd.DataFrame(rows)


def greedy_select(source_z: np.ndarray, candidate_z: np.ndarray, n_select: int) -> list[int]:
    # Distance of each candidate to its nearest source climate.
    d_to_source = np.sqrt(
        np.square(candidate_z[:, None, :] - source_z[None, :, :]).sum(axis=2)
    ).min(axis=1)

    selected = [int(np.argmax(d_to_source))]

    while len(selected) < min(n_select, len(candidate_z)):
        d_to_selected = np.sqrt(
            np.square(
                candidate_z[:, None, :]
                - candidate_z[np.asarray(selected)][None, :, :]
            ).sum(axis=2)
        ).min(axis=1)

        # Externality and diversity receive equal geometric weight.
        score = np.sqrt(np.maximum(d_to_source, 1e-12) * np.maximum(d_to_selected, 1e-12))
        score[np.asarray(selected)] = -np.inf
        selected.append(int(np.argmax(score)))

    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.source_physics)
    source_climate = (
        source.groupby("city")[CLIMATE_FEATURES]
        .first()
        .reset_index()
    )

    candidates = scan_epws(
        args.candidate_weather_root,
        set(args.exclude_city),
        args.include_vertical_solar,
    )
    candidates.to_csv(
        args.output_dir / "external_epw_candidate_catalog.csv",
        index=False,
    )

    usable = candidates.dropna(subset=CLIMATE_FEATURES).copy()
    if len(usable) < args.n_select:
        raise RuntimeError(
            f"Only {len(usable)} usable EPW candidates; requested {args.n_select}."
        )

    scaler = StandardScaler().fit(source_climate[CLIMATE_FEATURES])
    source_z = scaler.transform(source_climate[CLIMATE_FEATURES])
    candidate_z = scaler.transform(usable[CLIMATE_FEATURES])

    selected_idx = greedy_select(source_z, candidate_z, args.n_select)
    selected = usable.iloc[selected_idx].copy()

    d_to_source = np.sqrt(
        np.square(
            candidate_z[selected_idx][:, None, :] - source_z[None, :, :]
        ).sum(axis=2)
    )
    selected["nearest_source_distance_z"] = d_to_source.min(axis=1)
    selected["nearest_source_city"] = source_climate.iloc[
        d_to_source.argmin(axis=1)
    ]["city"].to_numpy()

    selected.insert(
        0,
        "external_order",
        np.arange(1, len(selected) + 1),
    )
    selected.to_csv(
        args.output_dir / "selected_external_climates.csv",
        index=False,
    )

    print(selected[
        [
            "external_order",
            "candidate_city",
            "nearest_source_city",
            "nearest_source_distance_z",
            *CLIMATE_FEATURES,
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
