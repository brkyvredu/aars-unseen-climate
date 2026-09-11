from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classguard_v2.weather import find_epw, summarize_epw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITIES = ("Ankara", "Antalya", "Bursa", "Erzurum", "Istanbul", "Izmir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add EPW climate and orientation-specific solar features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "energy_results_classguard_v2_6000.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "energy_results_classguard_v2_6000_physics.csv",
    )
    parser.add_argument(
        "--weather-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "weather",
    )
    parser.add_argument("--skip-vertical-solar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    cities = sorted(data["city"].dropna().unique())

    climate_rows = []
    for city in cities:
        epw = find_epw(args.weather_root, city)
        summary = summarize_epw(
            city, epw, include_vertical_solar=not args.skip_vertical_solar
        )
        climate_rows.append(summary)
        print(f"Climate features: {city} -> {epw}")

    climate = pd.DataFrame(climate_rows)
    enriched = data.merge(climate.drop(columns=["epw_path"]), on="city", how="left")

    orientation_map = {
        "N": "climate_vertical_n_kwh_m2",
        "E": "climate_vertical_e_kwh_m2",
        "S": "climate_vertical_s_kwh_m2",
        "W": "climate_vertical_w_kwh_m2",
    }
    if not args.skip_vertical_solar:
        enriched["climate_orientation_vertical_solar_kwh_m2"] = [
            row[orientation_map[str(row["orientation"]).upper()]]
            for _, row in enriched.iterrows()
        ]

    # Physics-derived design-climate interactions.
    width = enriched["room_width_m"].astype(float)
    depth = enriched["room_depth_m"].astype(float)
    height = enriched["room_height_m"].astype(float)
    floor_area = enriched["floor_area_m2"].astype(float)
    volume = enriched["volume_m3"].astype(float)
    window_area = enriched["window_area_m2"].astype(float)
    gross_wall_area = 2 * (width + depth) * height
    opaque_wall_area = (gross_wall_area - window_area).clip(lower=0)

    enriched["envelope_area_to_volume_ratio"] = (
        gross_wall_area + 2 * floor_area
    ) / volume
    enriched["ua_envelope_w_k"] = (
        opaque_wall_area * enriched["wall_u_value_w_m2k"].astype(float)
        + floor_area * enriched["roof_u_value_w_m2k"].astype(float)
        + floor_area * enriched["floor_u_value_w_m2k"].astype(float)
        + window_area * enriched["glazing_u_value"].astype(float)
    )
    enriched["ua_intensity_w_m2k"] = enriched["ua_envelope_w_k"] / floor_area
    enriched["solar_aperture_ratio"] = (
        window_area * enriched["shgc"].astype(float) / floor_area
    )
    enriched["heating_transmission_proxy_kwh_m2"] = (
        enriched["ua_intensity_w_m2k"]
        * enriched["climate_heating_degree_days_18c"]
        * 24
        / 1000
    )
    enriched["cooling_transmission_proxy_kwh_m2"] = (
        enriched["ua_intensity_w_m2k"]
        * enriched["climate_cooling_degree_days_22c"]
        * 24
        / 1000
    )
    if not args.skip_vertical_solar:
        enriched["orientation_solar_gain_proxy_kwh_m2"] = (
            enriched["solar_aperture_ratio"]
            * enriched["climate_orientation_vertical_solar_kwh_m2"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False, encoding="utf-8")
    climate.to_csv(args.output.with_name("climate_features_classguard_v2.csv"), index=False)
    print(f"Saved: {args.output}")
    print(f"Rows: {len(enriched)}")


if __name__ == "__main__":
    main()
