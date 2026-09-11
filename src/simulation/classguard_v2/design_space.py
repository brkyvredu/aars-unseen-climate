from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


CITIES = ("Ankara", "Antalya", "Bursa", "Erzurum", "Istanbul", "Izmir")
ORIENTATIONS = ("N", "E", "S", "W")
ORIENTATION_DEG = {"N": 0, "E": 90, "S": 180, "W": 270}

GLAZING_OPTIONS = {
    "clear_double": {"glazing_u_value": 2.8, "shgc": 0.70, "vlt": 0.78},
    "low_e_double": {"glazing_u_value": 1.6, "shgc": 0.55, "vlt": 0.70},
    "high_performance_double": {
        "glazing_u_value": 1.2,
        "shgc": 0.40,
        "vlt": 0.62,
    },
}

ENVELOPE_OPTIONS = {
    "low_insulation": {
        "wall_u_value_w_m2k": 0.80,
        "roof_u_value_w_m2k": 0.60,
        "floor_u_value_w_m2k": 0.70,
    },
    "medium_insulation": {
        "wall_u_value_w_m2k": 0.50,
        "roof_u_value_w_m2k": 0.35,
        "floor_u_value_w_m2k": 0.45,
    },
    "high_insulation": {
        "wall_u_value_w_m2k": 0.30,
        "roof_u_value_w_m2k": 0.20,
        "floor_u_value_w_m2k": 0.30,
    },
}


@dataclass(frozen=True)
class DesignSpaceConfig:
    n_base_designs: int = 1000
    seed: int = 20260804
    fixed_shading_depth_m: float = 0.0

    width_range_m: tuple[float, float] = (6.0, 8.5)
    depth_range_m: tuple[float, float] = (7.0, 10.5)
    height_range_m: tuple[float, float] = (3.0, 3.8)
    wwr_range: tuple[float, float] = (0.20, 0.45)

    def validate(self) -> None:
        if self.n_base_designs <= 0:
            raise ValueError("n_base_designs must be positive.")
        if not 0 < self.wwr_range[0] < self.wwr_range[1] <= 0.45:
            raise ValueError("WWR range must lie inside (0, 0.45].")


def _balanced_labels(
    labels: tuple[str, ...] | list[str], n: int, rng: np.random.Generator
) -> np.ndarray:
    values = np.resize(np.asarray(labels, dtype=object), n)
    rng.shuffle(values)
    return values


def generate_base_designs(config: DesignSpaceConfig) -> pd.DataFrame:
    """Generate common base designs using a reproducible Latin hypercube."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    sampler = qmc.LatinHypercube(d=4, seed=config.seed)
    unit = sampler.random(n=config.n_base_designs)
    lower = np.array(
        [
            config.width_range_m[0],
            config.depth_range_m[0],
            config.height_range_m[0],
            config.wwr_range[0],
        ],
        dtype=float,
    )
    upper = np.array(
        [
            config.width_range_m[1],
            config.depth_range_m[1],
            config.height_range_m[1],
            config.wwr_range[1],
        ],
        dtype=float,
    )
    values = qmc.scale(unit, lower, upper)

    orientation = _balanced_labels(ORIENTATIONS, config.n_base_designs, rng)
    glazing_type = _balanced_labels(
        tuple(GLAZING_OPTIONS), config.n_base_designs, rng
    )
    envelope_type = _balanced_labels(
        tuple(ENVELOPE_OPTIONS), config.n_base_designs, rng
    )

    rows: list[dict[str, object]] = []
    for i in range(config.n_base_designs):
        width, depth, height, wwr = values[i]
        glazing = GLAZING_OPTIONS[str(glazing_type[i])]
        envelope = ENVELOPE_OPTIONS[str(envelope_type[i])]
        floor_area = width * depth
        volume = floor_area * height

        rows.append(
            {
                "base_design_id": f"base_{i + 1:04d}",
                "design_schema_version": "classguard_v2_paired_1",
                "room_type": "classroom",
                "room_width_m": round(float(width), 4),
                "room_depth_m": round(float(depth), 4),
                "room_height_m": round(float(height), 4),
                "floor_area_m2": round(float(floor_area), 6),
                "volume_m3": round(float(volume), 6),
                "aspect_ratio": round(float(depth / width), 6),
                "orientation": str(orientation[i]),
                "orientation_deg": ORIENTATION_DEG[str(orientation[i])],
                "window_wall_ratio": round(float(wwr), 5),
                "shading_depth_m": config.fixed_shading_depth_m,
                "glazing_type": str(glazing_type[i]),
                "wall_u_category": str(envelope_type[i]),
                "wall_u_value_w_m2k": envelope["wall_u_value_w_m2k"],
                "roof_u_value_w_m2k": envelope["roof_u_value_w_m2k"],
                "floor_u_value_w_m2k": envelope["floor_u_value_w_m2k"],
                "glazing_u_value": glazing["glazing_u_value"],
                "shgc": glazing["shgc"],
                "vlt": glazing["vlt"],
                "energy_simulation_ready": 1,
            }
        )

    return pd.DataFrame(rows)


def expand_across_cities(
    base_designs: pd.DataFrame, cities: tuple[str, ...] = CITIES
) -> pd.DataFrame:
    required = {"base_design_id", "orientation"}
    missing = required - set(base_designs.columns)
    if missing:
        raise ValueError(f"Missing base-design columns: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    for city in cities:
        block = base_designs.copy()
        block.insert(1, "city", city)
        block.insert(
            2,
            "design_id",
            [f"{city}_{value}" for value in block["base_design_id"]],
        )
        block.insert(
            3,
            "energy_design_id",
            [f"cg2_{city.lower()}_{i + 1:04d}" for i in range(len(block))],
        )
        frames.append(block)

    paired = pd.concat(frames, ignore_index=True)
    return paired.sort_values(["base_design_id", "city"], kind="stable").reset_index(
        drop=True
    )


PAIRED_INVARIANT_COLUMNS = (
    "base_design_id",
    "design_schema_version",
    "room_type",
    "room_width_m",
    "room_depth_m",
    "room_height_m",
    "floor_area_m2",
    "volume_m3",
    "aspect_ratio",
    "orientation",
    "orientation_deg",
    "window_wall_ratio",
    "requested_wwr",
    "actual_wwr",
    "window_area_m2",
    "shading_depth_m",
    "glazing_type",
    "wall_u_category",
    "wall_u_value_w_m2k",
    "roof_u_value_w_m2k",
    "floor_u_value_w_m2k",
    "glazing_u_value",
    "shgc",
    "vlt",
    "energy_simulation_ready",
)


def validate_paired_design_space(
    paired: pd.DataFrame, cities: tuple[str, ...] = CITIES
) -> dict[str, object]:
    design_columns = [c for c in PAIRED_INVARIANT_COLUMNS if c in paired.columns]
    counts = paired.groupby("base_design_id")["city"].nunique()
    incomplete = counts[counts != len(cities)]

    mismatch_columns: list[str] = []
    for col in design_columns:
        if col == "base_design_id":
            continue
        if (paired.groupby("base_design_id")[col].nunique(dropna=False) > 1).any():
            mismatch_columns.append(col)

    expected_rows = paired["base_design_id"].nunique() * len(cities)
    duplicate_energy_ids = int(paired["energy_design_id"].duplicated().sum())

    report = {
        "rows": int(len(paired)),
        "expected_rows": int(expected_rows),
        "base_designs": int(paired["base_design_id"].nunique()),
        "cities": sorted(paired["city"].unique().tolist()),
        "incomplete_base_designs": int(len(incomplete)),
        "mismatched_design_columns": mismatch_columns,
        "duplicate_energy_design_ids": duplicate_energy_ids,
        "city_counts": {
            str(k): int(v) for k, v in paired["city"].value_counts().sort_index().items()
        },
        "orientation_counts_per_city": {
            f"{city}:{orientation}": int(value)
            for (city, orientation), value in paired.groupby(
                ["city", "orientation"]
            ).size().items()
        },
    }
    report["valid"] = bool(
        len(paired) == expected_rows
        and len(incomplete) == 0
        and not mismatch_columns
        and duplicate_energy_ids == 0
        and set(report["cities"]) == set(cities)
    )
    return report


def write_design_space(
    output_csv: Path, config: DesignSpaceConfig = DesignSpaceConfig()
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = generate_base_designs(config)
    paired = expand_across_cities(base)
    report = validate_paired_design_space(paired)
    if not report["valid"]:
        raise RuntimeError(f"Paired design-space validation failed: {report}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_csv, index=False, encoding="utf-8")
    return paired, report
