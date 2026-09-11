from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from .idf import ModelMetadata, to_float


GJ_TO_KWH = 277.7777777778


def gj_to_kwh_m2(value_gj: float, area_m2: float) -> float:
    return value_gj * GJ_TO_KWH / area_m2


def parse_error_counts(err_text: str) -> tuple[int, int]:
    return err_text.count("** Warning"), err_text.count("** Severe")


def _result_base(row: dict[str, object], metadata: ModelMetadata) -> dict[str, object]:
    return {
        "energy_design_id": row.get("energy_design_id", ""),
        "base_design_id": row.get("base_design_id", ""),
        "source_design_id": row.get("design_id", ""),
        "design_schema_version": row.get("design_schema_version", ""),
        "city": row.get("city", ""),
        "orientation": row.get("orientation", ""),
        "room_width_m": row.get("room_width_m", ""),
        "room_depth_m": row.get("room_depth_m", ""),
        "room_height_m": row.get("room_height_m", ""),
        "floor_area_m2": f"{metadata.floor_area_m2:.6f}",
        "volume_m3": f"{metadata.volume_m3:.6f}",
        "requested_wwr": row.get("window_wall_ratio", ""),
        "actual_wwr": f"{metadata.actual_wwr:.6f}",
        "window_area_m2": f"{metadata.window_area_m2:.6f}",
        "glazing_type": row.get("glazing_type", ""),
        "wall_u_category": row.get("wall_u_category", ""),
        "wall_u_value_w_m2k": row.get("wall_u_value_w_m2k", ""),
        "roof_u_value_w_m2k": row.get("roof_u_value_w_m2k", ""),
        "floor_u_value_w_m2k": row.get("floor_u_value_w_m2k", ""),
        "glazing_u_value": row.get("glazing_u_value", ""),
        "shgc": row.get("shgc", ""),
        "vlt": row.get("vlt", ""),
        "shading_depth_m": row.get("shading_depth_m", ""),
        "ground_temperature_source": metadata.ground_temperature_source,
        "design_day_names": " | ".join(metadata.design_day_names),
        "actual_wall_u_no_film_w_m2k": None,
        "actual_roof_u_no_film_w_m2k": None,
        "actual_floor_u_no_film_w_m2k": None,
        "actual_glazing_u_w_m2k": None,
        "actual_glazing_shgc": None,
        "total_site_energy_gj": None,
        "total_site_energy_kwh_m2": None,
        "heating_gj": None,
        "heating_kwh_m2": None,
        "cooling_gj": None,
        "cooling_kwh_m2": None,
        "lighting_gj": None,
        "lighting_kwh_m2": None,
        "equipment_gj": None,
        "equipment_kwh_m2": None,
        "status": "ok",
        "warning_count": None,
        "severe_count": None,
        "error_message": "",
    }


def _find_table_start(rows: list[list[str]], table_name: str) -> int | None:
    for i, row in enumerate(rows):
        clean = [value.strip() for value in row]
        if clean == [table_name]:
            return i
    return None


def _extract_envelope_performance(rows: list[list[str]], result: dict[str, object]) -> None:
    start = _find_table_start(rows, "Opaque Exterior")
    if start is not None:
        header_index = start + 2
        if header_index < len(rows):
            header = [value.strip() for value in rows[header_index]]
            index = {name: i for i, name in enumerate(header) if name}
            for row in rows[header_index + 1 :]:
                clean = [value.strip() for value in row]
                if not any(clean):
                    break
                if len(clean) <= max(index.values(), default=0):
                    continue
                surface = clean[1].upper() if len(clean) > 1 else ""
                u = to_float(clean[index.get("U-Factor no Film [W/m2-K]", -1)], math.nan)
                if surface in {"SOUTH WALL", "EAST WALL", "NORTH WALL", "WEST WALL"}:
                    if result["actual_wall_u_no_film_w_m2k"] is None and math.isfinite(u):
                        result["actual_wall_u_no_film_w_m2k"] = f"{u:.6f}"
                elif surface == "ROOF" and math.isfinite(u):
                    result["actual_roof_u_no_film_w_m2k"] = f"{u:.6f}"
                elif surface == "FLOOR" and math.isfinite(u):
                    result["actual_floor_u_no_film_w_m2k"] = f"{u:.6f}"

    for table_name in ("Exterior Fenestration", "Exterior Window"):
        start = _find_table_start(rows, table_name)
        if start is None:
            continue
        for row in rows[start : start + 30]:
            clean = [value.strip() for value in row]
            joined = "|".join(clean).upper()
            if "MAIN WINDOW" not in joined:
                continue
            numeric = [to_float(value, math.nan) for value in clean]
            finite = [value for value in numeric if math.isfinite(value)]
            # More reliable parsing is handled from known headers below when present.
            if finite:
                break

    # Header-driven scan for glazing fields across all tabular sections.
    for i, row in enumerate(rows):
        header = [value.strip() for value in row]
        if "Glass U-Factor [W/m2-K]" not in header:
            continue
        index = {name: j for j, name in enumerate(header) if name}
        for data_row in rows[i + 1 : i + 20]:
            clean = [value.strip() for value in data_row]
            if len(clean) <= max(index.values(), default=0):
                continue
            if not any("WINDOW" in value.upper() for value in clean[:3]):
                continue
            u = to_float(clean[index["Glass U-Factor [W/m2-K]"]], math.nan)
            shgc = to_float(clean[index.get("Glass SHGC", -1)], math.nan)
            if math.isfinite(u):
                result["actual_glazing_u_w_m2k"] = f"{u:.6f}"
            if math.isfinite(shgc):
                result["actual_glazing_shgc"] = f"{shgc:.6f}"
            return


def extract_summary(
    eplustbl_path: Path, row: dict[str, object], metadata: ModelMetadata
) -> dict[str, object]:
    with eplustbl_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.reader(f))

    result = _result_base(row, metadata)
    area_m2 = metadata.floor_area_m2

    for table_row in rows:
        clean = [value.strip() for value in table_row]
        if len(clean) >= 3 and clean[1] == "Total Site Energy":
            total_gj = to_float(clean[2])
            result["total_site_energy_gj"] = f"{total_gj:.6f}"
            result["total_site_energy_kwh_m2"] = f"{gj_to_kwh_m2(total_gj, area_m2):.6f}"
            break

    end_uses_start = _find_table_start(rows, "End Uses")
    if end_uses_start is None:
        raise RuntimeError("End Uses table not found in eplustbl.csv")
    header = [value.strip() for value in rows[end_uses_start + 2]]
    col_index = {name: idx for idx, name in enumerate(header) if name}

    for table_row in rows[end_uses_start + 3 :]:
        clean = [value.strip() for value in table_row]
        if len(clean) > 1 and clean[1] == "Total End Uses":
            break
        if len(clean) <= 1:
            continue
        name = clean[1]
        if name == "Heating":
            value = to_float(clean[col_index["District Heating Water [GJ]"]])
            result["heating_gj"] = f"{value:.6f}"
            result["heating_kwh_m2"] = f"{gj_to_kwh_m2(value, area_m2):.6f}"
        elif name == "Cooling":
            value = to_float(clean[col_index["District Cooling [GJ]"]])
            result["cooling_gj"] = f"{value:.6f}"
            result["cooling_kwh_m2"] = f"{gj_to_kwh_m2(value, area_m2):.6f}"
        elif name == "Interior Lighting":
            value = to_float(clean[col_index["Electricity [GJ]"]])
            result["lighting_gj"] = f"{value:.6f}"
            result["lighting_kwh_m2"] = f"{gj_to_kwh_m2(value, area_m2):.6f}"
        elif name == "Interior Equipment":
            value = to_float(clean[col_index["Electricity [GJ]"]])
            result["equipment_gj"] = f"{value:.6f}"
            result["equipment_kwh_m2"] = f"{gj_to_kwh_m2(value, area_m2):.6f}"

    _extract_envelope_performance(rows, result)
    return result


def failed_result(
    row: dict[str, object],
    metadata: ModelMetadata,
    warning_count: int,
    severe_count: int,
    error_message: str,
) -> dict[str, object]:
    result = _result_base(row, metadata)
    result.update(
        {
            "status": "failed",
            "warning_count": warning_count,
            "severe_count": severe_count,
            "error_message": re.sub(r"\s+", " ", error_message).strip()[:2000],
        }
    )
    return result
