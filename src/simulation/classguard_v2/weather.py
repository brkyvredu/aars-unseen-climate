from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EPW_MISSING_VALUES = {99.9, 999.0, 9999.0, 99999.0, 999999.0}


@dataclass(frozen=True)
class EpwLocation:
    name: str
    latitude: float
    longitude: float
    time_zone: float
    elevation_m: float


@dataclass(frozen=True)
class WeatherContext:
    city: str
    epw_path: Path
    location: EpwLocation
    ground_temperatures_c: tuple[float, ...]
    ground_temperature_source: str
    design_day_idf: str = ""
    design_day_names: tuple[str, ...] = ()
    ddy_path: Path | None = None


def find_epw(weather_root: Path, city: str) -> Path:
    city_dir = weather_root / city
    if not city_dir.exists():
        raise FileNotFoundError(f"Weather city folder not found: {city_dir}")
    epws = sorted(city_dir.rglob("*.epw"))
    if not epws:
        raise FileNotFoundError(f"EPW not found under: {city_dir}")
    return epws[0]


def find_companion_file(epw_path: Path, suffix: str) -> Path | None:
    direct = epw_path.with_suffix(suffix)
    if direct.exists():
        return direct
    candidates = sorted(epw_path.parent.glob(f"*{suffix}"))
    return candidates[0] if candidates else None


def read_epw_header(epw_path: Path) -> list[list[str]]:
    with epw_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        return [next(reader) for _ in range(8)]


def parse_location(header: list[list[str]]) -> EpwLocation:
    row = header[0]
    if not row or row[0].strip().upper() != "LOCATION" or len(row) < 10:
        raise ValueError("Invalid EPW LOCATION header.")
    place = ", ".join(part.strip() for part in row[1:5] if part.strip())
    return EpwLocation(
        name=place,
        latitude=float(row[6]),
        longitude=float(row[7]),
        time_zone=float(row[8]),
        elevation_m=float(row[9]),
    )


def _parse_ground_temperature_sets(row: list[str]) -> list[tuple[float, tuple[float, ...]]]:
    if not row or row[0].strip().upper() != "GROUND TEMPERATURES":
        return []
    try:
        n_sets = int(float(row[1]))
    except (ValueError, IndexError):
        return []

    values = row[2:]
    sets: list[tuple[float, tuple[float, ...]]] = []
    cursor = 0
    for _ in range(n_sets):
        # EPW ground-temperature record: depth, conductivity, density,
        # specific heat, then Jan-Dec monthly temperatures.
        if cursor + 16 > len(values):
            break
        try:
            depth = float(values[cursor])
            months = tuple(float(x) for x in values[cursor + 4 : cursor + 16])
        except ValueError:
            cursor += 16
            continue
        if len(months) == 12 and all(math.isfinite(x) for x in months):
            sets.append((depth, months))
        cursor += 16
    return sets


def read_epw_hourly(epw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(epw_path, skiprows=8, header=None, low_memory=False)
    if raw.shape[1] < 22:
        raise ValueError(f"EPW has only {raw.shape[1]} columns: {epw_path}")

    def numeric(index: int) -> pd.Series:
        series = pd.to_numeric(raw.iloc[:, index], errors="coerce")
        return series.mask(series.isin(EPW_MISSING_VALUES))

    out = pd.DataFrame(
        {
            "year": pd.to_numeric(raw.iloc[:, 0], errors="coerce"),
            "month": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "day": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "hour": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "minute": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
            "dry_bulb_c": numeric(6),
            "dew_point_c": numeric(7),
            "relative_humidity_pct": numeric(8),
            "atmospheric_pressure_pa": numeric(9),
            "global_horizontal_radiation_wh_m2": numeric(13),
            "direct_normal_radiation_wh_m2": numeric(14),
            "diffuse_horizontal_radiation_wh_m2": numeric(15),
            "wind_direction_deg": numeric(20),
            "wind_speed_m_s": numeric(21),
        }
    )
    return out


def ground_temperatures_from_epw(
    epw_path: Path, preferred_depth_m: float = 0.5
) -> tuple[tuple[float, ...], str]:
    header = read_epw_header(epw_path)
    sets = _parse_ground_temperature_sets(header[3])
    if sets:
        depth, temperatures = min(sets, key=lambda item: abs(item[0] - preferred_depth_m))
        return temperatures, f"epw_ground_temperature_depth_{depth:g}m"

    hourly = read_epw_hourly(epw_path)
    monthly = hourly.groupby("month")["dry_bulb_c"].mean().reindex(range(1, 13))
    if monthly.isna().any():
        annual = float(hourly["dry_bulb_c"].mean())
        monthly = monthly.fillna(annual)
    return tuple(float(x) for x in monthly), "fallback_epw_monthly_mean_dry_bulb"


def _idf_objects(text: str, object_type: str) -> list[str]:
    cleaned_lines = []
    for line in text.splitlines():
        cleaned_lines.append(line.split("!", 1)[0])
    cleaned = "\n".join(cleaned_lines)
    pattern = re.compile(
        rf"(?is)(?:^|;)\s*({re.escape(object_type)}\s*,.*?;)"
    )
    return [match.group(1).strip() for match in pattern.finditer(cleaned)]


def _object_name(idf_object: str) -> str:
    fields = [part.strip() for part in idf_object.split(",")]
    return fields[1].split(";", 1)[0].strip() if len(fields) > 1 else ""


def select_design_days(ddy_path: Path) -> tuple[str, tuple[str, ...]]:
    text = ddy_path.read_text(encoding="utf-8-sig", errors="ignore")
    objects = _idf_objects(text, "SizingPeriod:DesignDay")
    if not objects:
        return "", ()

    names = [_object_name(obj) for obj in objects]

    def choose(patterns: tuple[str, ...]) -> int | None:
        for pattern in patterns:
            for i, name in enumerate(names):
                if re.search(pattern, name, flags=re.IGNORECASE):
                    return i
        return None

    winter_idx = choose((r"99\.6%.*DB", r"99%.*DB", r"heating.*99", r"winter"))
    summer_idx = choose(
        (r"0\.4%.*DB", r"1%.*DB", r"cooling.*0\.4", r"summer")
    )

    selected_indices: list[int] = []
    for index in (winter_idx, summer_idx):
        if index is not None and index not in selected_indices:
            selected_indices.append(index)

    if not selected_indices:
        selected_indices = list(range(min(2, len(objects))))

    selected = [objects[i] for i in selected_indices]
    selected_names = tuple(names[i] for i in selected_indices)
    return "\n\n".join(selected), selected_names


def build_weather_context(
    weather_root: Path,
    city: str,
    *,
    include_ddy: bool = True,
    preferred_ground_depth_m: float = 0.5,
) -> WeatherContext:
    epw_path = find_epw(weather_root, city)
    header = read_epw_header(epw_path)
    location = parse_location(header)
    ground, ground_source = ground_temperatures_from_epw(
        epw_path, preferred_depth_m=preferred_ground_depth_m
    )

    ddy_path = find_companion_file(epw_path, ".ddy") if include_ddy else None
    design_day_idf = ""
    design_day_names: tuple[str, ...] = ()
    if ddy_path is not None:
        design_day_idf, design_day_names = select_design_days(ddy_path)

    return WeatherContext(
        city=city,
        epw_path=epw_path,
        location=location,
        ground_temperatures_c=ground,
        ground_temperature_source=ground_source,
        design_day_idf=design_day_idf,
        design_day_names=design_day_names,
        ddy_path=ddy_path,
    )


def degree_days_from_hourly(temp_c: pd.Series, base: float, mode: str) -> float:
    values = pd.to_numeric(temp_c, errors="coerce").dropna().to_numpy()
    if mode == "heating":
        degree_hours = np.maximum(base - values, 0.0).sum()
    elif mode == "cooling":
        degree_hours = np.maximum(values - base, 0.0).sum()
    else:
        raise ValueError(f"Unsupported degree-day mode: {mode}")
    return float(degree_hours / 24.0)


def _fixed_offset_index(hourly: pd.DataFrame, location: EpwLocation) -> pd.DatetimeIndex:
    year = hourly["year"].fillna(2001).astype(int).to_numpy()
    year = np.where(year <= 0, 2001, year)
    # EPW hours are 1-24 and represent the end of each hour. Use the midpoint
    # of the reporting interval for solar-position/transposition calculations.
    base = pd.to_datetime(
        {
            "year": year,
            "month": hourly["month"].astype(int),
            "day": hourly["day"].astype(int),
        },
        errors="coerce",
    )
    index = base + pd.to_timedelta(hourly["hour"] - 1, unit="h") + pd.Timedelta(
        minutes=30
    )
    tz = timezone(timedelta(hours=location.time_zone))
    return pd.DatetimeIndex(index).tz_localize(tz)


def vertical_solar_exposure_kwh_m2(
    hourly: pd.DataFrame, location: EpwLocation
) -> dict[str, float]:
    try:
        import pvlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pvlib is required for orientation-specific vertical irradiation. "
            "Install it with: pip install pvlib"
        ) from exc

    index = _fixed_offset_index(hourly, location)
    solar_position = pvlib.solarposition.get_solarposition(
        index,
        latitude=location.latitude,
        longitude=location.longitude,
        altitude=location.elevation_m,
        pressure=hourly["atmospheric_pressure_pa"].fillna(101325).to_numpy(),
        temperature=hourly["dry_bulb_c"].fillna(15).to_numpy(),
    )
    dni_extra = pvlib.irradiance.get_extra_radiation(index)
    ghi = hourly["global_horizontal_radiation_wh_m2"].clip(lower=0).fillna(0)
    dni = hourly["direct_normal_radiation_wh_m2"].clip(lower=0).fillna(0)
    dhi = hourly["diffuse_horizontal_radiation_wh_m2"].clip(lower=0).fillna(0)

    result: dict[str, float] = {}
    for orientation, azimuth in {"n": 0, "e": 90, "s": 180, "w": 270}.items():
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=90,
            surface_azimuth=azimuth,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=dni.to_numpy(),
            ghi=ghi.to_numpy(),
            dhi=dhi.to_numpy(),
            dni_extra=np.asarray(dni_extra),
            albedo=0.2,
            model="haydavies",
        )
        result[f"climate_vertical_{orientation}_kwh_m2"] = float(
            np.nansum(np.clip(np.asarray(poa["poa_global"]), 0, None)) / 1000.0
        )
    return result


def summarize_epw(city: str, epw_path: Path, include_vertical_solar: bool = True) -> dict[str, object]:
    header = read_epw_header(epw_path)
    location = parse_location(header)
    hourly = read_epw_hourly(epw_path)
    temp = hourly["dry_bulb_c"]

    row: dict[str, object] = {
        "city": city,
        "epw_path": str(epw_path),
        "climate_latitude": location.latitude,
        "climate_longitude": location.longitude,
        "climate_elevation_m": location.elevation_m,
        "climate_annual_mean_temp_c": float(temp.mean()),
        "climate_annual_min_temp_c": float(temp.min()),
        "climate_annual_max_temp_c": float(temp.max()),
        "climate_temp_p01_c": float(temp.quantile(0.01)),
        "climate_temp_p05_c": float(temp.quantile(0.05)),
        "climate_temp_p95_c": float(temp.quantile(0.95)),
        "climate_temp_p99_c": float(temp.quantile(0.99)),
        "climate_hours_below_0c": int((temp < 0).sum()),
        "climate_hours_above_30c": int((temp > 30).sum()),
        "climate_heating_degree_days_18c": degree_days_from_hourly(temp, 18, "heating"),
        "climate_cooling_degree_days_22c": degree_days_from_hourly(temp, 22, "cooling"),
        "climate_cooling_degree_days_24c": degree_days_from_hourly(temp, 24, "cooling"),
        "climate_annual_ghi_kwh_m2": float(
            hourly["global_horizontal_radiation_wh_m2"].clip(lower=0).sum() / 1000
        ),
        "climate_annual_dni_kwh_m2": float(
            hourly["direct_normal_radiation_wh_m2"].clip(lower=0).sum() / 1000
        ),
        "climate_annual_dhi_kwh_m2": float(
            hourly["diffuse_horizontal_radiation_wh_m2"].clip(lower=0).sum() / 1000
        ),
        "climate_mean_relative_humidity_pct": float(
            hourly["relative_humidity_pct"].mean()
        ),
        "climate_mean_wind_speed_m_s": float(hourly["wind_speed_m_s"].mean()),
    }
    ground, ground_source = ground_temperatures_from_epw(epw_path)
    row["ground_temperature_source"] = ground_source
    for month, value in enumerate(ground, start=1):
        row[f"ground_temperature_month_{month:02d}_c"] = value

    if include_vertical_solar:
        row.update(vertical_solar_exposure_kwh_m2(hourly, location))
    return row
