from __future__ import annotations

from dataclasses import dataclass

from .config import SimulationAssumptions
from .weather import WeatherContext


@dataclass(frozen=True)
class ModelMetadata:
    actual_wwr: float
    window_area_m2: float
    floor_area_m2: float
    volume_m3: float
    wall_insulation_thickness_m: float
    roof_insulation_thickness_m: float
    floor_insulation_thickness_m: float
    ground_temperature_source: str
    design_day_names: tuple[str, ...]


def to_float(value: object, default: float | None = None) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        if default is None:
            raise ValueError(f"Cannot convert to float: {value!r}")
        return default


def insulation_thickness_for_target_u(
    target_u: float, fixed_r: float, conductivity: float = 0.040
) -> float:
    if target_u <= 0:
        raise ValueError("Target U-value must be positive.")
    target_r = 1.0 / target_u
    insulation_r = target_r - fixed_r
    if insulation_r <= 0:
        raise ValueError(
            f"Target U-value {target_u} cannot be achieved with fixed R={fixed_r:.4f}."
        )
    return insulation_r * conductivity


def window_vertices(
    orientation: str,
    width: float,
    depth: float,
    height: float,
    requested_wwr: float,
) -> tuple[str, list[tuple[float, float, float]], float, float]:
    orientation = orientation.strip().upper()
    if orientation in {"S", "N"}:
        wall_dim = width
        parent_wall = "South Wall" if orientation == "S" else "North Wall"
    elif orientation in {"E", "W"}:
        wall_dim = depth
        parent_wall = "East Wall" if orientation == "E" else "West Wall"
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")

    wall_area = wall_dim * height
    requested_wwr = max(0.05, min(float(requested_wwr), 0.45))
    target_area = wall_area * requested_wwr

    sill_height = 0.80
    top_margin = 0.30
    side_margin = 0.20
    max_window_height = max(0.50, height - sill_height - top_margin)
    max_window_width = max(0.50, wall_dim - 2 * side_margin)

    win_h = max_window_height
    win_w = target_area / win_h
    if win_w > max_window_width:
        win_w = max_window_width
        win_h = min(target_area / win_w, max_window_height)

    actual_area = win_w * win_h
    actual_wwr = actual_area / wall_area
    z1 = sill_height
    z2 = sill_height + win_h
    a1 = (wall_dim - win_w) / 2
    a2 = a1 + win_w

    if orientation == "S":
        vertices = [(a1, 0, z2), (a1, 0, z1), (a2, 0, z1), (a2, 0, z2)]
    elif orientation == "N":
        vertices = [
            (a2, depth, z2),
            (a2, depth, z1),
            (a1, depth, z1),
            (a1, depth, z2),
        ]
    elif orientation == "E":
        vertices = [
            (width, a1, z2),
            (width, a1, z1),
            (width, a2, z1),
            (width, a2, z2),
        ]
    else:
        vertices = [(0, a2, z2), (0, a2, z1), (0, a1, z1), (0, a1, z2)]
    return parent_wall, vertices, actual_wwr, actual_area


def _format_vertices(vertices: list[tuple[float, float, float]]) -> str:
    lines = []
    for i, (x, y, z) in enumerate(vertices):
        ending = ";" if i == len(vertices) - 1 else ","
        lines.append(f"  {x:.4f}, {y:.4f}, {z:.4f}{ending}")
    return "\n".join(lines)


def _schedule_blocks(assumptions: SimulationAssumptions) -> str:
    start = assumptions.occupied_start_hour
    end = assumptions.occupied_end_hour
    return f"""
ScheduleTypeLimits,
  Fraction, 0, 1, CONTINUOUS;

ScheduleTypeLimits,
  Temperature, -60, 200, CONTINUOUS;

ScheduleTypeLimits,
  Any Number;

Schedule:Compact,
  Always On,
  Fraction,
  Through: 12/31,
  For: AllDays,
  Until: 24:00, 1.0;

Schedule:Compact,
  Dual Setpoint Control Type,
  Any Number,
  Through: 12/31,
  For: AllDays,
  Until: 24:00, 4;

Schedule:Compact,
  School Occupancy,
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: {start:02d}:00, 0.0,
  Until: {end:02d}:00, 1.0,
  Until: 24:00, 0.0,
  For: SummerDesignDay WinterDesignDay,
  Until: {start:02d}:00, 0.0,
  Until: {end:02d}:00, 1.0,
  Until: 24:00, 0.0,
  For: Weekends Holidays AllOtherDays,
  Until: 24:00, 0.0;

Schedule:Compact,
  School Lights,
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: {start:02d}:00, {assumptions.lighting_unoccupied_fraction:.4f},
  Until: {end:02d}:00, 1.0,
  Until: 24:00, {assumptions.lighting_unoccupied_fraction:.4f},
  For: SummerDesignDay WinterDesignDay,
  Until: {start:02d}:00, {assumptions.lighting_unoccupied_fraction:.4f},
  Until: {end:02d}:00, 1.0,
  Until: 24:00, {assumptions.lighting_unoccupied_fraction:.4f},
  For: Weekends Holidays AllOtherDays,
  Until: 24:00, {assumptions.lighting_unoccupied_fraction:.4f};

Schedule:Compact,
  School Equipment,
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: {start:02d}:00, {assumptions.equipment_unoccupied_fraction:.4f},
  Until: {end:02d}:00, 1.0,
  Until: 24:00, {assumptions.equipment_unoccupied_fraction:.4f},
  For: SummerDesignDay WinterDesignDay,
  Until: {start:02d}:00, {assumptions.equipment_unoccupied_fraction:.4f},
  Until: {end:02d}:00, 1.0,
  Until: 24:00, {assumptions.equipment_unoccupied_fraction:.4f},
  For: Weekends Holidays AllOtherDays,
  Until: 24:00, {assumptions.equipment_unoccupied_fraction:.4f};

Schedule:Compact,
  Heating Setpoint,
  Temperature,
  Through: 12/31,
  For: Weekdays,
  Until: {start:02d}:00, {assumptions.unoccupied_heating_setpoint_c:.2f},
  Until: {end:02d}:00, {assumptions.occupied_heating_setpoint_c:.2f},
  Until: 24:00, {assumptions.unoccupied_heating_setpoint_c:.2f},
  For: WinterDesignDay,
  Until: 24:00, {assumptions.occupied_heating_setpoint_c:.2f},
  For: SummerDesignDay,
  Until: 24:00, {assumptions.unoccupied_heating_setpoint_c:.2f},
  For: Weekends Holidays AllOtherDays,
  Until: 24:00, {assumptions.unoccupied_heating_setpoint_c:.2f};

Schedule:Compact,
  Cooling Setpoint,
  Temperature,
  Through: 12/31,
  For: Weekdays,
  Until: {start:02d}:00, {assumptions.unoccupied_cooling_setpoint_c:.2f},
  Until: {end:02d}:00, {assumptions.occupied_cooling_setpoint_c:.2f},
  Until: 24:00, {assumptions.unoccupied_cooling_setpoint_c:.2f},
  For: SummerDesignDay,
  Until: 24:00, {assumptions.occupied_cooling_setpoint_c:.2f},
  For: WinterDesignDay,
  Until: 24:00, {assumptions.unoccupied_cooling_setpoint_c:.2f},
  For: Weekends Holidays AllOtherDays,
  Until: 24:00, {assumptions.unoccupied_cooling_setpoint_c:.2f};

Schedule:Compact,
  Activity Schedule,
  Any Number,
  Through: 12/31,
  For: AllDays,
  Until: 24:00, {assumptions.activity_w_per_person:.2f};
""".strip()


def make_idf(
    row: dict[str, object],
    weather: WeatherContext,
    assumptions: SimulationAssumptions,
    *,
    run_sizing_periods: bool = False,
) -> tuple[str, ModelMetadata]:
    assumptions.validate()
    design_id = str(row["energy_design_id"])
    width = to_float(row["room_width_m"])
    depth = to_float(row["room_depth_m"])
    height = to_float(row["room_height_m"])
    area = width * depth
    volume = area * height
    orientation = str(row["orientation"]).strip().upper()
    requested_wwr = to_float(row["window_wall_ratio"])

    wall_u = to_float(row.get("wall_u_value_w_m2k"), 0.50)
    roof_u = to_float(row.get("roof_u_value_w_m2k"), 0.35)
    floor_u = to_float(row.get("floor_u_value_w_m2k"), 0.45)
    glazing_u = to_float(row.get("glazing_u_value"), 1.60)
    shgc = to_float(row.get("shgc"), 0.55)
    vlt = to_float(row.get("vlt"), 0.70)

    gypsum_r = 0.013 / 0.160
    brick_r = 0.200 / 0.720
    concrete_r = 0.200 / 1.400
    wall_ins_t = insulation_thickness_for_target_u(wall_u, gypsum_r + brick_r)
    roof_ins_t = insulation_thickness_for_target_u(roof_u, gypsum_r + concrete_r)
    floor_ins_t = insulation_thickness_for_target_u(floor_u, concrete_r)

    parent_wall, window_vertices_list, actual_wwr, window_area = window_vertices(
        orientation, width, depth, height, requested_wwr
    )
    ground = ", ".join(f"{value:.3f}" for value in weather.ground_temperatures_c)
    design_days = weather.design_day_idf.strip()
    sizing_enabled = bool(run_sizing_periods and design_days)
    design_day_section = f"\n{design_days}\n" if design_days else "\n"

    simulation_control = (
        "Yes, No, No, Yes, Yes, No" if sizing_enabled else "No, No, No, No, Yes, No"
    )

    idf = f"""Version, 26.1;

SimulationControl,
  {simulation_control};

Building,
  Classroom_{design_id},
  0.0,
  {assumptions.terrain},
  0.04,
  0.40,
  FullExterior,
  {assumptions.max_warmup_days},
  {assumptions.min_warmup_days};

Timestep, {assumptions.timestep_per_hour};
{design_day_section}
RunPeriod,
  Annual,
  1,
  1,
  ,
  12,
  31,
  ,
  Tuesday,
  Yes,
  Yes,
  No,
  Yes,
  Yes;

Site:GroundTemperature:BuildingSurface,
  {ground};

GlobalGeometryRules,
  UpperLeftCorner,
  CounterClockWise,
  World;

{_schedule_blocks(assumptions)}

Material,
  Generic Brick,
  MediumRough,
  0.200,
  0.720,
  1800,
  840,
  0.90,
  0.60,
  0.60;

Material,
  Wall Insulation,
  MediumRough,
  {wall_ins_t:.6f},
  0.040,
  30,
  1400,
  0.90,
  0.50,
  0.50;

Material,
  Roof Insulation,
  MediumRough,
  {roof_ins_t:.6f},
  0.040,
  30,
  1400,
  0.90,
  0.50,
  0.50;

Material,
  Floor Insulation,
  MediumRough,
  {floor_ins_t:.6f},
  0.040,
  30,
  1400,
  0.90,
  0.50,
  0.50;

Material,
  Generic Concrete,
  MediumRough,
  0.200,
  1.400,
  2200,
  1000,
  0.90,
  0.65,
  0.65;

Material,
  Generic Gypsum Board,
  Smooth,
  0.013,
  0.160,
  800,
  1090,
  0.90,
  0.50,
  0.50;

WindowMaterial:SimpleGlazingSystem,
  Parametric Glazing,
  {glazing_u:.4f},
  {shgc:.4f},
  {vlt:.4f};

Construction,
  Exterior Wall V2,
  Generic Gypsum Board,
  Wall Insulation,
  Generic Brick;

Construction,
  Roof V2,
  Generic Gypsum Board,
  Roof Insulation,
  Generic Concrete;

Construction,
  Floor V2,
  Floor Insulation,
  Generic Concrete;

Construction,
  Window V2,
  Parametric Glazing;

Zone,
  Classroom_Zone,
  0,
  0, 0, 0,
  1,
  1,
  {height:.4f},
  {volume:.4f};

BuildingSurface:Detailed,
  South Wall,
  Wall,
  Exterior Wall V2,
  Classroom_Zone,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0.5,
  4,
  0, 0, {height:.4f},
  0, 0, 0,
  {width:.4f}, 0, 0,
  {width:.4f}, 0, {height:.4f};

BuildingSurface:Detailed,
  East Wall,
  Wall,
  Exterior Wall V2,
  Classroom_Zone,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0.5,
  4,
  {width:.4f}, 0, {height:.4f},
  {width:.4f}, 0, 0,
  {width:.4f}, {depth:.4f}, 0,
  {width:.4f}, {depth:.4f}, {height:.4f};

BuildingSurface:Detailed,
  North Wall,
  Wall,
  Exterior Wall V2,
  Classroom_Zone,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0.5,
  4,
  {width:.4f}, {depth:.4f}, {height:.4f},
  {width:.4f}, {depth:.4f}, 0,
  0, {depth:.4f}, 0,
  0, {depth:.4f}, {height:.4f};

BuildingSurface:Detailed,
  West Wall,
  Wall,
  Exterior Wall V2,
  Classroom_Zone,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0.5,
  4,
  0, {depth:.4f}, {height:.4f},
  0, {depth:.4f}, 0,
  0, 0, 0,
  0, 0, {height:.4f};

BuildingSurface:Detailed,
  Roof,
  Roof,
  Roof V2,
  Classroom_Zone,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0,
  4,
  0, {depth:.4f}, {height:.4f},
  0, 0, {height:.4f},
  {width:.4f}, 0, {height:.4f},
  {width:.4f}, {depth:.4f}, {height:.4f};

BuildingSurface:Detailed,
  Floor,
  Floor,
  Floor V2,
  Classroom_Zone,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  1,
  4,
  0, 0, 0,
  0, {depth:.4f}, 0,
  {width:.4f}, {depth:.4f}, 0,
  {width:.4f}, 0, 0;

FenestrationSurface:Detailed,
  Main Window,
  Window,
  Window V2,
  {parent_wall},
  ,
  0.5,
  ,
  1.0,
  4,
{_format_vertices(window_vertices_list)}

People,
  Classroom People,
  Classroom_Zone,
  School Occupancy,
  People/Area,
  ,
  {assumptions.people_per_m2:.6f},
  ,
  0.3,
  autocalculate,
  Activity Schedule;

Lights,
  Classroom Lights,
  Classroom_Zone,
  School Lights,
  Watts/Area,
  ,
  {assumptions.lighting_w_per_m2:.4f},
  ,
  0.0,
  0.50,
  0.20;

ElectricEquipment,
  Classroom Equipment,
  Classroom_Zone,
  School Equipment,
  Watts/Area,
  ,
  {assumptions.equipment_w_per_m2:.4f},
  ,
  0.0,
  0.50,
  0.0;

ZoneInfiltration:DesignFlowRate,
  Classroom Infiltration,
  Classroom_Zone,
  Always On,
  AirChanges/Hour,
  ,
  ,
  ,
  {assumptions.infiltration_air_changes_per_hour:.4f},
  1.0,
  0.0,
  0.0,
  0.0;

DesignSpecification:OutdoorAir,
  Classroom Outdoor Air,
  Sum,
  {assumptions.outdoor_air_m3_s_per_person:.6f},
  {assumptions.outdoor_air_m3_s_per_m2:.7f},
  0.0,
  0.0,
  School Occupancy;

ThermostatSetpoint:DualSetpoint,
  Classroom Dual Setpoint,
  Heating Setpoint,
  Cooling Setpoint;

ZoneControl:Thermostat,
  Classroom Thermostat,
  Classroom_Zone,
  Dual Setpoint Control Type,
  ThermostatSetpoint:DualSetpoint,
  Classroom Dual Setpoint;

ZoneHVAC:IdealLoadsAirSystem,
  Classroom Ideal Loads,
  Always On,
  Classroom Supply Inlet,
  ,
  ,
  50,
  13,
  0.015,
  0.009,
  NoLimit,
  ,
  ,
  NoLimit,
  ,
  ,
  ,
  ,
  ConstantSupplyHumidityRatio,
  ,
  ConstantSupplyHumidityRatio,
  Classroom Outdoor Air,
  ,
  OccupancySchedule,
  NoEconomizer,
  None,
  0,
  0;

ZoneHVAC:EquipmentList,
  Classroom Equipment List,
  SequentialLoad,
  ZoneHVAC:IdealLoadsAirSystem,
  Classroom Ideal Loads,
  1,
  1;

ZoneHVAC:EquipmentConnections,
  Classroom_Zone,
  Classroom Equipment List,
  Classroom Inlet Node List,
  ,
  Classroom Zone Air Node,
  Classroom Return Air Node;

NodeList,
  Classroom Inlet Node List,
  Classroom Supply Inlet;

Sizing:Zone,
  Classroom_Zone,
  SupplyAirTemperature,
  13.0,
  11.11,
  SupplyAirTemperature,
  50.0,
  11.11,
  0.009,
  0.015,
  ,
  1.0,
  1.0,
  DesignDay,
  0.0,
  0.0,
  0.0,
  0.0,
  DesignDay,
  0.0,
  0.0,
  0.0,
  0.0;

Output:Table:SummaryReports,
  AllSummary;

OutputControl:Table:Style,
  Comma;

Output:Variable,
  *, Zone Ideal Loads Supply Air Total Heating Energy, Hourly;

Output:Variable,
  *, Zone Ideal Loads Supply Air Total Cooling Energy, Hourly;

Output:Variable,
  *, Zone Mean Air Temperature, Hourly;

Output:Meter,
  Electricity:Facility, Monthly;

Output:Meter,
  Heating:EnergyTransfer, Monthly;

Output:Meter,
  Cooling:EnergyTransfer, Monthly;
"""

    metadata = ModelMetadata(
        actual_wwr=actual_wwr,
        window_area_m2=window_area,
        floor_area_m2=area,
        volume_m3=volume,
        wall_insulation_thickness_m=wall_ins_t,
        roof_insulation_thickness_m=roof_ins_t,
        floor_insulation_thickness_m=floor_ins_t,
        ground_temperature_source=weather.ground_temperature_source,
        design_day_names=weather.design_day_names,
    )
    return idf, metadata
