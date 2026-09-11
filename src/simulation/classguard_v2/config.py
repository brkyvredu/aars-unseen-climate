from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SimulationAssumptions:
    """Operational assumptions used by the v2 classroom model.

    The ventilation and infiltration values are deliberately centralized here.
    They are provisional defaults and must be cited/confirmed against the
    standard adopted in the manuscript before the final production run.
    """

    occupied_start_hour: int = 8
    occupied_end_hour: int = 17

    occupied_heating_setpoint_c: float = 20.0
    unoccupied_heating_setpoint_c: float = 16.0
    occupied_cooling_setpoint_c: float = 26.0
    unoccupied_cooling_setpoint_c: float = 30.0

    people_per_m2: float = 0.536
    activity_w_per_person: float = 120.0
    lighting_w_per_m2: float = 9.0
    equipment_w_per_m2: float = 5.0
    lighting_unoccupied_fraction: float = 0.0
    equipment_unoccupied_fraction: float = 0.05

    # Provisional classroom outdoor-air defaults. Confirm the final standard.
    outdoor_air_m3_s_per_person: float = 0.005
    outdoor_air_m3_s_per_m2: float = 0.0006
    infiltration_air_changes_per_hour: float = 0.30

    timestep_per_hour: int = 4
    terrain: str = "City"
    max_warmup_days: int = 100
    min_warmup_days: int = 6

    def validate(self) -> None:
        if not 0 <= self.occupied_start_hour < self.occupied_end_hour <= 24:
            raise ValueError("Occupied hours must satisfy 0 <= start < end <= 24.")
        if self.occupied_heating_setpoint_c >= self.occupied_cooling_setpoint_c:
            raise ValueError("Occupied heating setpoint must be below cooling setpoint.")
        if self.unoccupied_heating_setpoint_c >= self.unoccupied_cooling_setpoint_c:
            raise ValueError("Unoccupied heating setpoint must be below cooling setpoint.")
        if self.people_per_m2 <= 0:
            raise ValueError("people_per_m2 must be positive.")
        if self.outdoor_air_m3_s_per_person < 0 or self.outdoor_air_m3_s_per_m2 < 0:
            raise ValueError("Outdoor-air rates cannot be negative.")
        if self.infiltration_air_changes_per_hour < 0:
            raise ValueError("Infiltration ACH cannot be negative.")
        if self.timestep_per_hour not in {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}:
            raise ValueError("Unsupported EnergyPlus timestep_per_hour value.")

    def to_dict(self) -> dict[str, float | int | str]:
        self.validate()
        return asdict(self)


DEFAULT_ASSUMPTIONS = SimulationAssumptions()
