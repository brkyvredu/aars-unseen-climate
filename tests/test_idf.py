from pathlib import Path

from classguard_v2.config import DEFAULT_ASSUMPTIONS
from classguard_v2.idf import make_idf
from classguard_v2.weather import EpwLocation, WeatherContext


def test_idf_contains_v2_physics_and_operational_assumptions(tmp_path: Path):
    row = {
        "energy_design_id": "cg2_test_0001",
        "base_design_id": "base_0001",
        "city": "Bursa",
        "orientation": "S",
        "room_width_m": 7,
        "room_depth_m": 8,
        "room_height_m": 3.3,
        "window_wall_ratio": 0.4,
        "wall_u_value_w_m2k": 0.5,
        "roof_u_value_w_m2k": 0.35,
        "floor_u_value_w_m2k": 0.45,
        "glazing_u_value": 1.6,
        "shgc": 0.55,
        "vlt": 0.7,
    }
    weather = WeatherContext(
        city="Bursa",
        epw_path=tmp_path / "bursa.epw",
        location=EpwLocation("Bursa", 40.2, 29.0, 3, 100),
        ground_temperatures_c=tuple(float(x) for x in range(10, 22)),
        ground_temperature_source="test",
    )
    idf, metadata = make_idf(row, weather, DEFAULT_ASSUMPTIONS)
    assert "DesignSpecification:OutdoorAir" in idf
    assert "ZoneInfiltration:DesignFlowRate" in idf
    assert "Until: 08:00, 16.00" in idf
    assert "Until: 17:00, 20.00" in idf
    assert "10.000, 11.000, 12.000" in idf
    assert "Bursa Summer Design Day" not in idf
    assert metadata.actual_wwr > 0
