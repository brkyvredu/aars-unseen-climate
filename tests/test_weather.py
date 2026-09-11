from pathlib import Path

from classguard_v2.weather import ground_temperatures_from_epw, read_epw_header


def test_epw_ground_temperature_header_parser(tmp_path: Path):
    epw = tmp_path / "sample.epw"
    ground = [10 + i for i in range(12)]
    header = [
        "LOCATION,Sample,State,Country,Source,123,40,29,3,100",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,1,0.5,1.0,1000,1000," + ",".join(map(str, ground)),
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,",
        "COMMENTS 2,",
        "DATA PERIODS,1,1,Data,Sunday,1/1,12/31",
    ]
    # The fallback hourly row is not needed because valid ground data exist.
    epw.write_text("\n".join(header), encoding="utf-8")
    parsed, source = ground_temperatures_from_epw(epw)
    assert parsed == tuple(float(x) for x in ground)
    assert "0.5m" in source
    assert len(read_epw_header(epw)) == 8
