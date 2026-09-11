from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CLASS-Guard v2 smoke run.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "energy_results_classguard_v2_6000.csv",
    )
    parser.add_argument("--expected-rows", type=int, default=12)
    parser.add_argument("--u-tolerance", type=float, default=0.03)
    parser.add_argument("--wwr-tolerance", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    checks: dict[str, object] = {}

    checks["row_count"] = int(len(df))
    checks["expected_row_count"] = args.expected_rows
    checks["all_status_ok"] = bool(df["status"].astype(str).str.lower().eq("ok").all())
    severe = pd.to_numeric(df["severe_count"], errors="coerce").fillna(0)
    checks["all_severe_zero"] = bool(severe.eq(0).all())

    required_energy = [
        "total_site_energy_kwh_m2",
        "heating_kwh_m2",
        "cooling_kwh_m2",
    ]
    finite_energy = True
    for column in required_energy:
        values = pd.to_numeric(df[column], errors="coerce")
        finite_energy &= bool(np.isfinite(values).all() and (values >= 0).all())
    checks["energy_outputs_finite_nonnegative"] = finite_energy

    wwr_error = (
        pd.to_numeric(df["requested_wwr"], errors="coerce")
        - pd.to_numeric(df["actual_wwr"], errors="coerce")
    ).abs()
    checks["max_wwr_absolute_error"] = float(wwr_error.max())
    checks["wwr_within_tolerance"] = bool(wwr_error.max() <= args.wwr_tolerance)

    u_pairs = [
        ("wall_u_value_w_m2k", "actual_wall_u_no_film_w_m2k", "wall"),
        ("roof_u_value_w_m2k", "actual_roof_u_no_film_w_m2k", "roof"),
        ("floor_u_value_w_m2k", "actual_floor_u_no_film_w_m2k", "floor"),
        ("glazing_u_value", "actual_glazing_u_w_m2k", "glazing"),
    ]
    u_ok = True
    for requested, actual, label in u_pairs:
        req = pd.to_numeric(df[requested], errors="coerce")
        act = pd.to_numeric(df[actual], errors="coerce")
        error = (req - act).abs()
        checks[f"{label}_u_max_absolute_error"] = float(error.max())
        checks[f"{label}_u_within_tolerance"] = bool(
            error.notna().all() and error.max() <= args.u_tolerance
        )
        u_ok &= bool(checks[f"{label}_u_within_tolerance"])

    base_counts = df.groupby("base_design_id")["city"].nunique()
    checks["base_designs_have_six_cities"] = bool((base_counts == 6).all())
    checks["row_count_matches"] = len(df) == args.expected_rows

    checks["passed"] = bool(
        checks["row_count_matches"]
        and checks["all_status_ok"]
        and checks["all_severe_zero"]
        and checks["energy_outputs_finite_nonnegative"]
        and checks["wwr_within_tolerance"]
        and u_ok
        and checks["base_designs_have_six_cities"]
    )

    output = args.input.with_suffix(".smoke_validation.json")
    output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"Saved: {output}")
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
