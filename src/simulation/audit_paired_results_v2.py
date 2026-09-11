from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classguard_v2.design_space import CITIES, validate_paired_design_space


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CLASS-Guard v2 paired results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "energy_results_classguard_v2_6000.csv",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    structure = validate_paired_design_space(df, CITIES)
    ok = df[df["status"].astype(str).str.lower().eq("ok")].copy()

    for column in [
        "total_site_energy_kwh_m2",
        "heating_kwh_m2",
        "cooling_kwh_m2",
        "lighting_kwh_m2",
        "equipment_kwh_m2",
    ]:
        ok[column] = pd.to_numeric(ok[column], errors="coerce")

    ok["energy_balance_residual_kwh_m2"] = (
        ok["total_site_energy_kwh_m2"]
        - ok["heating_kwh_m2"]
        - ok["cooling_kwh_m2"]
        - ok["lighting_kwh_m2"]
        - ok["equipment_kwh_m2"]
    )

    paired_success = ok.groupby("base_design_id")["city"].nunique()
    report = {
        "structure": structure,
        "status_counts": {
            str(k): int(v) for k, v in df["status"].value_counts(dropna=False).items()
        },
        "successful_rows": int(len(ok)),
        "fully_successful_base_designs": int((paired_success == len(CITIES)).sum()),
        "energy_balance_residual_mean": float(ok["energy_balance_residual_kwh_m2"].mean()),
        "energy_balance_residual_std": float(ok["energy_balance_residual_kwh_m2"].std()),
        "energy_balance_residual_max_abs": float(
            ok["energy_balance_residual_kwh_m2"].abs().max()
        ),
        "severe_error_rows": int((pd.to_numeric(df["severe_count"], errors="coerce") > 0).sum()),
    }
    output = args.output or args.input.with_suffix(".audit.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
