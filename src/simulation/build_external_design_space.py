from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Replicate the frozen 1,000 paired base designs across selected "
            "external climates."
        )
    )
    p.add_argument("--source-design-space", type=Path, required=True)
    p.add_argument("--selected-climates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "external"


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source_design_space)
    selected = pd.read_csv(args.selected_climates)

    city_col = (
        "candidate_city"
        if "candidate_city" in selected.columns
        else "city"
    )
    if city_col not in selected.columns:
        raise ValueError(
            "Selected climate file must contain candidate_city or city."
        )

    required = {"base_design_id", "city", "design_id", "energy_design_id"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Source design-space missing columns: {missing}")

    # One canonical row for every base design. Because the source experiment is
    # fully paired, design variables are identical across its six city copies.
    canonical = (
        source.sort_values(["base_design_id", "city"])
        .drop_duplicates("base_design_id")
        .copy()
    )
    if canonical["base_design_id"].nunique() != 1000:
        raise ValueError(
            f"Expected 1,000 base designs, found {canonical['base_design_id'].nunique()}."
        )

    outputs = []
    for city in selected[city_col].astype(str).tolist():
        block = canonical.copy()
        city_slug = slug(city)
        block["city"] = city
        block["design_id"] = city + "_" + block["base_design_id"].astype(str)
        block["energy_design_id"] = (
            "ext_" + city_slug + "_" + block["base_design_id"].str.replace("base_", "", regex=False)
        )
        block["design_schema_version"] = "classguard_v2_external_frozen_1"
        outputs.append(block)

    result = pd.concat(outputs, ignore_index=True)

    pair_counts = result.groupby("base_design_id")["city"].nunique()
    expected = selected[city_col].nunique()
    if not (pair_counts == expected).all():
        raise RuntimeError("External design-space pairing audit failed.")
    if result["energy_design_id"].duplicated().any():
        raise RuntimeError("Duplicate external energy_design_id detected.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Rows: {len(result)}")
    print(f"Base designs: {result['base_design_id'].nunique()}")
    print(f"External climates: {result['city'].nunique()}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
