from __future__ import annotations

import argparse
import json
from pathlib import Path

from classguard_v2.design_space import DesignSpaceConfig, write_design_space


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired CLASS-Guard v2 designs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "design_space" / "design_space_classguard_v2_6000.csv",
    )
    parser.add_argument("--n-base-designs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DesignSpaceConfig(n_base_designs=args.n_base_designs, seed=args.seed)
    paired, report = write_design_space(args.output, config)
    report_path = args.output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved: {args.output}")
    print(f"Rows: {len(paired)}")
    print(f"Base designs: {paired['base_design_id'].nunique()}")
    print(f"Audit: {report_path}")
    print(f"Valid paired structure: {report['valid']}")


if __name__ == "__main__":
    main()
