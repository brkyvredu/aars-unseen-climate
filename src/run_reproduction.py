from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "src/strict"


def run(cmd):
    print("\n>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Run the strict AARS reproduction pipeline.")
    p.add_argument(
        "--quick-reference-check",
        action="store_true",
        help="Validate the distributed reference result tables only; does not refit models.",
    )
    p.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional target-city subset for AARS refit smoke checking. Full baseline/statistics stages require all eight cities.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick_reference_check:
        run([sys.executable, ROOT / "src/verify_reference_results.py"])
        return

    aars_cmd = [sys.executable, STRICT / "evaluate_aars_strict.py"]
    if args.cities:
        aars_cmd += ["--cities", *args.cities]
        run(aars_cmd)
        print("Subset run completed. Baselines/statistics are intentionally skipped for subset mode.")
        return

    run(aars_cmd)
    run([sys.executable, STRICT / "external_baseline_benchmark_strict.py"])
    run([sys.executable, STRICT / "statistical_robustness_strict.py"])
    run([sys.executable, ROOT / "src/verify_reproduced_outputs.py"])


if __name__ == "__main__":
    main()
