from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from classguard_v2.config import DEFAULT_ASSUMPTIONS
from classguard_v2.idf import make_idf
from classguard_v2.results import (
    extract_summary,
    failed_result,
    parse_error_counts,
)
from classguard_v2.weather import build_weather_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paired CLASS-Guard v2 EnergyPlus experiment."
    )
    parser.add_argument(
        "--energyplus-exe",
        type=Path,
        default=Path(r"C:\EnergyPlusV26-1-0\energyplus.exe"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "design_space" / "design_space_classguard_v2_6000.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "energy_results_classguard_v2_6000.csv",
    )
    parser.add_argument(
        "--weather-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "weather",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "generated_classguard_v2_6000",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "classguard_v2_6000",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-run-dirs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-ddy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-sizing-periods", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    # Assumption overrides for sensitivity runs.
    parser.add_argument("--infiltration-ach", type=float, default=None)
    parser.add_argument("--oa-lps-person", type=float, default=None)
    parser.add_argument("--oa-lps-m2", type=float, default=None)
    return parser.parse_args()


def read_existing(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    successful = {
        str(row["energy_design_id"])
        for row in rows
        if str(row.get("status", "")).lower() == "ok"
    }
    return rows, successful


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.energyplus_exe.exists():
        raise FileNotFoundError(f"EnergyPlus executable not found: {args.energyplus_exe}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input design CSV not found: {args.input}")

    assumptions = DEFAULT_ASSUMPTIONS
    if args.infiltration_ach is not None:
        assumptions = replace(
            assumptions, infiltration_air_changes_per_hour=args.infiltration_ach
        )
    if args.oa_lps_person is not None:
        assumptions = replace(
            assumptions, outdoor_air_m3_s_per_person=args.oa_lps_person / 1000
        )
    if args.oa_lps_m2 is not None:
        assumptions = replace(
            assumptions, outdoor_air_m3_s_per_m2=args.oa_lps_m2 / 1000
        )
    assumptions.validate()

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        designs = list(csv.DictReader(f))
    if args.max_runs is not None:
        designs = designs[: args.max_runs]

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results, successful = read_existing(args.output) if args.resume else ([], set())
    if not args.resume and args.output.exists():
        args.output.unlink()

    weather_contexts = {}
    for city in sorted({str(row["city"]) for row in designs}):
        context = build_weather_context(
            args.weather_root, city, include_ddy=args.include_ddy
        )
        weather_contexts[city] = context
        print(
            f"Weather: {city} | EPW={context.epw_path.name} | "
            f"ground={context.ground_temperature_source} | "
            f"DDY={list(context.design_day_names) or 'not used'}"
        )

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "energyplus_exe": str(args.energyplus_exe),
        "weather_root": str(args.weather_root),
        "include_ddy": args.include_ddy,
        "run_sizing_periods": args.run_sizing_periods,
        "assumptions": assumptions.to_dict(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    started = time.time()
    executed = 0
    for position, row in enumerate(designs, start=1):
        design_id = str(row["energy_design_id"])
        city = str(row["city"])
        if design_id in successful:
            print(f"[{position}/{len(designs)}] Skip successful: {design_id}")
            continue

        print(
            f"[{position}/{len(designs)}] Run {design_id} | "
            f"base={row.get('base_design_id')} | {city} | {row.get('orientation')}"
        )
        executed += 1
        weather = weather_contexts[city]
        idf_text, metadata = make_idf(
            row,
            weather,
            assumptions,
            run_sizing_periods=args.run_sizing_periods,
        )
        idf_path = args.model_dir / f"{design_id}.idf"
        idf_path.write_text(idf_text, encoding="utf-8")

        run_dir = args.runs_dir / design_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        completed = subprocess.run(
            [
                str(args.energyplus_exe),
                "-w",
                str(weather.epw_path),
                "-d",
                str(run_dir),
                str(idf_path),
            ],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
        )

        err_path = run_dir / "eplusout.err"
        err_text = (
            err_path.read_text(encoding="utf-8", errors="ignore")
            if err_path.exists()
            else completed.stderr
        )
        warning_count, severe_count = parse_error_counts(err_text)

        try:
            if completed.returncode != 0 or "EnergyPlus Completed Successfully" not in err_text:
                result = failed_result(
                    row,
                    metadata,
                    warning_count,
                    severe_count,
                    err_text[-2000:] or completed.stderr[-2000:],
                )
                print(f"  FAILED rc={completed.returncode}, severe={severe_count}")
            else:
                result = extract_summary(run_dir / "eplustbl.csv", row, metadata)
                result["warning_count"] = warning_count
                result["severe_count"] = severe_count
                print(
                    f"  OK total={result['total_site_energy_kwh_m2']} "
                    f"heat={result['heating_kwh_m2']} cool={result['cooling_kwh_m2']}"
                )
        except Exception as exc:
            result = failed_result(
                row,
                metadata,
                warning_count,
                severe_count,
                f"Result extraction failed: {type(exc).__name__}: {exc}",
            )
            if args.fail_fast:
                raise

        # Replace any prior failed row for the same design.
        results = [r for r in results if str(r.get("energy_design_id")) != design_id]
        results.append(result)
        write_results(args.output, results)

        if not args.keep_run_dirs and str(result.get("status")) == "ok":
            shutil.rmtree(run_dir, ignore_errors=True)

        if args.fail_fast and str(result.get("status")) != "ok":
            raise RuntimeError(f"EnergyPlus failed for {design_id}")

    elapsed = time.time() - started
    ok_count = sum(str(row.get("status", "")).lower() == "ok" for row in results)
    failed_count = len(results) - ok_count
    print(f"Saved: {args.output}")
    print(f"Executed this run: {executed}")
    print(f"Total recorded: {len(results)} | OK={ok_count} | failed={failed_count}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Completed rows remain saved.", file=sys.stderr)
        raise SystemExit(130)
