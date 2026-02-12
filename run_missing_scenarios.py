"""
Run missing scenarios and append to existing experiment results.
"""
import subprocess
import shutil
import glob
import os
from pathlib import Path

# Configuration - point to your existing experiment
EXPERIMENT_DIR = Path("experiments/2026-02-06_064047")
RESULTS_DIR = EXPERIMENT_DIR / "results"
RAW_DIR = EXPERIMENT_DIR / "raw"

RUNS = 3
BASE_SEED = 26
END_TIME = 3600
DARP_SOLVER = "simple_rerouting"  # Use simple_rerouting instead of exhaustive_search

# Define ALL expected scenarios (same as run_experiment.py)
EXPECTED_SCENARIOS = [
    # (folder_name, mode, route_file, traffic, demand_label, period, binomial)
    ("DRT3_thigh_d30", "drt", "bus_routes_5x5_veh_3.rou.xml", "high", "d30", 120, 2),
    ("DRT6_thigh_d60", "drt", "bus_routes_5x5_veh_6.rou.xml", "high", "d60", 60, 2),
    ("DRT6_thigh_d95", "drt", "bus_routes_5x5_veh_6.rou.xml", "high", "d95", 38, 2),
    ("DRT15_thigh_d60", "drt", "bus_routes_5x5_veh_15.rou.xml", "high", "d60", 60, 2),
    ("DRT15_thigh_d95", "drt", "bus_routes_5x5_veh_15.rou.xml", "high", "d95", 38, 2),
    ("DRT15_thigh_d133", "drt", "bus_routes_5x5_veh_15.rou.xml", "high", "d133", 27, 2),
    ("DRT30_thigh_d133", "drt", "bus_routes_5x5_veh.rou.xml", "high", "d133", 27, 2),
    ("DRT30_thigh_d171", "drt", "bus_routes_5x5_veh.rou.xml", "high", "d171", 21, 2),
    ("DRT30_thigh_d240", "drt", "bus_routes_5x5_veh.rou.xml", "high", "d240", 15, 4),
    ("FixedLow_thigh_d30", "fixed", "bus_routes_fixed_5x5.rou.xml", "high", "d30", 120, 2),
    ("FixedLow_thigh_d60", "fixed", "bus_routes_fixed_5x5.rou.xml", "high", "d60", 60, 2),
    ("FixedLow_thigh_d95", "fixed", "bus_routes_fixed_5x5.rou.xml", "high", "d95", 38, 2),
    ("FixedLow_thigh_d133", "fixed", "bus_routes_fixed_5x5.rou.xml", "high", "d133", 27, 2),
    ("FixedHigh_thigh_d133", "fixed", "bus_routes_fixed_5x5_high.rou.xml", "high", "d133", 27, 2),
    ("FixedHigh_thigh_d171", "fixed", "bus_routes_fixed_5x5_high.rou.xml", "high", "d171", 21, 2),
    ("FixedHigh_thigh_d240", "fixed", "bus_routes_fixed_5x5_high.rou.xml", "high", "d240", 15, 4),
    # Low traffic scenarios
    ("DRT15_tlow_d133", "drt", "bus_routes_5x5_veh_15.rou.xml", "low", "d133", 27, 2),
    ("FixedLow_tlow_d133", "fixed", "bus_routes_fixed_5x5.rou.xml", "low", "d133", 27, 2),
    ("DRT3_tlow_d30", "drt", "bus_routes_5x5_veh_3.rou.xml", "low", "d30", 120, 2),
    ("FixedLow_tlow_d30", "fixed", "bus_routes_fixed_5x5.rou.xml", "low", "d30", 120, 2),
]


def get_existing_scenarios():
    """Get list of scenario folders that already exist."""
    if not RAW_DIR.exists():
        return set()
    return {f.name for f in RAW_DIR.iterdir() if f.is_dir()}


def get_network_label(folder_name):
    """Extract network label from folder name (e.g., DRT6_thigh_d95 -> 5x5_DRT6)."""
    parts = folder_name.split("_")
    return f"5x5_{parts[0]}"


def run_scenario(folder_name, mode, route_file, traffic, demand_label, period, binomial):
    """Run a single scenario."""
    file_label = f"p{period}b{binomial}"
    route_arg = "--drt-routes" if mode == "drt" else "--fixed-routes"
    
    print(f"\n{'='*60}")
    print(f"Running: {folder_name}")
    print(f"{'='*60}")
    
    # Run simulation
    cmd = [
        "python", "bus.py",
        "--mode", mode,
        route_arg, route_file,
        "--traffic", traffic,
        "--demand-period", str(period),
        "--demand-binomial", str(binomial),
        "--runs", str(RUNS),
        "--seed", str(BASE_SEED),
        "--end-time", str(END_TIME),
        "--no-gui",
    ]
    
    # Add DARP solver for DRT mode
    if mode == "drt":
        cmd.extend(["--darp-solver", DARP_SOLVER])
    
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"  ERROR: Simulation failed with code {result.returncode}")
        return False
    
    # Analyze and append to existing CSVs
    network_label = get_network_label(folder_name)
    analyze_cmd = [
        "python", "analyze_multiple_runs.py",
        "--traffic", traffic,
        "--demand", file_label,
        "--runs", str(RUNS),
        "--seed", str(BASE_SEED),
        "--mode", mode,
        "--network", network_label,
        "--summary-csv", str(RESULTS_DIR / "experiment_results_detailed.csv"),
        "--averaged-csv", str(RESULTS_DIR / "experiment_results_averaged.csv"),
        "--no-plots",
    ]
    print(f"  Analyzing...")
    subprocess.run(analyze_cmd, check=False)
    
    # Move files to scenario folder
    scenario_folder = RAW_DIR / folder_name
    scenario_folder.mkdir(parents=True, exist_ok=True)
    
    moved = 0
    for run in range(1, RUNS + 1):
        seed = BASE_SEED + (run - 1)
        patterns = [
            f"tripinfo_seed{seed}_{mode}_t{traffic}_{file_label}_run{run}",
            f"stop_seed{seed}_{mode}_t{traffic}_{file_label}_run{run}",
            f"fcd_seed{seed}_{mode}_t{traffic}_{file_label}_run{run}",
        ]
        for pattern in patterns:
            for ext in [".xml", ".xml.gz", ".csv"]:
                src = Path(f"{pattern}{ext}")
                if src.exists():
                    shutil.move(str(src), str(scenario_folder / src.name))
                    moved += 1
    
    print(f"  Moved {moved} files to {scenario_folder}")
    return True


def main():
    existing = get_existing_scenarios()
    print(f"Found {len(existing)} existing scenario folders in {RAW_DIR}")
    
    missing = []
    for scenario in EXPECTED_SCENARIOS:
        folder_name = scenario[0]
        if folder_name not in existing:
            missing.append(scenario)
    
    if not missing:
        print("\nAll scenarios are complete!")
        return
    
    print(f"\nMissing {len(missing)} scenarios:")
    for s in missing:
        print(f"  - {s[0]}")
    
    response = input("\nRun missing scenarios? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    success = 0
    failed = 0
    for scenario in missing:
        if run_scenario(*scenario):
            success += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {success} succeeded, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()