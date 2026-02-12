"""
Targeted experiment runner for DRT vs Fixed-route comparison.
Uses logical fleet-demand pairings to minimize simulation count
while maximizing insight.

Hypotheses tested:
- H1: DRT outperforms Fixed at low demand (km/pax, waiting time)
- H2: Fixed outperforms DRT at high demand (travel time, completion)
- H3: Crossover point exists around d133 (~133 pax/h)
- H4: Traffic affects DRT more than Fixed (requires TRAFFIC_LEVELS=["low","high"])
- H5: Fleet sizing has diminishing returns
- H6: DRT performance degrades faster with demand than Fixed
- H7: Optimal fleet size for low demand (DRT3 vs DRT6 vs DRT15)
- H8: At very low demand (d30), minimal DRT fleet outperforms Fixed (DRT3 vs FixedLow)
"""
import subprocess
import os
import glob
import shutil
import sys
from pathlib import Path
from datetime import datetime

# === CONFIGURATION ===
RUNS = 3  # Minimum for statistics (mean ± SE)
BASE_SEED = 26
END_TIME = 3600

SUMMARY_CSV = "experiment_results_detailed.csv"
AVERAGED_CSV = "experiment_results_averaged.csv"

# === DEMAND LOOKUP ===
# Format: label -> (period, binomial)
# file_label is what bus.py will use in output filenames: p{period}b{binomial}
DEMAND_SPECS = {
    "d30":  (120, 2),   # ~30 pax/h  -> files named "p120b2"
    "d60":  (60, 2),    # ~60 pax/h  -> files named "p60b2"
    "d95":  (38, 2),    # ~95 pax/h  -> files named "p38b2"
    "d133": (27, 2),    # ~133 pax/h -> files named "p27b2"
    "d171": (21, 2),    # ~171 pax/h -> files named "p21b2"
    "d212": (17, 2),    # ~212 pax/h -> files named "p17b2"
    "d240": (15, 4),    # ~240 pax/h -> files named "p15b4"
}

# === TARGETED SCENARIOS ===
# Format: (label, mode, route_file, [compatible_demands])
TARGETED_EXPERIMENTS = [
    # Minimal DRT fleet - H7, H8 (optimal fleet for very low demand)
    ("DRT3", "drt", "bus_routes_5x5_veh_3.rou.xml", 
     ["d30"]),
    
    # Small DRT fleet - H1, H7
    ("DRT6", "drt", "bus_routes_5x5_veh_6.rou.xml", 
     ["d60", "d95"]),
    
    # Medium DRT fleet - H1, H3, H5, H6, H7
    ("DRT15", "drt", "bus_routes_5x5_veh_15.rou.xml", 
     ["d60", "d95", "d133"]),
    
    # Large DRT fleet - H2, H3, H5, H6
    ("DRT30", "drt", "bus_routes_5x5_veh.rou.xml", 
     ["d133", "d171", "d240"]),
    
    # Fixed low frequency - H1, H3, H6, H8
    ("FixedLow", "fixed", "bus_routes_fixed_5x5.rou.xml", 
     ["d30", "d60", "d95", "d133"]),
    
    # Fixed high frequency - H2, H3, H6
    ("FixedHigh", "fixed", "bus_routes_fixed_5x5_high.rou.xml", 
     ["d133", "d171", "d240"]),
]

# === LOW TRAFFIC SCENARIOS ===
# These scenarios run at low traffic when TRAFFIC_LEVELS doesn't include "low"
# H4: Traffic affects DRT more than Fixed (DRT15 vs FixedLow at d133)
# H8: At very low demand, minimal DRT outperforms Fixed (DRT3 vs FixedLow at d30)
LOW_TRAFFIC_SCENARIOS = [
    # H4 scenarios
    ("DRT15", "drt", "bus_routes_5x5_veh_15.rou.xml", "d133"),
    ("FixedLow", "fixed", "bus_routes_fixed_5x5.rou.xml", "d133"),
    # H8 scenarios
    ("DRT3", "drt", "bus_routes_5x5_veh_3.rou.xml", "d30"),
    ("FixedLow", "fixed", "bus_routes_fixed_5x5.rou.xml", "d30"),
]

TRAFFIC_LEVELS = ["high"]
ENABLE_LOW_TRAFFIC_SCENARIOS = True  # Run H4/H8 scenarios at low traffic


def convert_xml_to_csv():
    """Convert all SUMO XML output files to CSV format.
    
    Converts tripinfo*.xml, stop*.xml, and fcd*.xml(.gz) files.
    Uses SUMO's xml2csv.py tool.
    """
    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        print("  WARNING: SUMO_HOME not set, cannot convert XML to CSV")
        return False
    
    xml2csv = Path(sumo_home) / "tools" / "xml" / "xml2csv.py"
    if not xml2csv.exists():
        print(f"  WARNING: xml2csv.py not found at {xml2csv}")
        return False
    
    # Patterns to convert (in current directory)
    patterns = [
        "tripinfo_*.xml",
        "stop_*.xml",
        "fcd_*.xml",
        "fcd_*.xml.gz",
    ]
    
    converted = 0
    for pattern in patterns:
        for xml_file in glob.glob(pattern):
            # Skip if CSV already exists and is newer
            csv_file = xml_file.replace(".xml.gz", ".csv").replace(".xml", ".csv")
            if Path(csv_file).exists():
                xml_mtime = Path(xml_file).stat().st_mtime
                csv_mtime = Path(csv_file).stat().st_mtime
                if csv_mtime >= xml_mtime:
                    continue  # CSV is up to date
            
            # Convert XML to CSV
            cmd = ["python", str(xml2csv), "-s", ";", "-o", csv_file, xml_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                converted += 1
            else:
                print(f"    WARNING: Failed to convert {xml_file}: {result.stderr[:100]}")
    
    if converted > 0:
        print(f"  Converted {converted} XML files to CSV")
    return True


def create_experiment_folder():
    """Create timestamped experiment folder structure."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = Path("experiments") / timestamp
    
    (exp_dir / "raw").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)
    
    print(f"Created experiment folder: {exp_dir}")
    return exp_dir


def get_demand_params(demand_label):
    """Convert demand label to (period, binomial, file_label).
    
    file_label is what bus.py generates in filenames: p{period}b{binomial}
    """
    if demand_label not in DEMAND_SPECS:
        raise ValueError(f"Unknown demand label: {demand_label}")
    
    period, binomial = DEMAND_SPECS[demand_label]
    file_label = f"p{period}b{binomial}"  # This is what bus.py uses!
    return period, binomial, file_label


def get_scenario_folder(exp_dir, label, traffic, demand_label):
    """Get/create folder for a specific scenario."""
    folder = exp_dir / "raw" / f"{label}_t{traffic}_{demand_label}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def move_scenario_outputs(scenario_folder, mode, traffic, file_label, num_runs, base_seed):
    """Move all output files for a scenario to its folder.
    
    file_label is the p{X}b{Y} format that bus.py actually uses.
    Moves both XML and CSV files.
    """
    moved = 0
    
    for run in range(1, num_runs + 1):
        seed = base_seed + (run - 1)
        # bus.py names files with p{period}b{binomial}, e.g., "p60b2"
        # Pattern: tripinfo_seed26_drt_thigh_p120b2_run1.csv
        base_patterns = [
            f"tripinfo_seed{seed}_{mode}_t{traffic}_{file_label}_run{run}",
            f"stop_seed{seed}_{mode}_t{traffic}_{file_label}_run{run}",
            f"fcd_seed{seed}_{mode}_t{traffic}_{file_label}_run{run}",
        ]
        
        # Move both XML and CSV versions
        for base in base_patterns:
            for ext in [".xml", ".xml.gz", ".csv"]:
                src = Path(f"{base}{ext}")
                if src.exists():
                    dst = scenario_folder / src.name
                    shutil.move(str(src), str(dst))
                    moved += 1
    
    if moved > 0:
        print(f"    Moved {moved} files to {scenario_folder}")
    return moved


def run_simulation(mode, route_file, traffic, period, binomial, runs, seed, end_time):
    """Run simulation with bus.py using numeric period/binomial."""
    route_arg = "--drt-routes" if mode == "drt" else "--fixed-routes"
    
    cmd = [
        "python", "bus.py",
        "--mode", mode,
        route_arg, route_file,
        "--traffic", traffic,
        "--demand-period", str(period),
        "--demand-binomial", str(binomial),
        "--runs", str(runs),
        "--seed", str(seed),
        "--end-time", str(end_time),
        "--no-gui",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  WARNING: Simulation returned code {result.returncode}")
    
    # Convert XML outputs to CSV for analysis
    convert_xml_to_csv()
    
    return result.returncode


def analyze_results(mode, traffic, file_label, runs, seed, network_label, results_dir):
    """Analyze results with analyze_multiple_runs.py.
    
    file_label must match what bus.py used (p{period}b{binomial}).
    """
    summary_path = results_dir / SUMMARY_CSV
    averaged_path = results_dir / AVERAGED_CSV
    
    # analyze_multiple_runs.py uses --demand to find files
    # The files are named: tripinfo_seed{X}_{mode}_t{traffic}_{file_label}_run{Y}.csv
    cmd = [
        "python", "analyze_multiple_runs.py",
        "--traffic", traffic,
        "--demand", file_label,  # Must match bus.py's output naming!
        "--runs", str(runs),
        "--seed", str(seed),
        "--mode", mode,  # Only analyze the mode we just ran
        "--summary-csv", str(summary_path),
        "--averaged-csv", str(averaged_path),
        "--network", network_label,
        "--no-plots",
    ]
    print(f"  Analyzing: {network_label}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  WARNING: Analysis returned code {result.returncode}")
    return result.returncode


def count_total_experiments():
    """Count total number of scenario combinations."""
    total = 0
    for label, mode, route_file, demands in TARGETED_EXPERIMENTS:
        for traffic in TRAFFIC_LEVELS:
            total += len(demands)
    
    if ENABLE_LOW_TRAFFIC_SCENARIOS and "low" not in TRAFFIC_LEVELS:
        total += len(LOW_TRAFFIC_SCENARIOS)
    
    return total


def print_experiment_plan():
    """Print the experiment plan before running."""
    total = count_total_experiments()
    total_sims = total * RUNS
    est_time = total_sims * 2.5 / 60
    
    print("\n" + "="*70)
    print("EXPERIMENT PLAN")
    print("="*70)
    print(f"Total scenarios: {total}")
    print(f"Runs per scenario: {RUNS}")
    print(f"Total simulations: {total_sims}")
    print(f"Estimated time: {est_time:.1f} hours")
    print("-"*70)
    
    print(f"\nMain scenarios (traffic={TRAFFIC_LEVELS}):\n")
    for label, mode, route_file, demands in TARGETED_EXPERIMENTS:
        print(f"  {label:12} ({mode:5}) @ {demands}")
    
    if ENABLE_LOW_TRAFFIC_SCENARIOS and "low" not in TRAFFIC_LEVELS:
        print("\nLow traffic scenarios (H4/H8):\n")
        for label, mode, route_file, demand in LOW_TRAFFIC_SCENARIOS:
            print(f"  {label:12} ({mode:5}) @ {demand}")
    
    print("\nHypotheses covered:")
    print("  H1: DRT vs Fixed @ low demand (d60, d95)")
    print("  H2: Fixed vs DRT @ high demand (d240)")
    print("  H3: Crossover @ d133")
    print("  H4: Traffic sensitivity @ d133" + (" [ENABLED]" if ENABLE_LOW_TRAFFIC_SCENARIOS else " [DISABLED]"))
    print("  H5: Fleet scaling (DRT15 vs DRT30 @ d133)")
    print("  H6: Demand sensitivity (slope d60→d133, d133→d240)")
    print("  H7: Optimal small fleet (DRT3 vs DRT6 vs DRT15 @ d60)")
    print("  H8: Minimal DRT vs Fixed @ very low demand (d30)" + (" [ENABLED]" if ENABLE_LOW_TRAFFIC_SCENARIOS else " [DISABLED]"))
    print("="*70 + "\n")


def main():
    print_experiment_plan()
    
    response = input("Proceed with experiment? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        sys.exit(0)
    
    exp_dir = create_experiment_folder()
    results_dir = exp_dir / "results"
    
    completed = 0
    failed = 0
    total = count_total_experiments()
    
    # === Run main experiments ===
    for label, mode, route_file, demands in TARGETED_EXPERIMENTS:
        for traffic in TRAFFIC_LEVELS:
            for demand_label in demands:
                completed += 1
                period, binomial, file_label = get_demand_params(demand_label)
                
                print(f"\n[{completed}/{total}] {label} @ {traffic} traffic, {demand_label} ({file_label})")
                
                # Run simulation
                ret = run_simulation(mode, route_file, traffic, period, binomial, 
                                    RUNS, BASE_SEED, END_TIME)
                if ret != 0:
                    failed += 1
                    print(f"  SKIPPING analysis due to simulation failure")
                    continue
                
                # Analyze BEFORE moving (files need to be in current directory for analyze_multiple_runs)
                network_label = f"5x5_{label}"
                analyze_results(mode, traffic, file_label, RUNS, BASE_SEED, network_label, results_dir)
                
                # Move outputs to scenario folder AFTER analysis
                scenario_folder = get_scenario_folder(exp_dir, label, traffic, demand_label)
                move_scenario_outputs(scenario_folder, mode, traffic, file_label, RUNS, BASE_SEED)
    
    # === Run low traffic scenarios (H4, H8) ===
    if ENABLE_LOW_TRAFFIC_SCENARIOS and "low" not in TRAFFIC_LEVELS:
        print("\n" + "-"*70)
        print("Running low traffic scenarios (H4, H8)")
        print("-"*70)
        
        for label, mode, route_file, demand_label in LOW_TRAFFIC_SCENARIOS:
            completed += 1
            traffic = "low"
            period, binomial, file_label = get_demand_params(demand_label)
            
            print(f"\n[{completed}/{total}] {label} @ {traffic} traffic, {demand_label} ({file_label})")
            
            ret = run_simulation(mode, route_file, traffic, period, binomial,
                                RUNS, BASE_SEED, END_TIME)
            if ret != 0:
                failed += 1
                print(f"  SKIPPING analysis due to simulation failure")
                continue
            
            # Analyze BEFORE moving
            network_label = f"5x5_{label}"
            analyze_results(mode, traffic, file_label, RUNS, BASE_SEED, network_label, results_dir)
            
            # Move outputs AFTER analysis
            scenario_folder = get_scenario_folder(exp_dir, label, traffic, demand_label)
            move_scenario_outputs(scenario_folder, mode, traffic, file_label, RUNS, BASE_SEED)
    
    # === Summary ===
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results folder: {exp_dir}")
    print(f"Successful: {total - failed}/{total} scenarios")
    if failed > 0:
        print(f"FAILED: {failed} scenarios")
    print(f"\nDetailed CSV: {results_dir / SUMMARY_CSV}")
    print(f"Averaged CSV: {results_dir / AVERAGED_CSV}")
    print("\nTo visualize results:")
    print(f"  python visualize_experiment.py \"{results_dir / AVERAGED_CSV}\"")


if __name__ == "__main__":
    main()