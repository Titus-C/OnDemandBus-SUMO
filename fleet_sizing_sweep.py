"""
Quick fleet sizing sweep to determine appropriate DRT vehicle counts
for 1km x 1km network before running full experiment.

This script:
1. Creates route files with different fleet sizes (distributed evenly across depots)
2. Runs a DRT simulation for each fleet size under worst-case conditions
3. Analyzes completion rates and wait times from the output
4. Recommends the minimum fleet size for >=90% completion rate
5. Generates plots and CSV summary of results

Flow: fleet_sizing_sweep.py -> bus.py -> drtOnline_local.py -> traci.close()
Each step uses subprocess.run() which blocks until complete.
"""
import subprocess
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Test these fleet sizes
FLEET_SIZES = [15, 18, 21, 24, 27, 30]  # Multiples of 3 for even depot distribution

# Test conditions (use your hardest case)
TRAFFIC = "high"
DEMAND = "high"
RUNS = 1  # Single run for quick exploration
SEED = 42
END_TIME = 3600

# Template file (your 30-vehicle file)
TEMPLATE_FILE = "bus_routes_5x5_veh.rou.xml"

# Depot configuration (vehicles per depot in template)
DEPOTS = ["depot_NW", "depot_C", "depot_SE"]
VEHICLES_PER_DEPOT = 10  # In the 30-vehicle template


def create_fleet_file(num_vehicles):
    """Create route file with N vehicles distributed evenly across depots."""
    output_file = f"bus_routes_5x5_veh_{num_vehicles}.rou.xml"
    
    if Path(output_file).exists():
        print(f"  Using existing: {output_file}")
        return output_file
    
    # Read template
    with open(TEMPLATE_FILE, 'r') as f:
        content = f.read()
    
    # Find all vehicle definitions with their depot info
    # Pattern matches vehicle elements with parkingArea stop
    vehicle_pattern = r'(<vehicle[^>]+id="([^"]+)"[^>]*>.*?<stop[^>]+parkingArea="([^"]+)"[^>]*/>\s*</vehicle>)'
    vehicles = re.findall(vehicle_pattern, content, re.DOTALL)
    
    if not vehicles:
        print(f"  ERROR: Could not find DRT vehicles in {TEMPLATE_FILE}")
        return None
    
    print(f"  Found {len(vehicles)} vehicles in template")
    
    # Group vehicles by depot
    depot_vehicles = {depot: [] for depot in DEPOTS}
    for full_xml, veh_id, depot in vehicles:
        if depot in depot_vehicles:
            depot_vehicles[depot].append((full_xml, veh_id))
    
    print(f"  Vehicles per depot: {', '.join(f'{d}: {len(v)}' for d, v in depot_vehicles.items())}")
    
    # Calculate how many vehicles per depot for the target fleet size
    num_depots = len(DEPOTS)
    base_per_depot = num_vehicles // num_depots
    extra = num_vehicles % num_depots
    
    # Select vehicles to keep (distribute evenly, extras go to first depots)
    vehicles_to_keep = []
    for i, depot in enumerate(DEPOTS):
        count = base_per_depot + (1 if i < extra else 0)
        vehicles_to_keep.extend([v[0] for v in depot_vehicles[depot][:count]])
        print(f"    {depot}: keeping {count} vehicles")
    
    # Remove vehicles not in keep list
    new_content = content
    for full_xml, veh_id, depot in vehicles:
        if full_xml not in vehicles_to_keep:
            new_content = new_content.replace(full_xml, '')
    
    # Clean up multiple empty lines
    new_content = re.sub(r'\n\s*\n\s*\n', '\n\n', new_content)
    
    with open(output_file, 'w') as f:
        f.write(new_content)
    
    print(f"  Created: {output_file} with {num_vehicles} vehicles")
    return output_file


def run_single_test(num_vehicles, route_file):
    """Run one simulation and extract key metrics.
    
    Uses subprocess.run() which blocks until the simulation completes.
    bus.py calls drtOnline_local.py which properly closes TraCI when done.
    """
    cmd = [
        "python", "bus.py",
        "--mode", "drt",
        "--drt-routes", route_file,
        "--traffic", TRAFFIC,
        "--demand", DEMAND,
        "--runs", "1",
        "--seed", str(SEED),
        "--end-time", str(END_TIME),
        "--no-gui",
        "-v",
    ]
    
    print(f"  Running simulation (this may take several minutes)...")
    
    try:
        # subprocess.run blocks until the child process completes
        # No timeout - let simulation run to completion
        result = subprocess.run(cmd, check=True)
        print(f"  Simulation complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Simulation failed (fleet likely too small): exit code {e.returncode}")
        return False


def analyze_run(num_vehicles):
    """Extract metrics from the run output."""
    # Find the output file
    pattern = f"tripinfo_seed{SEED}_drt_t{TRAFFIC}_d{DEMAND}_run1.csv"
    
    if not Path(pattern).exists():
        print(f"  WARNING: Output file not found: {pattern}")
        return None
    
    try:
        df = pd.read_csv(pattern, sep=';')
        
        # Person metrics
        persons = df[df['personinfo_id'].notna()]
        all_ids = persons['personinfo_id'].unique()
        completed = persons[persons['personinfo_duration'] != -1]
        completed_ids = completed['personinfo_id'].unique()
        
        # Ride metrics (for completed riders)
        rides = completed[completed['ride_waitingTime'].notna()]
        
        metrics = {
            'fleet_size': num_vehicles,
            'total_demand': len(all_ids),
            'completed': len(completed_ids),
            'completion_rate': len(completed_ids) / len(all_ids) * 100 if len(all_ids) > 0 else 0,
            'avg_wait': rides['ride_waitingTime'].mean() if not rides.empty else None,
            'max_wait': rides['ride_waitingTime'].max() if not rides.empty else None,
            'avg_ride': rides['ride_duration'].mean() if not rides.empty else None,
        }
        
        return metrics
        
    except Exception as e:
        print(f"  Analysis error: {e}")
        return None


def cleanup_previous_outputs():
    """Remove previous output files to avoid confusion."""
    patterns = [
        f"tripinfo_seed{SEED}_drt_t{TRAFFIC}_d{DEMAND}_run1.*",
        f"stop_seed{SEED}_drt_t{TRAFFIC}_d{DEMAND}_run1.*",
        f"fcd_seed{SEED}_drt_t{TRAFFIC}_d{DEMAND}_run1.*",
    ]
    for pattern in patterns:
        for f in Path(".").glob(pattern):
            f.unlink()


def main():
    print("=" * 60)
    print("FLEET SIZING SWEEP")
    print(f"Testing: {FLEET_SIZES} vehicles")
    print(f"Conditions: Traffic={TRAFFIC}, Demand={DEMAND}")
    print("=" * 60)
    
    results = []
    
    for size in FLEET_SIZES:
        print(f"\n--- Testing {size} vehicles ---")
        
        # Clean up previous outputs
        cleanup_previous_outputs()
        
        # Create route file (delete existing to regenerate with correct distribution)
        route_file_path = Path(f"bus_routes_5x5_veh_{size}.rou.xml")
        if route_file_path.exists():
            route_file_path.unlink()  # Force regeneration
        
        # Create route file
        route_file = create_fleet_file(size)
        if not route_file:
            continue
        
        # Run simulation (blocks until complete - no overlap between runs)
        success = run_single_test(size, route_file)
        
        # Analyze (even partial results from crashed runs)
        metrics = analyze_run(size)
        if metrics:
            if not success:
                metrics['note'] = 'crashed'
            results.append(metrics)
            print(f"  === Results ===")
            print(f"  Demand: {metrics['total_demand']} passengers")
            print(f"  Completed: {metrics['completed']} ({metrics['completion_rate']:.1f}%)")
            print(f"  Avg wait: {metrics['avg_wait']:.1f}s" if metrics['avg_wait'] else "  Avg wait: N/A")
            print(f"  Max wait: {metrics['max_wait']:.1f}s" if metrics['max_wait'] else "  Max wait: N/A")
        elif not success:
            # Record failed run
            results.append({
                'fleet_size': size,
                'total_demand': None,
                'completed': 0,
                'completion_rate': 0,
                'avg_wait': None,
                'max_wait': None,
                'avg_ride': None,
                'note': 'crashed/overloaded'
            })
    
    if not results:
        print("\nNo results collected!")
        return
    
    # Summary
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("FLEET SIZING RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Filter valid results for plotting
    df_valid = df[df['completion_rate'] > 0].copy()
    
    if df_valid.empty:
        print("\nNo valid results to plot!")
        return
    
    # Find "good enough" fleet size (>90% completion, reasonable wait)
    good = df_valid[(df_valid['completion_rate'] >= 90)]
    if not good.empty:
        recommended = good.iloc[0]['fleet_size']
        print(f"\n✓ Minimum fleet for 90%+ completion: {int(recommended)} vehicles")
    else:
        print(f"\n⚠ No fleet size achieved 90% completion - may need more vehicles")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].plot(df_valid['fleet_size'], df_valid['completion_rate'], 'g-o', linewidth=2)
    axes[0].axhline(y=90, color='r', linestyle='--', label='90% target')
    axes[0].set_xlabel('Fleet Size')
    axes[0].set_ylabel('Completion Rate (%)')
    axes[0].set_title('Service Coverage')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(df_valid['fleet_size'], df_valid['avg_wait'], 'b-o', linewidth=2)
    axes[1].set_xlabel('Fleet Size')
    axes[1].set_ylabel('Avg Wait Time (s)')
    axes[1].set_title('Wait Time')
    axes[1].grid(True)
    
    axes[2].plot(df_valid['fleet_size'], df_valid['max_wait'], 'r-o', linewidth=2)
    axes[2].axhline(y=900, color='orange', linestyle='--', label='max-wait limit')
    axes[2].set_xlabel('Fleet Size')
    axes[2].set_ylabel('Max Wait Time (s)')
    axes[2].set_title('Worst Case Wait')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('fleet_sizing_results.png', dpi=150)
    plt.show()
    
    print(f"\nSaved: fleet_sizing_results.png")
    
    # Save CSV
    df.to_csv('fleet_sizing_results.csv', index=False)
    print(f"Saved: fleet_sizing_results.csv")


if __name__ == "__main__":
    main()