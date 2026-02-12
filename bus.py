import os
import sys
import random
import argparse
import subprocess

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

import xml.etree.ElementTree as ET
import math
import heapq
import time
import socket
from typing import Dict, List, Tuple, Any

# ============================================================================
# CONFIGURATION - Edit these variables to change network/file settings
# ============================================================================
# These defaults can be overridden by CLI arguments (e.g., --net-file)

CONFIG = {
    # Network and infrastructure files
    
    # old 300x300m 4x4 network
    # 'net_file': 'bus_with_tls.net.xml',      # SUMO network file
    # 'add_file': 'bus.add.xml',               # Bus stops additional file
    # 'drt_routes': 'bus_routes.rou.xml',      # DRT vehicle definitions
    # 'fixed_routes': 'bus_routes_fixed.rou.xml',  # Fixed bus routes
    # 'min_distance': 300,                     # Min trip distance (meters)
    
    # new 1x1km 5x5 network
    'net_file': 'bus_5x5_250_with_ped_staticTLS.net.xml',      # SUMO network file
    'add_file': 'bus_5x5.add.xml',               # Bus stops additional file
    'drt_routes': 'bus_routes_5x5_veh.rou.xml',      # DRT vehicle definitions
    'fixed_routes': 'bus_routes_fixed_5x5.rou.xml',  # Fixed bus routes
    # 'fixed_routes': 'bus_routes_fixed_5x5_high.rou.xml',  # Fixed bus routes
    'min_distance': 700,                     # Min trip distance (meters)
    
    # Simulation parameters
    'end_time': 3600,                        # Simulation end time (seconds)
    'seed': 42,                              # Default random seed
    
    # DRT parameters
    'drf': 2.0,                              # Detour factor
    'max_wait': 900,                         # Max pickup wait (seconds)
    'darp_solver': 'exhaustive_search',      # DARP algorithm
    'sim_step': 10,                          # Dispatch interval in seconds (default was 30)
    
    # Demand generation
    'traffic': 'none',                       # Background traffic level
    'demand': 'none',                        # Passenger demand level
    # 'min_distance': 700,                     # Min trip distance (meters)
}

# ============================================================================
# Optional: Load config from external JSON file if it exists
# ============================================================================
def load_config_file(config_path='bus_config.json'):
    """Load configuration from JSON file, if it exists."""
    import json
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            CONFIG.update(file_config)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Could not load {config_path}: {e}")

# Try to load external config (optional)
load_config_file()

# ============================================================================
# DEMAND/TRAFFIC SCENARIOS
# ============================================================================

CAR_TRAFFIC_SCENARIOS = {
    'none':   None,
    'low':    {'period': 10,  'binomial': 2},   # ~360 cars/hour
    'medium': {'period': 5,  'binomial': 3},   # ~720 cars/hour  
    'high':   {'period': 2,  'binomial': 4},   # ~1800 cars/hour
    'very_high': {'period': 1, 'binomial': 5}, # ~3600 cars/hour
}

# Demand can be specified numerically via --demand-period and --demand-binomial
# Or use preset: 'none' to disable
# Approximate pax/hour ≈ 3600 / period * (binomial/2)

PASSENGER_DEMAND_SCENARIOS = {
    'none':   None,
    'low':    {'period': 60,  'binomial': 2},   # ~60 passengers/hour
    'medium': {'period': 30,  'binomial': 3},   # ~120 passengers/hour
    'high':   {'period': 15,  'binomial': 4},   # ~240 passengers/hour
    'very_high': {'period': 10, 'binomial': 5}, # ~360 passengers/hour
}


def resolve_demand_params(args):
    """
    Resolve demand parameters from either preset or custom values.
    Custom --demand-period/--demand-binomial override --demand preset.
    Sets args.demand_period and args.demand_binomial.
    Returns a label string for filenames.
    """
    if args.demand_period is not None:
        # Custom numeric values override preset
        args.demand_period = args.demand_period
        args.demand_binomial = args.demand_binomial if args.demand_binomial is not None else 2
        approx = int(3600 / args.demand_period * (args.demand_binomial / 2)) if args.demand_period > 0 else 0
        return f"p{args.demand_period}b{args.demand_binomial}"  # e.g., "p60b2"
    else:
        # Use preset
        scenario = PASSENGER_DEMAND_SCENARIOS.get(args.demand)
        if scenario is None:
            args.demand_period = None
            args.demand_binomial = None
            return "none"
        else:
            args.demand_period = scenario['period']
            args.demand_binomial = scenario['binomial']
            return args.demand  # e.g., "low", "high"


def generate_car_traffic(args):
    """Generate background car traffic using randomTrips.py with pre-computed routes"""
    if args.traffic == 'none':
        return None
    
    scenario = CAR_TRAFFIC_SCENARIOS[args.traffic]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    random_trips = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    
    # Output files
    trips_file = os.path.join(script_dir, f'_traffic_seed{args.seed}_{args.traffic}.trips.xml')
    output_file = os.path.join(script_dir, f'traffic_seed{args.seed}_{args.traffic}.rou.xml')
    
    if os.path.exists(output_file) and not args.regenerate:
        print(f"Using existing traffic file: {output_file}")
        return output_file
    
    print(f"Generating {args.traffic} car traffic with seed {args.seed}...")
    
    # Use args.net_file instead of hardcoded network file
    cmd = [
        sys.executable, random_trips,
        '-n', args.net_file,
        '--seed', str(args.seed),
        '--begin', '0', 
        '--end', str(args.end_time),
        '--period', str(scenario['period']),
        '--binomial', str(scenario['binomial']),
        '--min-distance', '300',
        '--fringe-factor', '5',
        '-o', trips_file,           # Intermediate trips file
        '-r', output_file,          # Final routed file (calls duarouter)
        '--validate',
    ]
    
    if args.verbose:
        print(f"Command: {' '.join(cmd)}")
    
    subprocess.run(cmd, cwd=script_dir, check=True)
    
    # Clean up intermediate trips file
    if os.path.exists(trips_file):
        os.remove(trips_file)
    
    print(f"Generated: {output_file}")
    return output_file


def generate_passenger_demand(args, mode):
    """Generate passenger trips between bus stops"""
    # Check if demand is disabled
    if args.demand_period is None or args.demand_period <= 0:
        return None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    random_trips = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    
    # Use demand_label for filename (set by resolve_demand_params)
    output_file = os.path.join(script_dir, f'passengers_seed{args.seed}_{args.demand_label}_{mode}.rou.xml')
    
    if os.path.exists(output_file) and not args.regenerate:
        print(f"Using existing: {output_file}")
        return output_file
    
    approx_pax = int(3600 / args.demand_period)
    print(f"Generating passenger demand: period={args.demand_period}, binomial={args.demand_binomial} (~{approx_pax} pax/h) for {mode} mode")
    
    cmd = [
        sys.executable, random_trips,
        '-n', args.net_file,
        '-a', args.add_file,
        '--seed', str(args.seed + 1000),
        '--begin', '0',
        '--end', str(args.end_time),
        '--period', str(args.demand_period),
        '--binomial', str(args.demand_binomial),
        '--from-stops', 'busStop',
        '--to-stops', 'busStop',
        '--min-distance', str(args.min_distance),
        '-o', output_file,
        '--validate',
    ]
    
    # Mode-specific trip type
    if mode == 'drt':
        # cmd.extend(['--personrides', 'taxi'])

        cmd.extend(['--persontrips'])
        cmd.extend(['--trip-attributes', 'modes="taxi"'])
    else:  # fixed
        cmd.extend(['--persontrips'])
        # Optionally restrict to public+walk only (no taxi):
        cmd.extend(['--trip-attributes', 'modes="public"'])
    
    subprocess.run(cmd, cwd=script_dir, check=True)
    return output_file


# ============ MULTI-RUN SUPPORT ============
def get_output_suffix(args, run_num=None):
    """Generate output file suffix based on run configuration"""
    parts = []
    if args.output_prefix:
        parts.append(args.output_prefix)
    parts.append(f"seed{args.seed}")
    parts.append(f"{args.mode}")
    parts.append(f"t{args.traffic}")
    parts.append(f"d{args.demand_label}" if not args.demand_label.startswith('p') else args.demand_label)  # Handle both preset ("low") and custom ("p60b2")
    if run_num is not None:
        parts.append(f"run{run_num}")
    return '_'.join(parts)


def rename_output_files(args, run_num):
    """Rename output files with run-specific suffix"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = get_output_suffix(args, run_num)
    
    # Files to rename
    output_files = {
        'tripinfo.output.xml': f'tripinfo_{suffix}.xml',
        'stop.output.xml': f'stop_{suffix}.xml',
        'statistics.output.xml': f'statistics_{suffix}.xml',
        'taxi_dispatch.output.xml': f'taxi_dispatch_{suffix}.xml',
        'fcd.xml.gz': f'fcd_{suffix}.xml.gz',
        'simulation.log.txt': f'simulation_{suffix}.log.txt',
    }
    
    renamed_files = {}
    for original, new_name in output_files.items():
        original_path = os.path.join(script_dir, original)
        new_path = os.path.join(script_dir, new_name)
        if os.path.exists(original_path):
            # Remove existing file with same name if it exists
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(original_path, new_path)
            renamed_files[original] = new_name
            if args.verbose:
                print(f"  Renamed: {original} -> {new_name}")
    
    return renamed_files


def cleanup_generated_files(args):
    """Remove all generated seed files (traffic and passenger demand files only)"""
    import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Patterns for generated INPUT files only (not outputs)
    patterns = [
        'traffic_seed*.rou.xml',           # Traffic route files
        '_traffic_seed*.trips.xml',        # Traffic trip files (intermediate)
        'passengers_seed*.rou.xml',        # Passenger demand files
        '_passengers_seed*.trips.xml',     # Passenger trip files (intermediate)
    ]
    
    # Optionally include common SUMO output/log files
    if getattr(args, 'cleanup_outputs', False):
        patterns.extend([
            'tripinfo_*.xml', 'tripinfo.output.xml', 'tripinfo_*.xml.gz',
            'stop_*.xml', 'stop.output.xml',
            'statistics_*.xml', 'statistics.output.xml',
            'taxi_dispatch_*.xml', 'taxi_dispatch.output.xml',
            'fcd_*.xml.gz', 'fcd.xml.gz',
            'simulation_*.log.txt', 'simulation.log.txt',
            'sumo_error.log',
            'tripinfo_drt.xml',
            'simulation_*.log', 'simulation.log'
        ])

    removed_count = 0
    for pattern in patterns:
        full_pattern = os.path.join(script_dir, pattern)
        files = glob.glob(full_pattern)
        for f in files:
            try:
                os.remove(f)
                if args.verbose:
                    print(f"  Removed: {os.path.basename(f)}")
                removed_count += 1
            except OSError as e:
                print(f"  Error removing {f}: {e}")
    
    print(f"Cleanup complete: {removed_count} files removed.")
    return removed_count


def run_multiple_simulations(args):
    """Run multiple simulations with different seeds"""
    results = []
    base_seed = args.seed
    
    # Resolve demand params once (they don't change per seed)
    if not hasattr(args, 'demand_label'):
        args.demand_label = resolve_demand_params(args)
    
    print("=" * 60)
    print(f"MULTI-RUN SIMULATION: {args.runs} runs starting from seed {base_seed}")
    print(f"Mode: {args.mode.upper()}, Traffic: {args.traffic}, Demand: {args.demand_label}")
    print("=" * 60)
    
    for run_num in range(1, args.runs + 1):
        # Calculate seed for this run
        current_seed = base_seed + (run_num - 1)
        args.seed = current_seed
        
        print(f"\n{'#' * 60}")
        print(f"# RUN {run_num}/{args.runs} - Seed: {current_seed}")
        print(f"{'#' * 60}")
        
        # Generate traffic and demand files for this seed
        traffic_file = generate_car_traffic(args)
        demand_file = generate_passenger_demand(args, args.mode)
        
        args.traffic_file = traffic_file
        args.demand_file = demand_file
        
        # Run the simulation
        if args.mode == 'drt':
            return_code = run_drt_mode(args)
        else:
            return_code = run_fixed_mode(args)
        
        # Rename output files with run-specific names
        renamed_files = rename_output_files(args, run_num)
        
        results.append({
            'run': run_num,
            'seed': current_seed,
            'return_code': return_code,
            'output_files': renamed_files,
        })
        
        print(f"\nRun {run_num} completed with return code: {return_code}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MULTI-RUN SUMMARY")
    print("=" * 60)
    print(f"{'Run':<6} {'Seed':<10} {'Status':<12} {'Output Files'}")
    print("-" * 60)
    for r in results:
        status = "SUCCESS" if r['return_code'] == 0 else f"FAILED({r['return_code']})"
        files = ', '.join(r['output_files'].values()) if r['output_files'] else 'None'
        # Truncate files list if too long
        if len(files) > 40:
            files = files[:37] + '...'
        print(f"{r['run']:<6} {r['seed']:<10} {status:<12} {files}")
    
    print("=" * 60)
    
    # Return 0 if all runs succeeded, otherwise return 1
    failed_runs = [r for r in results if r['return_code'] != 0]
    if failed_runs:
        print(f"\nWARNING: {len(failed_runs)} run(s) failed!")
        return 1
    else:
        print(f"\nAll {args.runs} runs completed successfully!")
        return 0


# ============ ARGUMENT PARSING ============
def parse_args():
    parser = argparse.ArgumentParser(description='Bus simulation runner')
    
    # Add cleanup argument near the top with other flags
    parser.add_argument('--cleanup', action='store_true',
                         help='Remove all generated seed files (traffic/passenger) and exit')
    parser.add_argument('--cleanup-outputs', action='store_true',
                        help='Also remove simulation output files (tripinfo, stop, statistics, logs, fcd)')
    
    parser.add_argument('--mode', choices=['fixed', 'drt'], default='fixed',
                        help='Simulation mode: "fixed" for fixed routes (default), "drt" for on-demand')
    parser.add_argument('--gui', action='store_true', default=True,
                        help='Run with SUMO GUI (default: True)')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run without GUI')
    
    # DRT-specific options (using CONFIG defaults)
    parser.add_argument('--drf', type=float, default=CONFIG['drf'],
                        help=f"DRT detour factor (default: {CONFIG['drf']})")
    parser.add_argument('--max-wait', type=int, default=CONFIG['max_wait'],
                        help=f"Max waiting time for pickup in seconds (default: {CONFIG['max_wait']})")
    parser.add_argument('--end-time', type=int, default=CONFIG['end_time'],
                        help=f"Simulation end time in seconds (default: {CONFIG['end_time']})")
    parser.add_argument('--darp-solver', default=CONFIG['darp_solver'],
                        choices=['exhaustive_search', 'simple_rerouting'],
                        help=f"DARP solver method (default: {CONFIG['darp_solver']})")
    parser.add_argument('--sim-step', type=int, default=CONFIG['sim_step'],
                        help=f"Dispatch interval in seconds (default: {CONFIG['sim_step']})")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    # Demand generation options (using CONFIG defaults)
    parser.add_argument('--seed', type=int, default=CONFIG['seed'],
                        help=f"Random seed for reproducible demand (default: {CONFIG['seed']})")
    parser.add_argument('--traffic', choices=['none', 'low', 'medium', 'high', 'very_high'], 
                        default=CONFIG['traffic'],
                        help=f"Background car traffic level (default: {CONFIG['traffic']})")
    
    # Demand: can use preset OR custom numeric values
    parser.add_argument('--demand', choices=['none', 'low', 'medium', 'high', 'very_high'], 
                        default=CONFIG['demand'],
                        help=f"Passenger demand preset (default: {CONFIG['demand']}). Overridden by --demand-period if specified.")
    parser.add_argument('--demand-period', type=int, default=None,
                        help='Custom passenger arrival period in seconds (overrides --demand preset). Lower = more demand.')
    parser.add_argument('--demand-binomial', type=int, default=None,
                        help='Custom binomial parameter for demand (default: 2 if --demand-period set). Higher = more variance.')
    
    parser.add_argument('--min-distance', type=int, default=CONFIG['min_distance'],
                        help=f"Minimum trip distance in meters (default: {CONFIG['min_distance']})")
    parser.add_argument('--regenerate', action='store_true',
                        help='Force regenerate demand files even if they exist')
    
    # Multi-run options
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of simulation runs with different seeds (default: 1)')
    parser.add_argument('--output-prefix', type=str, default='',
                        help='Prefix for output files (default: empty)')
    
    # Network and file configuration (using CONFIG defaults)
    parser.add_argument('--net-file', type=str, default=CONFIG['net_file'],
                        help=f"SUMO network file (default: {CONFIG['net_file']})")
    parser.add_argument('--add-file', type=str, default=CONFIG['add_file'],
                        help=f"Additional file with bus stops (default: {CONFIG['add_file']})")
    parser.add_argument('--drt-routes', type=str, default=CONFIG['drt_routes'],
                        help=f"DRT vehicle routes file (default: {CONFIG['drt_routes']})")
    parser.add_argument('--fixed-routes', type=str, default=CONFIG['fixed_routes'],
                        help=f"Fixed bus routes file (default: {CONFIG['fixed_routes']})")
    # parser.add_argument('--fixed-config', type=str, default=CONFIG['fixed_config'],
    #                     help=f"Fixed mode SUMO config (default: {CONFIG['fixed_config']})")
    
    # Optional: config file argument
    parser.add_argument('--config', type=str, default=None,
                        help='Load configuration from JSON file')
    
    return parser.parse_args()


def run_drt_mode(args):
    """Run on-demand DRT simulation using drtOnline.py via subprocess"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # drt_online_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'drt', 'drtOnline.py')
    drt_online_path = os.path.join(script_dir, 'drtOnline_local.py')
    if not os.path.exists(drt_online_path):
        drt_online_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'drt', 'drtOnline.py')
    
    # CHANGED: Use args.drt_routes instead of hardcoded 'bus_routes.rou.xml'
    route_files = [args.drt_routes]  # DRT vehicles only
    
    # REMOVED: No longer include legacy routes.rou.drt.xml at all
    # That file contains old test passengers and is not needed for experiments
    
    if args.traffic_file:
        route_files.append(os.path.basename(args.traffic_file))
    if args.demand_file:
        route_files.append(os.path.basename(args.demand_file))
    
    # CHANGED: Use args.net_file and args.add_file in temp config
    temp_config = os.path.join(script_dir, '_temp_drt.sumocfg')
    with open(temp_config, 'w') as f:
        f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration>
    <input>
        <net-file value="{args.net_file}"/>
        <route-files value="{','.join(route_files)}"/>
        <additional-files value="{args.add_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{args.end_time}"/>
    </time>
    <routing>
        <persontrip.transfer.walk-taxi value="ptStops"/>
        <persontrip.transfer.taxi-walk value="ptStops"/>
    </routing>
    <processing>
        <device.taxi.dispatch-algorithm value="traci"/>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <verbose value="true"/>
        <log value="simulation.log.txt"/>
        <error-log value="sumo_error.log"/>
    </report>
    <output>
        <tripinfo-output.write-unfinished value="true"/>
        <tripinfo-output value="tripinfo.output.xml"/>
        <stop-output value="stop.output.xml"/>
        <statistic-output value="statistics.output.xml"/>
    </output>
</sumoConfiguration>
''')
    
    # Build the command using the temp config
    cmd = [
        sys.executable,
        drt_online_path,
        '--sumocfg', temp_config,
        '-s', 'sumo-gui' if (args.gui and not args.no_gui) else 'sumo',
        '--drf', str(args.drf),
        '--max-wait', str(args.max_wait),
        '--end-time', str(args.end_time),
        '--darp-solver', args.darp_solver,
        '--sim-step', str(args.sim_step),
        '-o', 'tripinfo_drt.xml',
    ]
    
    if args.verbose:
        cmd.append('--verbose')
    
    print("=" * 50)
    print("Running DRT (On-Demand) mode")
    print("=" * 50)
    print(f"Network: {args.net_file}")
    print(f"Bus stops: {args.add_file}")
    print(f"DRT routes: {args.drt_routes}")
    print(f"Traffic: {args.traffic} -> {args.traffic_file or 'none'}")
    print(f"Demand: {args.demand} -> {args.demand_file or 'none'}")
    print(f"DRT params: max-wait={args.max_wait}s, sim-step={args.sim_step}s")
    print(f"Route files: {route_files}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    result = subprocess.run(cmd, cwd=script_dir)
    
    # Clean up temp config
    # if os.path.exists(temp_config):
    #     os.remove(temp_config)
    
    return result.returncode


def run_fixed_mode(args):
    """Run fixed-route bus simulation (original bus.py logic)"""
    
    print("=" * 50)
    print("Running FIXED route mode")
    print("=" * 50)
    print(f"Network: {args.net_file}")
    print(f"Bus stops: {args.add_file}")
    print(f"Fixed routes: {args.fixed_routes}")
    print(f"Traffic: {args.traffic} -> {args.traffic_file or 'none'}")
    print(f"Demand: {args.demand} -> {args.demand_file or 'none'}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sumo_binary = 'sumo-gui' if (args.gui and not args.no_gui) else 'sumo'
    
    # Build route files list
    route_files = [args.fixed_routes]
    if args.traffic_file:
        route_files.append(os.path.basename(args.traffic_file))
    if args.demand_file:
        route_files.append(os.path.basename(args.demand_file))
    
    # Generate dynamic config (same approach as DRT mode)
    temp_config = os.path.join(script_dir, '_temp_fixed.sumocfg')
    with open(temp_config, 'w') as f:
        f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration>
    <input>
        <net-file value="{args.net_file}"/>
        <route-files value="{','.join(route_files)}"/>
        <additional-files value="{args.add_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{args.end_time}"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <device.fcd.probability value="0"/>
    </processing>
    <report>
        <verbose value="true"/>
        <log value="simulation.log.txt"/>
        <error-log value="sumo_error.log"/>
    </report>
    <output>
        <tripinfo-output.write-unfinished value="true"/>
        <tripinfo-output value="tripinfo.output.xml"/>
        <stop-output value="stop.output.xml"/>
        <statistic-output value="statistics.output.xml"/>
        <fcd-output value="fcd.xml.gz"/>
        <fcd-output.distance value="true"/>
    </output>
</sumoConfiguration>
''')
    
    Sumo_config = [sumo_binary, '-c', temp_config]
    
    traci.start(Sumo_config)
    
    step = 0

    # ============ LEGACY PERSON GENERATION (COMMENTED OUT) ============
    # These functions were used for dynamic person generation during simulation.
    # Now using pre-generated demand files for fair comparison with DRT mode.
    
    # def addPerson(pid, startEdgeID, destinationEdgeID):
    #     print("startEdgeID:", startEdgeID)
    #
    #     # laneNumber = traci.edge.getLaneNumber(startEdgeID)
    #     # laneIDs =[]
    #     # for i in range(laneNumber):
    #     #     laneIDs.append(e + "_" + str(i))
    #     # # find the longest lane
    #     # min_len = 999999
    #     # for lane in laneIDs:
    #     #     length = traci.lane.getLength(lane)
    #     #     if length < min_len:
    #     #         min_len = length
    #     # print("min_len:", min_len)
    #
    #     startEdge = network.getEdge(startEdgeID)
    #     startEdgeLength = startEdge.getLength()
    #     print("startEdgeLength:", startEdgeLength)
    #     # startPos = random.uniform(0, min_len)
    #     startPos = random.uniform(0, startEdgeLength)
    #     print("startPos", startPos)
    #     traci.person.add(personID=pid, edgeID = startEdgeID, pos = startPos)
    #     walkingRoute = traci.simulation.findIntermodalRoute(fromEdge=startEdgeID, toEdge=destinationEdgeID, modes="public", departPos=startPos)
    #     print(f"{pid} route: {walkingRoute}")
    #
    #     for stage in walkingRoute:
    #         traci.person.appendStage(pid, stage)
    #     # print(f"Max speed of person {pid}: {traci.person.getMaxSpeed(pid)}")
        
    # def addPersonAtBusStop(pid):
    #     busStops = traci.busstop.getIDList()
    #     # print(f"Bus Stop List: ", busStops)
    #
    #     startBusStop = random.choice(busStops)
    #     print("Starting Bus Stop", startBusStop)
    #     # startBSIndex = busStops.index(startBusStop)
    #     destinationBusStop = random.choice(busStops)
    #     while destinationBusStop == startBusStop:
    #         destinationBusStop = random.choice(busStops)
    #
    #     startBSLaneID = traci.busstop.getLaneID(startBusStop)
    #     print("LaneID:", startBSLaneID)
    #     startBSEdgeID = startBSLaneID.split("_")[0]
    #     print("EdgeID:", startBSEdgeID)
    #     destLaneID = traci.busstop.getLaneID(destinationBusStop)
    #     destEdgeID = destLaneID.split("_")[0]
    #
    #     startBSStartPos = traci.busstop.getStartPos(startBusStop)
    #     startBSEndPos = traci.busstop.getEndPos(startBusStop)
    #     spawnPos = (startBSStartPos + startBSEndPos) / 2
    #
    #     traci.person.add(pid, startBSEdgeID, spawnPos)
    #     # traci.person.appendWalkingStage(pid, edges=[startBSEdgeID], stopID=startBusStop, arrivalPos=spawnPos)
    #     # traci.person.appendWaitingStage(pid, duration=9999, description="waiting", stopID=startBusStop)
    #     # traci.person.appendDrivingStage(pid, toEdge=destEdgeID, lines="taxi", stopID=startBusStop)
    #     walkingRoute = traci.simulation.findIntermodalRoute(fromEdge=startBSEdgeID, toEdge=destEdgeID, modes="public", departPos=spawnPos)
    #     print(f"{pid} route: {walkingRoute}")
    #
    #     for stage in walkingRoute:
    #         traci.person.appendStage(pid, stage)

    # def randomEdges(edgeIDs):
    #     startEdgeID = random.choice(edgeIDs)
    #     startEdgeIndex = edgeIDs.index(startEdgeID)
    #     edgeIDs.pop(startEdgeIndex)
    #     destinationEdgeID = random.choice(edgeIDs)
    #     return startEdgeID, destinationEdgeID

    def displayStats():
        perIDlist = traci.person.getIDList()
        perCount = traci.person.getIDCount()
        drivingCount = 0
        walkingCount = 0

        for pid in perIDlist:
            pNstage = traci.person.getStage(pid).type
            if pNstage == 3:
                drivingCount += 1
            if pNstage == 2:
                walkingCount += 1

            pRoute = traci.person.getEdges(pid)
            # print("ID: ", pid, "Next Stage: ", pNstage)
            # print("Edges: ", pRoute)
        print("Total: ", perCount, "Driving: ", drivingCount, "Walking: ", walkingCount)
        # if perIDlist and perIDlist[0] is not None:
        #     print(f"Max speed of person {perIDlist[0]}: {traci.person.getMaxSpeed(perIDlist[0])}")

    # CHANGED: Use args.add_file instead of hardcoded 'bus.add.xml'
    def parse_bus_stops():
        tree = ET.parse(args.add_file)
        root = tree.getroot()

        print("Bus Stops XML File:")
        print(root.tag)
        
        for bs in root.findall('busStop'):
            sid = bs.get('id')
            lane_id = bs.get('lane')
            startPos = float(bs.get('startPos', '0'))
            endPos = float(bs.get('endPos', '0'))
            print(f"Stop ID: {sid}, lane_id: {lane_id}, startPos: {startPos}, endPos: {endPos}")

    # CHANGED: Use args.drt_routes instead of hardcoded 'bus_routes.rou.xml'
    def parse_bus_routes():
        tree = ET.parse(args.drt_routes)
        root = tree.getroot()

        print("Bus Routes XML File: ")
        print(root.tag)
        print(root.attrib)
        for child in root:
            print(child.tag, child.attrib)

        for route in root.findall('route'):
            route_id = route.get('id')
            edges = route.get('edges')  # string of edge IDs
            edge_list = edges.split()  # list of edge IDs
            print("Route", route_id, "edges:", edge_list)
            for stop in route.findall('stop'):
                bus_stop = stop.get('busStop')
                duration = stop.get('duration')
                # until = stop.get('until')
                # print("  Stop:", bus_stop, "duration:", duration, "until:", until)
                print("  Stop:", bus_stop, "duration:", duration)

    # ============ LEGACY EDGE PROCESSING (COMMENTED OUT) ============
    # This was used for dynamic person generation
    # edgeIDs = traci.edge.getIDList()
    # normalEdgeIDs = [e for e in edgeIDs if not e.startswith(":")]
    # startEdgeID, destinationEdgeID = randomEdges(normalEdgeIDs)

    # edgeConnectionDict = {}
    # for e in normalEdgeIDs:
    #     if e not in edgeConnectionDict.keys():
    #         edgeConnectionDict[e] = {}
    #         laneNumber = traci.edge.getLaneNumber(e)
    #         laneIDs =[]
    #         for i in range(laneNumber):
    #             laneIDs.append(e + "_" + str(i))
    #         nextEdgeSet = set()
    #         for lane in laneIDs:
    #             nextLaneTuple = traci.lane.getLinks(lane)
    #             for l in nextLaneTuple:
    #                 next = l[0]
    #                 if not next.startswith(":"):
    #                     nextEdgeSet.add(next.split("_")[0])
    #                     # nextEdgeSet.add((next.split("_")[0], "Walk", traci.lane.getLength(next)/traci.lane.getMaxSpeed(next)))    
    #                     for nextEdge in nextEdgeSet:
    #                         walkTime = traci.lane.getLength(next)/1.38889
    #                         rideTime = traci.lane.getLength(next)/min(traci.lane.getMaxSpeed(next), 23.61111111111111)
    #                         edgeConnectionDict[e][nextEdge] = (walkTime, rideTime)
    #         # print(nextEdgeSet)
    #         # print(e, laneIDs)
    #     # edgeConnectionDict[e] = list(nextEdgeSet)
    # print(edgeConnectionDict)

    # ============ MAIN SIMULATION LOOP ============
    # passengers come from pre-generated demand files
    
    # stepCounter = 0  # Legacy: used for periodic person generation
    while traci.simulation.getMinExpectedNumber() > 0:
        # Respect end time - stop simulation when reached
        if traci.simulation.getTime() >= args.end_time:
            print(f"Reached end time: {args.end_time}")
            break
            
        traci.simulationStep()
        step += 1
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step: {step}")
            displayStats()

        # ============ LEGACY DYNAMIC PERSON GENERATION (COMMENTED OUT) ============
        # pid = "p"+ str(step)
        # edgeIDs = traci.edge.getIDList()
        # normalEdgeIDs = [e for e in edgeIDs if not e.startswith(":")]
        # startEdgeID, destinationEdgeID = randomEdges(normalEdgeIDs)
        # if step == 1:
        #     # addPerson(pid, startEdgeID, destinationEdgeID)
        #     addPersonAtBusStop(pid)
        #
        # if stepCounter == 20:
        #     # addPerson(pid, startEdgeID, destinationEdgeID)
        #     addPersonAtBusStop(pid)
        #     stepCounter = 0
        #
        # if step >= 3600:
        #     stepCounter = 0
        # 
        # if step == 3600:
        #     break
        #
        # stepCounter += 1
        # displayStats()

    # Final statistics
    print("=" * 50)
    print("Simulation Complete")
    print(f"Total steps: {step}")
    displayStats()

    # traci.close()
    # return 0

    # Wait for remaining persons to leave before closing (safety timeout)
    WAIT_PERSONS_MAX = 1200  # seconds
    max_wait_steps = max(1, int(WAIT_PERSONS_MAX))
    wait_steps = 0
    # Use simulation time queries for logging; each traci.simulationStep() advances 1s by default
    while traci.person.getIDCount() > 0 and wait_steps < max_wait_steps:
        if args.verbose:
            print(f"Waiting for {traci.person.getIDCount()} person(s) to leave (sim time {traci.simulation.getTime()})")
        traci.simulationStep()
        wait_steps += 1

    if traci.person.getIDCount() > 0:
        print(f"Warning: {traci.person.getIDCount()} person(s) still present when closing TraCI (timeout reached).")
    else:
        if args.verbose:
            print("No persons left in simulation.")

    traci.close()
    return 0


# ============ MAIN ENTRY POINT ============
if __name__ == "__main__":
    args = parse_args()
    
    # Cleanup mode - remove generated files and exit
    if args.cleanup:
        cleanup_generated_files(args)
        sys.exit(0)
    
    # Resolve demand parameters (preset or custom)
    args.demand_label = resolve_demand_params(args)
    
    # Multi-run mode
    if args.runs > 1:
        sys.exit(run_multiple_simulations(args))
    
    # Single run mode (original behavior)
    # Generate traffic and demand files
    traffic_file = generate_car_traffic(args)
    demand_file = generate_passenger_demand(args, args.mode)
    
    # Store for use in mode functions
    args.traffic_file = traffic_file
    args.demand_file = demand_file
    
    if args.mode == 'drt':
        return_code = run_drt_mode(args)
    else:
        return_code = run_fixed_mode(args)
    
    # Rename output files for single run (use run_num=None for no run number in suffix)
    rename_output_files(args, run_num=None)
    
    sys.exit(return_code)