"""Multi-Run Analysis for SUMO Bus Simulation Experiments.

Usage:
    # Multiple runs with naming convention:
    python analyze_multiple_runs.py --traffic low --demand high --runs 5 --seed 42
    
    # Individual files (DRT vs Fixed comparison):
    python analyze_multiple_runs.py --drt-files tripinfo_drt.csv --fixed-files tripinfo_fixed.csv
    
    # Multiple individual files per mode:
    python analyze_multiple_runs.py --drt-files drt1.csv drt2.csv --fixed-files fixed1.csv fixed2.csv
    
    # Single mode only:
    python analyze_multiple_runs.py --drt-files tripinfo_drt.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats
import csv
import itertools

from analyze_csv_simple import (
    load_csv, analyze_pedestrians, analyze_buses_drt,
    analyze_buses_fixed, compute_km_per_passenger,
    compute_occupied_distance_km_from_stops, write_summary_csv
)

sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("paper", font_scale=1.2)


def load_run_data(mode, traffic, demand, seed, run_num):
    """Load CSV data for a single run.

    This will try, in order:
    1) exact filename using the provided seed and demand label,
    2) filename using seed + (run_num-1),
    3) any matching file `tripinfo_seed*_{mode}_t{traffic}_{demand_part}_run{run_num}.csv`
       choosing the candidate whose seed is closest to the requested seed.
    
    The `demand` parameter can be a preset name ('low', 'high') or a custom label ('p60b2').
    """
    import re
    from pathlib import Path

    # demand can be preset name or custom label like 'p60b2'
    # For custom labels (starting with 'p'), use as-is; for presets, add 'd' prefix
    if demand.startswith('p'):
        demand_part = demand  # Custom: p120b2 -> p120b2
    else:
        demand_part = f"d{demand}"  # Preset: low -> dlow
    
    requested = f"tripinfo_seed{seed}_{mode}_t{traffic}_{demand_part}_run{run_num}.csv"
    alt_seed = seed + (run_num - 1)
    alt = f"tripinfo_seed{alt_seed}_{mode}_t{traffic}_{demand_part}_run{run_num}.csv"

    # 1) exact
    p = Path(requested)
    if p.exists():
        tripinfo_file = p
    # 2) seed incremented
    elif Path(alt).exists():
        tripinfo_file = Path(alt)
    else:
        # 3) glob for any seed that matches the rest of pattern
        pattern = f"tripinfo_seed*_{mode}_t{traffic}_{demand_part}_run{run_num}.csv"
        candidates = list(Path(".").glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No files matching: {pattern}")
        # pick candidate whose seed number is closest to requested seed
        def seed_of(path):
            m = re.search(r'seed(\d+)', path.name)
            return int(m.group(1)) if m else 0
        candidates.sort(key=lambda pth: abs(seed_of(pth) - seed))
        tripinfo_file = candidates[0]

    stop_file = Path(str(tripinfo_file).replace("tripinfo_", "stop_"))
    if not tripinfo_file.exists():
        raise FileNotFoundError(f"Missing: {tripinfo_file}")
    # stop file may be absent for some modes — load_csv handles missing files (returns empty DF)
    fcd_file = Path(str(tripinfo_file).replace('tripinfo_', 'fcd_'))
    fcd_df = load_csv(str(fcd_file)) if fcd_file.exists() else pd.DataFrame()
    return load_csv(str(tripinfo_file)), load_csv(str(stop_file)), fcd_df


def analyze_single_run(mode, traffic, demand, seed, run_num):
    """Analyze a single run and return metrics dict plus raw ped/bus results and stops_df."""
    tripinfo_df, stops_df, fcd_df = load_run_data(mode, traffic, demand, seed, run_num)
    ped = analyze_pedestrians(tripinfo_df)
    # choose appropriate bus analyzer
    if mode == 'drt':
        bus = analyze_buses_drt(tripinfo_df, stops_df)
    else:
        bus = analyze_buses_fixed(tripinfo_df, stops_df, fcd_df=fcd_df)

    km_per_pax = compute_km_per_passenger(bus, ped)

    # top-level summary fields (keeps compatibility with previous callers)
    summary = {
        'run': run_num,
        'mode': mode,
        'avg_travel_time': ped.get('avg_travel_time', 0),
        'avg_waiting_time': ped.get('avg_waiting_time', 0),
        'avg_riding_time': ped.get('avg_riding_time', 0),
        'riders_count': ped.get('riders_count', 0),
        'total_persons': ped.get('total_persons', 0),
        'total_mileage_km': bus.get('total_mileage_km', 0),
        'avg_occupancy': bus.get('avg_occupancy', 0),
        'km_per_passenger': km_per_pax,
        # include raw dicts for full output / csv writing
        'ped_results': ped,
        'bus_results': bus,
        # include stops_df so caller can compute stop-based metrics (if needed)
        'stops_df': stops_df,
        'fcd_df': fcd_df,
    }
    return summary


def analyze_single_file(tripinfo_path, mode, run_num=1):
    """Analyze a single tripinfo CSV file directly."""
    tripinfo_df = load_csv(tripinfo_path)
    ped = analyze_pedestrians(tripinfo_df)
    
    # Auto-detect mode if not specified, otherwise use provided mode
    if mode == 'auto':
        mode = 'drt' if 'on_demand' in tripinfo_df['tripinfo_vType'].astype(str).str.cat() else 'fixed'
    
    bus = analyze_buses_drt(tripinfo_df) if mode == 'drt' else analyze_buses_fixed(tripinfo_df, pd.DataFrame())
    
    return {
        'run': run_num, 
        'mode': mode,
        'file': Path(tripinfo_path).name,
        'avg_travel_time': ped.get('avg_travel_time', 0),
        'avg_waiting_time': ped.get('avg_waiting_time', 0),
        'avg_riding_time': ped.get('avg_riding_time', 0),
        'riders_count': ped.get('riders_count', 0),
        'walkers_count': ped.get('walkers_count', 0),
        'total_persons': ped.get('total_persons', 0),
        'total_mileage_km': bus.get('total_mileage_km', 0),
        'avg_occupancy': bus.get('avg_occupancy', 0),
        'km_per_passenger': compute_km_per_passenger(bus, ped),
    }


def analyze_file_list(file_list, mode):
    """Analyze a list of tripinfo CSV files."""
    print(f"\nAnalyzing {mode.upper()}: {len(file_list)} file(s)")
    
    metrics = []
    for i, fpath in enumerate(file_list, 1):
        try:
            m = analyze_single_file(fpath, mode, run_num=i)
            metrics.append(m)
            print(f"  {Path(fpath).name}: travel={m['avg_travel_time']:.1f}s, wait={m['avg_waiting_time']:.1f}s, riders={m['riders_count']}")
        except Exception as e:
            print(f"  {fpath}: ERROR - {e}")
    
    if not metrics:
        raise ValueError(f"No valid files for {mode}")
    
    df = pd.DataFrame(metrics)
    
    # Calculate summary stats (SE=0 if only 1 file)
    n = len(df)
    summary = {}
    for col in df.columns:
        if col not in ['run', 'mode', 'file'] and pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            se_val = df[col].std() / np.sqrt(n) if n > 1 else 0
            summary[col] = {'mean': mean_val, 'se': se_val}
    
    return df, summary


def analyze_multiple_runs(mode, traffic, demand, num_runs, seed, summary_csv=None, network_label="", netfile=None):
    """
    Analyze all runs, return DataFrame and summary stats.
    If `summary_csv` is provided, append a detailed row per run using write_summary_csv().
    If `netfile` is provided and stop CSVs are present, compute occupied_distance_km from stops.
    """
    print(f"\nAnalyzing {mode.upper()}: {num_runs} runs (seed={seed})")
    
    metrics = []
    for run in range(1, num_runs + 1):
        try:
            m = analyze_single_run(mode, traffic, demand, seed, run)
            metrics.append(m)
            print(f"  Run {run}: travel={m.get('avg_travel_time', 0):.1f}s, wait={m.get('avg_waiting_time', 0):.1f}s")
            # If user requested a combined summary CSV, append the full ped/bus row
            if summary_csv:
                ped = m.get('ped_results', {})
                bus = m.get('bus_results', {})
                stops_df = m.get('stops_df', None)

                # occupied distance:
                # - preferred: already computed inside analyze_buses_fixed() when an fcd_*.csv exists
                # - fallback: if user provided --netfile, compute a coarse estimate from stops + net geometry
                if bus.get('occupied_distance_km', 0.0) in (0, 0.0) and netfile and stops_df is not None and not stops_df.empty:
                    try:
                        bus['occupied_distance_km'] = compute_occupied_distance_km_from_stops(stops_df, netfile)
                    except Exception as e:
                        print(f"    Warning: occupied-distance calc failed for run {run}: {e}")
                        bus.setdefault('occupied_distance_km', 0.0)

                km_per_pax = m.get('km_per_passenger', compute_km_per_passenger(bus, ped))
                try:
                    write_summary_csv(summary_csv, network_label, mode, ped, bus, km_per_pax)
                except Exception as e:
                    print(f"    Warning: failed to write summary CSV row for run {run}: {e}")



        except FileNotFoundError as e:
            print(f"  Run {run}: MISSING - {e}")
    
    if not metrics:
        raise ValueError(f"No valid runs for {mode}")
    
    df = pd.DataFrame(metrics)
    # numeric summary for top-level columns (exclude nested dicts)
    numeric_cols = [c for c in df.columns if c not in ['run', 'mode', 'ped_results', 'bus_results', 'stops_df']]
    summary = {col: {'mean': df[col].mean(), 'se': df[col].std() / np.sqrt(len(df))} 
               for col in numeric_cols if pd.api.types.is_numeric_dtype(df[col])}
    return df, summary


def write_averaged_summary_csv(outfile, mode, traffic, demand, num_runs, summary_stats, network_label=""):
    """Write a single row with averaged metrics (mean ± SE) across multiple runs."""
    import csv
    from pathlib import Path
    
    row = {
        "network": network_label,
        "mode": mode,
        "traffic": traffic,
        "demand": demand,
        "num_runs": num_runs,
    }
    
    # Add all summary stats with mean and se columns
    for metric, vals in summary_stats.items():
        if isinstance(vals, dict) and 'mean' in vals:
            row[f"{metric}_mean"] = vals['mean']
            row[f"{metric}_se"] = vals['se']
        else:
            row[metric] = vals
    
    write_header = not Path(outfile).exists()
    with open(outfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"  ✓ Appended averaged summary to {outfile}")


def print_comparison(drt_sum, fixed_sum, show_se=True):
    """Print DRT vs Fixed comparison table."""
    print("\n" + "="*70)
    print(f"{'Metric':<25} {'DRT':<18} {'Fixed':<18} {'Diff %':<10}")
    print("-"*70)
    
    metrics_to_show = ['avg_travel_time', 'avg_waiting_time', 'avg_riding_time',
                       'riders_count', 'total_mileage_km', 'km_per_passenger']
    
    for m in metrics_to_show:
        if m in drt_sum and m in fixed_sum:
            d, f = drt_sum[m]['mean'], fixed_sum[m]['mean']
            d_se, f_se = drt_sum[m]['se'], fixed_sum[m]['se']
            diff = (d - f) / f * 100 if f else 0
            
            if show_se and (d_se > 0 or f_se > 0):
                print(f"{m:<25} {d:>7.1f}±{d_se:<5.1f}   {f:>7.1f}±{f_se:<5.1f}   {diff:>+6.1f}%")
            else:
                print(f"{m:<25} {d:>13.1f}   {f:>13.1f}   {diff:>+6.1f}%")
    print("="*70)


def print_single_summary(summary, mode):
    """Print summary for a single mode."""
    print(f"\n{'='*50}")
    print(f"{mode.upper()} Summary")
    print(f"{'='*50}")
    
    for m, vals in summary.items():
        if vals['se'] > 0:
            print(f"  {m:<25}: {vals['mean']:>8.2f} ± {vals['se']:.2f}")
        else:
            print(f"  {m:<25}: {vals['mean']:>8.2f}")
    print("="*50)


def statistical_tests(drt_df, fixed_df):
    """Run paired t-tests on key metrics."""
    if len(drt_df) < 2 or len(fixed_df) < 2:
        print("\n(Statistical tests require at least 2 samples per group)")
        return
        
    if len(drt_df) != len(fixed_df):
        print("\n(Statistical tests require equal sample sizes - skipping)")
        return
    
    print("\nStatistical Tests (paired t-test):")
    for m in ['avg_travel_time', 'avg_waiting_time', 'km_per_passenger']:
        if m in drt_df.columns and m in fixed_df.columns:
            _, p = stats.ttest_rel(drt_df[m], fixed_df[m])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {m}: p={p:.4f} {sig}")


def combine_for_plot(drt_df, fixed_df):
    """Combine DRT and Fixed DataFrames for seaborn plotting."""
    drt, fixed = drt_df.copy(), fixed_df.copy()
    drt['Mode'], fixed['Mode'] = 'DRT', 'Fixed'
    return pd.concat([drt, fixed])


def create_comparison_plot(drt_df, fixed_df, output_file='comparison.png'):
    """Create 2x3 comparison plot."""
    combined = combine_for_plot(drt_df, fixed_df)
    colors = {'DRT': '#3498db', 'Fixed': '#e74c3c'}
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('DRT vs Fixed Route Comparison', fontsize=14, fontweight='bold')
    
    # Use bar plots if only 1 sample per mode, otherwise boxplot/violin
    single_sample = len(drt_df) == 1 and len(fixed_df) == 1
    
    plots = [
        ('avg_travel_time', 'Travel Time (s)'),
        ('avg_waiting_time', 'Waiting Time (s)'),
        ('riders_count', 'Passengers Served'),
        ('total_mileage_km', 'Total Distance (km)'),
        ('avg_occupancy', 'Avg Occupancy'),
        ('km_per_passenger', 'KM per Passenger'),
    ]
    
    for ax, (col, title) in zip(axes.flat, plots):
        if col in combined.columns:
            if single_sample:
                sns.barplot(data=combined, x='Mode', y=col, ax=ax, palette=colors)
            else:
                sns.boxplot(data=combined, x='Mode', y=col, ax=ax, palette=colors)
            ax.set_title(title)
            ax.set_xlabel('')
        else:
            ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze multiple simulation runs or compare individual files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multiple runs with naming convention:
  python analyze_multiple_runs.py --traffic low --demand high --runs 5 --seed 42
  
  # Compare individual DRT and Fixed files:
  python analyze_multiple_runs.py --drt-files tripinfo_drt.csv --fixed-files tripinfo_fixed.csv
  
  # Multiple files per mode (for statistics):
  python analyze_multiple_runs.py --drt-files drt1.csv drt2.csv --fixed-files fixed1.csv fixed2.csv
  
  # Single mode only:
  python analyze_multiple_runs.py --drt-files tripinfo_drt.csv
        """
    )
    
    # File-based mode
    parser.add_argument('--drt-files', nargs='+', help='DRT tripinfo CSV file(s)')
    parser.add_argument('--fixed-files', nargs='+', help='Fixed route tripinfo CSV file(s)')
    
    # Legacy run-based mode
    parser.add_argument('--traffic', help='Traffic level (for run-based mode)')
    parser.add_argument('--demand', help='Demand level (for run-based mode)')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs (default: 3)')
    parser.add_argument('--seed', type=int, default=42, help='Starting seed (default: 42)')
    parser.add_argument('--mode', choices=['drt', 'fixed', 'both'], default='both',
                        help='Which mode to analyze: drt, fixed, or both (default: both)')
    
    # Common options
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--output', default='comparison.png', help='Output plot filename')
    parser.add_argument('--summary-csv', type=str, default=None, help='Append per-run detailed summary rows to this CSV')
    parser.add_argument('--network', type=str, default='', help='Network label stored in summary CSV rows')
    parser.add_argument('--netfile', type=str, default=None, help='Path to SUMO .net.xml (used to compute occupied distance from stop files)')
    parser.add_argument('--averaged-csv', type=str, default=None, 
                        help='Export averaged summary (mean±SE) across runs to this CSV')

    
    args = parser.parse_args()
    
    drt_df = drt_sum = None
    fixed_df = fixed_sum = None
    
    # File-based mode
    if args.drt_files or args.fixed_files:
        if args.drt_files:
            try:
                drt_df, drt_sum = analyze_file_list(args.drt_files, 'drt')
            except Exception as e:
                print(f"DRT error: {e}")
        
        if args.fixed_files:
            try:
                fixed_df, fixed_sum = analyze_file_list(args.fixed_files, 'fixed')
            except Exception as e:
                print(f"Fixed error: {e}")
    
    # Legacy run-based mode
    elif args.traffic and args.demand:
        if args.mode in ('drt', 'both'):
            try:
                drt_df, drt_sum = analyze_multiple_runs('drt', args.traffic, args.demand, args.runs, args.seed, summary_csv=args.summary_csv, network_label=args.network, netfile=args.netfile)
                if args.averaged_csv and drt_sum:
                    write_averaged_summary_csv(args.averaged_csv, 'drt', args.traffic, args.demand, args.runs, drt_sum, args.network)
            except Exception as e:
                print(f"DRT error: {e}")
        
        if args.mode in ('fixed', 'both'):
            try:
                fixed_df, fixed_sum = analyze_multiple_runs('fixed', args.traffic, args.demand, args.runs, args.seed, summary_csv=args.summary_csv, network_label=args.network, netfile=args.netfile)
                if args.averaged_csv and fixed_sum:
                    write_averaged_summary_csv(args.averaged_csv, 'fixed', args.traffic, args.demand, args.runs, fixed_sum, args.network)
            except Exception as e:
                print(f"Fixed error: {e}")
    
    else:
        parser.print_help()
        print("\nError: Specify either --drt-files/--fixed-files OR --traffic/--demand")
        return
    
    # Print results
    if drt_sum and fixed_sum:
        print_comparison(drt_sum, fixed_sum)
        statistical_tests(drt_df, fixed_df)
        if not args.no_plots:
            create_comparison_plot(drt_df, fixed_df, args.output)
    elif drt_sum:
        print_single_summary(drt_sum, 'DRT')
    elif fixed_sum:
        print_single_summary(fixed_sum, 'Fixed')
    else:
        print("No valid data to analyze.")
        return
    
    print("\nDone!")


if __name__ == "__main__":
    main()