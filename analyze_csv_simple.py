"""
Simple SUMO Results Analyzer using xml2csv output files.

Metrics computed:
- Pedestrian: Average travel time, waiting time, walking time, riding transit/walking only split
- Bus: Average occupancy, mileage  
- KM per served passenger

Usage:
    python analyze_csv_simple.py tripinfo.csv stops.csv               # DRT mode
    python analyze_csv_simple.py tripinfo.output.csv stop.output.csv --fcd fcd.csv  # Fixed mode (occupied km)
    python analyze_csv_simple.py --compare file1.csv file2.csv ...    # Compare multiple files
    python analyze_csv_simple.py --compare file1.csv:Label1 file2.csv:Label2  # With custom labels
"""

import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
try:
    import sumolib
except Exception:
    sumolib = None

# --- Helper: compute coordinate on lane geometry at position (meters) ---
def _lane_coord_at_pos(net, lane_id, pos):
    lane = net.getLane(lane_id)
    if lane is None:
        return None
    shape = lane.getShape()
    if not shape:
        return None
    length = lane.getLength()
    p = max(0.0, min(pos, length))
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(shape[:-1], shape[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if acc + seg >= p:
            frac = (p - acc) / seg if seg > 1e-9 else 0.0
            return (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))
        acc += seg
    return shape[-1]

# --- Occupancy / distance helpers ---

def compute_occupied_distance_km_from_stops(stop_df, netfile):
    """Backward-compatible helper used by analyze_multiple_runs.py.

    Returns *occupied* vehicle-km computed from stops + network geometry.
    This is coarse (straight-line between stop coordinates), but requires no FCD.
    """
    return compute_occupied_and_passenger_km_from_stops(stop_df, netfile).get("occupied_km", 0.0)


def _pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_km_stats_from_fcd_and_stops(fcd_df, stops_df, vehicle_type="bus"):
    """Compute vehicle_km / occupied_km / passenger_km using FCD distance + stop occupancy events.

    Why this is usually the easiest/most accurate for fixed-route buses:
    - FCD provides *actual traveled distance* per vehicle via the `distance` attribute.
    - stop-output provides load/unload counts and onboard count at stops.

    Assumptions / notes:
    - Occupancy is assumed piecewise-constant between stops.
    - For the time before a vehicle's first recorded stop, we assume occupancy equals
      `initialPersons` at that first stop (typically 0 for fixed buses).
    - Distance deltas are taken from FCD `distance` and clamped at >=0.
    """
    import numpy as np

    if fcd_df is None or fcd_df.empty or stops_df is None or stops_df.empty:
        return {"vehicle_km": 0.0, "occupied_km": 0.0, "passenger_km": 0.0}

    # --- column detection (xml2csv naming varies slightly) ---
    time_col = _pick_first_existing_col(fcd_df, ["timestep_time", "time", "timestep"])
    veh_id_col = _pick_first_existing_col(fcd_df, ["vehicle_id", "id"])
    veh_type_col = _pick_first_existing_col(fcd_df, ["vehicle_type", "type"])
    dist_col = _pick_first_existing_col(fcd_df, ["vehicle_distance", "distance"])

    if time_col is None or veh_id_col is None or dist_col is None:
        missing = [k for k, v in [("time", time_col), ("vehicle id", veh_id_col), ("distance", dist_col)] if v is None]
        raise ValueError(f"FCD CSV is missing required columns: {', '.join(missing)}. "
                         f"Found columns={list(fcd_df.columns)[:20]}...")

    # Normalize types / numeric
    f = fcd_df.copy()
    f[time_col] = pd.to_numeric(f[time_col], errors="coerce")
    f[dist_col] = pd.to_numeric(f[dist_col], errors="coerce")
    f = f.dropna(subset=[time_col, veh_id_col, dist_col])

    # Filter to buses (if type column exists)
    if veh_type_col is not None and vehicle_type is not None:
        f = f[f[veh_type_col].astype(str).str.lower() == str(vehicle_type).lower()]

    # Stops: ensure numeric time columns exist
    s = stops_df.copy()
    sid = _pick_first_existing_col(s, ["stopinfo_id", "id"])
    s_end = _pick_first_existing_col(s, ["stopinfo_ended", "ended"])
    s_init = _pick_first_existing_col(s, ["stopinfo_initialPersons", "initialPersons"])
    s_loaded = _pick_first_existing_col(s, ["stopinfo_loadedPersons", "loadedPersons"])
    s_unloaded = _pick_first_existing_col(s, ["stopinfo_unloadedPersons", "unloadedPersons"])

    if sid is None or s_end is None:
        raise ValueError("Stops CSV is missing required columns (need stopinfo_id and stopinfo_ended).")

    for c in [s_end, s_init, s_loaded, s_unloaded]:
        if c and c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0)

    vehicle_m = 0.0
    occupied_m = 0.0
    passenger_m = 0.0

    for vid, fgrp in f.groupby(veh_id_col):
        fgrp = fgrp.sort_values(time_col)
        if len(fgrp) < 2:
            continue

        sgrp = s[s[sid] == vid].sort_values(s_end)
        if sgrp.empty:
            d = fgrp[dist_col].to_numpy()
            dd = np.diff(d)
            dd = np.where(dd > 0, dd, 0.0)
            vehicle_m += float(dd.sum())
            continue

        ends = sgrp[s_end].to_numpy(dtype=float)
        init_occ0 = int(sgrp.iloc[0].get(s_init, 0) or 0)

        occ_after = []
        occ = init_occ0
        for _, row in sgrp.iterrows():
            loaded = int(row.get(s_loaded, 0) or 0) if s_loaded else 0
            unloaded = int(row.get(s_unloaded, 0) or 0) if s_unloaded else 0
            occ = int(occ + loaded - unloaded)
            occ_after.append(occ)
        occ_after = np.asarray(occ_after, dtype=int)

        times = fgrp[time_col].to_numpy(dtype=float)
        dist = fgrp[dist_col].to_numpy(dtype=float)

        dd = np.diff(dist)
        dd = np.where(dd > 0, dd, 0.0)

        t0 = times[:-1]
        idx = np.searchsorted(ends, t0, side="right") - 1
        occ_interval = np.where(idx >= 0, occ_after[idx], init_occ0).astype(int)

        vehicle_m += float(dd.sum())

        mask_occ = occ_interval > 0
        if mask_occ.any():
            occupied_m += float(dd[mask_occ].sum())
            passenger_m += float((dd[mask_occ] * occ_interval[mask_occ]).sum())

    return {
        "vehicle_km": vehicle_m / 1000.0,
        "occupied_km": occupied_m / 1000.0,
        "passenger_km": passenger_m / 1000.0,
    }


def compute_peak_onboard_from_stops(stops_df, vehicle_ids=None, type_contains=None):
    """Compute peak onboard persons per vehicle (and overall) from stop-output.

    Uses:
      - initialPersons (occupancy when stop starts)
      - loadedPersons / unloadedPersons (change at the stop)

    Peak onboard is approximated as max over {initialPersons, initialPersons + loaded - unloaded} for each stop.

    Args:
        stops_df: DataFrame from stop-output (xml2csv).
        vehicle_ids: optional set/list of vehicle IDs to include (others ignored).
        type_contains: optional substring filter applied to stopinfo_type (case-insensitive), e.g. "bus".

    Returns:
        dict with:
          - max_onboard: int
          - max_onboard_vehicle_id: str or ""
          - per_vehicle_peak: dict[str,int] (may be empty)
    """
    if stops_df is None or getattr(stops_df, "empty", True):
        return {"max_onboard": 0, "max_onboard_vehicle_id": "", "per_vehicle_peak": {}}

    s = stops_df.copy()

    sid = _pick_first_existing_col(s, ["stopinfo_id", "id", "vehicle_id"])
    stype = _pick_first_existing_col(s, ["stopinfo_type", "type"])
    s_end = _pick_first_existing_col(s, ["stopinfo_ended", "ended"])
    s_start = _pick_first_existing_col(s, ["stopinfo_started", "started"])
    s_init = _pick_first_existing_col(s, ["stopinfo_initialPersons", "initialPersons"])
    s_loaded = _pick_first_existing_col(s, ["stopinfo_loadedPersons", "loadedPersons"])
    s_unloaded = _pick_first_existing_col(s, ["stopinfo_unloadedPersons", "unloadedPersons"])

    if sid is None:
        return {"max_onboard": 0, "max_onboard_vehicle_id": "", "per_vehicle_peak": {}}

    # Optional filters
    if vehicle_ids is not None:
        vset = set(map(str, vehicle_ids))
        s = s[s[sid].astype(str).isin(vset)]
    if type_contains and stype in s.columns:
        s = s[s[stype].astype(str).str.contains(type_contains, case=False, na=False)]

    if s.empty:
        return {"max_onboard": 0, "max_onboard_vehicle_id": "", "per_vehicle_peak": {}}

    # Ensure numeric
    for c in [s_end, s_start, s_init, s_loaded, s_unloaded]:
        if c and c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0)

    order_col = s_end or s_start

    per_vehicle_peak = {}
    max_onboard = 0
    max_vid = ""

    for vid, grp in s.groupby(sid):
        if order_col:
            grp = grp.sort_values(order_col)
        peak = 0
        for _, row in grp.iterrows():
            occ_before = int(row.get(s_init, 0) or 0) if s_init else 0
            loaded = int(row.get(s_loaded, 0) or 0) if s_loaded else 0
            unloaded = int(row.get(s_unloaded, 0) or 0) if s_unloaded else 0
            occ_after = occ_before + loaded - unloaded

            if occ_before > peak:
                peak = occ_before
            if occ_after > peak:
                peak = occ_after

        per_vehicle_peak[str(vid)] = int(peak)
        if peak > max_onboard:
            max_onboard = int(peak)
            max_vid = str(vid)

    return {"max_onboard": max_onboard, "max_onboard_vehicle_id": max_vid, "per_vehicle_peak": per_vehicle_peak}

# --- end occupancy helpers ---

def compute_occupied_and_passenger_km_from_stops(stop_df, netfile):
    """
    From a stop CSV and SUMO net, compute:
      - vehicle_km: total vehicle kilometers (sum of distances between consecutive stops per vehicle)
      - occupied_km: total vehicle-km where occupancy > 0 (binary)
      - passenger_km: sum over segments (occupancy * segment_length) -> passenger-km

    Returns dict with keys: vehicle_km, occupied_km, passenger_km (all in km).
    """
    from math import hypot

    if stop_df is None or stop_df.empty:
        return {"vehicle_km": 0.0, "occupied_km": 0.0, "passenger_km": 0.0}

    if sumolib is None:
        raise RuntimeError("sumolib not available; set SUMO_HOME/tools on PYTHONPATH")

    net = sumolib.net.readNet(netfile, withInternal=True)

    df = stop_df.copy()
    for col in ('stopinfo_pos', 'stopinfo_started', 'stopinfo_ended'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    def _coord_row(r):
        try:
            return _lane_coord_at_pos(net, r['stopinfo_lane'], float(r['stopinfo_pos']))
        except Exception:
            return None

    df['_coord'] = df.apply(_coord_row, axis=1)

    vehicle_m = 0.0
    occupied_m = 0.0
    passenger_m = 0.0

    group_col = 'stopinfo_id'
    time_col = 'stopinfo_started' if 'stopinfo_started' in df.columns else 'stopinfo_ended'

    for vid, grp in df.groupby(group_col):
        grp = grp.sort_values(time_col).reset_index(drop=True)
        if len(grp) < 2:
            continue
        # occupancy before first travel segment
        occ = int(grp.loc[0].get('stopinfo_initialPersons', 0) or 0)
        for i in range(len(grp) - 1):
            row = grp.loc[i]
            next_row = grp.loc[i + 1]
            a = row['_coord']
            b = next_row['_coord']
            if a is None or b is None:
                # cannot compute this segment
                # still update occ using this row's load/unload so next segment uses correct occ
                loaded = int(row.get('stopinfo_loadedPersons', 0) or 0)
                unloaded = int(row.get('stopinfo_unloadedPersons', 0) or 0)
                occ = int(occ + loaded - unloaded)
                continue
            dist_m = hypot(b[0] - a[0], b[1] - a[1])
            # update occupancy using load/unload at this stop before travel to next stop
            loaded = int(row.get('stopinfo_loadedPersons', 0) or 0)
            unloaded = int(row.get('stopinfo_unloadedPersons', 0) or 0)
            occ = int(occ + loaded - unloaded)
            vehicle_m += dist_m
            if occ > 0:
                occupied_m += dist_m
                passenger_m += occ * dist_m

    return {
        "vehicle_km": vehicle_m / 1000.0,
        "occupied_km": occupied_m / 1000.0,
        "passenger_km": passenger_m / 1000.0,
    }

def load_csv(filepath, sep=';'):
    """Load CSV file with semicolon separator (xml2csv default).

    Note: We set keep_default_na=False to preserve literal "NULL" strings
    in the data, which SUMO uses to indicate cancelled DRT requests.
    """
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        return pd.DataFrame()
    return pd.read_csv(filepath, sep=sep, keep_default_na=False, na_values=[''])


def analyze_pedestrians(tripinfo_df):
    """
    Analyze pedestrian metrics from tripinfo CSV.
    
    Returns dict with:
    - avg_travel_time: Average total duration from depart to arrival
    - avg_waiting_time: Average time waiting for transit
    - avg_walking_time: Average walking duration (for riders with walks)
    - avg_riding_time: Average time spent on transit
    - avg_trip_distance_km: Average distance traveled on transit
    - riders_count: Number of people who used transit
    - walkers_count: Number of people who walked only
    - Plus new metrics for walking distance, speed, time loss
    """
    # # Filter to rows with personinfo data (personinfo_id is not empty)
    # persons = tripinfo_df[tripinfo_df['personinfo_id'].notna()].copy()
    
    # if persons.empty:
    #     return {'error': 'No person data found'}
    
    # # Filter out unfinished trips (personinfo_duration == -1)
    # # These are persons who didn't complete their trip before simulation ended
    # persons = persons[persons['personinfo_duration'] != -1].copy()
    
    # if persons.empty:
    #     return {'error': 'No completed person trips found'}

    # Filter to rows with personinfo data (personinfo_id is not empty)
    persons_all = tripinfo_df[tripinfo_df['personinfo_id'].notna()].copy()

    if persons_all.empty:
        return {'error': 'No person data found'}

    # compute generated / unfinished / completed person ids
    all_ids = persons_all['personinfo_id'].unique()
    unfinished_ids = persons_all[persons_all['personinfo_duration'] == -1]['personinfo_id'].unique()
    unfinished_set = set(unfinished_ids)
    completed_set = [pid for pid in all_ids if pid not in unfinished_set]

    # keep only completed persons for the metrics below
    persons = persons_all[persons_all['personinfo_id'].isin(completed_set)].copy()

    if persons.empty:
        return {'error': 'No completed person trips found'}
    
    # === CLASSIFICATION: 3 categories ===
    # 1. Transit Riders: Have valid ride_vehicle (not empty, not NaN, not "NULL")
    # 2. Cancelled DRT: Have ride_vehicle = "NULL" (DRT request was cancelled)  
    # 3. Walk-Only: No ride_vehicle at all (actual pedestrians or no transit trip)
    
    # Identify actual riders (have a valid ride_vehicle, not "NULL")
    valid_ride_mask = (
        persons['ride_vehicle'].notna() & 
        (persons['ride_vehicle'] != '') &
        (persons['ride_vehicle'].astype(str) != 'NULL')
    )
    riders = persons[valid_ride_mask].copy()
    
    # Identify cancelled DRT requests (ride_vehicle = "NULL" means cancelled)
    cancelled_mask = (persons['ride_vehicle'].astype(str) == 'NULL')
    cancelled_drt = persons[cancelled_mask].copy()
    
    # Get unique person IDs for each category
    all_person_ids = persons['personinfo_id'].unique()
    rider_ids = riders['personinfo_id'].unique()
    cancelled_ids = cancelled_drt['personinfo_id'].unique()
    
    # Walk-only: persons not in riders AND not in cancelled
    walker_ids = [pid for pid in all_person_ids 
                  if pid not in rider_ids and pid not in cancelled_ids]
    
    walkers = persons[persons['personinfo_id'].isin(walker_ids)]
    
    # Get ALL rows for riders (including walk stages)
    rider_all_rows = persons[persons['personinfo_id'].isin(rider_ids)].copy()
    
    # Build aggregation dict dynamically based on available columns
    agg_dict = {
        'personinfo_duration': 'first',     # Total travel time (same in all rows for a person)
        'personinfo_waitingTime': 'first',  # Time waiting for transit
        'personinfo_timeLoss': 'first',     # Time loss
        'ride_duration': 'sum',              # Total riding time (from ride rows)
        'ride_routeLength': 'sum',           # Total distance on transit (meters)
        'ride_timeLoss': 'sum',              # Time loss during ride
    }
    
    # Add walk columns if they exist
    if 'walk_duration' in rider_all_rows.columns:
        agg_dict['walk_duration'] = 'sum'
    if 'walk_routeLength' in rider_all_rows.columns:
        agg_dict['walk_routeLength'] = 'sum'
    if 'walk_timeLoss' in rider_all_rows.columns:
        agg_dict['walk_timeLoss'] = 'sum'
    if 'ride_waitingTime' in rider_all_rows.columns:
        agg_dict['ride_waitingTime'] = 'sum'
    
    rider_stats = rider_all_rows.groupby('personinfo_id').agg(agg_dict).reset_index()
    
    # Fill NaN values and ensure columns exist
    for col in ['walk_duration', 'walk_routeLength', 'walk_timeLoss', 'ride_duration', 
                'ride_routeLength', 'ride_timeLoss', 'ride_waitingTime', 'personinfo_timeLoss']:
        if col not in rider_stats.columns:
            rider_stats[col] = 0.0
        else:
            rider_stats[col] = rider_stats[col].fillna(0)
    
    # === Cancelled DRT stats ===
    cancelled_agg_dict = {
        'personinfo_duration': 'first',
        'personinfo_timeLoss': 'first',
        'personinfo_waitingTime': 'first',
    }
    
    if not cancelled_drt.empty:
        cancelled_stats = cancelled_drt.groupby('personinfo_id').agg(cancelled_agg_dict).reset_index()
    else:
        cancelled_stats = pd.DataFrame()
    
    # === Walker-only stats ===
    walker_agg_dict = {
        'personinfo_duration': 'first',
        'personinfo_timeLoss': 'first',
    }
    if 'walk_duration' in walkers.columns:
        walker_agg_dict['walk_duration'] = 'sum'
    if 'walk_routeLength' in walkers.columns:
        walker_agg_dict['walk_routeLength'] = 'sum'
    if 'walk_timeLoss' in walkers.columns:
        walker_agg_dict['walk_timeLoss'] = 'sum'
    
    if not walkers.empty:
        walker_stats = walkers.groupby('personinfo_id').agg(walker_agg_dict).reset_index()
        for col in ['walk_duration', 'walk_routeLength', 'walk_timeLoss']:
            if col not in walker_stats.columns:
                walker_stats[col] = 0.0
            else:
                walker_stats[col] = walker_stats[col].fillna(0)
    else:
        walker_stats = pd.DataFrame()
    
    # === Calculate results ===
    results = {
        'riders_count': len(rider_ids),
        'walkers_count': len(walker_ids),
        'cancelled_drt_count': len(cancelled_ids),  # NEW: Track cancelled DRT requests
        'total_persons': len(all_person_ids),
        'rider_percentage': len(rider_ids) / len(all_person_ids) * 100 if all_person_ids.size > 0 else 0,
        'total_persons_generated': len(all_ids),
        'unfinished_persons': len(unfinished_set),
        'completion_rate_pct': 100.0 * len(completed_set) / len(all_ids) if len(all_ids) else 0.0,
    }
    
    # --- Cancelled DRT metrics ---
    if not cancelled_stats.empty:
        results.update({
            'cancelled_avg_travel_time': cancelled_stats['personinfo_duration'].mean(),
            'cancelled_avg_waiting_time': cancelled_stats['personinfo_waitingTime'].mean(),
        })
    else:
        results.update({
            'cancelled_avg_travel_time': 0,
            'cancelled_avg_waiting_time': 0,
        })
    
    # --- Rider metrics ---
    if not rider_stats.empty:
        # Calculate total travel distance (walk + ride) for speed calculation
        rider_stats['total_distance'] = rider_stats['walk_routeLength'] + rider_stats['ride_routeLength']
        
        # Safe speed calculation - avoid division by zero and filter invalid values
        rider_stats['avg_speed_mps'] = rider_stats.apply(
            lambda row: row['total_distance'] / row['personinfo_duration'] 
            if row['personinfo_duration'] > 0 else 0, 
            axis=1
        )
        rider_stats['avg_speed_kmh'] = rider_stats['avg_speed_mps'] * 3.6
        
        # Filter out unreasonable speeds (negative or > 100 km/h for pedestrians)
        valid_speeds = rider_stats['avg_speed_kmh'][(rider_stats['avg_speed_kmh'] >= 0) & (rider_stats['avg_speed_kmh'] <= 100)]
        avg_rider_speed = valid_speeds.mean() if not valid_speeds.empty else 0
        
        results.update({
            # Travel time
            'avg_travel_time': rider_stats['personinfo_duration'].mean(),
            'avg_waiting_time': rider_stats['personinfo_waitingTime'].mean(),
            'avg_riding_time': rider_stats['ride_duration'].mean(),
            'avg_trip_distance_km': rider_stats['ride_routeLength'].mean() / 1000,
            
            # Rider walking metrics
            'rider_avg_walking_time': rider_stats['walk_duration'].mean(),
            'rider_avg_walking_distance_m': rider_stats['walk_routeLength'].mean(),
            'rider_avg_walking_distance_km': rider_stats['walk_routeLength'].mean() / 1000,
            
            # Speed (using filtered valid speeds)
            'rider_avg_speed_kmh': avg_rider_speed,
            
            # Time loss
            'rider_avg_time_loss': rider_stats['personinfo_timeLoss'].mean(),
            'rider_avg_ride_time_loss': rider_stats['ride_timeLoss'].mean(),
            'rider_avg_walk_time_loss': rider_stats['walk_timeLoss'].mean(),
            'rider_avg_ride_wait_time': rider_stats['ride_waitingTime'].mean(),
        })
    else:
        results.update({
            'avg_travel_time': 0, 'avg_waiting_time': 0, 'avg_riding_time': 0,
            'avg_trip_distance_km': 0,
            'rider_avg_walking_time': 0, 'rider_avg_walking_distance_m': 0, 
            'rider_avg_walking_distance_km': 0,
            'rider_avg_speed_kmh': 0,
            'rider_avg_time_loss': 0, 'rider_avg_ride_time_loss': 0,
            'rider_avg_walk_time_loss': 0, 'rider_avg_ride_waiting_time': 0,
        })
    
    # --- Walker-only metrics ---
    if not walker_stats.empty:
        # Safe speed calculation for walkers
        walker_stats['avg_speed_mps'] = walker_stats.apply(
            lambda row: row['walk_routeLength'] / row['personinfo_duration'] 
            if row['personinfo_duration'] > 0 else 0, 
            axis=1
        )
        walker_stats['avg_speed_kmh'] = walker_stats['avg_speed_mps'] * 3.6
        
        # Filter out unreasonable speeds
        valid_walker_speeds = walker_stats['avg_speed_kmh'][(walker_stats['avg_speed_kmh'] >= 0) & (walker_stats['avg_speed_kmh'] <= 20)]
        avg_walker_speed = valid_walker_speeds.mean() if not valid_walker_speeds.empty else 0
        
        results.update({
            'walker_avg_travel_time': walker_stats['personinfo_duration'].mean(),
            'walker_avg_walking_time': walker_stats['walk_duration'].mean(),
            'walker_avg_walking_distance_m': walker_stats['walk_routeLength'].mean(),
            'walker_avg_walking_distance_km': walker_stats['walk_routeLength'].mean() / 1000,
            'walker_avg_speed_kmh': avg_walker_speed,
            'walker_avg_time_loss': walker_stats['personinfo_timeLoss'].mean(),
            'walker_avg_walk_time_loss': walker_stats['walk_timeLoss'].mean(),
        })
    else:
        results.update({
            'walker_avg_travel_time': 0, 'walker_avg_walking_time': 0,
            'walker_avg_walking_distance_m': 0, 'walker_avg_walking_distance_km': 0,
            'walker_avg_speed_kmh': 0, 'walker_avg_time_loss': 0, 'walker_avg_walk_time_loss': 0,
        })
    
    # --- Combined walking metrics (backwards compatibility) ---
    # Average walking time across all persons (riders + walkers)
    all_walk_times = []
    all_walk_distances = []
    all_travel_times = []
    all_time_losses = []
    all_walk_time_losses = []
    all_speeds = []
    
    if not rider_stats.empty:
        all_walk_times.extend(rider_stats['walk_duration'].tolist())
        all_walk_distances.extend(rider_stats['walk_routeLength'].tolist())
        all_travel_times.extend(rider_stats['personinfo_duration'].tolist())
        all_time_losses.extend(rider_stats['personinfo_timeLoss'].tolist())
        all_walk_time_losses.extend(rider_stats['walk_timeLoss'].tolist())
        # Get valid rider speeds
        valid_rider_speeds = rider_stats['avg_speed_kmh'][(rider_stats['avg_speed_kmh'] >= 0) & (rider_stats['avg_speed_kmh'] <= 100)]
        all_speeds.extend(valid_rider_speeds.tolist())
        
    if not walker_stats.empty:
        all_walk_times.extend(walker_stats['walk_duration'].tolist())
        all_walk_distances.extend(walker_stats['walk_routeLength'].tolist())
        all_travel_times.extend(walker_stats['personinfo_duration'].tolist())
        all_time_losses.extend(walker_stats['personinfo_timeLoss'].tolist())
        all_walk_time_losses.extend(walker_stats['walk_timeLoss'].tolist())
        # Get valid walker speeds
        valid_walker_speeds = walker_stats['avg_speed_kmh'][(walker_stats['avg_speed_kmh'] >= 0) & (walker_stats['avg_speed_kmh'] <= 20)]
        all_speeds.extend(valid_walker_speeds.tolist())
    
    results['avg_walking_time'] = sum(all_walk_times) / len(all_walk_times) if all_walk_times else 0
    results['avg_walking_distance_m'] = sum(all_walk_distances) / len(all_walk_distances) if all_walk_distances else 0
    results['all_avg_travel_time'] = sum(all_travel_times) / len(all_travel_times) if all_travel_times else 0
    results['all_avg_time_loss'] = sum(all_time_losses) / len(all_time_losses) if all_time_losses else 0
    results['all_avg_walk_time_loss'] = sum(all_walk_time_losses) / len(all_walk_time_losses) if all_walk_time_losses else 0
    results['all_avg_speed_kmh'] = sum(all_speeds) / len(all_speeds) if all_speeds else 0
    
    return results



def analyze_buses_drt(tripinfo_df, stops_df=None):
    """
    Analyze DRT bus metrics from tripinfo CSV.

    Adds (easy) extra stats:
      1) number of vehicles used (unique vehicle IDs)
      2) max onboard on a single vehicle (from stop-output, if provided)
      3) requests/customers served (taxi_customers sum)

    Returns dict with:
    - bus_count: number of DRT vehicles used
    - total_mileage_km / avg_mileage_km
    - total_customers: total customers served (sum taxi_customers)
    - drt_requests_served: same as total_customers (kept explicit)
    - occupied_distance_km / occupancy_rate (from taxi_occupiedDistance)
    - vehicle_km, passenger_km, avg_occupancy (from stops if available)
    - max_onboard / max_onboard_vehicle_id (if stops_df provided)
    """
    # DRT buses have tripinfo_vType containing 'on_demand' and taxi_customers column
    buses = tripinfo_df[
        (tripinfo_df['tripinfo_vType'].astype(str).str.contains('on_demand', case=False, na=False)) &
        (tripinfo_df.get('taxi_customers') is not None) &
        (tripinfo_df['taxi_customers'].notna())
    ].copy()

    if buses.empty:
        return {'error': 'No DRT bus data found'}

    vid_col = _pick_first_existing_col(buses, ["tripinfo_id", "id", "vehicle_id"])
    if vid_col is None:
        # tripinfo xml2csv usually includes tripinfo_id
        vid_col = buses.columns[0]

    unique_bus_ids = buses[vid_col].astype(str)
    bus_count = int(unique_bus_ids.nunique())

    # Convert meters to km
    buses['mileage_km'] = pd.to_numeric(buses['tripinfo_routeLength'], errors="coerce").fillna(0.0) / 1000.0
    buses['occupied_km'] = pd.to_numeric(buses.get('taxi_occupiedDistance', 0.0), errors="coerce").fillna(0.0) / 1000.0

    total_mileage = float(buses['mileage_km'].sum())
    total_customers = float(pd.to_numeric(buses['taxi_customers'], errors="coerce").fillna(0.0).sum())
    occupied_distance_km = float(buses['occupied_km'].sum())

    # === EXACT passenger_km from ride_routeLength ===
    passenger_km = _compute_passenger_km_from_rides(tripinfo_df)

    # avg_occupancy = passenger_km / occupied_km
    if occupied_distance_km > 0:
        avg_occupancy = passenger_km / occupied_distance_km
    elif total_mileage > 0:
        avg_occupancy = passenger_km / total_mileage
    else:
        avg_occupancy = 0.0

    results = {
        'bus_count': bus_count,
        'total_mileage_km': total_mileage,
        'avg_mileage_km': float(buses['mileage_km'].mean()) if bus_count > 0 else 0.0,
        'total_customers': int(total_customers),
        'drt_requests_served': int(total_customers),
        'avg_customers_per_bus': (total_customers / bus_count) if bus_count > 0 else 0.0,
        'occupied_distance_km': occupied_distance_km,
        'occupancy_rate': (occupied_distance_km / total_mileage * 100.0) if total_mileage > 0 else 0.0,
        'vehicle_km': total_mileage,
        'occupied_km': occupied_distance_km,
        'passenger_km': passenger_km,
        'avg_occupancy': avg_occupancy,
    }

    # Peak onboard from stop-output (if provided)
    if stops_df is not None and not getattr(stops_df, "empty", True):
        peak = compute_peak_onboard_from_stops(stops_df, vehicle_ids=set(unique_bus_ids))
        results["max_onboard"] = int(peak.get("max_onboard", 0))
        results["max_onboard_vehicle_id"] = peak.get("max_onboard_vehicle_id", "")
    else:
        results["max_onboard"] = 0
        results["max_onboard_vehicle_id"] = ""

    return results


def _compute_passenger_km_from_rides(tripinfo_df):
    """Compute exact passenger-km by summing ride_routeLength for all person rides.
    
    This is accurate for both DRT and Fixed-route modes because SUMO records
    the actual route length traveled by each person in each ride stage.
    """
    ride_col = _pick_first_existing_col(tripinfo_df, ['ride_routeLength', 'routeLength'])
    if ride_col is None:
        return 0.0
    
    ride_lengths = pd.to_numeric(tripinfo_df[ride_col], errors="coerce")
    return float(ride_lengths.dropna().sum()) / 1000.0


def analyze_buses_fixed(tripinfo_df, stops_df, netfile=None, fcd_df=None):
    """
    Analyze fixed-route bus metrics.
    Fixed buses have tripinfo_vType == 'bus'.
    
    Returns dict with:
    - total_mileage_km: Total distance traveled by all buses
    - avg_mileage_km: Average mileage per bus
    - total_boardings: Total passengers boarding
    - avg_occupancy: Average passengers per stop
    """
    # Fixed buses have vType containing 'bus'
    buses = tripinfo_df[
        tripinfo_df['tripinfo_vType'].str.contains('bus', case=False, na=False)
    ].copy()

    if buses.empty:
        return {'error': 'No fixed-route bus data found'}

    vid_col = _pick_first_existing_col(buses, ['tripinfo_id', 'id', 'vehicle_id'])
    bus_ids = buses[vid_col].astype(str) if vid_col in buses.columns else buses.index.astype(str)
    bus_count = int(bus_ids.nunique())

    buses['mileage_km'] = buses['tripinfo_routeLength'] / 1000.0
    total_mileage = float(buses['mileage_km'].sum())

    # === EXACT passenger_km from ride_routeLength ===
    passenger_km = _compute_passenger_km_from_rides(tripinfo_df)

    # avg_occupancy = passenger_km / vehicle_km
    avg_occupancy = (passenger_km / total_mileage) if total_mileage > 0 else 0.0

    # Boardings from stops
    total_boardings = 0
    total_alightings = 0
    if stops_df is not None and not stops_df.empty:
        loaded_col = _pick_first_existing_col(stops_df, ['stopinfo_loadedPersons', 'loadedPersons'])
        unloaded_col = _pick_first_existing_col(stops_df, ['stopinfo_unloadedPersons', 'unloadedPersons'])
        if loaded_col:
            total_boardings = int(pd.to_numeric(stops_df[loaded_col], errors='coerce').fillna(0).sum())
        if unloaded_col:
            total_alightings = int(pd.to_numeric(stops_df[unloaded_col], errors='coerce').fillna(0).sum())

    total_customers = total_boardings  # for fixed, customers = boardings

    results = {
        'bus_count': bus_count,
        'total_mileage_km': total_mileage,
        'avg_mileage_km': (total_mileage / bus_count) if bus_count > 0 else 0.0,
        'total_boardings': total_boardings,
        'total_alightings': total_alightings,
        'total_customers': total_customers,
        'avg_customers_per_bus': (total_customers / bus_count) if bus_count > 0 else 0.0,
        'vehicle_km': total_mileage,
        'passenger_km': passenger_km,
        'avg_occupancy': avg_occupancy,
        # occupied_km is harder for fixed (no taxi device), estimate from FCD or stops if available
        'occupied_km': 0.0,
        'occupied_distance_km': 0.0,
        'occupancy_rate': 0.0,
    }

    # Compute occupied_km from FCD + stops if available
    if (fcd_df is not None) and (not getattr(fcd_df, "empty", True)) and (stops_df is not None) and (not stops_df.empty):
        try:
            km_stats = compute_km_stats_from_fcd_and_stops(fcd_df, stops_df, vehicle_type="bus")
            results['occupied_km'] = km_stats.get('occupied_km', 0.0)
            results['occupied_distance_km'] = km_stats.get('occupied_km', 0.0)
            if results['vehicle_km'] > 0:
                results['occupancy_rate'] = results['occupied_km'] / results['vehicle_km'] * 100.0
        except Exception:
            pass

    # Peak onboard
    if stops_df is not None and not getattr(stops_df, "empty", True):
        peak = compute_peak_onboard_from_stops(stops_df, type_contains="bus")
        results["max_onboard"] = int(peak.get("max_onboard", 0))
        results["max_onboard_vehicle_id"] = peak.get("max_onboard_vehicle_id", "")
    else:
        results["max_onboard"] = 0
        results["max_onboard_vehicle_id"] = ""

    return results


def compute_km_per_passenger(bus_results, pedestrian_results):
    """
    Compute KM per served passenger.
    
    KM per passenger = Total bus mileage / Number of passengers served
    """
    total_km = bus_results.get('total_mileage_km', 0)
    
    # For DRT: use total_customers from taxi device
    # For Fixed: use riders_count from pedestrian analysis
    passengers_served = bus_results.get('total_customers', pedestrian_results.get('riders_count', 0))
    
    if passengers_served > 0:
        return total_km / passengers_served
    return 0


def detect_mode(tripinfo_df):
    """Detect if simulation is DRT or Fixed-route."""
    if 'taxi_customers' in tripinfo_df.columns:
        drt_buses = tripinfo_df['taxi_customers'].notna().sum()
        if drt_buses > 0:
            return 'drt'
    return 'fixed'


def print_results(ped_results, bus_results, km_per_pax, mode):
    """Print formatted results."""
    print("\n" + "="*60)
    print(f"SIMULATION RESULTS ({mode.upper()} MODE)")
    print("="*60)
    
    print("\n📊 PEDESTRIAN METRICS")
    print("-"*40)
    print(f"  Total persons:        {ped_results.get('total_persons', 0)}")
    print(f"  Transit riders:       {ped_results.get('riders_count', 0)} ({ped_results.get('rider_percentage', 0):.1f}%)")
    print(f"  Walk-only:            {ped_results.get('walkers_count', 0)}")
    
    # Show cancelled DRT if any (DRT mode specific)
    cancelled_count = ped_results.get('cancelled_drt_count', 0)
    if cancelled_count > 0:
        print(f"  Cancelled DRT:        {cancelled_count} (request removed/timeout)")

    print(f"  Total persons generated: {ped_results.get('total_persons_generated', ped_results.get('total_persons', 0))}")
    print(f"  Unfinished persons:      {ped_results.get('unfinished_persons', 0)}")
    print(f"  Completion rate:         {ped_results.get('completion_rate_pct', 0):.1f}%")
    
    print("\n  --- Transit Riders ---")
    print(f"  Avg travel time:      {ped_results.get('avg_travel_time', 0):.1f} s")
    print(f"  Avg waiting time:     {ped_results.get('avg_waiting_time', 0):.1f} s")
    print(f"  Avg riding time:      {ped_results.get('avg_riding_time', 0):.1f} s")
    print(f"  Avg ride distance:    {ped_results.get('avg_trip_distance_km', 0):.3f} km")
    print(f"  Avg walking time:     {ped_results.get('rider_avg_walking_time', 0):.1f} s")
    print(f"  Avg walking distance: {ped_results.get('rider_avg_walking_distance_m', 0):.1f} m")
    print(f"  Avg speed:            {ped_results.get('rider_avg_speed_kmh', 0):.2f} km/h")
    print(f"  Avg time loss:        {ped_results.get('rider_avg_time_loss', 0):.1f} s")
    print(f"  Avg ride time loss:   {ped_results.get('rider_avg_ride_time_loss', 0):.1f} s")
    print(f"  Avg walk time loss:   {ped_results.get('rider_avg_walk_time_loss', 0):.1f} s")
    print(f"  Avg ride wait time:   {ped_results.get('rider_avg_ride_waiting_time', 0):.1f} s")
    
    # Cancelled DRT section (if any)
    if cancelled_count > 0:
        print("\n  --- Cancelled DRT Requests ---")
        print(f"  Count:                {cancelled_count}")
        print(f"  Avg time in system:   {ped_results.get('cancelled_avg_travel_time', 0):.1f} s")
        print(f"  Avg waiting time:     {ped_results.get('cancelled_avg_waiting_time', 0):.1f} s")
        print("  (These are persons whose DRT request was cancelled/timed out)")
    
    if ped_results.get('walkers_count', 0) > 0:
        print("\n  --- Walk-Only Travelers ---")
        print(f"  Avg travel time:      {ped_results.get('walker_avg_travel_time', 0):.1f} s")
        print(f"  Avg walking time:     {ped_results.get('walker_avg_walking_time', 0):.1f} s")
        print(f"  Avg walking distance: {ped_results.get('walker_avg_walking_distance_m', 0):.1f} m")
        print(f"  Avg speed:            {ped_results.get('walker_avg_speed_kmh', 0):.2f} km/h")
        print(f"  Avg time loss:        {ped_results.get('walker_avg_time_loss', 0):.1f} s")
        print(f"  Avg walk time loss:   {ped_results.get('walker_avg_walk_time_loss', 0):.1f} s")
    
    print("\n  --- All Pedestrians (Combined) ---")
    print(f"  Avg travel time:      {ped_results.get('all_avg_travel_time', 0):.1f} s")
    print(f"  Avg walking time:     {ped_results.get('avg_walking_time', 0):.1f} s")
    print(f"  Avg walking distance: {ped_results.get('avg_walking_distance_m', 0):.1f} m")
    print(f"  Avg speed:            {ped_results.get('all_avg_speed_kmh', 0):.2f} km/h")
    print(f"  Avg time loss:        {ped_results.get('all_avg_time_loss', 0):.1f} s")
    print(f"  Avg walk time loss:   {ped_results.get('all_avg_walk_time_loss', 0):.1f} s")
    
    print("\n🚌 BUS METRICS")
    print("-"*40)
    print(f"  Number of buses:      {bus_results.get('bus_count', 0)}")
    print(f"  Total mileage:        {bus_results.get('total_mileage_km', 0):.2f} km")
    print(f"  Avg mileage per bus:  {bus_results.get('avg_mileage_km', 0):.2f} km")
    
    if mode == 'drt':
        print(f"  Total customers:      {bus_results.get('total_customers', 0)}")
        print(f"  DRT requests served:  {bus_results.get('drt_requests_served', bus_results.get('total_customers', 0))}")
        print(f"  Avg customers/bus:    {bus_results.get('avg_customers_per_bus', 0):.1f}")
        print(f"  Occupied distance:    {bus_results.get('occupied_distance_km', 0):.2f} km")
        print(f"  Occupancy rate:       {bus_results.get('occupancy_rate', 0):.1f}%")
    else:
        # Provide DRT-equivalent metrics for fixed-route buses using available fields
        total_customers = bus_results.get('total_customers', bus_results.get('total_boardings', 0))
        bus_count = bus_results.get('bus_count', 0)
        avg_customers_per_bus = bus_results.get(
            'avg_customers_per_bus',
            (total_customers / bus_count) if bus_count else 0
        )
        occupied_distance_km = bus_results.get('occupied_distance_km', 0)
        occupancy_rate = bus_results.get('occupancy_rate', bus_results.get('avg_occupancy_at_stops', 0))

        print(f"  Total customers:      {int(total_customers)}")
        print(f"  Avg customers/bus:    {avg_customers_per_bus:.1f}")
        print(f"  Occupied distance:    {occupied_distance_km:.2f} km")
        print(f"  Occupancy rate:       {occupancy_rate:.1f}%")

    # === NEW v2 metrics: vehicle_km, passenger_km, avg_occupancy, peak onboard ===
    print("\n  --- Distance & Occupancy (v2) ---")
    vehicle_km = bus_results.get('vehicle_km', bus_results.get('total_mileage_km', 0))
    occupied_km = bus_results.get('occupied_km', bus_results.get('occupied_distance_km', 0))
    passenger_km = bus_results.get('passenger_km', 0)
    avg_occupancy = bus_results.get('avg_occupancy', 0)
    
    print(f"  Vehicle-km:           {vehicle_km:.2f} km")
    print(f"  Occupied-km:          {occupied_km:.2f} km")
    print(f"  Passenger-km:         {passenger_km:.2f} pax·km")
    print(f"  Avg occupancy:        {avg_occupancy:.2f} pax/vehicle")
    
    # Peak onboard
    max_onboard = bus_results.get('max_onboard', 0)
    max_onboard_vid = bus_results.get('max_onboard_vehicle_id', '')
    if max_onboard > 0:
        print(f"  Peak onboard:         {max_onboard} (vehicle: {max_onboard_vid})")
    else:
        print(f"  Peak onboard:         {max_onboard}")
    
    print("\n📈 EFFICIENCY")
    print("-"*40)
    print(f"  KM per served passenger: {km_per_pax:.3f} km")
    print("="*60 + "\n")


def extract_comparison_metrics(tripinfo_df):
    """
    Extract key metrics for comparison from a tripinfo CSV.
    
    Returns dict with:
    - avg_speed_kmh: Average speed of all persons (km/h)
    - avg_time_loss: Average time loss of all persons (seconds)
    - avg_ride_distance_km: Average ride distance (km)
    """
    ped_results = analyze_pedestrians(tripinfo_df)
    
    if 'error' in ped_results:
        return {
            'avg_speed_kmh': 0,
            'avg_time_loss': 0,
            'avg_ride_distance_km': 0,
        }
    
    return {
        'avg_speed_kmh': ped_results.get('all_avg_speed_kmh', 0),
        'avg_time_loss': ped_results.get('all_avg_time_loss', 0),
        'avg_ride_distance_km': ped_results.get('avg_trip_distance_km', 0),
    }


def compare_csv_files(csv_files, labels=None):
    """
    Compare metrics across multiple CSV files.
    
    Args:
        csv_files: List of paths to tripinfo CSV files
        labels: Optional list of labels for each file (defaults to filenames)
    
    Returns:
        dict with lists of metrics for each file
    """
    if labels is None:
        labels = [Path(f).stem for f in csv_files]
    
    comparison_data = {
        'labels': labels,
        'avg_speed_kmh': [],
        'avg_time_loss': [],
        'avg_ride_distance_km': [],
    }
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        df = load_csv(csv_file)
        
        if df.empty:
            print(f"  Warning: Empty or missing file: {csv_file}")
            comparison_data['avg_speed_kmh'].append(0)
            comparison_data['avg_time_loss'].append(0)
            comparison_data['avg_ride_distance_km'].append(0)
            continue
        
        metrics = extract_comparison_metrics(df)
        comparison_data['avg_speed_kmh'].append(metrics['avg_speed_kmh'])
        comparison_data['avg_time_loss'].append(metrics['avg_time_loss'])
        comparison_data['avg_ride_distance_km'].append(metrics['avg_ride_distance_km'])
    
    return comparison_data


def plot_comparison_bar_graph(comparison_data, output_file='comparison_chart.png'):
    """
    Create a grouped bar graph comparing metrics across multiple CSV files.
    
    Args:
        comparison_data: Dict from compare_csv_files()
        output_file: Path to save the chart (default: comparison_chart.png)
    """
    labels = comparison_data['labels']
    n_files = len(labels)
    
    metrics = {
        'Avg Speed (km/h)': comparison_data['avg_speed_kmh'],
        'Avg Time Loss (s)': comparison_data['avg_time_loss'],
        'Avg Ride Distance (km)': comparison_data['avg_ride_distance_km'],
    }
    
    n_metrics = len(metrics)
    x = np.arange(n_files)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(10, n_files * 2), 6))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']  # Green, Red, Blue
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i])
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Simulation Scenario')
    ax.set_ylabel('Value')
    ax.set_title('Comparison of Simulation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()
    
    print(f"\nChart saved to: {output_file}")
    return fig, ax


def print_comparison_table(comparison_data):
    """Print a formatted comparison table."""
    labels = comparison_data['labels']
    
    print("\n" + "="*70)
    print("COMPARISON OF SIMULATION METRICS")
    print("="*70)
    print(f"{'Scenario':<25} {'Avg Speed':<15} {'Avg Time Loss':<15} {'Avg Ride Dist':<15}")
    print(f"{'':<25} {'(km/h)':<15} {'(s)':<15} {'(km)':<15}")
    print("-"*70)
    
    for i, label in enumerate(labels):
        speed = comparison_data['avg_speed_kmh'][i]
        time_loss = comparison_data['avg_time_loss'][i]
        ride_dist = comparison_data['avg_ride_distance_km'][i]
        print(f"{label:<25} {speed:<15.2f} {time_loss:<15.2f} {ride_dist:<15.3f}")
    
    print("="*70 + "\n")


def compare_main():
    """Main function for comparison mode."""
    if len(sys.argv) < 3:
        print("Usage: python analyze_csv_simple.py --compare file1.csv file2.csv [file3.csv ...]")
        print("       python analyze_csv_simple.py --compare file1.csv:Label1 file2.csv:Label2 ...")
        print("\nExample:")
        print("  python analyze_csv_simple.py --compare drt_low.csv drt_high.csv fixed.csv")
        print("  python analyze_csv_simple.py --compare drt.csv:DRT fixed.csv:Fixed")
        return
    
    csv_files = []
    labels = []
    
    for arg in sys.argv[2:]:
        if ':' in arg:
            filepath, label = arg.rsplit(':', 1)
            csv_files.append(filepath)
            labels.append(label)
        else:
            csv_files.append(arg)
            labels.append(Path(arg).stem)
    
    comparison_data = compare_csv_files(csv_files, labels)
    print_comparison_table(comparison_data)
    plot_comparison_bar_graph(comparison_data)
    
    return comparison_data


def write_summary_csv(outfile, network, mode, ped, bus, km_per_pax):
    """Write a comprehensive summary row with all metrics to CSV."""
    
    # Get v2 metrics with fallbacks
    vehicle_km = bus.get('vehicle_km', bus.get('total_mileage_km', 0))
    occupied_km = bus.get('occupied_km', bus.get('occupied_distance_km', 0))
    passenger_km = bus.get('passenger_km', 0)
    avg_occupancy = bus.get('avg_occupancy', 0)
    
    # For fixed-route, compute derived values if not present
    total_customers = bus.get('total_customers', bus.get('total_boardings', 0))
    bus_count = bus.get('bus_count', 0)
    avg_customers_per_bus = bus.get(
        'avg_customers_per_bus',
        (total_customers / bus_count) if bus_count else 0
    )
    
    row = {
        # === Metadata ===
        "network": network,
        "mode": mode,
        
        # === Pedestrian: Counts ===
        "total_persons": ped.get("total_persons", 0),
        "total_persons_generated": ped.get("total_persons_generated", ped.get("total_persons", 0)),
        "unfinished_persons": ped.get("unfinished_persons", 0),
        "completion_rate_pct": ped.get("completion_rate_pct", 0),
        "transit_riders": ped.get("riders_count", 0),
        "walk_only": ped.get("walkers_count", 0),
        "transit_share_pct": ped.get("rider_percentage", 0),
        
        # === Pedestrian: Transit Riders ===
        "rider_avg_travel_time": ped.get("avg_travel_time", 0),
        "rider_avg_waiting_time": ped.get("avg_waiting_time", 0),
        "rider_avg_riding_time": ped.get("avg_riding_time", 0),
        "rider_avg_ride_distance_km": ped.get("avg_trip_distance_km", 0),
        "rider_avg_walking_time": ped.get("rider_avg_walking_time", 0),
        "rider_avg_walking_distance_m": ped.get("rider_avg_walking_distance_m", 0),
        "rider_avg_speed_kmh": ped.get("rider_avg_speed_kmh", 0),
        "rider_avg_time_loss": ped.get("rider_avg_time_loss", 0),
        "rider_avg_ride_time_loss": ped.get("rider_avg_ride_time_loss", 0),
        "rider_avg_walk_time_loss": ped.get("rider_avg_walk_time_loss", 0),
        "rider_avg_ride_waiting_time": ped.get("rider_avg_ride_waiting_time", 0),
        
        # === Pedestrian: Walk-Only ===
        "walker_avg_travel_time": ped.get("walker_avg_travel_time", 0),
        "walker_avg_walking_time": ped.get("walker_avg_walking_time", 0),
        "walker_avg_walking_distance_m": ped.get("walker_avg_walking_distance_m", 0),
        "walker_avg_speed_kmh": ped.get("walker_avg_speed_kmh", 0),
        "walker_avg_time_loss": ped.get("walker_avg_time_loss", 0),
        "walker_avg_walk_time_loss": ped.get("walker_avg_walk_time_loss", 0),
        
        # === Pedestrian: All Combined ===
        "all_avg_travel_time": ped.get("all_avg_travel_time", 0),
        "all_avg_walking_time": ped.get("avg_walking_time", 0),
        "all_avg_walking_distance_m": ped.get("avg_walking_distance_m", 0),
        "all_avg_speed_kmh": ped.get("all_avg_speed_kmh", 0),
        "all_avg_time_loss": ped.get("all_avg_time_loss", 0),
        "all_avg_walk_time_loss": ped.get("all_avg_walk_time_loss", 0),
        
        # === Bus: Basic ===
        "bus_count": bus_count,
        "total_mileage_km": bus.get("total_mileage_km", 0),
        "avg_mileage_km": bus.get("avg_mileage_km", 0),
        "total_customers": total_customers,
        "avg_customers_per_bus": avg_customers_per_bus,
        
        # === Bus: DRT-specific (will be 0 for fixed) ===
        "drt_requests_served": bus.get("drt_requests_served", bus.get("total_customers", 0)),
        "occupied_distance_km": bus.get("occupied_distance_km", 0),
        "occupancy_rate_pct": bus.get("occupancy_rate", 0),
        
        # === Bus: v2 Distance & Occupancy ===
        "vehicle_km": vehicle_km,
        "occupied_km": occupied_km,
        "passenger_km": passenger_km,
        "avg_occupancy_pax_per_veh": avg_occupancy,
        "max_onboard": bus.get("max_onboard", 0),
        "max_onboard_vehicle_id": bus.get("max_onboard_vehicle_id", ""),
        
        # === Efficiency ===
        "km_per_passenger": km_per_pax,
    }
    
    import csv
    write_header = not Path(outfile).exists()
    with open(outfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tripinfo", nargs="?", default="tripinfo.output.csv")
    parser.add_argument("stops", nargs="?", default="stop.output.csv")
    parser.add_argument("--compare", action="store_true")

    parser.add_argument("--fcd", type=str, default=None,
                    help="Optional FCD CSV (xml2csv output). Enables occupied_distance_km for fixed buses without TraCI.")
    parser.add_argument("--netfile", type=str, default=None,
                    help="Optional SUMO .net.xml (fallback occupied-km estimate from stop positions if --fcd not provided).")
    parser.add_argument("--summary-csv", type=str, help="Output summary CSV for plotting")
    parser.add_argument("--network", type=str, default="", help="Network label for summary CSV")
    parser.add_argument("--mode", type=str, default="", help="Mode label for summary CSV (drt/fixed)")
    args, unknown = parser.parse_known_args()

    if args.compare:
        return compare_main()

    print(f"Loading: {args.tripinfo}")
    tripinfo_df = load_csv(args.tripinfo)
    print(f"Loading: {args.stops}")
    stops_df = load_csv(args.stops)
    fcd_df = load_csv(args.fcd) if args.fcd else pd.DataFrame()
    if tripinfo_df.empty:
        print("Error: Could not load tripinfo data")
        return

    mode = detect_mode(tripinfo_df)
    print(f"Detected mode: {mode.upper()}")
    ped_results = analyze_pedestrians(tripinfo_df)
    if mode == 'drt':
        bus_results = analyze_buses_drt(tripinfo_df, stops_df)
    else:
        bus_results = analyze_buses_fixed(tripinfo_df, stops_df, netfile=args.netfile, fcd_df=fcd_df)
    km_per_pax = compute_km_per_passenger(bus_results, ped_results)
    print_results(ped_results, bus_results, km_per_pax, mode)

    # Write summary CSV if requested
    if args.summary_csv:
        write_summary_csv(
            args.summary_csv,
            args.network,
            args.mode or mode,
            ped_results,
            bus_results,
            km_per_pax
        )

    return ped_results, bus_results, km_per_pax

if __name__ == "__main__":
    main()