"""
Generate priority hypothesis graphs.
Focus on H1/H2, H3, H5, and Efficiency metrics.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("experiments/2026-02-06_064047/results")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(RESULTS_DIR / "experiment_results_averaged.csv")

# Parse system and demand
df['system'] = df['network'].str.extract(r'5x5_(\w+)')
demand_map = {'p120b2': 30, 'p60b2': 60, 'p38b2': 95, 'p27b2': 133, 'p21b2': 171, 'p15b4': 240}
df['demand_level'] = df['demand'].map(demand_map)

# Filter high traffic (main comparison)
high_traffic = df[df['traffic'] == 'high'].copy()
low_traffic = df[df['traffic'] == 'low'].copy()

# Color scheme
colors = {
    'DRT3': '#1f77b4', 'DRT6': '#2ca02c', 'DRT15': '#ff7f0e', 'DRT30': '#d62728',
    'FixedLow': '#9467bd', 'FixedHigh': '#8c564b'
}

# =============================================================================
# GRAPH 1: Travel Time vs Demand
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for system in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    data = high_traffic[high_traffic['system'] == system].sort_values('demand_level')
    if not data.empty:
        ax.plot(data['demand_level'], data['avg_travel_time_mean'], 
                marker='o', linewidth=2, markersize=8, 
                color=colors.get(system, 'gray'), label=system)
        ax.fill_between(data['demand_level'],
                        data['avg_travel_time_mean'] - data['avg_travel_time_se'],
                        data['avg_travel_time_mean'] + data['avg_travel_time_se'],
                        alpha=0.2, color=colors.get(system, 'gray'))

ax.set_xlabel('Passenger Demand (pax/hour)', fontsize=12)
ax.set_ylabel('Average Travel Time (seconds)', fontsize=12)
ax.set_title('Average Travel Time by System and Demand Level', fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(20, 260)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph1_TravelTime_vs_Demand.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph1_TravelTime_vs_Demand.png")
plt.close()

# =============================================================================
# GRAPH 2: Waiting Time vs Demand
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for system in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    data = high_traffic[high_traffic['system'] == system].sort_values('demand_level')
    if not data.empty:
        ax.plot(data['demand_level'], data['avg_waiting_time_mean'], 
                marker='s', linewidth=2, markersize=8,
                color=colors.get(system, 'gray'), label=system)

ax.set_xlabel('Passenger Demand (pax/hour)', fontsize=12)
ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
ax.set_title('Average Waiting Time by System and Demand Level', fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph2_WaitingTime_vs_Demand.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph2_WaitingTime_vs_Demand.png")
plt.close()

# =============================================================================
# GRAPH 3: KM per Passenger (Efficiency) vs Demand
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for system in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    data = high_traffic[high_traffic['system'] == system].sort_values('demand_level')
    if not data.empty:
        ax.plot(data['demand_level'], data['km_per_passenger_mean'], 
                marker='o', linewidth=2, markersize=8,
                color=colors.get(system, 'gray'), label=system)
        ax.fill_between(data['demand_level'],
                        data['km_per_passenger_mean'] - data['km_per_passenger_se'],
                        data['km_per_passenger_mean'] + data['km_per_passenger_se'],
                        alpha=0.2, color=colors.get(system, 'gray'))

ax.set_xlabel('Passenger Demand (pax/hour)', fontsize=12)
ax.set_ylabel('Vehicle-KM per Passenger', fontsize=12)
ax.set_title('Vehicle Efficiency: KM per Passenger by System', fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph3_KM_per_Passenger.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph3_KM_per_Passenger.png")
plt.close()

# =============================================================================
# GRAPH 4: Total Fleet Mileage vs Demand
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for system in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    data = high_traffic[high_traffic['system'] == system].sort_values('demand_level')
    if not data.empty:
        ax.plot(data['demand_level'], data['total_mileage_km_mean'], 
                marker='^', linewidth=2, markersize=8,
                color=colors.get(system, 'gray'), label=system)

ax.set_xlabel('Passenger Demand (pax/hour)', fontsize=12)
ax.set_ylabel('Total Fleet Mileage (km)', fontsize=12)
ax.set_title('Total Fleet Mileage by System and Demand Level', fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph4_Total_Mileage.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph4_Total_Mileage.png")
plt.close()

# =============================================================================
# GRAPH 5: Vehicle Occupancy vs Demand
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for system in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    data = high_traffic[high_traffic['system'] == system].sort_values('demand_level')
    if not data.empty:
        ax.plot(data['demand_level'], data['avg_occupancy_mean'], 
                marker='D', linewidth=2, markersize=8,
                color=colors.get(system, 'gray'), label=system)

ax.set_xlabel('Passenger Demand (pax/hour)', fontsize=12)
ax.set_ylabel('Average Vehicle Occupancy (passengers)', fontsize=12)
ax.set_title('Average Vehicle Occupancy by System and Demand Level', fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph5_Vehicle_Occupancy.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph5_Vehicle_Occupancy.png")
plt.close()

# =============================================================================
# GRAPH 6: Fleet Capacity Limits - Maximum Sustainable Demand per Fleet
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Define the maximum tested demand for each fleet configuration
# DRT6@95 failed, so DRT6 max sustainable = 60; DRT6@95 not in data means failure
capacity_data = [
    {'system': 'DRT3', 'fleet_size': 3, 'max_demand': 30, 'status': 'tested'},
    {'system': 'DRT6', 'fleet_size': 6, 'max_demand': 60, 'status': 'tested'},  # Failed at 95
    {'system': 'DRT15', 'fleet_size': 15, 'max_demand': 133, 'status': 'tested'},
    {'system': 'DRT30', 'fleet_size': 30, 'max_demand': 240, 'status': 'tested'},
]

capacity_df = pd.DataFrame(capacity_data)

# Create bar chart showing max sustainable demand
bars = ax.bar(capacity_df['system'], capacity_df['max_demand'], 
              color=[colors[s] for s in capacity_df['system']],
              edgecolor='black', linewidth=1.5)

# Add demand level labels on bars
for bar, (_, row) in zip(bars, capacity_df.iterrows()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)} pax/h',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add fleet size labels
ax.set_xticks(range(len(capacity_df)))
ax.set_xticklabels([f"{row['system']}\n({row['fleet_size']} vans)" 
                    for _, row in capacity_df.iterrows()], fontsize=11)

ax.set_xlabel('Fleet Configuration', fontsize=12)
ax.set_ylabel('Maximum Sustainable Demand (pax/hour)', fontsize=12)
ax.set_title('DRT Fleet Capacity Limits: Maximum Tested Demand Level', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 280)

# Add annotation about DRT6 failure
ax.annotate('DRT6 failed at 95 pax/h\n(system overwhelmed)', 
            xy=(1, 60), xytext=(1.5, 120),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray', ha='left')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph6_Fleet_Capacity_Limits.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph6_Fleet_Capacity_Limits.png")
plt.close()

# =============================================================================
# GRAPH 7: Traffic Impact on Travel Time (High vs Low Traffic)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Get systems with both high and low traffic data at d133
systems_with_both = ['DRT15', 'FixedLow']

x = np.arange(len(systems_with_both))
width = 0.35

high_times = []
low_times = []
high_se = []
low_se = []

for system in systems_with_both:
    # High traffic at d133
    h_data = high_traffic[(high_traffic['system'] == system) & (high_traffic['demand_level'] == 133)]
    if not h_data.empty:
        high_times.append(h_data['avg_travel_time_mean'].values[0])
        high_se.append(h_data['avg_travel_time_se'].values[0])
    else:
        high_times.append(0)
        high_se.append(0)
    
    # Low traffic at d133
    l_data = low_traffic[(low_traffic['system'] == system) & (low_traffic['demand_level'] == 133)]
    if not l_data.empty:
        low_times.append(l_data['avg_travel_time_mean'].values[0])
        low_se.append(l_data['avg_travel_time_se'].values[0])
    else:
        low_times.append(0)
        low_se.append(0)

bars1 = ax.bar(x - width/2, high_times, width, yerr=high_se, capsize=5,
               label='High Traffic', color='#e74c3c', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, low_times, width, yerr=low_se, capsize=5,
               label='Low Traffic', color='#3498db', edgecolor='black', linewidth=1.2)

ax.set_xlabel('System', fontsize=12)
ax.set_ylabel('Average Travel Time (seconds)', fontsize=12)
ax.set_title('Traffic Condition Impact on Travel Time (at 133 pax/hour)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(systems_with_both, fontsize=11)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

# Add percentage difference annotations
for i, (h, l) in enumerate(zip(high_times, low_times)):
    if h > 0 and l > 0:
        pct_diff = ((h - l) / l) * 100
        ax.annotate(f'+{pct_diff:.1f}%', xy=(i, max(h, l) + 30), 
                    ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph7_Traffic_Impact_TravelTime.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph7_Traffic_Impact_TravelTime.png")
plt.close()

# =============================================================================
# GRAPH 8: Traffic Impact on Waiting Time (High vs Low Traffic)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

high_wait = []
low_wait = []
high_wait_se = []
low_wait_se = []

for system in systems_with_both:
    # High traffic at d133
    h_data = high_traffic[(high_traffic['system'] == system) & (high_traffic['demand_level'] == 133)]
    if not h_data.empty:
        high_wait.append(h_data['avg_waiting_time_mean'].values[0])
        high_wait_se.append(h_data['avg_waiting_time_se'].values[0])
    else:
        high_wait.append(0)
        high_wait_se.append(0)
    
    # Low traffic at d133
    l_data = low_traffic[(low_traffic['system'] == system) & (low_traffic['demand_level'] == 133)]
    if not l_data.empty:
        low_wait.append(l_data['avg_waiting_time_mean'].values[0])
        low_wait_se.append(l_data['avg_waiting_time_se'].values[0])
    else:
        low_wait.append(0)
        low_wait_se.append(0)

bars1 = ax.bar(x - width/2, high_wait, width, yerr=high_wait_se, capsize=5,
               label='High Traffic', color='#e74c3c', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, low_wait, width, yerr=low_wait_se, capsize=5,
               label='Low Traffic', color='#3498db', edgecolor='black', linewidth=1.2)

ax.set_xlabel('System', fontsize=12)
ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
ax.set_title('Traffic Condition Impact on Waiting Time (at 133 pax/hour)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(systems_with_both, fontsize=11)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

# Add percentage difference annotations
for i, (h, l) in enumerate(zip(high_wait, low_wait)):
    if h > 0 and l > 0:
        pct_diff = ((h - l) / l) * 100
        ax.annotate(f'+{pct_diff:.1f}%', xy=(i, max(h, l) + 15), 
                    ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph8_Traffic_Impact_WaitingTime.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph8_Traffic_Impact_WaitingTime.png")
plt.close()

# =============================================================================
# GRAPH 9: Small Fleet Comparison - DRT3 vs DRT6 vs DRT15 at d60
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Get data for small fleets at d60
small_fleets = ['DRT3', 'DRT6', 'DRT15']
d60_data = high_traffic[(high_traffic['demand_level'] == 60) | 
                         ((high_traffic['system'] == 'DRT3') & (high_traffic['demand_level'] == 30))]

x = np.arange(len(small_fleets))
width = 0.35

travel_times = []
travel_se = []
wait_times = []
wait_se = []

for system in small_fleets:
    if system == 'DRT3':
        # DRT3 only tested at d30
        data = high_traffic[(high_traffic['system'] == 'DRT3') & (high_traffic['demand_level'] == 30)]
    else:
        data = high_traffic[(high_traffic['system'] == system) & (high_traffic['demand_level'] == 60)]
    
    if not data.empty:
        travel_times.append(data['avg_travel_time_mean'].values[0])
        travel_se.append(data['avg_travel_time_se'].values[0])
        wait_times.append(data['avg_waiting_time_mean'].values[0])
        wait_se.append(data['avg_waiting_time_se'].values[0])
    else:
        travel_times.append(0)
        travel_se.append(0)
        wait_times.append(0)
        wait_se.append(0)

bars1 = ax.bar(x - width/2, travel_times, width, yerr=travel_se, capsize=5,
               label='Travel Time', color=[colors[s] for s in small_fleets], 
               edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, wait_times, width, yerr=wait_se, capsize=5,
               label='Waiting Time', color=[colors[s] for s in small_fleets], 
               edgecolor='black', linewidth=1.2, alpha=0.6, hatch='//')

ax.set_xlabel('Fleet Configuration', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Small Fleet Performance Comparison\n(DRT3@d30, DRT6@d60, DRT15@d60)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{s}\n({s[3:]} vans)' for s in small_fleets], fontsize=11)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph9_Small_Fleet_Comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph9_Small_Fleet_Comparison.png")
plt.close()

# =============================================================================
# GRAPH 10: H8 - DRT3 vs FixedLow at Very Low Demand (d30)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

systems_h8 = ['DRT3', 'FixedLow']
x = np.arange(len(systems_h8))
width = 0.25

# Metrics to compare
metrics = {
    'Travel Time': ('avg_travel_time_mean', 'avg_travel_time_se'),
    'Waiting Time': ('avg_waiting_time_mean', 'avg_waiting_time_se'),
    'KM/Passenger': ('km_per_passenger_mean', 'km_per_passenger_se'),
}

# Get data at d30
d30_high = high_traffic[high_traffic['demand_level'] == 30]

travel = []
travel_se_list = []
wait = []
wait_se_list = []

for system in systems_h8:
    data = d30_high[d30_high['system'] == system]
    if not data.empty:
        travel.append(data['avg_travel_time_mean'].values[0])
        travel_se_list.append(data['avg_travel_time_se'].values[0])
        wait.append(data['avg_waiting_time_mean'].values[0])
        wait_se_list.append(data['avg_waiting_time_se'].values[0])
    else:
        travel.append(0)
        travel_se_list.append(0)
        wait.append(0)
        wait_se_list.append(0)

bar_colors = [colors['DRT3'], colors['FixedLow']]
bars1 = ax.bar(x - width/2, travel, width, yerr=travel_se_list, capsize=5,
               label='Travel Time', color=bar_colors, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, wait, width, yerr=wait_se_list, capsize=5,
               label='Waiting Time', color=bar_colors, edgecolor='black', linewidth=1.2, 
               alpha=0.6, hatch='//')

ax.set_xlabel('System', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Minimal DRT vs Fixed at Very Low Demand (30 pax/hour)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['DRT3\n(3 vans, 24 pax capacity)', 'FixedLow\n(6 buses, 510 pax capacity)'], fontsize=10)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

# Add improvement annotation
if travel[0] > 0 and travel[1] > 0:
    improvement = ((travel[1] - travel[0]) / travel[1]) * 100
    ax.annotate(f'DRT3 {improvement:.0f}% faster', 
                xy=(0.5, max(travel) * 0.7), fontsize=11, ha='center',
                color='green' if improvement > 0 else 'red')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph10_H8_DRT3_vs_FixedLow_d30.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph10_H8_DRT3_vs_FixedLow_d30.png")
plt.close()

# =============================================================================
# GRAPH 11: Stacked Bar - Time Breakdown (Waiting + Riding + Walking)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Select representative scenarios at d133 for comparison
systems_stack = ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']
d133_data = high_traffic[high_traffic['demand_level'] == 133]

waiting = []
riding = []
walking = []
labels = []

for system in systems_stack:
    data = d133_data[d133_data['system'] == system]
    if not data.empty:
        wait_val = data['avg_waiting_time_mean'].values[0]
        ride_val = data['avg_riding_time_mean'].values[0]
        # Walking time: travel - wait - ride (for Fixed, includes walk to/from stops)
        travel_val = data['avg_travel_time_mean'].values[0]
        walk_val = max(0, travel_val - wait_val - ride_val)
        
        waiting.append(wait_val)
        riding.append(ride_val)
        walking.append(walk_val)
        labels.append(system)

x = np.arange(len(labels))
width = 0.6

# Stacked bars
p1 = ax.bar(x, waiting, width, label='Waiting Time', color='#e74c3c', edgecolor='black')
p2 = ax.bar(x, riding, width, bottom=waiting, label='Riding Time', color='#3498db', edgecolor='black')
p3 = ax.bar(x, walking, width, bottom=np.array(waiting) + np.array(riding), 
            label='Walking Time', color='#2ecc71', edgecolor='black')

# Add value labels on each segment
for i, (w, r, wlk) in enumerate(zip(waiting, riding, walking)):
    # Waiting
    ax.text(i, w/2, f'{w:.0f}s', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    # Riding
    ax.text(i, w + r/2, f'{r:.0f}s', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    # Walking (only if significant)
    if wlk > 50:
        ax.text(i, w + r + wlk/2, f'{wlk:.0f}s', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    # Total on top
    total = w + r + wlk
    ax.text(i, total + 15, f'Total: {total:.0f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('System', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Travel Time Breakdown: Waiting vs Riding vs Walking\n(at 133 pax/hour, High Traffic)', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{l}\n({"DRT" if "DRT" in l else "Fixed"})' for l in labels], fontsize=11)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(np.array(waiting) + np.array(riding) + np.array(walking)) * 1.15)

# Add insights annotation
ax.annotate('DRT: Higher wait, but door-to-door (no walking)\nFixed: Lower wait, but requires walking to stops', 
            xy=(0.5, 0.02), xycoords='axes fraction', fontsize=9, 
            ha='center', style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph11_Stacked_Time_Breakdown.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph11_Stacked_Time_Breakdown.png")
plt.close()

# =============================================================================
# GRAPH 12: Service Completion Rate (riders served / total demand)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate completion rate from existing data
for system in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    data = high_traffic[high_traffic['system'] == system].sort_values('demand_level')
    if not data.empty:
        # Completion rate = riders / total_persons * 100
        completion_rate = (data['riders_count_mean'] / data['total_persons_mean']) * 100
        ax.plot(data['demand_level'], completion_rate,
                marker='o', linewidth=2, markersize=8,
                label=system, color=colors.get(system, 'gray'))

ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% served')
ax.axhline(y=90, color='orange', linestyle=':', alpha=0.5, label='90% threshold')

ax.set_xlabel('Passenger Demand (pax/hour)', fontsize=12)
ax.set_ylabel('Completion Rate (%)', fontsize=12)
ax.set_title('Service Completion Rate', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(50, 105)
ax.set_xlim(20, 260)

# Add insight annotation
ax.annotate('DRT: 100% completion (door-to-door guarantee)\nFixed: Some passengers walk-only or unserved', 
            xy=(0.98, 0.02), xycoords='axes fraction', fontsize=9, 
            ha='right', va='bottom', style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph12_Completion_Rate.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph12_Completion_Rate.png")
plt.close()

# =============================================================================
# GRAPH 13: Vehicle Utilization (Avg Occupancy Rate)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for system in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    data = high_traffic[high_traffic['system'] == system].sort_values('demand_level')
    if not data.empty:
        ax.plot(data['demand_level'], data['avg_occupancy_mean'],
                marker='o', linewidth=2, markersize=8,
                label=system, color=colors.get(system, 'gray'))
        # Add error bars
        ax.fill_between(data['demand_level'],
                       data['avg_occupancy_mean'] - data['avg_occupancy_se'],
                       data['avg_occupancy_mean'] + data['avg_occupancy_se'],
                       alpha=0.2, color=colors.get(system, 'gray'))

ax.set_xlabel('Passenger Demand (pax/hour)', fontsize=12)
ax.set_ylabel('Average Vehicle Occupancy (passengers)', fontsize=12)
ax.set_title('Vehicle Utilization: How Full Are the Vehicles?\n(Higher = More Efficient Use of Capacity)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(20, 260)

# Add capacity reference lines
ax.axhline(y=8, color='blue', linestyle=':', alpha=0.4, label='DRT van capacity (8)')
# Note: Fixed capacity is 85 but occupancy is far below, so no need to show

# Add insight
ax.annotate('DRT vans: 1.5-1.8 pax avg (capacity: 8)\nFixed buses: 0.1-0.6 pax avg (capacity: 85)', 
            xy=(0.98, 0.98), xycoords='axes fraction', fontsize=9, 
            ha='right', va='top', style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph13_Vehicle_Utilization.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph13_Vehicle_Utilization.png")
plt.close()

# =============================================================================
# GRAPH 14: Resource Efficiency Summary (Bar Chart)
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 7))

# Compare at d133 (crossover point) where we have data for all systems
d133 = high_traffic[high_traffic['demand_level'] == 133]

systems = ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']
x = np.arange(len(systems))
width = 0.35

km_per_pax = []
completion = []
labels = []

for system in systems:
    data = d133[d133['system'] == system]
    if not data.empty:
        km_per_pax.append(data['km_per_passenger_mean'].values[0])
        comp = (data['riders_count_mean'].values[0] / data['total_persons_mean'].values[0]) * 100
        completion.append(comp)
        labels.append(system)
    else:
        km_per_pax.append(0)
        completion.append(0)
        labels.append(system)

# Create twin axis
ax2 = ax.twinx()

# Bars for KM per passenger
bar_colors = [colors.get(s, 'gray') for s in systems]
bars1 = ax.bar(x - width/2, km_per_pax, width, label='KM per Passenger', 
               color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.8)

# Bars for completion rate
bars2 = ax2.bar(x + width/2, completion, width, label='Completion Rate (%)', 
                color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.5, hatch='//')

# Labels
ax.set_xlabel('System', fontsize=12)
ax.set_ylabel('KM per Passenger (lower = better)', fontsize=12, color='black')
ax2.set_ylabel('Completion Rate % (higher = better)', fontsize=12, color='gray')

ax.set_title('Resource Efficiency vs Service Reliability\nat 133 pax/hour (Crossover Demand)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{s}\n({"DRT" if "DRT" in s else "Fixed"})' for s in labels], fontsize=10)

# Add value labels
for i, (kpp, comp) in enumerate(zip(km_per_pax, completion)):
    if kpp > 0:
        ax.text(i - width/2, kpp + 0.05, f'{kpp:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if comp > 0:
        ax2.text(i + width/2, comp + 1, f'{comp:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='gray')

# Legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)

ax.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(60, 110)

# Add insight
ax.annotate('Trade-off: DRT uses more km/pax but serves everyone\nFixed is km-efficient but may leave passengers unserved', 
            xy=(0.5, 0.02), xycoords='axes fraction', fontsize=9, 
            ha='center', style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Graph14_Resource_vs_Reliability.png', dpi=300, bbox_inches='tight')
print(f"Saved: Graph14_Resource_vs_Reliability.png")
plt.close()

# =============================================================================
# Print summary statistics
# =============================================================================
print("\n" + "="*70)
print("KEY METRICS SUMMARY")
print("="*70)

print("\n📊 Travel Time (seconds)")
for sys in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    d = high_traffic[high_traffic['system'] == sys]['avg_travel_time_mean'].mean()
    if not np.isnan(d):
        print(f"   {sys}: {d:.0f}s")

print("\n📊 KM per Passenger")
for sys in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    d = high_traffic[high_traffic['system'] == sys]['km_per_passenger_mean'].mean()
    if not np.isnan(d):
        print(f"   {sys}: {d:.2f} km/pax")

print("\n📊 Average Occupancy")
for sys in ['DRT15', 'DRT30', 'FixedLow', 'FixedHigh']:
    d = high_traffic[high_traffic['system'] == sys]['avg_occupancy_mean'].mean()
    if not np.isnan(d):
        print(f"   {sys}: {d:.2f} pax/veh")

print("\n📊 Traffic Impact at d133")
for sys in systems_with_both:
    h = high_traffic[(high_traffic['system'] == sys) & (high_traffic['demand_level'] == 133)]['avg_travel_time_mean']
    l = low_traffic[(low_traffic['system'] == sys) & (low_traffic['demand_level'] == 133)]['avg_travel_time_mean']
    if not h.empty and not l.empty:
        pct = ((h.values[0] - l.values[0]) / l.values[0]) * 100
        print(f"   {sys}: High traffic adds +{pct:.1f}% travel time")

print("\n" + "="*70)
print(f"All graphs saved to: {OUTPUT_DIR.absolute()}")
print("="*70)