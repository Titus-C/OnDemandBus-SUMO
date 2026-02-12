"""
Visualize experiment results comparing DRT vs Fixed-route.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load averaged results - accept command line argument or use default
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = "experiment_results_averaged.csv"

df = pd.read_csv(csv_path)

# Parse network label to extract mode info
df['system'] = df['network'].str.extract(r'5x5_(\w+)')
df['is_drt'] = df['mode'] == 'drt'

# Key metrics to plot - use actual column names from averaged CSV
metrics = [
    ('avg_travel_time_mean', 'Avg Travel Time (s)'),
    ('avg_waiting_time_mean', 'Avg Waiting Time (s)'),
    ('km_per_passenger_mean', 'KM per Passenger'),
    ('total_mileage_km_mean', 'Total Mileage (km)'),
    ('avg_occupancy_mean', 'Avg Occupancy'),
    ('riders_count_mean', 'Riders Count'),
]

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('DRT vs Fixed-Route Bus Comparison', fontsize=14, fontweight='bold')

palette = {'DRT15': '#3498db', 'DRT30': '#2980b9', 
           'FixedLow': '#e74c3c', 'FixedHigh': '#c0392b'}

for ax, (col, title) in zip(axes.flat, metrics):
    if col in df.columns:
        # Group by system and demand
        pivot = df.pivot_table(values=col, index=['traffic', 'demand'], 
                               columns='system', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax, color=[palette.get(c, 'gray') for c in pivot.columns])
        ax.set_title(title)
        ax.set_xlabel('')
        ax.legend(title='System')
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('experiment_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
print("\n" + "="*80)
print("SUMMARY: Mean metrics by system and demand level")
print("="*80)
summary = df.groupby(['system', 'traffic', 'demand'])[
    ['avg_travel_time_mean', 'avg_waiting_time_mean', 
     'km_per_passenger_mean', 'avg_occupancy_mean']
].mean().round(2)
print(summary.to_string())