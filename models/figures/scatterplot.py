import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV files
df_extent = pd.read_csv('data/results/IME_monthly_extent_2003_2004.csv')
df_delta = pd.read_csv('data/hotspots/IME/chl_hotspots_2003_2024_filled.csv')

# Step 2: Merge the two DataFrames on atoll, year, and month
df_merged = pd.merge(df_extent, df_delta, on=['atoll', 'year', 'month'])

# Step 3: Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_merged, x='spatial_extent_km2', y='delta', hue='atoll', alpha=0.7, legend=False)

plt.xlabel('Spatial extent (km²)')
plt.ylabel('IME magnitude (mg/m$^3$)')
plt.grid(True)
plt.tight_layout()
plt.show()
