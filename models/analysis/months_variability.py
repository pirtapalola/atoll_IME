# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import geopandas as gpd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr
from matplotlib.ticker import MaxNLocator, MultipleLocator

# Load the data
df = pd.read_csv("data/results/monthly_counts_v2.csv")
annual_delta_df = pd.read_csv("data/results/annual_delta_chl_2003_2024.csv")
coords = pd.read_csv('data/atolls/roi_3_atolls.csv')
coords = coords.drop(labels=["region", "total_atoll_area_sqkm", "rois", "lat_max", "lon_min", "lat_min", "lon_max"],
                     axis=1)

# Remove high islands
exclude_islands = [
    "Tahiti", "Moorea", "Raiatea", "Bora_Bora", "Huahine", "Gambiers",
    "Rimatara", "Rurutu", "Tubuai", "Raivavae", "Akiaki", "Nukutavake",
    "Maiao", "Maupiti", "Henderson", "Pitcairn", "Makatea", "Tepoto_Nord", "Mauke"
]
df = df.loc[~df['atoll'].isin(exclude_islands)]

df_avg = df.groupby('atoll')['months_IME'].mean().reset_index(name='months_IME')

sns.histplot(df_avg['months_IME'], bins=70, kde=False, color='skyblue')
median_value = df_avg['months_IME'].median()
plt.axvline(median_value, color='darkblue', linestyle='--', linewidth=2, label=f'Median = {median_value:.2f}')
plt.xlim(left=0, right=12)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.xlabel('Average annual IME duration (months)', fontsize=12)
plt.ylabel('Number of atolls', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(median_value)

# Step 1: Merge the two DataFrames on 'atoll'
df_delta_avg = annual_delta_df.groupby('atoll')['delta_chl_a'].mean().reset_index(name='delta_chl_a')
df_delta_avg['log_delta_chl_a'] = np.log(df_delta_avg['delta_chl_a'])
df_scatter = pd.merge(df_delta_avg, df_avg, on='atoll')

# Step 2: Calculate Pearson correlation
# r_val, p_val = pearsonr(df_scatter['months_IME'], df_scatter['log_delta_chl_a'])
r_val, p_val = spearmanr(df_scatter['months_IME'], df_scatter['log_delta_chl_a'])
print(r_val, p_val)

# Step 2: Scatter plot
sns.regplot(
    data=df_scatter,
    x='months_IME',
    y='log_delta_chl_a',
    scatter_kws={'color': 'seagreen', 's': 60},
    line_kws={'color': 'black', 'linestyle': '--'},
    ci=None
)

plt.text(
    0.05, 0.95, f"$\\rho$ = {r_val:.2f}, $p$ < 0.001",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top'
)

# Step 3: Add labels and formatting
plt.xlabel('IME duration (months)', fontsize=12)
plt.ylabel('Log(IME magnitude) [mg/m$^3$]', fontsize=12)
plt.ylim(bottom=0)
plt.xlim(left=2)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

"""
# Open a NetCDF file
chl_path = "data/masked/roi3_masked_4km_1997_2024.nc"
chl_masked = xr.open_dataset(chl_path)
chl_roi = chl_masked["chlor_a"]
chl_roi = chl_roi.sel(time=slice("2003", "2024"))
chl_roi_mean = chl_roi.mean(dim="time")

total_lats = chl_roi['lat'].values
total_lons = chl_roi['lon'].values

high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")

# Load atoll data
df1 = pd.read_csv("data/atolls/atoll_data.csv", encoding='latin1')

# Calculate lagoon_openness and add it as a new column
df1['lagoon_openness'] = df1['total_width_channels_km'] / df1['outer_perimeter_km']

# Create a new column with distance to the closest island
df1['distance_nearest_km'] = \
    df1[['distance_nearest_atoll_km', 'distance_nearest_high_island_km']].min(axis=1)

# Keep only necessary columns
df1_subset = df1[['atoll', 'total_atoll_area_sqkm', 'lagoon_openness', 'no_channels', 'distance_nearest_km']]

# Compute mean and std for each atoll
summary = df.groupby('atoll')['months_IME'].agg(['mean', 'std']).reset_index()
summary['mean'] = [round(i, 0) for i in summary['mean']]
summary['std'] = [round(i, 2) for i in summary['std']]
summary.to_csv('summary_IME_months.csv', index=False)

# Specify a threshold
total_years = len(range(2003, 2025))  # 22 years
threshold = 0.5
min_years_with_x = int(total_years * threshold)

# Count how many years each atoll had a specific number of months
no_months = 11
counts_x = df[df['months_IME'] >= no_months]
print(counts_x)
atoll_x_counts = counts_x.groupby('atoll').size().reset_index(name='years_with_x')

# Filter atolls meeting or exceeding the threshold
majority_x_atolls = atoll_x_counts[atoll_x_counts['years_with_x'] >= min_years_with_x]

# Save the result
print(len(majority_x_atolls['atoll']))
print(majority_x_atolls.sort_values(by='years_with_x', ascending=False))

merged_df = pd.merge(summary, df1_subset, how='inner', left_on='atoll', right_on='atoll')

# Merge with coordinates
data = pd.merge(summary, coords, on='atoll')

# Create figure
fig, ax = plt.subplots(figsize=(20, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot background data first
mesh = ax.pcolormesh(total_lons, total_lats, chl_roi_mean,
                     transform=ccrs.PlateCarree(), cmap='viridis', alpha=0.9)

# Plot ellipses on top of pcolormesh
atoll_ellipses1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, transform=ccrs.PlateCarree())
atoll_ellipses2.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, transform=ccrs.PlateCarree())
high_island_ellipses.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, transform=ccrs.PlateCarree())

# Plot scatter on top to prevent override
sc = ax.scatter(
    data['lon'], data['lat'],
    c=data['mean'],  s=20 + data['std'] * 2,
    cmap='plasma', edgecolors='black', alpha=0.5, transform=ccrs.PlateCarree()
)

# Add gridlines with labels
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.9, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Add a north arrow to the top right corner
ax.annotate('N',
            xy=(0.97, 0.93), xycoords='axes fraction',
            ha='center', va='center',
            fontsize=14, fontweight='bold')
# bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black"))

ax.annotate('', xy=(0.97, 0.90), xytext=(0.97, 0.82),
            xycoords='axes fraction',
            arrowprops=dict(facecolor='black', width=0.5, headwidth=10))

cbar2 = plt.colorbar(sc, ax=ax, orientation='vertical', ticks=np.arange(1, 13, 1), shrink=0.7, pad=0.07)
cbar2.set_label('IME months')  # Chlorophyll-a (mg/m$^3$)

# Title and display
plt.savefig('data/results/figures/IME_months_map.png', dpi=500, bbox_inches='tight')

# Sort by mean chl-a and atoll name
mean_order = data.reset_index().sort_values(by=['mean', 'atoll'],
    ascending=[False, True]
)['atoll']

# Create the box plot
plt.figure(figsize=(20, 8))
ax = sns.boxplot(
    x='atoll',
    y='months_IME',
    data=df,
    order=mean_order,
    palette='plasma',
    showfliers=False
)

# Set axes
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel('Atoll')
ax.set_ylabel('IME months')
ax.set_yticks(np.arange(1, 13, 1))

# Save the plot
plt.savefig('data/results/figures/IME_months_boxplot.png', dpi=500, bbox_inches='tight')

# Include log-transformed values
merged_df['log_area'] = np.log(merged_df['total_atoll_area_sqkm'])
merged_df['log_months'] = np.log(merged_df['mean'])

# Save as a csv file
merged_df.to_csv('data/results/atoll_months_summary.csv')

# Calculate Pearson correlation
r, p = pearsonr(merged_df['log_area'], merged_df['log_months'])

# Create scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x='log_area', y='log_months', data=merged_df, scatter_kws={'s': 40}, line_kws={'color': 'red'})

# Add annotation with r and p
plt.annotate(f'r = {r: .2f}\n p = {p: .2f}', xy=(0.05, 0.95), xycoords='axes fraction',
             ha='left', va='top', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

# Axis labels and save
plt.xlabel('Log Atoll Area')
plt.ylabel('Log IME months')
plt.tight_layout()
plt.savefig('data/results/figures/area_months.png', dpi=500)"""
