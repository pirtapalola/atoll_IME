# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import geopandas as gpd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from matplotlib.ticker import MaxNLocator, MultipleLocator

# Load the data
df = pd.read_csv("data/results/annual_delta_chl_2003_2024.csv")
coords = pd.read_csv('data/atolls/roi_3_atolls.csv')
coords = coords.drop(labels=["region", "total_atoll_area_sqkm", "rois", "lat_max", "lon_min", "lat_min", "lon_max"],
                     axis=1)
atoll_months = pd.read_csv('data/results/atoll_months_summary.csv')
atoll_months = atoll_months[['atoll', 'mean', 'std']]
atoll_months = atoll_months.rename(columns={"mean": "IME_months", "std": "IME_months_std"})

# Remove high islands
exclude_islands = [
    "Tahiti", "Moorea", "Raiatea", "Bora_Bora", "Huahine", "Gambiers",
    "Rimatara", "Rurutu", "Tubuai", "Raivavae", "Akiaki", "Nukutavake",
    "Maiao", "Maupiti", "Henderson", "Pitcairn", "Makatea", "Tepoto_Nord", "Mauke"
]
df = df.loc[~df['atoll'].isin(exclude_islands)]

df['log_delta_chl_a'] = np.log(df['delta_chl_a'])
df_avg = df.groupby('atoll')['delta_chl_a'].mean().reset_index(name='delta_chl_a')

sns.histplot(df_avg['delta_chl_a'], bins=70, kde=False, color='lightgreen')
median_value = df_avg['delta_chl_a'].median()
plt.axvline(median_value, color='darkgreen', linestyle='--', linewidth=2, label=f'Median = {median_value:.2f}')
plt.xlim(left=0)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.xlabel('Average annual IME magnitude (mg/m$^3$)', fontsize=12)
plt.ylabel('Number of atolls', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(f"Median delta_chl_a: {median_value:.2f} mg/m^3")

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
df1_subset = df1[['atoll', 'total_atoll_area_sqkm', 'total_width_channels_km',
                  'lagoon_openness', 'no_channels', 'distance_nearest_km']]

# Compute mean and std for each atoll
summary = df.groupby('atoll')['delta_chl_a'].agg(['mean', 'std']).reset_index()

merged_df = pd.merge(summary, df1_subset, how='inner', left_on='atoll', right_on='atoll')
print(merged_df)

# Print the 5 lowest and highest atolls
print("5 Atolls with Lowest Values:")
print(summary.head(5))

print("5 Atolls with Highest Values:")
print(summary.tail(5))

# Merge with coordinates
data = pd.merge(merged_df, coords, on='atoll')
print(data)

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

cbar2 = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7, pad=0.07)
cbar2.set_label('IME magnitude (mg/m$^3$)')

# Title and display
plt.savefig('data/results/figures/IME_map.png', dpi=500, bbox_inches='tight')

# Compute the mean chl-a for sorting
mean_order = data.groupby('atoll')['mean'].mean().sort_values(ascending=False).index

# Create the box plot
plt.figure(figsize=(20, 8))
ax = sns.boxplot(
    x='atoll',
    y='delta_chl_a',
    data=df,
    order=mean_order,
    palette='plasma',
    showfliers=False
)

# Rotate and align x-tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Set axis labels
ax.set_xlabel('Atoll')
ax.set_ylabel('IME magnitude (mg/m$^3$)')

plt.tight_layout()

# Save the plot
plt.savefig('data/results/figures/IME_boxplot.png', dpi=500, bbox_inches='tight')


# Include log-transformed values
data['log_area'] = np.log(data['total_atoll_area_sqkm'])
data['log_chl'] = np.log(data['mean'])

# Merge with IME months data
data_summary = pd.merge(data, atoll_months, on='atoll')

# Save as a csv file
data_summary.to_csv('data/results/atoll_summary.csv', index=False)

# Calculate Pearson correlation
r, p = pearsonr(merged_df['log_area'], merged_df['log_chl'])

# Create scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x='log_area', y='log_chl', data=merged_df, scatter_kws={'s': 40}, line_kws={'color': 'red'})

# Add annotation with r and p
plt.annotate(f'r = {r: .2f}\np < 0.0001', xy=(0.05, 0.95), xycoords='axes fraction',
             ha='left', va='top', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

# Axis labels and save
plt.xlabel('Log Atoll Area')
plt.ylabel('Log Chlorophyll-a')
plt.tight_layout()
plt.savefig('data/results/figures/area_chl.png', dpi=500)"""
