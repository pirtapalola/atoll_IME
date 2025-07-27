# Import libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import colors

# Define the variable of interest
dataset = pd.read_csv('el_nino/MHW/MHW_p90_duration.csv')
column_of_interest = 'duration'
var_name = "90th percentile MHW duration"  # Chlorophyll-a (mg/m$^3$) SST (°C)
output_name = "MHW_p90_duration_map"

print(dataset[column_of_interest].min())
print(dataset[column_of_interest].max())

# Load atoll center coordinates
coords = pd.read_csv("data/atolls/roi_3_atolls.csv")

# Load island contours
high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")
atoll_ellipses = gpd.GeoDataFrame(pd.concat([atoll_ellipses1, atoll_ellipses2, high_island_ellipses],
                                            ignore_index=True))

# Filter to only include atolls for which data is available
hotspot_df = pd.read_csv("data/hotspots/IME/chl_hotspots_2003_2024_filled.csv")
valid_atolls = set(hotspot_df['atoll'])
atoll_ellipses = atoll_ellipses[atoll_ellipses["name"].isin(valid_atolls)]

# Create plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Define the extent
ax.set_extent([-155, -130, -14, -24.5], crs=ccrs.PlateCarree())

# Add a light grey background
ax.set_facecolor('lightgrey')

# Plot island contours
atoll_ellipses.plot(ax=ax, color='darkgrey', edgecolor='black', linewidth=1, transform=ccrs.PlateCarree())

# Merge data with coordinates
coords = coords.merge(dataset, on="atoll", how="left")
coords = coords.dropna(subset=[column_of_interest])  # Remove NaN

# Center the color bar to zero
# norm = colors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
sc = ax.scatter(coords['lon'], coords['lat'],
                c=coords[column_of_interest],
                cmap='seismic',
                # norm=norm,
                edgecolor='black',
                s=30,
                transform=ccrs.PlateCarree())

# Add gridlines with labels
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Add a north arrow to the top right corner
ax.annotate('N',
            xy=(0.97, 0.93), xycoords='axes fraction',
            ha='center', va='center',
            fontsize=14, fontweight='bold')

ax.annotate('', xy=(0.97, 0.90), xytext=(0.97, 0.82),
            xycoords='axes fraction',
            arrowprops=dict(facecolor='black', width=0.5, headwidth=10))

# Add colorbar
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.6)
cbar.set_label(var_name, fontsize=12)
plt.tight_layout()
plt.savefig(f"figures/{output_name}.png", dpi=500, bbox_inches='tight')
plt.show()
