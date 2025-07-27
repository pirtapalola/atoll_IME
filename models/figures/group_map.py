# Import libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Load IME delta difference data
dataset = pd.read_csv('ANALYSIS/delta_difference_2014_2016.csv')
column_of_interest = 'delta_difference'
var_name = "Difference in IME magnitude (mg/m$^3$)"
output_name = "delta_difference_sign_behavior_map"

# Load atoll center coordinates
coords = pd.read_csv("data/atolls/roi_3_atolls.csv")

# Load island contours (ellipses)
high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")
atoll_ellipses = gpd.GeoDataFrame(pd.concat([atoll_ellipses1, atoll_ellipses2, high_island_ellipses],
                                            ignore_index=True))

# Filter to only include valid atolls
hotspot_df = pd.read_csv("data/hotspots/IME/chl_hotspots_2003_2024_filled.csv")
valid_atolls = set(hotspot_df['atoll'])
atoll_ellipses = atoll_ellipses[atoll_ellipses["name"].isin(valid_atolls)]

# Merge dataset into coords
coords = coords.merge(dataset, on="atoll", how="left")
coords = coords.dropna(subset=[column_of_interest])

# Load sign behavior data (generated from second script)
sign_behavior = pd.read_csv("data/sign_behavior_consistency.csv")
coords = coords.merge(sign_behavior, on="atoll", how="left")

# Map consistency to color
color_map = {
    'Consistently positive': 'red',
    'Consistently negative': 'blue',
    'Sign changed': 'yellow',
    'No difference': 'white'
}
coords['behavior_color'] = coords['consistency'].map(color_map)

# Create plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([-155, -130, -14, -24.5], crs=ccrs.PlateCarree())
ax.set_facecolor('lightgrey')

# Plot island contours
atoll_ellipses.plot(ax=ax, color='darkgrey', edgecolor='black', linewidth=1, transform=ccrs.PlateCarree())

# Plot atoll points colored by behavior
for behavior, group in coords.groupby('consistency'):
    ax.scatter(group['lon'], group['lat'],
               color=color_map.get(behavior, 'black'),
               label=behavior,
               edgecolor='black',
               s=50,
               transform=ccrs.PlateCarree())

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Add north arrow
ax.annotate('N',
            xy=(0.97, 0.93), xycoords='axes fraction',
            ha='center', va='center',
            fontsize=14, fontweight='bold')
ax.annotate('', xy=(0.97, 0.90), xytext=(0.97, 0.82),
            xycoords='axes fraction',
            arrowprops=dict(facecolor='black', width=0.5, headwidth=10))

# Add legend
ax.legend(loc='lower left', fontsize=10)

# Final layout
plt.tight_layout()
plt.savefig(f"figures/{output_name}.png", dpi=500, bbox_inches='tight')
plt.show()
