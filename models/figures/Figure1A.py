# Import libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib as mpl

# Set global font sizes
mpl.rcParams.update({
    'font.size': 30,           # Base font size for most elements
    'axes.labelsize': 34,      # Axis labels
    'xtick.labelsize': 30,     # X tick labels
    'ytick.labelsize': 30,     # Y tick labels
    'legend.fontsize': 30,     # Legend font
    'figure.titlesize': 34     # Title size
})


# Load atoll center coordinates
coords = pd.read_csv("data/atolls/roi_3_atolls.csv")

# Load island contours
high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")
atoll_ellipses = gpd.GeoDataFrame(pd.concat([atoll_ellipses1, atoll_ellipses2], ignore_index=True))

# Create plot
fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()})

# Add a light blue background
ax.set_facecolor('#caf0f8')

# Define the extent
ax.set_extent([-155, -130, -14, -24], crs=ccrs.PlateCarree())

# Plot atoll contours
atoll_ellipses.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=1, transform=ccrs.PlateCarree())

# Plot high island contours
high_island_ellipses.plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=1, transform=ccrs.PlateCarree())

# Add gridlines with labels
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.ylocator = mticker.FixedLocator(np.arange(-24, -13 + 2, 2))

# Add a north arrow to the top right corner
ax.annotate('N',
            xy=(0.97, 0.93), xycoords='axes fraction',
            ha='center', va='center',
            fontsize=14, fontweight='bold')

ax.annotate('', xy=(0.97, 0.90), xytext=(0.97, 0.82),
            xycoords='axes fraction',
            arrowprops=dict(facecolor='black', width=0.5, headwidth=10))

# Save the figure
plt.tight_layout()
plt.savefig(f"figures/study_region.png", dpi=500, bbox_inches='tight')
plt.close()
