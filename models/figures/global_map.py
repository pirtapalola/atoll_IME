import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box

# Define map projection
proj = ccrs.Orthographic(central_longitude=-160, central_latitude=0)

# Create the figure and axis
fig = plt.figure(figsize=(4, 4))
ax = plt.axes(projection=proj)

# Add base map features
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='#caf0f8')
ax.coastlines()
ax.gridlines()

# Define the rectangle as a shapely box (in lon/lat)
lon_min = -155
lon_max = lon_min + 25
lat_min = -15
lat_max = lat_min + 10

geom = box(lon_min, lat_min, lon_max, lat_max)

# Add the geometry to the map
ax.add_geometries(
    [geom],
    crs=ccrs.PlateCarree(),     # Geographic coordinates
    facecolor='none',
    edgecolor='black',
    linewidth=2
)

# Save the figure
plt.tight_layout()
plt.savefig(f"figures/globe.png", dpi=500, bbox_inches='tight')
plt.close()
