# Import libraries
import xarray as xr
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from models.tools import shallow_water_mask
import pandas as pd

"""STEP 1. Load the data and define key variables."""

# Define the study region extent (roi3)
lon_min = -157.684
lon_max = -127.737
lat_min = -25.9255
lat_max = -12.1711

# Define the filepath for GEBCO bathymetry data
gebco_filepath = "data/bathymetry/GEBCO_2024_sub_ice_topo.nc"

# Create a shallow water mask
shallow_mask_da = shallow_water_mask(gebco_filepath)

# Open the ocean current dataset
ds = xr.open_dataset("data/copernicus_data/copernicus_wind_1995_2024.nc")

# Define the variable
current_speed_depth_avg = ds["wind_speed"]

# Compute depth-averaged current speed
# current_speed_depth_avg = current_speed.mean(dim="depth")
# current_speed_depth_avg.name = "mlotst"
# print("Depth-averaged speed", current_speed_depth_avg.shape)

# Select a subset within the study region
# current_data = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
current_data = current_speed_depth_avg.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
# print("Current data shape", current_data.shape)

total_lats = current_data['latitude'].values
total_lons = current_data['longitude'].values

# Load the shapefiles
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")

atoll_ellipses = gpd.GeoDataFrame(pd.concat([atoll_ellipses1, atoll_ellipses2, high_island_ellipses],
                                            ignore_index=True))

"""STEP 2. Mask shallow waters and lagoon pixels."""

# Create a meshgrid of lon, lat for the entire region
lon1, lat1 = np.meshgrid(total_lons, total_lats)
points = [Point(x, y) for x, y in zip(lon1.ravel(), lat1.ravel())]

# Initialize a global mask
global_mask_interp_numeric = shallow_mask_da.interp(lon=current_data['longitude'],
                                                    lat=current_data['latitude'],
                                                    method='nearest')
print("Global mask data shape", global_mask_interp_numeric.shape)
global_mask_interp = global_mask_interp_numeric.astype(bool)
global_mask = np.zeros_like(global_mask_interp.values, dtype=bool)

counter = 0

# Loop through each atoll to mask the lagoon pixels
for ellipse_geo in atoll_ellipses.geometry:
    # Create the lagoon mask for points inside the ellipse
    lagoon_mask = np.array([ellipse_geo.contains(point) for point in points])

    # Reshape the lagoon mask to match the current data shape
    lagoon_mask = lagoon_mask.reshape(global_mask_interp_numeric.shape)

    counter += 1
    print(f"Loop iteration: {counter}")

    # Update the global mask
    global_mask |= lagoon_mask

# Apply the global mask to the current data
current_masked_time = []

for time_step in current_data['time']:
    # Slice the current data for the current time step
    current_data_time = current_data.sel(time=time_step)

    # Mask the current data for this time step using the global mask
    current_masked_time_step = current_data_time.where(~global_mask)

    # Append to the list
    current_masked_time.append(current_masked_time_step)

# Combine the masked data across all time steps
current_masked_time = xr.concat(current_masked_time, dim='time')

# Save to a NetCDF file
output_path = "data/masked/wind_1995_2024.nc"
current_masked_time.to_netcdf(output_path)

print(f"Masked current data saved to: {output_path}")
