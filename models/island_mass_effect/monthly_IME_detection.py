# Import libraries
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
from shapely.geometry import Point
from models.tools import create_offshore_buffer

"""Step 1. Load & Process Oceanic Monthly Means."""

# Load atoll bounding box data
bbox_df = pd.read_csv("data/atolls_bbox/bbox_coordinates_100km.csv")

# Load precomputed oceanic chlorophyll-a monthly means
oceanic_means_df = pd.read_csv("data/buffer_data/oceanic_30km_buffer_150km_extent.csv")

# Filter bbox_df to include only atolls present in oceanic_means_df
valid_atolls = set(oceanic_means_df["atoll"].unique())  # Get unique atoll names from oceanic_means_df
bbox_df_filtered = bbox_df[bbox_df["name"].isin(valid_atolls)]  # Keep only matching atolls

# Convert to dictionary format for iteration
atolls = bbox_df_filtered.to_dict(orient="records")

# Make sure the months and years are integers
oceanic_means_df["time"] = pd.to_datetime(oceanic_means_df["time"])  # Convert to datetime
oceanic_means_df["year"] = oceanic_means_df["time"].dt.year
oceanic_means_df["month"] = oceanic_means_df["time"].dt.month
oceanic_means_df["year"] = oceanic_means_df["year"].astype(int)
oceanic_means_df["month"] = oceanic_means_df["month"].astype(int)
print(oceanic_means_df)

# Create dictionary with (atoll, year, month) as keys
monthly_means_dict = {
    (row["atoll"], int(row["year"]), int(row["month"])): row["chlor_a"]
    for _, row in oceanic_means_df.iterrows()
}

# Create dictionary with (atoll, year, month) as keys
monthly_std_dict = {
    (row["atoll"], int(row["year"]), int(row["month"])): row["std"]
    for _, row in oceanic_means_df.iterrows()
}

""" STEP 2: Load Chlorophyll-a Data from NetCDF """

# Load the shapefiles
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")
atoll_ellipses = gpd.GeoDataFrame(pd.concat([atoll_ellipses1, atoll_ellipses2, high_island_ellipses],
                                            ignore_index=True))

# Open the netCDF file containing the preprocessed chl-a data
nc_file = "data/masked/roi3_masked_4km_1997_2024.nc"
chl_preprocessed = xr.open_dataset(nc_file)

# Extract the variable of interest
chl_data = chl_preprocessed['chlor_a']

# Slice to the time period of interest
chl_data = chl_data.sel(time=slice("2003", "2024"))

# Initialize an empty GeoDataFrame to store hotspot pixels
hotspot_gdf = gpd.GeoDataFrame(columns=["atoll", "year", "month", "geometry"], geometry="geometry", crs="EPSG:4326")

# Keep count of the number of loop iterations
counter = 0

# Loop through each atoll
for atoll in atolls:
    lat_min = atoll["lat_min"]
    lat_max = atoll["lat_max"]
    lon_min = atoll["lon_min"]
    lon_max = atoll["lon_max"]
    atoll_name = atoll["name"]

    # Slice chlorophyll-a data to the area of interest
    chl_a_data_subset = chl_data.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

    # Filter the GeoDataFrame to get the geometry of the specific atoll
    atoll_ellipses = atoll_ellipses.to_crs("EPSG:4326")
    atoll_geometry = atoll_ellipses.loc[atoll_ellipses["name"] == atoll["name"], "geometry"].values
    ellipse_geo = atoll_geometry[0]
    ellipse_gdf = gpd.GeoDataFrame(geometry=[ellipse_geo])

    if ellipse_gdf.crs is None:
        ellipse_gdf = ellipse_gdf.set_crs("EPSG:4326")

    # Create the buffer around the ellipse
    buffer_km = 30
    analysis_buffer_distance = buffer_km * 1000  # Convert km to meters
    analysis_buffer, analysis_buffer_gdf = create_offshore_buffer(ellipse_gdf, analysis_buffer_distance)

    if analysis_buffer_gdf.crs is None:
        analysis_buffer_gdf = analysis_buffer_gdf.set_crs("EPSG:4326")

    # Extract unique years & months from dataset
    time_series = pd.to_datetime(chl_a_data_subset['time'].values)
    unique_years = sorted(time_series.year.unique())
    unique_months = range(1, 13)  # Months from 1 to 12

    # Loop through each year & month
    for year in unique_years:
        for month in unique_months:
            if (atoll_name, year, month) not in monthly_means_dict:
                print(f"Warning: No oceanic data found for {atoll_name} in {year, month}. Skipping...")
                continue

            threshold = (monthly_means_dict[(atoll_name, year, month)]
                         + (2 * monthly_std_dict[(atoll_name, year, month)]))

            # Filter data for the current month-year
            chl_a_month = chl_a_data_subset.sel(time=(time_series.year == year) & (time_series.month == month))

            # Clip chlorophyll-a data to the buffer area
            chl_a_month = chl_a_month.rio.write_crs("EPSG:4326", inplace=True)
            chl_a_month_buffer = chl_a_month.rio.clip(analysis_buffer.geometry, analysis_buffer.crs, all_touched=True)

            # Inverse the log-transformation
            chl_a_month_buffer = 10 ** chl_a_month_buffer

            hotspot_indices = np.argwhere(chl_a_month_buffer.values[0] > threshold)  # Extract first time slice

            # Extract corresponding lat/lon
            hotspot_lats = chl_a_month_buffer.lat.values[hotspot_indices[:, 0]]
            hotspot_lons = chl_a_month_buffer.lon.values[hotspot_indices[:, 1]]

            # Create Point geometries
            hotspot_points = [Point(lon, lat) for lon, lat in zip(hotspot_lons, hotspot_lats)]

            # Create a GeoDataFrame for this month-year-atoll
            atoll_hotspot_gdf = gpd.GeoDataFrame({
                "atoll": atoll_name,
                "year": year,
                "month": month,
                "geometry": hotspot_points
            }, crs="EPSG:4326")  # Ensure correct CRS

            # Append to the main GeoDataFrame
            hotspot_gdf = pd.concat([hotspot_gdf, atoll_hotspot_gdf], ignore_index=True)

            counter += 1
            print(f"Loop iteration: {counter}")

# Save the GeoDataFrame
hotspot_gdf.to_file("data/IME/hotspots_monthly_30km_2003_2024.gpkg", driver="GPKG")
