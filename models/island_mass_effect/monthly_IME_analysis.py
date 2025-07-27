"""

This code is part of the project:
Palola et al. "Disentangling geosphere and biosphere drivers for the island mass effect around atolls"

A. Calculate the increase in chlorophyll-a relative to oceanic background conditions.
STEP 1. Load the data.
STEP 2. Compute the difference in chlorophyll-a for each pixel.
STEP 3. Calculate the sum over all pixels.

INPUT:
A GeoPackage file containing the hotspot pixels.
A netCDF file containing the masked chlorophyll-a data.
A csv file containing the oceanic background value associated with each atoll.

OUTPUT:
A csv file containing the increase in chlorophyll-a for each atoll, each month, each year.

Last updated on 23 April 2025

"""

# Import libraries
import pandas as pd
import geopandas as gpd
import xarray as xr

"""Step 1. Load the data."""

# Load hotspot points
hotspot_gdf = gpd.read_file("data/hotspots/IME/hotspots_chl_2003_2024_cleaned.gpkg")

# Load oceanic means data
oceanic_means_df = pd.read_csv("data/buffer_data/oceanic_30km_buffer_150km_extent.csv")

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

nc_file = "data/masked/roi3_masked_4km_1997_2024.nc"
chl_preprocessed = xr.open_dataset(nc_file)

# Ensure CRS consistency
if hotspot_gdf.crs is None:
    hotspot_gdf = hotspot_gdf.set_crs("EPSG:4326")
elif hotspot_gdf.crs.to_epsg() != 4326:
    hotspot_gdf = hotspot_gdf.to_crs("EPSG:4326")

if chl_preprocessed.rio.crs is None:
    chl_preprocessed.rio.write_crs("EPSG:4326", inplace=True)
elif chl_preprocessed.rio.crs.to_epsg() != 4326:
    chl_preprocessed = chl_preprocessed.rio.reproject("EPSG:4326")

# Extract the variable of interest
chl_data = chl_preprocessed['chlor_a']
print(chl_data)

"""STEP 2. Compute the difference in chlorophyll-a for each pixel."""

# Create an empty list to store the results.
results = []

# Loop through each data point
for idx, row in hotspot_gdf.iterrows():
    atoll = row["atoll"]
    year = int(row["year"])
    month = int(row["month"])
    lon, lat = row["geometry"].x, row["geometry"].y

    # Extract chl-a value at hotspot location and time
    chl_a_value = chl_data.sel(
        lon=lon, lat=lat, method="nearest", tolerance=0.01
    ).sel(time=f"{year}-{month:02d}").values.item()

    # Inverse the log-transformation
    chl_a_value = 10 ** chl_a_value

    # Get oceanic mean value for that atoll, year, and month
    mean_chl_a = monthly_means_dict.get((atoll, year, month), None)

    if mean_chl_a is not None:
        # Compute delta chl-a
        delta_chl = chl_a_value / mean_chl_a

        # Save both chl_a_value and delta_chl
        results.append([atoll, year, month, mean_chl_a, chl_a_value, delta_chl])

# Convert to DataFrame
results_df = pd.DataFrame(
    results, columns=["atoll", "year", "month", "mean_chl_a", "chl_a_value", "delta_chl_a"]
)

"""STEP 3. Calculate the sum over all pixels."""

monthly_sum_df = results_df.groupby(
    ["atoll", "year", "month", "mean_chl_a"])[["chl_a_value", "delta_chl_a"]].mean().reset_index()

# Save to CSV
output_file = "data/relative_chl_hotspots_2003_2024_raw.csv"
monthly_sum_df.to_csv(output_file, index=False)
