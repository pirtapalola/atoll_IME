"""

This code is part of the project:
Palola et al. "Disentangling geosphere and biosphere drivers for the island mass effect around atolls"

A. Calculate the monthly background oceanic value for each atoll.
STEP 1. Load the data.
STEP 2. Compute the monthly mean oceanic background value for each atoll.

INPUT:
A netCDF file containing the masked chlorophyll-a data.
A csv file containing the bounding box coordinates for each atoll.

OUTPUT:
A csv file containing the monthly oceanic background values computed for each atoll.

Last updated on 23 April 2025

"""

# Import libraries
import pandas as pd
import xarray as xr

"""STEP 1. Load the data."""

# Define bbox coordinates around each atoll
bbox_df = pd.read_csv("data/atolls_bbox/bbox_coordinates_150km.csv")
atolls = bbox_df.to_dict(orient="records")  # Converts each row into a dictionary

# Open the netCDF file
nc_file = "data/masked/roi3_buffer_masked_30km_1997_2024.nc"
chl_nc = xr.open_dataset(nc_file)

# Extract the variable of interest
chl_a = chl_nc['chlor_a']

"""STEP 2. Compute the monthly mean oceanic background value for each atoll."""

# Create an empty list to store the results
chl_results = []

# Loop through each atoll
for atoll in atolls:
    lat_min = atoll["lat_min"]
    lat_max = atoll["lat_max"]
    lon_min = atoll["lon_min"]
    lon_max = atoll["lon_max"]

    # Slice the chlorophyll-a data to the area of interest
    chl_a_data_subset = chl_a.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

    # Slice to the time period of interest
    chl_a_data_subset = chl_a_data_subset.sel(time=slice("2003", "2024"))

    # Inverse the log-transformation
    chl_a_data_subset = 10 ** chl_a_data_subset

    # Compute mean, standard deviation, and median
    chl_monthly_mean = chl_a_data_subset.mean(dim=["lon", "lat"])
    chl_monthly_std = chl_a_data_subset.std(dim=["lon", "lat"])
    chl_monthly_median = chl_a_data_subset.median(dim=["lon", "lat"])

    # Convert results to dataframes
    df_mean = chl_monthly_mean.to_dataframe().reset_index()
    df_std = chl_monthly_std.to_dataframe().reset_index()
    df_median = chl_monthly_median.to_dataframe().reset_index()

    # Merge all statistics into a single DataFrame
    df_mean["std"] = df_std["chlor_a"]
    df_mean["median"] = df_median["chlor_a"]

    # Add an 'atoll' column for identification
    df_mean["atoll"] = atoll["name"]

    # Append to the results list
    chl_results.append(df_mean)

# Concatenate all results into a single DataFrame
chl_results_df = pd.concat(chl_results, ignore_index=True)

# Save the DataFrame to a CSV file
chl_results_df.to_csv("data/buffer_data/oceanic_30km_buffer_150km_extent.csv", index=False)
