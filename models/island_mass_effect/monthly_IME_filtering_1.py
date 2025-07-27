"""

This code is part of the project:
Palola et al. "Disentangling geosphere and biosphere drivers for the island mass effect around atolls"

A. Only include hotspot pixels closer to the atoll of interest than any other island.
STEP 1. Load the data.
STEP 2. Filter through the hotspot pixels.

Last updated on 23 April 2025

"""

# Import libraries
import pandas as pd
import geopandas as gpd

"""STEP 1. Load the data."""

bbox_df = pd.read_csv("data/atolls_bbox/bbox_coordinates_100km.csv")
oceanic_means_df = pd.read_csv("data/buffer_data/oceanic_30km_buffer_150km_extent.csv")

# Filter bbox_df to include only relevant atolls
valid_atolls = set(oceanic_means_df["atoll"].unique())
bbox_df_filtered = bbox_df[bbox_df["name"].isin(valid_atolls)]

# Load atoll geometries
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")

# Combine into one GeoDataFrame
atoll_ellipses = gpd.GeoDataFrame(pd.concat([atoll_ellipses1, atoll_ellipses2, high_island_ellipses],
                                            ignore_index=True))

# Filter atolls by valid names
atoll_ellipses = atoll_ellipses[atoll_ellipses["name"].isin(valid_atolls)]

# Reproject atoll geometries to a projected CRS (e.g., EPSG:3395 - World Mercator)
atoll_ellipses = atoll_ellipses.to_crs(epsg=3395)

# Load hotspot data
hotspot_gdf = gpd.read_file("data/IME/hotspots_monthly_30km_2003_2024.gpkg")

# Reproject hotspot data to the same CRS as atoll geometries
hotspot_gdf = hotspot_gdf.to_crs(epsg=3395)

"""STEP 2. Filter through the hotspot pixels."""

# Create an empty list to store filtered DataFrames
filtered_gdfs = []

# Extract unique year-month combinations
unique_year_months = hotspot_gdf[["year", "month"]].drop_duplicates()

# Keep count of loop iterations
counter = 0

for _, row in unique_year_months.iterrows():
    year = row["year"]
    month = row["month"]

    # Filter hotspots for this specific year and month
    hotspot_subset = hotspot_gdf[
        (hotspot_gdf["year"] == year) &
        (hotspot_gdf["month"] == month)
        ].copy()

    # Perform spatial join to find nearest atolls for this subset
    hotspot_subset = hotspot_subset.sjoin_nearest(atoll_ellipses[["name", "geometry"]], distance_col="distance")

    # Check the number of rows after spatial join (debugging)
    print(f"Rows in hotspot_subset after sjoin_nearest (Year {year}, Month {month}): {len(hotspot_subset)}")

    # Keep only hotspot pixels closest to their assigned atoll
    hotspot_filtered = hotspot_subset[hotspot_subset["atoll"] == hotspot_subset["name"]]

    # Check how many rows are kept after filtering
    print(f"Rows in hotspot_filtered after filtering (Year {year}, Month {month}): {len(hotspot_filtered)}")

    # Append filtered data to the list
    filtered_gdfs.append(hotspot_filtered)

    counter += 1
    print(f"Loop iteration: {counter} (Year: {year}, Month: {month})")

# Combine all filtered subsets
hotspot_gdf_filtered = gpd.GeoDataFrame(pd.concat(filtered_gdfs, ignore_index=True))

# Save the filtered GeoDataFrame
hotspot_gdf_filtered.to_file("data/IME/hotspots_monthly_30km_2003_2024_filtered.gpkg",
                             driver="GPKG")
