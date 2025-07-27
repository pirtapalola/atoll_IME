"""

This code is part of the project:
Palola et al. "Disentangling geosphere and biosphere drivers for the island mass effect around atolls"

A. Only include hotspot pixels in clusters near the atoll.
STEP 1. Load the data.
STEP 2. Filter through the hotspot pixels.

Last updated on 23 April 2025

"""

# Import libraries
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np

"""STEP 1. Load the data."""

# Load filtered hotspot data
hotspot_gdf = gpd.read_file("data/IME/"
                            "hotspots_monthly_30km_2003_2024_filtered.gpkg")

# Load atoll geometries
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")

# Combine into one GeoDataFrame
atoll_ellipses = gpd.GeoDataFrame(pd.concat([atoll_ellipses1, atoll_ellipses2, high_island_ellipses],
                                            ignore_index=True))
atoll_ellipses = atoll_ellipses.to_crs(epsg=3395)

# Reproject hotspot data to the same CRS as atoll geometries
hotspot_gdf = hotspot_gdf.to_crs(epsg=3395)

# Create an empty list to store filtered DataFrames
filtered_gdfs = []

# Extract unique year-month-atoll combinations
unique_combinations = hotspot_gdf[["year", "month", "atoll"]].drop_duplicates()

"""STEP 2. Filter through the hotspot pixels."""

# Keep count of loop iterations
counter = 0

# Loop through each combination
for _, row in unique_combinations.iterrows():
    year = row["year"]
    month = row["month"]
    atoll = row["atoll"]

    # Filter for the specific atoll, year, and month
    hotspot_subset = hotspot_gdf[
        (hotspot_gdf["year"] == year) &
        (hotspot_gdf["month"] == month) &
        (hotspot_gdf["atoll"] == atoll)
        ].copy()

    if hotspot_subset.empty:
        print("Hotspot subset is empty for atoll: ", atoll, month, year)
        continue

    # Get the atoll geometry
    atoll_geom = atoll_ellipses.loc[atoll_ellipses["name"] == atoll, "geometry"]

    if atoll_geom.empty:
        print("Atoll geometry is empty for atoll: ", atoll)
        continue

    atoll_geom = atoll_geom.iloc[0]  # Take the first matching atoll geometry
    print(atoll_geom)

    # Cluster the points
    coords = np.array(list(zip(hotspot_subset.geometry.x, hotspot_subset.geometry.y)))

    # DBSCAN clustering
    db = DBSCAN(eps=2000, min_samples=2).fit(coords)  # eps in meters
    hotspot_subset["cluster"] = db.labels_

    # Print cluster stats
    total_clusters = len(set(hotspot_subset["cluster"])) - (1 if -1 in hotspot_subset["cluster"] else 0)
    noise_points = sum(hotspot_subset["cluster"] == -1)
    print(f"Atoll={atoll}, Year={year}, Month={month}: {total_clusters} clusters found, {noise_points} noise points")

    # Create a buffer around the atoll contour
    buffer_distance = 1000  # Distance in meters
    atoll_buffer = atoll_geom.buffer(buffer_distance)

    # Find clusters where at least one point is within the buffer
    clusters_to_keep = set(
        hotspot_subset.loc[
            hotspot_subset.geometry.within(atoll_buffer), "cluster"
        ]
    )

    # Print number of clusters passing the filter
    print(f"Atoll={atoll}, Year={year}, Month={month}: {len(clusters_to_keep)} clusters retained after 5 km filter")

    if clusters_to_keep:
        # Match cluster labels exactly and ensure datatype consistency
        hotspot_filtered = hotspot_subset[
            (hotspot_subset["cluster"] != -1) &
            (hotspot_subset["cluster"].astype(int).isin([int(c) for c in clusters_to_keep]))
            ]
    else:
        hotspot_filtered = gpd.GeoDataFrame()  # Avoid NoneType issues

    # If no points are left, print a warning
    if hotspot_filtered.empty:
        print(f"WARNING: No points retained for Atoll={atoll}, Year={year}, Month={month}")

    filtered_gdfs.append(hotspot_filtered)

    counter += 1
    print(f"Loop {counter}: Atoll={atoll}, Year={year}, Month={month} -> Kept {len(hotspot_filtered)} points")

# Combine all filtered subsets
hotspot_gdf_filtered = gpd.GeoDataFrame(pd.concat(filtered_gdfs, ignore_index=True))

# Save the filtered GeoDataFrame
hotspot_gdf_filtered.to_file("data/IME/hotspots_chl_2003_2024_final.gpkg", driver="GPKG")
