"""

This code is part of the project:
Palola et al. "Disentangling geosphere and biosphere drivers for the island mass effect around atolls"

A. Define the region of interest (roi) around each atoll.
STEP 1. Define the names and coordinates of the atolls.
STEP 2. Define the region of interest around each atoll.

INPUT:
A dataframe: Atoll names and central coordinates

OUTPUT:
A shapefile: Rectangular region of interest around each atoll

Last updated on 3 March 2025

"""

# Import libraries
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

"""STEP 1. Define the names and coordinates of the atolls."""
atoll_df = pd.read_csv("data/atoll_data/roi3_atolls.csv")

# Atoll names
name_list = list(atoll_df["name"])

# Central coordinates
central_lat_list = list(atoll_df["center_lat"])
central_lon_list = list(atoll_df["center_lon"])

# Half the width of the region of interest, i.e. the radius of a circle drawn inside the bounding box
half_width_km = 100

"""STEP 2. Define the region of interest around each atoll."""


# Find the coordinates of the bounding box
def get_bounding_box(lat, lon, box_size_km):
    lat_per_km = 1 / 111.32
    lon_per_km = 1 / (111.32 * np.cos(np.radians(lat)))

    delta_lat = box_size_km * lat_per_km
    delta_lon = box_size_km * lon_per_km

    return {
        'lat_min': lat - delta_lat,
        'lat_max': lat + delta_lat,
        'lon_min': lon - delta_lon,
        'lon_max': lon + delta_lon
    }


# Loop through all the atolls to create shapefiles
for i in range(len(name_list)):
    bbox = get_bounding_box(central_lat_list[i], central_lon_list[i], half_width_km)
    rectangle = Polygon([
        (bbox['lon_min'], bbox['lat_min']),
        (bbox['lon_max'], bbox['lat_min']),
        (bbox['lon_max'], bbox['lat_max']),
        (bbox['lon_min'], bbox['lat_max']),
        (bbox['lon_min'], bbox['lat_min'])])
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[rectangle])
    # Save as a shapefile
    gdf.to_file("data/atolls_bbox/" + name_list[i] + "_roi.shp")

# Create an empty list to store bounding box data
bbox_data = []

# Loop through all the atolls
for i in range(len(name_list)):
    bbox = get_bounding_box(central_lat_list[i], central_lon_list[i], half_width_km)

    # Append bounding box details to the list
    bbox_data.append([
        name_list[i],
        bbox['lat_min'], bbox['lat_max'],
        bbox['lon_min'], bbox['lon_max']
    ])

# Convert the list into a DataFrame
bbox_df = pd.DataFrame(bbox_data, columns=["name", "lat_min", "lat_max", "lon_min", "lon_max"])

# Save the bounding box data to a CSV file
bbox_df.to_csv("data/atolls_bbox/bbox_coordinates_100km.csv", index=False)
