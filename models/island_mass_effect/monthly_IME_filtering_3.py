# Import libraries
import geopandas as gpd
import pandas as pd
import xarray as xr
from models.tools import create_offshore_buffer
import numpy as np
from shapely.geometry import Point

# Load atoll bounding box data
bbox_df = pd.read_csv("data/atolls_bbox/bbox_coordinates_100km.csv")

# Load precomputed oceanic chlorophyll-a monthly means
oceanic_means_df = pd.read_csv("data/buffer_data/oceanic_30km_buffer_150km_extent.csv")

# Filter bbox_df to include only atolls present in oceanic_means_df
valid_atolls = set(oceanic_means_df["atoll"].unique())  # Get unique atoll names from oceanic_means_df
bbox_df_filtered = bbox_df[bbox_df["name"].isin(valid_atolls)]  # Keep only matching atolls

# Convert to dictionary format for iteration
atolls = bbox_df_filtered.to_dict(orient="records")

# Load hotspot data
hotspot_gdf = gpd.read_file("data/IME/hotspots_chl_2003_2024_final.gpkg")

# Load chl data
nc_file = "data/masked/roi3_masked_4km_1997_2024.nc"
chl_preprocessed = xr.open_dataset(nc_file)

# Load atoll geometries
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")

# Combine into one GeoDataFrame
atoll_ellipses = gpd.GeoDataFrame(pd.concat([atoll_ellipses1, atoll_ellipses2, high_island_ellipses],
                                            ignore_index=True))
atoll_ellipses = atoll_ellipses.to_crs(epsg=4326)

# Reproject hotspot data to the same CRS as atoll geometries
hotspot_gdf = hotspot_gdf.to_crs(epsg=4326)

if chl_preprocessed.rio.crs is None:
    chl_preprocessed.rio.write_crs("EPSG:4326", inplace=True)
elif chl_preprocessed.rio.crs.to_epsg() != 4326:
    chl_preprocessed = chl_preprocessed.rio.reproject("EPSG:4326")

# Extract the variable of interest
chl_data = chl_preprocessed['chlor_a']

# Initialize lists to store results
high_nan_records = []
no_valid_records = []

# Keep count of loop iterations
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

    # Get atoll geometry
    atoll_geometry = atoll_ellipses.loc[atoll_ellipses["name"] == atoll_name, "geometry"].values[0]
    ellipse_gdf = gpd.GeoDataFrame(geometry=[atoll_geometry], crs="EPSG:4326")

    # Create 10km offshore buffer
    buffer_km = 10
    analysis_buffer_distance = buffer_km * 1000  # Convert km to meters
    analysis_buffer, analysis_buffer_gdf = create_offshore_buffer(ellipse_gdf, analysis_buffer_distance)

    # Extract unique years & months
    time_series = pd.to_datetime(chl_a_data_subset['time'].values)
    unique_years = sorted(time_series.year.unique())
    unique_months = range(1, 13)  # Months from 1 to 12
    unique_years = unique_years[3:]  # Drop the first years

    # Loop through each year & month
    for year in unique_years:
        for month in unique_months:
            # Filter data for the current month-year
            chl_a_month = chl_a_data_subset.sel(time=(time_series.year == year) & (time_series.month == month))

            # Ensure CRS is set before clipping
            chl_a_month = chl_a_month.rio.write_crs("EPSG:4326", inplace=True)

            # Clip chlorophyll-a data to the buffer area
            chl_a_month_buffer = chl_a_month.rio.clip(analysis_buffer.geometry, analysis_buffer.crs, all_touched=True)

            nan_indices = np.argwhere(np.isnan(chl_a_month_buffer.values[0]))
            valid_indices = np.argwhere(~np.isnan(chl_a_month_buffer.values[0]))

            # Extract corresponding lat/lon
            nan_lats = chl_a_month_buffer.lat.values[nan_indices[:, 0]]
            nan_lons = chl_a_month_buffer.lon.values[nan_indices[:, 1]]

            # Create Point geometries
            nan_points = [Point(lon, lat) for lon, lat in zip(nan_lons, nan_lats)]

            # Create a GeoDataFrame
            nan_gdf = gpd.GeoDataFrame(geometry=nan_points, crs="EPSG:4326")

            # Keep the NaN pixels that lie inside the buffer zone
            nan_gdf = nan_gdf[nan_gdf.geometry.within(analysis_buffer.unary_union)]

            # Remove the NaN pixels that lie inside the atoll contour lines
            nan_gdf = nan_gdf[~nan_gdf.geometry.within(atoll_ellipses.unary_union)]

            # Calculate the percentage of NaN pixels
            total_pixels = len(valid_indices)
            nan_pixels_no = nan_gdf.size
            print(total_pixels, nan_pixels_no)

            # Check for no valid pixels
            if total_pixels == 0:
                no_valid_records.append((year, month, atoll_name))

            # Check for a high percentage of NaN pixels
            if (nan_pixels_no > 0) and (total_pixels > 0):
                nan_percentage = (nan_pixels_no / total_pixels) * 100
                # Check if NaN percentage exceeds 20%
                if nan_percentage > 20:
                    high_nan_records.append((year, month, atoll_name, nan_percentage))
                    print(f"NaN pixel percentage for atoll {atoll_name}: ", nan_percentage)

            counter += 1
            print(f"Loop iteration: {counter} (Year: {year}, Month: {month})")

            """fig, ax = plt.subplots(figsize=(10, 8))

            # Extract extent values
            lon_min, lon_max = float(chl_a_month_buffer.lon.min().item()), float(chl_a_month_buffer.lon.max().item())
            lat_min, lat_max = float(chl_a_month_buffer.lat.min().item()), float(chl_a_month_buffer.lat.max().item())

            chl_a_month_buffer_avg = chl_a_month_buffer.mean(dim="time")
            img = ax.imshow(
                chl_a_month_buffer_avg.values,
                origin="upper",
                extent=(
                    float(chl_a_month_buffer.lon.min().values), float(chl_a_month_buffer.lon.max().values),
                    float(chl_a_month_buffer.lat.min().values), float(chl_a_month_buffer.lat.max().values)
                ),
                cmap="viridis",
                zorder=1)
            plt.colorbar(img, ax=ax, label="Log chl-a concentration (mg/m³)")

            # Plot atoll contours
            atoll_ellipses1.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=2)
            atoll_ellipses2.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=2)
            high_island_ellipses.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=2)

            if hotspot_gdf.empty:
                print("No NaN pixels to plot.")
            if not hotspot_gdf.empty:
                hotspot_gdf.plot(ax=ax, color='yellow', label='Hotspots', markersize=1, zorder=3)

            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            
            plt.show()"""

# Convert results to DataFrames
nan_records_df = pd.DataFrame(high_nan_records, columns=["year", "month", "atoll", "NaN_Percentage"])
no_valid_records_df = pd.DataFrame(no_valid_records, columns=["year", "month", "atoll"])

# Save results to CSV files
nan_records_df.to_csv("data/IME/high_nan_atolls.csv", index=False)
no_valid_records_df.to_csv("data/IME/no_valid_records.csv", index=False)

# Convert to a set for fast lookup
nan_records_set = set(zip(nan_records_df["year"], nan_records_df["month"], nan_records_df["atoll"]))
no_valid_records_set = set(zip(no_valid_records_df["year"], no_valid_records_df["month"], no_valid_records_df["atoll"]))

# Filter out records present in nan_records
hotspot_gdf_filtered = hotspot_gdf[
    ~hotspot_gdf.apply(lambda row: (row["year"], row["month"], row["atoll"]) in nan_records_set, axis=1)]

# Filter out records present in nan_records
hotspot_gdf_final = hotspot_gdf_filtered[
    ~hotspot_gdf_filtered.apply(lambda row: (row["year"], row["month"], row["atoll"]) in no_valid_records_set, axis=1)]

# Save the cleaned dataset
hotspot_gdf_final.to_file("data/IME/hotspots_chl_2003_2024_cleaned.gpkg", driver="GPKG")
