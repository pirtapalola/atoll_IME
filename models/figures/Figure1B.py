# Import libraries
import geopandas as gpd
import pandas as pd
import xarray as xr
from models.tools import create_offshore_buffer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter

# Set global font size for all plots
mpl.rcParams.update({
    'font.size': 14,          # General font size
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 14,     # Axis label font size
    'xtick.labelsize': 12,    # X-axis tick label font size
    'ytick.labelsize': 12,    # Y-axis tick label font size
    'legend.fontsize': 12,    # Legend font size
    'figure.titlesize': 16   # Figure title font size
})


def lon_formatter(x, pos):
    direction = 'E' if x >= 0 else 'W'
    return f"{abs(x):.1f}°{direction}"


def lat_formatter(x, pos):
    direction = 'N' if x >= 0 else 'S'
    return f"{abs(x):.1f}°{direction}"


# Define buffer width
buffer_width = 30

# Load atoll bounding box data
bbox_df = pd.read_csv("data/atolls_bbox/bbox_coordinates_150km.csv")

# Load the shapefiles
atoll_ellipses1 = gpd.read_file("data/ellipse/atoll_ellipses_0.shp")
atoll_ellipses2 = gpd.read_file("data/ellipse/atoll_ellipses_30.shp")
high_island_ellipses = gpd.read_file("data/ellipse/high_island_ellipses_0.shp")
atoll_ellipses = gpd.GeoDataFrame(pd.concat([atoll_ellipses1, atoll_ellipses2, high_island_ellipses],
                                            ignore_index=True))

# Load the filtered hotspot data
hotspot_gdf = gpd.read_file("data/hotspots/IME/hotspots_chl_2003_2024_cleaned.gpkg")
if hotspot_gdf.crs is None:
    hotspot_gdf = hotspot_gdf.set_crs("EPSG:4326")
elif hotspot_gdf.crs.to_epsg() != 4326:
    hotspot_gdf = hotspot_gdf.to_crs("EPSG:4326")

# Define the variable of interest
variable_name = "chlor_a"

# Open the dataset
ds_combined = xr.open_dataset("data/masked/roi3_masked_4km_1997_2024.nc")

# Extract variable and slice time
ds_variable = ds_combined[variable_name]
ds_variable = ds_variable.sel(time=slice("2003", "2024"))

# Ensure CRS consistency
if atoll_ellipses.crs is None:
    atoll_ellipses = atoll_ellipses.set_crs("EPSG:4326")
elif atoll_ellipses.crs.to_epsg() != 4326:
    atoll_ellipses = atoll_ellipses.to_crs("EPSG:4326")

ds_variable.rio.write_crs("EPSG:4326", inplace=True)
if "lon" in ds_variable.coords and "lat" in ds_variable.coords:
    ds_variable = ds_variable.assign_coords(
        lon=ds_variable["lon"].assign_attrs(crs="EPSG:4326"),
        lat=ds_variable["lat"].assign_attrs(crs="EPSG:4326")
    )

# Remove high islands
exclude_islands = [
    "Tahiti", "Moorea", "Raiatea", "Bora_Bora", "Huahine", "Gambiers",
    "Rimatara", "Rurutu", "Tubuai", "Raivavae", "Akiaki", "Nukutavake",
    "Maiao", "Maupiti", "Henderson", "Pitcairn", "Makatea", "Tepoto_Nord", "Mauke"
]

atolls_list = ["Anaa"]

bbox_df_filtered = bbox_df[bbox_df["name"].isin(atolls_list)]  # Keep only matching atolls

# Convert to dictionary format for iteration
atolls = bbox_df_filtered.to_dict(orient="records")

# Keep count of the number of loop iterations
counter = 0
results = []

# Loop through each atoll
for atoll in atolls:
    lat_min = atoll["lat_min"]
    lat_max = atoll["lat_max"]
    lon_min = atoll["lon_min"]
    lon_max = atoll["lon_max"]
    atoll_name = atoll["name"]

    # Slice the data to the area of interest
    data_subset = ds_variable.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    print(data_subset.shape)

    # Filter the GeoDataFrame to get the geometry of the specific atoll
    atoll_ellipses = atoll_ellipses.to_crs("EPSG:4326")
    atoll_geometry = atoll_ellipses.loc[atoll_ellipses["name"] == atoll["name"], "geometry"].values
    ellipse_geo = atoll_geometry[0]
    ellipse_gdf = gpd.GeoDataFrame(geometry=[ellipse_geo])

    if ellipse_gdf.crs is None:
        ellipse_gdf = ellipse_gdf.set_crs("EPSG:4326")

    # Create the buffer around the ellipse
    buffer_km = buffer_width
    analysis_buffer_distance = buffer_km * 1000  # Convert km to meters
    analysis_buffer, analysis_buffer_gdf = create_offshore_buffer(ellipse_gdf, analysis_buffer_distance)

    if analysis_buffer_gdf.crs is None:
        analysis_buffer_gdf = analysis_buffer_gdf.set_crs("EPSG:4326")

    # Extract specific years & months from dataset
    time_series = pd.to_datetime(data_subset['time'].values)
    unique_years = [2011, 2012]
    unique_months = range(4, 5)  # Months from 1 to 12

    # Loop through each year & month
    for year in unique_years:
        for month in unique_months:
            # Filter data for the current month-year
            values_month = data_subset.sel(time=(time_series.year == year) & (time_series.month == month))

            # Clip chlorophyll-a data to the buffer area
            values_month = values_month.rio.write_crs("EPSG:4326", inplace=True)

            # Clip to buffer
            values_month_buffer = values_month.rio.clip(analysis_buffer.geometry,
                                                        analysis_buffer.crs,
                                                        all_touched=True)

            # Define data range for plotting
            min_val_buffer = float(np.nanpercentile(values_month_buffer.values, 2))
            max_val_buffer = float(np.nanpercentile(values_month_buffer.values, 98))

            # Regional map
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            im = values_month.plot(ax=ax1, cmap="viridis", alpha=0.9, vmin=min_val_buffer, vmax=max_val_buffer,
                                   add_colorbar=False)
            cbar = fig1.colorbar(im, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label("Chlorophyll-a (mg/m³)")

            atoll_ellipses.plot(ax=ax1, color='lightgrey', edgecolor='darkgrey', linewidth=2)
            ellipse_gdf.plot(ax=ax1, color='lightgrey', edgecolor='darkgrey', linewidth=2)

            atoll_hotspots = hotspot_gdf[
                (hotspot_gdf["atoll"] == atoll_name) &
                (hotspot_gdf["year"] == str(year)) &
                (hotspot_gdf["month"] == str(month))
                ]

            analysis_buffer_gdf.plot(ax=ax1, color='none', edgecolor='lightgrey', linestyle='--', linewidth=2)

            ax1.set_xlim([lon_min, lon_max])
            ax1.set_ylim([lat_min, lat_max])
            ax1.set_xlabel("Longitude")
            ax1.set_ylabel("Latitude")

            # Format ticks
            ax1.xaxis.set_major_locator(MultipleLocator(1))
            ax1.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
            ax1.yaxis.set_major_formatter(FuncFormatter(lat_formatter))

            ax1.grid(True)
            fig1.tight_layout()
            fig1.savefig(f"figures/{atoll_name}-{year}-{month:02}_full.png", dpi=500, bbox_inches='tight')
            plt.close(fig1)

            # Zoomed-in map
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            im = values_month.plot(ax=ax2, cmap="viridis", alpha=0.9, vmin=min_val_buffer, vmax=max_val_buffer,
                                   add_colorbar=False)
            cbar = fig2.colorbar(im, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label("Chlorophyll-a (mg/m³)")

            ellipse_gdf.plot(ax=ax2, color='lightgrey', edgecolor='darkgrey', linewidth=2)
            analysis_buffer_gdf.plot(ax=ax2, color='none', edgecolor='lightgrey', linestyle='--', linewidth=2)

            atoll_hotspots = hotspot_gdf[
                (hotspot_gdf["atoll"] == atoll_name) &
                (hotspot_gdf["year"] == str(year)) &
                (hotspot_gdf["month"] == str(month))
                ]

            ax2.scatter(atoll_hotspots.geometry.x, atoll_hotspots.geometry.y,
                        color='#fb5607', label='Hotspots', s=3, alpha=0.8, zorder=2)

            buffer_bounds = analysis_buffer_gdf.total_bounds
            ax2.set_xlim([buffer_bounds[0], buffer_bounds[2]])
            ax2.set_ylim([buffer_bounds[1], buffer_bounds[3]])

            ax2.set_xlabel("Longitude")
            ax2.set_ylabel("Latitude")

            # Set x-tick step and custom format
            ax2.xaxis.set_major_locator(MultipleLocator(0.2))
            ax2.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
            ax2.yaxis.set_major_formatter(FuncFormatter(lat_formatter))

            ax2.grid(True)
            fig2.tight_layout()
            fig2.savefig(f"figures/{atoll_name}-{year}-{month:02}_zoomed.png", dpi=500, bbox_inches='tight')
            plt.close(fig2)
