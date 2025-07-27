# Import libraries
import geopandas as gpd
import pandas as pd
import xarray as xr
from models.tools import create_offshore_buffer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
"""
# Load the filtered hotspot data
hotspot_gdf = gpd.read_file("data/hotspots/IME/hotspots_chl_2003_2024_cleaned.gpkg")
if hotspot_gdf.crs is None:
    hotspot_gdf = hotspot_gdf.set_crs("EPSG:4326")
elif hotspot_gdf.crs.to_epsg() != 4326:
    hotspot_gdf = hotspot_gdf.to_crs("EPSG:4326")"""

# Define the variable of interest
variable_name = "current"

# Open the dataset
# ds_combined = xr.open_dataset("data/masked/roi3_masked_4km_1997_2024.nc")

# Open both datasets
ds1 = xr.open_dataset(f"data/masked/{variable_name}_masked_1993_2021.nc")
ds2 = xr.open_dataset(f"data/masked/{variable_name}_masked_2021_2024.nc")

# Concatenate along the time dimension
ds_combined = xr.concat([ds1, ds2], dim="time")
# ds_combined = ds_combined.mean(dim="depth")
print(ds_combined)

# Extract variable and slice time
ds_variable = ds_combined["KE"]
# ds_variable = ds_variable.sel(time=slice("2003", "2024"))

# Ensure CRS consistency
if atoll_ellipses.crs is None:
    atoll_ellipses = atoll_ellipses.set_crs("EPSG:4326")
elif atoll_ellipses.crs.to_epsg() != 4326:
    atoll_ellipses = atoll_ellipses.to_crs("EPSG:4326")

ds_variable.rio.write_crs("EPSG:4326", inplace=True)
if "longitude" in ds_variable.coords and "latitude" in ds_variable.coords:
    ds_variable = ds_variable.assign_coords(
        longitude=ds_variable["longitude"].assign_attrs(crs="EPSG:4326"),
        latitude=ds_variable["latitude"].assign_attrs(crs="EPSG:4326")
    )

# Remove high islands
exclude_islands = [
    "Tahiti", "Moorea", "Raiatea", "Bora_Bora", "Huahine", "Gambiers",
    "Rimatara", "Rurutu", "Tubuai", "Raivavae", "Akiaki", "Nukutavake",
    "Maiao", "Maupiti", "Henderson", "Pitcairn", "Makatea", "Tepoto_Nord", "Mauke"
]
atolls_list = [atoll for atoll in bbox_df['name'].unique() if atoll not in exclude_islands]
# atolls_list = ["Anaa"]
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
    data_subset = ds_variable.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
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

    # Extract unique years & months from dataset
    time_series = pd.to_datetime(data_subset['time'].values)
    unique_years = sorted(time_series.year.unique())
    # unique_years = [2011, 2012]
    unique_months = range(1, 13)  # Months from 1 to 12

    # Loop through each year & month
    for year in unique_years:
        for month in unique_months:
            # Filter data for the current month-year
            values_month = data_subset.sel(time=(time_series.year == year) & (time_series.month == month))

            # Clip chlorophyll-a data to the buffer area
            values_month = values_month.rio.write_crs("EPSG:4326", inplace=True)
            values_month_buffer = values_month.rio.clip(analysis_buffer.geometry,
                                                        analysis_buffer.crs,
                                                        all_touched=True)

            # Inverse the log-transformation
            # values_month_buffer_log = 10 ** values_month_buffer

            # Compute stats
            mean_val = float(values_month_buffer.mean().values)
            sum_val = float(values_month_buffer.sum().values)
            std_val = float(values_month_buffer.std().values)
            median_val = float(values_month_buffer.median().values)
            p5_val = float(np.nanpercentile(values_month_buffer.values, 5))
            p95_val = float(np.nanpercentile(values_month_buffer.values, 95))

            # For plotting
            # min_val = float(np.nanpercentile(values_month_buffer_log.values, 2))
            # max_val = float(np.nanpercentile(values_month_buffer_log.values, 98))
            # min_val_buffer = float(np.nanpercentile(values_month_buffer.values, 2))
            # max_val_buffer = float(np.nanpercentile(values_month_buffer.values, 98))

            # Append to results list
            results.append({
                "atoll": atoll_name,
                "year": year,
                "month": month,
                "sum": sum_val,
                "mean": mean_val,
                "std": std_val,
                "median": median_val,
                "p5": p5_val,
                "p95": p95_val
            })

            """ds_south_pac_raw = ds_variable.sel(time=(time_series.year == year) & (time_series.month == month))
            ds_south_pac = 10 ** ds_south_pac_raw

            # --- First Figure: Entire Dataset Region ---
            fig1, ax0 = plt.subplots(figsize=(8, 6))
            ds_south_pac.plot(ax=ax0, cmap="viridis")  # , vmin=min_val, vmax=max_val
            ellipse_gdf.plot(ax=ax0, color='none', edgecolor='#072ac8', linewidth=0.5)
            analysis_buffer_gdf.plot(ax=ax0, color='none', edgecolor='black', linewidth=1.5)
            ax0.set_xlabel("Longitude")
            ax0.set_ylabel("Latitude")
            ax0.grid(True)
            fig1.tight_layout()
            fig1.savefig(f"data/buffer_data/maps/{atoll_name}-{year}-{month:02}_region.png", dpi=500,
                         bbox_inches='tight')
            plt.close(fig1)

            # --- Second Figure: Full Region View ---
            fig2, ax1 = plt.subplots(figsize=(8, 6))
            values_month.plot(ax=ax1, cmap="viridis", alpha=0.9, vmin=min_val_buffer, vmax=max_val_buffer)
            atoll_ellipses.plot(ax=ax1, color='none', edgecolor='darkgrey', linewidth=2)
            ellipse_gdf.plot(ax=ax1, color='none', edgecolor='#072ac8', linewidth=2)
            atoll_hotspots = hotspot_gdf[(hotspot_gdf["atoll"] == atoll_name) &
                                         (hotspot_gdf["year"] == str(year)) &
                                         (hotspot_gdf["month"] == str(month))]
            ax1.scatter(atoll_hotspots.geometry.x, atoll_hotspots.geometry.y,
                        color='darkgreen', label='Hotspots', s=0.1, alpha=0.8, zorder=2)
            analysis_buffer_gdf.plot(ax=ax1, color='none', edgecolor='#072ac8', linestyle='--', linewidth=2)
            ax1.set_xlim([lon_min, lon_max])
            ax1.set_ylim([lat_min, lat_max])
            ax1.set_xlabel("Longitude")
            ax1.set_ylabel("Latitude")
            ax1.grid(True)
            fig2.tight_layout()
            fig2.savefig(f"data/buffer_data/maps/{atoll_name}-{year}-{month:02}_full.png", dpi=500, bbox_inches='tight')
            plt.close(fig2)

            # --- Third Figure: Zoomed Buffer Region ---
            fig3, ax2 = plt.subplots(figsize=(8, 6))
            values_month.plot(ax=ax2, cmap="viridis", alpha=0.9, vmin=min_val_buffer, vmax=max_val_buffer)  #
            ellipse_gdf.plot(ax=ax2, color='none', edgecolor='#072ac8', linewidth=2)
            analysis_buffer_gdf.plot(ax=ax2, color='none', edgecolor='#072ac8', linestyle='--', linewidth=2)
            atoll_hotspots = hotspot_gdf[(hotspot_gdf["atoll"] == atoll_name) &
                                         (hotspot_gdf["year"] == str(year)) &
                                         (hotspot_gdf["month"] == str(month))]

            ax2.scatter(atoll_hotspots.geometry.x, atoll_hotspots.geometry.y,
                        color='darkgreen', label='Hotspots', s=3, alpha=0.8, zorder=2)
            buffer_bounds = analysis_buffer_gdf.total_bounds
            ax2.set_xlim([buffer_bounds[0], buffer_bounds[2]])
            ax2.set_ylim([buffer_bounds[1], buffer_bounds[3]])
            ax2.set_xlabel("Longitude")
            ax2.set_ylabel("Latitude")
            ax2.grid(True)
            fig3.tight_layout()
            fig3.savefig(f"data/buffer_data/maps/{atoll_name}-{year}-{month:02}_zoomed.png", dpi=500,
                         bbox_inches='tight')
            plt.close(fig3)

            # --- Fourth Figure: Histogram ---
            fig4, ax3 = plt.subplots(figsize=(10, 6))
            all_values = 10 ** values_month.values.flatten()
            buffer_values = 10 ** values_month_buffer.values.flatten()
            all_values = all_values[~np.isnan(all_values)]
            buffer_values = buffer_values[~np.isnan(buffer_values)]
            combined_values = np.concatenate([all_values, buffer_values])
            vmin_hist, vmax_hist = np.percentile(combined_values, [1, 99])
            bins = np.linspace(vmin_hist, vmax_hist, 50)
            median_all = np.median(all_values)
            median_buffer = np.median(buffer_values)

            ax3.hist(all_values, bins=bins, alpha=0.6, color='skyblue', edgecolor='black', label='Full Region',
                     density=True)
            ax3.hist(buffer_values, bins=bins, alpha=0.6, color='lightcoral', edgecolor='black', label='Buffer Zone',
                     density=True)
            ax3.axvline(median_all, color='blue', linestyle='--', linewidth=2, label=f'Full Median: {median_all:.3f}')
            ax3.axvline(median_buffer, color='red', linestyle='--', linewidth=2,
                        label=f'Buffer Median: {median_buffer:.3f}')
            ax3.set_xlim(vmin_hist, vmax_hist)
            ax3.set_xlabel("Chl-a (mg/m³)")
            ax3.set_ylabel("Density")
            ax3.legend()
            ax3.grid(True)
            fig4.tight_layout()
            fig4.savefig(f"data/buffer_data/maps/{atoll_name}-{year}-{month:02}_hist.png", dpi=500, bbox_inches='tight')
            plt.close(fig4)

            counter += 1
            print(f"Loop iteration: {counter}")"""

# Convert to DataFrame
# results_df = pd.DataFrame(results)

# Save to CSV
# results_df.to_csv(f"data/buffer_data/{variable_name}/monthly_{variable_name}_{buffer_width}km_1993_2024.csv",
#                  index=False)
