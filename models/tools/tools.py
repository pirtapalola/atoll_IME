import pandas as pd
import xarray as xr
import numpy as np
import netCDF4
import cftime
from skimage import measure
import cv2
import geopandas as gpd
from pyproj import CRS
from scipy.ndimage import binary_dilation
from shapely.affinity import scale
import pyproj
from shapely.geometry import Point, LineString, Polygon
from dtaidistance import dtw
from PyEMD import EMD
from scipy.signal import hilbert


def shallow_water_mask(filepath):
    gebco_df = netCDF4.Dataset(filepath)

    # Extract elevation, longitude, and latitude as NumPy arrays
    elevation = np.array(gebco_df['elevation'][:])
    lon_gebco = np.array(gebco_df['lon'][:])
    lat_gebco = np.array(gebco_df['lat'][:])

    # Create mask for elevation >= -30 meters
    mask = elevation >= -30

    # Extend the mask
    extended_mask = binary_dilation(mask, structure=np.ones((3, 3)), iterations=8)

    # Convert the boolean mask to a numeric type and then to a xarray DataArray
    mask_numeric_da = extended_mask.astype(int)
    mask_da = xr.DataArray(mask_numeric_da, coords=[lat_gebco, lon_gebco], dims=["lat", "lon"])

    return mask_da


def extract_chl_a(roi_coordinates, t_intervals):
    with xr.open_dataset('https://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-1km-DAILY',
                         decode_cf=False) as ds:
        time_units = ds['time'].attrs['units']
        time_calendar = ds['time'].attrs.get('calendar', 'standard')

    # List to accumulate each monthly chunk of chlorophyll-a data
    chlor_a_data = []

    # Loop over each month, downloading data chunk by chunk
    for i in range(len(t_intervals) - 1):
        start_date = t_intervals[i]
        end_date = t_intervals[i + 1]

        # Convert start and end dates to numeric values
        start_num = cftime.date2num(start_date, units=time_units, calendar=time_calendar)
        end_num = cftime.date2num(end_date, units=time_units, calendar=time_calendar)

        # Slice data for the current month and region of interest
        chunk = ds.sel(lon=slice(roi_coordinates["lon1"][0], roi_coordinates["lon2"][0]),
                       lat=slice(roi_coordinates["lat1"][0], roi_coordinates["lat2"][0]),
                       time=slice(start_num, end_num))

        # Extract chlorophyll-a data
        chlor_a_chunk = chunk['chlor_a']

        # Handle the _FillValue by masking it out
        fill_value = chlor_a_chunk.attrs.get('_FillValue', None)
        if fill_value is not None:
            chlor_a_chunk = chlor_a_chunk.where(chlor_a_chunk != fill_value)

        # Append the cleaned data chunk to our list
        chlor_a_data.append(chlor_a_chunk)

    # Concatenate all chunks along the time dimension
    chl_a_combined = xr.concat(chlor_a_data, dim='time')
    return chl_a_combined


def geodesic_buffer(geometry, distance):
    """
    Create a geodesic buffer around a geometry.

    Parameters:
    - geometry: The input geometry (shapely object).
    - distance: Buffer distance in meters.

    Returns:
    - A buffered geometry in WGS84 coordinates (longitude, latitude).
    """
    # Define the WGS84 projection (longitude/latitude)
    proj_wgs84 = pyproj.CRS("EPSG:4326")

    # Use Geod (geodesic) for accurate distance buffering on the ellipsoid
    geod = pyproj.Geod(ellps="WGS84")

    def geodesic_point_buffer(lon, lat, dist):
        """
        Create a geodesic buffer around a single point.

        Parameters:
        - lon: Longitude of the point.
        - lat: Latitude of the point.
        - dist: Distance in meters.

        Returns:
        - A polygon representing the geodesic buffer.
        """
        # Create a series of points representing the buffer
        angle_coords = []
        for angle in range(0, 360):
            # Compute the destination point for each angle and distance
            dest_lon, dest_lat, _ = geod.fwd(lon, lat, angle, dist)
            angle_coords.append((dest_lon, dest_lat))

        # Return as a shapely Polygon
        return Polygon(angle_coords)

    # If the geometry is a Point, apply the geodesic_point_buffer function
    if geometry.geom_type == 'Point':
        lon1, lat1 = geometry.x, geometry.y
        return geodesic_point_buffer(lon1, lat1, distance)

    # If the geometry is a LineString, create buffers around each coordinate point and merge
    elif geometry.geom_type == 'LineString':
        buffered = [geodesic_point_buffer(lon, lat, distance) for lon, lat in geometry.coords]
        return gpd.GeoSeries(buffered).unary_union  # Merge all buffer parts

    # If the geometry is a Polygon, buffer around its exterior
    elif geometry.geom_type == 'Polygon':
        buffered = [geodesic_point_buffer(lon, lat, distance) for lon, lat in geometry.exterior.coords]
        return gpd.GeoSeries(buffered).unary_union  # Merge all buffer parts

    return None  # Return None if geometry type is unsupported


def contours_to_gdf(mask_interpolated, chl_masked):
    # Extract the mask as a NumPy array
    mask_interp_array = mask_interpolated.values

    # Extract the mask contours. The "level" argument defines the threshold value for contour detection.
    mask_contours = measure.find_contours(mask_interp_array, 0)

    # Convert to OpenCV
    contours = []  # Create an empty list
    for con_i in mask_contours:  # Loop through the mask contours
        cv_contour = []
        for point in con_i:
            integer_list = [int(point[1]), int(point[0])]
            cv_contour.append([integer_list])
        cv_contour = np.array(cv_contour)  # Convert to NumPy array
        contours.append(cv_contour)

    # Remove small contours that likely result from noise
    filtered_contours = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])  # Calculate the area contained by each contour line
        if area > 1:  # The area must be greater than one pixel
            filtered_contours.append(mask_contours[i])

    # Convert filtered contours to LineString objects
    lines = []
    for contour in filtered_contours:
        # Convert contour points to (lon, lat) based on the grid of the mask
        coords = [(chl_masked['lon'].values[int(pt[1])], chl_masked['lat'].values[int(pt[0])]) for pt in contour]
        lines.append(LineString(coords))

    # Create a GeoDataFrame from the lines
    contour_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

    return contour_gdf


def identify_closest_contour(contour_geodataframe, atoll_name):
    # Define the atoll location (latitude, longitude)
    atoll_df = pd.read_csv("data/atolls/atoll_center_points.csv")
    target_location = atoll_df[atoll_df["atoll"] == atoll_name]
    longitude = target_location.iloc[0]['lon']
    latitude = target_location.iloc[0]['lat']
    center_point = Point(longitude, latitude)  # (lon, lat)

    # Create a GeoDataFrame for the specified location
    specified_location_gdf = gpd.GeoDataFrame(geometry=[center_point], crs=contour_geodataframe.crs)

    # Project both to a suitable projected CRS for accurate distance calculations
    # Using an Azimuthal Equidistant projection centered on the specified location
    aeqd_proj = CRS(proj='aeqd', lat_0=latitude, lon_0=longitude, datum='WGS84').to_proj4()

    contours_gdf_proj = contour_geodataframe.to_crs(aeqd_proj)
    specified_location_gdf_proj = specified_location_gdf.to_crs(aeqd_proj)

    # Calculate distance from specified location to each contour line
    contours_gdf_proj['distance'] = contours_gdf_proj.geometry.distance(
        specified_location_gdf_proj.geometry.iloc[0])

    # Select the contour line with the minimum distance
    min_distance_index = contours_gdf_proj['distance'].idxmin()
    closest_contour = contour_geodataframe.loc[[min_distance_index]]  # Keep as GeoDataFrame

    # Save the selected contour line as a shapefile
    selected_contour_gdf = gpd.GeoDataFrame(geometry=closest_contour.geometry, crs="EPSG:4326")

    return selected_contour_gdf, center_point


def create_offshore_buffer(single_contour_gdf, distance_m):
    # Create a buffer containing the pixels closest to the contour
    single_contour_gdf['buffered_geometry'] = single_contour_gdf['geometry'].apply(
        lambda geom: geodesic_buffer(geom, distance_m))
    offshore_buffer = single_contour_gdf['buffered_geometry']

    # Create a new GeoDataFrame with the buffered geometries
    offshore_buffer_gdf = gpd.GeoDataFrame(
        single_contour_gdf[['buffered_geometry']],
        geometry='buffered_geometry',
        crs=single_contour_gdf.crs
    )

    return offshore_buffer, offshore_buffer_gdf


def draw_ellipse(contours_to_include_gdf, centre_pt):

    # Step 1: Reproject to a local projected CRS
    # Use UTM zone based on the contour's centroid
    contour_centroid = contours_to_include_gdf.geometry.centroid.iloc[0]
    utm_crs = f"EPSG:{32600 + int((contour_centroid.x + 180) // 6 + 1)}"  # Calculate UTM zone EPSG
    projected_gdf = contours_to_include_gdf.to_crs(utm_crs)

    # Step 2: Calculate convex hull and bounding ellipse
    convex_hull = projected_gdf.geometry.union_all().convex_hull

    # Compute the minimal bounding ellipse parameters
    def minimal_bounding_ellipse(geom):
        """
        Compute a minimal bounding ellipse for a given geometry.
        """
        ellipse = scale(geom, xfact=1.1, yfact=1.1, origin="center")  # Adjust factors as needed
        return ellipse

    ellipse_geom = minimal_bounding_ellipse(convex_hull)

    # Step 3: Transform ellipse back to EPSG:4326
    ellipse_gdf = gpd.GeoDataFrame(geometry=[ellipse_geom], crs=utm_crs).to_crs("EPSG:4326")

    # Step 4: Set the center to the provided atoll center point
    ellipse_gdf["geometry"] = ellipse_gdf.translate(
        xoff=centre_pt.x - contour_centroid.x,
        yoff=centre_pt.y - contour_centroid.y,
    )
    ellipse_gdf = gpd.GeoDataFrame(geometry=[ellipse_geom], crs=utm_crs).to_crs("EPSG:4326")

    return ellipse_gdf


def log_transform(chl_dataset, epsilon=1e-6):
    return np.log10(chl_dataset + epsilon)


# Extract the coordinates that define the rectangular region of interest
def get_roi_coordinates(shapefile_name):
    roi_gdf = gpd.read_file("data/roi_atolls/" + shapefile_name + "_roi.shp")
    roi_coordinates = list(roi_gdf["geometry"][0].exterior.coords)
    roi_lat1 = roi_coordinates[2][1]
    roi_lat2 = roi_coordinates[0][1]
    roi_lon1 = roi_coordinates[0][0]
    roi_lon2 = roi_coordinates[2][0]
    roi_df = pd.DataFrame()
    roi_df["lon1"] = [roi_lon1]
    roi_df["lon2"] = [roi_lon2]
    roi_df["lat1"] = [roi_lat1]
    roi_df["lat2"] = [roi_lat2]
    return roi_df


def slice_to_roi(roi_coordinates, ds):
    # Slice data to the region of interest
    chlor_a_chunk = ds.sel(lon=slice(roi_coordinates["lon1"][0], roi_coordinates["lon2"][0]),
                           lat=slice(roi_coordinates["lat1"][0], roi_coordinates["lat2"][0]))
    return chlor_a_chunk


def draw_ellipse2(contour_geometry):
    """
    Create an ellipse geometry from a single contour geometry (LineString).
    """
    # Reproject the geometry to a local projected CRS
    projected_geom = gpd.GeoSeries([contour_geometry], crs="EPSG:4326")

    # Calculate convex hull and bounding ellipse
    convex_hull = projected_geom.geometry.union_all().convex_hull

    # Compute the minimal bounding ellipse parameters
    def minimal_bounding_ellipse(geom):
        """
        Compute a minimal bounding ellipse for a given geometry.
        """
        ellipse = scale(geom, xfact=1.1, yfact=1.1, origin="center")  # Adjust factors as needed
        return ellipse

    ellipse_geom = minimal_bounding_ellipse(convex_hull)

    # Transform ellipse back to EPSG:4326
    ellipse_gdf = gpd.GeoDataFrame(geometry=[ellipse_geom], crs="EPSG:4326")
    return ellipse_gdf


# Function to calculate MAPE
def calc_mape(series1, series2):
    mask = (series1 != 0) & (series2 != 0)  # Avoid division by zero

    return np.mean(np.abs((series1[mask] - series2[mask]) / series1[mask]))


# Function to calculate the derivative
def compute_derivative(series):
    return np.gradient(series)


# Function to calculate Derivative Dynamic Time Warping (DDTW)
def ddtw_distance(series1, series2):
    # Compute derivatives
    d_series1 = compute_derivative(series1)
    d_series2 = compute_derivative(series2)

    # Apply DTW on the derivatives
    dist = dtw.distance(d_series1, d_series2)
    return dist


def decompose_with_multiple_s(signal1, s_range):
    all_imfs = []
    max_imfs = 0

    for S in s_range:
        emd1 = EMD()
        emd1.S_number = S
        imfs = emd1(signal1)
        all_imfs.append(imfs)
        max_imfs = max(max_imfs, imfs.shape[0])

    # Pad IMFs to have same shape
    padded_imfs = []
    for imfs_var in all_imfs:
        num_imfs1, length = imfs_var.shape[0], imfs_var.shape[1]
        pad = np.zeros((max_imfs - num_imfs1, length))
        padded = np.vstack([imfs_var, pad])
        padded_imfs.append(padded)

    all_imfs_array = np.stack(padded_imfs)
    imf_mean1 = np.mean(all_imfs_array, axis=0)
    imf_std1 = np.std(all_imfs_array, axis=0)

    return imf_mean1, imf_std1


def group_imfs(imf_array):
    """
    Groups IMFs into seasonal (first 2), interannual (middle), and trend (last) components.
    """
    num_imfs = imf_array.shape[0]

    seasonal = np.sum(imf_array[0:2], axis=0)
    interannual = np.sum(imf_array[2:], axis=0) if num_imfs > 3 else np.zeros_like(imf_array[0])
    # trend = imf_array[-1]

    return seasonal, interannual #, trend


def get_instantaneous_amplitude(signal_data):
    """
    Computes the instantaneous amplitude (envelope) of a signal using Hilbert transform.
    """
    analytic_signal = hilbert(signal_data)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def phase_randomize(signal1):
    n1 = len(signal1)
    fft_vals = np.fft.fft(signal1)
    amplitudes = np.abs(fft_vals)
    phases = np.angle(fft_vals)

    random_phases = np.random.uniform(0, 2*np.pi, n1//2 - 1)
    phases[1:n1//2] = random_phases
    phases[-(n1//2)+1:] = -random_phases[::-1]

    randomized_fft = amplitudes * np.exp(1j * phases)
    surrogate1 = np.fft.ifft(randomized_fft).real
    return surrogate1

