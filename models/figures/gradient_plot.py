import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define the variable of interest
dataset = pd.read_csv('el_nino/MHW/MHW_max_intensity.csv')
column_of_interest = 'MHW_intensity'
var_name = "Max MHW intensity"  # Chlorophyll-a (mg/m$^3$) SST (°C)
output_name = "MHW_max_intensity_plot"

print(dataset[column_of_interest].min())
print(dataset[column_of_interest].max())

# Load atoll center coordinates
coords = pd.read_csv("data/atolls/roi_3_atolls.csv")

# Merge data with coordinates
coords = coords.merge(dataset, on="atoll", how="left")
coords = coords.dropna(subset=[column_of_interest])  # Remove NaN

# Use your full atoll coordinate dataset
lons = coords['lon'].values
lats = coords['lat'].values
positions = np.column_stack([lons, lats])

# Fit a line (principal direction) using PCA
pca = PCA(n_components=1)
pca.fit(positions)
axis_vector = pca.components_[0]  # NE–SW direction

# Choose a reference point (e.g., mean location)
reference_point = positions.mean(axis=0)

# Center positions and project onto the axis vector
centered_positions = positions - reference_point
projected_distance = centered_positions @ axis_vector

# Add to dataframe
coords['NE_SW_position'] = projected_distance

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(coords['NE_SW_position'], coords[column_of_interest],
            c=coords[column_of_interest], cmap='seismic', edgecolor='k')
plt.xlabel('Position along NE–SW axis')
plt.ylabel(var_name)
plt.grid(True)
plt.tight_layout()
# plt.savefig('figures/MHW_90p_trend_NESW.png', dpi=500)
plt.show()

# Save the dataframe as a CSV file
coords.to_csv("MHW_max_intensity_gradient_data.csv", index=False)
