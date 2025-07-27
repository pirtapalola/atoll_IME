import copernicusmarine
# copernicusmarine.login(username="ppalola", password="Appels11n1t!")

# cmems_mod_glo_phy_myint_0.083deg_P1M-m cmems_mod_glo_phy_my_0.083deg_P1M-m
# mlotst uo vo thetao
copernicusmarine.subset(
  dataset_id="cmems_obs-wind_glo_phy_my_l4_P1M",
  variables=["wind_speed"],
  minimum_longitude=-157.684,
  maximum_longitude=-127.737,
  minimum_latitude=-25.9255,
  maximum_latitude=-12.1711,
  # minimum_depth=0.5,
  # maximum_depth=30,
  start_datetime="1995-01-01T00:00:00",
  end_datetime="2024-12-01T00:00:00",
  output_filename="copernicus_wind_1995_2024.nc",
  output_directory="data/copernicus_data"
)
