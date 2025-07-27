import pandas as pd

# Load the data
high_thetao_df = pd.read_csv('el_nino/MHW/thetao_anomalies/high_thetao_1mo_anomaly.csv')

# Step 1: Assign seasonal year (July–June, named after the second year)
def assign_seasonal_year(row):
    if row['month'] >= 7:
        return row['year'] + 1
    else:
        return row['year']

high_thetao_df['seasonal_year'] = high_thetao_df.apply(assign_seasonal_year, axis=1)

# Step 2: All unique atolls
all_atolls = set(high_thetao_df['atoll'].unique())
total_atolls = len(all_atolls)

# Step 3: Prepare to collect qualifying seasonal years
qualified_seasonal_years = []

# Step 4: Loop through seasonal years (e.g., 2004 to 2024)
for season in range(2004, 2025):
    # Filter for rows that belong to this seasonal year
    season_subset = high_thetao_df[high_thetao_df['seasonal_year'] == season]

    # Atolls that had at least one high thetao anomaly during the seasonal year
    atolls_with_heatwave = set(season_subset['atoll'].unique())

    # Atolls that had NO heatwave this seasonal year
    atolls_without_heatwave = all_atolls - atolls_with_heatwave

    # Compute % of atolls without heatwave
    percent_no_heatwave = len(atolls_without_heatwave) / total_atolls

    if percent_no_heatwave >= 0.98:
        qualified_seasonal_years.append(season)

# Output the result
print("Seasonal years (Jul–Jun) where 98% of atolls had no high thetao anomalies:")
print(qualified_seasonal_years)
