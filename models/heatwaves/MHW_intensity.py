# Import libraries
import pandas as pd
import numpy as np

# -----------------
# 1. Load the data
# -----------------

# Read the csv file
df = pd.read_csv(f'data/buffer_data/thetao/monthly_thetao_30km_1993_2024.csv')
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# ----------------------------------------
# 2. Compute MHW intensity for all months
# ----------------------------------------

# Calculate monthly climatology and 90th percentile per atoll and month
climatology = (
    df.groupby(['atoll', 'month'])['mean']
    .agg(clim_mean='mean', clim_90th=lambda x: x.quantile(0.9))
    .reset_index()
)

# Merge climatology back with original data
df = df.merge(climatology, on=['atoll', 'month'])

# Compute intensity
df['MHW_intensity'] = (df['mean'] - df['clim_mean']) / (df['clim_90th'] - df['clim_mean'])
MHW_df = df[['atoll', 'year', 'month', 'MHW_intensity']]
# MHW_df.to_csv('el_nino/MHW_monthly_intensity_1993_2024.csv', index=False)

"""
# Step 1: Filter for February and March only
feb_mar_df = MHW_df[MHW_df['month'].isin([2, 3])]

# Step 2: Compute the average intensity per year (across all atolls and the two months)
yearly_avg_intensity = (
    feb_mar_df.groupby('year')['MHW_intensity']
    .mean()
    .reset_index(name='avg_MHW_intensity_FebMar')
)

# Step 3: Sort by how close the intensity is to neutral (0)
yearly_avg_intensity['abs_deviation'] = yearly_avg_intensity['avg_MHW_intensity_FebMar'].abs()
neutral_years = yearly_avg_intensity.sort_values(by='abs_deviation')

# Step 4: Output top N most neutral years
print(neutral_years.head(10))"""

# -----------------------------------------
# 2. Compute mean exceedance of each event
# -----------------------------------------

# Compute anomaly and exceedance
# df['anomaly'] = df['value'] - df['clim_mean']
# df['exceedance'] = df['value'] - df['clim_90th']  # How much above the 90th percentile

df = df[df['year'] >= 2003]
df = df[df['MHW_intensity'] >= 1]

# Group by atoll and group_id (i.e., MHW event), then calculate mean exceedance
event_summary = df.groupby(['atoll']).agg(
    start_year=('year', 'min'),
    start_month=('month', 'min'),
    end_year=('year', 'max'),
    end_month=('month', 'max'),
    duration=('month', 'count'),
    # mean_exceedance=('exceedance', 'mean')
).reset_index()

# Compute the max duration for each atoll

MHW_max_duration = event_summary.groupby('atoll', as_index=False)['duration'].quantile(0.9)
MHW_max_duration.to_csv('el_nino/MHW/MHW_p90_duration.csv', index=False)

# Record the number of events per atoll
# event_summary_month_count = event_summary.groupby('atoll', as_index=False)['group_id'].max()

# Save to CSV
# event_summary_month_count.to_csv('el_nino/MHW_event_summary_event_count.csv', index=False)

# --------------------------------------------------
# 3. Compute mean exceedance on log-transformed data
# --------------------------------------------------
"""
# Apply log-transformation
event_summary1 = event_summary.copy()
event_summary1['log_mean_exceedance'] = np.log(event_summary1['mean_exceedance'])

# Compute mean of log-transformed values
MHW_log_mean_exceedance = event_summary1.groupby('atoll', as_index=False)['log_mean_exceedance'].mean()
# MHW_log_mean_exceedance.to_csv('el_nino/MHW/MHW_log_mean_exceedance.csv', index=False)

# ---------------------------------------
# 4. Compute max exceedance and duration
# ---------------------------------------

# Compute the max exceedance for each atoll
MHW_max_exceedance = event_summary.groupby('atoll', as_index=False)['mean_exceedance'].max()
# MHW_max_exceedance.to_csv('el_nino/MHW_max_exceedance.csv', index=False)"""


# -----------------------------
# 5. Compute monthly intensity
# -----------------------------

# Compute intensity
# df['MHW_intensity'] = (df['thetao'] - df['clim_mean']) / (df['clim_90th'] - df['clim_mean'])
# MHW_df = df[['atoll', 'year', 'month', 'MHW_intensity']]
# MHW_df.to_csv('el_nino/MHW_monthly_intensity.csv', index=False)"""

# -------------------------
# 6. Compute max intensity
# -------------------------

# Compute the max intensity for each atoll
# MHW_max_intensity = MHW_df.groupby('atoll', as_index=False)['MHW_intensity'].max()
MHW_df = MHW_df[MHW_df['year'] >= 2003]
MHW_df = MHW_df[MHW_df['MHW_intensity'] >= 1]
MHW_90p_intensity = MHW_df.groupby('atoll', as_index=False)['MHW_intensity'].median()
# MHW_90p_intensity.to_csv('el_nino/MHW/MHW_median_intensity.csv', index=False)

"""
# --------------------------------------------------
# 7. Compute mean intensity on log-transformed data
# --------------------------------------------------

# Apply log-transformation
MHW_df = MHW_df.copy()
MHW_df['log_MHW_intensity'] = np.log(MHW_df['MHW_intensity'])

# Compute mean of log-transformed values
MHW_log_mean_intensity = MHW_df.groupby('atoll', as_index=False)['log_MHW_intensity'].mean()
# MHW_log_mean_intensity.to_csv('el_nino/MHW_log_mean_intensity.csv', index=False)"""
