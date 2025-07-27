import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
MHW_intensity_df = pd.read_csv('el_nino/MHW/MHW_monthly_intensity_1993_2024.csv')
MHW_intensity_df = MHW_intensity_df[MHW_intensity_df['year'] >= 2003]


# Step 1: Create 'seasonal_year' column (July–June, named after the second year)
def assign_seasonal_year(row):
    if row['month'] >= 7:  # July to Dec
        return row['year'] + 1
    else:  # Jan to June
        return row['year']


MHW_intensity_df['seasonal_year'] = MHW_intensity_df.apply(assign_seasonal_year, axis=1)
seasonal_df = MHW_intensity_df.copy()

# Max intensity per atoll per seasonal year
max_intensity_per_atoll_season = seasonal_df.groupby(['atoll', 'seasonal_year'])['MHW_intensity'].max().reset_index()

# Mean intensity across all atolls per seasonal year
mean_intensity_per_season = max_intensity_per_atoll_season.groupby('seasonal_year')['MHW_intensity'].mean().reset_index()
top_3_mean_intensity = mean_intensity_per_season.sort_values(by='MHW_intensity', ascending=False).head(3)
print(top_3_mean_intensity)

# For each atoll, find the seasonal year with max intensity
max_season_per_atoll = max_intensity_per_atoll_season.loc[
    max_intensity_per_atoll_season.groupby('atoll')['MHW_intensity'].idxmax()]

# Count how often each seasonal year was the max for an atoll
year_counts = max_season_per_atoll['seasonal_year'].value_counts()

# Most common max-intensity seasonal year
most_common_year = year_counts.idxmax()
count = year_counts.max()
top_3_years = year_counts.sort_values(ascending=False).head(3)
print(top_3_years)
print(f"The most common max-intensity seasonal year (Jul–Jun) across atolls was {most_common_year}, appearing {count} times.")

# Count specific years (e.g., 2016, 2024)
count_2016 = (max_season_per_atoll['seasonal_year'] == 2016).sum()
count_2024 = (max_season_per_atoll['seasonal_year'] == 2024).sum()

print(f"2016 (Jul 2015–Jun 2016) was the max-intensity seasonal year for {count_2016} atolls.")
print(f"2024 (Jul 2023–Jun 2024) was the max-intensity seasonal year for {count_2024} atolls.")

"""SECTION 2"""

# Count the number of MHW months per atoll
MHW_2016 = seasonal_df[(seasonal_df['seasonal_year'] == 2016) & (seasonal_df['MHW_intensity'] >= 1)]
MHW_month_counts_2016 = MHW_2016.groupby('atoll').size().reset_index(name='MHW_months_2016')
MHW_month_counts_2016.to_csv('el_nino/MHW_revised/MHW_month_counts_2016.csv')
print(MHW_month_counts_2016['MHW_months_2016'].max())

MHW_2024 = seasonal_df[(seasonal_df['seasonal_year'] == 2024) & (seasonal_df['MHW_intensity'] >= 1)]

# Count the number of such months per atoll
MHW_month_counts_2024 = MHW_2024.groupby('atoll').size().reset_index(name='MHW_months_2024')
MHW_month_counts_2024.to_csv('el_nino/MHW_revised/MHW_month_counts_2024.csv')
print(MHW_month_counts_2024['MHW_months_2024'].max())

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(MHW_month_counts_2024['MHW_months_2024'], bins=50)
plt.tight_layout()
plt.show()
