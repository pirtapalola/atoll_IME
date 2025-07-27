import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
dataset_2014_2016 = pd.read_csv('ANALYSIS/delta_difference_2014_2016.csv')
dataset_2014_2024 = pd.read_csv('ANALYSIS/delta_difference_2014_2024.csv')

# Add a label column to each dataset
dataset_2014_2016['Period'] = '2014-2016'
dataset_2014_2024['Period'] = '2014-2024'

# Concatenate datasets
combined = pd.concat([dataset_2014_2016, dataset_2014_2024])

# Plot
plt.figure(figsize=(12,7))
sns.histplot(
    data=combined,
    x='delta_difference',
    hue='Period',
    bins=70,
    kde=False,
    palette='Set2',
    alpha=0.6,
    element='step'
)

plt.xlabel("Difference in IME magnitude (mg/m$^3$)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Merge on 'atoll'
merged = pd.merge(dataset_2014_2016[["delta_difference", "atoll"]], dataset_2014_2024[["delta_difference", "atoll"]],
                  on='atoll')
print(merged)

# Define a function to get sign: +1 for positive, -1 for negative, 0 for zero
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# Apply sign function to both columns
merged['sign_2014_2016'] = merged['delta_difference_x'].apply(sign)
merged['sign_2014_2024'] = merged['delta_difference_y'].apply(sign)

# Define a new column for consistency
def consistency(row):
    if row['sign_2014_2016'] == row['sign_2014_2024']:
        if row['sign_2014_2016'] == 0:
            return 'No difference'
        elif row['sign_2014_2016'] == 1:
            return 'Consistently positive'
        elif row['sign_2014_2016'] == -1:
            return 'Consistently negative'
    else:
        return 'Sign changed'

merged['consistency'] = merged.apply(consistency, axis=1)

# Count how many atolls fall into each category
count_consistency = merged['consistency'].value_counts()

print(count_consistency)

# Plot the results
plt.figure(figsize=(8,5))
sns.countplot(data=merged, x='consistency', order=[
    'Consistently positive',
    'Consistently negative',
    'Sign changed',
    'No difference'
], palette='Set1')

plt.xlabel('Consistency Category')
plt.ylabel('Number of Atolls')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Save the consistency results to CSV
output = merged[['atoll', 'consistency']]
output.to_csv('data/sign_behavior_consistency.csv', index=False)
print("Saved sign behavior consistency to: data/sign_behavior_consistency.csv")
