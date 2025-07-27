import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

df_scatter = pd.read_csv('ANALYSIS/atoll_IME_MHW_summary_mean.csv')
df_scatter['log_IME_MHW_perc'] = np.log(df_scatter['IME_MHW_perc'])
df_scatter['log_IME_chl'] = np.log(df_scatter['IME_chl'])

# Calculate the correlation coefficient
# r_val, p_val = pearsonr(df_scatter['IME_MHW_perc'], df_scatter['log_IME_chl'])
r_val, p_val = spearmanr(df_scatter['IME_MHW_perc'], df_scatter['log_IME_chl'])
print(r_val, p_val)

# Step 2: Scatter plot
sns.regplot(
    data=df_scatter,
    x='IME_MHW_perc',
    y='log_IME_chl',
    scatter_kws={'color': '#90e8db', 's': 60},  # seagreen
    line_kws={'color': 'black', 'linestyle': '--'},
    ci=None
)

plt.text(
    0.05, 0.95, f"$\\rho$ = {r_val:.2f}, $p$ < 0.001",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top'
)

# Step 3: Add labels and formatting
plt.xlabel('Frequency of IME during MHW (%)', fontsize=12)
plt.ylabel('Log(annual IME magnitude [mg/m$^3$])', fontsize=12)
plt.ylim(bottom=0)
plt.xlim(left=2)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
