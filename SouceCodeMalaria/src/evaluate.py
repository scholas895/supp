import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming you already have a DataFrame `results_df` with columns:
# Model, MSE, RMSE, MAE, R2, MAPE (%)
# For example, from the previous step:
# results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE (%)'])

# If not, create it manually:
# results_df = pd.DataFrame({
#     'Model': ['XGBoost (simple)', 'Random Forest', 'XGBoost (tuned)', 
#               'ST‑AWXE (adaptive)', 'ST‑AWXE (combined)'],
#     'RMSE': [42.1987, 40.6321, 38.6123, 36.0559, 35.9303],
#     'MAE': [20.3456, 19.8765, 18.4321, 17.1234, 16.9876],
#     'R2': [0.8523, 0.8612, 0.8734, 0.8912, 0.8934],
#     'MAPE (%)': [15.67, 14.98, 13.45, 12.11, 11.89]
# })

# -----------------------------
# Figure: Multi-panel comparison
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

models = results_df['Model'].values

# Panel 1: RMSE
ax1.bar(models, results_df['RMSE'], color='#2E86AB', edgecolor='black', alpha=0.8)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_title('Root Mean Square Error (lower is better)', fontsize=12)
ax1.tick_params(axis='x', rotation=45)

# Panel 2: MAE
ax2.bar(models, results_df['MAE'], color='#A23B72', edgecolor='black', alpha=0.8)
ax2.set_ylabel('MAE', fontsize=12)
ax2.set_title('Mean Absolute Error (lower is better)', fontsize=12)
ax2.tick_params(axis='x', rotation=45)

# Panel 3: R²
ax3.bar(models, results_df['R2'], color='#F18F01', edgecolor='black', alpha=0.8)
ax3.set_ylabel('R²', fontsize=12)
ax3.set_title('Coefficient of Determination (higher is better)', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
ax3.set_ylim(0, 1)  # R² ranges 0-1

# Panel 4: MAPE (%)
ax4.bar(models, results_df['MAPE (%)'], color='#C73E1D', edgecolor='black', alpha=0.8)
ax4.set_ylabel('MAPE (%)', fontsize=12)
ax4.set_title('Mean Absolute Percentage Error (lower is better)', fontsize=12)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('evaluation_metrics_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()