import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
df = pd.read_csv('malaria_fully_combined_filled.csv')
target_cols = ['cases', 'deaths_rich']
identifier_cols = ['country', 'year']
exclude_from_features = identifier_cols + target_cols + ['deaths_who', 'high_risk']
feature_cols = [c for c in df.columns if c not in exclude_from_features and np.issubdtype(df[c].dtype, np.number)]

data = df[feature_cols + target_cols + identifier_cols].copy().dropna()

# Encode country
le = LabelEncoder()
data['country_encoded'] = le.fit_transform(data['country'])
feature_cols.append('country_encoded')

X = data[feature_cols].values
y_cases = data['cases'].values
y_deaths = data['deaths_rich'].values
y = np.column_stack([y_cases, y_deaths])

years = data['year'].values
countries = data['country_encoded'].values
country_names = data['country'].values  # for plotting

# Temporal split
train_mask = years < 2013
val_mask = (years >= 2013) & (years <= 2017)
test_mask = years >= 2018

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

country_val = countries[val_mask]
year_val = years[val_mask]
country_test = countries[test_mask]
year_test = years[test_mask]
country_names_test = country_names[test_mask]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Train samples: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# We will predict only 'cases' for clarity (same pipeline works for 'deaths')
y_train_1 = y_train[:, 0]
y_val_1 = y_val[:, 0]
y_test_1 = y_test[:, 0]

# -----------------------------
# 2. Train baseline models
# -----------------------------
print("\n[1/5] Training XGBoost (simple)...")
xgb_simple = xgb.XGBRegressor(random_state=42, n_estimators=100)
xgb_simple.fit(X_train_scaled, y_train_1)
y_pred_simple = xgb_simple.predict(X_test_scaled)

print("[2/5] Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_1)
y_pred_rf = rf.predict(X_test_scaled)

print("[3/5] Training XGBoost (tuned)...")
xgb_tuned = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
xgb_tuned.fit(X_train_scaled, y_train_1)
y_pred_tuned = xgb_tuned.predict(X_test_scaled)

# -----------------------------
# 3. ST‑AWXE base models (ensemble components)
# -----------------------------
def train_base_models(X_train, y_train, X_val, y_val):
    models = {}
    # XGBoost
    xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_m.fit(X_train, y_train)
    models['xgb'] = xgb_m
    # LightGBM
    lgb_m = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_m.fit(X_train, y_train)
    models['lgb'] = lgb_m
    # CatBoost
    cb_m = cb.CatBoostRegressor(iterations=200, depth=5, learning_rate=0.05, random_seed=42, verbose=False)
    cb_m.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
    models['cb'] = cb_m
    # Random Forest
    rf_m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_m.fit(X_train, y_train)
    models['rf'] = rf_m
    return models

print("[4/5] Training ST‑AWXE base models...")
base_models = train_base_models(X_train_scaled, y_train_1, X_val_scaled, y_val_1)

# Validation predictions from base models
val_preds_base = np.column_stack([m.predict(X_val_scaled) for m in base_models.values()])

# Meta‑learner (Ridge) using base predictions + country + year
meta_features_val = np.column_stack([val_preds_base, country_val, year_val])
meta_scaler = StandardScaler()
meta_features_val_scaled = meta_scaler.fit_transform(meta_features_val)
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_features_val_scaled, y_val_1)

# Adaptive weighted prediction
def adaptive_weighted_predict(test_feat, test_country, test_year, base_models_dict, X_val_orig, y_val_orig, val_preds_base_orig, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_val_scaled)
    _, indices = nn.kneighbors(test_feat.reshape(1, -1))
    neighbor_idx = indices[0]
    weights = []
    for i, name in enumerate(base_models_dict.keys()):
        pred_neighbors = val_preds_base_orig[neighbor_idx, i]
        true_neighbors = y_val_orig[neighbor_idx]
        mae = np.mean(np.abs(pred_neighbors - true_neighbors))
        weights.append(1.0 / (mae + 1e-6))
    weights = np.array(weights)
    weights /= weights.sum()
    test_preds = np.array([m.predict(test_feat.reshape(1, -1))[0] for m in base_models_dict.values()])
    return np.dot(weights, test_preds)

print("   Computing adaptive weighted predictions...")
y_pred_adapt = np.array([adaptive_weighted_predict(X_test_scaled[i], country_test[i], year_test[i],
                                                   base_models, X_val_scaled, y_val_1, val_preds_base, k=5)
                         for i in range(len(X_test_scaled))])

# Meta‑learner predictions on test
test_preds_base = np.column_stack([m.predict(X_test_scaled) for m in base_models.values()])
meta_features_test = np.column_stack([test_preds_base, country_test, year_test])
meta_features_test_scaled = meta_scaler.transform(meta_features_test)
y_pred_meta = meta_learner.predict(meta_features_test_scaled)

# Combined prediction
y_pred_combined = (y_pred_adapt + y_pred_meta) / 2

print("[5/5] All models trained.")

# -----------------------------
# 4. Evaluation & metrics (MSE, RMSE, MAE, R², MAPE)
# -----------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE (%) - protects against zero values by adding a small epsilon"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_mask = y_true != 0
    if np.sum(nonzero_mask) == 0:
        return 100.0
    return np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100

models_dict = {
    'XGBoost (simple)': y_pred_simple,
    'Random Forest': y_pred_rf,
    'XGBoost (tuned)': y_pred_tuned,
    'ST‑AWXE (adaptive)': y_pred_adapt,
    'ST‑AWXE (combined)': y_pred_combined
}

results = []
for name, pred in models_dict.items():
    mse = mean_squared_error(y_test_1, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_1, pred)
    r2 = r2_score(y_test_1, pred)
    mape = mean_absolute_percentage_error(y_test_1, pred)
    results.append((name, mse, rmse, mae, r2, mape))

# Hyperparameter search space table (for reference)
hyperparams = {
    'Parameter': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                  'colsample_bytree', 'reg_alpha', 'reg_lambda'],
    'Search range / values': [''] * 7,
    'Description': [
        'Number of boosting rounds',
        'Maximum tree depth (controls overfitting)',
        'Step size shrinkage (eta)',
        'Fraction of samples used per tree',
        'Fraction of features used per tree',
        'L1 regularisation on weights',
        'L2 regularisation on weights'
    ]
}
df_hyper = pd.DataFrame(hyperparams)

print("\n" + "="*80)
print("XGBoost Hyperparameter Search Space")
print("="*80)
print(df_hyper.to_string(index=False))
print("="*80)

df_hyper.to_csv('xgb_hyperparameters_empty.csv', index=False)
print("\nEmpty table saved as 'xgb_hyperparameters_empty.csv' – fill in the ranges manually.")

# Print performance table
print("\n" + "="*90)
print("PERFORMANCE COMPARISON ON TEST SET (2018-2021) – Target: cases")
print("="*90)
print(f"{'Model':<25} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<10} {'MAPE (%)':<10}")
print("-"*90)
for name, mse, rmse, mae, r2, mape in results:
    print(f"{name:<25} {mse:<12.4f} {rmse:<12.4f} {mae:<12.4f} {r2:<10.4f} {mape:<10.2f}")

# Save results to CSV
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE (%)'])
results_df.to_csv('model_performance.csv', index=False)
print("\nPerformance table saved to 'model_performance.csv'")

# -----------------------------
# 5. Figures (saved as PDF)
# -----------------------------
# Figure 1: Bar chart comparing RMSE and MAE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
model_names = [r[0] for r in results]
rmse_vals = [r[2] for r in results]
mae_vals = [r[3] for r in results]

ax1.bar(model_names, rmse_vals, color='steelblue', edgecolor='black')
ax1.set_title('Root Mean Square Error (RMSE)', fontsize=14)
ax1.set_ylabel('RMSE')
ax1.tick_params(axis='x', rotation=45)

ax2.bar(model_names, mae_vals, color='salmon', edgecolor='black')
ax2.set_title('Mean Absolute Error (MAE)', fontsize=14)
ax2.set_ylabel('MAE')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison_bar.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Figure 2: Scatter plots (actual vs predicted) for each model
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (name, pred) in enumerate(models_dict.items()):
    ax = axes[idx]
    ax.scatter(y_test_1, pred, alpha=0.5, edgecolors='k', s=60)
    ax.plot([y_test_1.min(), y_test_1.max()], [y_test_1.min(), y_test_1.max()], 'r--', lw=2)
    ax.set_xlabel('Actual cases')
    ax.set_ylabel('Predicted cases')
    ax.set_title(name)
    ax.grid(True)
# Hide extra subplot if any
for j in range(len(models_dict), len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig('scatter_actual_vs_predicted.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Figure 3: Time series for one country (most frequent in test set)
test_df = pd.DataFrame({'country': country_names_test, 'year': year_test, 'actual': y_test_1,
                        'pred_combined': y_pred_combined})
country_counts = test_df['country'].value_counts()
example_country = country_counts.index[0]  # country with most test samples
country_data = test_df[test_df['country'] == example_country].sort_values('year')

plt.figure(figsize=(12, 6))
plt.plot(country_data['year'], country_data['actual'], 'o-', label='Actual cases', linewidth=2, markersize=8)
plt.plot(country_data['year'], country_data['pred_combined'], 's--', label='ST‑AWXE (combined)', linewidth=2, markersize=8)
plt.xlabel('Year')
plt.ylabel('Malaria cases')
plt.title(f'Actual vs Predicted Cases in {example_country} (Test Period)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'time_series_{example_country}.pdf', format='pdf', bbox_inches='tight')
plt.show()

print("\nAll figures saved as PDF files.")