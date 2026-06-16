import pandas as pd
df = pd.read_csv('malaria_fully_combined_filled.csv')
print(len(df))                # total rows
print(df.dropna().shape[0])   # rows with no missing values

print(df.shape[1])   # number of columns
# or
print(len(df.columns))


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. LOAD DATASET
# ==============================
df = pd.read_csv('malaria_fully_combined_filled.csv')
df = df.dropna(subset=['cases'])

# Feature columns (all numeric except country, year, cases)
feature_cols = [col for col in df.columns if col not in ['country', 'year', 'cases']]
feature_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']]

# Log transform target (small offset to avoid log(0))
offset = 0.01
df['cases_log'] = np.log(df['cases'] + offset)

# Keep only countries with at least some data
country_counts = df['country'].value_counts()
valid_countries = country_counts[country_counts >= 2].index.tolist()
df = df[df['country'].isin(valid_countries)]

# Split into training (years ≤ 2017) and test (years ≥ 2018)
train_df = df[df['year'] <= 2017].copy()
test_df = df[df['year'] >= 2018].copy()

# ==============================
# 2. FUNCTION: TRAIN SIMPLIFIED ST-AWXE
# ==============================
def train_st_awxe_simple(X_tr, y_tr_log, X_va, y_va_log, tr_years, va_years):
    """
    Train simplified ST-AWXE (temporal weighting + unweighted XGBoost + Ridge meta‑learner).
    Returns a predictor function that expects X_new and new_year.
    """
    temporal_decay = 2.0
    ridge_alpha = 1.0

    # Get validation predictions from base learners
    pred_temp_val = []
    pred_unw_val = []

    for i, val_year in enumerate(va_years):
        # Temporally weighted XGBoost
        w_temp = np.exp(-np.abs(tr_years - val_year) / temporal_decay)
        model_temp = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_temp.fit(X_tr, y_tr_log, sample_weight=w_temp)
        pred_temp_val.append(model_temp.predict(X_va[i:i+1])[0])

        # Unweighted XGBoost
        model_unw = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_unw.fit(X_tr, y_tr_log)
        pred_unw_val.append(model_unw.predict(X_va[i:i+1])[0])

    # Train meta‑learner
    meta_features_val = np.column_stack([pred_temp_val, pred_unw_val])
    meta_model = Ridge(alpha=ridge_alpha)
    meta_model.fit(meta_features_val, y_va_log)

    # Retrain base learners on full training data (for final predictor)
    # Note: For each new test point, we should ideally retrain with temporal weights,
    # but to keep LOCO feasible we retrain once on the whole training set and use a fixed meta‑learner.
    # This is an approximation.
    model_temp_full = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    # Use dummy weights (mean temporal weight) – better than nothing
    model_temp_full.fit(X_tr, y_tr_log, sample_weight=np.exp(-np.abs(tr_years - tr_years.mean()) / temporal_decay))
    model_unw_full = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model_unw_full.fit(X_tr, y_tr_log)

    def predictor(X_new, new_year):
        # For a single test point, recompute temporal weights using the training years
        # (this is still an approximation; we omit per‑point retraining for speed)
        w_temp = np.exp(-np.abs(tr_years - new_year) / temporal_decay)
        model_temp_point = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_temp_point.fit(X_tr, y_tr_log, sample_weight=w_temp)
        model_unw_point = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_unw_point.fit(X_tr, y_tr_log)
        pred_temp = model_temp_point.predict(X_new)
        pred_unw = model_unw_point.predict(X_new)
        meta_feat = np.column_stack([pred_temp, pred_unw])
        pred_log = meta_model.predict(meta_feat)
        return np.exp(pred_log) - offset
    return predictor

# ==============================
# 3. COUNTRY‑WISE LOCO (RMSE per country)
# ==============================
all_countries = test_df['country'].unique()
loco_rmse = []
loco_mae = []

print("Running LOCO cross‑validation...")
for held_out in all_countries:
    print(f"Held‑out: {held_out}")
    # Training: all countries except held_out, years ≤ 2017
    train_loco = df[(df['country'] != held_out) & (df['year'] <= 2017)].copy()
    # Validation: same as training (we use a subset for meta‑learner? We'll use the whole training set for simplicity)
    # But to keep consistency, we split training into train/val internally? For LOCO we need a validation set for meta‑learner.
    # We'll use a fixed validation set (years 2013-2017 from training countries)
    val_loco = train_loco[(train_loco['year'] >= 2013) & (train_loco['year'] <= 2017)].copy()
    train_loco = train_loco[train_loco['year'] <= 2012].copy()  # use earlier years for training base learners

    if len(train_loco) < 50 or len(val_loco) < 10:
        loco_rmse.append(np.nan)
        loco_mae.append(np.nan)
        continue

    # Features and target
    X_train_loco = train_loco[feature_cols]
    y_train_loco_log = train_loco['cases_log'].values
    X_val_loco = val_loco[feature_cols]
    y_val_loco_log = val_loco['cases_log'].values
    train_years_loco = train_loco['year'].values
    val_years_loco = val_loco['year'].values

    # Standardise
    scaler = StandardScaler()
    X_train_loco_scaled = scaler.fit_transform(X_train_loco)
    X_val_loco_scaled = scaler.transform(X_val_loco)

    # Train model
    predictor = train_st_awxe_simple(X_train_loco_scaled, y_train_loco_log,
                                     X_val_loco_scaled, y_val_loco_log,
                                     train_years_loco, val_years_loco)

    # Test on held‑out country, years 2018–2021
    test_loco = test_df[test_df['country'] == held_out].copy()
    if len(test_loco) == 0:
        loco_rmse.append(np.nan)
        loco_mae.append(np.nan)
        continue

    X_test_loco = test_loco[feature_cols]
    y_test_true = test_loco['cases'].values
    y_test_pred = []
    for idx, row in test_loco.iterrows():
        X_row = X_test_loco.loc[[idx]].values
        # Need to scale using the same scaler
        X_row_scaled = scaler.transform(X_row)
        pred = predictor(X_row_scaled, row['year'])
        y_test_pred.append(pred)

    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mae = mean_absolute_error(y_test_true, y_test_pred)
    loco_rmse.append(rmse)
    loco_mae.append(mae)

# Create country‑wise error DataFrame
loco_df = pd.DataFrame({
    'country': all_countries,
    'RMSE': loco_rmse,
    'MAE': loco_mae
}).dropna()

# Save to CSV for supplementary table
loco_df.to_csv('loco_errors_by_country.csv', index=False)

# Plot country‑wise RMSE (bar chart)
plt.figure(figsize=(12,6))
plt.bar(loco_df['country'], loco_df['RMSE'], color='skyblue')
plt.xticks(rotation=90)
plt.ylabel('RMSE (cases)')
plt.title('Leave‑One‑Country‑Out RMSE per Country')
plt.tight_layout()
plt.savefig('loco_rmse_by_country.png', dpi=300)
plt.savefig('loco_rmse_by_country.pdf')
plt.show()

# ==============================
# 4. YEAR‑WISE MAE ACROSS ALL COUNTRIES (2018–2021)
# ==============================
# We need predictions for all test countries and years.
# We'll train a single model on the full training set (all countries, years ≤2017)
train_all = df[df['year'] <= 2017].copy()
val_all = train_all[(train_all['year'] >= 2013) & (train_all['year'] <= 2017)].copy()
train_all_base = train_all[train_all['year'] <= 2012].copy()

X_train_all = train_all_base[feature_cols]
y_train_all_log = train_all_base['cases_log'].values
X_val_all = val_all[feature_cols]
y_val_all_log = val_all['cases_log'].values
train_years_all = train_all_base['year'].values
val_years_all = val_all['year'].values

# Standardise
scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_val_all_scaled = scaler_all.transform(X_val_all)

predictor_all = train_st_awxe_simple(X_train_all_scaled, y_train_all_log,
                                     X_val_all_scaled, y_val_all_log,
                                     train_years_all, val_years_all)

# Predict for all test samples (2018–2021)
test_all = df[df['year'] >= 2018].copy()
X_test_all = test_all[feature_cols]
y_test_true_all = test_all['cases'].values
y_test_pred_all = []
years_list = test_all['year'].values

for idx, row in test_all.iterrows():
    X_row = X_test_all.loc[[idx]].values
    X_row_scaled = scaler_all.transform(X_row)
    pred = predictor_all(X_row_scaled, row['year'])
    y_test_pred_all.append(pred)

# Compute MAE per year
test_all['predicted'] = y_test_pred_all
yearly_mae = test_all.groupby('year').apply(lambda g: mean_absolute_error(g['cases'], g['predicted'])).reset_index()
yearly_mae.columns = ['Year', 'MAE']

# Save as supplementary table
yearly_mae.to_csv('yearly_mae_table.csv', index=False)

# Plot year‑wise MAE
plt.figure(figsize=(8,5))
plt.plot(yearly_mae['Year'], yearly_mae['MAE'], marker='o', linestyle='-', color='red')
plt.xlabel('Year')
plt.ylabel('Mean Absolute Error (cases)')
plt.title('Year‑wise MAE Across All Countries (2018–2021)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig('yearly_mae_plot.png', dpi=300)
plt.savefig('yearly_mae_plot.pdf')
plt.show()

print("Country‑wise LOCO results saved to 'loco_errors_by_country.csv'")
print("Year‑wise MAE table saved to 'yearly_mae_table.csv'")
print("Plots saved as PNG and PDF.")