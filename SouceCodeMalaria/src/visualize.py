import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. Load dataset
# ==============================
df = pd.read_csv('malaria_fully_combined_filled.csv')
df = df.dropna(subset=['cases'])

# Feature columns (all numeric except country, year, cases)
feature_cols = [col for col in df.columns if col not in ['country', 'year', 'cases']]
feature_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64']]

# ==============================
# 2. Split data
# ==============================
train_df = df[df['year'] <= 2017].copy()
test_df = df[df['year'] >= 2018].copy()

X_train = train_df[feature_cols]
y_train = train_df['cases']
X_test = test_df[feature_cols]
y_test = test_df['cases']
test_years = test_df['year'].values
test_countries = test_df['country'].values

# Standardise
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 3. Helper functions for ST‑AWXE (simplified)
# ==============================
def st_awxe_adaptive(X_train, y_train, X_test, train_years, test_years):
    # Use validation set 2013-2017
    val_mask = (train_df['year'] >= 2013) & (train_df['year'] <= 2017)
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    val_years = train_df['year'][val_mask].values

    temporal_decay = 2.0
    pred_temp_val, pred_unw_val = [], []
    for i, v_year in enumerate(val_years):
        w = np.exp(-np.abs(train_years - v_year) / temporal_decay)
        model_temp = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_temp.fit(X_train, y_train, sample_weight=w)
        pred_temp_val.append(model_temp.predict(X_val[i:i+1])[0])
        model_unw = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_unw.fit(X_train, y_train)
        pred_unw_val.append(model_unw.predict(X_val[i:i+1])[0])

    meta_features_val = np.column_stack([pred_temp_val, pred_unw_val])
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features_val, y_val)

    pred_temp_test, pred_unw_test = [], []
    for i, t_year in enumerate(test_years):
        w = np.exp(-np.abs(train_years - t_year) / temporal_decay)
        model_temp = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_temp.fit(X_train, y_train, sample_weight=w)
        pred_temp_test.append(model_temp.predict(X_test[i:i+1])[0])
        model_unw = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_unw.fit(X_train, y_train)
        pred_unw_test.append(model_unw.predict(X_test[i:i+1])[0])

    meta_features_test = np.column_stack([pred_temp_test, pred_unw_test])
    return meta_model.predict(meta_features_test)

def st_awxe_combined(X_train, y_train, X_test):
    val_mask = (train_df['year'] >= 2013) & (train_df['year'] <= 2017)
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]

    base_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    base_model.fit(X_train, y_train)
    pred_val = base_model.predict(X_val)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(pred_val.reshape(-1, 1), y_val)

    pred_test = base_model.predict(X_test)
    return meta_model.predict(pred_test.reshape(-1, 1))

# ==============================
# 4. Train models and get predictions
# ==============================
train_years = train_df['year'].values
predictions = {}

# XGBoost simple
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.3, random_state=42)
model.fit(X_train_scaled, y_train)
predictions['XGBoost simple'] = model.predict(X_test_scaled)

# XGBoost tuned
model = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=1.0, reg_lambda=1.0, random_state=42)
model.fit(X_train_scaled, y_train)
predictions['XGBoost tuned'] = model.predict(X_test_scaled)

# Random Forest
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train_scaled, y_train)
predictions['Random Forest'] = model.predict(X_test_scaled)

# ST‑AWXE adaptive
print("Training ST‑AWXE adaptive...")
pred_adaptive = st_awxe_adaptive(X_train_scaled, y_train.values, X_test_scaled, train_years, test_years)
predictions['ST-AWXE adaptive'] = pred_adaptive

# ST‑AWXE combined
print("Training ST‑AWXE combined...")
pred_combined = st_awxe_combined(X_train_scaled, y_train.values, X_test_scaled)
predictions['ST-AWXE combined'] = pred_combined

# ==============================
# 5. Compute year‑wise MAE for each model
# ==============================
# Create DataFrame with year and actual values
results = pd.DataFrame({'year': test_years, 'actual': y_test.values})

for name, pred in predictions.items():
    results[name] = pred

# Compute MAE per year for each model
yearly_mae = results.groupby('year').apply(
    lambda g: pd.Series({
        'actual_MAE': mean_absolute_error(g['actual'], g['actual']),  # reference (zero)
        **{name: mean_absolute_error(g['actual'], g[name]) for name in predictions.keys()}
    })
).reset_index()

print("\nYear‑wise MAE (cases):")
print(yearly_mae.round(2))

# Save to CSV
yearly_mae.to_csv('yearly_mae_all_models.csv', index=False)

# ==============================
# 6. Plot year‑wise MAE for all models
# ==============================
plt.figure(figsize=(10,6))
for name in predictions.keys():
    plt.plot(yearly_mae['year'], yearly_mae[name], marker='o', label=name)

plt.xlabel('Year')
plt.ylabel('Mean Absolute Error (cases)')
plt.title('Year‑wise MAE across all models (2018–2021)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig('yearly_mae_plot.png', dpi=300)
plt.show()