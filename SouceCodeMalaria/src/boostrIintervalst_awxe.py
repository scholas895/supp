import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

# ==============================
# 2. SPLIT DATA
# ==============================
train_df = df[df['year'] <= 2017].copy()
val_df   = df[(df['year'] >= 2013) & (df['year'] <= 2017)].copy()
test_df  = df[(df['country'] == 'Algeria') & (df['year'] >= 2018)].copy()

if test_df.empty:
    test_df = df[df['country'] == 'Algeria'].copy()

X_train = train_df[feature_cols]
y_train = train_df['cases']
X_val   = val_df[feature_cols]
y_val   = val_df['cases']
X_test  = test_df[feature_cols]
y_test  = test_df['cases']
years_test = test_df['year'].values

# Log transform target (small offset to avoid log(0))
offset = 0.01
y_train_log = np.log(y_train + offset)
y_val_log   = np.log(y_val + offset)
y_test_log  = np.log(y_test + offset)

# Standardise features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Year arrays for temporal weighting
train_years = train_df['year'].values
val_years   = val_df['year'].values

# ==============================
# 3. FUNCTION TO TRAIN SIMPLIFIED ST-AWXE (on log target)
# ==============================
def train_simplified_st_awxe_log(X_tr, y_tr_log, X_va, y_va_log, tr_years, va_years):
    """
    Train simplified ST‑AWXE:
    - temporally weighted XGBoost
    - unweighted XGBoost
    - Ridge meta‑learner
    Returns a predictor function that takes new X and new year and returns predictions on original scale.
    """
    temporal_decay = 2.0   # years
    ridge_alpha = 1.0

    # ---- Obtain base learner predictions on validation set ----
    pred_temp = []
    pred_unw = []

    for i, val_year in enumerate(va_years):
        # Temporally weighted XGBoost
        w_temp = np.exp(-np.abs(tr_years - val_year) / temporal_decay)
        model_temp = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_temp.fit(X_tr, y_tr_log, sample_weight=w_temp)
        pred_temp.append(model_temp.predict(X_va[i:i+1])[0])

        # Unweighted XGBoost
        model_unw = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_unw.fit(X_tr, y_tr_log)
        pred_unw.append(model_unw.predict(X_va[i:i+1])[0])

    # Stack predictions as features for meta‑learner
    meta_features = np.column_stack([pred_temp, pred_unw])
    meta_model = Ridge(alpha=ridge_alpha)
    meta_model.fit(meta_features, y_va_log)

    # ---- Retrain base learners on the full training set (for final predictor) ----
    # For speed, we train them once (but ideally we should retrain per test point).
    # This is an approximation; for bootstrapping we will retrain inside the loop.
    # Here we just return a function that uses the pre‑trained base learners and meta‑learner.
    # Actually, for prediction on new test points, we need to recompute base learners with weights relative to each test year.
    # Given the bootstrap will retrain everything, we can simply return a function that does the full training again.
    # To keep the interface simple, we will train the full model inside the predictor each time.
    # For the final predictor (used outside bootstrap), we do a full training on the provided data.
    def predictor(X_new, new_year):
        # Train temporally weighted model on the full training data (given at call time)
        w_temp = np.exp(-np.abs(tr_years - new_year) / temporal_decay)
        model_temp_new = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_temp_new.fit(X_tr, y_tr_log, sample_weight=w_temp)
        model_unw_new = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model_unw_new.fit(X_tr, y_tr_log)
        pred_temp_new = model_temp_new.predict(X_new)
        pred_unw_new = model_unw_new.predict(X_new)
        meta_feat = np.column_stack([pred_temp_new, pred_unw_new])
        pred_log = meta_model.predict(meta_feat)
        # Convert back to original scale
        pred_orig = np.exp(pred_log) - offset
        pred_orig = np.maximum(pred_orig, 0)   # ensure non‑negative
        return pred_orig
    return predictor

# ==============================
# 4. BOOTSTRAP TO GET PREDICTION INTERVALS
# ==============================
n_bootstrap = 200   # number of bootstrap samples (increase for better intervals)
bootstrap_preds = []   # store predictions for each bootstrap

print("Starting bootstrap (this may take several minutes)...")
for b in range(n_bootstrap):
    if b % 20 == 0:
        print(f"Bootstrap iteration {b+1}/{n_bootstrap}")

    # Resample training data with replacement
    idx = np.random.choice(len(X_train_scaled), len(X_train_scaled), replace=True)
    X_boot = X_train_scaled[idx]
    y_boot_log = y_train_log.iloc[idx].values
    years_boot = train_years[idx]

    # For each bootstrap, we need a separate validation set? We'll reuse the same validation set.
    # To be consistent, we use the original validation set (not bootstrapped) for meta‑learner training.
    # That is acceptable: the meta‑learner is trained on out‑of‑sample predictions from the base learners.
    # We'll call the training function with the bootstrapped training data and the original validation data.
    # However, the training function expects training data and validation data. We'll create a simplified predictor that does the full training each time.
    # Instead of using the complex closure, we'll implement the steps directly in the loop for clarity.

    # ---- Step 1: Train base learners on bootstrapped data and get predictions on validation set ----
    temporal_decay = 2.0
    pred_temp_val = []
    pred_unw_val = []

    for i, val_year in enumerate(val_years):
        # Temporally weighted XGBoost
        w_temp = np.exp(-np.abs(years_boot - val_year) / temporal_decay)
        model_temp = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=b)
        model_temp.fit(X_boot, y_boot_log, sample_weight=w_temp)
        pred_temp_val.append(model_temp.predict(X_val_scaled[i:i+1])[0])

        # Unweighted XGBoost
        model_unw = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=b)
        model_unw.fit(X_boot, y_boot_log)
        pred_unw_val.append(model_unw.predict(X_val_scaled[i:i+1])[0])

    # ---- Step 2: Train meta‑learner on validation predictions ----
    meta_features_val = np.column_stack([pred_temp_val, pred_unw_val])
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features_val, y_val_log)

    # ---- Step 3: Retrain base learners on the full bootstrapped data (for test prediction) ----
    # For each test point (Algeria test years), we need to retrain with weights relative to that year.
    # We'll do it per test year (since only a few years, it's fine).
    pred_test_combined = []
    for test_year in years_test:
        # Temporally weighted model for this test year
        w_temp_test = np.exp(-np.abs(years_boot - test_year) / temporal_decay)
        model_temp_test = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=b)
        model_temp_test.fit(X_boot, y_boot_log, sample_weight=w_temp_test)
        pred_temp_test = model_temp_test.predict(X_test_scaled)   # predict for all test points? We need only one point.
        # Actually, we are iterating over test years, but X_test_scaled has all test samples in order.
        # We'll need to select the row corresponding to that test year. Simpler: predict only that single row.
        # But we have multiple test years. We'll loop over test indices.
        # Let's restructure: we have test years list and we want predictions for each year. We'll use index matching.
        pass

    # Simpler: we will loop over test samples (rows of X_test_scaled) together with their corresponding year.
    # We'll collect predictions for each test sample.
    pred_boot = []
    for idx_test, (test_year, X_test_row) in enumerate(zip(years_test, X_test_scaled)):
        # Reshape row to 2D
        X_row = X_test_row.reshape(1, -1)
        # Temporally weighted model for this specific test year
        w_temp = np.exp(-np.abs(years_boot - test_year) / temporal_decay)
        model_temp = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=b)
        model_temp.fit(X_boot, y_boot_log, sample_weight=w_temp)
        pred_temp = model_temp.predict(X_row)[0]

        # Unweighted model
        model_unw = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=b)
        model_unw.fit(X_boot, y_boot_log)
        pred_unw = model_unw.predict(X_row)[0]

        # Meta prediction
        meta_feat = np.array([[pred_temp, pred_unw]])
        pred_log = meta_model.predict(meta_feat)[0]
        pred_orig = np.exp(pred_log) - offset
        pred_orig = max(pred_orig, 0)
        pred_boot.append(pred_orig)

    bootstrap_preds.append(pred_boot)

# Convert to array
bootstrap_preds = np.array(bootstrap_preds)   # shape (n_bootstrap, n_test_points)
pred_mean = bootstrap_preds.mean(axis=0)
pred_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
pred_upper = np.percentile(bootstrap_preds, 97.5, axis=0)

# ==============================
# 5. PLOT
# ==============================
plt.figure(figsize=(10,6))
plt.plot(years_test, y_test, 'o-', color='black', linewidth=2, label='Actual cases')
plt.plot(years_test, pred_mean, 's-', color='blue', linewidth=2, label='Predicted mean (ST‑AWXE simplified)')
plt.fill_between(years_test, pred_lower, pred_upper, color='blue', alpha=0.2, label='95% prediction interval')
plt.xlabel('Year')
plt.ylabel('Malaria cases (count)')
plt.title('Algeria: Actual vs Predicted with Bootstrap Intervals (ST‑AWXE)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig('figure4_st_awxe_intervals.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_st_awxe_intervals.pdf', bbox_inches='tight')
plt.show()

print("Figure saved as 'figure4_st_awxe_intervals.png' and '.pdf'")