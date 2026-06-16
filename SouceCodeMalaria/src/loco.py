import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. Load and preprocess data
# ------------------------------------------------------------
df = pd.read_csv('malaria_fully_combined_filled.csv')
target_cols = ['cases', 'deaths_rich']
identifier_cols = ['country', 'year']
exclude_from_features = identifier_cols + target_cols + ['deaths_who', 'high_risk']
feature_cols = [c for c in df.columns if c not in exclude_from_features and np.issubdtype(df[c].dtype, np.number)]

data = df[feature_cols + target_cols + identifier_cols].copy().dropna()

le = LabelEncoder()
data['country_encoded'] = le.fit_transform(data['country'])
feature_cols.append('country_encoded')

X = data[feature_cols].values
y = data['cases'].values
years = data['year'].values
countries = data['country'].values

# ------------------------------------------------------------
# 2. ST‑AWXE combined training function
# ------------------------------------------------------------
def train_st_awxe_combined(X_train, y_train, X_val, y_val, X_test):
    models = {}
    models['xgb'] = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42).fit(X_train, y_train)
    models['lgb'] = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1).fit(X_train, y_train)
    models['cb'] = cb.CatBoostRegressor(iterations=200, depth=5, learning_rate=0.05, random_seed=42, verbose=False).fit(
        X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
    models['rf'] = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)

    val_preds_base = np.column_stack([m.predict(X_val) for m in models.values()])
    meta_scaler = StandardScaler()
    meta_features_val_scaled = meta_scaler.fit_transform(val_preds_base)
    meta_learner = Ridge(alpha=1.0).fit(meta_features_val_scaled, y_val)

    def adaptive_weighted_predict(test_feat, base_models_dict, X_ref, y_ref, val_preds_ref, k=5):
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref)
        _, indices = nn.kneighbors(test_feat.reshape(1, -1))
        neighbor_idx = indices[0]
        weights = []
        for i, name in enumerate(base_models_dict.keys()):
            pred_neighbors = val_preds_ref[neighbor_idx, i]
            true_neighbors = y_ref[neighbor_idx]
            mae = np.mean(np.abs(pred_neighbors - true_neighbors))
            weights.append(1.0 / (mae + 1e-6))
        weights = np.array(weights) / np.sum(weights)
        test_preds = np.array([m.predict(test_feat.reshape(1, -1))[0] for m in base_models_dict.values()])
        return np.dot(weights, test_preds)

    y_pred_adapt = np.array([adaptive_weighted_predict(X_test[i], models, X_val, y_val, val_preds_base, k=5) for i in range(len(X_test))])
    test_preds_base = np.column_stack([m.predict(X_test) for m in models.values()])
    meta_features_test_scaled = meta_scaler.transform(test_preds_base)
    y_pred_meta = meta_learner.predict(meta_features_test_scaled)
    return (y_pred_adapt + y_pred_meta) / 2

# ------------------------------------------------------------
# 3. Leave‑one‑country‑out cross‑validation
# ------------------------------------------------------------
rmse_per_country = {}
unique_countries = data['country'].unique()

print("Running LOCO cross‑validation (may take 20-40 minutes)...")
for i, country in enumerate(unique_countries, 1):
    print(f"  [{i}/{len(unique_countries)}] Processing {country}...")
    country_mask = data['country'] == country

    train_inner_mask = (data['country'] != country) & (years < 2016)
    val_mask = (data['country'] != country) & (years >= 2016) & (years < 2018)
    test_mask = country_mask & (years >= 2018)

    X_train = X[train_inner_mask]
    y_train = y[train_inner_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print(f"    Skipping {country} (insufficient data)")
        continue

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    y_pred = train_st_awxe_combined(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_per_country[country] = rmse
    print(f"    RMSE = {rmse:.2f}")

rmse_df = pd.DataFrame(list(rmse_per_country.items()), columns=['country', 'RMSE'])

# ------------------------------------------------------------
# 4. Load shapefile and map RMSE (PNG + PDF)
# ------------------------------------------------------------
# Try to load from local file first, if not present then download.
shapefile_path = 'ne_110m_admin_0_countries.zip'
try:
    world = gpd.read_file(shapefile_path)
except:
    print("Local shapefile not found, downloading from Natural Earth...")
    world = gpd.read_file('https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip')
    world.to_file(shapefile_path, driver='ESRI Shapefile')  # save for next time

# Filter Africa (column name is 'CONTINENT' uppercase)
africa = world[world['CONTINENT'] == 'Africa'].copy()

# Find the correct country name column
possible_names = ['NAME', 'ADMIN', 'name_long', 'admin', 'name']
country_col = None
for col in possible_names:
    if col in africa.columns:
        country_col = col
        break
if country_col is None:
    raise KeyError("No recognised country name column found in shapefile. Columns: " + str(africa.columns.tolist()))

print(f"Using column '{country_col}' for merging.")

# Fix country name mismatches between your CSV and shapefile
name_mapping = {
    'Democratic Republic of the Congo': 'Dem. Rep. Congo',
    'Cabo Verde': 'Cape Verde',
    'Central African Republic': 'Central African Rep.',
    'Equatorial Guinea': 'Eq. Guinea',
    'Eswatini': 'Swaziland',
    'Sao Tome and Principe': 'São Tomé and Principe',
    'South Sudan': 'S. Sudan',
    'Tanzania': 'United Republic of Tanzania',
    'Ivory Coast': "Côte d'Ivoire",
}
rmse_df['country_std'] = rmse_df['country'].replace(name_mapping)

# Merge
africa_rmse = africa.merge(rmse_df, left_on=country_col, right_on='country_std', how='left')

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
africa_rmse.plot(column='RMSE', ax=ax, legend=True,
                 cmap='RdYlBu_r', edgecolor='black', linewidth=0.3,
                 missing_kwds={'color': 'lightgrey', 'label': 'No data'},
                 legend_kwds={'label': 'RMSE (cases)', 'shrink': 0.6})
ax.set_title('Leave‑One‑Country‑Out RMSE for ST‑AWXE (combined)\n(Test period: 2018–2021)', fontsize=14)
ax.set_axis_off()

# Save as PNG and PDF
plt.savefig('loco_map.png', dpi=300, bbox_inches='tight')
plt.savefig('loco_map.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Maps saved as 'loco_map.png' and 'loco_map.pdf'")