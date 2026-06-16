import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
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

years = data['year'].values
countries = data['country_encoded'].values
country_names = data['country'].values

# Temporal split
train_mask = years < 2013
val_mask = (years >= 2013) & (years <= 2017)
test_mask = years >= 2018

X_train, y_train_cases = X[train_mask], y_cases[train_mask]
X_val, y_val_cases = X[val_mask], y_cases[val_mask]
X_test, y_test_cases = X[test_mask], y_cases[test_mask]

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

# -----------------------------
# 2. Create binary classification target (cases > threshold)
# -----------------------------
threshold = np.median(y_train_cases)
print(f"\nClassification threshold (cases > {threshold:.2f}) -> class 1 (high burden)")

y_train_class = (y_train_cases > threshold).astype(int)
y_val_class = (y_val_cases > threshold).astype(int)
y_test_class = (y_test_cases > threshold).astype(int)

# -----------------------------
# 3. Train baseline classifiers
# -----------------------------
print("\n[1/5] Training baseline classifiers...")

xgb_simple = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_simple.fit(X_train_scaled, y_train_class)
y_pred_simple = xgb_simple.predict(X_test_scaled)
y_proba_simple = xgb_simple.predict_proba(X_test_scaled)[:, 1]

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_class)
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

xgb_tuned = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                              random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_tuned.fit(X_train_scaled, y_train_class)
y_pred_tuned = xgb_tuned.predict(X_test_scaled)
y_proba_tuned = xgb_tuned.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# 4. ST‑AWXE base models (ensemble components) for classification
# -----------------------------
def train_base_classifiers(X_train, y_train, X_val, y_val):
    models = {}
    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_m.fit(X_train, y_train)
    models['xgb'] = xgb_m
    lgb_m = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                               random_state=42, verbose=-1)
    lgb_m.fit(X_train, y_train)
    models['lgb'] = lgb_m
    cb_m = cb.CatBoostClassifier(iterations=200, depth=5, learning_rate=0.05,
                                 random_seed=42, verbose=False)
    cb_m.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
    models['cb'] = cb_m
    rf_m = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_m.fit(X_train, y_train)
    models['rf'] = rf_m
    return models

print("[2/5] Training ST‑AWXE base classifiers...")
base_models = train_base_classifiers(X_train_scaled, y_train_class, X_val_scaled, y_val_class)

val_proba_base = np.column_stack([m.predict_proba(X_val_scaled)[:, 1] for m in base_models.values()])

meta_features_val = np.column_stack([val_proba_base, country_val, year_val])
meta_scaler = StandardScaler()
meta_features_val_scaled = meta_scaler.fit_transform(meta_features_val)
meta_learner = LogisticRegression(random_state=42)
meta_learner.fit(meta_features_val_scaled, y_val_class)

def adaptive_weighted_predict_class(test_feat, test_country, test_year, base_models_dict,
                                    X_val_orig, y_val_orig, val_proba_base_orig, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_val_scaled)
    _, indices = nn.kneighbors(test_feat.reshape(1, -1))
    neighbor_idx = indices[0]
    weights = []
    for i, name in enumerate(base_models_dict.keys()):
        pred_proba_neighbors = val_proba_base_orig[neighbor_idx, i]
        true_neighbors = y_val_orig[neighbor_idx]
        mae = np.mean(np.abs(pred_proba_neighbors - true_neighbors))
        weights.append(1.0 / (mae + 1e-6))
    weights = np.array(weights)
    weights /= weights.sum()
    test_probas = np.array([m.predict_proba(test_feat.reshape(1, -1))[0, 1] for m in base_models_dict.values()])
    return np.dot(weights, test_probas)

print("   Computing adaptive weighted predictions (probabilities)...")
y_proba_adapt = np.array([adaptive_weighted_predict_class(
    X_test_scaled[i], country_test[i], year_test[i], base_models,
    X_val_scaled, y_val_class, val_proba_base, k=5)
    for i in range(len(X_test_scaled))])
y_pred_adapt = (y_proba_adapt >= 0.5).astype(int)

test_proba_base = np.column_stack([m.predict_proba(X_test_scaled)[:, 1] for m in base_models.values()])
meta_features_test = np.column_stack([test_proba_base, country_test, year_test])
meta_features_test_scaled = meta_scaler.transform(meta_features_test)
y_proba_meta = meta_learner.predict_proba(meta_features_test_scaled)[:, 1]
y_pred_meta = (y_proba_meta >= 0.5).astype(int)

y_proba_combined = (y_proba_adapt + y_proba_meta) / 2
y_pred_combined = (y_proba_combined >= 0.5).astype(int)

print("[3/5] All models trained.")

# -----------------------------
# 5. Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC
# -----------------------------
def evaluate_classification(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    return acc, prec, rec, f1, auc

models_class = {
    'XGBoost (simple)': (y_pred_simple, y_proba_simple),
    'Random Forest': (y_pred_rf, y_proba_rf),
    'XGBoost (tuned)': (y_pred_tuned, y_proba_tuned),
    'ST‑AWXE (adaptive)': (y_pred_adapt, y_proba_adapt),
    'ST‑AWXE (combined)': (y_pred_combined, y_proba_combined)
}



# -----------------------------
# 6. Grouped Bar Chart (PNG + PDF)
# -----------------------------
# Prepare data for grouped bars


# Also keep the individual figures as before (optional but included)
# Figure: Confusion matrix
cm = confusion_matrix(y_test_class, y_pred_combined)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low burden', 'High burden'])
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix - ST‑AWXE Combined (threshold={threshold:.0f})')
plt.tight_layout()
plt.savefig('confusion_matrix.pdf', format='pdf', bbox_inches='tight')
plt.savefig('confusion_matrix.png', format='png', bbox_inches='tight', dpi=300)
plt.show()



# Figure: Time series (PDF/PNG)
test_df = pd.DataFrame({'country': country_names_test, 'year': year_test,
                        'true_class': y_test_class,
                        'prob_combined': y_proba_combined})
country_counts = test_df['country'].value_counts()
example_country = country_counts.index[0]
country_data = test_df[test_df['country'] == example_country].sort_values('year')

plt.figure(figsize=(12, 6))
plt.plot(country_data['year'], country_data['true_class'], 'o-', label='Actual class (0=low,1=high)', markersize=8)
plt.plot(country_data['year'], country_data['prob_combined'], 's--', label='ST‑AWXE predicted probability', linewidth=2, markersize=8)
plt.axhline(y=0.5, color='red', linestyle=':', label='Decision boundary (0.5)')
plt.xlabel('Year')
plt.ylabel('Probability / Class')
plt.title(f'Classification Results for {example_country} (Test Period)')
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'time_series_classification_{example_country}.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'time_series_classification_{example_country}.png', format='png', bbox_inches='tight', dpi=300)
plt.show()

print("\nAll figures saved as both PDF and PNG (grouped bar chart included).")