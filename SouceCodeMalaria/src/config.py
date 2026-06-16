import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Load and explore data
# -----------------------------
df = pd.read_csv('malaria_fully_combined_filled.csv')
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First few rows:\n", df.head())

# -----------------------------
# 2. Feature & target selection
# -----------------------------
# We'll predict 'cases' (malaria cases) and 'deaths_rich' (deaths from rich data)
# as two regression tasks.
target_cols = ['cases', 'deaths_rich']

# Drop columns that are not features or are leakage (e.g., lag targets we might use? We keep lags as features)
# We'll keep all numeric columns except target columns and identifier columns.
identifier_cols = ['country', 'year']
exclude_from_features = identifier_cols + target_cols + ['deaths_who', 'high_risk']  # deaths_who is alternative target
feature_cols = [c for c in df.columns if c not in exclude_from_features and np.issubdtype(df[c].dtype, np.number)]

print("Using features:", feature_cols)
print("Targets:", target_cols)

# Prepare data: drop rows where any target is missing (none are missing in this file)
data = df[feature_cols + target_cols + identifier_cols].copy()
data = data.dropna()

# Encode 'country' as categorical feature
le = LabelEncoder()
data['country_encoded'] = le.fit_transform(data['country'])
feature_cols.append('country_encoded')

# Final feature matrix and target matrix
X = data[feature_cols].values
y_cases = data['cases'].values
y_deaths = data['deaths_rich'].values
y = np.column_stack([y_cases, y_deaths])  # shape (n_samples, 2)

# Temporal split: train (year < 2013), val (2013-2017), test (2018-2021)
years = data['year'].values
train_mask = years < 2013
val_mask = (years >= 2013) & (years <= 2017)
test_mask = years >= 2018

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

# Scale features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3. Baseline models: Random Forest & XGBoost (Multi-output)
# -----------------------------
def evaluate_model(name, y_true, y_pred):
    rmse_cases = np.sqrt(mean_squared_error(y_true[:,0], y_pred[:,0]))
    mae_cases = mean_absolute_error(y_true[:,0], y_pred[:,0])
    rmse_deaths = np.sqrt(mean_squared_error(y_true[:,1], y_pred[:,1]))
    mae_deaths = mean_absolute_error(y_true[:,1], y_pred[:,1])
    print(f"{name:20s} | Cases RMSE: {rmse_cases:.2f}, MAE: {mae_cases:.2f} | Deaths RMSE: {rmse_deaths:.2f}, MAE: {mae_deaths:.2f}")
    return rmse_cases, mae_cases, rmse_deaths, mae_deaths

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print("\n=== Random Forest Results ===")
evaluate_model("Random Forest", y_test, y_pred_rf)

# XGBoost (multi-output via separate models or sklearn wrapper)
# Use XGBoost's multi-output regression with objective='reg:squarederror' and num_class=2? Simpler: train two models.
xgb_cases = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
xgb_deaths = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
xgb_cases.fit(X_train_scaled, y_train[:,0])
xgb_deaths.fit(X_train_scaled, y_train[:,1])
y_pred_xgb = np.column_stack([xgb_cases.predict(X_test_scaled), xgb_deaths.predict(X_test_scaled)])
print("\n=== XGBoost Results ===")
evaluate_model("XGBoost", y_test, y_pred_xgb)

# -----------------------------
# 4. ST-WEMTL: Weighted Deep Ensemble Multi-Task Learning
# -----------------------------
# We build an ensemble of K multi-task neural networks.
# Each network: shared layers + two task-specific heads.
# Ensemble weights are learned from validation performance (inverse validation loss weighting).
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim, shared_dims=[128, 64], task_dims=[32, 16]):
        super().__init__()
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            shared_layers.append(nn.Linear(prev_dim, dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(0.2))
            prev_dim = dim
        self.shared = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.head_cases = self._make_head(prev_dim, task_dims, 1)
        self.head_deaths = self._make_head(prev_dim, task_dims, 1)
        
    def _make_head(self, in_dim, hidden_dims, out_dim):
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        shared_out = self.shared(x)
        cases_out = self.head_cases(shared_out).squeeze(1)
        deaths_out = self.head_deaths(shared_out).squeeze(1)
        return cases_out, deaths_out

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_cases_batch, y_deaths_batch in train_loader:
            X_batch = X_batch.to(device)
            y_cases_batch = y_cases_batch.to(device)
            y_deaths_batch = y_deaths_batch.to(device)
            optimizer.zero_grad()
            pred_cases, pred_deaths = model(X_batch)
            loss = criterion(pred_cases, y_cases_batch) + criterion(pred_deaths, y_deaths_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_cases_batch, y_deaths_batch in val_loader:
                X_batch = X_batch.to(device)
                y_cases_batch = y_cases_batch.to(device)
                y_deaths_batch = y_deaths_batch.to(device)
                pred_cases, pred_deaths = model(X_batch)
                loss = criterion(pred_cases, y_cases_batch) + criterion(pred_deaths, y_deaths_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    
    model.load_state_dict(best_model_state)
    return model

# Prepare PyTorch datasets and loaders
batch_size = 64
train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                              torch.tensor(y_train[:,0], dtype=torch.float32),
                              torch.tensor(y_train[:,1], dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                            torch.tensor(y_val[:,0], dtype=torch.float32),
                            torch.tensor(y_val[:,1], dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                             torch.tensor(y_test[:,0], dtype=torch.float32),
                             torch.tensor(y_test[:,1], dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Train ensemble of K models (here K=3)
K = 3
ensemble_models = []
val_losses = []  # to compute weights

print("\n=== Training ST-AWXE  Ensemble ===")
for i in range(K):
    print(f"\nTraining model {i+1}/{K}")
    model = MultiTaskNet(input_dim=X_train_scaled.shape[1])
    model = train_model(model, train_loader, val_loader, epochs=100, lr=0.001)
    ensemble_models.append(model)
    
    # Compute validation loss for weighting (inverse loss weighting)
    model.eval()
    val_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for X_b, yc_b, yd_b in val_loader:
            X_b = X_b.to(device)
            yc_b = yc_b.to(device)
            yd_b = yd_b.to(device)
            pred_c, pred_d = model(X_b)
            loss = criterion(pred_c, yc_b) + criterion(pred_d, yd_b)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

# Compute ensemble weights (inverse validation loss, normalized)
inv_losses = 1.0 / np.array(val_losses)
ensemble_weights = inv_losses / inv_losses.sum()
print(f"Ensemble weights: {ensemble_weights}")

# Predict on test set using weighted ensemble
def ensemble_predict(models, weights, loader):
    all_preds_cases = []
    all_preds_deaths = []
    with torch.no_grad():
        for X_b, _, _ in loader:
            X_b = X_b.to(device)
            pred_cases_ensemble = torch.zeros(X_b.size(0), device=device)
            pred_deaths_ensemble = torch.zeros(X_b.size(0), device=device)
            for w, model in zip(weights, models):
                model.eval()
                pc, pd = model(X_b)
                pred_cases_ensemble += w * pc
                pred_deaths_ensemble += w * pd
            all_preds_cases.append(pred_cases_ensemble.cpu().numpy())
            all_preds_deaths.append(pred_deaths_ensemble.cpu().numpy())
    y_pred_cases = np.concatenate(all_preds_cases)
    y_pred_deaths = np.concatenate(all_preds_deaths)
    return np.column_stack([y_pred_cases, y_pred_deaths])

y_pred_st_wemtl = ensemble_predict(ensemble_models, ensemble_weights, test_loader)
print("\n=== ST-AWXE  Results ===")
evaluate_model("ST-AWXE ", y_test, y_pred_st_wemtl)

# Optional: save models and scaler for later use
import joblib
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(xgb_cases, 'xgb_cases.pkl')
joblib.dump(xgb_deaths, 'xgb_deaths.pkl')
joblib.dump(scaler, 'scaler.pkl')
torch.save([ensemble_models, ensemble_weights], 'st_wemtl_ensemble.pt')
print("\nModels saved to disk.")