import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score

def calculate_regression_metrics(y_true, y_pred, digits=4):
    """
    Calculate comprehensive regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    digits : int
        Number of decimal places for rounding (default=4)
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    
    # MAPE (%) - handles zero values by ignoring them
    non_zero = y_true != 0
    if np.sum(non_zero) == 0:
        mape = 100.0
    else:
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    
    # Optionally: SMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-6, denominator)  # avoid division by zero
    smape = np.mean(100 * np.abs(y_pred - y_true) / denominator)
    
    # Maximum error
    max_error = np.max(np.abs(y_true - y_pred))
    
    metrics = {
        'MSE': round(mse, digits),
        'RMSE': round(rmse, digits),
        'MAE': round(mae, digits),
        'MedAE': round(medae, digits),
        'R² (R-squared)': round(r2, digits),
        'Explained Variance': round(ev, digits),
        'MAPE (%)': round(mape, 2),
        'SMAPE (%)': round(smape, 2),
        'Max Error': round(max_error, digits)
    }
    return metrics

def print_metrics_table(metrics_dict, model_name="Model"):
    """
    Print a formatted table of metrics.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary returned by calculate_regression_metrics
    model_name : str
        Name of the model (for display)
    """
    print(f"\n{'='*50}")
    print(f"Evaluation Metrics for {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics_dict.items():
        print(f"{metric:<20}: {value}")
    print(f"{'='*50}")

# =============================================
# Example usage with your test predictions
# =============================================

# Assume you have y_test_1 (ground truth) and predictions from your models
# Example:
# metrics = calculate_regression_metrics(y_test_1, y_pred_simple)
# print_metrics_table(metrics, "XGBoost (simple)")

# To compare multiple models, store results in a DataFrame
import pandas as pd

def compare_models_metrics(y_true, models_dict):
    """
    Compare multiple models and return a DataFrame with all metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth values
    models_dict : dict
        Dictionary with model names as keys and predictions as values
    
    Returns:
    --------
    pd.DataFrame : DataFrame with models as rows and metrics as columns
    """
    all_metrics = []
    for name, y_pred in models_dict.items():
        metrics = calculate_regression_metrics(y_true, y_pred)
        metrics['Model'] = name
        all_metrics.append(metrics)
    
    df = pd.DataFrame(all_metrics)
    # Reorder columns to have Model first
    cols = ['Model'] + [c for c in df.columns if c != 'Model']
    df = df[cols]
    return df

# =============================================
# Integration with your existing script:
# =============================================

# After you have all predictions (y_pred_simple, y_pred_rf, y_pred_tuned, y_pred_adapt, y_pred_combined)
# Create a dictionary of predictions:
# predictions_dict = {
#     'XGBoost (simple)': y_pred_simple,
#     'Random Forest': y_pred_rf,
#     'XGBoost (tuned)': y_pred_tuned,
#     'ST-AWXE (adaptive)': y_pred_adapt,
#     'ST-AWXE (combined)': y_pred_combined
# }
# 
# # Generate comparison table
# comparison_df = compare_models_metrics(y_test_1, predictions_dict)
# print(comparison_df.to_string(index=False))
# 
# # Save to CSV
# comparison_df.to_csv('model_metrics_comparison.csv', index=False)# After your training and prediction code
predictions_dict = {
    'XGBoost (simple)': y_pred_simple,
    'Random Forest': y_pred_rf,
    'XGBoost (tuned)': y_pred_tuned,
    'ST-AWXE (adaptive)': y_pred_adapt,
    'ST-AWXE (combined)': y_pred_combined
}

comparison_df = compare_models_metrics(y_test_1, predictions_dict)
print(comparison_df.to_string(index=False))
comparison_df.to_csv('model_metrics_comparison.csv', index=False)