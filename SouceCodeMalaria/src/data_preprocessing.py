import pandas as pd
import numpy as np

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv('malaria_fully_combined_filled.csv')

# -----------------------------
# 2. Select numeric columns, but exclude 'year'
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove 'year' if present
if 'year' in numeric_cols:
    numeric_cols.remove('year')

# -----------------------------
# 3. Compute basic descriptive statistics (no percentiles)
# -----------------------------
desc_stats = df[numeric_cols].describe(percentiles=[]).round(1)
# Keep only: count, mean, std, min, max
desc_stats = desc_stats.loc[['count', 'mean', 'std', 'min', 'max']]

# Transpose: variables become rows, statistics become columns
desc_stats_transposed = desc_stats.T

# Convert 'count' to integer
desc_stats_transposed['count'] = desc_stats_transposed['count'].astype(int)

# -----------------------------
# 4. Save as CSV
# -----------------------------
desc_stats_transposed.to_csv('descriptive_statistics_simple.csv')
print("Simplified descriptive statistics saved to 'descriptive_statistics_simple.csv'")

# -----------------------------
# 5. Print dataset description (for your document)
# -----------------------------
total_rows = len(df)
unique_countries = df['country'].nunique()
year_min = df['year'].min()
year_max = df['year'].max()
high_risk_pct = df['high_risk'].mean() * 100

print("\n" + "="*60)
print("DATASET DESCRIPTION")
print("="*60)
print(f"• Number of records: {total_rows}")
print(f"• Time span: {year_min} – {year_max}")
print(f"• Number of African countries: {unique_countries}")
print(f"• Percentage of high‑risk records (high_risk=1): {high_risk_pct:.1f}%")
print(f"• Missing values: {df.isnull().sum().sum()} (dataset is complete after pre‑filling)")
print("\nKey variables: cases, deaths_rich (targets), itn_coverage (intervention),")
print("incidence_rate, mortality_rate, infection_prevalence, plus multiple lag features")
print("(lag1, lag2) and a 3‑year moving average (incidence_ma3).")
print("The dataset is well‑suited for spatio‑temporal forecasting and multi‑task learning.\n")

# Optional: display the first few rows of the simplified table
print("Preview of simplified transposed statistics (first 10 variables):\n")
print(desc_stats_transposed.head(10))



================================================================================================
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('malaria_fully_combined_filled.csv')

# Filter for 2021 test data (note: your test set is years >= 2018)
test_2021 = df[df['year'] == 2021].copy()

# Print basic info
print("=" * 50)
print("2021 TEST DATA INSPECTION")
print("=" * 50)
print(f"\nNumber of country‑year observations in 2021: {len(test_2021)}")
print(f"Countries represented: {sorted(test_2021['country'].unique())}\n")

# Value counts for cases
print("Distribution of 'cases' values in 2021:")
print(test_2021['cases'].value_counts().sort_index())

# Summary statistics
print("\nSummary statistics for 'cases' in 2021:")
print(test_2021['cases'].describe())

# Check if all cases are zero or near‑zero
zero_count = (test_2021['cases'] == 0).sum()
non_zero_count = (test_2021['cases'] > 0).sum()
print(f"\nZero cases: {zero_count} observations ({zero_count/len(test_2021)*100:.1f}%)")
print(f"Non‑zero cases: {non_zero_count} observations ({non_zero_count/len(test_2021)*100:.1f}%)")

# If there are non‑zero cases, show them
if non_zero_count > 0:
    print("\nNon‑zero case observations in 2021:")
    non_zero = test_2021[test_2021['cases'] > 0][['country', 'cases']]
    print(non_zero.to_string(index=False))

# Check for potential data leakage: 
# Verify that 2021 data were not used in training (training should be <= 2017)
train_years = df[df['year'] <= 2017]['year'].unique()
print(f"\nTraining years: {sorted(train_years)[:5]}... to {sorted(train_years)[-1]}")
print(f"2021 in training? {2021 in train_years}")

# Also check if any feature in 2021 contains implausible values (e.g., future lags)
print("\nFirst few rows of 2021 data (selected columns):")
selected_cols = ['country', 'year', 'cases', 'cases_lag1', 'cases_lag2', 'itn_coverage', 'high_risk']
existing_cols = [col for col in selected_cols if col in test_2021.columns]
print(test_2021[existing_cols].head(10))