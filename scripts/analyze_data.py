import pandas as pd
import numpy as np
import os

dataset = "mcf"
df = pd.read_csv(f'datasets/{dataset}_50M_labeled.csv')

# Convert decision column to numeric labels
df['decision'] = df['decision'].map({'Cached': 1, 'Not Cached': 0})

# Look at the data distribution
target = 'decision'
counts = df[target].value_counts(normalize=True) * 100
print(f"\n--- Class Distribution ---")
print(f"Cache-Averse (0): {counts.get(0, 0):.2f}%")
print(f"Cache-Friendly (1): {counts.get(1, 0):.2f}%\n")

# Train/Test Split (First 70% Train, Last 30% Test)
split_idx = int(len(df) * 0.70)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"--- Dataset Split ---")
print(f"Train Set: {len(train_df)} rows (0 to {split_idx})")
print(f"Test Set:  {len(test_df)} rows ({split_idx} to {len(df)})\n")

# Save the splits so the LLM pipeline can use them later
train_df.to_csv(f'datasets/{dataset}_train_70.csv', index=False)
test_df.to_csv(f'datasets/{dataset}_test_30.csv', index=False)
print("Saved Train/Test CSVs!\n")

# Correlation (R^2) Analysis on Training Data
print("--- Statistical Analysis (Feature Importance) ---")
# Keep only numeric feature columns
features = train_df.select_dtypes(include=[np.number]).columns.drop(target)

for feature in features:
    # Calculate Pearson Correlation (r)
    corr = train_df[feature].corr(train_df[target])
    # Calculate Variance Explained (R^2)
    r2 = corr ** 2
    print(f"Feature: {feature:15} | r = {corr:>7.4f} | R^2 = {r2:>7.4f}")

print("\nInsight: Features with higher R^2 explain more of the 'Cache-Friendly' variance and should be highlighted in the LLM Prompt.")
