import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('datasets/mcf_train_70.csv', nrows=50000)

print("=" * 80)
print("CACHE REPLACEMENT HEURISTIC - DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"Decision distribution:\n{df['decision'].value_counts(normalize=True)}\n")

# 1. Type distribution per class
print("ACCESS TYPE distribution per decision class:")
type_names = {0: 'LOAD', 1: 'RFO', 2: 'PREFETCH', 3: 'WRITE', 4: 'TRANSLATION'}
for dec in [0, 1]:
    subset = df[df['decision'] == dec]['type']
    print(f"  Decision={dec}: {subset.value_counts().to_dict()}")
print()

# 2. Page-alignment: does (ip >> 12) == (full_addr >> 12)?
df['ip_int64'] = df['ip'].astype('int64')
df['addr_int64'] = df['full_addr'].astype('int64')
df['page_match'] = (df['ip_int64'] >> 12) == (df['addr_int64'] >> 12)
print("PAGE ALIGNMENT (ip_page == addr_page) by decision:")
page_corr = pd.crosstab(df['page_match'], df['decision'], margins=True)
print(page_corr)
print()

# 3. Low-order bits: last 6 bits of ip and full_addr
df['ip_low6'] = df['ip_int64'] & 0x3F
df['addr_low6'] = df['addr_int64'] & 0x3F
print("IP low 6 bits (mean) by decision:", df.groupby('decision')['ip_low6'].mean().to_dict())
print("ADDR low 6 bits (mean) by decision:", df.groupby('decision')['addr_low6'].mean().to_dict())
print()

# 4. Set and Way distribution
print("SET distribution by decision (first 5 values):")
print(df.groupby('decision')['set'].value_counts().head(10))
print("\nWAY distribution by decision:")
print(pd.crosstab(df['way'], df['decision'], margins=True))
print()

# 5. Hit rate by decision
print("HIT distribution by decision:")
print(pd.crosstab(df['hit'], df['decision'], margins=True))
print()

# 6. CPU distribution
print("CPU distribution by decision:")
print(pd.crosstab(df['triggering_cpu'], df['decision'], margins=True))

print("\n" + "=" * 80)