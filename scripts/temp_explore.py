import pandas as pd
import numpy as np

# Load dataset with row limit for speed
df = pd.read_csv('datasets/mcf_train_70.csv', nrows=50000)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"\nDecision distribution:\n{df['decision'].value_counts()}")
print(f"Decision proportions:\n{df['decision'].value_counts(normalize=True)}\n")

# Split by decision class
df_keep = df[df['decision'] == 1]
df_evict = df[df['decision'] == 0]

print("=" * 70)
print("ACCESS TYPE CORRELATION (0=LOAD, 1=RFO, 2=PREFETCH, 3=WRITE, 4=TRANSLATION)")
print("=" * 70)
print(f"Keep (decision=1):\n{df_keep['type'].value_counts(normalize=True)}")
print(f"\nEvict (decision=0):\n{df_evict['type'].value_counts(normalize=True)}\n")

# Page alignment: check if (ip >> 12) == (full_addr >> 12)
df['page_aligned'] = (df['ip'] >> 12) == (df['full_addr'] >> 12)
print("=" * 70)
print("PAGE ALIGNMENT ANALYSIS")
print("=" * 70)
print(f"Keep (decision=1): {df_keep['page_aligned'].value_counts(normalize=True).to_dict()}")
print(f"Evict (decision=0): {df_evict['page_aligned'].value_counts(normalize=True).to_dict()}\n")

# Low-order bits patterns
print("=" * 70)
print("LOW-ORDER BITS ANALYSIS")
print("=" * 70)
df['ip_low6'] = df['ip'] & 0x3F
df['addr_low6'] = df['full_addr'] & 0x3F
print(f"IP low 6 bits (Keep):\n{df_keep['ip_low6'].value_counts().head(5)}")
print(f"\nIP low 6 bits (Evict):\n{df_evict['ip_low6'].value_counts().head(5)}")
print(f"\nAddr low 6 bits (Keep):\n{df_keep['addr_low6'].value_counts().head(5)}")
print(f"\nAddr low 6 bits (Evict):\n{df_evict['addr_low6'].value_counts().head(5)}\n")

# Set and way distribution
print("=" * 70)
print("SET & WAY DISTRIBUTION")
print("=" * 70)
print(f"Set cardinality (Keep): {df_keep['set'].nunique()}, (Evict): {df_evict['set'].nunique()}")
print(f"Way cardinality (Keep): {df_keep['way'].nunique()}, (Evict): {df_evict['way'].nunique()}")
print(f"Way distribution (Keep):\n{df_keep['way'].value_counts(normalize=True)}")
print(f"Way distribution (Evict):\n{df_evict['way'].value_counts(normalize=True)}\n")

# Hit ratio by decision
print("=" * 70)
print("HIT BEHAVIOR")
print("=" * 70)
print(f"Hit rate (Keep): {df_keep['hit'].mean():.3f}")
print(f"Hit rate (Evict): {df_evict['hit'].mean():.3f}\n")