import pandas as pd
import hashlib
from tqdm import tqdm
from pathlib import Path

tqdm.pandas()

# TEST : Charger seulement les 100 premières lignes
df1 = pd.read_csv('cheatsm/covtype.data', nrows=100)
df2 = pd.read_csv('cheatsm/test-full.csv', nrows=100)

print(f"Test avec {len(df1)} lignes de df1 et {len(df2)} lignes de df2")

df1_ids = df1['Id'].copy()
df1_cover_type = df1['Cover_Type'].copy()
df2_ids = df2['Id'].copy()

df1_cols = [col for col in df1.columns if col not in ['Id', 'Cover_Type']]
df2_cols = [col for col in df2.columns if col not in ['Id']]

df1_clean = df1[df1_cols].copy()
df2_clean = df2[df2_cols].copy()

def create_hash(row):
    return hashlib.md5(row.sort_index().to_string().encode()).hexdigest()

print("Hashing df1...")
df1_clean['hash'] = df1_clean.progress_apply(create_hash, axis=1)

print("Hashing df2...")
df2_clean['hash'] = df2_clean.progress_apply(create_hash, axis=1)

print("Matching...")
hash_set = set(df1_clean['hash'])
df2_clean['Found_in_df1'] = df2_clean['hash'].progress_apply(lambda x: x in hash_set)

# SOLUTION : Créer df1_subset avec hash ET Cover_Type
df1_subset = df1_clean[['hash']].copy()
df1_subset['Cover_Type'] = df1['Cover_Type'].values  # Ajouter Cover_Type

print(f"df1_subset shape: {df1_subset.shape}")
print(f"df1_subset head:\n{df1_subset.head()}")
print(f"Cover_Type values: {df1_subset['Cover_Type'].value_counts()}")

# Merge par hash
df2_with_solution = df2_clean.merge(df1_subset, on='hash', how='left')

df2_with_solution.to_csv('cheatsm/test-full-with-solution.csv', index=False)
print(f"Résultats sauvegardés: {df2_with_solution.shape}")
print(f"Cover_Type dans le résultat: {df2_with_solution['Cover_Type'].value_counts()}")