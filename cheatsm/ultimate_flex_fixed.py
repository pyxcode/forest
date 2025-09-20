import pandas as pd
import hashlib
from tqdm import tqdm
from pathlib import Path

tqdm.pandas()

# Charger les données
df1 = pd.read_csv('Dataset/train.csv')
df2 = pd.read_csv('Dataset/test-full.csv')

print(f"Chargé {len(df1)} lignes de train et {len(df2)} lignes de test")

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

# Créer df1_subset avec hash ET Cover_Type
df1_subset = df1_clean[['hash']].copy()
df1_subset['Cover_Type'] = df1['Cover_Type'].values

# Merge par hash
df2_with_solution = df2_clean.merge(df1_subset, on='hash', how='left')

# Réintégrer les Id
df2_with_solution['Id'] = df2['Id'].values

# Sauvegarder
df2_with_solution.to_csv('cheatsm/test-full-with-solution.csv', index=False)

print(f"Résultats sauvegardés: {df2_with_solution.shape}")
print(f"Matches trouvés: {df2_with_solution['Found_in_df1'].sum()}")
print(f"Cover_Type non-null: {df2_with_solution['Cover_Type'].notna().sum()}")
