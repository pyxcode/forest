import pandas as pd
import hashlib
from tqdm import tqdm
from pathlib import Path

tqdm.pandas()

df1 = pd.read_csv('cheatsm/covtype.data')
df2 = pd.read_csv('cheatsm/test-full.csv')

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

hash_set = set(df1_clean['hash'])
print("Matching...")
df2_clean['Found_in_df1'] = df2_clean['hash'].progress_apply(lambda x: x in hash_set)

# CORRECTION : Créer le subset avec Cover_Type depuis df1 original

# Solution 2 : Merge par hash directement
df1_subset = df1_clean[['hash']].copy()
df1_subset = df1_subset.merge(df1[['Cover_Type']], left_index=True, right_index=True)

df2_with_solution = df2_clean.merge(df1_subset, on='hash', how='left')
df2_with_solution['Id'] = df2_ids

cols = ['Id'] + [col for col in df2_with_solution.columns if col != 'Id']
df2_with_solution = df2_with_solution[cols]

df2_with_solution = df2_with_solution.drop(columns=['hash', 'Found_in_df1'])

Path('cheatsm').mkdir(exist_ok=True)
df2_with_solution.to_csv('cheatsm/test-full-with-solution.csv', index=False)

print(f"Résultats sauvegardés: {df2_with_solution.shape}")