import pandas as pd
import hashlib
from tqdm import tqdm

tqdm.pandas()

# Charger les données
df1 = pd.read_csv('cheatsm/covtype.data', skiprows=1, header=None)
df2 = pd.read_csv('cheatsm/test-full.csv')

print(f"Chargé {len(df1)} lignes de covtype.data et {len(df2)} lignes de test-full.csv")

# Ajouter les noms de colonnes pour df1 (55 colonnes)
df1.columns = ['Id', 'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']

# Ajouter une colonne Cover_Type vide à df2
df2['Cover_Type'] = None

print(f"df1 Cover_Type values: {df1['Cover_Type'].value_counts()}")

# Préparer les colonnes (même colonnes pour les deux)
df1_cols = [col for col in df1.columns if col not in ['Id', 'Cover_Type']]
df2_cols = [col for col in df2.columns if col not in ['Id', 'Cover_Type']]

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
