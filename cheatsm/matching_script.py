import pandas as pd
import hashlib
from tqdm import tqdm

tqdm.pandas()

print("=== MATCHING PARFAIT DE LIGNES ===")
print("Objectif: Trouver les correspondances exactes entre covtype.data et test-full.csv")
print()

# 1. Charger les données
print("1. Chargement des données...")
df_train = pd.read_csv('cheatsm/covtype.data', header=None, skiprows=2)  # Skip le header multi-ligne
df_test = pd.read_csv('cheatsm/test-full.csv', header=None, skiprows=2)  # Skip le header multi-ligne

# 2. Masquer la colonne ID (première colonne)
train_without_id = df_train.iloc[:, 1:]  # Toutes les colonnes sauf la première
test_without_id = df_test.iloc[:, 1:]    # Toutes les colonnes sauf la première

print("\nPremière ligne train (sans Id):")
print(train_without_id.iloc[0])
print("\nPremière ligne test (sans Id):")
print(test_without_id.iloc[0])
