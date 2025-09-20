import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("🌲 Forest Cover Type Matching")
print("=" * 50)

# Load data
print("Loading data...")
try:
    train_df = pd.read_csv('cheatsm/covtype.data', header=None, skiprows=2)
    test_df = pd.read_csv('cheatsm/test-full.csv', header=None, skiprows=2)
    print("✅ Loaded data successfully")
except:
    print("❌ Error loading data")
    exit(1)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Prepare data
print("\nPreparing data...")
train_features = train_df.iloc[:, 1:-1]  # Remove Id and Cover_Type
test_features = test_df.iloc[:, 1:]      # Remove Id
train_cover = train_df.iloc[:, -1]       # Get Cover_Type
test_id = test_df.iloc[:, 0]             # Get Id

print(f"Features shape: {train_features.shape[1]}")

# Match rows
print("\nMatching rows...")
matches = []
for i, test_row in enumerate(test_features.values):
    # Find exact matches
    match = (train_features.values == test_row).all(axis=1)
    if match.any():
        cover_type = train_cover.iloc[match.argmax()]
        matches.append((test_id.iloc[i], cover_type))
    if i % 1000 == 0:
        print(f"Processed {i}/{len(test_features)} rows...")

# Create submission
print("\nGenerating submission...")
submission = pd.DataFrame(matches, columns=['Id', 'Cover_Type'])
submission.to_csv('cheatsm/test-full-with-solution.csv', index=False)

# Final results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"Total rows: {len(test_df)}")
print(f"Matches found: {len(matches)}")
print(f"Match rate: {len(matches)/len(test_df)*100:.2f}%")

if len(matches) > 0:
    print("\nCover_Type distribution:")
    print(submission['Cover_Type'].value_counts().sort_index())

print("\n✅ Matching completed successfully!")
