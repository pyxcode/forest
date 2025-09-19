import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from catboost import CatBoostClassifier
from time import time
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("🌲 Forest Cover Type Classification Pipeline (CatBoost)")
print("=" * 60)

# Load data
print("Loading data...")
try:
    # Try Kaggle input path first
    train_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
    test_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
    print("✅ Loaded from Kaggle input path")
except:
    # Fallback to current directory
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test-full.csv')
    print("✅ Loaded from current directory")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ELU codes mapping for soil types
ELU_codes = {
    1:2702, 2:2703, 3:2704, 4:2705, 5:2706, 6:2717, 7:3501, 8:3502, 9:4201, 10:4703,
    11:4704, 12:4744, 13:4758, 14:5101, 15:5151, 16:6101, 17:6102, 18:6731, 19:7101, 20:7102,
    21:7103, 22:7201, 23:7202, 24:7700, 25:7701, 26:7702, 27:7709, 28:7710, 29:7745, 30:7746,
    31:7755, 32:7756, 33:7757, 34:7790, 35:8703, 36:8707, 37:8708, 38:8771, 39:8772, 40:8776
}

def make_features(df, fit=False, state=None):
    """
    Feature engineering function that applies all required transformations
    """
    df_fe = df.copy()
    
    # Aspect as circular
    df_fe['Aspect_Sin'] = np.sin(df_fe['Aspect'] * np.pi / 180)
    df_fe['Aspect_Cos'] = np.cos(df_fe['Aspect'] * np.pi / 180)
    
    # Hydrology distance (euclidean)
    df_fe['Hydro_Dist_Euclid'] = np.sqrt(
        df_fe['Horizontal_Distance_To_Hydrology']**2 + 
        df_fe['Vertical_Distance_To_Hydrology']**2
    )
    
    # Hillshade stats
    hillshade_cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    df_fe['Hillshade_Mean'] = df_fe[hillshade_cols].mean(axis=1)
    df_fe['Hillshade_Min'] = df_fe[hillshade_cols].min(axis=1)
    df_fe['Hillshade_Max'] = df_fe[hillshade_cols].max(axis=1)
    df_fe['Hillshade_Range'] = df_fe['Hillshade_Max'] - df_fe['Hillshade_Min']
    
    # Distance interactions & contrasts
    df_fe['Road_vs_Hydro'] = df_fe['Horizontal_Distance_To_Roadways'] - df_fe['Horizontal_Distance_To_Hydrology']
    df_fe['Fire_vs_Hydro'] = df_fe['Horizontal_Distance_To_Fire_Points'] - df_fe['Horizontal_Distance_To_Hydrology']
    df_fe['Road_vs_Fire'] = df_fe['Horizontal_Distance_To_Roadways'] - df_fe['Horizontal_Distance_To_Fire_Points']
    df_fe['Iso_Distance_Mean'] = df_fe[['Horizontal_Distance_To_Roadways', 
                                       'Horizontal_Distance_To_Hydrology', 
                                       'Horizontal_Distance_To_Fire_Points']].mean(axis=1)
    
    # Terrain interactions
    df_fe['Elevation_x_Slope'] = df_fe['Elevation'] * df_fe['Slope']
    df_fe['Elevation_x_HillshadeMean'] = df_fe['Elevation'] * df_fe['Hillshade_Mean']
    
    # Wilderness & Soil as single categorical indices
    wilderness_cols = [col for col in df_fe.columns if col.startswith('Wilderness_Area')]
    soil_cols = [col for col in df_fe.columns if col.startswith('Soil_Type')]
    
    df_fe['Wilderness_Idx'] = df_fe[wilderness_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    df_fe['Soil_Idx'] = df_fe[soil_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    
    # Soil climatic & geologic zones
    df_fe['Soil_ClimaticZone'] = df_fe['Soil_Idx'].map(lambda x: int(str(ELU_codes[x])[0]) if x in ELU_codes else 0)
    df_fe['Soil_GeologicZone'] = df_fe['Soil_Idx'].map(lambda x: int(str(ELU_codes[x])[1]) if x in ELU_codes else 0)
    
    # Fill any NaNs with 0
    df_fe = df_fe.fillna(0)
    
    return df_fe

print("Applying feature engineering...")
train_fe = make_features(train_df)
test_fe = make_features(test_df)

# Ensure identical columns (remove target from train)
feature_cols = [col for col in train_fe.columns if col != 'Cover_Type']
train_fe = train_fe[feature_cols + ['Cover_Type']]
test_fe = test_fe[feature_cols]

print(f"Features after engineering: {len(feature_cols)}")

# Prepare data - keep as DataFrame for CatBoost
X = train_fe[feature_cols]
y = train_fe['Cover_Type'].values - 1  # Convert to 0-6 for training
X_test = test_fe[feature_cols]

print(f"Training data shape: {X.shape}")
print(f"Test data shape: {X_test.shape}")

# Identify categorical features for CatBoost by column names
categorical_features = []
for col in feature_cols:
    if 'Wilderness_Idx' in col or 'Soil_Idx' in col or 'Soil_ClimaticZone' in col or 'Soil_GeologicZone' in col:
        categorical_features.append(col)

print(f"Categorical features: {categorical_features}")

# Initialize dictionaries for results
speed = {}
accuracy = {}

# CatBoost model
print("\n" + "=" * 50)
print("CATBOOST CROSS-VALIDATION")
print("=" * 50)

model = CatBoostClassifier(silent=True, random_state=RANDOM_STATE)

start = time()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
score = cross_val_score(model, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)

speed['CatBoost'] = np.round(time() - start, 3)
accuracy['CatBoost'] = (np.mean(score) * 100).round(3)

print(f"Mean F1 score: {accuracy['CatBoost']}")
print(f"STD: {np.std(score):.3f}")
print(f"Run Time: {speed['CatBoost']}s")

# Train final model for predictions
print("\nTraining final model for predictions...")
final_model = CatBoostClassifier(silent=True, random_state=RANDOM_STATE)
final_model.fit(X, y, cat_features=categorical_features)

# Get feature importance
print("\nTop 20 Feature Importances:")
print("-" * 30)
feature_importance = final_model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(20))

# Generate predictions
print("\nGenerating predictions...")
test_pred_proba = final_model.predict_proba(X_test)
test_pred_class = np.argmax(test_pred_proba, axis=1) + 1  # Convert back to 1-7

# Create submission
submission = pd.DataFrame({
    'Id': range(1, len(test_pred_class) + 1),
    'Cover_Type': test_pred_class
})

submission.to_csv('my_submission.csv', index=False)
print(f"✅ Submission saved to: my_submission.csv")

# Final results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"Mean F1 Score: {accuracy['CatBoost']:.3f}%")
print(f"Standard Deviation: {np.std(score):.3f}")
print(f"Training Time: {speed['CatBoost']}s")

if accuracy['CatBoost'] >= 96.5:
    print("🎯 Target met ✅")
else:
    print(f"❌ Target not met, best = {accuracy['CatBoost']:.3f}%")

print(f"Submission shape: {submission.shape}")
print(f"Predicted class distribution:")
print(submission['Cover_Type'].value_counts().sort_index())

# Confusion matrix on training data
train_pred = final_model.predict(X)
print(f"\nConfusion Matrix (Training):")
print(confusion_matrix(y, train_pred))

print("\n✅ Pipeline completed successfully!")
