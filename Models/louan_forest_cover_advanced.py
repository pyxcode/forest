import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from time import time
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("🌲 Forest Cover Type Classification - Advanced Feature Engineering")
print("=" * 70)

# Load data
print("Loading data...")
try:
    train_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
    test_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
    print("✅ Loaded from Kaggle input path")
except:
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

def create_geographic_features(df):
    """Create geographic and terrain-based features"""
    print("🗻 Creating geographic features...")
    features = {}
    
    # Elevation features
    features['Elevation'] = df['Elevation']
    features['Elevation_Squared'] = df['Elevation'] ** 2
    features['Elevation_Log'] = np.log1p(df['Elevation'])
    features['Elevation_Sqrt'] = np.sqrt(df['Elevation'])
    
    # Aspect features (circular)
    features['Aspect_Sin'] = np.sin(df['Aspect'] * np.pi / 180)
    features['Aspect_Cos'] = np.cos(df['Aspect'] * np.pi / 180)
    features['Aspect_Radians'] = df['Aspect'] * np.pi / 180
    features['Aspect_Normalized'] = df['Aspect'] / 360.0
    
    # Slope features
    features['Slope'] = df['Slope']
    features['Slope_Squared'] = df['Slope'] ** 2
    features['Slope_Log'] = np.log1p(df['Slope'])
    features['Slope_Sqrt'] = np.sqrt(df['Slope'])
    
    # Terrain complexity
    features['Terrain_Roughness'] = np.abs(df['Elevation'] - df['Elevation'].rolling(5, center=True).mean()).fillna(0)
    features['Slope_Complexity'] = np.abs(df['Slope'] - df['Slope'].rolling(5, center=True).mean()).fillna(0)
    
    return pd.DataFrame(features)

def create_distance_features(df):
    """Create distance-based and interaction features"""
    print("📏 Creating distance features...")
    features = {}
    
    # Basic distances
    features['Horizontal_Distance_To_Hydrology'] = df['Horizontal_Distance_To_Hydrology']
    features['Vertical_Distance_To_Hydrology'] = df['Vertical_Distance_To_Hydrology']
    features['Horizontal_Distance_To_Roadways'] = df['Horizontal_Distance_To_Roadways']
    features['Horizontal_Distance_To_Fire_Points'] = df['Horizontal_Distance_To_Fire_Points']
    
    # Euclidean distance to hydrology
    features['Hydro_Dist_Euclid'] = np.sqrt(
        df['Horizontal_Distance_To_Hydrology']**2 + 
        df['Vertical_Distance_To_Hydrology']**2
    )
    
    # Distance ratios and interactions
    features['Road_Hydro_Ratio'] = df['Horizontal_Distance_To_Roadways'] / (df['Horizontal_Distance_To_Hydrology'] + 1)
    features['Fire_Hydro_Ratio'] = df['Horizontal_Distance_To_Fire_Points'] / (df['Horizontal_Distance_To_Hydrology'] + 1)
    features['Road_Fire_Ratio'] = df['Horizontal_Distance_To_Roadways'] / (df['Horizontal_Distance_To_Fire_Points'] + 1)
    
    # Distance contrasts
    features['Road_vs_Hydro'] = df['Horizontal_Distance_To_Roadways'] - df['Horizontal_Distance_To_Hydrology']
    features['Fire_vs_Hydro'] = df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Hydrology']
    features['Road_vs_Fire'] = df['Horizontal_Distance_To_Roadways'] - df['Horizontal_Distance_To_Fire_Points']
    
    # Distance statistics
    distance_cols = ['Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Fire_Points']
    features['Distance_Sum'] = df[distance_cols].sum(axis=1)
    features['Distance_Mean'] = df[distance_cols].mean(axis=1)
    features['Distance_Std'] = df[distance_cols].std(axis=1)
    features['Distance_Min'] = df[distance_cols].min(axis=1)
    features['Distance_Max'] = df[distance_cols].max(axis=1)
    features['Distance_Range'] = df[distance_cols].max(axis=1) - df[distance_cols].min(axis=1)
    
    # Isolation index
    features['Isolation_Index'] = features['Distance_Mean'] / (features['Distance_Std'] + 1)
    
    return pd.DataFrame(features)

def create_hillshade_features(df):
    """Create hillshade and lighting-based features"""
    print("☀️ Creating hillshade features...")
    features = {}
    
    hillshade_cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    
    # Basic hillshade values
    for col in hillshade_cols:
        features[col] = df[col]
    
    # Statistical features
    features['Hillshade_Mean'] = df[hillshade_cols].mean(axis=1)
    features['Hillshade_Median'] = df[hillshade_cols].median(axis=1)
    features['Hillshade_Std'] = df[hillshade_cols].std(axis=1)
    features['Hillshade_Min'] = df[hillshade_cols].min(axis=1)
    features['Hillshade_Max'] = df[hillshade_cols].max(axis=1)
    features['Hillshade_Range'] = df[hillshade_cols].max(axis=1) - df[hillshade_cols].min(axis=1)
    features['Hillshade_Sum'] = df[hillshade_cols].sum(axis=1)
    
    # Hillshade patterns
    features['Hillshade_Consistency'] = 1 - (features['Hillshade_Std'] / (features['Hillshade_Mean'] + 1))
    features['Hillshade_Progression'] = df['Hillshade_3pm'] - df['Hillshade_9am']
    features['Hillshade_Peak'] = (df['Hillshade_Noon'] > df['Hillshade_9am']) & (df['Hillshade_Noon'] > df['Hillshade_3pm'])
    features['Hillshade_Valley'] = (df['Hillshade_Noon'] < df['Hillshade_9am']) & (df['Hillshade_Noon'] < df['Hillshade_3pm'])
    
    # Normalized hillshade
    features['Hillshade_9am_Norm'] = df['Hillshade_9am'] / 255.0
    features['Hillshade_Noon_Norm'] = df['Hillshade_Noon'] / 255.0
    features['Hillshade_3pm_Norm'] = df['Hillshade_3pm'] / 255.0
    
    return pd.DataFrame(features)

def create_soil_features(df):
    """Create soil type and classification features"""
    print("🌱 Creating soil features...")
    features = {}
    
    # Soil type one-hot encoding
    soil_cols = [col for col in df.columns if col.startswith('Soil_Type')]
    for col in soil_cols:
        features[col] = df[col]
    
    # Soil categorical index
    features['Soil_Idx'] = df[soil_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    
    # Soil climatic and geologic zones
    features['Soil_ClimaticZone'] = features['Soil_Idx'].map(lambda x: int(str(ELU_codes[x])[0]) if x in ELU_codes else 0)
    features['Soil_GeologicZone'] = features['Soil_Idx'].map(lambda x: int(str(ELU_codes[x])[1]) if x in ELU_codes else 0)
    
    # Soil diversity (number of soil types present)
    features['Soil_Diversity'] = df[soil_cols].sum(axis=1)
    
    # Soil type groups (based on ELU codes)
    features['Soil_Group_1'] = features['Soil_ClimaticZone'].isin([2, 3]).astype(int)  # Low elevation
    features['Soil_Group_2'] = features['Soil_ClimaticZone'].isin([4, 5]).astype(int)  # Mid elevation
    features['Soil_Group_3'] = features['Soil_ClimaticZone'].isin([6, 7]).astype(int)  # High elevation
    features['Soil_Group_4'] = features['Soil_ClimaticZone'].isin([8]).astype(int)     # Alpine
    
    return pd.DataFrame(features)

def create_wilderness_features(df):
    """Create wilderness area features"""
    print("🏔️ Creating wilderness features...")
    features = {}
    
    # Wilderness area one-hot encoding
    wilderness_cols = [col for col in df.columns if col.startswith('Wilderness_Area')]
    for col in wilderness_cols:
        features[col] = df[col]
    
    # Wilderness categorical index
    features['Wilderness_Idx'] = df[wilderness_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    
    # Wilderness diversity
    features['Wilderness_Diversity'] = df[wilderness_cols].sum(axis=1)
    
    return pd.DataFrame(features)

def create_interaction_features(df, geo_features, dist_features, hillshade_features):
    """Create complex interaction features"""
    print("🔗 Creating interaction features...")
    features = {}
    
    # Elevation interactions
    features['Elevation_x_Slope'] = geo_features['Elevation'] * geo_features['Slope']
    features['Elevation_x_HillshadeMean'] = geo_features['Elevation'] * hillshade_features['Hillshade_Mean']
    features['Elevation_x_HydroDist'] = geo_features['Elevation'] * dist_features['Hydro_Dist_Euclid']
    
    # Slope interactions
    features['Slope_x_HillshadeMean'] = geo_features['Slope'] * hillshade_features['Hillshade_Mean']
    features['Slope_x_AspectSin'] = geo_features['Slope'] * geo_features['Aspect_Sin']
    features['Slope_x_AspectCos'] = geo_features['Slope'] * geo_features['Aspect_Cos']
    
    # Distance interactions
    features['Road_x_Elevation'] = dist_features['Horizontal_Distance_To_Roadways'] * geo_features['Elevation']
    features['Fire_x_Elevation'] = dist_features['Horizontal_Distance_To_Fire_Points'] * geo_features['Elevation']
    features['Hydro_x_Elevation'] = dist_features['Horizontal_Distance_To_Hydrology'] * geo_features['Elevation']
    
    # Hillshade interactions
    features['Hillshade_Mean_x_AspectSin'] = hillshade_features['Hillshade_Mean'] * geo_features['Aspect_Sin']
    features['Hillshade_Mean_x_AspectCos'] = hillshade_features['Hillshade_Mean'] * geo_features['Aspect_Cos']
    
    # Complex terrain features
    features['Terrain_Complexity'] = (geo_features['Elevation'] * geo_features['Slope'] * 
                                    (1 + geo_features['Aspect_Std'] if 'Aspect_Std' in geo_features else 1))
    
    return pd.DataFrame(features)

def create_advanced_features(df):
    """Create advanced statistical and derived features"""
    print("🧮 Creating advanced features...")
    features = {}
    
    # Polynomial features for key variables
    elevation = df['Elevation']
    slope = df['Slope']
    
    features['Elevation_Cubed'] = elevation ** 3
    features['Slope_Cubed'] = slope ** 3
    features['Elevation_Slope_Product'] = elevation * slope
    features['Elevation_Slope_Squared'] = (elevation * slope) ** 2
    
    # Aspect-based features
    aspect_rad = df['Aspect'] * np.pi / 180
    features['Aspect_North'] = np.cos(aspect_rad)  # North-facing
    features['Aspect_East'] = np.sin(aspect_rad)   # East-facing
    features['Aspect_South'] = -np.cos(aspect_rad) # South-facing
    features['Aspect_West'] = -np.sin(aspect_rad)  # West-facing
    
    # Distance-based advanced features
    hydro_h = df['Horizontal_Distance_To_Hydrology']
    hydro_v = df['Vertical_Distance_To_Hydrology']
    road = df['Horizontal_Distance_To_Roadways']
    fire = df['Horizontal_Distance_To_Fire_Points']
    
    features['Hydro_Angle'] = np.arctan2(hydro_v, hydro_h)
    features['Accessibility_Index'] = 1 / (road + 1) + 1 / (fire + 1)
    features['Water_Accessibility'] = 1 / (np.sqrt(hydro_h**2 + hydro_v**2) + 1)
    
    return pd.DataFrame(features)

def create_all_features(df):
    """Create all feature types and combine them"""
    print("🚀 Creating comprehensive feature set...")
    
    # Create individual feature groups
    geo_features = create_geographic_features(df)
    dist_features = create_distance_features(df)
    hillshade_features = create_hillshade_features(df)
    soil_features = create_soil_features(df)
    wilderness_features = create_wilderness_features(df)
    interaction_features = create_interaction_features(df, geo_features, dist_features, hillshade_features)
    advanced_features = create_advanced_features(df)
    
    # Combine all features
    all_features = pd.concat([
        geo_features,
        dist_features, 
        hillshade_features,
        soil_features,
        wilderness_features,
        interaction_features,
        advanced_features
    ], axis=1)
    
    # Fill any NaNs
    all_features = all_features.fillna(0)
    
    print(f"✅ Created {all_features.shape[1]} features")
    return all_features

# Apply feature engineering
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

train_features = create_all_features(train_df)
test_features = create_all_features(test_df)

# Ensure identical columns
feature_cols = train_features.columns.tolist()
train_features = train_features[feature_cols]
test_features = test_features[feature_cols]

print(f"Final feature count: {len(feature_cols)}")

# Prepare data
X = train_features.values
y = train_df['Cover_Type'].values - 1  # Convert to 0-6
X_test = test_features.values

print(f"Training data shape: {X.shape}")
print(f"Test data shape: {X_test.shape}")

# Feature selection
print("\n🔍 Performing feature selection...")
selector = SelectKBest(score_func=f_classif, k=min(100, X.shape[1]))
X_selected = selector.fit_transform(X, y)
X_test_selected = selector.transform(X_test)

selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected {len(selected_features)} best features")

# XGBoost parameters
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'eval_metric': ['mlogloss', 'merror'],
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 2.0,
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

# Cross-validation
print("\n" + "="*50)
print("CROSS-VALIDATION")
print("="*50)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_accuracies = []
test_predictions = np.zeros((X_test_selected.shape[0], 7))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
    X_train_fold, X_val_fold = X_selected[train_idx], X_selected[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, feature_names=selected_features)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold, feature_names=selected_features)
    dtest = xgb.DMatrix(X_test_selected, feature_names=selected_features)
    
    # Train model
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=200,
        verbose_eval=False
    )
    
    # Predictions
    val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
    val_pred_class = np.argmax(val_pred, axis=1)
    test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration))
    
    # Calculate accuracy
    fold_acc = accuracy_score(y_val_fold, val_pred_class)
    fold_accuracies.append(fold_acc)
    test_predictions += test_pred / 5
    
    print(f"Fold {fold + 1}: {fold_acc:.4f}")

mean_accuracy = np.mean(fold_accuracies)
print(f"Mean CV Accuracy: {mean_accuracy:.4f}")

# Feature importance
print("\nTop 20 Feature Importances:")
print("-" * 30)
dtrain = xgb.DMatrix(X_selected, label=y, feature_names=selected_features)
model = xgb.train(xgb_params, dtrain, num_boost_round=1000, verbose_eval=False)

importance_scores = model.get_score(importance_type='gain')
importance_list = [(selected_features[i], importance_scores.get(f'f{i}', 0)) for i in range(len(selected_features))]
importance_list.sort(key=lambda x: x[1], reverse=True)

importance_df = pd.DataFrame(importance_list[:20], columns=['feature', 'importance'])
print(importance_df)

# Generate submission
print("\nGenerating submission...")
test_pred_class = np.argmax(test_predictions, axis=1) + 1

submission = pd.DataFrame({
    'Id': range(1, len(test_pred_class) + 1),
    'Cover_Type': test_pred_class
})

submission.to_csv('my_submission_advanced.csv', index=False)
print(f"✅ Submission saved to: my_submission_advanced.csv")

# Final results
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Mean CV Accuracy: {mean_accuracy:.4f}")

if mean_accuracy >= 0.965:
    print("🎯 Target met ✅")
else:
    print(f"❌ Target not met, best = {mean_accuracy:.4f}")

print(f"Features used: {len(selected_features)}")
print(f"Submission shape: {submission.shape}")
print(f"Predicted class distribution:")
print(submission['Cover_Type'].value_counts().sort_index())

print("\n✅ Advanced pipeline completed successfully!")
