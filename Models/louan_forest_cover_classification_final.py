import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("🌲 Forest Cover Type Classification Pipeline")
print("=" * 50)

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

# Prepare data
X = train_fe[feature_cols].values
y = train_fe['Cover_Type'].values - 1  # Convert to 0-6 for training
X_test = test_fe[feature_cols].values

print(f"Training data shape: {X.shape}")
print(f"Test data shape: {X_test.shape}")

def fit_and_evaluate(model_name, params, X, y, X_test, feature_names):
    """
    Fit model with cross-validation and return results
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    fold_accuracies = []
    test_predictions = np.zeros((X_test.shape[0], 7))
    oof_predictions = np.zeros((X.shape[0], 7))
    
    print(f"\n{model_name} Cross-Validation:")
    print("-" * 30)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, feature_names=feature_names)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        
        # Train model
        model = xgb.train(
            params,
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
        
        # Store predictions
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / 5
        
        print(f"Fold {fold + 1}: {fold_acc:.4f}")
    
    mean_accuracy = np.mean(fold_accuracies)
    print(f"Mean CV Accuracy: {mean_accuracy:.4f}")
    
    return fold_accuracies, mean_accuracy, test_predictions, oof_predictions

# XGBoost parameters
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'eval_metric': ['mlogloss', 'merror'],
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

# Train initial XGBoost model
fold_accs, mean_acc, test_preds, oof_preds = fit_and_evaluate(
    "XGBoost", xgb_params, X, y, X_test, feature_cols
)

# Get feature importance
print("\nTop 20 Feature Importances:")
print("-" * 30)
dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
model = xgb.train(xgb_params, dtrain, num_boost_round=1000, verbose_eval=False)

# Get feature importance scores
importance_scores = model.get_score(importance_type='gain')
importance_list = [(feature, importance_scores.get(f'f{i}', 0)) for i, feature in enumerate(feature_cols)]
importance_list.sort(key=lambda x: x[1], reverse=True)

importance_df = pd.DataFrame(importance_list[:20], columns=['feature', 'importance'])
print(importance_df)

# Auto-escalation if mean CV < 0.965
if mean_acc < 0.965:
    print(f"\n⚠️  Mean CV accuracy {mean_acc:.4f} < 0.965, starting auto-escalation...")
    
    # Try 1: Reduce learning rate
    print("\nTrying reduced learning rate...")
    xgb_params_tuned = xgb_params.copy()
    xgb_params_tuned['learning_rate'] = 0.03
    
    fold_accs_tuned, mean_acc_tuned, test_preds_tuned, oof_preds_tuned = fit_and_evaluate(
        "XGBoost (LR=0.03)", xgb_params_tuned, X, y, X_test, feature_cols
    )
    
    if mean_acc_tuned > mean_acc:
        mean_acc = mean_acc_tuned
        test_preds = test_preds_tuned
        oof_preds = oof_preds_tuned
        print(f"✅ Improved to {mean_acc:.4f} with reduced learning rate")
    
    # Try 2: Random search if still below target
    if mean_acc < 0.965:
        print("\nTrying random search...")
        best_acc = mean_acc
        best_params = xgb_params.copy()
        
        for trial in range(30):
            params = xgb_params.copy()
            params['max_depth'] = np.random.choice([4, 6, 8, 10])
            params['min_child_weight'] = np.random.choice([1, 3, 5, 7])
            params['subsample'] = np.random.choice([0.7, 0.8, 0.9])
            params['colsample_bytree'] = np.random.choice([0.7, 0.8, 0.9])
            params['reg_lambda'] = np.random.choice([0.0, 0.5, 1.0, 5.0])
            
            _, trial_acc, _, _ = fit_and_evaluate(
                f"Trial {trial+1}", params, X, y, X_test, feature_cols
            )
            
            if trial_acc > best_acc:
                best_acc = trial_acc
                best_params = params
                print(f"✅ New best: {best_acc:.4f}")
                
            if best_acc >= 0.965:
                break
        
        if best_acc > mean_acc:
            mean_acc = best_acc
            test_preds = fit_and_evaluate("Best XGBoost", best_params, X, y, X_test, feature_cols)[2]
            print(f"✅ Final XGBoost: {mean_acc:.4f}")

# Generate submission
print("\nGenerating submission...")
test_pred_class = np.argmax(test_preds, axis=1) + 1  # Convert back to 1-7

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
print(f"Mean CV Accuracy: {mean_acc:.4f}")

if mean_acc >= 0.965:
    print("🎯 Target met ✅")
else:
    print(f"❌ Target not met, best = {mean_acc:.4f}")

print(f"Submission shape: {submission.shape}")
print(f"Predicted class distribution:")
print(submission['Cover_Type'].value_counts().sort_index())

# Confusion matrix on out-of-fold predictions
oof_pred_class = np.argmax(oof_preds, axis=1)
print(f"\nConfusion Matrix (Out-of-Fold):")
print(confusion_matrix(y, oof_pred_class))

print("\n✅ Pipeline completed successfully!")
