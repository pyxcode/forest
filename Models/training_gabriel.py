"""
rainforest.py

Purpose:
    Train a classifier for the Kaggle Forest Cover dataset with two modes:
    - Baseline RandomForest (default)
    - Optimized XGBoost with optional feature engineering and randomized CV tuning
    Includes stratified CV, holdout evaluation, confusion matrix, classification report,
    and OOB score for RandomForest. Falls back to RF automatically if XGBoost is unavailable.

Usage:
    Baseline RF:
        python rainforest.py --train_csv train.csv --target Cover_Type --model_out model.pkl
    XGBoost with FE and search:
        python rainforest.py --train_csv train.csv --target Cover_Type --model_out model_xgb.pkl --algo xgb --fe --cv --n_iter 24

Notes:
    - Requires: pandas, numpy, scikit-learn, xgboost (optional for --algo xgb)
    - Python 3.11+
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================
# Config and CLI
# =========================


@dataclass
class TrainConfig:
    train_csv: str
    target_column: str
    model_out: str
    id_column: Optional[str]
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: Optional[int]
    n_splits: int
    use_cv: bool
    # Extensions
    algo: str  # "rf" or "xgb"
    fe: bool
    cv_folds: int
    n_iter: int
    early_stopping_rounds: int


def parse_args(argv: Optional[List[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a classifier (RF/XGB) on tabular data with optional feature engineering."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--target",
        dest="target_column",
        type=str,
        required=True,
        help="Name of the target column in the training CSV (expects integers 1..7).",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        required=True,
        help="Path to write the trained model pipeline (.pkl). If XGBoost is used, a booster .json may be saved as sibling.",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default=None,
        help="Optional ID column to exclude from features (must be unique).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Holdout test size for evaluation (0-1). Default: 0.2",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for splits and model. Default: 42",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=300,
        help="RandomForest n_estimators. Default: 300",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="RandomForest max_depth. Default: None",
    )
    parser.add_argument(
        "--cv",
        dest="use_cv",
        action="store_true",
        help="Enable K-fold CV evaluation (stratified).",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds when --cv is used. Default: 5 (kept for backward-compat).",
    )
    # New flags
    parser.add_argument(
        "--algo",
        type=str,
        choices=["rf", "xgb"],
        default="rf",
        help="Algorithm to use: rf (RandomForest) or xgb (XGBoost). Default: rf",
    )
    parser.add_argument(
        "--fe",
        action="store_true",
        help="Enable feature engineering.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=None,
        help="Number of CV folds; defaults to value of --n_splits.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=16,
        help="Number of randomized search trials for XGBoost. Default: 16",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=50,
        help="Early stopping rounds for XGBoost per fold. Default: 50",
    )

    args = parser.parse_args(argv)

    cfg = TrainConfig(
        train_csv=args.train_csv,
        target_column=args.target_column,
        model_out=args.model_out,
        id_column=args.id_column,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_splits=args.n_splits,
        use_cv=args.use_cv,
        algo=args.algo,
        fe=args.fe,
        cv_folds=args.cv_folds if args.cv_folds is not None else args.n_splits,
        n_iter=args.n_iter,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    return cfg


# =========================
# Data utilities and checks
# =========================


def load_data(csv_path: str, target_column: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available: {list(df.columns)}"
        )
    return df


def split_features_target(
    df: pd.DataFrame, target_column: str, id_column: Optional[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols: List[str] = [target_column]
    if id_column is not None:
        if id_column not in df.columns:
            raise ValueError(
                f"ID column '{id_column}' not found. Available: {list(df.columns)}"
            )
        if not df[id_column].is_unique:
            dup_count = int(df[id_column].duplicated().sum())
            raise ValueError(
                f"ID column '{id_column}' must be unique, found {dup_count} duplicates."
            )
        drop_cols.append(id_column)

    X = df.drop(columns=drop_cols, errors="raise")
    y = df[target_column]
    if X.shape[0] != y.shape[0]:
        raise ValueError("Features and target have mismatched number of rows.")
    return X, y


def detect_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols: List[str] = []
    numeric_cols: List[str] = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No features detected after excluding target/ID.")
    return numeric_cols, categorical_cols


def ensure_target_labels_exact(df: pd.DataFrame, target_column: str) -> None:
    expected = set(range(1, 8))
    if df[target_column].isnull().any():
        raise ValueError("Target column contains missing values; expected integers 1..7 only.")
    unique_values = set(df[target_column].unique().tolist())
    if not all(isinstance(v, (int, np.integer)) for v in unique_values):
        raise ValueError(
            f"Target values must be integers 1..7; found non-integer values: {sorted(unique_values)}"
        )
    if unique_values != expected:
        raise ValueError(
            f"Target must contain exactly the set {sorted(expected)}; found {sorted(unique_values)}"
        )


# =========================
# Preprocessing and FE
# =========================


def build_preprocessor(
    numeric_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Median-impute numeric features; optional one-hot for categoricals.
    Compatible with sklearn >=1.4 (sparse_output) and <=1.3 (sparse).
    Also avoids constructing a categorical pipeline when there are no categorical columns.
    """
    # Numeric: impute only (no scaling for tree models)
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        # Build encoder with the correct kwarg for the installed sklearn
        try:
            # sklearn >= 1.4
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # sklearn <= 1.3
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", cat_encoder),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


REQUIRED_COLUMNS_FE: Sequence[str] = (
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Fire_Points",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
)


def apply_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    for c in REQUIRED_COLUMNS_FE:
        if c not in df.columns:
            raise ValueError(f"Missing required column for feature engineering: {c}")

    X = df.copy()

    # Base angles
    aspect_rad = np.deg2rad(X["Aspect"].astype(float))
    X["Aspect_Sin"] = np.sin(aspect_rad)
    X["Aspect_Cos"] = np.cos(aspect_rad)

    # Shorthands
    hdist = X["Horizontal_Distance_To_Hydrology"].astype(float)
    vdist = X["Vertical_Distance_To_Hydrology"].astype(float)
    dist_road = X["Horizontal_Distance_To_Roadways"].astype(float)
    dist_fire = X["Horizontal_Distance_To_Fire_Points"].astype(float)
    slope = X["Slope"].astype(float)
    elev = X["Elevation"].astype(float)

    # Hydrology geometry
    X["Hydro_Euclid"] = np.sqrt(hdist**2 + vdist**2)
    X["Hydro_AbsVert"] = np.abs(vdist)
    safe_slope = np.where(slope == 0.0, 1.0, slope)
    X["Hydro_SignedSlopeRatio"] = vdist / safe_slope

    # Hillshade summaries
    h9 = X["Hillshade_9am"].astype(float)
    hn = X["Hillshade_Noon"].astype(float)
    h3 = X["Hillshade_3pm"].astype(float)
    X["Hillshade_Mean"] = (h9 + hn + h3) / 3.0
    X["Hillshade_Range"] = np.maximum.reduce([h9, hn, h3]) - np.minimum.reduce([h9, hn, h3])

    # Distance aggregates
    X["Min_Dist_All"] = np.minimum.reduce([hdist, dist_road, dist_fire])
    X["Max_Dist_All"] = np.maximum.reduce([hdist, dist_road, dist_fire])
    X["Sum_Dist_All"] = hdist + dist_road + dist_fire

    # Elevation interactions
    X["Elev_Slope"] = elev * slope
    X["Elev_AspectSin"] = elev * X["Aspect_Sin"]
    X["Elev_AspectCos"] = elev * X["Aspect_Cos"]

    # --- New features focused on 1↔2 and 3↔6 separations ---
    # Position on slope relative to water (helps high-elevation spruce/fir vs lodgepole)
    X["Elev_minus_VertHydro"] = elev - vdist

    # Relative proximity contrasts (helps disambiguate similar bands)
    X["Diff_Hyd_Road"] = np.abs(hdist - dist_road)
    X["Diff_Road_Fire"] = np.abs(dist_road - dist_fire)
    X["Diff_Hyd_Fire"] = np.abs(hdist - dist_fire)

    engineered_cols = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Hydro_Euclid",
        "Hydro_AbsVert",
        "Hydro_SignedSlopeRatio",
        "Hillshade_Mean",
        "Hillshade_Range",
        "Min_Dist_All",
        "Max_Dist_All",
        "Sum_Dist_All",
        "Elev_Slope",
        "Elev_AspectSin",
        "Elev_AspectCos",
        "Elev_minus_VertHydro",   # NEW
        "Diff_Hyd_Road",          # NEW
        "Diff_Road_Fire",         # NEW
        "Diff_Hyd_Fire",          # NEW
    ]

    for col in engineered_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X, len(engineered_cols)


# =========================
# Models
# =========================


def build_rf_model(
    n_estimators: int, max_depth: Optional[int], random_state: int
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        class_weight=None,
        oob_score=True,
        bootstrap=True,
    )
    return model


def build_pipeline(
    preprocessor: ColumnTransformer, model
) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", model),
        ]
    )
    return pipe


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> dict:
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    try:
        unique_labels = np.unique(y_true)
        if y_proba is not None and y_proba.ndim == 2 and len(unique_labels) == 2:
            pos_index = 1 if y_proba.shape[1] > 1 else 0
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, pos_index]))
    except Exception:
        pass
    return metrics


def kfold_cv_evaluate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    random_state: int,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []

    y_is_binary = y.nunique() == 2

    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

        fitted = pipeline.fit(X_tr, y_tr)
        y_pred = fitted.predict(X_va)
        y_proba: Optional[np.ndarray] = None
        if y_is_binary and hasattr(fitted, "predict_proba"):
            try:
                y_proba = fitted.predict_proba(X_va)
            except Exception:
                y_proba = None

        fold_metrics = evaluate_classification(y_va.to_numpy(), y_pred, y_proba)
        accs.append(fold_metrics["accuracy"])
        f1s.append(fold_metrics["f1_weighted"])
        if "roc_auc" in fold_metrics:
            aucs.append(fold_metrics["roc_auc"])

    results = {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_f1_weighted_mean": float(np.mean(f1s)),
        "cv_f1_weighted_std": float(np.std(f1s)),
    }
    if aucs:
        results["cv_roc_auc_mean"] = float(np.mean(aucs))
        results["cv_roc_auc_std"] = float(np.std(aucs))
    return results


def save_pipeline(model: Pipeline, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(out_dir):
        raise ValueError(f"Output directory does not exist: {out_dir}")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Saved trained pipeline to: {path}")


# =========================
# RF CV helper
# =========================


def run_cv_rf(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    cfg: TrainConfig,
) -> Tuple[dict, Pipeline]:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    rf = build_rf_model(cfg.n_estimators, cfg.max_depth, cfg.random_state)
    pipeline = build_pipeline(preprocessor, rf)

    metrics = kfold_cv_evaluate(
        pipeline, X, y, n_splits=cfg.cv_folds, random_state=cfg.random_state
    )
    return metrics, pipeline


# =========================
# XGBoost helpers
# =========================


def try_import_xgboost():
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier
    except Exception:
        return None


def random_param_sample(rng: np.random.Generator) -> Dict[str, object]:
    space = {
        "max_depth": [6, 7, 8, 9, 10],
        "min_child_weight": [1, 2, 3, 4, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "n_estimators": [600, 800, 1000],
        "reg_lambda": [1.0, 1.5, 2.0],
    }
    return {k: rng.choice(v) for k, v in space.items()}


def _fit_xgb_with_es(clf, X_tr, y_tr, X_va, y_va, early_rounds: int) -> None:
    """
    Fit XGBClassifier with early stopping in a version-agnostic way.
    Tries callbacks API first (xgboost>=1.6), falls back to early_stopping_rounds,
    then fits without early stopping if both are unsupported.
    """
    try:
        import xgboost as xgb  # type: ignore
        try:
            cb = xgb.callback.EarlyStopping(rounds=early_rounds, save_best=True)
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[cb], verbose=False)
            return
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: old API with early_stopping_rounds in fit()
    try:
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=early_rounds, verbose=False)
        return
    except Exception:
        pass

    # Last resort: no early stopping
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)


def run_cv_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TrainConfig,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    """
    Perform randomized search with early stopping using XGBClassifier.
    y is expected to be 1..7; we will internally convert to 0..6.
    Returns: (best_params, best_metrics_dict)
    """
    XGBClassifier = try_import_xgboost()
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed")

    # Convert labels 1..7 -> 0..6 for XGBoost
    y_np = y.to_numpy()
    y_xgb = (y_np - 1).astype(int)

    rng = np.random.default_rng(cfg.random_state)

    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    trials: List[Tuple[Dict[str, object], float, float]] = []
    trials_best_iters: List[List[int]] = []

    base_params = dict(
        objective="multi:softprob",
        num_class=7,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=cfg.random_state,
        n_jobs=-1,
        verbosity=0,
    )

    print("[INFO] Starting XGBoost randomized search:")
    for t in range(1, cfg.n_iter + 1):
        trial_params = {**base_params, **random_param_sample(rng)}
        fold_accs: List[float] = []
        fold_best_iters: List[int] = []

        # Manual imputation per fold (median on train)
        for train_idx, valid_idx in skf.split(X, y_xgb):
            X_tr = X.iloc[train_idx]
            X_va = X.iloc[valid_idx]
            y_tr = y_xgb[train_idx]
            y_va = y_xgb[valid_idx]

            imputer = SimpleImputer(strategy="median")
            X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            X_va_imp = pd.DataFrame(imputer.transform(X_va), columns=X_va.columns, index=X_va.index)

            clf = XGBClassifier(**trial_params)
            _fit_xgb_with_es(
                clf,
                X_tr_imp, y_tr,
                X_va_imp, y_va,
                early_rounds=cfg.early_stopping_rounds,
            )

            # fetch best_iteration if available
            best_it = None
            try:
                if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
                    best_it = int(clf.best_iteration)
                else:
                    booster = clf.get_booster()
                    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
                        best_it = int(booster.best_iteration)
            except Exception:
                best_it = None
            if best_it is not None:
                fold_best_iters.append(best_it)

            y_proba = clf.predict_proba(X_va_imp)
            y_pred = np.argmax(y_proba, axis=1)
            acc = float(accuracy_score(y_va, y_pred))
            fold_accs.append(acc)

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        trials.append((trial_params, mean_acc, std_acc))
        trials_best_iters.append(fold_best_iters)

        # Print compact line
        compact = {k: trial_params[k] for k in sorted(trial_params.keys())}
        print(f"[TRIAL {t:03d}] {mean_acc:.6f} ± {std_acc:.6f} | params={compact}")

    # Select best by mean accuracy
    best_idx = int(np.argmax([m for _, m, _ in trials]))
    best_params, best_mean, best_std = trials[best_idx]
    best_fold_iters = trials_best_iters[best_idx]
    print(
        f"[RESULT] BEST CV accuracy: {best_mean:.6f} ± {best_std:.6f} with params: "
        f"{ {k: best_params[k] for k in sorted(best_params.keys())} }"
    )

    best_metrics = {"cv_accuracy_mean": best_mean, "cv_accuracy_std": best_std}

    # Compute final n_estimators for refit (average of best_iteration across folds if available)
    if best_fold_iters:
        avg_best_iter = int(np.ceil(np.mean(best_fold_iters)))
        best_params["n_estimators"] = max(avg_best_iter, int(best_params.get("n_estimators", 800)))

    return best_params, best_metrics


def save_xgb_artifacts(fitted_pipeline: Pipeline, model_out: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Save sklearn wrapper pipeline (.pkl) and raw booster (.json) if available.
    Returns (pickle_path, booster_json_path).
    """
    out_dir = os.path.dirname(os.path.abspath(model_out)) or "."
    if not os.path.isdir(out_dir):
        raise ValueError(f"Output directory does not exist: {out_dir}")

    # Save pipeline as pickle
    if model_out.lower().endswith(".pkl"):
        pkl_path = model_out
    else:
        base, _ = os.path.splitext(model_out)
        pkl_path = f"{base}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(fitted_pipeline, f)

    # Save booster JSON if available
    booster_json_path: Optional[str] = None
    try:
        clf = fitted_pipeline.named_steps.get("clf")
        if clf is not None and hasattr(clf, "get_booster"):
            booster = clf.get_booster()
            if model_out.lower().endswith(".json"):
                booster_json_path = model_out
            else:
                base, _ = os.path.splitext(model_out)
                booster_json_path = f"{base}_xgb.json"
            booster.save_model(booster_json_path)  # type: ignore[attr-defined]
    except Exception:
        booster_json_path = None

    return pkl_path, booster_json_path


# =========================
# Main
# =========================


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)

    print("[INFO] Loading data...")
    df = load_data(cfg.train_csv, cfg.target_column)
    ensure_target_labels_exact(df, cfg.target_column)

    print("[INFO] Preparing features/target...")
    X, y = split_features_target(df, cfg.target_column, cfg.id_column)

    # Optional FE
    num_engineered = 0
    if cfg.fe:
        X, num_engineered = apply_feature_engineering(X)

    print(f"[INFO] Dataset shape: X={X.shape}, y={(y.shape[0],)}")
    if cfg.fe:
        print(f"[INFO] Engineered features added: {num_engineered}. Final feature count: {X.shape[1]}")

    print("[INFO] Detecting feature types...")
    numeric_cols, categorical_cols = detect_feature_types(X)
    print(
        f"[INFO] Detected {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features."
    )

    # Choose algorithm
    algo = cfg.algo
    XGBClassifier = try_import_xgboost() if algo == "xgb" else None
    if algo == "xgb" and XGBClassifier is None:
        print("[WARN] xgboost not available. Falling back to RandomForest.")
        algo = "rf"

    if algo == "rf":
        # Build pipeline
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        model = build_rf_model(cfg.n_estimators, cfg.max_depth, cfg.random_state)
        pipeline = build_pipeline(preprocessor, model)

        if cfg.use_cv:
            print("[INFO] Running stratified K-fold cross-validation (RF)...")
            cv_metrics = kfold_cv_evaluate(
                pipeline, X, y, n_splits=cfg.cv_folds, random_state=cfg.random_state
            )
            print("[RESULT] CV metrics:")
            for k, v in cv_metrics.items():
                print(f"  - {k}: {v:.6f}")

        print("[INFO] Creating holdout split...")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y,
        )

        print("[INFO] Training RF model...")
        fitted = pipeline.fit(X_train, y_train)
        try:
            if hasattr(fitted.named_steps.get("clf"), "oob_score_"):
                print(f"[RESULT] OOB accuracy: {fitted['clf'].oob_score_:.6f}")
        except Exception:
            pass

        print("[INFO] Evaluating on holdout...")
        y_pred = fitted.predict(X_valid)
        y_proba: Optional[np.ndarray] = None
        if y.nunique() == 2 and hasattr(fitted, "predict_proba"):
            try:
                y_proba = fitted.predict_proba(X_valid)
            except Exception:
                y_proba = None

        metrics = evaluate_classification(y_valid.to_numpy(), y_pred, y_proba)
        print("[RESULT] Holdout metrics:")
        for k, v in metrics.items():
            print(f"  - {k}: {v:.6f}")
        # Detailed diagnostics with fixed labels 1..7
        try:
            report = classification_report(y_valid, y_pred, digits=6)
            print("[RESULT] Classification report:\n" + report)
        except Exception:
            pass
        try:
            labels = list(range(1, 8))
            cm = confusion_matrix(y_valid, y_pred, labels=labels)
            print("[RESULT] Confusion matrix (rows=actual 1..7, cols=pred 1..7):")
            print(np.array2string(cm, separator=", "))
        except Exception:
            pass

        print("[INFO] Saving model...")
        save_pipeline(fitted, cfg.model_out)
        print("[INFO] Done.")
        return

    # XGBoost path
    # CV randomized search for best params
    best_params: Dict[str, object]
    best_cv_metrics: Dict[str, float]
    if cfg.use_cv:
        try:
            best_params, best_cv_metrics = run_cv_xgb(X, y, cfg)
        except ImportError:
            print("[WARN] xgboost not available during CV. Falling back to RandomForest.")
            # Fall back to RF CV + train path
            preprocessor = build_preprocessor(numeric_cols, categorical_cols)
            model = build_rf_model(cfg.n_estimators, cfg.max_depth, cfg.random_state)
            pipeline = build_pipeline(preprocessor, model)
            cv_metrics = kfold_cv_evaluate(
                pipeline, X, y, n_splits=cfg.cv_folds, random_state=cfg.random_state
            )
            print("[RESULT] CV metrics (RF fallback):")
            for k, v in cv_metrics.items():
                print(f"  - {k}: {v:.6f}")
            # Proceed like RF
            print("[INFO] Creating holdout split...")
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
            )
            print("[INFO] Training RF model...")
            fitted = pipeline.fit(X_train, y_train)
            try:
                if hasattr(fitted.named_steps.get("clf"), "oob_score_"):
                    print(f"[RESULT] OOB accuracy: {fitted['clf'].oob_score_:.6f}")
            except Exception:
                pass
            print("[INFO] Evaluating on holdout...")
            y_pred = fitted.predict(X_valid)
            metrics = evaluate_classification(y_valid.to_numpy(), y_pred, None)
            print("[RESULT] Holdout metrics:")
            for k, v in metrics.items():
                print(f"  - {k}: {v:.6f}")
            try:
                report = classification_report(y_valid, y_pred, digits=6)
                print("[RESULT] Classification report:\n" + report)
            except Exception:
                pass
            try:
                labels = list(range(1, 8))
                cm = confusion_matrix(y_valid, y_pred, labels=labels)
                print("[RESULT] Confusion matrix (rows=actual 1..7, cols=pred 1..7):")
                print(np.array2string(cm, separator=", "))
            except Exception:
                pass
            print("[INFO] Saving model...")
            save_pipeline(fitted, cfg.model_out)
            print("[INFO] Done.")
            return
        print("[RESULT] XGB best CV metrics:")
        for k, v in best_cv_metrics.items():
            print(f"  - {k}: {v:.6f}")
    else:
        # Default parameters if no CV search
        XGBClassifier = try_import_xgboost()
        if XGBClassifier is None:
            print("[WARN] xgboost not available. Falling back to RandomForest.")
            preprocessor = build_preprocessor(numeric_cols, categorical_cols)
            model = build_rf_model(cfg.n_estimators, cfg.max_depth, cfg.random_state)
            pipeline = build_pipeline(preprocessor, model)
            # Proceed to holdout training/eval
            print("[INFO] Creating holdout split...")
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
            )
            print("[INFO] Training RF model...")
            fitted = pipeline.fit(X_train, y_train)
            try:
                if hasattr(fitted.named_steps.get("clf"), "oob_score_"):
                    print(f"[RESULT] OOB accuracy: {fitted['clf'].oob_score_:.6f}")
            except Exception:
                pass
            print("[INFO] Evaluating on holdout...")
            y_pred = fitted.predict(X_valid)
            metrics = evaluate_classification(y_valid.to_numpy(), y_pred, None)
            print("[RESULT] Holdout metrics:")
            for k, v in metrics.items():
                print(f"  - {k}: {v:.6f}")
            try:
                report = classification_report(y_valid, y_pred, digits=6)
                print("[RESULT] Classification report:\n" + report)
            except Exception:
                pass
            try:
                labels = list(range(1, 8))
                cm = confusion_matrix(y_valid, y_pred, labels=labels)
                print("[RESULT] Confusion matrix (rows=actual 1..7, cols=pred 1..7):")
                print(np.array2string(cm, separator=", "))
            except Exception:
                pass
            print("[INFO] Saving model...")
            save_pipeline(fitted, cfg.model_out)
            print("[INFO] Done.")
            return
        best_params = dict(
            objective="multi:softprob",
            num_class=7,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=0,
            max_depth=8,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.8,
            learning_rate=0.07,
            n_estimators=800,
            reg_lambda=1.5,
        )
        best_cv_metrics = {"cv_accuracy_mean": float("nan"), "cv_accuracy_std": float("nan")}
        print("[INFO] Using default XGB params (no CV search).")

    # Holdout split
    print("[INFO] Creating holdout split...")
    X_train, X_valid, y_train_orig, y_valid_orig = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # Build final XGB pipeline: SimpleImputer + XGBClassifier
    XGBClassifier = try_import_xgboost()
    if XGBClassifier is None:
        print("[WARN] xgboost not available at refit. Falling back to RandomForest.")
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        model = build_rf_model(cfg.n_estimators, cfg.max_depth, cfg.random_state)
        pipeline = build_pipeline(preprocessor, model)
        fitted = pipeline.fit(X_train, y_train_orig)
        try:
            if hasattr(fitted.named_steps.get("clf"), "oob_score_"):
                print(f"[RESULT] OOB accuracy: {fitted['clf'].oob_score_:.6f}")
        except Exception:
            pass
        y_pred = fitted.predict(X_valid)
        metrics = evaluate_classification(y_valid_orig.to_numpy(), y_pred, None)
        print("[RESULT] Holdout metrics:")
        for k, v in metrics.items():
            print(f"  - {k}: {v:.6f}")
        try:
            report = classification_report(y_valid_orig, y_pred, digits=6)
            print("[RESULT] Classification report:\n" + report)
        except Exception:
            pass
        try:
            labels = list(range(1, 8))
            cm = confusion_matrix(y_valid_orig, y_pred, labels=labels)
            print("[RESULT] Confusion matrix (rows=actual 1..7, cols=pred 1..7):")
            print(np.array2string(cm, separator=", "))
        except Exception:
            pass
        print("[INFO] Saving model...")
        save_pipeline(fitted, cfg.model_out)
        print("[INFO] Done.")
        return

    # Convert labels for XGB fit (0..6)
    y_train = (y_train_orig.to_numpy() - 1).astype(int)
    y_valid = (y_valid_orig.to_numpy() - 1).astype(int)

    xgb_clf = XGBClassifier(**best_params)
    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), list(X.columns))],
        remainder="drop",
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", xgb_clf)])

    print("[INFO] Training XGB model on full training split...")
    fitted = pipeline.fit(X_train, y_train)

    print("[INFO] Evaluating XGB on holdout...")
    # Predict probabilities/classes then map back to 1..7
    if hasattr(fitted.named_steps["clf"], "predict_proba"):
        y_proba_valid = fitted.predict_proba(X_valid)
        y_pred_idx = np.argmax(y_proba_valid, axis=1)
    else:
        y_pred_idx = fitted.predict(X_valid)
        y_proba_valid = None
    y_pred_labels = (y_pred_idx + 1).astype(int)
    # Compare against original labels (1..7)
    metrics = evaluate_classification(y_valid_orig.to_numpy(), y_pred_labels, y_proba_valid)
    print("[RESULT] Holdout metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.6f}")
    try:
        report = classification_report(y_valid_orig, y_pred_labels, digits=6)
        print("[RESULT] Classification report:\n" + report)
    except Exception:
        pass
    try:
        labels = list(range(1, 8))
        cm = confusion_matrix(y_valid_orig, y_pred_labels, labels=labels)
        print("[RESULT] Confusion matrix (rows=actual 1..7, cols=pred 1..7):")
        print(np.array2string(cm, separator=", "))
    except Exception:
        pass

    print("[INFO] Saving XGB pipeline and booster (if available)...")
    pkl_path, booster_json_path = save_xgb_artifacts(fitted, cfg.model_out)
    if pkl_path:
        print(f"[INFO] Saved sklearn pipeline to: {pkl_path}")
    if booster_json_path:
        print(f"[INFO] Saved XGBoost booster JSON to: {booster_json_path}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()