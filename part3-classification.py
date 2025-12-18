"""
Part 3 - Classification: CDC Diabetes Health Indicators with AdaBoost

This script:
  - Loads the CDC Diabetes dataset from UCI (id = 891)
  - Splits it into train / val / test
  - Trains multiple AdaBoost models with different hyperparameters
  - Logs performance metrics and saves plots + a summary file

Outputs are written to: results_classification/
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from kagglehub import KaggleDatasetAdapter
from textwrap import wrap
from matplotlib.backends.backend_pdf import PdfPages
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.size"] = 11

OUTPUT_DIR = "results_classification"
ADABOOST_OUTPUT_DIR = OUTPUT_DIR + "/adaboost"
GRADBOOST_OUTPUT_DIR = OUTPUT_DIR + "/gradboost"
TREE_OUTPUT_DIR = OUTPUT_DIR + "/baseline_tree"
os.makedirs(TREE_OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADABOOST_OUTPUT_DIR, exist_ok=True)
os.makedirs(GRADBOOST_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------

def load_cdc_diabetes_data():
    """Load CDC Diabetes Health Indicators dataset from UCI."""
    print("Loading CDC Diabetes Health Indicators dataset (UCI id=891)...")
    cdc_diabetes = fetch_ucirepo(id=891)

    X = cdc_diabetes.data.features
    y = cdc_diabetes.data.targets

    # Use the first (and only) target column as label
    target_col = y.columns[0]
    y_series = y[target_col].astype(int)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y_series.shape}")
    print(f"Target column: {target_col}")
    print(f"Class distribution:\n{y_series.value_counts()}\n")

    return X, y_series, target_col

def load_cdc_diabetes_data_OLD():
    """
    Load CDC Diabetes Health Indicators dataset from Kaggle.
    Uses the binary target version: diabetes_binary_health_indicators_BRFSS2015.csv
    """
    print("Loading CDC Diabetes Health Indicators dataset from Kaggle...")

    data_dir = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    print(f"Dataset downloaded to: {data_dir}")

    csv_path = os.path.join(data_dir, "diabetes_binary_health_indicators_BRFSS2015.csv")
    print(f"Loading CSV from: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="latin1")
    except Exception as e:
        print("Standard read_csv failed, retrying with python engine and explicit sep=',' ...")
        df = pd.read_csv(
            csv_path,
            encoding="latin1",
            engine="python",
            sep=",",
            on_bad_lines="error"
        )

    print(f"Raw dataframe shape: {df.shape[0]} samples, {df.shape[1]} columns")

    target_col = "Diabetes_binary"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns are: {df.columns.tolist()}")

    y_series = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y_series.shape}")
    print(f"Target column: {target_col}")
    print("Class distribution (0=no diabetes, 1=diabetes):")
    print(y_series.value_counts(), "\n")

    return X, y_series, target_col


# ---------------------------------------------------------------------
# 2. DATA SPLIT
# ---------------------------------------------------------------------

def split_data(X, y, random_state=42):
    """Split into train (60%), validation (20%), and test (20%), stratified."""
    print("Splitting data into train (60%), val (20%), test (20%)...")

    # First: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    # Then: train vs val from train_val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,  # 0.25 of 0.8 = 0.2, so 60/20/20 overall
        stratify=y_train_val,
        random_state=random_state,
    )

    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val   shape: {X_val.shape}, {y_val.shape}")
    print(f"Test  shape: {X_test.shape}, {y_test.shape}\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------
# 3. TRAINING & HYPERPARAMETER EXPLORATION
# ---------------------------------------------------------------------

def train_eval_adaboost_grid(X_train, y_train, X_val, y_val, random_state=42):
    """
    Run a manual grid search over AdaBoost hyperparameters.
    Returns:
      - results_df: DataFrame with metrics per config
      - best_config: dict with best hyperparameters based on validation ROC-AUC
    """
    print("=" * 80)
    print("ADA BOOST HYPERPARAMETER EXPLORATION")
    print("=" * 80)

    # Hyperparameter grid (adjust if too slow / fast)
    n_estimators_list = [50, 100, 200, 400]
    learning_rates = [0.05, 0.1, 0.5]
    max_depths = [1, 2]  # depth of base trees

    records = []

    total_configs = len(n_estimators_list) * len(learning_rates) * len(max_depths)
    config_idx = 0

    for max_depth in max_depths:
        for lr in learning_rates:
            for n_estimators in n_estimators_list:
                config_idx += 1
                print(f"\nConfig {config_idx}/{total_configs}: "f"n_estimators={n_estimators}, learning_rate={lr}, max_depth={max_depth}")

                base_tree = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=random_state)
                model = AdaBoostClassifier(estimator=base_tree, n_estimators=n_estimators, learning_rate=lr, random_state=random_state)

                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time

                # Validation predictions
                val_pred = model.predict(X_val)
                val_proba = model.predict_proba(X_val)[:, 1]

                # Metrics (treat class 1 = Diabetes as positive)
                acc = accuracy_score(y_val, val_pred)
                prec = precision_score(y_val, val_pred, zero_division=0)
                rec = recall_score(y_val, val_pred, zero_division=0)
                f1 = f1_score(y_val, val_pred, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_val, val_proba)
                except ValueError:
                    roc_auc = np.nan

                print(f"  Val accuracy:   {acc:.4f}")
                print(f"  Val precision:  {prec:.4f}")
                print(f"  Val recall:     {rec:.4f}")
                print(f"  Val F1:         {f1:.4f}")
                print(f"  Val ROC-AUC:    {roc_auc:.4f}")
                print(f"  Train time (s): {train_time:.2f}")

                records.append(
                    {
                        "n_estimators": n_estimators,
                        "learning_rate": lr,
                        "max_depth": max_depth,
                        "val_accuracy": acc,
                        "val_precision": prec,
                        "val_recall": rec,
                        "val_f1": f1,
                        "val_roc_auc": roc_auc,
                        "train_time_sec": train_time,
                    }
                )

    results_df = pd.DataFrame(records)
    results_path = os.path.join(ADABOOST_OUTPUT_DIR, "adaboost_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved all config results to: {results_path}")

    # Choose best config by validation ROC-AUC, then F1 as a tiebreaker
    results_df_sorted = results_df.sort_values(by=["val_roc_auc", "val_f1"], ascending=[False, False])
    best_row = results_df_sorted.iloc[0]

    best_config = {
        "n_estimators": int(best_row["n_estimators"]),
        "learning_rate": float(best_row["learning_rate"]),
        "max_depth": int(best_row["max_depth"]),
    }

    print("\nBest config (by validation ROC-AUC):")
    print(best_config)
    print(
        f"  Val ROC-AUC={best_row['val_roc_auc']:.4f}, "
        f"Val F1={best_row['val_f1']:.4f}, "
        f"Val Accuracy={best_row['val_accuracy']:.4f}"
    )

    summary_txt_path = os.path.join(ADABOOST_OUTPUT_DIR, "hyperparameter_grid_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write("AdaBoost Hyperparameter Grid Search Results\n")
        f.write("="*70 + "\n\n")

        for _, row in results_df_sorted.iterrows():
            is_best = (
                row["n_estimators"] == best_config["n_estimators"] and
                row["learning_rate"] == best_config["learning_rate"] and
                row["max_depth"] == best_config["max_depth"]
            )

            marker = "  <--- BEST CONFIG" if is_best else ""
            f.write(
                f"n_estimators={row['n_estimators']}, "
                f"learning_rate={row['learning_rate']}, "
                f"max_depth={row['max_depth']} | "
                f"ROC-AUC={row['val_roc_auc']:.4f}, "
                f"F1={row['val_f1']:.4f}{marker}\n"
            )

    print(f"Saved readable grid summary to: {summary_txt_path}")

    return results_df, best_config

def train_eval_gradientboost_grid(X_train, y_train, X_val, y_val, random_state=42):
    """
    Run a manual grid search over Gradient Boosting hyperparameters.
    Returns:
      - results_df: DataFrame with metrics per config
      - best_config: dict with best hyperparameters based on validation ROC-AUC
    Output file: results_classification/gradientboost_results.csv
    """
    print("=" * 80)
    print("GRADIENT BOOSTING HYPERPARAMETER EXPLORATION")
    print("=" * 80)

    # Hyperparameter grid (keep modest; GB is slower than AdaBoost)
    n_estimators_list = [50, 100, 200]
    learning_rates = [0.05, 0.1, 0.2]
    max_depths = [2, 3]  # depth of individual trees

    records = []

    total_configs = len(n_estimators_list) * len(learning_rates) * len(max_depths)
    config_idx = 0

    for max_depth in max_depths:
        for lr in learning_rates:
            for n_estimators in n_estimators_list:
                config_idx += 1
                print(
                    f"\n[GB] Config {config_idx}/{total_configs}: "
                    f"n_estimators={n_estimators}, learning_rate={lr}, max_depth={max_depth}"
                )

                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=lr,
                    max_depth=max_depth,
                    random_state=random_state,
                )

                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time

                # Validation predictions
                val_pred = model.predict(X_val)
                val_proba = model.predict_proba(X_val)[:, 1]

                # Metrics (class 1 = diabetes)
                acc = accuracy_score(y_val, val_pred)
                prec = precision_score(y_val, val_pred, zero_division=0)
                rec = recall_score(y_val, val_pred, zero_division=0)
                f1 = f1_score(y_val, val_pred, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_val, val_proba)
                except ValueError:
                    roc_auc = np.nan

                print(f"  [GB] Val accuracy:   {acc:.4f}")
                print(f"  [GB] Val precision:  {prec:.4f}")
                print(f"  [GB] Val recall:     {rec:.4f}")
                print(f"  [GB] Val F1:         {f1:.4f}")
                print(f"  [GB] Val ROC-AUC:    {roc_auc:.4f}")
                print(f"  [GB] Train time (s): {train_time:.2f}")

                records.append(
                    {
                        "n_estimators": n_estimators,
                        "learning_rate": lr,
                        "max_depth": max_depth,
                        "val_accuracy": acc,
                        "val_precision": prec,
                        "val_recall": rec,
                        "val_f1": f1,
                        "val_roc_auc": roc_auc,
                        "train_time_sec": train_time,
                    }
                )

    results_df = pd.DataFrame(records)
    results_path = os.path.join(GRADBOOST_OUTPUT_DIR, "gradientboost_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[GB] Saved all config results to: {results_path}")

    # Choose best config by validation ROC-AUC, then F1 as tiebreaker
    results_df_sorted = results_df.sort_values(
        by=["val_roc_auc", "val_f1"], ascending=[False, False]
    )
    best_row = results_df_sorted.iloc[0]

    best_config = {
        "n_estimators": int(best_row["n_estimators"]),
        "learning_rate": float(best_row["learning_rate"]),
        "max_depth": int(best_row["max_depth"]),
    }

    print("\n[GB] Best config (by validation ROC-AUC):")
    print(best_config)
    print(
        f"  [GB] Val ROC-AUC={best_row['val_roc_auc']:.4f}, "
        f"Val F1={best_row['val_f1']:.4f}, "
        f"Val Accuracy={best_row['val_accuracy']:.4f}"
    )

    summary_txt_path = os.path.join(GRADBOOST_OUTPUT_DIR, "gradientboost_hyperparameter_grid_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write("Gradient Boosting Hyperparameter Grid Search Results\n")
        f.write("=" * 70 + "\n\n")

        for _, row in results_df_sorted.iterrows():
            is_best = (
                row["n_estimators"] == best_config["n_estimators"]
                and row["learning_rate"] == best_config["learning_rate"]
                and row["max_depth"] == best_config["max_depth"]
            )

            marker = "  <--- BEST CONFIG" if is_best else ""
            f.write(
                f"n_estimators={row['n_estimators']}, "
                f"learning_rate={row['learning_rate']}, "
                f"max_depth={row['max_depth']} | "
                f"ROC-AUC={row['val_roc_auc']:.4f}, "
                f"F1={row['val_f1']:.4f}{marker}\n"
            )

    print(f"[GB] Saved readable grid summary to: {summary_txt_path}")

    return results_df, best_config

# ---------------------------------------------------------------------
# 4. FINAL MODEL TRAINING & TEST EVALUATION
# ---------------------------------------------------------------------

def train_final_model_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_config, random_state=42):
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 80)

    # Combine train + val
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    base_tree = DecisionTreeClassifier(max_depth=best_config["max_depth"], class_weight="balanced", random_state=random_state)
    model = AdaBoostClassifier(estimator=base_tree, n_estimators=best_config["n_estimators"], learning_rate=best_config["learning_rate"], random_state=random_state)

    print("Fitting final model on train+val...")
    start_time = time.time()
    model.fit(X_train_full, y_train_full)
    train_time = time.time() - start_time
    print(f"Final training time: {train_time:.2f} seconds")

    # Test predictions
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    test_acc = accuracy_score(y_test, test_pred)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, test_proba)

    alt_threshold = 0.30  # you can later tune this
    test_pred_alt = (test_proba >= alt_threshold).astype(int)

    alt_acc = accuracy_score(y_test, test_pred_alt)
    alt_prec = precision_score(y_test, test_pred_alt, zero_division=0)
    alt_rec = recall_score(y_test, test_pred_alt, zero_division=0)
    alt_f1 = f1_score(y_test, test_pred_alt, zero_division=0)

    print(f"\nALTERNATIVE THRESHOLD METRICS (threshold={alt_threshold:.2f}):")
    print(f"  Accuracy: {alt_acc:.4f}")
    print(f"  Precision (class 1): {alt_prec:.4f}")
    print(f"  Recall (class 1):    {alt_rec:.4f}")
    print(f"  F1 (class 1):        {alt_f1:.4f}")

    print("\nTEST METRICS (final model):")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision (class 1): {test_prec:.4f}")
    print(f"  Recall (class 1):    {test_rec:.4f}")
    print(f"  F1 (class 1):        {test_f1:.4f}")
    print(f"  ROC-AUC:             {test_roc_auc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, test_pred, digits=4))

    # ------------------------------
    # Model size / "number of parameters"
    # ------------------------------
    base_estimators = model.estimators_
    n_trees = len(base_estimators)
    total_nodes = sum(est.tree_.node_count for est in base_estimators)
    total_leaves = sum(
        np.count_nonzero(est.tree_.children_left == -1)
        for est in base_estimators
    )

    print("\nMODEL SIZE:")
    print(f"  Number of trees:      {n_trees}")
    print(f"  Total tree nodes:     {total_nodes}")
    print(f"  Total leaf nodes:     {total_leaves}")

    # ------------------------------
    # Feature importance ("which features are relevant?")
    # ------------------------------
    importances = model.feature_importances_
    feature_names = X_train.columns  # X was a pandas DataFrame

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    # Save full table
    fi_path = os.path.join(ADABOOST_OUTPUT_DIR, "feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"Saved feature importances to: {fi_path}")

    # Plot top 15 features
    top_k = 15
    fi_top = fi_df.head(top_k).sort_values("importance")  # for nice horizontal bar

    plt.figure(figsize=(8, 6))
    plt.barh(fi_top["feature"], fi_top["importance"])
    plt.xlabel("Importance")
    plt.title("AdaBoost Feature Importances (top 15)")
    plt.tight_layout()
    fi_plot_path = os.path.join(ADABOOST_OUTPUT_DIR, "adaboost_feature_importances.png")
    plt.savefig(fi_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot: {fi_plot_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    plot_confusion_matrix(cm, title="AdaBoost - Confusion Matrix", filename="adaboost_confusion_matrix.png", output_dir=ADABOOST_OUTPUT_DIR)

    # ROC curve
    plot_roc_curve(y_test, test_proba, title="AdaBoost - ROC Curve", filename="adaboost_roc_curve.png", output_dir=ADABOOST_OUTPUT_DIR)

    # Save summary to text file
    summary_path = os.path.join(ADABOOST_OUTPUT_DIR, "adaboost_best_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Best AdaBoost configuration (by validation ROC-AUC):\n")
        for k, v in best_config.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nTest metrics (final model):\n")
        f.write(f"  Accuracy: {test_acc:.4f}\n")
        f.write(f"  Precision (class 1): {test_prec:.4f}\n")
        f.write(f"  Recall (class 1):    {test_rec:.4f}\n")
        f.write(f"  F1 (class 1):        {test_f1:.4f}\n")
        f.write(f"  ROC-AUC:             {test_roc_auc:.4f}\n")

        f.write("\nTraining time (seconds): {:.2f}\n".format(train_time))

        f.write("\nModel size:\n")
        f.write(f"  Number of trees:      {n_trees}\n")
        f.write(f"  Total tree nodes:     {total_nodes}\n")
        f.write(f"  Total leaf nodes:     {total_leaves}\n")

    print(f"\nSaved summary to: {summary_path}")

    metrics = {
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "alt_threshold": alt_threshold,
        "alt_accuracy": alt_acc,
        "alt_precision": alt_prec,
        "alt_recall": alt_rec,
        "alt_f1": alt_f1,
        "train_time": train_time,
        "n_trees": n_trees,
        "total_nodes": total_nodes,
        "total_leaves": total_leaves,
    }

    return model, metrics

def train_final_gradientboost_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_config, random_state=42):
    """
    Train final Gradient Boosting model on train+val using best_config,
    evaluate on test set, write result files, and return (model, metrics_dict).
    """
    print("\n" + "=" * 80)
    print("TRAINING FINAL GRADIENT BOOSTING MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 80)

    # Combine train + val
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    model = GradientBoostingClassifier(
        n_estimators=best_config["n_estimators"],
        learning_rate=best_config["learning_rate"],
        max_depth=best_config["max_depth"],
        random_state=random_state,
    )

    print("[GB] Fitting final model on train+val...")
    start_time = time.time()
    model.fit(X_train_full, y_train_full)
    train_time = time.time() - start_time
    print(f"[GB] Final training time: {train_time:.2f} seconds")

    # Test predictions
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    # Standard metrics
    test_acc = accuracy_score(y_test, test_pred)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, test_proba)

    # Alternative threshold (same as AdaBoost for fair comparison)
    alt_threshold = 0.30
    test_pred_alt = (test_proba >= alt_threshold).astype(int)

    alt_acc = accuracy_score(y_test, test_pred_alt)
    alt_prec = precision_score(y_test, test_pred_alt, zero_division=0)
    alt_rec = recall_score(y_test, test_pred_alt, zero_division=0)
    alt_f1 = f1_score(y_test, test_pred_alt, zero_division=0)

    print(f"\n[GB] ALTERNATIVE THRESHOLD METRICS (threshold={alt_threshold:.2f}):")
    print(f"  Accuracy: {alt_acc:.4f}")
    print(f"  Precision (class 1): {alt_prec:.4f}")
    print(f"  Recall (class 1):    {alt_rec:.4f}")
    print(f"  F1 (class 1):        {alt_f1:.4f}")

    print("\n[GB] TEST METRICS (final model):")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision (class 1): {test_prec:.4f}")
    print(f"  Recall (class 1):    {test_rec:.4f}")
    print(f"  F1 (class 1):        {test_f1:.4f}")
    print(f"  ROC-AUC:             {test_roc_auc:.4f}")

    print("\n[GB] Classification report:")
    print(classification_report(y_test, test_pred, digits=4))

    # Model size
    # estimators_ is an array of shape (n_estimators, 1) for binary problems
    gb_estimators = model.estimators_.ravel()
    n_trees = len(gb_estimators)
    total_nodes = sum(est.tree_.node_count for est in gb_estimators)
    total_leaves = sum(
        np.count_nonzero(est.tree_.children_left == -1)
        for est in gb_estimators
    )

    print("\n[GB] MODEL SIZE:")
    print(f"  Number of trees:      {n_trees}")
    print(f"  Total tree nodes:     {total_nodes}")
    print(f"  Total leaf nodes:     {total_leaves}")

    # Feature importance
    importances = model.feature_importances_
    feature_names = X_train.columns

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    fi_path = os.path.join(GRADBOOST_OUTPUT_DIR, "gradientboost_feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"[GB] Saved feature importances to: {fi_path}")

    # Plot top 15 features
    top_k = 15
    fi_top = fi_df.head(top_k).sort_values("importance")

    plt.figure(figsize=(8, 6))
    plt.barh(fi_top["feature"], fi_top["importance"])
    plt.xlabel("Importance")
    plt.title("Gradient Boosting Feature Importances (top 15)")
    plt.tight_layout()
    fi_plot_path = os.path.join(GRADBOOST_OUTPUT_DIR, "gradientboost_feature_importances.png")
    plt.savefig(fi_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[GB] Saved feature importance plot: {fi_plot_path}")

    # Confusion matrix + ROC
    cm = confusion_matrix(y_test, test_pred)
    plot_confusion_matrix(cm, title="Gradient Boosting - Confusion Matrix", filename="gradientboost_confusion_matrix.png", output_dir=GRADBOOST_OUTPUT_DIR)

    plot_roc_curve(y_test, test_proba, title="Gradient Boosting - ROC Curve",filename="gradientboost_roc_curve.png", output_dir=GRADBOOST_OUTPUT_DIR)

    # Summary text file
    summary_path = os.path.join(GRADBOOST_OUTPUT_DIR, "gradientboost_best_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Best Gradient Boosting configuration (by validation ROC-AUC):\n")
        for k, v in best_config.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nTest metrics (final model):\n")
        f.write(f"  Accuracy: {test_acc:.4f}\n")
        f.write(f"  Precision (class 1): {test_prec:.4f}\n")
        f.write(f"  Recall (class 1):    {test_rec:.4f}\n")
        f.write(f"  F1 (class 1):        {test_f1:.4f}\n")
        f.write(f"  ROC-AUC:             {test_roc_auc:.4f}\n")

        f.write("\nTraining time (seconds): {:.2f}\n".format(train_time))

        f.write("\nModel size:\n")
        f.write(f"  Number of trees:      {n_trees}\n")
        f.write(f"  Total tree nodes:     {total_nodes}\n")
        f.write(f"  Total leaf nodes:     {total_leaves}\n")

        f.write("\nAlternative threshold metrics (threshold={:.2f}):\n".format(alt_threshold))
        f.write(f"  Accuracy: {alt_acc:.4f}\n")
        f.write(f"  Precision (class 1): {alt_prec:.4f}\n")
        f.write(f"  Recall (class 1):    {alt_rec:.4f}\n")
        f.write(f"  F1 (class 1):        {alt_f1:.4f}\n")

    print(f"\n[GB] Saved summary to: {summary_path}")

    metrics = {
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "alt_threshold": alt_threshold,
        "alt_accuracy": alt_acc,
        "alt_precision": alt_prec,
        "alt_recall": alt_rec,
        "alt_f1": alt_f1,
        "train_time": train_time,
        "n_trees": n_trees,
        "total_nodes": total_nodes,
        "total_leaves": total_leaves,
    }

    return model, metrics

def kfold_evaluation_adaboost(X, y, best_config, random_state=42, n_splits=5):
    """Run K-fold CV on the whole train+val set with the best hyperparameters."""
    print("\n" + "=" * 80)
    print(f"K-FOLD CROSS-VALIDATION ADA BOOST (n_splits={n_splits})")
    print("=" * 80)

    base_tree = DecisionTreeClassifier(
        max_depth=best_config["max_depth"],
        random_state=random_state,
    )

    model = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=best_config["n_estimators"],
        learning_rate=best_config["learning_rate"],
        random_state=random_state,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Accuracy
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    # ROC-AUC
    roc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"CV accuracy: mean={acc_scores.mean():.4f}, std={acc_scores.std():.4f}")
    print(f"CV ROC-AUC:  mean={roc_scores.mean():.4f}, std={roc_scores.std():.4f}")

    # Save to a small text file
    cv_path = os.path.join(ADABOOST_OUTPUT_DIR, "adaboost_kfold_results.txt")
    with open(cv_path, "w") as f:
        f.write(f"{n_splits}-fold cross-validation results\n")
        f.write(f"Best config: {best_config}\n\n")
        f.write(f"Accuracy: mean={acc_scores.mean():.4f}, std={acc_scores.std():.4f}\n")
        f.write(f"ROC-AUC:  mean={roc_scores.mean():.4f}, std={roc_scores.std():.4f}\n")
    print(f"Saved K-fold CV results to: {cv_path}")

def kfold_evaluation_gradientboost(X, y, best_config, random_state=42, n_splits=5):
    """Run K-fold CV on the whole train+val set with the best GB hyperparameters."""
    print("\n" + "=" * 80)
    print(f"K-FOLD CROSS-VALIDATION GRADIENT BOOST (n_splits={n_splits})")
    print("=" * 80)

    model = GradientBoostingClassifier(
        n_estimators=best_config["n_estimators"],
        learning_rate=best_config["learning_rate"],
        max_depth=best_config["max_depth"],
        random_state=random_state,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    roc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"[GB] CV accuracy: mean={acc_scores.mean():.4f}, std={acc_scores.std():.4f}")
    print(f"[GB] CV ROC-AUC:  mean={roc_scores.mean():.4f}, std={roc_scores.std():.4f}")

    cv_path = os.path.join(GRADBOOST_OUTPUT_DIR, "gradientboost_kfold_results.txt")
    with open(cv_path, "w") as f:
        f.write(f"{n_splits}-fold cross-validation results (Gradient Boosting)\n")
        f.write(f"Best config: {best_config}\n\n")
        f.write(f"Accuracy: mean={acc_scores.mean():.4f}, std={acc_scores.std():.4f}\n")
        f.write(f"ROC-AUC:  mean={roc_scores.mean():.4f}, std={roc_scores.std():.4f}\n")
    print(f"[GB] Saved K-fold CV results to: {cv_path}")

def train_final_baseline_tree_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    max_depth=4,
    random_state=42,
):
    """
    Train a single DecisionTreeClassifier as a baseline model (no boosting),
    evaluate on the test set, save its own summary/plots, and return (model, metrics_dict).
    This is used just for comparison with AdaBoost and Gradient Boosting.
    """
    print("\n" + "=" * 80)
    print("TRAINING BASELINE DECISION TREE MODEL")
    print("=" * 80)

    # Combine train + val (same pattern as boosted models)
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )

    print(f"[Tree] Fitting baseline tree (max_depth={max_depth}) on train+val...")
    start_time = time.time()
    model.fit(X_train_full, y_train_full)
    train_time = time.time() - start_time
    print(f"[Tree] Training time: {train_time:.2f} seconds")

    # Test predictions
    test_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        test_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: use decision_function + sigmoid-ish transform if needed
        # but for DecisionTreeClassifier, predict_proba exists, so this is just safety.
        test_proba = test_pred.astype(float)

    # Standard metrics at default threshold 0.5
    test_acc = accuracy_score(y_test, test_pred)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, test_proba)

    print("\n[Tree] TEST METRICS (baseline tree):")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision (class 1): {test_prec:.4f}")
    print(f"  Recall (class 1):    {test_rec:.4f}")
    print(f"  F1 (class 1):        {test_f1:.4f}")
    print(f"  ROC-AUC:             {test_roc_auc:.4f}")

    print("\n[Tree] Classification report:")
    print(classification_report(y_test, test_pred, digits=4))

    # Model size info
    tree_ = model.tree_
    n_trees = 1
    total_nodes = tree_.node_count
    total_leaves = np.count_nonzero(tree_.children_left == -1)

    print("\n[Tree] MODEL SIZE:")
    print(f"  Number of trees:      {n_trees}")
    print(f"  Total tree nodes:     {total_nodes}")
    print(f"  Total leaf nodes:     {total_leaves}")

    # Feature importances
    importances = model.feature_importances_
    feature_names = X_train.columns

    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    fi_path = os.path.join(TREE_OUTPUT_DIR, "tree_feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"[Tree] Saved feature importances to: {fi_path}")

    # Plot top 15 features
    top_k = 15
    fi_top = fi_df.head(top_k).sort_values("importance")

    plt.figure(figsize=(8, 6))
    plt.barh(fi_top["feature"], fi_top["importance"])
    plt.xlabel("Importance")
    plt.title("Baseline Decision Tree Feature Importances (top 15)")
    plt.tight_layout()
    fi_plot_path = os.path.join(TREE_OUTPUT_DIR, "tree_feature_importances.png")
    plt.savefig(fi_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Tree] Saved feature importance plot: {fi_plot_path}")

    # Confusion matrix + ROC
    cm = confusion_matrix(y_test, test_pred)
    plot_confusion_matrix(
        cm,
        title="Baseline Decision Tree - Confusion Matrix",
        filename="tree_confusion_matrix.png",
        output_dir=TREE_OUTPUT_DIR,
    )

    plot_roc_curve(
        y_test,
        test_proba,
        title="Baseline Decision Tree - ROC Curve",
        filename="tree_roc_curve.png",
        output_dir=TREE_OUTPUT_DIR,
    )

    # Save summary
    summary_path = os.path.join(TREE_OUTPUT_DIR, "tree_baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Baseline Decision Tree configuration:\n")
        f.write(f"  max_depth: {max_depth}\n\n")

        f.write("Test metrics (baseline tree):\n")
        f.write(f"  Accuracy: {test_acc:.4f}\n")
        f.write(f"  Precision (class 1): {test_prec:.4f}\n")
        f.write(f"  Recall (class 1):    {test_rec:.4f}\n")
        f.write(f"  F1 (class 1):        {test_f1:.4f}\n")
        f.write(f"  ROC-AUC:             {test_roc_auc:.4f}\n")

        f.write("\nTraining time (seconds): {:.2f}\n".format(train_time))

        f.write("\nModel size:\n")
        f.write(f"  Number of trees:      {n_trees}\n")
        f.write(f"  Total tree nodes:     {total_nodes}\n")
        f.write(f"  Total leaf nodes:     {total_leaves}\n")

    print(f"[Tree] Saved baseline tree summary to: {summary_path}")

    metrics = {
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "train_time": train_time,
        "n_trees": n_trees,
        "total_nodes": total_nodes,
        "total_leaves": total_leaves,
    }

    return model, metrics


# ---------------------------------------------------------------------
# 5. PLOTS
# ---------------------------------------------------------------------

def plot_confusion_matrix(cm, title, filename, output_dir):
    """Plot and save confusion matrix as heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot: {out_path}")

def plot_roc_curve(y_true, y_proba, title, filename, output_dir):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve plot: {out_path}")

def plot_accuracy_vs_trees(results_df, best_config, output_dir, model_name, filename):
    """Plot validation and (if available) test accuracy vs number of trees for best depth & lr."""
    # Filter to best learning_rate and max_depth
    mask = ((results_df["learning_rate"] == best_config["learning_rate"]) & (results_df["max_depth"] == best_config["max_depth"]))
    df_best_line = results_df[mask].copy().sort_values("n_estimators")

    plt.figure(figsize=(8, 6))
    plt.plot(df_best_line["n_estimators"], df_best_line["val_accuracy"], marker="o", label="Validation accuracy")
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name}: Accuracy vs Trees\n"f"(lr={best_config['learning_rate']}, max_depth={best_config['max_depth']})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy vs trees plot: {out_path}")


# ---------------------------------------------------------------------
# 6. HELPERS
# ---------------------------------------------------------------------

def save_model_comparison_summary(adaboost_metrics, gradientboost_metrics, tree_metrics, output_dir):
    """
    Save a side-by-side comparison of:
      - Baseline Decision Tree
      - AdaBoost
      - Gradient Boosting
    to results_classification/model_comparison_summary.txt
    """
    out_path = os.path.join(output_dir, "model_comparison_summary.txt")
    with open(out_path, "w") as f:
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("Baseline Decision Tree test metrics:\n")
        f.write(f"  Accuracy: {tree_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Precision (class 1): {tree_metrics['test_precision']:.4f}\n")
        f.write(f"  Recall (class 1):    {tree_metrics['test_recall']:.4f}\n")
        f.write(f"  F1 (class 1):        {tree_metrics['test_f1']:.4f}\n")
        f.write(f"  ROC-AUC:             {tree_metrics['test_roc_auc']:.4f}\n")
        f.write(f"  Train time (s):      {tree_metrics['train_time']:.2f}\n")
        f.write(
            f"  Trees: {tree_metrics['n_trees']}, "
            f"nodes: {tree_metrics['total_nodes']}, "
            f"leaves: {tree_metrics['total_leaves']}\n\n"
        )

        f.write("AdaBoost test metrics:\n")
        f.write(f"  Accuracy: {adaboost_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Precision (class 1): {adaboost_metrics['test_precision']:.4f}\n")
        f.write(f"  Recall (class 1):    {adaboost_metrics['test_recall']:.4f}\n")
        f.write(f"  F1 (class 1):        {adaboost_metrics['test_f1']:.4f}\n")
        f.write(f"  ROC-AUC:             {adaboost_metrics['test_roc_auc']:.4f}\n")
        f.write(f"  Train time (s):      {adaboost_metrics['train_time']:.2f}\n")
        f.write(
            f"  Trees: {adaboost_metrics['n_trees']}, "
            f"nodes: {adaboost_metrics['total_nodes']}, "
            f"leaves: {adaboost_metrics['total_leaves']}\n\n"
        )

        f.write("Gradient Boosting test metrics:\n")
        f.write(f"  Accuracy: {gradientboost_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Precision (class 1): {gradientboost_metrics['test_precision']:.4f}\n")
        f.write(f"  Recall (class 1):    {gradientboost_metrics['test_recall']:.4f}\n")
        f.write(f"  F1 (class 1):        {gradientboost_metrics['test_f1']:.4f}\n")
        f.write(f"  ROC-AUC:             {gradientboost_metrics['test_roc_auc']:.4f}\n")
        f.write(f"  Train time (s):      {gradientboost_metrics['train_time']:.2f}\n")
        f.write(
            f"  Trees: {gradientboost_metrics['n_trees']}, "
            f"nodes: {gradientboost_metrics['total_nodes']}, "
            f"leaves: {gradientboost_metrics['total_leaves']}\n\n"
        )

        f.write("Notes:\n")
        f.write("- All three models use the same train/val/test split and metrics.\n")
        f.write("- The baseline decision tree shows performance without boosting.\n")
        f.write("- ROC-AUC is the main ranking metric; F1 highlights performance on class 1.\n")

    print(f"Saved Decision Tree vs AdaBoost vs Gradient Boosting comparison to: {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load data
    X, y, target_col = load_cdc_diabetes_data()

    # 2. Split into train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ------------------------------------------------------------------
    # AdaBoost pipeline
    # ------------------------------------------------------------------
    results_df_ada, best_config_ada = train_eval_adaboost_grid(X_train, y_train, X_val, y_val)

    # K-fold CV (AdaBoost) on train+val with best config
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    kfold_evaluation_adaboost(X_train_val, y_train_val, best_config_ada)

    # Plot accuracy vs trees (AdaBoost)
    plot_accuracy_vs_trees(results_df_ada, best_config_ada, model_name="Ada Boost", filename="adaboost_accuracy_vs_trees.png", output_dir=ADABOOST_OUTPUT_DIR)

    # Train final AdaBoost model and evaluate on test set
    final_adaboost_model, adaboost_metrics = train_final_model_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_config_ada)

    # ------------------------------------------------------------------
    # Gradient Boosting pipeline
    # ------------------------------------------------------------------
    results_df_gb, best_config_gb = train_eval_gradientboost_grid(X_train, y_train, X_val, y_val)

    # K-fold CV (Gradient Boosting) on train+val with best config
    kfold_evaluation_gradientboost(X_train_val, y_train_val, best_config_gb)

    # Plot accuracy vs trees (Gradient Boosting)
    plot_accuracy_vs_trees(results_df_gb, best_config_gb, model_name="Gradient Boost", filename="gradientboost_accuracy_vs_trees.png", output_dir=GRADBOOST_OUTPUT_DIR)

    # Train final Gradient Boosting model and evaluate on test set
    final_gb_model, gradientboost_metrics = train_final_gradientboost_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_config_gb)

    baseline_tree_model, tree_metrics = train_final_baseline_tree_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, max_depth=4)
    print(tree_metrics)

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    save_model_comparison_summary(adaboost_metrics, gradientboost_metrics, tree_metrics, OUTPUT_DIR)

    print("\nDone. Check the 'results_classification/' folder for plots and result files.")