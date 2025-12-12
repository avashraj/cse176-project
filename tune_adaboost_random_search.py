"""Random search hyperparameter tuning (Colab-friendly).

Inputs:
- df_manhattan_iqr.csv (preferred, in current working directory e.g. /content in Colab)
- OR data/df_manhattan_iqr.csv (exported from avash-part3-reg-eda.py)
- AdaBoostRegressor_*.joblib and LinearRegression_*.joblib (optional) baseline saved pipelines
  Can be in the current directory OR in models/

What it does:
- Loads the dataset
- Creates a train/test split
- (Optional) loads baseline pipelines (LinearRegression + AdaBoost) and evaluates them on the same test split
- Runs RandomizedSearchCV for AdaBoostRegressor with a small, high-signal search space
- Saves the tuned pipeline to models/AdaBoostRegressor_df_manhattan_iqr_tuned.joblib

Why these parameters:
- n_estimators: number of weak learners. More can reduce bias but increases runtime and can overfit.
- learning_rate: shrinkage applied to each learner’s contribution. Smaller values often generalize better but need more estimators.
- base tree max_depth: controls weak-learner complexity (interaction capacity). Too deep overfits, too shallow underfits.
- loss: affects robustness to outliers (even after IQR, tails can remain).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def report(name: str, y_true, y_pred) -> None:
    print(f"\n{name}")
    print(f"  MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"  R2:   {r2_score(y_true, y_pred):.4f}")


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    # Explicitly treat these as categorical (CSV won’t preserve pandas Categorical dtype)
    categorical_cols = [c for c in ["time_bin", "day_of_week", "month", "day"] if c in feature_df.columns]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def load_baseline(filename: str):
    # In Colab users often upload directly to /content (cwd).
    # Support both cwd and ./models/
    candidates = [
        Path(filename),
        Path("models") / filename,
    ]
    found = first_existing(candidates)
    if found is None:
        return None
    return joblib.load(found)


def main() -> None:
    # Prefer reading directly from cwd (Colab uploads land in /content).
    data_path = first_existing(
        [
            Path("df_manhattan_iqr.csv"),
            Path("data") / "df_manhattan_iqr.csv",
            Path("/content/df_manhattan_iqr.csv"),
        ]
    )
    if data_path is None:
        raise FileNotFoundError(
            "Could not find `df_manhattan_iqr.csv`. Put it in the current directory "
            "(e.g. /content in Colab) or in ./data/ as data/df_manhattan_iqr.csv."
        )

    df = pd.read_csv(data_path)
    if "fare_amount" not in df.columns:
        raise ValueError("Expected target column `fare_amount` in CSV")

    df = df.dropna(subset=["fare_amount"]).copy()

    y = df["fare_amount"].to_numpy()
    X = df.drop(columns=["fare_amount"])  # keep all engineered features

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pre = build_preprocessor(X)

    # Optional: evaluate saved baselines (if you copied them into Colab)
    lin = load_baseline("LinearRegression_df_manhattan_iqr.joblib")
    if lin is not None:
        report("Baseline LinearRegression (loaded)", y_test, lin.predict(X_test))

    ada_base = load_baseline("AdaBoostRegressor_df_manhattan_iqr.joblib")
    if ada_base is not None:
        report("Baseline AdaBoostRegressor (loaded)", y_test, ada_base.predict(X_test))

    # Build AdaBoost with an explicit base estimator
    base_tree = DecisionTreeRegressor(random_state=42)

    try:
        ada = AdaBoostRegressor(estimator=base_tree, random_state=42)
        depth_param = "model__estimator__max_depth"
    except TypeError:
        # Older sklearn
        ada = AdaBoostRegressor(base_estimator=base_tree, random_state=42)
        depth_param = "model__base_estimator__max_depth"

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", ada),
    ])

    # Small, high-impact search space
    # Note: scipy is available in Colab by default.
    from scipy.stats import randint, loguniform

    param_distributions = {
        "model__n_estimators": randint(100, 800),
        "model__learning_rate": loguniform(1e-3, 1.0),
        depth_param: randint(1, 7),  # keep trees shallow-ish
        "model__loss": ["linear", "square", "exponential"],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print("\nBest params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    best_pipe = search.best_estimator_
    report("Tuned AdaBoostRegressor", y_test, best_pipe.predict(X_test))

    # Save in ./models if present/desired, else save to cwd.
    models_dir = Path("models")
    out_path = (
        (models_dir / "AdaBoostRegressor_df_manhattan_iqr_tuned.joblib")
        if models_dir.exists()
        else Path("AdaBoostRegressor_df_manhattan_iqr_tuned.joblib")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, out_path)
    print(f"\nSaved tuned pipeline -> {out_path}")


if __name__ == "__main__":
    main()
