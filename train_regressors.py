"""Train baseline regressors on Manhattan datasets.

Loads datasets from `avash-part3-reg-eda.py` (datasets 3/4 + IQR), then trains:
- Linear Regression baseline
- AdaBoost Regressor

Runs on:
- df_manhattan_clean (dataset 4: fares >= $1)
- df_manhattan_iqr (IQR filtered)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_eda_module() -> object:
    """Load `avash-part3-reg-eda.py` as a module (file name has hyphens)."""
    eda_path = Path(__file__).with_name("avash-part3-reg-eda.py")
    spec = importlib.util.spec_from_file_location("uber_eda", eda_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {eda_path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure the module is registered so dataclasses/type resolution works.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[ColumnTransformer, list[str]]:
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]

    categorical_cols = [
        c for c in feature_cols
        if (isinstance(X[c].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(X[c]))
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor, feature_cols


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Some sklearn versions don't support `squared=...` keyword, so compute explicitly.
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_report(df: pd.DataFrame, dataset_name: str, random_state: int = 42) -> None:
    target_col = "fare_amount"
    df = df.dropna(subset=[target_col]).copy()

    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    preprocessor, _ = make_preprocessor(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    models = {
        "LinearRegression": LinearRegression(),
        "AdaBoostRegressor": AdaBoostRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            loss="square",
        ),
    }

    print("=" * 70)
    print(f"DATASET: {dataset_name}  |  rows={len(df):,}  features={X.shape[1]}")
    print("=" * 70)

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate(y_test, preds)

        print(f"\nModel: {name}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  R2:   {metrics['R2']:.4f}")

        # Persist the full pipeline (preprocessing + model) for reuse (e.g., in Colab)
        models_dir = Path(__file__).with_name("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        safe_ds = dataset_name.split()[0].replace("(", "").replace(")", "").replace(":", "").strip()
        out_path = models_dir / f"{name}_{safe_ds}.joblib"
        joblib.dump(pipe, out_path)
        print(f"  Saved model pipeline -> {out_path}")


def main() -> None:
    eda = load_eda_module()
    ds = eda.build_datasets("data/uber.csv")

    # Dataset 4
    train_and_report(ds.df_manhattan_clean, "df_manhattan_clean (fares >= $1)")

    # IQR dataset
    train_and_report(ds.df_manhattan_iqr, "df_manhattan_iqr (IQR-filtered fares)")


if __name__ == "__main__":
    main()
