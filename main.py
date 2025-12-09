from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

DATA_PATH = Path(__file__).resolve().parent / "diabetes_012_health_indicators_BRFSS2015.csv"


def get_data(data_path: Path = DATA_PATH):
    """
    Load the CDC Diabetes Health Indicators dataset from a local CSV.

    Args:
        data_path (Path): Path to the CSV file.

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.DataFrame): Target vector
        metadata (dict): Dataset metadata
    """
    print(f"Loading dataset from {data_path} ...")
    df = pd.read_csv(data_path)

    target_col = "Diabetes_012"
    y = df[[target_col]]
    X = df.drop(columns=[target_col])

    metadata = {
        "source": "local CSV",
        "path": str(data_path),
        "n_samples": len(df),
        "n_features": X.shape[1],
    }

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, metadata


# ============================================================================
# BASIC PREPROCESSING FUNCTIONS
# ============================================================================

def handle_missing_values(X, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        X (pd.DataFrame): Feature matrix
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
    
    Returns:
        X_processed (pd.DataFrame): Processed feature matrix
    """
    X_processed = X.copy()
    
    if strategy == 'drop':
        X_processed = X_processed.dropna()
    elif strategy == 'mean':
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
    elif strategy == 'mode':
        for col in X_processed.columns:
            X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0] if len(X_processed[col].mode()) > 0 else 0)
    
    print(f"Missing values handled using '{strategy}' strategy")
    return X_processed


def one_hot_encode(X, columns=None, drop_first=False):
    """
    Perform one-hot encoding on categorical columns.
    
    Args:
        X (pd.DataFrame): Feature matrix
        columns (list): List of column names to encode. If None, encodes all categorical columns
        drop_first (bool): Whether to drop the first category to avoid multicollinearity
    
    Returns:
        X_encoded (pd.DataFrame): One-hot encoded feature matrix
        encoded_columns (list): List of newly created column names
    """
    X_encoded = X.copy()
    
    # Identify categorical columns
    if columns is None:
        categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in X_encoded.columns]
    
    if len(categorical_cols) == 0:
        print("No categorical columns found for one-hot encoding")
        return X_encoded, []
    
    # Perform one-hot encoding
    encoded_columns = []
    for col in categorical_cols:
        dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=drop_first)
        encoded_columns.extend(dummies.columns.tolist())
        X_encoded = pd.concat([X_encoded.drop(columns=[col]), dummies], axis=1)
    
    print(f"One-hot encoded {len(categorical_cols)} categorical columns into {len(encoded_columns)} new features")
    return X_encoded, encoded_columns


def label_encode(X, columns=None):
    """
    Perform label encoding on categorical columns (for ordinal or when categories are too many).
    
    Args:
        X (pd.DataFrame): Feature matrix
        columns (list): List of column names to encode. If None, encodes all categorical columns
    
    Returns:
        X_encoded (pd.DataFrame): Label encoded feature matrix
        encoders (dict): Dictionary of LabelEncoder objects for each column
    """
    X_encoded = X.copy()
    encoders = {}
    
    if columns is None:
        categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_cols = [col for col in columns if col in X_encoded.columns]
    
    if len(categorical_cols) == 0:
        print("No categorical columns found for label encoding")
        return X_encoded, {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le
    
    print(f"Label encoded {len(categorical_cols)} categorical columns")
    return X_encoded, encoders


def remove_duplicates(X, y=None):
    """
    Remove duplicate rows from the dataset.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.DataFrame, optional): Target vector
    
    Returns:
        X_unique (pd.DataFrame): Feature matrix without duplicates
        y_unique (pd.DataFrame, optional): Target vector without duplicates
    """
    initial_shape = X.shape[0]
    X_unique = X.drop_duplicates()
    
    if y is not None:
        # Remove corresponding target rows
        y_unique = y.loc[X_unique.index]
        print(f"Removed {initial_shape - X_unique.shape[0]} duplicate rows")
        return X_unique, y_unique
    
    print(f"Removed {initial_shape - X_unique.shape[0]} duplicate rows")
    return X_unique


# ============================================================================
# DATA TRANSFORMATION FUNCTIONS
# ============================================================================

def standardize_features(X, method='standard'):
    """
    Standardize features to have zero mean and unit variance (or other scaling methods).
    
    Args:
        X (pd.DataFrame): Feature matrix
        method (str): Scaling method ('standard', 'minmax', 'robust')
            - 'standard': Zero mean, unit variance (StandardScaler)
            - 'minmax': Scale to [0, 1] range (MinMaxScaler)
            - 'robust': Use median and IQR (RobustScaler, good for outliers)
    
    Returns:
        X_scaled (pd.DataFrame): Scaled feature matrix
        scaler: Fitted scaler object for inverse transformation
    """
    X_scaled = X.copy()
    numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'standard':
        scaler = StandardScaler()
        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
        print(f"Standardized features (zero mean, unit variance) using StandardScaler")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
        print(f"Normalized features to [0, 1] range using MinMaxScaler")
    elif method == 'robust':
        scaler = RobustScaler()
        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
        print(f"Robust scaled features (median-centered, IQR-scaled) using RobustScaler")
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    return X_scaled, scaler


def zero_mean_normalize(X):
    """
    Normalize features to have zero mean (centering only, no variance scaling).
    
    Args:
        X (pd.DataFrame): Feature matrix
    
    Returns:
        X_normalized (pd.DataFrame): Zero-mean normalized feature matrix
        means (pd.Series): Mean values for each feature (for inverse transformation)
    """
    X_normalized = X.copy()
    numeric_cols = X_normalized.select_dtypes(include=[np.number]).columns.tolist()
    
    means = X_normalized[numeric_cols].mean()
    X_normalized[numeric_cols] = X_normalized[numeric_cols] - means
    
    print(f"Centered features to zero mean")
    return X_normalized, means


def feature_selection_variance(X, threshold=0.0):
    """
    Remove features with low variance (below threshold).
    
    Args:
        X (pd.DataFrame): Feature matrix
        threshold (float): Variance threshold. Features with variance below this are removed.
    
    Returns:
        X_selected (pd.DataFrame): Feature matrix with selected features
        selected_features (list): List of selected feature names
        selector: Fitted VarianceThreshold object
    """
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    print(f"Selected {len(selected_features)} features (removed {X.shape[1] - len(selected_features)} low-variance features)")
    return X_selected, selected_features, selector


def feature_selection_univariate(X, y, k=10, score_func=f_classif):
    """
    Select top k features based on univariate statistical tests.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series or np.array): Target vector
        k (int): Number of top features to select
        score_func: Scoring function (f_classif, mutual_info_classif, etc.)
    
    Returns:
        X_selected (pd.DataFrame): Feature matrix with selected features
        selected_features (list): List of selected feature names
        selector: Fitted SelectKBest object
    """
    # Ensure y is 1D array
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0].values
    
    k = min(k, X.shape[1])  # Don't select more features than available
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    score_func_name = score_func.__name__ if hasattr(score_func, '__name__') else str(score_func)
    print(f"Selected top {k} features using {score_func_name}")
    return X_selected, selected_features, selector


def feature_selection_model_based(X, y, estimator=None, max_features=None, threshold='median'):
    """
    Select features based on importance from a tree-based model.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series or np.array): Target vector
        estimator: Model to use for feature importance. If None, uses RandomForestClassifier
        max_features (int): Maximum number of features to select. If None, uses threshold.
        threshold (str or float): Threshold for feature selection
    
    Returns:
        X_selected (pd.DataFrame): Feature matrix with selected features
        selected_features (list): List of selected feature names
        selector: Fitted SelectFromModel object
    """
    # Ensure y is 1D array
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0].values
    
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    selector = SelectFromModel(estimator, max_features=max_features, threshold=threshold)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    print(f"Selected {len(selected_features)} features using model-based selection ({type(estimator).__name__})")
    return X_selected, selected_features, selector


def feature_selection_correlation(X, threshold=0.95):
    """
    Remove highly correlated features (keep one from each highly correlated pair).
    
    Args:
        X (pd.DataFrame): Feature matrix
        threshold (float): Correlation threshold. Features with correlation above this are removed.
    
    Returns:
        X_selected (pd.DataFrame): Feature matrix with selected features
        selected_features (list): List of selected feature names
        removed_features (list): List of removed feature names
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = X[numeric_cols].corr().abs()
    
    # Find pairs of highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to remove
    to_remove = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    # Keep all non-numeric columns and non-removed numeric columns
    selected_features = [col for col in X.columns if col not in to_remove]
    X_selected = X[selected_features]
    
    print(f"Removed {len(to_remove)} highly correlated features (threshold={threshold})")
    return X_selected, selected_features, to_remove


def apply_pca(X, n_components=None, variance_ratio=0.95):
    """
    Apply Principal Component Analysis for dimensionality reduction.
    
    Args:
        X (pd.DataFrame): Feature matrix
        n_components (int): Number of components. If None, uses variance_ratio.
        variance_ratio (float): Cumulative variance ratio to retain (if n_components is None)
    
    Returns:
        X_pca (pd.DataFrame): Transformed feature matrix
        pca: Fitted PCA object
        explained_variance (np.array): Explained variance ratio
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    if n_components is None:
        # Find number of components that explain variance_ratio of variance
        pca_temp = PCA()
        pca_temp.fit(X_numeric)
        cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= variance_ratio) + 1
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_numeric)
    
    # Create DataFrame with component names
    component_names = [f'PC{i+1}' for i in range(n_components)]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=X.index)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"Applied PCA: {n_components} components explain {explained_variance.sum():.2%} of variance")
    
    return X_pca, pca, explained_variance


def log_transform(X, columns=None):
    """
    Apply log transformation to specified columns (useful for skewed data).
    
    Args:
        X (pd.DataFrame): Feature matrix
        columns (list): List of column names to transform. If None, transforms all positive numeric columns.
    
    Returns:
        X_transformed (pd.DataFrame): Log-transformed feature matrix
    """
    X_transformed = X.copy()
    
    if columns is None:
        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
        # Only transform columns with all positive values
        columns = [col for col in numeric_cols if (X_transformed[col] > 0).all()]
    
    for col in columns:
        if col in X_transformed.columns:
            X_transformed[col] = np.log1p(X_transformed[col])  # log1p handles zeros
    
    print(f"Applied log transformation to {len(columns)} features")
    return X_transformed


def remove_outliers_iqr(X, factor=1.5):
    """
    Remove outliers using Interquartile Range (IQR) method.
    
    Args:
        X (pd.DataFrame): Feature matrix
        factor (float): IQR multiplier for outlier detection
    
    Returns:
        X_clean (pd.DataFrame): Feature matrix without outliers
        outlier_mask (pd.Series): Boolean mask indicating which rows were kept
    """
    X_clean = X.copy()
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_mask = pd.Series([True] * len(X_clean), index=X_clean.index)
    
    for col in numeric_cols:
        Q1 = X_clean[col].quantile(0.25)
        Q3 = X_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Mark outliers
        col_outliers = (X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)
        outlier_mask = outlier_mask & ~col_outliers
    
    X_clean = X_clean[outlier_mask]
    
    n_removed = len(X) - len(X_clean)
    print(f"Removed {n_removed} rows with outliers (IQR factor={factor})")
    
    return X_clean, outlier_mask


# ============================================================================
# PIPELINE FUNCTIONS (Combining multiple preprocessing steps)
# ============================================================================

def preprocessing_pipeline_basic(X, y=None, handle_missing='mean', encode_categorical='onehot', 
                                  drop_first=False, remove_dups=True):
    """
    Basic preprocessing pipeline combining common steps.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.DataFrame, optional): Target vector
        handle_missing (str): Strategy for missing values
        encode_categorical (str): Encoding method ('onehot', 'label', or None)
        drop_first (bool): Whether to drop first category in one-hot encoding
        remove_dups (bool): Whether to remove duplicates
    
    Returns:
        X_processed (pd.DataFrame): Processed feature matrix
        y_processed (pd.DataFrame, optional): Processed target vector
    """
    print("\n" + "="*80)
    print("BASIC PREPROCESSING PIPELINE")
    print("="*80)
    
    X_processed = X.copy()
    y_processed = y.copy() if y is not None else None
    
    # Handle missing values
    if X_processed.isnull().sum().sum() > 0:
        X_processed = handle_missing_values(X_processed, strategy=handle_missing)
    
    # Encode categorical variables
    if encode_categorical == 'onehot':
        X_processed, _ = one_hot_encode(X_processed, drop_first=drop_first)
    elif encode_categorical == 'label':
        X_processed, _ = label_encode(X_processed)
    
    # Remove duplicates
    if remove_dups:
        if y_processed is not None:
            X_processed, y_processed = remove_duplicates(X_processed, y_processed)
        else:
            X_processed = remove_duplicates(X_processed)
    
    print(f"\nFinal shape: {X_processed.shape}")
    return X_processed, y_processed


def preprocessing_pipeline_advanced(X, y=None, scaling='standard', feature_selection=None, 
                                    n_features=None, pca=False, n_components=None):
    """
    Advanced preprocessing pipeline with scaling and feature selection.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.DataFrame, optional): Target vector
        scaling (str): Scaling method ('standard', 'minmax', 'robust', or None)
        feature_selection (str): Feature selection method ('variance', 'univariate', 'model', 'correlation', or None)
        n_features (int): Number of features to select (for univariate/model-based)
        pca (bool): Whether to apply PCA
        n_components (int): Number of PCA components (if pca=True)
    
    Returns:
        X_processed (pd.DataFrame): Processed feature matrix
        transformers (dict): Dictionary of fitted transformers
    """
    print("\n" + "="*80)
    print("ADVANCED PREPROCESSING PIPELINE")
    print("="*80)
    
    X_processed = X.copy()
    transformers = {}
    
    # Scaling
    if scaling:
        X_processed, scaler = standardize_features(X_processed, method=scaling)
        transformers['scaler'] = scaler
    
    # Feature selection
    if feature_selection == 'variance':
        X_processed, _, selector = feature_selection_variance(X_processed, threshold=0.01)
        transformers['feature_selector'] = selector
    elif feature_selection == 'univariate' and y is not None:
        k = n_features if n_features else min(20, X_processed.shape[1])
        X_processed, _, selector = feature_selection_univariate(X_processed, y, k=k)
        transformers['feature_selector'] = selector
    elif feature_selection == 'model' and y is not None:
        max_feat = n_features if n_features else None
        X_processed, _, selector = feature_selection_model_based(X_processed, y, max_features=max_feat)
        transformers['feature_selector'] = selector
    elif feature_selection == 'correlation':
        X_processed, _, _ = feature_selection_correlation(X_processed, threshold=0.95)
    
    # PCA
    if pca:
        X_processed, pca_obj, _ = apply_pca(X_processed, n_components=n_components)
        transformers['pca'] = pca_obj
    
    print(f"\nFinal shape: {X_processed.shape}")
    return X_processed, transformers


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run a single, streamlined preprocessing strategy.
    """
    print("="*80)
    print("DATA PREPROCESSING (SINGLE STRATEGY)")
    print("="*80)
    
    # Load data
    X, y, metadata = get_data()
    print(f"Metadata: {metadata}")

    # Single, streamlined preprocessing strategy:
    #   1) Drop rows with missing values and duplicates
    #   2) Standard scale numeric features
    #   3) Remove near-zero-variance features
    X_basic, y_basic = preprocessing_pipeline_basic(
        X,
        y,
        handle_missing="drop",
        encode_categorical=None,
        remove_dups=True,
    )

    X_scaled, scaler = standardize_features(X_basic, method="standard")
    X_selected, selected_features, _ = feature_selection_variance(
        X_scaled, threshold=0.01
    )

    print("\nFinal shapes after preprocessing:")
    print(f"  Original: {X.shape}")
    print(f"  After basic cleanup: {X_basic.shape}")
    print(f"  After scaling: {X_scaled.shape}")
    print(f"  After variance filtering: {X_selected.shape}")
    print(f"Selected feature count: {len(selected_features)}")

    print("\nSample of processed features:")
    print(X_selected.head())


if __name__ == "__main__":
    main()
