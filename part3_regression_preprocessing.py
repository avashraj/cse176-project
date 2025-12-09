import os
import pandas as pd
import numpy as np
import kagglehub
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ============================================================================
# DATA LOADING
# ============================================================================

def load_uber_data():
    """
    Load the Uber fares dataset from Kaggle.
    
    Returns:
        df (pd.DataFrame): Raw dataset
    """
    print("Downloading dataset...")
    path = kagglehub.dataset_download("yasserh/uber-fares-dataset")
    print(f"Path to dataset files: {path}")
    
    # Find the CSV file in the downloaded directory
    csv_files = list(Path(path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    csv_file = csv_files[0]
    print(f"Loading data from: {csv_file}")
    
    # Load the dataset without key as its not useful for prediction
    df = pd.read_csv(csv_file, index_col=0, encoding='latin-1')
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth using Haversine formula.
    
    Args:
        lat1, lon1: Pickup coordinates
        lat2, lon2: Dropoff coordinates
    
    Returns:
        distance: Distance in kilometers
    """
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def extract_temporal_features(df):
    """
    Extract temporal features from pickup_datetime.
    
    Args:
        df (pd.DataFrame): Dataset with pickup_datetime column
    
    Returns:
        df (pd.DataFrame): Dataset with temporal features added
    """
    if 'pickup_datetime' in df.columns:
        # Convert to datetime
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        
        # Extract temporal features
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day'] = df['pickup_datetime'].dt.day
        df['pickup_month'] = df['pickup_datetime'].dt.month
        df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
        
        print("Extracted temporal features: hour, day, month, dayofweek")
    
    return df


def one_hot_encode_temporal_features(df):
    """
    One-hot encode temporal features (hour, day, month, dayofweek).
    
    Args:
        df (pd.DataFrame): Dataset with temporal feature columns
    
    Returns:
        df (pd.DataFrame): Dataset with one-hot encoded temporal features
    """
    temporal_features = ['pickup_hour', 'pickup_day', 'pickup_month', 'pickup_dayofweek']
    existing_features = [f for f in temporal_features if f in df.columns]
    
    if not existing_features:
        print("No temporal features found to one-hot encode")
        return df
    
    # One-hot encode each temporal feature
    for feature in existing_features:
        # Use get_dummies with prefix to create named columns
        dummies = pd.get_dummies(df[feature], prefix=feature, dtype=int)
        # Drop the original column and add the dummy columns
        df = df.drop(columns=[feature])
        df = pd.concat([df, dummies], axis=1)
    
    print(f"One-hot encoded temporal features: {existing_features}")
    print(f"  Created {sum(len(pd.unique(df.filter(like=f).columns)) for f in existing_features)} new binary columns")
    
    return df


def add_distance_feature(df):
    """
    Calculate trip distance from coordinates.
    
    Args:
        df (pd.DataFrame): Dataset with coordinate columns
    
    Returns:
        df (pd.DataFrame): Dataset with distance_km column added
    """
    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 
                                          'dropoff_latitude', 'dropoff_longitude']):
        df['distance_km'] = calculate_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        print("Added distance_km feature")
    
    return df


# ============================================================================
# DATA CLEANING
# ============================================================================

def remove_invalid_coordinates(df):
    """
    Remove rows with invalid coordinates (outside reasonable bounds for NYC area).
    NYC approximate bounds: lat 40.5-41.0, lon -74.3 to -73.7
    
    Args:
        df (pd.DataFrame): Dataset with coordinate columns
    
    Returns:
        df (pd.DataFrame): Dataset with invalid coordinates removed
    """
    initial_shape = df.shape[0]
    
    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 
                                          'dropoff_latitude', 'dropoff_longitude']):
        # Remove rows with invalid coordinates (0, 0 or outside reasonable bounds)
        valid_mask = (
            (df['pickup_latitude'] != 0) & (df['pickup_longitude'] != 0) &
            (df['dropoff_latitude'] != 0) & (df['dropoff_longitude'] != 0) &
            (df['pickup_latitude'].between(-90, 90)) & 
            (df['pickup_longitude'].between(-180, 180)) &
            (df['dropoff_latitude'].between(-90, 90)) & 
            (df['dropoff_longitude'].between(-180, 180))
        )
        df = df[valid_mask].copy()
        
        removed = initial_shape - df.shape[0]
        print(f"Removed {removed} rows with invalid coordinates ({removed/initial_shape*100:.2f}%)")
    
    return df


def remove_invalid_fares(df, target_col='fare_amount', min_fare=0, max_fare=500):
    """
    Remove rows with invalid fare amounts.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
        min_fare (float): Minimum valid fare
        max_fare (float): Maximum valid fare
    
    Returns:
        df (pd.DataFrame): Dataset with invalid fares removed
    """
    initial_shape = df.shape[0]
    
    if target_col in df.columns:
        valid_mask = df[target_col].between(min_fare, max_fare)
        df = df[valid_mask].copy()
        
        removed = initial_shape - df.shape[0]
        print(f"Removed {removed} rows with invalid fare amounts ({removed/initial_shape*100:.2f}%)")
    
    return df


def remove_invalid_passenger_count(df, max_passengers=8):
    """
    Remove rows with invalid passenger counts.
    
    Args:
        df (pd.DataFrame): Dataset
        max_passengers (int): Maximum valid passenger count
    
    Returns:
        df (pd.DataFrame): Dataset with invalid passenger counts removed
    """
    initial_shape = df.shape[0]
    
    if 'passenger_count' in df.columns:
        valid_mask = (df['passenger_count'] > 0) & (df['passenger_count'] <= max_passengers)
        df = df[valid_mask].copy()
        
        removed = initial_shape - df.shape[0]
        print(f"Removed {removed} rows with invalid passenger counts ({removed/initial_shape*100:.2f}%)")
    
    return df


def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Dataset
        strategy (str): Strategy for handling missing values ('drop', 'mean', 'median')
    
    Returns:
        df (pd.DataFrame): Dataset with missing values handled
    """
    initial_shape = df.shape[0]
    missing_count = df.isnull().sum().sum()
    
    if missing_count == 0:
        print("No missing values found")
        return df
    
    print(f"Found {missing_count} missing values")
    
    if strategy == 'drop':
        df = df.dropna()
        removed = initial_shape - df.shape[0]
        print(f"Dropped {removed} rows with missing values")
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        print("Filled missing values with mean (numeric columns only)")
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print("Filled missing values with median (numeric columns only)")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df


# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================

def preprocess_uber_data(df=None, remove_outliers=True, scale_features=False):
    """
    Main preprocessing function for Uber fares dataset.
    
    Args:
        df (pd.DataFrame, optional): Raw dataset. If None, loads from Kaggle
        remove_outliers (bool): Whether to remove outliers based on IQR
        scale_features (bool): Whether to scale features using StandardScaler
    
    Returns:
        X (pd.DataFrame): Processed feature matrix
        y (pd.Series): Target variable (fare_amount)
        scaler (StandardScaler or None): Fitted scaler if scale_features=True
    """
    # Load data if not provided
    if df is None:
        df = load_uber_data()
    
    target_col = 'fare_amount'
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Initial dataset shape: {df.shape}")
    
    # Step 1: Remove key column (unique identifier, not useful for prediction)
    # Note: Values were already removed during loading by setting index_col=0 but this removes key from header
    if 'key' in df.columns:
        df = df.drop(columns=['key'])
        print("Removed 'key' column (unique identifier)")
    
    # Step 2: Extract temporal features
    df = extract_temporal_features(df)
    
    # Step 3: One-hot encode temporal features
    df = one_hot_encode_temporal_features(df)
    
    # Step 4: Add distance feature
    df = add_distance_feature(df)
    
    # Step 5: Remove invalid coordinates
    df = remove_invalid_coordinates(df)
    
    # Step 6: Remove invalid fares
    df = remove_invalid_fares(df, target_col=target_col, min_fare=0, max_fare=500)
    
    # Step 7: Remove invalid passenger counts
    df = remove_invalid_passenger_count(df, max_passengers=8)
    
    # Step 8: Handle missing values
    df = handle_missing_values(df, strategy='drop')
    
    # Step 9: Remove outliers using IQR method (optional)
    if remove_outliers and target_col in df.columns:
        initial_shape = df.shape[0]
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Also remove outliers in distance if it exists
        if 'distance_km' in df.columns:
            Q1_dist = df['distance_km'].quantile(0.25)
            Q3_dist = df['distance_km'].quantile(0.75)
            IQR_dist = Q3_dist - Q1_dist
            lower_bound_dist = Q1_dist - 1.5 * IQR_dist
            upper_bound_dist = Q3_dist + 1.5 * IQR_dist
            
            valid_mask = (
                (df[target_col] >= lower_bound) & (df[target_col] <= upper_bound) &
                (df['distance_km'] >= lower_bound_dist) & (df['distance_km'] <= upper_bound_dist)
            )
        else:
            valid_mask = (df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)
        
        df = df[valid_mask].copy()
        removed = initial_shape - df.shape[0]
        print(f"Removed {removed} outliers using IQR method ({removed/initial_shape*100:.2f}%)")
    
    # Step 10: Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Remove datetime column if it exists (we've extracted features from it)
    if 'pickup_datetime' in X.columns:
        X = X.drop(columns=['pickup_datetime'])
        print("Removed 'pickup_datetime' column (features already extracted)")
    
    # Step 11: Scale features (optional)
    scaler = None
    if scale_features:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        print("Scaled numeric features using StandardScaler")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {target_col}")
    print(f"Target statistics:")
    print(f"  Mean: ${y.mean():.2f}")
    print(f"  Std: ${y.std():.2f}")
    print(f"  Min: ${y.min():.2f}")
    print(f"  Max: ${y.max():.2f}")
    print("="*80)
    
    if scaler:
        return X, y, scaler
    return X, y


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run preprocessing
    X, y = preprocess_uber_data(remove_outliers=True, scale_features=False)
    
    # Optionally save preprocessed data
    X.to_csv('preprocessed_features.csv', index=False)
    y.to_csv('preprocessed_target.csv', index=False)
    print("\nPreprocessed data saved to CSV files")
