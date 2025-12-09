import pandas as pd
import numpy as np 
from part3_regression_preprocessing import preprocess_uber_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_SEED = 42

## ============================================================================
## DATA LOADING AND SPLITTING FUNCTION
## ============================================================================

def load_and_split_data(load_from_csv=True):
    """Loads data and splits it into 70/15/15 train/validation/test sets."""
    
    if load_from_csv:
        # Load X features (as DataFrame for feature names)
        X = pd.read_csv("preprocessed_features.csv")
        # Load y target (as Series)
        y_series = pd.read_csv("preprocessed_target.csv").squeeze() 
        
        X_data = X.to_numpy()
        y_data = y_series.to_numpy()
        print("Data loaded from CSV files.")
    else:
        X, y_series = preprocess_uber_data(remove_outliers=True, scale_features=False)
        X_data = X.to_numpy()
        y_data = y_series.to_numpy()
        
    # Step 1: Split data into 70% Training and 30% Temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_data, y_data, test_size=0.3, random_state=RANDOM_SEED
    )

    # Step 2: Split 30% Temporary pool into two equal halves (15% Validation and 15% Test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
    )
    
    # Return all split data, plus the original DataFrame and Series for feature names/preview
    return X_train, X_val, X_test, y_train, y_val, y_test, X, y_series


## ============================================================================
## MODEL TRAINING FUNCTION
## ============================================================================

def train_and_evaluate_adaboost(X_train, y_train, X_val, y_val, X_features_df, **kwargs):
    """
    Trains and evaluates the AdaBoostRegressor using parameters passed via kwargs.
    
    The function intelligently separates base estimator params (prefixed with 'estimator__')
    from the main ensemble parameters.
    """
    print("\n" + " TRAINING AdaBoost Regressor ")
    
    # Separate parameters into two groups: base estimator and ensemble
    ada_params = {}
    base_params = {}
    
    for key, value in kwargs.items():
        if key.startswith('estimator__'):
            # Remove 'estimator__' prefix and add to base_params
            base_params[key[11:]] = value
        else:
            ada_params[key] = value

    # 1. Define the Base Estimator using the separated parameters
    base_estimator = DecisionTreeRegressor(random_state=RANDOM_SEED, **base_params)

    # 2. Initialize the AdaBoost Regressor
    ada_reg_model = AdaBoostRegressor(
        estimator=base_estimator,
        random_state=RANDOM_SEED,
        **ada_params  # Pass remaining parameters (n_estimators, learning_rate, loss)
    )
    
    # Log the parameters being used
    print(f"Ensemble Params: {ada_params}")
    print(f"Base Estimator Params: {base_params}")
    
    # ... (rest of the training and evaluation logic remains the same) ...
    # Train the model
    ada_reg_model.fit(X_train, y_train)

    # Predict and Evaluate
    y_val_pred = ada_reg_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\nVAL SET PERFORMANCE: RMSE=${val_rmse:.2f}, R²={val_r2:.4f}")
    
    # ... (code to calculate and return feature importance, model, and metrics) ...
    feature_importance = ada_reg_model.feature_importances_
    feature_names = X_features_df.columns
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    top_features = {feature_names[sorted_idx[i]]: feature_importance[sorted_idx[i]] for i in range(5)}
    
    return ada_reg_model, val_rmse, val_r2, top_features

## ============================================================================
## MAIN EXECUTION BLOCK (BASELINE RUN)
## ============================================================================

if __name__ == "__main__":
    
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test, X_df, y_series = load_and_split_data(load_from_csv=True)
    
    print("-" * 50)
    print(f"Data Splitting Complete: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    # Display preview (Optional)
    pd.set_option('display.max_columns', None) 
    print(f"\n1. X Features (First 5 Rows):\n{X_df.head()}") 
    pd.reset_option('display.max_columns') 
    
    # Define baseline hyperparameters
    BASELINE_PARAMS = {
        'n_estimators': 100, 
        'learning_rate': 0.01,
        'loss': 'linear',
        'estimator__max_depth': 9,
        # 'estimator__min_samples_leaf': 10 
    }

    # Train and evaluate the baseline model
    model, rmse, r2, top_features = train_and_evaluate_adaboost(
        X_train, y_train, X_val, y_val, X_df, **BASELINE_PARAMS
    )
    
    print("\nTop 5 Feature Importances:")
    for i, (name, imp) in enumerate(top_features.items()):
        print(f"  {i+1}. {name}: {imp:.4f}")

    print("\nModel training and initial validation complete.")