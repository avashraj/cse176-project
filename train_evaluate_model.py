import pandas as pd
import numpy as np 
from avash_part3_reg_eda import build_datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

RANDOM_SEED = 42


## ============================================================================
## DATA LOADING AND SPLITTING FUNCTION
## ============================================================================

def load_and_split_data():
    """
    Loads data from the new preprocessing pipeline and splits it into 
    80/20 train/test sets (no validation set needed since hyperparameters 
    are already tuned).
    """
    # Load data from new preprocessing files
    ds = build_datasets()
    data = ds.df_manhattan_iqr.copy()
    print(f"Total data loaded: {len(data)} samples")
    
    # Separate features (X) and target (y)
    target_column = 'fare_amount'
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Identify categorical columns (both 'category' dtype and 'object' dtype)
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_columns:
        print(f"\nCategorical columns found: {categorical_columns}")
        print("Applying one-hot encoding...")
        
        # Convert categorical dtypes to object first (for consistent handling)
        for col in categorical_columns:
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype(str)
        
        # One-hot encode categorical columns
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        print(f"Features after encoding: {X.shape[1]} columns")
    
    print(f"\nFinal feature columns: {list(X.columns)}")
    
    # Split into 80% Training and 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, X


## ============================================================================
## MODEL TRAINING FUNCTION
## ============================================================================

def train_and_evaluate_adaboost(X_train, y_train, X_test, y_test, X_features_df, **kwargs):
    """
    Trains and evaluates the AdaBoostRegressor using parameters passed via kwargs.
    
    The function intelligently separates base estimator params (prefixed with 'estimator__')
    from the main ensemble parameters.
    """
    print("\n" + "="*50)
    print(" TRAINING AdaBoost Regressor ")
    print("="*50)
    
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
    print(f"\nEnsemble Params: {ada_params}")
    print(f"Base Estimator Params: {base_params}")
    
    # Train the model
    print("\nTraining model...")
    ada_reg_model.fit(X_train, y_train)
    print("Training complete!")

    # Predict and Evaluate on Test Set
    print("\nEvaluating on test set...")
    y_test_pred = ada_reg_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\nTEST SET PERFORMANCE:")
    print(f"  MSE:  ${test_mse:.2f}")
    print(f"  RMSE: ${test_rmse:.2f}")
    print(f"  MAE:  ${test_mae:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # Calculate feature importance
    feature_importance = ada_reg_model.feature_importances_
    feature_names = X_features_df.columns
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    # Get top 5 features
    top_features = {
        feature_names[sorted_idx[i]]: feature_importance[sorted_idx[i]] 
        for i in range(min(5, len(feature_names)))
    }
    
    return ada_reg_model, test_rmse, test_r2, test_mae, top_features, y_test_pred


## ============================================================================
## MAIN EXECUTION BLOCK
## ============================================================================

if __name__ == "__main__":
    
    print("="*50)
    print(" AdaBoost Regressor - Final Model Training ")
    print("="*50)
    
    # Load and split data
    X_train, X_test, y_train, y_test, X_df = load_and_split_data()
    
    # Display feature preview (Optional)
    print("\n" + "-"*50)
    print("Feature Preview (First 5 Rows):")
    print("-"*50)
    pd.set_option('display.max_columns', None) 
    print(X_df.head()) 
    pd.reset_option('display.max_columns') 
    
    # Best hyperparameters from grid search
    BEST_PARAMS = {
        'n_estimators': 100, 
        'learning_rate': 0.01,
        'loss': 'linear',
        'estimator__max_depth': 9,
        'estimator__min_samples_split': 2 
    }

    # Train and evaluate the final model
    model, rmse, r2, mae, top_features, y_test_pred = train_and_evaluate_adaboost(
        X_train, y_train, X_test, y_test, X_df, **BEST_PARAMS
    )
    
    # Display top 5 feature importances
    print("\n" + "="*50)
    print("Top 5 Feature Importances:")
    print("="*50)
    for i, (name, imp) in enumerate(top_features.items(), 1):
        print(f"  {i}. {name}: {imp:.4f}")

    print("\n" + "="*50)
    print("Model training and evaluation complete!")
    print("="*50)
    
    # Commented out graph generation code - moved to part3_evaluation_graphs.py
    # Uncomment and import functions from part3_evaluation_graphs.py when needed:
    #
    # from part3_evaluation_graphs import (
    #     plot_validation_curve_n_estimators,
    #     plot_predictions_vs_actual,
    #     plot_residuals,
    #     plot_error_distribution
    # )
    #
    # plot_predictions_vs_actual(y_test, y_test_pred, save_filename="adaboost_predictions_test.png")
    # plot_residuals(y_test, y_test_pred, save_filename="adaboost_residual_plot.png")
    # plot_error_distribution(y_test, y_test_pred, save_filename="adaboost_error_distribution.png")