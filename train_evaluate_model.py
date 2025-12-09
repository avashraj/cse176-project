import pandas as pd
import numpy as np 
from part3_regression_preprocessing import preprocess_uber_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt 
from sklearn.model_selection import validation_curve # <-- NEW: For generating curve data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer # <-- Add make_scorer

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
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)

    print(f"\nVAL SET PERFORMANCE: MSE = ${val_mse} RMSE=${val_rmse:.2f}, R²={val_r2:.4f} MAE = ${val_mae:.2f}")
    
    # ... (code to calculate and return feature importance, model, and metrics) ...
    feature_importance = ada_reg_model.feature_importances_
    feature_names = X_features_df.columns
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    top_features = {feature_names[sorted_idx[i]]: feature_importance[sorted_idx[i]] for i in range(5)}
    
    return ada_reg_model, val_rmse, val_r2, top_features, y_val_pred

## ============================================================================
## PLOTTING FUNCTION
## ============================================================================

def plot_predictions_vs_actual(y_actual, y_predicted, save_filename="predictions_vs_actual.png"):
    """
    Generates a scatter plot comparing actual values to predicted values and saves it.
    
    Args:
        y_actual (np.array): The true target values (y_val).
        y_predicted (np.array): The model's predicted values (y_val_pred).
        save_filename (str): The name of the file to save the plot as.
    """
    
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of actual vs. predicted values
    # Use alpha for transparency due to large data size
    plt.scatter(y_actual, y_predicted, alpha=0.3, color='skyblue')
    
    # Add the ideal prediction line (y=x)
    # Get the minimum and maximum values for the axis limits
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal Prediction (y=x)')
    
    plt.title('AdaBoost Regressor: Predicted Fares vs. Actual Fares')
    plt.xlabel('Actual Fare Amount (USD)')
    plt.ylabel('Predicted Fare Amount (USD)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot to a file
    plt.savefig(save_filename)
    plt.close() # Close the figure to free up memory
    print(f"\nPlot saved to {save_filename}")

## ============================================================================
## PLOTTING FUNCTION: N_ESTIMATORS VALIDATION CURVE
## ============================================================================

def plot_validation_curve_n_estimators(X_train, y_train, X_val, y_val, base_params, save_filename="validation_curve_n_estimators.png"):
    """
    Generates and saves a validation curve plot showing error vs. n_estimators.
    
    Args:
        X_train, y_train, X_val, y_val: Training and validation data.
        base_params (dict): Fixed hyperparameters for the AdaBoost Regressor.
        save_filename (str): The name of the file to save the plot as.
    """
    print("\nGenerating Validation Curve for n_estimators...")
    
    # 1. Define the Fixed Base Estimator
    base_estimator = DecisionTreeRegressor(random_state=RANDOM_SEED, **base_params)

    # 2. Define the Model (n_estimators is a placeholder that validation_curve will iterate over)
    ada_reg_model = AdaBoostRegressor(
        estimator=base_estimator,
        random_state=RANDOM_SEED,
        learning_rate=0.01,  # Fixed
        loss='linear'        # Fixed
    )
    
    # 3. Define the parameter range to test (10 to 200 trees)
    param_range = np.arange(10, 201, 10) 
    
    # Use neg_mean_squared_error because validation_curve maximizes score
    neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # 4. Generate the curve data
    # We combine X_train and X_val into a single set (X_train_full)
    # The scoring function will handle the internal splits or use the full data for the curve.
    
    # NOTE: validation_curve runs cross-validation internally (cv=3).
    # We use the full available training data for this process.
    train_scores, valid_scores = validation_curve(
        estimator=ada_reg_model,
        X=np.vstack((X_train, X_val)), 
        y=np.hstack((y_train, y_val)),
        param_name="n_estimators",
        param_range=param_range,
        cv=3, # Use 3-fold cross-validation
        scoring=neg_mse_scorer,
        n_jobs=-1
    )
    
    # 5. Convert scores to positive RMSE
    train_scores_mean = np.sqrt(-train_scores.mean(axis=1))
    valid_scores_mean = np.sqrt(-valid_scores.mean(axis=1))

    # 6. Plotting
    plt.figure(figsize=(9, 6))
    plt.plot(param_range, train_scores_mean, label="Training RMSE", color="blue", lw=2)
    plt.plot(param_range, valid_scores_mean, label="Cross-Validation RMSE", color="red", lw=2)
    
    # Highlight the optimal point (n_estimators=100)
    plt.axvline(x=100, color='gray', linestyle='--', label=r'Tuned $\mathbf{n\_estimators=100}$')

    plt.title(f"AdaBoost Regressor: Performance vs. Number of Estimators")
    plt.xlabel(r"Number of Estimators ($n\_estimators$)")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.ylim(1.5, 3.5) # Set fixed limits for better visualization
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(save_filename)
    plt.close()
    print(f"Validation Curve saved to {save_filename}")

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
        'estimator__min_samples_leaf': 5 
    }

    # Train and evaluate the baseline model
    model, rmse, r2, top_features, y_val_pred = train_and_evaluate_adaboost(
        X_train, y_train, X_val, y_val, X_df, **BASELINE_PARAMS
    )
    
    # NEW FUNCTION CALL: Generate and save the plot
    plot_predictions_vs_actual(y_val, y_val_pred, save_filename="adaboost_predictions_val.png")
    
    # ... inside if __name__ == "__main__":

    # ... (code to load and split data) ...

    # Define the FIXED Base Hyperparameters (excluding n_estimators, learning_rate, and loss)
    BASE_ESTIMATOR_PARAMS = {
        'max_depth': 9,
        # IMPORTANT: Use min_samples_split=2 from your results (original code used min_samples_leaf=10)
        'min_samples_split': 2 
    }
    
    # We are using the best parameters for the TRAINING FUNCTION call:
    BEST_PARAMS_FOR_TRAINING = {
        'n_estimators': 100, 
        'learning_rate': 0.01,
        'loss': 'linear',
        'estimator__max_depth': 9,
        'estimator__min_samples_split': 2 
    }

    # 1. Train and evaluate the final model
    model, rmse, r2, top_features, y_val_pred = train_and_evaluate_adaboost(
        X_train, y_train, X_val, y_val, X_df, **BEST_PARAMS_FOR_TRAINING
    )
    
    # 2. Generate the Validation Curve
    plot_validation_curve_n_estimators(
        X_train, y_train, X_val, y_val, 
        base_params=BASE_ESTIMATOR_PARAMS, 
        save_filename="adaboost_validation_curve.png"
    )
    
    # 3. Generate the Final Prediction Plot
    plot_predictions_vs_actual(y_val, y_val_pred, save_filename="adaboost_predictions_val.png")

    # ... (rest of the print statements) ...
    
    print("\nTop 5 Feature Importances:")
    for i, (name, imp) in enumerate(top_features.items()):
        print(f"  {i+1}. {name}: {imp:.4f}")

    print("\nModel training and initial validation complete.")