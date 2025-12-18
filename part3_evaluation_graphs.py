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

## ============================================================================
## PLOTTING FUNCTION
## ============================================================================

def plot_predictions_vs_actual(y_actual, y_predicted, save_filename="predictions_vs_actual.png"):
    """
    Generates a scatter plot comparing actual values to predicted values and saves it.
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

# --- NEW FUNCTION FOR RESIDUAL PLOT ---
def plot_residuals(y_actual, y_predicted, save_filename="residual_plot.png"):
    """
    Generates and saves a residual plot.
    
    Args:
        y_actual (np.array): The true target values (y_val).
        y_predicted (np.array): The model's predicted values (y_val_pred).
        save_filename (str): The name of the file to save the plot as.
    """
    # 1. Calculate the residuals (Error = Actual - Predicted)
    residuals = y_actual - y_predicted
    
    plt.figure(figsize=(8, 6))
    
    # 2. Scatter plot of Predicted Values (X-axis) vs. Residuals (Y-axis)
    # The predicted value is the best independent variable for spotting heteroscedasticity.
    plt.scatter(y_predicted, residuals, alpha=0.3, color='darkorange')
    
    # 3. Add the centerline at Residual = 0
    plt.hlines(y=0, xmin=y_predicted.min(), xmax=y_predicted.max(), 
               color='red', linestyle='-', linewidth=2, label='Zero Error Line')
    
    plt.title('AdaBoost Regressor: Residual Plot')
    plt.xlabel('Predicted Fare Amount (USD)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot
    plt.savefig(save_filename)
    plt.close() # Close the figure
    print(f"Residual plot saved to {save_filename}")
# --- END NEW FUNCTION ---

## ============================================================================
## PLOTTING FUNCTION: N_ESTIMATORS VALIDATION CURVE
## ============================================================================

def plot_validation_curve_n_estimators(X_train, y_train, X_val, y_val, base_params, save_filename="validation_curve_n_estimators.png"):
    """
    Generates and saves a validation curve plot showing error vs. n_estimators.
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
    param_range = np.arange(5, 50, 5) 
    
    # Use neg_mean_squared_error because validation_curve maximizes score
    neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # 4. Generate the curve data
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
    # The actual range used here is 5-45, so 100 is just a visual reference line.
    # The visual reference line should be updated based on the actual range of param_range or removed.
    # We will remove the static 100 line for this example since the range is small.
    # plt.axvline(x=100, color='gray', linestyle='--', label=r'Tuned $\mathbf{n\_estimators=100}$')

    plt.title(f"AdaBoost Regressor: Performance vs. Number of Estimators")
    plt.xlabel(r"Number of Estimators ($n\_estimators$)")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.ylim(1.5, 3.5) # Set fixed limits for better visualization
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(save_filename)
    plt.close()
    print(f"Validation Curve saved to {save_filename}")
    
def plot_error_distribution(y_actual, y_predicted, save_filename="error_distribution_histogram.png"):
    """
    Generates and saves a histogram showing the distribution of the residuals.
    
    A properly performing model should have residuals normally distributed
    around a mean of zero.
    
    Args:
        y_actual (np.array): The true target values (y_val).
        y_predicted (np.array): The model's predicted values (y_val_pred).
        save_filename (str): The name of the file to save the plot as.
    """
    # 1. Calculate the residuals (Error = Actual - Predicted)
    residuals = y_actual - y_predicted
    
    # Calculate key statistics for annotation
    mean_residual = np.mean(residuals)
    median_residual = np.median(residuals)
    
    plt.figure(figsize=(10, 6))
    
    # 2. Plot the Histogram
    # Use density=True to visualize the shape, and a reasonable number of bins.
    plt.hist(residuals, bins=50, density=True, color='purple', alpha=0.7, 
             label='Residuals Distribution', edgecolor='black')
    
    # 3. Add Diagnostic Lines
    
    # Line for Mean Residual
    plt.axvline(mean_residual, color='darkgreen', linestyle='dashed', linewidth=2,
                label=f'Mean Residual: ${mean_residual:.3f}')
    
    # Line for Zero Error
    plt.axvline(0, color='red', linestyle='-', linewidth=1.5,
                label='Zero Error Line')

    plt.title('AdaBoost Regressor: Error Distribution Histogram')
    plt.xlabel('Residuals (Actual Fare - Predicted Fare)')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot
    plt.savefig(save_filename)
    plt.close() # Close the figure
    print(f"Error distribution histogram saved to {save_filename}")