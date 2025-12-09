import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV # The core tool for tuning
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from train_evaluate_model import load_and_split_data, train_and_evaluate_adaboost, RANDOM_SEED
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import time

# --- CUSTOM SCORER ---
# GridSearchCV needs a single score to optimize. We'll use negative Mean Squared Error (MSE)
# because GridSearchCV maximizes score, and we want to minimize MSE.
neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

## ============================================================================
## 1. LOAD DATA AND INITIAL SETUP
## ============================================================================

# Load the split data from your utility file
X_train, X_val, X_test, y_train, y_val, y_test, X_df, y_series = load_and_split_data(load_from_csv=True)

# Combine X_train and y_train for use in GridSearchCV's Cross-Validation
# GridSearchCV will handle the internal splits on this combined data.
X_train_cv = X_train
y_train_cv = y_train

print("-" * 60)
print(f"Starting Grid Search on {X_train_cv.shape[0]} training samples...")
print("-" * 60)

## ============================================================================
## 2. DEFINE THE HYPERPARAMETER GRID
## ============================================================================

# Define the Decision Tree Regressor as the base estimator for AdaBoost
base_tree = DecisionTreeRegressor(random_state=RANDOM_SEED)

# Define the AdaBoost Regressor model
ada_reg = AdaBoostRegressor(estimator=base_tree, random_state=RANDOM_SEED)

# The parameter grid to explore
param_grid = {
    # AdaBoost Ensemble Parameters
    'n_estimators': [50, 100, 200],         # Number of trees/iterations
    'learning_rate': [0.01, 0.1, 0.5],      # Contribution of each tree
    'loss': ['linear', 'square'],           # Loss function for regression
    
    # Base Estimator (DecisionTreeRegressor) Parameters
    'estimator__max_depth': [3, 6, 9],      # Crucial for controlling complexity
    'estimator__min_samples_split': [2, 5]  # Minimum samples required to split a node
}

# Calculate the total number of models to train
total_combinations = np.prod([len(v) for v in param_grid.values()])
CV_FOLDS = 3 # Use 3 folds for faster initial testing on large data
total_fits = total_combinations * CV_FOLDS

print(f"Grid Size: {total_combinations} combinations.")
print(f"Cross-Validation Folds: {CV_FOLDS}")
print(f"Total model fits to perform: {total_fits}")
print("-" * 60)

## ============================================================================
## 3. RUN GRID SEARCH
## ============================================================================

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=ada_reg,
    param_grid=param_grid,
    scoring=neg_mse_scorer,  # Optimize for minimum MSE (maximum negative MSE)
    cv=CV_FOLDS,             # Number of cross-validation folds
    verbose=2,               # Set to 1 or 2 to see progress
    n_jobs=-1                # Use all available CPU cores for speed
)

start_time = time.time()
grid_search.fit(X_train_cv, y_train_cv)
end_time = time.time()

## ============================================================================
## 4. REPORT RESULTS
## ============================================================================

print("\n" + "🏆" * 5 + " GRID SEARCH RESULTS " + "🏆" * 5)
print(f"Time taken for Grid Search: {(end_time - start_time):.2f} seconds")

# Get the best parameters found
best_params = grid_search.best_params_
print("\nBest Hyperparameters Found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Get the best score (which is negative MSE)
best_mse = -grid_search.best_score_
best_rmse = np.sqrt(best_mse)

print(f"\nBest Cross-Validated Score (Training Data):")
print(f"  RMSE: ${best_rmse:.4f}")

# Retrieve the best model object
best_model = grid_search.best_estimator_

## ============================================================================
## 5. FINAL VALIDATION (Using the separate 15% Validation Set)
## ============================================================================

print("-" * 60)
print("Evaluating BEST MODEL on separate 15% Validation Set...")

y_val_pred = best_model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val, y_val_pred)

print(f"\nFINAL VALIDATION PERFORMANCE:")
print(f"  RMSE: ${val_rmse:.4f}")
print(f"  R²: {val_r2:.4f}")
print("-" * 60)

# You can now save this best_model object for later use!