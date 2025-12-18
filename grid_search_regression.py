import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from train_evaluate_model import load_and_split_data, RANDOM_SEED
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import time
from tqdm.auto import tqdm

# --- CUSTOM SCORER ---
# GridSearchCV needs a single score to optimize. We'll use negative Mean Squared Error (MSE)
# because GridSearchCV maximizes score, and we want to minimize MSE.
neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

## ============================================================================
## PROGRESS TRACKING SCORER
## ============================================================================

# Global progress bar (will be created later)
_pbar = None

def progress_scorer(estimator, X, y):
    """Custom scorer that updates progress bar and returns negative MSE."""
    global _pbar
    if _pbar is not None:
        _pbar.update(1)
    y_pred = estimator.predict(X)
    return -mean_squared_error(y, y_pred)

## ============================================================================
## 1. LOAD DATA AND INITIAL SETUP
## ============================================================================

print("="*70)
print(" AdaBoost Grid Search - Hyperparameter Tuning ")
print("="*70)

# Load the split data (now 80/20 train/test split)
X_train, X_test, y_train, y_test, X_df = load_and_split_data()

print("\n" + "-" * 70)
print(f"Starting Grid Search on {X_train.shape[0]} training samples...")
print(f"Test set size: {X_test.shape[0]} samples")
print("-" * 70)

## ============================================================================
## 2. DEFINE THE HYPERPARAMETER GRID
## ============================================================================

# Define the Decision Tree Regressor as the base estimator for AdaBoost
base_tree = DecisionTreeRegressor(random_state=RANDOM_SEED)

# Define the AdaBoost Regressor model
ada_reg = AdaBoostRegressor(estimator=base_tree, random_state=RANDOM_SEED)

# The parameter grid to explore
# EXPANDED GRID for more thorough search
# param_grid = {
#     # AdaBoost Ensemble Parameters
#     'n_estimators': [50, 100, 150, 200, 300],     # Extended range to check convergence
#     'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0], # More granular learning rates
#     'loss': ['linear', 'square', 'exponential'],  # All three loss functions
    
#     # Base Estimator (DecisionTreeRegressor) Parameters
#     'estimator__max_depth': [3, 5, 7, 9, 12],     # Wider range of depths
#     'estimator__min_samples_split': [2, 5, 10],   # Added 10 for more regularization
#     'estimator__min_samples_leaf': [1, 2, 4]      # NEW: Controls leaf size
# }

param_grid = {
    # AdaBoost Ensemble Parameters
    'n_estimators': [50],     # Extended range to check convergence
    'learning_rate': [0.01], # More granular learning rates
    'loss': ['linear'],  # All three loss functions
    
    # Base Estimator (DecisionTreeRegressor) Parameters
    'estimator__max_depth': [3],     # Wider range of depths
    'estimator__min_samples_split': [2],   # Added 10 for more regularization
    'estimator__min_samples_leaf': [1]      # NEW: Controls leaf size
}

# Calculate the total number of model fits
total_combinations = np.prod([len(v) for v in param_grid.values()])
CV_FOLDS = 5  # 5-fold CV is standard and more reliable than 3-fold
CV_FOLDS = 3  # Reduced to 3-fold CV to limit total computation time
total_fits = total_combinations * CV_FOLDS

print(f"\nGrid Configuration:")
print(f"   • Grid Size: {total_combinations} combinations")
print(f"   • Cross-Validation Folds: {CV_FOLDS}")
print(f"   • Total model fits: {total_combinations * CV_FOLDS}")
print(f"\nParameter Ranges:")
for param, values in param_grid.items():
    print(f"   • {param}: {values}")
print("-" * 70)

## ============================================================================
## 3. RUN GRID SEARCH
## ============================================================================

print("\n🔍 Starting Grid Search...")
print("   Progress bar will show below:\n")

# Create global progress bar
# global _pbar
_pbar = tqdm(total=total_fits, desc="Grid Search Progress", 
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} fits [{elapsed}<{remaining}]')

# Initialize GridSearchCV with progress-tracking scorer
grid_search = GridSearchCV(
    estimator=ada_reg,
    param_grid=param_grid,
    scoring=progress_scorer,  # Use custom scorer that updates progress bar
    cv=CV_FOLDS,              # Number of cross-validation folds
    verbose=0,                # Turn off verbose to let progress bar work cleanly
    n_jobs=1,                 # MUST be 1 for progress bar to work (no parallel processing)
    return_train_score=True   # Also return training scores to check overfitting
)

start_time = time.time()
try:
    grid_search.fit(X_train, y_train)
finally:
    _pbar.close()  # Ensure progress bar closes even if there's an error
    _pbar = None
end_time = time.time()

## ============================================================================
## 4. REPORT RESULTS
## ============================================================================

print("\n" + "="*70)
print("GRID SEARCH COMPLETE")
print("="*70)
print(f"Time taken: {(end_time - start_time)/60:.2f} minutes")

# Get the best parameters found
best_params = grid_search.best_params_
print("\nBest Hyperparameters Found:")
for k, v in best_params.items():
    print(f"   • {k}: {v}")

# Get the best score (which is negative MSE)
best_mse = -grid_search.best_score_
best_rmse = np.sqrt(best_mse)

print(f"\nBest Cross-Validated Performance (CV on Training Data):")
print(f"   • MSE:  ${best_mse:.2f}")
print(f"   • RMSE: ${best_rmse:.2f}")

# Retrieve the best model object
best_model = grid_search.best_estimator_

## ============================================================================
## 5. ANALYZE TOP MODELS
## ============================================================================

print("\n" + "-" * 70)
print("Top 10 Model Configurations:")
print("-" * 70)

# Get all results and sort by score
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results['mean_rmse'] = np.sqrt(-cv_results['mean_test_score'])
cv_results = cv_results.sort_values('mean_test_score', ascending=False)

# Display top 10
top_10 = cv_results.head(10)[['params', 'mean_rmse', 'std_test_score', 'mean_fit_time']]
for idx, row in top_10.iterrows():
    print(f"\n{idx+1}. RMSE: ${row['mean_rmse']:.2f} (±{np.sqrt(row['std_test_score']):.2f})")
    print(f"   Fit time: {row['mean_fit_time']:.2f}s")
    print(f"   Params: {row['params']}")

## ============================================================================
## 6. CHECK FOR CONVERGENCE (n_estimators analysis)
## ============================================================================

print("\n" + "-" * 70)
print("Convergence Analysis (n_estimators):")
print("-" * 70)

# Group by n_estimators and show average performance
estimator_analysis = cv_results.groupby(
    cv_results['params'].apply(lambda x: x['n_estimators'])
)['mean_rmse'].agg(['mean', 'min', 'max', 'std'])

print("\nPerformance by number of estimators:")
print(estimator_analysis.to_string())

# Check if performance still improving at max n_estimators
max_estimators = max(param_grid['n_estimators'])
best_at_max = cv_results[
    cv_results['params'].apply(lambda x: x['n_estimators'] == max_estimators)
]['mean_rmse'].min()
best_overall = cv_results['mean_rmse'].min()

if best_at_max == best_overall:
    print(f"\nWARNING: Best model uses maximum n_estimators ({max_estimators}).")
    print(f"   Consider expanding the range to check for further improvement.")
else:
    print(f"\nModel appears to have converged before max n_estimators.")

## ============================================================================
## 7. FINAL EVALUATION ON TEST SET
## ============================================================================

print("\n" + "="*70)
print("FINAL EVALUATION ON HELD-OUT TEST SET")
print("="*70)

y_test_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTest Set Performance:")
print(f"   • MSE:  ${test_mse:.2f}")
print(f"   • RMSE: ${test_rmse:.2f}")
print(f"   • R²:   {test_r2:.4f}")

# Compare to CV performance
print(f"\nPerformance Comparison:")
print(f"   • CV RMSE:   ${best_rmse:.2f}")
print(f"   • Test RMSE: ${test_rmse:.2f}")
print(f"   • Difference: ${abs(test_rmse - best_rmse):.2f}")

if abs(test_rmse - best_rmse) / best_rmse > 0.1:
    print(f"\n WARNING: Test performance differs from CV by >10%")
    print(f"   This may indicate overfitting or distribution shift.")
else:
    print(f"\nTest performance is consistent with CV results.")

## ============================================================================
## 8. SAVE RESULTS
## ============================================================================

print("\n" + "-" * 70)
print("Saving Results...")
print("-" * 70)

# Save the full CV results
cv_results.to_csv('grid_search_results.csv', index=False)
print("   • Saved: grid_search_results.csv")

# Save best parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv('best_hyperparameters.csv', index=False)
print("   • Saved: best_hyperparameters.csv")

# Save summary
summary = {
    'cv_rmse': best_rmse,
    'test_rmse': test_rmse,
    'test_r2': test_r2,
    'best_n_estimators': best_params['n_estimators'],
    'best_learning_rate': best_params['learning_rate'],
    'total_search_time_minutes': (end_time - start_time) / 60
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv('grid_search_summary.csv', index=False)
print("   • Saved: grid_search_summary.csv")

print("\n" + "="*70)
print("Grid Search Complete! ✨")
print("="*70)
print("\nNext Steps:")
print("   1. Review the top 10 configurations above")
print("   2. Check convergence analysis for n_estimators")
print("   3. Use best parameters in train_evaluate_model.py")
print("   4. Consider feature engineering if performance is still limited")
print("="*70)