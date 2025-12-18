import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from train_evaluate_model import load_and_split_data, RANDOM_SEED
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import time

# --- CONFIGURATION ---
# Set to 'grid' for exhaustive search or 'random' for faster randomized search
SEARCH_TYPE = 'random'  # Change to 'grid' for exhaustive search
N_RANDOM_ITERATIONS = 100  # Only used if SEARCH_TYPE='random'

# --- CUSTOM SCORER ---
neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

## ============================================================================
## 1. LOAD DATA AND INITIAL SETUP
## ============================================================================

print("="*70)
print(f" AdaBoost {'Randomized' if SEARCH_TYPE == 'random' else 'Grid'} Search - Hyperparameter Tuning ")
print("="*70)

# Load the split data (now 80/20 train/test split)
X_train, X_test, y_train, y_test, X_df = load_and_split_data()

print("\n" + "-" * 70)
print(f"Starting search on {X_train.shape[0]} training samples...")
print(f"Test set size: {X_test.shape[0]} samples")
print("-" * 70)

## ============================================================================
## 2. DEFINE THE HYPERPARAMETER GRID/DISTRIBUTIONS
## ============================================================================

# Define the Decision Tree Regressor as the base estimator for AdaBoost
base_tree = DecisionTreeRegressor(random_state=RANDOM_SEED)

# Define the AdaBoost Regressor model
ada_reg = AdaBoostRegressor(estimator=base_tree, random_state=RANDOM_SEED)

if SEARCH_TYPE == 'grid':
    # Grid Search: Define discrete parameter grid
    param_space = {
        'n_estimators': [50, 100, 150, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
        'loss': ['linear', 'square', 'exponential'],
        'estimator__max_depth': [3, 5, 7, 9, 12],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4]
    }
    
    total_combinations = np.prod([len(v) for v in param_space.values()])
    CV_FOLDS = 5
    CV_FOLDS = 3  # Reduce folds for grid search to limit total fits
    total_fits = total_combinations * CV_FOLDS
    
    print(f"\n⚙️  Grid Search Configuration:")
    print(f"   • Total combinations: {total_combinations}")
    print(f"   • Cross-Validation Folds: {CV_FOLDS}")
    print(f"   • Total model fits: {total_fits}")
    
else:
    # Randomized Search: Define parameter distributions
    param_space = {
        'n_estimators': [50, 75, 100, 125, 150, 200, 250, 300, 400, 500],
        'learning_rate': [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0],
        'loss': ['linear', 'square', 'exponential'],
        'estimator__max_depth': list(range(3, 16)),  # 3 to 15
        'estimator__min_samples_split': list(range(2, 21)),  # 2 to 20
        'estimator__min_samples_leaf': list(range(1, 11))  # 1 to 10
    }
    
    CV_FOLDS = 5
    total_fits = N_RANDOM_ITERATIONS * CV_FOLDS
    
    print(f"\n⚙️  Randomized Search Configuration:")
    print(f"   • Random iterations: {N_RANDOM_ITERATIONS}")
    print(f"   • Cross-Validation Folds: {CV_FOLDS}")
    print(f"   • Total model fits: {total_fits}")

print(f"\n📊 Parameter Ranges:")
for param, values in param_space.items():
    if len(values) <= 10:
        print(f"   • {param}: {values}")
    else:
        print(f"   • {param}: range({min(values)}, {max(values)+1}) [{len(values)} values]")
print("-" * 70)

## ============================================================================
## 3. RUN SEARCH WITH PARALLEL PROCESSING
## ============================================================================

print(f"\n🔍 Starting {'Randomized' if SEARCH_TYPE == 'random' else 'Grid'} Search...")
print("   (Using all CPU cores - updates will print periodically)\n")

# Custom verbose class to print progress
class VerboseCallback:
    def __init__(self, total_fits):
        self.total_fits = total_fits
        self.fit_count = 0
        self.last_print_time = time.time()
        self.start_time = time.time()
        
    def __call__(self, scores):
        self.fit_count += 1
        current_time = time.time()
        
        # Print every 5 seconds or every 10% of progress
        if (current_time - self.last_print_time > 5) or (self.fit_count % max(1, self.total_fits // 10) == 0):
            elapsed = current_time - self.start_time
            percent = (self.fit_count / self.total_fits) * 100
            
            # Estimate time remaining
            if self.fit_count > 0:
                avg_time_per_fit = elapsed / self.fit_count
                remaining_fits = self.total_fits - self.fit_count
                eta_seconds = avg_time_per_fit * remaining_fits
                eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
            else:
                eta_str = "calculating..."
            
            print(f"   [{percent:5.1f}%] Completed {self.fit_count}/{self.total_fits} fits | "
                  f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | ETA: {eta_str}")
            self.last_print_time = current_time

# Initialize search
if SEARCH_TYPE == 'grid':
    search = GridSearchCV(
        estimator=ada_reg,
        param_grid=param_space,
        scoring=neg_mse_scorer,
        cv=CV_FOLDS,
        verbose=3,  # Shows some progress
        n_jobs=-1,  # Use most CPU cores
        return_train_score=True
    )
else:
    search = RandomizedSearchCV(
        estimator=ada_reg,
        param_distributions=param_space,
        n_iter=N_RANDOM_ITERATIONS,
        scoring=neg_mse_scorer,
        cv=CV_FOLDS,
        verbose=3,  # Shows some progress
        n_jobs=-1,  # Use all CPU cores
        random_state=RANDOM_SEED,
        return_train_score=True
    )

start_time = time.time()
search.fit(X_train, y_train)
end_time = time.time()

## ============================================================================
## 4. REPORT RESULTS
## ============================================================================

print("\n" + "="*70)
print(f"🏆 {'RANDOMIZED' if SEARCH_TYPE == 'random' else 'GRID'} SEARCH COMPLETE 🏆")
print("="*70)
print(f"Time taken: {(end_time - start_time)/60:.2f} minutes")

# Get the best parameters found
best_params = search.best_params_
print("\n✅ Best Hyperparameters Found:")
for k, v in best_params.items():
    print(f"   • {k}: {v}")

# Get the best score (which is negative MSE)
best_mse = -search.best_score_
best_rmse = np.sqrt(best_mse)

print(f"\n📈 Best Cross-Validated Performance (CV on Training Data):")
print(f"   • MSE:  ${best_mse:.2f}")
print(f"   • RMSE: ${best_rmse:.2f}")

# Retrieve the best model object
best_model = search.best_estimator_

## ============================================================================
## 5. ANALYZE TOP MODELS
## ============================================================================

print("\n" + "-" * 70)
print("📊 Top 10 Model Configurations:")
print("-" * 70)

# Get all results and sort by score
cv_results = pd.DataFrame(search.cv_results_)
cv_results['mean_rmse'] = np.sqrt(-cv_results['mean_test_score'])
cv_results = cv_results.sort_values('mean_test_score', ascending=False)

# Display top 10
top_10 = cv_results.head(10)[['params', 'mean_rmse', 'std_test_score', 'mean_fit_time']]
for idx, row in enumerate(top_10.itertuples(), 1):
    print(f"\n{idx}. RMSE: ${row.mean_rmse:.2f} (±{np.sqrt(row.std_test_score):.2f})")
    print(f"   Fit time: {row.mean_fit_time:.2f}s")
    print(f"   Params: {row.params}")

## ============================================================================
## 6. CHECK FOR CONVERGENCE (n_estimators analysis)
## ============================================================================

print("\n" + "-" * 70)
print("🔬 Convergence Analysis (n_estimators):")
print("-" * 70)

# Group by n_estimators and show average performance
estimator_analysis = cv_results.groupby(
    cv_results['params'].apply(lambda x: x['n_estimators'])
)['mean_rmse'].agg(['mean', 'min', 'max', 'std', 'count'])

print("\nPerformance by number of estimators:")
print(estimator_analysis.sort_index().to_string())

# Check if performance still improving at max n_estimators
n_estimators_values = [p['n_estimators'] for p in cv_results['params']]
max_estimators = max(n_estimators_values)
best_at_max = cv_results[
    cv_results['params'].apply(lambda x: x['n_estimators'] == max_estimators)
]['mean_rmse'].min()
best_overall = cv_results['mean_rmse'].min()

if abs(best_at_max - best_overall) < 0.01:  # Within $0.01
    print(f"\n⚠️  WARNING: Best model uses n_estimators={max_estimators} (near maximum).")
    print(f"   Consider expanding the range to check for further improvement.")
else:
    print(f"\n✅ Model appears to have converged before max n_estimators.")
    optimal_n = best_params['n_estimators']
    print(f"   Optimal n_estimators: {optimal_n}")

## ============================================================================
## 7. FINAL EVALUATION ON TEST SET
## ============================================================================

print("\n" + "="*70)
print("🧪 FINAL EVALUATION ON HELD-OUT TEST SET")
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
print(f"\n📊 Performance Comparison:")
print(f"   • CV RMSE:   ${best_rmse:.2f}")
print(f"   • Test RMSE: ${test_rmse:.2f}")
print(f"   • Difference: ${abs(test_rmse - best_rmse):.2f}")

if abs(test_rmse - best_rmse) / best_rmse > 0.1:
    print(f"\n⚠️  WARNING: Test performance differs from CV by >10%")
    print(f"   This may indicate overfitting or distribution shift.")
else:
    print(f"\n✅ Test performance is consistent with CV results.")

## ============================================================================
## 8. SAVE RESULTS
## ============================================================================

print("\n" + "-" * 70)
print("💾 Saving Results...")
print("-" * 70)

# Save the full CV results
search_type_str = 'random' if SEARCH_TYPE == 'random' else 'grid'
cv_results.to_csv(f'{search_type_str}_search_results.csv', index=False)
print(f"   • Saved: {search_type_str}_search_results.csv")

# Save best parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv('best_hyperparameters.csv', index=False)
print("   • Saved: best_hyperparameters.csv")

# Save summary
summary = {
    'search_type': SEARCH_TYPE,
    'cv_rmse': best_rmse,
    'test_rmse': test_rmse,
    'test_r2': test_r2,
    'best_n_estimators': best_params['n_estimators'],
    'best_learning_rate': best_params['learning_rate'],
    'best_loss': best_params['loss'],
    'best_max_depth': best_params['estimator__max_depth'],
    'total_search_time_minutes': (end_time - start_time) / 60,
    'total_fits': len(cv_results)
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(f'{search_type_str}_search_summary.csv', index=False)
print(f"   • Saved: {search_type_str}_search_summary.csv")

print("\n" + "="*70)
print("✨ Search Complete! ✨")
print("="*70)
print("\nNext Steps:")
print("   1. Review the top 10 configurations above")
print("   2. Check convergence analysis for n_estimators")
print("   3. Update BEST_PARAMS in train_evaluate_model.py with the best parameters")
print("   4. Consider feature engineering if performance is still limited")
if SEARCH_TYPE == 'random':
    print("   5. If needed, run grid search around the best parameters for fine-tuning")
print("="*70)