import pandas as pd
import numpy as np 
from part3_regression_preprocessing import preprocess_uber_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor # Needed for the base estimator
from sklearn.metrics import mean_squared_error, r2_score

# Load in features and target variable
load_from_csv = True

if load_from_csv:

    # Load X features
    X = pd.read_csv("preprocessed_features.csv")
    
    # Load y target. It was saved as a single Series/column.
    y_series = pd.read_csv("preprocessed_target.csv").squeeze() # .squeeze() makes it a single Series
    
    # Convert to NumPy arrays if required by your model (though pandas DataFrames work too)
    X_data = X.to_numpy()
    y_data = y_series.to_numpy()
    
    print("Data loaded from CSV files.")
else:
    # This block will run the full preprocessing pipeline (assuming import is fixed)
    # The X and y from the pipeline are already DataFrames/Series
    from part3_regression_preprocessing import preprocess_uber_data 
    X, y = preprocess_uber_data(remove_outliers=True, scale_features=False)
    
# Set a random state for reproducibility
RANDOM_SEED = 42

# Step 1: Split data into 70% Training and 30% Temporary (Validation + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, 
    test_size=0.3,          # 30% goes to the temporary pool
    random_state=RANDOM_SEED
)

# Step 2: Split 30% Temporary pool into two equal halves (15% Validation and 15% Test)
# Since X_temp is 30% of the original data, 0.5 of X_temp is 15% of the original.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5,          # 50% of the temp data goes to the test set
    random_state=RANDOM_SEED
)

## --- CHECK DATA PREVIEW ---
print("\n" + "=" * 50)
print("FINAL DATASET PREVIEW (Features and Target)")
print("=" * 50)

# Set Pandas option to display ALL columns (features) without truncation
pd.set_option('display.max_columns', None) 

print("1. X Features (First 5 Rows):")
# Using the DataFrame object X to show the header and first 5 rows cleanly
print(X.head()) 

# Reset the Pandas option to the default after printing
pd.reset_option('display.max_columns') 

print("\n2. y Target (First 5 Values):")
# Printing the first 5 values of the target series
print(y_series.head())
print("=" * 50)
# --- END CHECK DATA PREVIEW ---

# --- Verification ---
print("-" * 35)
print("Data Splitting Complete:")
print(f"X_train (70%): {X_train.shape[0]} samples")
print(f"X_val (15%):   {X_val.shape[0]} samples")
print(f"X_test (15%):  {X_test.shape[0]} samples")

# --- Verification ---
print("-" * 35)
print("Data Splitting Complete:")
print(f"X_train (70%): {X_train.shape[0]} samples")
print(f"X_val (15%):   {X_val.shape[0]} samples")
print(f"X_test (15%):  {X_test.shape[0]} samples")
print("-" * 35)


# ============================================================================
# MODEL TRAINING AND EVALUATION (AdaBoostRegressor)
# ============================================================================

print("\n" + " TRAINING AdaBoost Regressor ")

# 1. Define the Base Estimator (Weak Learner)
# A shallow tree is crucial for boosting algorithms to work effectively.
# max_depth=6 is a reasonable starting point for regression.
base_estimator = DecisionTreeRegressor(max_depth=6, random_state=RANDOM_SEED)

# 2. Initialize the AdaBoost Regressor
# n_estimators=150: Number of weak learners.
# learning_rate=0.1: Moderate rate of contribution from each learner.
# loss='square': Better for minimizing squared error in regression than the default 'linear'.
ada_reg_model = AdaBoostRegressor(
    estimator=base_estimator,
    n_estimators=150,
    learning_rate=0.1,
    loss='square',         
    random_state=RANDOM_SEED
)

print(f"Model: {type(ada_reg_model).__name__} initialized with Base Estimator: DecisionTreeRegressor(max_depth=6).")
print("Starting training on X_train...")

# Train the model on the Training set
ada_reg_model.fit(X_train, y_train)

print("Training complete. Evaluating on Validation set...")

# Predict on the Validation set
y_val_pred = ada_reg_model.predict(X_val)

# Evaluate performance on the Validation set
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val, y_val_pred)

print("\n" + "=" * 50)
print("VALIDATION SET PERFORMANCE")
print(f"   Root Mean Squared Error (RMSE): ${val_rmse:.2f}")
print(f"   Mean Squared Error (MSE):       {val_mse:.2f}")
print(f"   R-squared (R²):                 {val_r2:.4f}")
print("=" * 50)


# ============================================================================
# FEATURE IMPORTANCE CHECK
# ============================================================================
feature_importance = ada_reg_model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[::-1]

print("\nTop 5 Feature Importances:")
for i in range(5):
    print(f"  {i+1}. {feature_names[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.4f}")

print("\nModel training and initial validation complete.")