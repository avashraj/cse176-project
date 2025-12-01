import numpy as np
from matplotlib import pyplot
from numpy._typing import NDArray
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from typing import Tuple


def load_data(path: str = "MNISTmini") -> Tuple[list[NDArray], list[NDArray], list[NDArray], list[NDArray]]:
    """Load MNISTmini dataset from .mat file.
    
    Args:
        path: Path to the MNISTmini .mat file
        
    Returns:
        Tuple containing (train, train_labels, test, test_labels)
    """
    data = loadmat(path)
    train: list[NDArray] = data["train_fea1"]
    train_labels: list[NDArray] = data["train_gnd1"]
    test: list[NDArray] = data["test_fea1"]
    test_labels: list[NDArray] = data["test_gnd1"]
    return train, train_labels, test, test_labels


def create_datasets(
    train: list[NDArray],
    train_labels: list[NDArray],
    test: list[NDArray],
    test_labels: list[NDArray]
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Filter, shuffle, and limit datasets to 1000 elements each.
    
    Filters for labels 0 (represented as 10) and 6, shuffles the data,
    and limits training, validation, and testing sets to 1000 elements each.
    
    Args:
        train: Training features
        train_labels: Training labels
        test: Testing features
        test_labels: Testing labels
        
    Returns:
        Tuple containing (train_np, train_labels_np, validation_np, 
                         validation_labels_np, test_np, test_labels_np)
    """
    training_fr = []
    training_labels_fr = []
    testing_fr = []
    testing_labels_fr = []

    # Filter for labels 0 (10) and 6
    for X, y in zip(train, train_labels):
        y_val = y[0]
        if y_val == [10] or y_val == [6]:
            training_fr.append(X)
            training_labels_fr.append(y_val)

    for X, y in zip(test, test_labels):
        y_val = y[0]
        if y_val == [10] or y_val == [6]:
            testing_fr.append(X)
            testing_labels_fr.append(y_val)

    # Convert to numpy arrays
    train_np = np.array(training_fr)
    train_labels_np = np.array(training_labels_fr)
    test_np = np.array(testing_fr)
    test_labels_np = np.array(testing_labels_fr)

    # Shuffle the training arrays
    train_indices = np.random.permutation(len(train_np))
    train_np = train_np[train_indices]
    train_labels_np = train_labels_np[train_indices]

    # Shuffle the testing arrays
    test_indices = np.random.permutation(len(test_np))
    test_np = test_np[test_indices]
    test_labels_np = test_labels_np[test_indices]

    # Split training data: first 1000 for training, next 1000 for validation
    validation_np = train_np[1000:2000] if len(train_np) >= 2000 else train_np[1000:]
    validation_labels_np = train_labels_np[1000:2000] if len(train_labels_np) >= 2000 else train_labels_np[1000:]
    
    # Limit training to 1000
    train_np = train_np[:1000]
    train_labels_np = train_labels_np[:1000]
    
    # Limit validation to 1000
    validation_np = validation_np[:1000]
    validation_labels_np = validation_labels_np[:1000]
    
    # Limit testing to 1000
    test_np = test_np[:1000]
    test_labels_np = test_labels_np[:1000]
    
    return train_np, train_labels_np, validation_np, validation_labels_np, test_np, test_labels_np


def train_logistic_regression(
    train_np: NDArray,
    train_labels_np: NDArray,
    validation_np: NDArray,
    validation_labels_np: NDArray,
    test_np: NDArray,
    test_labels_np: NDArray
) -> Tuple[LogisticRegression, float]:
    """Train a LogisticRegression model with hyperparameter tuning using cross-validation.
    
    Tunes the L2 regularization parameter (C) using cross-validation on the validation set.
    Also plots training and validation errors for each C value.
    
    Args:
        train_np: Training features
        train_labels_np: Training labels
        validation_np: Validation features
        validation_labels_np: Validation labels
        test_np: Testing features
        test_labels_np: Testing labels
        
    Returns:
        Tuple containing (trained_model, accuracy_score)
    """
    # Combine training and validation for hyperparameter tuning
    combined_train = np.vstack([train_np, validation_np])
    combined_train_labels = np.vstack([train_labels_np, validation_labels_np])
    
    # Define parameter grid for C (inverse of regularization strength)
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    # Track errors for plotting
    train_errors = []
    validation_errors = []
    
    # Evaluate each C value on training and validation sets
    print("Evaluating Logistic Regression hyperparameters...")
    for C in C_values:
        model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000, C=C)
        model.fit(train_np, train_labels_np.ravel())
        
        # Calculate training error
        train_pred = model.predict(train_np)
        train_acc = accuracy_score(train_labels_np, train_pred)
        train_error = 1 - train_acc
        train_errors.append(train_error)
        
        # Calculate validation error
        val_pred = model.predict(validation_np)
        val_acc = accuracy_score(validation_labels_np, val_pred)
        val_error = 1 - val_acc
        validation_errors.append(val_error)
        
        print(f"C={C}: Train Error={train_error:.4f}, Val Error={val_error:.4f}")
    
    # Plot training and validation errors
    pyplot.figure(figsize=(10, 6))
    pyplot.plot(C_values, train_errors, 'o-', label='Training Error', linewidth=2, markersize=8)
    pyplot.plot(C_values, validation_errors, 's-', label='Validation Error', linewidth=2, markersize=8)
    pyplot.xscale('log')
    pyplot.xlabel('C (Regularization Parameter)', fontsize=12)
    pyplot.ylabel('Error', fontsize=12)
    pyplot.title('Logistic Regression: Training vs Validation Error', fontsize=14)
    pyplot.legend(fontsize=11)
    pyplot.grid(True, alpha=0.3)
    pyplot.tight_layout()
    pyplot.savefig('logistic_regression_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    print("Saved plot to: logistic_regression_hyperparameter_tuning.png")
    pyplot.show()
    
    # Perform grid search with cross-validation for final model selection
    param_grid = {'C': C_values}
    base_model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit on combined training and validation data
    grid_search.fit(combined_train, combined_train_labels.ravel())
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_C = grid_search.best_params_['C']
    
    print(f"Logistic Regression - Best C: {best_C}")
    print(f"Logistic Regression - Best CV Score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_predict = best_model.predict(test_np)
    accuracy = accuracy_score(test_labels_np, y_predict)
    print(f"Logistic Regression - Test accuracy: {accuracy:.4f}")
    
    return best_model, accuracy


def train_random_forest(
    train_np: NDArray,
    train_labels_np: NDArray,
    validation_np: NDArray,
    validation_labels_np: NDArray,
    test_np: NDArray,
    test_labels_np: NDArray
) -> Tuple[RandomForestClassifier, float]:
    """Train a RandomForestClassifier model with hyperparameter tuning using cross-validation.
    
    Tunes the number of trees (n_estimators) using cross-validation on the validation set.
    Also plots training and validation errors for each n_estimators value.
    
    Args:
        train_np: Training features
        train_labels_np: Training labels
        validation_np: Validation features
        validation_labels_np: Validation labels
        test_np: Testing features
        test_labels_np: Testing labels
        
    Returns:
        Tuple containing (trained_model, accuracy_score)
    """
    # Combine training and validation for hyperparameter tuning
    combined_train = np.vstack([train_np, validation_np])
    combined_train_labels = np.vstack([train_labels_np, validation_labels_np])
    
    # Define parameter grid for n_estimators (number of trees)
    n_estimators_values = [10, 50, 100, 200, 500]
    
    # Track errors for plotting
    train_errors = []
    validation_errors = []
    
    # Evaluate each n_estimators value on training and validation sets
    print("Evaluating Random Forest hyperparameters...")
    for n_est in n_estimators_values:
        model = RandomForestClassifier(n_estimators=n_est, random_state=42)
        model.fit(train_np, train_labels_np.ravel())
        
        # Calculate training error
        train_pred = model.predict(train_np)
        train_acc = accuracy_score(train_labels_np, train_pred)
        train_error = 1 - train_acc
        train_errors.append(train_error)
        
        # Calculate validation error
        val_pred = model.predict(validation_np)
        val_acc = accuracy_score(validation_labels_np, val_pred)
        val_error = 1 - val_acc
        validation_errors.append(val_error)
        
        print(f"n_estimators={n_est}: Train Error={train_error:.4f}, Val Error={val_error:.4f}")
    
    # Plot training and validation errors
    pyplot.figure(figsize=(10, 6))
    pyplot.plot(n_estimators_values, train_errors, 'o-', label='Training Error', linewidth=2, markersize=8)
    pyplot.plot(n_estimators_values, validation_errors, 's-', label='Validation Error', linewidth=2, markersize=8)
    pyplot.xlabel('Number of Trees (n_estimators)', fontsize=12)
    pyplot.ylabel('Error', fontsize=12)
    pyplot.title('Random Forest: Training vs Validation Error', fontsize=14)
    pyplot.legend(fontsize=11)
    pyplot.grid(True, alpha=0.3)
    pyplot.tight_layout()
    pyplot.savefig('random_forest_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    print("Saved plot to: random_forest_hyperparameter_tuning.png")
    pyplot.show()
    
    # Perform grid search with cross-validation for final model selection
    param_grid = {'n_estimators': n_estimators_values}
    base_model = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit on combined training and validation data
    grid_search.fit(combined_train, combined_train_labels.ravel())
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_n_estimators = grid_search.best_params_['n_estimators']
    
    print(f"Random Forest - Best n_estimators: {best_n_estimators}")
    print(f"Random Forest - Best CV Score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_predict = best_model.predict(test_np)
    accuracy = accuracy_score(test_labels_np, y_predict)
    print(f"Random Forest - Test accuracy: {accuracy:.4f}")
    
    return best_model, accuracy


def main():
    # Load data
    train, train_labels, test, test_labels = load_data()
    
    # Create datasets (filter, shuffle, limit to 1000)
    train_np, train_labels_np, validation_np, validation_labels_np, test_np, test_labels_np = create_datasets(
        train, train_labels, test, test_labels
    )
    
    # Train LogisticRegression model with hyperparameter tuning
    lr_model, lr_accuracy = train_logistic_regression(
        train_np, train_labels_np, validation_np, validation_labels_np, test_np, test_labels_np
    )
    
    # Train RandomForestClassifier model with hyperparameter tuning
    rf_model, rf_accuracy = train_random_forest(
        train_np, train_labels_np, validation_np, validation_labels_np, test_np, test_labels_np
    )
    

if __name__ == "__main__":
    main()
