import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

def plot_error_vs_trees(tree_list, test_errors, title, filename):
    """Plot test error as a function of the number of trees and save to PDF."""
    plt.figure()
    plt.plot(tree_list, test_errors, marker="o")
    plt.xlabel("Number of Trees")
    plt.ylabel("Test Error")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

def plot_confusion_matrix(cm, classes, title, filename):
    """Plot confusion matrix as an image and save to PDF."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix: {filename}")

def load_mnist_pixel(mat_path):
    """
    Load MNIST.mat with raw pixel features.

    Expected keys:
        train_fea: (60000, 784)
        train_gnd: (60000, 1) labels 1..10
        test_fea:  (10000, 784)
        test_gnd:  (10000, 1)
    """
    data = sio.loadmat(mat_path)
    X_train = data["train_fea"].astype(np.float32) / 255.0
    y_train = data["train_gnd"].ravel().astype(int) - 1  # convert 1..10 -> 0..9
    X_test = data["test_fea"].astype(np.float32) / 255.0
    y_test = data["test_gnd"].ravel().astype(int) - 1
    return X_train, y_train, X_test, y_test

def load_mnist_lenet5(mat_path):
    """
    Load MNIST-LeNet5.mat with LeNet-5 features.

    Expected same label format as MNIST.mat:
        train_fea: (60000, 800)
        train_gnd: (60000, 1) labels 1..10
        test_fea:  (10000, 800)
        test_gnd:  (10000, 1)
    """
    data = sio.loadmat(mat_path)
    X_train = data["train_fea"].astype(np.float32)
    y_train = data["train_gnd"].ravel().astype(int) - 1  # 1..10 -> 0..9
    X_test = data["test_fea"].astype(np.float32)
    y_test = data["test_gnd"].ravel().astype(int) - 1
    return X_train, y_train, X_test, y_test


def run_xgboost_experiment(name, X_train, y_train, X_test, y_test, tree_list=None, random_state=42):
    """
    Steps:
        - Split training into train_sub + validation
        - Loop over number of trees, train model, record val & test error
        - Choose best number of trees based on validation error
        - Retrain on full training set with best settings
        - Compute final test error & confusion matrix
        - Save plots and result summary file
    """
    print(f"\n=== Running experiment: {name} ===")

    if tree_list is None:
        tree_list = [100, 300, 500, 800, 1200, 1500]

    # Split off a validation set (e.g., 5,000 examples)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=5000,
        stratify=y_train,
        random_state=random_state,
    )

    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")

    val_errors = []
    test_errors = []

    # Loop over number of trees
    for n_trees in tree_list:
        print(f"Training with {n_trees} trees...")

        model = XGBClassifier(
            n_estimators=n_trees,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=num_classes,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=8,
            random_state=random_state,
            tree_method="hist",
            device="cuda"
        )

        model.fit(X_train_sub, y_train_sub)

        # Validation error
        val_pred = model.predict(X_val)
        val_err = 1.0 - np.mean(val_pred == y_val)
        val_errors.append(val_err)

        # Test error (for the required plot)
        test_pred = model.predict(X_test)
        test_err = 1.0 - np.mean(test_pred == y_test)
        test_errors.append(test_err)

        print(f"  Validation error: {val_err * 100:.2f}%")
        print(f"  Test error:       {test_err * 100:.2f}%")

    # Choose best number of trees based on validation error
    best_idx = int(np.argmin(val_errors))
    best_trees = tree_list[best_idx]
    best_val_err = val_errors[best_idx]
    best_test_err_at_cv = test_errors[best_idx]

    print(f"\nBest number of trees (by validation error): {best_trees}")
    print(f"Validation error at best: {best_val_err * 100:.2f}%")
    print(f"Test error at best (from CV loop): {best_test_err_at_cv * 100:.2f}%")

    # Retrain on full training data with best number of trees
    print("\nRetraining final model on full training set...")
    final_model = XGBClassifier(
        n_estimators=best_trees,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=num_classes,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=8,
        random_state=random_state,
        tree_method="hist",
        device="cuda",
    )
    final_model.fit(X_train, y_train)

    final_test_pred = final_model.predict(X_test)
    final_test_err = 1.0 - np.mean(final_test_pred == y_test)

    print(f"\n=== Final test error ({name}) ===")
    print(f"{final_test_err * 100:.2f}%")

    # Confusion matrix on test set
    cm = confusion_matrix(y_test, final_test_pred)
    classes = list(range(num_classes))

    # Save plot: test error vs #trees
    plot_error_vs_trees(
        tree_list,
        test_errors,
        title=f"{name}: Test Error vs Number of Trees",
        filename=f"{name}_test_error_vs_trees.pdf",
    )

    # Save confusion matrix plot
    plot_confusion_matrix(
        cm,
        classes=classes,
        title=f"{name}: Confusion Matrix",
        filename=f"{name}_confusion_matrix.pdf",
    )

    # Save summary text file
    results_filename = f"{name}_results.txt"
    with open(results_filename, "w") as f:
        f.write(f"{name} results\n")
        f.write(f"Best number of trees: {best_trees}\n")
        f.write(f"Validation error at best: {best_val_err * 100:.4f}%\n")
        f.write(f"Test error at best (CV loop): {best_test_err_at_cv * 100:.4f}%\n")
        f.write(f"Final test error (retrained on all training data): {final_test_err * 100:.4f}%\n")
        f.write("Hyperparameters:\n")
        f.write("  max_depth = 6\n")
        f.write("  learning_rate = 0.1\n")
        f.write("  subsample = 0.8\n")
        f.write("  colsample_bytree = 0.8\n")

    print(f"Saved summary: {results_filename}")
    print(f"=== Done with {name} ===\n")

    return {
        "best_trees": best_trees,
        "final_test_error": final_test_err,
        "cm": cm,
        "tree_list": tree_list,
        "test_errors": test_errors,
        "val_errors": val_errors,
    }

if __name__ == "__main__":
    pixel_mat_path = "MNIST.mat"
    lenet_mat_path = "MNIST-LeNet5.mat"

    if not os.path.exists(pixel_mat_path):
        raise FileNotFoundError(f"Could not find {pixel_mat_path} in current directory.")

    # 1. Pixel features experiment
    X_train_pix, y_train_pix, X_test_pix, y_test_pix = load_mnist_pixel(pixel_mat_path)
    results_pixel = run_xgboost_experiment(
        name="MNIST_pixel",
        X_train=X_train_pix,
        y_train=y_train_pix,
        X_test=X_test_pix,
        y_test=y_test_pix,
    )

    # 2. LeNet-5 features experiment (if file is present)
    if os.path.exists(lenet_mat_path):
        X_train_l5, y_train_l5, X_test_l5, y_test_l5 = load_mnist_lenet5(lenet_mat_path)
        results_lenet5 = run_xgboost_experiment(
            name="MNIST_LeNet5",
            X_train=X_train_l5,
            y_train=y_train_l5,
            X_test=X_test_l5,
            y_test=y_test_l5,
        )
    else:
        print(f"\nWARNING: {lenet_mat_path} not found. Skipping LeNet-5 experiment.")