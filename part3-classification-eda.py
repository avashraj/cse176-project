import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

os.makedirs('eda_plots', exist_ok=True)

print("Fetching dataset...")
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

df = pd.concat([X, y], axis=1)

target_col = y.columns[0]

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Dataset shape: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"\nTarget column: {target_col}")

print("\n" + "="*80)
print("BASIC INFORMATION")
print("="*80)
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total missing values")
if df.isnull().sum().sum() > 0:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\nFeatures with missing values:\n{missing}")

# ============================================================================
# 1. CLASS DISTRIBUTION (Target Variable)
# ============================================================================
print("\n" + "="*80)
print("1. CLASS DISTRIBUTION")
print("="*80)
class_counts = y[target_col].value_counts().sort_index()
class_props = y[target_col].value_counts(normalize=True).sort_index()
print(f"\nClass distribution:\n{class_counts}")
print(f"\nClass proportions:\n{class_props}")

# Map class labels: 0 -> No Diabetes, 1 -> Diabetes
class_labels = {0: 'No Diabetes', 1: 'Diabetes'}
label_names = [class_labels.get(idx, f'Class {idx}') for idx in class_counts.index]

# Bar plot - separate figure
plt.figure(figsize=(8, 6))
bars = plt.bar(label_names, class_counts.values, color=['#3498db', '#e74c3c'])
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Class Distribution (Count)', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for i, (bar, v) in enumerate(zip(bars, class_counts.values)):
    plt.text(bar.get_x() + bar.get_width()/2, v + max(class_counts.values)*0.01, 
             str(v), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/01_class_distribution_count.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/01_class_distribution_count.png")

# Pie chart - separate figure
plt.figure(figsize=(8, 6))
pie_labels = [f'{label}\n({prop:.1%})' for label, prop in zip(label_names, class_props.values)]
plt.pie(class_counts.values, labels=pie_labels, autopct='%1.1f%%', startangle=90, 
        colors=['#3498db', '#e74c3c'])
plt.title('Class Distribution (Proportion)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/01_class_distribution_proportion.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/01_class_distribution_proportion.png")


# ============================================================================
# 2. FEATURE DISTRIBUTIONS (Histograms for numeric features)
# ============================================================================
print("\n" + "="*80)
print("2. FEATURE DISTRIBUTIONS")
print("="*80)

# Get numeric features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Number of numeric features: {len(numeric_features)}")

# Plot distributions for all numeric features
n_features = len(numeric_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, feature in enumerate(numeric_features):
    ax = axes[idx]
    X[feature].hist(bins=50, ax=ax, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Hide extra subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Feature Distributions (Histograms)', fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('eda_plots/03_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/03_feature_distributions.png")

# ============================================================================
# 3. FEATURE STATISTICS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("3. FEATURE STATISTICS SUMMARY")
print("="*80)
stats_summary = X.describe()
print(f"\nStatistical Summary:\n{stats_summary}")

# Save statistics to CSV
stats_summary.to_csv('eda_plots/04_feature_statistics.csv')
print("Saved: eda_plots/04_feature_statistics.csv")

# ============================================================================
# 4. CORRELATION MATRIX
# ============================================================================
print("\n" + "="*80)
print("4. CORRELATION MATRIX")
print("="*80)

# Calculate correlation matrix for numeric features
correlation_matrix = X[numeric_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Feature Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/05_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/05_correlation_matrix.png")

# Find highly correlated features (for tree ensembles, this is less critical but still informative)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:  # Threshold for high correlation
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))

if high_corr_pairs:
    print(f"\nHighly correlated feature pairs (|r| > 0.7):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")

# ============================================================================
# 5. FEATURE-TARGET RELATIONSHIPS (Box plots for numeric features)
# ============================================================================
print("\n" + "="*80)
print("5. FEATURE-TARGET RELATIONSHIPS")
print("="*80)

# Select top features by variance (to avoid too many plots)
feature_variances = X[numeric_features].var().sort_values(ascending=False)
top_features = feature_variances.head(min(12, len(numeric_features))).index.tolist()

n_features_plot = len(top_features)
n_cols = 3
n_rows = (n_features_plot + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten() if n_features_plot > 1 else [axes]

# Map class labels for plotting
class_labels = {0: 'No Diabetes', 1: 'Diabetes'}

for idx, feature in enumerate(top_features):
    ax = axes[idx]
    df_plot = pd.DataFrame({feature: X[feature], target_col: y[target_col]})
    # Map numeric class values to labels
    df_plot['Class_Label'] = df_plot[target_col].map(class_labels)
    sns.boxplot(data=df_plot, x='Class_Label', y=feature, ax=ax, palette=['#3498db', '#e74c3c'])
    ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Class', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Hide extra subplots
for idx in range(n_features_plot, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Feature-Target Relationships (Box Plots - Top Features by Variance)', 
             fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('eda_plots/06_feature_target_relationships.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/06_feature_target_relationships.png")

# ============================================================================
# 6. OUTLIER DETECTION (Box plots for all features)
# ============================================================================
print("\n" + "="*80)
print("6. OUTLIER DETECTION")
print("="*80)

# Create box plots to identify outliers
n_features = len(numeric_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, feature in enumerate(numeric_features):
    ax = axes[idx]
    X[feature].plot(kind='box', ax=ax, color='#3498db')
    ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
    ax.set_ylabel('Value', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Hide extra subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Outlier Detection (Box Plots)', fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('eda_plots/07_outlier_detection.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/07_outlier_detection.png")

# ============================================================================
# 7. FEATURE IMPORTANCE INDICATORS (Variance and Mutual Information-like)
# ============================================================================
print("\n" + "="*80)
print("7. FEATURE IMPORTANCE INDICATORS")
print("="*80)

# Calculate feature variances
feature_variances = X[numeric_features].var().sort_values(ascending=False)

# Calculate simple feature-target relationships (mean difference between classes)
feature_importance_indicators = []
for feature in numeric_features:
    class_0_mean = X[y[target_col] == y[target_col].unique()[0]][feature].mean()
    class_1_mean = X[y[target_col] == y[target_col].unique()[1]][feature].mean()
    mean_diff = abs(class_0_mean - class_1_mean)
    feature_importance_indicators.append({
        'feature': feature,
        'variance': X[feature].var(),
        'mean_diff_between_classes': mean_diff,
        'std': X[feature].std()
    })

importance_df = pd.DataFrame(feature_importance_indicators)
importance_df = importance_df.sort_values('mean_diff_between_classes', ascending=False)

# Plot top features by mean difference
top_n = min(15, len(importance_df))
plt.figure(figsize=(12, 8))
top_features_plot = importance_df.head(top_n)
plt.barh(range(len(top_features_plot)), top_features_plot['mean_diff_between_classes'], 
         color='#2ecc71')
plt.yticks(range(len(top_features_plot)), top_features_plot['feature'])
plt.xlabel('Absolute Mean Difference Between Classes', fontsize=12)
plt.title(f'Top {top_n} Features by Class Separation (Mean Difference)', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_plots/08_feature_importance_indicators.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/08_feature_importance_indicators.png")

# Save importance indicators to CSV
importance_df.to_csv('eda_plots/08_feature_importance_indicators.csv', index=False)
print("Saved: eda_plots/08_feature_importance_indicators.csv")

# ============================================================================
# 8. DATA TYPES AND CATEGORICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("8. CATEGORICAL FEATURES ANALYSIS")
print("="*80)

categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
if len(categorical_features) > 0:
    print(f"Number of categorical features: {len(categorical_features)}")
    for feature in categorical_features:
        print(f"\n{feature}:")
        print(X[feature].value_counts())
else:
    print("No categorical features found (all features are numeric).")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EDA COMPLETE - SUMMARY")
print("="*80)
print(f"\nTotal plots saved: 9")
print(f"Output directory: eda_plots/")
print("\nGenerated visualizations:")
print("  1. Class distribution (count)")
print("  1. Class distribution (proportion)")
print("  2. Missing values analysis")
print("  3. Feature distributions (histograms)")
print("  4. Feature statistics summary (CSV)")
print("  5. Correlation matrix")
print("  6. Feature-target relationships (box plots)")
print("  7. Outlier detection (box plots)")
print("  8. Feature importance indicators")
print("\nAll plots saved as high-resolution PNG images (300 DPI)")
print("="*80)
