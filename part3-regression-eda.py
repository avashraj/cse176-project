import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

os.makedirs('eda_plots', exist_ok=True)

print("Downloading dataset...")
path = kagglehub.dataset_download("yasserh/uber-fares-dataset")
print(f"Path to dataset files: {path}")

# Find the CSV file in the downloaded directory
csv_files = list(Path(path).glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {path}")
csv_file = csv_files[0]
print(f"Loading data from: {csv_file}")

# Load the dataset
df = pd.read_csv(csv_file)

target_col = 'fare_amount'

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")
print(f"Samples: {df.shape[0]}")
print(f"\nTarget column: {target_col}")
print(f"\nColumn names: {list(df.columns)}")

print("\n" + "="*80)
print("BASIC INFORMATION")
print("="*80)
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total missing values")
if df.isnull().sum().sum() > 0:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\nFeatures with missing values:\n{missing}")

# Convert pickup_datetime to datetime
if 'pickup_datetime' in df.columns:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    
    # Extract time features
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_dayofweek_name'] = df['pickup_datetime'].dt.day_name()

# Calculate distance from coordinates (Haversine distance approximation)
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth in kilometers"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 
                                      'dropoff_latitude', 'dropoff_longitude']):
    df['distance_km'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

# ============================================================================
# 1. TARGET VARIABLE DISTRIBUTION (fare_amount)
# ============================================================================
print("\n" + "="*80)
print("1. TARGET VARIABLE DISTRIBUTION")
print("="*80)
target_stats = df[target_col].describe()
print(f"\nTarget variable statistics:\n{target_stats}")

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(df[target_col], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
plt.xlabel('Fare Amount (USD)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Fare Amount', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_plots/01_target_distribution_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/01_target_distribution_histogram.png")

# Box plot
plt.figure(figsize=(8, 6))
plt.boxplot(df[target_col], vert=True, patch_artist=True,
            boxprops=dict(facecolor='#3498db', alpha=0.7))
plt.ylabel('Fare Amount (USD)', fontsize=12)
plt.title('Fare Amount Distribution (Box Plot)', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_plots/01_target_distribution_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/01_target_distribution_boxplot.png")

# Log transformation if needed (for skewed data)
if df[target_col].skew() > 1:
    plt.figure(figsize=(10, 6))
    plt.hist(np.log1p(df[target_col]), bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
    plt.xlabel('Log(Fare Amount + 1)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Log-Transformed Fare Amount', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/01_target_distribution_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/01_target_distribution_log.png")
    print(f"Note: Target is right-skewed (skewness={df[target_col].skew():.2f}), log transformation shown")

# ============================================================================
# 2. FEATURE DISTRIBUTIONS
# ============================================================================
print("\n" + "="*80)
print("2. FEATURE DISTRIBUTIONS")
print("="*80)

# Get numeric features (exclude target and datetime)
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_features:
    numeric_features.remove(target_col)
if 'key' in numeric_features:
    numeric_features.remove('key')

print(f"Number of numeric features: {len(numeric_features)}")
print(f"Numeric features: {numeric_features}")

# Plot distributions for numeric features
n_features = len(numeric_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
if n_features == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    ax = axes[idx]
    df[feature].hist(bins=50, ax=ax, color='#3498db', edgecolor='black', alpha=0.7)
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
stats_summary = df[numeric_features + [target_col]].describe()
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

# Calculate correlation matrix for numeric features including target
correlation_features = numeric_features + [target_col]
correlation_matrix = df[correlation_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Feature Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/05_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/05_correlation_matrix.png")

# Find features highly correlated with target
target_correlations = correlation_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
print(f"\nFeatures most correlated with {target_col}:")
for feature, corr in target_correlations.head(10).items():
    print(f"  {feature}: {corr:.3f}")

# Find highly correlated feature pairs
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
# 5. FEATURE-TARGET RELATIONSHIPS (Scatter plots for regression)
# ============================================================================
print("\n" + "="*80)
print("5. FEATURE-TARGET RELATIONSHIPS")
print("="*80)

# Select top features by correlation with target
top_features = target_correlations.head(min(6, len(numeric_features))).index.tolist()

n_features_plot = len(top_features)
n_cols = 3
n_rows = (n_features_plot + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
if n_features_plot == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, feature in enumerate(top_features):
    ax = axes[idx]
    # Sample data if too large for scatter plot
    if len(df) > 10000:
        sample_df = df.sample(n=10000, random_state=42)
    else:
        sample_df = df
    ax.scatter(sample_df[feature], sample_df[target_col], alpha=0.3, s=10, color='#3498db')
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel(target_col, fontsize=10)
    ax.set_title(f'{feature} vs {target_col}', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add correlation coefficient
    corr = df[feature].corr(df[target_col])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide extra subplots
for idx in range(n_features_plot, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Feature-Target Relationships (Scatter Plots - Top Features by Correlation)', 
             fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('eda_plots/06_feature_target_relationships.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/06_feature_target_relationships.png")

# ============================================================================
# 6. OUTLIER DETECTION
# ============================================================================
print("\n" + "="*80)
print("6. OUTLIER DETECTION")
print("="*80)

# Create box plots to identify outliers
n_features = len(numeric_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
if n_features == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    ax = axes[idx]
    df[feature].plot(kind='box', ax=ax, color='#3498db')
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
# 7. TEMPORAL ANALYSIS (if datetime available)
# ============================================================================
if 'pickup_datetime' in df.columns and df['pickup_datetime'].notna().sum() > 0:
    print("\n" + "="*80)
    print("7. TEMPORAL ANALYSIS")
    print("="*80)
    
    # Fare by hour
    if 'pickup_hour' in df.columns:
        fare_by_hour = df.groupby('pickup_hour')[target_col].mean()
        plt.figure(figsize=(12, 6))
        plt.plot(fare_by_hour.index, fare_by_hour.values, marker='o', color='#3498db', linewidth=2)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Average Fare Amount (USD)', fontsize=12)
        plt.title('Average Fare Amount by Hour of Day', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.xticks(range(24))
        plt.tight_layout()
        plt.savefig('eda_plots/08_temporal_hour.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: eda_plots/08_temporal_hour.png")
    
    # Fare by day of week
    if 'pickup_dayofweek_name' in df.columns:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fare_by_day = df.groupby('pickup_dayofweek_name')[target_col].mean().reindex(day_order)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(fare_by_day)), fare_by_day.values, color='#3498db', alpha=0.7)
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('Average Fare Amount (USD)', fontsize=12)
        plt.title('Average Fare Amount by Day of Week', fontsize=14, fontweight='bold')
        plt.xticks(range(len(fare_by_day)), fare_by_day.index, rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('eda_plots/08_temporal_dayofweek.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: eda_plots/08_temporal_dayofweek.png")

# ============================================================================
# 8. PASSENGER COUNT ANALYSIS
# ============================================================================
if 'passenger_count' in df.columns:
    print("\n" + "="*80)
    print("8. PASSENGER COUNT ANALYSIS")
    print("="*80)
    
    passenger_counts = df['passenger_count'].value_counts().sort_index()
    print(f"\nPassenger count distribution:\n{passenger_counts}")
    
    fare_by_passengers = df.groupby('passenger_count')[target_col].mean()
    
    plt.figure(figsize=(10, 6))
    plt.bar(fare_by_passengers.index, fare_by_passengers.values, color='#2ecc71', alpha=0.7)
    plt.xlabel('Passenger Count', fontsize=12)
    plt.ylabel('Average Fare Amount (USD)', fontsize=12)
    plt.title('Average Fare Amount by Passenger Count', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/09_passenger_count_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/09_passenger_count_analysis.png")

# ============================================================================
# 9. DISTANCE ANALYSIS (if calculated)
# ============================================================================
if 'distance_km' in df.columns:
    print("\n" + "="*80)
    print("9. DISTANCE ANALYSIS")
    print("="*80)
    
    # Distance vs Fare scatter plot
    plt.figure(figsize=(10, 6))
    if len(df) > 10000:
        sample_df = df.sample(n=10000, random_state=42)
    else:
        sample_df = df
    plt.scatter(sample_df['distance_km'], sample_df[target_col], alpha=0.3, s=10, color='#e74c3c')
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Fare Amount (USD)', fontsize=12)
    plt.title('Distance vs Fare Amount', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    # Add correlation
    corr = df['distance_km'].corr(df[target_col])
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('eda_plots/10_distance_fare_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/10_distance_fare_relationship.png")
    
    # Distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['distance_km'], bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Trip Distance', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/10_distance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/10_distance_distribution.png")

# ============================================================================
# 10. FEATURE IMPORTANCE INDICATORS (Correlation with target)
# ============================================================================
print("\n" + "="*80)
print("10. FEATURE IMPORTANCE INDICATORS")
print("="*80)

# Calculate feature correlations with target
feature_importance_indicators = []
for feature in numeric_features:
    corr = abs(df[feature].corr(df[target_col]))
    feature_importance_indicators.append({
        'feature': feature,
        'abs_correlation_with_target': corr,
        'variance': df[feature].var(),
        'std': df[feature].std()
    })

importance_df = pd.DataFrame(feature_importance_indicators)
importance_df = importance_df.sort_values('abs_correlation_with_target', ascending=False)

# Plot top features by correlation
top_n = min(15, len(importance_df))
plt.figure(figsize=(12, 8))
top_features_plot = importance_df.head(top_n)
plt.barh(range(len(top_features_plot)), top_features_plot['abs_correlation_with_target'], 
         color='#2ecc71')
plt.yticks(range(len(top_features_plot)), top_features_plot['feature'])
plt.xlabel('Absolute Correlation with Target', fontsize=12)
plt.title(f'Top {top_n} Features by Correlation with Fare Amount', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_plots/11_feature_importance_indicators.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/11_feature_importance_indicators.png")

# Save importance indicators to CSV
importance_df.to_csv('eda_plots/11_feature_importance_indicators.csv', index=False)
print("Saved: eda_plots/11_feature_importance_indicators.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EDA COMPLETE - SUMMARY")
print("="*80)
print(f"\nTotal plots saved: Multiple visualizations")
print(f"Output directory: eda_plots/")
print("\nGenerated visualizations:")
print("  1. Target distribution (histogram)")
print("  2. Target distribution (box plot)")
print("  3. Feature distributions (histograms)")
print("  4. Feature statistics summary (CSV)")
print("  5. Correlation matrix")
print("  6. Feature-target relationships (scatter plots)")
print("  7. Outlier detection (box plots)")
print("  8. Temporal analysis (if datetime available)")
print("  9. Passenger count analysis")
print("  10. Distance analysis (if coordinates available)")
print("  11. Feature importance indicators")
print("\nAll plots saved as high-resolution PNG images (300 DPI)")
print("="*80)
