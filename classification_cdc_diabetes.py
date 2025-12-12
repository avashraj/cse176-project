"""
CDC Diabetes Health Indicators (UCI ML Repo id=891)

This script:
- fetches the dataset via ucimlrepo
- prints dataset metadata + variable info
- prints basic statistics (shapes, missingness, describe, target distribution)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo


def main() -> None:
    # fetch dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

    # data (as pandas dataframes)
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets

    # metadata
    print("=== METADATA ===")
    print(cdc_diabetes_health_indicators.metadata)

    # variable information
    print("\n=== VARIABLES ===")
    print(cdc_diabetes_health_indicators.variables)

    # plots output directory
    os.makedirs("eda_plots", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # basic statistics
    print("\n=== BASIC STATS ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\n--- X dtypes (top 30) ---")
    print(X.dtypes.head(30))

    print("\n--- Missing values ---")
    x_missing = X.isna().sum()
    y_missing = y.isna().sum()
    print(f"X missing cells: {int(x_missing.sum())} / {X.shape[0] * X.shape[1]}")
    print(f"y missing cells: {int(y_missing.sum().sum())} / {y.shape[0] * y.shape[1]}")
    print("\nX missing by column (non-zero only, top 30):")
    print(x_missing[x_missing > 0].sort_values(ascending=False).head(30))
    print("\ny missing by column (non-zero only):")
    print(y_missing[y_missing > 0])

    print("\n--- X describe() ---")
    # include='all' can be very wide/noisy; default gives numeric summary if applicable
    print(X.describe().transpose())

    # Target distribution (handles single-column targets cleanly)
    print("\n--- Target distribution ---")
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        target_series = y.iloc[:, 0].astype(int)
        print(target_series.value_counts(dropna=False).sort_index())
        print("\nTarget proportions:")
        print(target_series.value_counts(normalize=True, dropna=False).sort_index())
    else:
        print("y columns:", list(y.columns) if isinstance(y, pd.DataFrame) else type(y))
        print(y.head())

    # ---------------------------
    # Requested plots
    # ---------------------------

    # 1) Target imbalance (pie chart)
    target_counts = target_series.value_counts().sort_index()
    target_pct = (target_counts / target_counts.sum() * 100).round(2)

    plt.figure(figsize=(7, 7))
    plt.pie(
        target_counts.values,
        labels=[
            f"{label}\n{count:,} ({pct:.2f}%)"
            for label, count, pct in zip(target_counts.index, target_counts.values, target_pct.values)
        ],
        startangle=90,
    )
    plt.title("Target imbalance (Diabetes_binary)")
    plt.tight_layout()
    plt.savefig("eda_plots/diabetes_target_imbalance_pie.png", dpi=200)
    plt.close()

    # 2) Age vs target + age summary table
    AGE_LABELS = {
        1: "Age 18 to 24",
        2: "Age 25 to 29",
        3: "Age 30 to 34",
        4: "Age 35 to 39",
        5: "Age 40 to 44",
        6: "Age 45 to 49",
        7: "Age 50 to 54",
        8: "Age 55 to 59",
        9: "Age 60 to 64",
        10: "Age 65 to 69",
        11: "Age 70 to 74",
        12: "Age 75 to 79",
        13: "Age 80 or older",
    }

    age = X["Age"].astype(int)
    age_freq = age.value_counts().sort_index()
    age_pct = age_freq / age_freq.sum() * 100

    age_summary = pd.DataFrame(
        {
            "Value": age_freq.index,
            "Value Label": [AGE_LABELS.get(v, f"Age code {v}") for v in age_freq.index],
            "Frequency": age_freq.values,
            "Percentage": age_pct.values,
            # No weights provided; match request by repeating Percentage as Weighted Percentage.
            "Weighted Percentage": age_pct.values,
        }
    )

    age_summary_out = age_summary.copy()
    age_summary_out["Percentage"] = age_summary_out["Percentage"].round(2)
    age_summary_out["Weighted Percentage"] = age_summary_out["Weighted Percentage"].round(2)

    print("\n=== AGE DISTRIBUTION TABLE ===")
    print(age_summary_out.to_string(index=False))
    age_summary_out.to_csv("eda_plots/age_distribution_table.csv", index=False)

    # Plot: frequency by age group
    plt.figure(figsize=(13, 6))
    sns.barplot(data=age_summary, x="Value Label", y="Frequency")
    plt.title("Age distribution (Frequency)")
    plt.xlabel("Age group")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("eda_plots/age_distribution_frequency.png", dpi=200)
    plt.close()

    # Plot: diabetes prevalence by age group (mean of binary target)
    age_prevalence = target_series.groupby(age).mean().reindex(age_freq.index) * 100
    prev_df = pd.DataFrame(
        {"Value": age_prevalence.index, "Value Label": [AGE_LABELS.get(v, str(v)) for v in age_prevalence.index], "Prevalence (%)": age_prevalence.values}
    )

    plt.figure(figsize=(13, 6))
    sns.barplot(data=prev_df, x="Value Label", y="Prevalence (%)", color=sns.color_palette()[1])
    plt.title("Diabetes prevalence by Age group")
    plt.xlabel("Age group")
    plt.ylabel("P(Diabetes_binary=1) (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("eda_plots/age_vs_target_prevalence.png", dpi=200)
    plt.close()

    # 2b) GenHlth vs target (distribution + prevalence)
    # GenHlth scale: 1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor (per common BRFSS coding)
    GENHLTH_LABELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
    gen = X["GenHlth"].astype(int)

    gen_freq = gen.value_counts().sort_index()
    gen_pct = gen_freq / gen_freq.sum() * 100
    gen_summary = pd.DataFrame(
        {
            "Value": gen_freq.index,
            "Value Label": [GENHLTH_LABELS.get(v, str(v)) for v in gen_freq.index],
            "Frequency": gen_freq.values,
            "Percentage": gen_pct.values,
            "Weighted Percentage": gen_pct.values,
        }
    )
    gen_summary_out = gen_summary.copy()
    gen_summary_out["Percentage"] = gen_summary_out["Percentage"].round(2)
    gen_summary_out["Weighted Percentage"] = gen_summary_out["Weighted Percentage"].round(2)

    print("\n=== GENHLTH DISTRIBUTION TABLE ===")
    print(gen_summary_out.to_string(index=False))
    gen_summary_out.to_csv("eda_plots/genhlth_distribution_table.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=gen_summary, x="Value Label", y="Frequency")
    plt.title("GenHlth distribution (Frequency)")
    plt.xlabel("GenHlth")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("eda_plots/genhlth_distribution_frequency.png", dpi=200)
    plt.close()

    gen_prevalence = target_series.groupby(gen).mean().reindex(gen_freq.index) * 100
    gen_prev_df = pd.DataFrame(
        {
            "Value": gen_prevalence.index,
            "Value Label": [GENHLTH_LABELS.get(v, str(v)) for v in gen_prevalence.index],
            "Prevalence (%)": gen_prevalence.values,
        }
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=gen_prev_df, x="Value Label", y="Prevalence (%)", color=sns.color_palette()[1])
    plt.title("Diabetes prevalence by GenHlth")
    plt.xlabel("GenHlth")
    plt.ylabel("P(Diabetes_binary=1) (%)")
    plt.tight_layout()
    plt.savefig("eda_plots/genhlth_vs_target_prevalence.png", dpi=200)
    plt.close()

    # 3) Max mean difference (top 10): |mean1 - mean0|
    X_num = X.astype(float)
    X_arr = X_num.to_numpy()
    y_arr = target_series.to_numpy()
    mask1 = y_arr == 1
    mask0 = y_arr == 0

    mu1 = X_arr[mask1].mean(axis=0)
    mu0 = X_arr[mask0].mean(axis=0)
    mmd = np.abs(mu1 - mu0)

    mmd_df = (
        pd.DataFrame({"feature": X.columns, "max_mean_diff": mmd})
        .sort_values("max_mean_diff", ascending=False)
        .reset_index(drop=True)
    )
    top10_mmd = mmd_df.head(10)
    top10_mmd.to_csv("eda_plots/top10_max_mean_difference.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top10_mmd, y="feature", x="max_mean_diff", orient="h")
    plt.title("Top 10 features by Max Mean Difference |mean(y=1) - mean(y=0)|")
    plt.xlabel("Max Mean Difference")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("eda_plots/top10_max_mean_difference.png", dpi=200)
    plt.close()

    # 4) Fisher's Discriminant Ratio (FDR) top 10:
    # FDR = (mu1 - mu0)^2 / (var1 + var0)
    var1 = X_arr[mask1].var(axis=0, ddof=0)
    var0 = X_arr[mask0].var(axis=0, ddof=0)
    fdr = (mu1 - mu0) ** 2 / (var1 + var0 + 1e-12)

    fdr_df = (
        pd.DataFrame({"feature": X.columns, "fdr": fdr})
        .sort_values("fdr", ascending=False)
        .reset_index(drop=True)
    )
    top10_fdr = fdr_df.head(10)
    top10_fdr.to_csv("eda_plots/top10_fdr.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top10_fdr, y="feature", x="fdr", orient="h")
    plt.title("Top 10 features by Fisher's Discriminant Ratio (FDR)")
    plt.xlabel("FDR")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("eda_plots/top10_fdr.png", dpi=200)
    plt.close()

    print("\n--- Preview ---")
    print("X head():")
    print(X.head())
    print("\ny head():")
    print(y.head())


if __name__ == "__main__":
    main()


