"""Uber fare EDA + dataset construction.

This file is designed to be:
- runnable as a script (generates plots + prints summaries)
- importable as a module (for model training), without executing plots

Datasets:
- df_all: raw dataset
- df_manhattan: Manhattan-only (pickup & dropoff in Manhattan-ish mask)
- df_manhattan_fe: feature engineered Manhattan dataset
- df_manhattan_clean: df_manhattan_fe with fares >= $1
- df_manhattan_iqr: df_manhattan_clean with IQR-filtered fare outliers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Filtering / feature engineering
# ----------------------------

MANHATTAN_BOUNDS = {
    "lon_min": -74.02,
    "lon_max": -73.93,
    "lat_min": 40.70,
    "lat_max": 40.88,
}


def is_in_manhattan(lon: pd.Series, lat: pd.Series) -> pd.Series:
    """Approximate Manhattan mask + small exclusions across the East River."""
    in_bounds = (
        (lon >= MANHATTAN_BOUNDS["lon_min"]) & (lon <= MANHATTAN_BOUNDS["lon_max"]) &
        (lat >= MANHATTAN_BOUNDS["lat_min"]) & (lat <= MANHATTAN_BOUNDS["lat_max"])
    )

    # Exclude bottom-right corner (Brooklyn/DUMBO-ish)
    in_brooklyn_corner = (lat < 40.735) & (lon > -73.97)

    # Exclude right lower-middle side (Queens/LIC-ish) – conservative threshold
    in_queens_side = (lat >= 40.735) & (lat < 40.76) & (lon > -73.945)

    return in_bounds & ~in_brooklyn_corner & ~in_queens_side


def haversine_distance_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in miles."""
    r = 3959.0  # Earth radius (miles)

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def get_time_bin(hour: int) -> str:
    """6-bin time-of-day bucket."""
    if hour < 6:
        return "Late Night"
    if hour < 10:
        return "Morning"
    if hour < 15:
        return "Midday"
    if hour < 18:
        return "Afternoon"
    if hour < 22:
        return "Evening"
    return "Night"


def load_all_data(csv_path: str | Path = "data/uber.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def build_manhattan(df_all: pd.DataFrame) -> pd.DataFrame:
    df_manhattan = df_all[
        (df_all["pickup_longitude"] != 0)
        & (df_all["pickup_latitude"] != 0)
        & (df_all["dropoff_longitude"] != 0)
        & (df_all["dropoff_latitude"] != 0)
        & is_in_manhattan(df_all["pickup_longitude"], df_all["pickup_latitude"])
        & is_in_manhattan(df_all["dropoff_longitude"], df_all["dropoff_latitude"])
    ].copy()
    return df_manhattan


def feature_engineer_manhattan(df_manhattan: pd.DataFrame) -> pd.DataFrame:
    df = df_manhattan.copy()

    # Parse datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    # Datetime features
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month.astype("category")
    df["day"] = df["pickup_datetime"].dt.day.astype("category")
    df["day_of_week"] = df["pickup_datetime"].dt.day_name().astype("category")

    hour = df["pickup_datetime"].dt.hour
    df["time_bin"] = hour.apply(get_time_bin).astype("category")

    # Trip distance
    df["trip_distance"] = haversine_distance_miles(
        df["pickup_latitude"].to_numpy(),
        df["pickup_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
    )

    # Drop unnecessary columns
    drop_cols = ["Unnamed: 0", "key", "pickup_datetime"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def clean_low_fares(df: pd.DataFrame, min_fare: float = 1.0) -> pd.DataFrame:
    return df[df["fare_amount"] >= min_fare].copy()


def iqr_filter(
    df: pd.DataFrame,
    target: str = "fare_amount",
    k: float = 1.5,
) -> Tuple[pd.DataFrame, float, float]:
    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    filtered = df[(df[target] >= lower) & (df[target] <= upper)].copy()
    return filtered, float(lower), float(upper)


@dataclass(frozen=True)
class Datasets:
    df_all: pd.DataFrame
    df_manhattan: pd.DataFrame
    df_manhattan_fe: pd.DataFrame
    df_manhattan_clean: pd.DataFrame
    df_manhattan_iqr: pd.DataFrame
    iqr_bounds: Tuple[float, float]


def build_datasets(csv_path: str | Path = "data/uber.csv") -> Datasets:
    df_all = load_all_data(csv_path)
    df_manhattan = build_manhattan(df_all)
    df_manhattan_fe = feature_engineer_manhattan(df_manhattan)
    df_manhattan_clean = clean_low_fares(df_manhattan_fe, min_fare=1.0)
    df_manhattan_iqr, lower, upper = iqr_filter(df_manhattan_clean, target="fare_amount", k=1.5)
    return Datasets(
        df_all=df_all,
        df_manhattan=df_manhattan,
        df_manhattan_fe=df_manhattan_fe,
        df_manhattan_clean=df_manhattan_clean,
        df_manhattan_iqr=df_manhattan_iqr,
        iqr_bounds=(lower, upper),
    )


# ----------------------------
# Plotting (only when run as script)
# ----------------------------

def run_eda(ds: Datasets, out_dir: str | Path = "eda_plots") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    df_clean = ds.df_manhattan_clean

    # Target distribution
    fare_reasonable = df_clean["fare_amount"][(df_clean["fare_amount"] >= 2.5) & (df_clean["fare_amount"] <= 100)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    fare_common = fare_reasonable[fare_reasonable <= 50]
    ax1.hist(fare_common, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.set_title("Fare Amount Distribution ($2.50 - $50)")
    ax1.set_xlabel("Fare Amount ($)")
    ax1.set_ylabel("Frequency")

    ax2 = axes[1]
    fare_zoomed = fare_reasonable[fare_reasonable <= 25]
    ax2.hist(fare_zoomed, bins=45, edgecolor="black", alpha=0.7, color="coral")
    ax2.set_title("Fare Amount Distribution ($2.50 - $25, Zoomed)")
    ax2.set_xlabel("Fare Amount ($)")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(out_dir / "fare_amount_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Trip distance vs fare
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_clean["trip_distance"], df_clean["fare_amount"], alpha=0.3, s=5, c="blue")
    ax.set_xlabel("Trip Distance (miles)")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_title("Trip Distance vs Fare Amount")
    ax.set_xlim(0, df_clean["trip_distance"].quantile(0.99))
    ax.set_ylim(0, df_clean["fare_amount"].quantile(0.99))
    plt.tight_layout()
    plt.savefig(out_dir / "trip_distance_vs_fare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Time bin vs fare
    time_bin_order = ["Late Night", "Morning", "Midday", "Afternoon", "Evening", "Night"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_clean, x="time_bin", y="fare_amount", order=time_bin_order, ax=ax)
    ax.set_title("Fare Amount by Time of Day (Binned)")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_ylim(0, df_clean["fare_amount"].quantile(0.95))
    plt.tight_layout()
    plt.savefig(out_dir / "time_bin_vs_fare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Day of week vs fare
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_clean, x="day_of_week", y="fare_amount", order=day_order, ax=ax)
    ax.set_title("Fare Amount by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_ylim(0, df_clean["fare_amount"].quantile(0.95))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "day_of_week_vs_fare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Month vs fare
    month_order = list(range(1, 13))
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_clean, x="month", y="fare_amount", order=month_order, ax=ax)
    ax.set_title("Fare Amount by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_ylim(0, df_clean["fare_amount"].quantile(0.95))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.tight_layout()
    plt.savefig(out_dir / "month_vs_fare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Year vs fare
    year_fare = df_clean.groupby("year")["fare_amount"].agg(["mean", "median", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(year_fare["year"], year_fare["mean"], marker="o", linewidth=2, label="Mean")
    ax.plot(year_fare["year"], year_fare["median"], marker="s", linewidth=2, label="Median")
    ax.fill_between(
        year_fare["year"],
        year_fare["mean"] - year_fare["std"],
        year_fare["mean"] + year_fare["std"],
        alpha=0.2,
        label="±1 std",
    )
    ax.set_title("Average Fare Amount by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fare Amount ($)")
    ax.legend()
    ax.set_xticks(year_fare["year"])
    plt.tight_layout()
    plt.savefig(out_dir / "year_vs_fare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Hour vs fare + heatmap (computed from original datetime, still filtered to fares >= $1)
    df_hour = ds.df_manhattan.copy()
    df_hour["pickup_datetime"] = pd.to_datetime(df_hour["pickup_datetime"], errors="coerce")
    df_hour["hour"] = df_hour["pickup_datetime"].dt.hour
    df_hour["day_of_week"] = df_hour["pickup_datetime"].dt.day_name()
    df_hour = df_hour[df_hour["fare_amount"] >= 1]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_hour, x="hour", y="fare_amount", ax=ax)
    ax.set_title("Fare Amount by Hour (No Binning)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_ylim(0, df_hour["fare_amount"].quantile(0.95))
    plt.tight_layout()
    plt.savefig(out_dir / "hour_vs_fare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pivot = df_hour.pivot_table(values="fare_amount", index="day_of_week", columns="hour", aggfunc="mean")
    pivot = pivot.reindex(day_order)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Avg Fare ($)"})
    ax.set_title("Average Fare Amount by Hour and Day of Week")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(out_dir / "hour_day_fare_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # IQR versions for hour/day/heatmap
    lower, upper = ds.iqr_bounds
    df_hour_iqr = df_hour[(df_hour["fare_amount"] >= lower) & (df_hour["fare_amount"] <= upper)].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_hour_iqr, x="hour", y="fare_amount", ax=ax)
    ax.set_title("Fare Amount by Hour (IQR Cleaned)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_ylim(0, df_hour_iqr["fare_amount"].quantile(0.95))
    plt.tight_layout()
    plt.savefig(out_dir / "hour_vs_fare_iqr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pivot_iqr = df_hour_iqr.pivot_table(values="fare_amount", index="day_of_week", columns="hour", aggfunc="mean")
    pivot_iqr = pivot_iqr.reindex(day_order)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot_iqr, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Avg Fare ($)"})
    ax.set_title("Average Fare Amount by Hour and Day of Week (IQR Cleaned)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(out_dir / "hour_day_fare_heatmap_iqr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=ds.df_manhattan_iqr, x="day_of_week", y="fare_amount", order=day_order, ax=ax)
    ax.set_title("Fare Amount by Day of Week (IQR Cleaned)")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_ylim(0, ds.df_manhattan_iqr["fare_amount"].quantile(0.95))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "day_of_week_vs_fare_iqr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------
    # Correlation / association with target (IQR-cleaned Manhattan)
    # ----------------------------
    df_iqr = ds.df_manhattan_iqr.copy()
    target = "fare_amount"

    # Numeric Pearson correlations with target
    numeric_cols = [
        c for c in df_iqr.select_dtypes(include=["number"]).columns
        if c != target
    ]
    pearson_corr = df_iqr[numeric_cols + [target]].corr(numeric_only=True)[target].drop(target)

    # Categorical association (correlation ratio / eta)
    # Produces a single 0..1 association score per categorical feature.
    def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
        cats = categories.astype("category")
        y = values.to_numpy()
        mask = ~pd.isna(cats)
        cats = cats[mask]
        y = y[mask.to_numpy()]
        if len(y) == 0:
            return float("nan")
        overall_mean = float(np.mean(y))
        denom = float(np.sum((y - overall_mean) ** 2))
        if denom == 0:
            return 0.0
        num = 0.0
        for level in cats.cat.categories:
            idx = (cats == level).to_numpy()
            if not np.any(idx):
                continue
            y_k = y[idx]
            num += len(y_k) * (float(np.mean(y_k)) - overall_mean) ** 2
        return float(np.sqrt(num / denom))

    categorical_cols = [
        c for c in df_iqr.columns
        if c != target and (isinstance(df_iqr[c].dtype, pd.CategoricalDtype) or df_iqr[c].dtype == object)
    ]
    eta_scores = {c: correlation_ratio(df_iqr[c], df_iqr[target]) for c in categorical_cols}
    eta = pd.Series(eta_scores).sort_values(ascending=False)

    # Save a combined table (for write-up)
    assoc_table = pd.DataFrame({
        "feature": list(pearson_corr.index) + list(eta.index),
        "association_type": (["pearson_corr"] * len(pearson_corr)) + (["eta_corr_ratio"] * len(eta)),
        "value": list(pearson_corr.values) + list(eta.values),
        "abs_value": list(np.abs(pearson_corr.values)) + list(np.abs(eta.values)),
    }).sort_values("abs_value", ascending=False)
    assoc_table.to_csv(out_dir / "feature_target_associations_iqr.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Numeric (signed)
    axn = axes[0]
    pc_sorted = pearson_corr.sort_values()
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in pc_sorted.values]
    axn.barh(pc_sorted.index, pc_sorted.values, color=colors)
    axn.set_title("Numeric feature correlation with fare (Pearson)")
    axn.set_xlabel("Correlation (signed)")

    # Categorical (eta, 0..1)
    axc = axes[1]
    eta_sorted = eta.sort_values()
    axc.barh(eta_sorted.index, eta_sorted.values, color="#2ca02c")
    axc.set_title("Categorical association with fare (η correlation ratio)")
    axc.set_xlabel("Association strength (0–1)")

    plt.tight_layout()
    plt.savefig(out_dir / "feature_target_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Manhattan map (optional; cartopy imported only here)
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        df_sample = ds.df_manhattan.sample(n=min(199000, len(ds.df_manhattan)), random_state=42)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={"projection": ccrs.PlateCarree()})
        extent = [
            MANHATTAN_BOUNDS["lon_min"] - 0.01,
            MANHATTAN_BOUNDS["lon_max"] + 0.01,
            MANHATTAN_BOUNDS["lat_min"] - 0.01,
            MANHATTAN_BOUNDS["lat_max"] + 0.01,
        ]

        ax1 = axes[0]
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor="lightgray")
        ax1.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.scatter(df_sample["pickup_longitude"], df_sample["pickup_latitude"], c="green", s=1, alpha=0.3)
        ax1.set_title(f"Manhattan Pickup Locations (n={len(df_sample)})")

        ax2 = axes[1]
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, facecolor="lightgray")
        ax2.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.scatter(df_sample["dropoff_longitude"], df_sample["dropoff_latitude"], c="red", s=1, alpha=0.3)
        ax2.set_title(f"Manhattan Dropoff Locations (n={len(df_sample)})")

        plt.tight_layout()
        plt.savefig(out_dir / "uber_manhattan_pickup_dropoff_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # Cartopy may not be installed in all environments
        pass


def main() -> None:
    ds = build_datasets("data/uber.csv")

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"df_all:            {len(ds.df_all):,}")
    print(f"df_manhattan:      {len(ds.df_manhattan):,}")
    print(f"df_manhattan_fe:   {len(ds.df_manhattan_fe):,}")
    print(f"df_manhattan_clean:{len(ds.df_manhattan_clean):,} (fares >= $1)")
    print(f"df_manhattan_iqr:  {len(ds.df_manhattan_iqr):,} (IQR filtered)")
    print(f"IQR bounds:        {ds.iqr_bounds}")

    # Basic stats
    print("\nColumns (df_manhattan_clean):")
    print(ds.df_manhattan_clean.columns.tolist())

    # Export for external training/tuning (e.g., Google Colab)
    out_csv = Path("data") / "df_manhattan_iqr.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ds.df_manhattan_iqr.to_csv(out_csv, index=False)
    print(f"\nSaved IQR dataset to: {out_csv}")

    run_eda(ds, out_dir="eda_plots")


if __name__ == "__main__":
    main()