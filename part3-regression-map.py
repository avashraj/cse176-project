import os
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional: cartopy for country borders / coastlines
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    CARTOPY_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback
    CARTOPY_AVAILABLE = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OUTPUT_PATH = "eda_plots/uber_rides_world_map.png"


def load_uber_data() -> pd.DataFrame:
    """Download and load the Uber fares dataset from Kaggle."""
    print("Downloading dataset...")
    path = kagglehub.dataset_download("yasserh/uber-fares-dataset")
    print(f"Dataset path: {path}")

    csv_files = list(Path(path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    csv_file = csv_files[0]
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {df.shape[0]:,} rows")
    return df


def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing/invalid coordinates.

    Notes:
    - The dataset is NYC taxi trips; we additionally clip to a broad NYC box
      to filter obvious outliers (points in the ocean or other continents).
    """
    coord_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    for col in coord_cols:
        if col not in df.columns:
            raise ValueError(f"Missing coordinate column: {col}")

    initial = len(df)
    df = df.dropna(subset=coord_cols)

    # Remove obvious invalid coordinates (0,0 or outside [-90, 90]/[-180, 180])
    valid_mask = (
        (df["pickup_latitude"].between(-90, 90))
        & (df["dropoff_latitude"].between(-90, 90))
        & (df["pickup_longitude"].between(-180, 180))
        & (df["dropoff_longitude"].between(-180, 180))
        & (
            (df["pickup_latitude"] != 0)
            | (df["pickup_longitude"] != 0)
            | (df["dropoff_latitude"] != 0)
            | (df["dropoff_longitude"] != 0)
        )
    )
    df = df[valid_mask].copy()
    removed_global = initial - len(df)

    # NYC bounding box (loose): lat 35..45, lon -80..-70
    nyc_mask = (
        df["pickup_latitude"].between(40.5, 41.0)
        & df["pickup_longitude"].between(-74.3, -73.7)
        & df["dropoff_latitude"].between(40.5, 41.0)
        & df["dropoff_longitude"].between(-74.3, -73.7)
    )
    df = df[nyc_mask].copy()
    removed_bbox = initial - removed_global - len(df)

    print(
        f"Removed {removed_global:,} rows (invalid/global), "
        f"{removed_bbox:,} rows (outside NYC box)"
    )
    return df


def plot_world_map(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    """
    Plot pickup and dropoff points on a global map.

    - Uses Cartopy for country borders/coastlines if available.
    - Falls back to a Mollweide scatter without borders otherwise.
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # Prepare coordinates (degrees)
    pickup_lons_deg = df["pickup_longitude"].values
    pickup_lats_deg = df["pickup_latitude"].values
    drop_lons_deg = df["dropoff_longitude"].values
    drop_lats_deg = df["dropoff_latitude"].values

    if CARTOPY_AVAILABLE:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.Mollweide())
        ax.set_title(
            "Uber Rides: Pickup and Dropoff Locations (Global View)",
            fontsize=12,
        )
        ax.add_feature(cfeature.LAND, facecolor="#f7f7f7")
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        ax.gridlines(draw_labels=True, linestyle="--", alpha=0.3)

        # Plot pickups and dropoffs
        ax.scatter(
            pickup_lons_deg,
            pickup_lats_deg,
            s=2,
            alpha=0.35,
            color="#3498db",
            label="Pickup",
            transform=ccrs.PlateCarree(),
        )
        ax.scatter(
            drop_lons_deg,
            drop_lats_deg,
            s=2,
            alpha=0.35,
            color="#e74c3c",
            label="Dropoff",
            transform=ccrs.PlateCarree(),
        )
    else:
        # Fallback: Mollweide scatter without country outlines
        pickup_lons = np.radians(pickup_lons_deg)
        pickup_lats = np.radians(pickup_lats_deg)
        drop_lons = np.radians(drop_lons_deg)
        drop_lats = np.radians(drop_lats_deg)

        fig = plt.figure(figsize=(12, 6))
        ax = plt.subplot(111, projection="mollweide")
        ax.set_title(
            "Uber Rides: Pickup and Dropoff Locations (Global View)",
            fontsize=12,
        )
        ax.grid(True, linestyle="--", alpha=0.4)

        ax.scatter(
            pickup_lons,
            pickup_lats,
            s=2,
            alpha=0.35,
            color="#3498db",
            label="Pickup",
        )
        ax.scatter(
            drop_lons,
            drop_lats,
            s=2,
            alpha=0.35,
            color="#e74c3c",
            label="Dropoff",
        )

        if not CARTOPY_AVAILABLE:
            print(
                "cartopy not installed; showing points without country outlines. "
                "Install cartopy for borders: pip install cartopy"
            )

    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved world map plot to {output_path}")


def main():
    df = load_uber_data()
    df = clean_coordinates(df)
    plot_world_map(df)


if __name__ == "__main__":
    main()
