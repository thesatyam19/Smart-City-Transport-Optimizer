
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = DATA_DIR / "raw_transport.csv"
CLEAN_FILE = DATA_DIR / "cleaned_transport.csv"


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "timestamp",
        "route_id",
        "route_name",
        "latitude",
        "longitude",
        "delay_minutes",
        "passengers",
        "speed_kmh",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    numeric_columns = [
        "latitude",
        "longitude",
        "delay_minutes",
        "passengers",
        "speed_kmh",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    return df


def interpolate_timestamps(df):
    df = df.copy()
    df["_original_order"] = np.arange(len(df))

    result = []

    for route_id, group in df.groupby(
        "route_id", sort=False
    ):
        group = group.sort_values("_original_order").copy()

        times = group["timestamp"]
        valid_times = times.dropna()

        if valid_times.empty:
            group = group.iloc[0:0]
            continue

        positions = np.arange(len(group))
        valid_positions = np.flatnonzero(times.notna())

        known_ns = (
    times.dropna()
    .astype("datetime64[ns]")
    .astype("int64")
    .to_numpy()
)

        interpolated_ns = np.interp(
            positions,
            valid_positions,
            known_ns
        ).astype("int64")

        group["timestamp"] = pd.to_datetime(
            interpolated_ns
        )

        result.append(group)

    if not result:
        raise ValueError(
            "No valid timestamps found in the dataset."
        )

    return pd.concat(result).sort_values(
        "_original_order"
    ).drop(columns="_original_order").reset_index(drop=True)


def clean_gps_noise(df):
    df = df.copy()

    valid_coordinates = (
        df["latitude"].between(25.9, 26.4)
        & df["longitude"].between(77.9, 78.4)
    )

    invalid_count = int((~valid_coordinates).sum())
    df = df.loc[valid_coordinates].copy()

    gps_outlier = pd.Series(False, index=df.index)

    for _, group in df.groupby("route_id"):
        for col in ["latitude", "longitude"]:
            median = group[col].median()
            mad = (group[col] - median).abs().median()

            if pd.isna(mad) or mad == 0:
                continue

            outlier = (
                (group[col] - median).abs()
                > 6 * 1.4826 * mad
            )

            gps_outlier.loc[group.index] |= outlier

    gps_outlier_count = int(gps_outlier.sum())

    df = df.loc[~gps_outlier].copy()

    return df, invalid_count, gps_outlier_count


def process_data():
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw = load_data()
    raw_count = len(raw)

    timestamp_missing_before = int(
        raw["timestamp"].isna().sum()
    )

    df = interpolate_timestamps(raw)

    df, invalid_gps, gps_outliers = clean_gps_noise(df)

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "delay_minutes",
            "passengers",
            "speed_kmh",
        ]
    ).copy()

    df = df[
        (df["delay_minutes"] >= 0)
        & (df["passengers"] >= 0)
        & (df["speed_kmh"] >= 0)
    ].copy()

    df["passengers"] = df["passengers"].round().astype(int)

    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["day_name"] = df["timestamp"].dt.day_name()

    df["is_peak_hour"] = df["hour"].isin(
        [8, 9, 10, 17, 18, 19]
    )

    df = df.sort_values(
        ["route_id", "timestamp"]
    ).reset_index(drop=True)

    df.to_csv(CLEAN_FILE, index=False)

    print("\nDATA CLEANING REPORT")
    print("-" * 40)
    print(f"Raw records: {raw_count:,}")
    print(
        "Missing timestamps before cleaning:",
        timestamp_missing_before
    )
    print("Invalid GPS coordinates:", invalid_gps)
    print("GPS outliers removed:", gps_outliers)
    print(f"Clean records: {len(df):,}")
    print(f"Records removed: {raw_count - len(df):,}")

    return df



def resample_hourly(df):
    data = df.copy()
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data = data.dropna(subset=["timestamp"])

    data["hour_start"] = data["timestamp"].dt.floor("h")

    hourly = (
        data.groupby(
            ["route_id", "route_name", "hour_start"],
            as_index=False
        )
        .agg(
            avg_delay=("delay_minutes", "mean"),
            passenger_demand=("passengers", "sum"),
            avg_speed=("speed_kmh", "mean"),
            trip_records=("route_id", "size")
        )
        .rename(columns={"hour_start": "timestamp"})
        .sort_values(["route_id", "timestamp"])
        .reset_index(drop=True)
    )

    return hourly



def find_top_congestion(hourly):
    active = hourly[
        hourly["trip_records"] > 0
    ].copy()

    if active.empty:
        return active

    active["delay_score"] = (
        active["avg_delay"].rank(pct=True)
    )

    active["demand_score"] = (
        active["passenger_demand"].rank(pct=True)
    )

    active["speed_score"] = (
        1 - active["avg_speed"].rank(pct=True)
    )

    active["congestion_score"] = 100 * (
        0.50 * active["delay_score"]
        + 0.30 * active["demand_score"]
        + 0.20 * active["speed_score"]
    )

    top3 = active.nlargest(
        3, "congestion_score"
    ).copy()

    top3["time_slot"] = (
        top3["timestamp"].dt.strftime("%H:%M")
        + " - "
        + (
            top3["timestamp"]
            + pd.Timedelta(hours=1)
        ).dt.strftime("%H:%M")
    )

    top3.to_csv(
        OUTPUT_DIR / "top_3_congestion.csv",
        index=False
    )

    return top3


def make_recommendation(row):
    recommendations = []

    if row["avg_delay"] >= 10:
        recommendations.append(
            "Review schedules and add buffer time."
        )

    if row["passenger_demand"] >= 20:
        recommendations.append(
            "Evaluate additional buses during this hour."
        )

    if row["avg_speed"] < 15:
        recommendations.append(
            "Inspect congestion hotspots and alternate routes."
        )

    if not recommendations:
        recommendations.append(
            "Continue monitoring this route and time slot."
        )

    return " ".join(recommendations)


def main():
    df = process_data()
    hourly = resample_hourly(df)
    top3 = find_top_congestion(hourly)

    if not top3.empty:
        top3["recommendation"] = top3.apply(
            make_recommendation, axis=1
        )

        top3.to_csv(
            OUTPUT_DIR / "top_3_congestion.csv",
            index=False
        )

    print("\nTOP 3 CONGESTION RESULTS")
    print("-" * 40)

    if top3.empty:
        print("No hourly data available.")
    else:
        print(
            top3[
                [
                    "route_id",
                    "timestamp",
                    "avg_delay",
                    "passenger_demand",
                    "avg_speed",
                    "congestion_score",
                    "recommendation",
                ]
            ].to_string(index=False)
        )

    print("\nOutput files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
