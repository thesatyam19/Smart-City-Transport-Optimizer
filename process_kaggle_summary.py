
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "kaggle_transport_cleaned.csv"
CHUNK_SIZE = 200_000

hourly_parts = []

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"], errors="coerce"
    )

    chunk["delay_minutes"] = pd.to_numeric(
        chunk["delay_minutes"], errors="coerce"
    )

    chunk["latitude"] = pd.to_numeric(
        chunk["latitude"], errors="coerce"
    )

    chunk["longitude"] = pd.to_numeric(
        chunk["longitude"], errors="coerce"
    )

    chunk = chunk.dropna(
        subset=["timestamp", "route_id", "delay_minutes"]
    )

    chunk["hour_start"] = chunk["timestamp"].dt.floor("h")

    summary = (
        chunk.groupby(["route_id", "hour_start"], as_index=False)
        .agg(
            avg_delay=("delay_minutes", "mean"),
            vehicle_records=("route_id", "size"),
            avg_distance_from_stop_m=("distance_from_stop_m", "mean")
        )
    )

    hourly_parts.append(summary)
    print(f"Processed {sum(len(part) for part in hourly_parts):,} grouped rows")

if not hourly_parts:
    raise ValueError("No valid records found in the input dataset.")

hourly = pd.concat(hourly_parts, ignore_index=True)

hourly = (
    hourly.groupby(["route_id", "hour_start"], as_index=False)
    .agg(
        avg_delay=("avg_delay", "mean"),
        vehicle_records=("vehicle_records", "sum"),
        avg_distance_from_stop_m=("avg_distance_from_stop_m", "mean")
    )
    .rename(columns={"hour_start": "timestamp"})
    .sort_values(["route_id", "timestamp"])
)

hourly["hour"] = hourly["timestamp"].dt.hour
hourly["is_peak_hour"] = hourly["hour"].isin(
    [7, 8, 9, 16, 17, 18, 19]
)

hourly.to_csv(
    OUTPUT_DIR / "kaggle_hourly_summary.csv",
    index=False
)

print("\nKAGGLE HOURLY SUMMARY COMPLETE")
print(f"Hourly route records: {len(hourly):,}")
print(f"Unique routes: {hourly['route_id'].nunique():,}")
print(f"Average delay: {hourly['avg_delay'].mean():.2f} minutes")
print("Saved: outputs/kaggle_hourly_summary.csv")
print("\nTop 10 route-hours by average delay:")
print(
    hourly.nlargest(10, "avg_delay")[
        ["route_id", "timestamp", "avg_delay", "vehicle_records"]
    ].to_string(index=False)
)
