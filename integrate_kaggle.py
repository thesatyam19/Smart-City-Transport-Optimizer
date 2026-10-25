
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    BASE_DIR / "mta_1706.csv",
    BASE_DIR / "mta_1708.csv",
    BASE_DIR / "mta_1710.csv",
    BASE_DIR / "mta_1712.csv"
]

frames = []

for file in FILES:
    if not file.exists():
        print(f"File not found: {file.name}")
        continue

    df = pd.read_csv(
    file,
    low_memory=False,
    on_bad_lines="warn"
)
    df["source_file"] = file.name
    frames.append(df)
    print(f"Loaded {file.name}: {len(df):,} records")

if not frames:
    raise FileNotFoundError("No Kaggle CSV files found in the project folder.")

raw = pd.concat(frames, ignore_index=True)

raw["RecordedAtTime"] = pd.to_datetime(
    raw["RecordedAtTime"], errors="coerce"
)

raw["ExpectedArrivalTime"] = pd.to_datetime(
    raw["ExpectedArrivalTime"], errors="coerce"
)

raw["ScheduledArrivalTime"] = pd.to_datetime(
    raw["ScheduledArrivalTime"],
    format="%H:%M:%S",
    errors="coerce"
)

scheduled_seconds = (
    raw["ScheduledArrivalTime"].dt.hour * 3600
    + raw["ScheduledArrivalTime"].dt.minute * 60
    + raw["ScheduledArrivalTime"].dt.second
)

recorded_seconds = (
    raw["RecordedAtTime"].dt.hour * 3600
    + raw["RecordedAtTime"].dt.minute * 60
    + raw["RecordedAtTime"].dt.second
)

scheduled_day_offset = (scheduled_seconds - recorded_seconds) % 86400

scheduled_datetime = (
    raw["RecordedAtTime"].dt.normalize()
    + pd.to_timedelta(scheduled_seconds, unit="s")
)

scheduled_datetime = scheduled_datetime.where(
    scheduled_day_offset <= 43200,
    scheduled_datetime - pd.Timedelta(days=1)
)

scheduled_datetime = scheduled_datetime.where(
    scheduled_day_offset > 43200,
    scheduled_datetime
)

raw["delay_minutes"] = (
    raw["ExpectedArrivalTime"] - scheduled_datetime
).dt.total_seconds() / 60

raw["delay_minutes"] = raw["delay_minutes"].clip(-180, 180)

result = pd.DataFrame({
    "timestamp": raw["RecordedAtTime"],
    "route_id": raw["PublishedLineName"].astype("string").str.strip(),
    "route_name": raw["OriginName"].astype("string").str.strip()
        + " → "
        + raw["DestinationName"].astype("string").str.strip(),
    "latitude": raw["VehicleLocation.Latitude"],
    "longitude": raw["VehicleLocation.Longitude"],
    "delay_minutes": raw["delay_minutes"],
    "distance_from_stop_m": raw["DistanceFromStop"],
    "arrival_proximity": raw["ArrivalProximityText"],
    "vehicle_id": raw["VehicleRef"],
    "next_stop": raw["NextStopPointName"],
    "source_file": raw["source_file"]
})

result = result.dropna(
    subset=["timestamp", "route_id", "latitude", "longitude"]
)

result = result[
    result["latitude"].between(40.4, 41.0)
    & result["longitude"].between(-74.3, -73.6)
]

result["hour"] = result["timestamp"].dt.hour
result["is_peak_hour"] = result["hour"].isin(
    [7, 8, 9, 16, 17, 18, 19]
)

result["passengers"] = np.nan
result["speed_kmh"] = np.nan

output_file = OUTPUT_DIR / "kaggle_transport_cleaned.csv"
result.to_csv(output_file, index=False)

print("\nKAGGLE DATA INTEGRATION REPORT")
print("-" * 45)
print(f"Combined records: {len(raw):,}")
print(f"Clean records: {len(result):,}")
print(f"Unique bus routes: {result['route_id'].nunique():,}")
print(f"Missing delay values: {result['delay_minutes'].isna().sum():,}")
print(f"Missing passenger values: {result['passengers'].isna().sum():,}")
print(f"Missing speed values: {result['speed_kmh'].isna().sum():,}")
print(f"\nSaved dataset: {output_file}")
print("\nSample records:")
print(result.head(5).to_string(index=False))
