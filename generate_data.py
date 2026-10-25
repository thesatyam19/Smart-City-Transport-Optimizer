
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

routes = {
    "R01": {"name": "Central Express", "start": (26.2183, 78.1828), "end": (26.2380, 78.2050)},
    "R02": {"name": "Market Connector", "start": (26.2100, 78.1700), "end": (26.2250, 78.2000)},
    "R03": {"name": "University Link", "start": (26.2300, 78.1750), "end": (26.2400, 78.1900)},
    "R04": {"name": "Railway Shuttle", "start": (26.2050, 78.1800), "end": (26.2200, 78.2150)},
    "R05": {"name": "Industrial Corridor", "start": (26.2150, 78.2000), "end": (26.2350, 78.2250)},
    "R06": {"name": "Residential Loop", "start": (26.2250, 78.1600), "end": (26.2450, 78.1800)},
}

records = []
timestamps = pd.date_range(
    start="2026-10-01 05:00:00",
    end="2026-10-07 23:00:00",
    freq="15min"
)

for route_id, route in routes.items():
    start_lat, start_lon = route["start"]
    end_lat, end_lon = route["end"]

    for timestamp in timestamps:
        hour = timestamp.hour
        weekday = timestamp.weekday()

        peak = hour in [8, 9, 10, 17, 18, 19]
        weekend = weekday >= 5

        base_delay = {
            "R01": 8,
            "R02": 12,
            "R03": 4,
            "R04": 10,
            "R05": 7,
            "R06": 5,
        }[route_id]

        delay = base_delay + (9 if peak else 0) + np.random.normal(0, 3)
        delay = max(0, delay)

        passengers = np.random.poisson(
            24 if peak else 10
        )
        if weekend:
            passengers = max(0, int(passengers * 0.8))

        progress = np.random.uniform(0, 1)
        latitude = start_lat + progress * (end_lat - start_lat)
        longitude = start_lon + progress * (end_lon - start_lon)

        latitude += np.random.normal(0, 0.00015)
        longitude += np.random.normal(0, 0.00015)

        records.append({
            "timestamp": timestamp,
            "route_id": route_id,
            "route_name": route["name"],
            "latitude": latitude,
            "longitude": longitude,
            "delay_minutes": round(delay, 2),
            "passengers": passengers,
            "speed_kmh": round(
                max(4, np.random.normal(24 if not peak else 15, 5)), 2
            ),
            "is_peak_hour": peak
        })

df = pd.DataFrame(records)

missing_indices = np.random.choice(
    df.index,
    size=int(len(df) * 0.01),
    replace=False
)
df.loc[missing_indices, "timestamp"] = pd.NaT

noise_indices = np.random.choice(
    df.index,
    size=int(len(df) * 0.005),
    replace=False
)
df.loc[noise_indices, "latitude"] += np.random.choice([-0.08, 0.08], len(noise_indices))
df.loc[noise_indices, "longitude"] += np.random.choice([-0.08, 0.08], len(noise_indices))

output_file = DATA_DIR / "raw_transport.csv"
df.to_csv(output_file, index=False)

print("Transport dataset generated successfully!")
print(f"Total records: {len(df):,}")
print(f"Routes: {df['route_id'].nunique()}")
print(f"Output file: {output_file}")
print("\nPreview:")
print(df.head())
