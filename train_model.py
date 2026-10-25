
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "kaggle_hourly_summary.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "delay_prediction_model.joblib"

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

required_columns = [
    "route_id",
    "timestamp",
    "avg_delay",
    "hour",
    "is_peak_hour",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df["avg_delay"] = pd.to_numeric(df["avg_delay"], errors="coerce")
df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
df["is_peak_hour"] = df["is_peak_hour"].astype(str).str.lower().map(
    {"true": 1, "false": 0, "1": 1, "0": 0}
)

df = df.dropna(
    subset=["timestamp", "route_id", "avg_delay", "hour"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

df = df[
    np.isfinite(df["avg_delay"])
    & np.isfinite(df["hour"])
    & (df["avg_delay"] >= 0)
]

if len(df) < 30:
    raise ValueError("Not enough valid records to train the model.")

df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

features = [
    "route_id",
    "hour",
    "is_peak_hour",
    "day_of_week",
    "month",
]

target = "avg_delay"

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Not enough data for a train/test split.")

categorical_features = ["route_id"]
numeric_features = [
    "hour",
    "is_peak_hour",
    "day_of_week",
    "month",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_features,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

model.fit(train_df[features], train_df[target])

predictions = model.predict(test_df[features])

mae = mean_absolute_error(test_df[target], predictions)
rmse = np.sqrt(mean_squared_error(test_df[target], predictions))
r2 = r2_score(test_df[target], predictions)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
test_results = pd.DataFrame({
    "timestamp": test_df["timestamp"].values,
    "route_id": test_df["route_id"].values,
    "Actual Delay": test_df[target].values,
    "Predicted Delay": predictions
})

test_results = pd.DataFrame({
    "timestamp": test_df["timestamp"].values,
    "route_id": test_df["route_id"].values,
    "Actual Delay": test_df[target].values,
    "Predicted Delay": predictions
})

test_results.to_csv(MODEL_DIR / "test_predictions.csv", index=False)


test_results_path = MODEL_DIR / "test_predictions.csv"
test_results.to_csv(test_results_path, index=False)

joblib.dump(
    {
        "model": model,
        "features": features,
        "target": target,
        "trained_until": str(train_df["timestamp"].max()),
        "tested_from": str(test_df["timestamp"].min()),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    },
    MODEL_PATH,
)

print("\nMODEL TRAINING COMPLETE")
print(f"Valid records: {len(df):,}")
print(f"Training records: {len(train_df):,}")
print(f"Testing records: {len(test_df):,}")
print(f"Training ends: {train_df['timestamp'].max()}")
print(f"Testing starts: {test_df['timestamp'].min()}")
print(f"MAE: {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")
print(f"R² score: {r2:.4f}")
print(f"Saved model: {MODEL_PATH}")
