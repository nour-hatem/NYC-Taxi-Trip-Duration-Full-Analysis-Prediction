import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


# ── Constants ─────────────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6_371.0

NUMERIC_FEATURES = [
    "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude",
    "passenger_count", "distance_km", "bearing",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

CATEGORICAL_FEATURES = ["vendor_id", "store_and_fwd_flag", "dayofweek", "month"]

INPUT_COLS = [
    "pickup_datetime", "vendor_id", "store_and_fwd_flag",
    "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude",
    "passenger_count",
]


# ── Spatial helpers ────────────────────────────────────────────────────────────

def calculate_distance(lon1, lat1, lon2, lat2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi      = np.radians(lat2 - lat1)
    d_lambda   = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def calculate_bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(y, x))


# ── Data cleaning ──────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["trip_duration"].between(10, 10_800) &
        df["pickup_latitude"].between(40.50, 40.93) &
        df["pickup_longitude"].between(-74.25, -73.65) &
        df["dropoff_latitude"].between(40.50, 40.93) &
        df["dropoff_longitude"].between(-74.25, -73.65) &
        (df["passenger_count"] > 0)
    )
    return df[mask].reset_index(drop=True)


# ── Feature engineering ────────────────────────────────────────────────────────

def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["pickup_datetime"] = pd.to_datetime(X["pickup_datetime"])
    dt = X["pickup_datetime"].dt

    X["distance_km"] = calculate_distance(
        X["pickup_longitude"], X["pickup_latitude"],
        X["dropoff_longitude"], X["dropoff_latitude"],
    )
    X["bearing"] = calculate_bearing(
        X["pickup_longitude"], X["pickup_latitude"],
        X["dropoff_longitude"], X["dropoff_latitude"],
    )

    X["hour"]      = dt.hour
    X["dayofweek"] = dt.dayofweek
    X["month"]     = dt.month

    X["hour_sin"] = np.sin(2 * np.pi * X["hour"]      / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"]      / 24)
    X["day_sin"]  = np.sin(2 * np.pi * X["dayofweek"] /  7)
    X["day_cos"]  = np.cos(2 * np.pi * X["dayofweek"] /  7)

    X.drop(columns=["pickup_datetime", "hour"], inplace=True)
    return X


# ── Pipeline ───────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators     = 2000,
        learning_rate    = 0.03,
        max_depth        = 10,
        subsample        = 0.9,
        colsample_bytree = 0.7,
        min_child_weight = 5,
        gamma            = 0.1,
        n_jobs           = -1,
        tree_method      = "hist",
        random_state     = 42,
        objective        = "reg:squarederror",
        eval_metric      = "rmse",
    )

    return Pipeline([
        ("features",  FunctionTransformer(feature_engineering)),
        ("processor", preprocessor),
        ("model",     model),
    ])


# ── Train / evaluate / save ────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    X_train = train_df[INPUT_COLS].copy()
    y_train = np.log1p(train_df["trip_duration"])

    X_val = val_df[INPUT_COLS].copy()
    y_val = val_df["trip_duration"]

    return X_train, X_val, y_train, y_val


def train(pipeline: Pipeline, X_train, X_val, y_train, y_val) -> Pipeline:
    fe   = pipeline.named_steps["features"]
    proc = pipeline.named_steps["processor"]

    X_train_fe = fe.transform(X_train)
    X_val_fe   = fe.transform(X_val)

    proc.fit(X_train_fe)
    X_train_proc = proc.transform(X_train_fe)
    X_val_proc   = proc.transform(X_val_fe)

    pipeline.named_steps["model"].fit(
        X_train_proc, y_train,
        eval_set=[(X_val_proc, np.log1p(y_val))],
        verbose=200,
    )
    return pipeline


def evaluate(pipeline: Pipeline, X_val, y_val):
    y_pred = np.expm1(pipeline.predict(X_val))

    r2    = r2_score(y_val, y_pred)
    rmse  = root_mean_squared_error(y_val, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(
        np.maximum(y_val, 0), np.maximum(y_pred, 0)
    ))

    print("=" * 40)
    print(f"  R²    : {r2:.4f}")
    print(f"  RMSE  : {rmse:.1f} s  ({rmse / 60:.1f} min)")
    print(f"  RMSLE : {rmsle:.4f}")
    print("=" * 40)

    return y_pred, r2, rmse, rmsle


def save_model(pipeline: Pipeline, path: str = "nyc_taxi_xgb_pipeline.pkl"):
    joblib.dump(pipeline, path)
    print(f"Pipeline saved → {path}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_model_eval(y_val, y_pred):
    residuals = y_val - y_pred
    cap = np.percentile(y_val, 99)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, y_val, alpha=0.1, s=3, color="steelblue")
    axes[0].plot([0, cap], [0, cap], "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set(xlim=(0, cap), ylim=(0, cap),
                xlabel="Predicted (s)", ylabel="Actual (s)", title="Predicted vs Actual")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.1, s=3, color="coral")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set(xlim=(0, cap), ylim=(-cap / 2, cap / 2),
                xlabel="Predicted (s)", ylabel="Residual (actual − predicted)", title="Residual Plot")

    plt.suptitle("Model Evaluation", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(pipeline: Pipeline):
    ohe       = pipeline.named_steps["processor"].named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feat_names = ohe_names + NUMERIC_FEATURES

    importance = (
        pd.Series(pipeline.named_steps["model"].feature_importances_, index=feat_names)
        .sort_values(ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    importance.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set(title="Top 20 Feature Importances (XGBoost gain)", xlabel="Importance score")
    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/train.zip")
    raw = clean_data(raw)

    X_train, X_val, y_train, y_val = split_data(raw)

    pipeline = build_pipeline()
    pipeline = train(pipeline, X_train, X_val, y_train, y_val)

    y_pred, r2, rmse, rmsle = evaluate(pipeline, X_val, y_val)

    plot_model_eval(y_val, y_pred)
    plot_feature_importance(pipeline)

    save_model(pipeline)