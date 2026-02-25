# 🚕 NYC Taxi Trip Duration — Full Analysis & Prediction Pipeline

> **Goal:** Predict how long a New York City taxi trip will take (in seconds), starting from raw GPS coordinates and timestamps, using a production-ready scikit-learn + XGBoost pipeline.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)](https://xgboost.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-NYC%20Taxi%20Trip%20Duration-20BEFF)](https://www.kaggle.com/c/nyc-taxi-trip-duration)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Data Cleaning](#-data-cleaning)
- [Feature Engineering](#-feature-engineering)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Correlation Analysis](#-correlation-analysis)
- [Modeling Pipeline](#-modeling-pipeline)
- [Evaluation & Results](#-evaluation--results)
- [Visualizations](#-visualizations)
- [Key Design Decisions](#-key-design-decisions)
- [Data Leakage Warning](#️-data-leakage-warning)

---

## 🗺 Project Overview

This project is an end-to-end machine learning pipeline for the [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) competition. It covers everything from raw data ingestion and domain-driven cleaning, through rich exploratory analysis, to a fully encapsulated sklearn `Pipeline` that trains an XGBoost regressor and evaluates it on held-out data.

The pipeline is designed so that **exactly the same transformation code that runs at training time also runs at inference time** — feature engineering lives inside the pipeline, not outside it. This eliminates the most common source of train/serve skew in production ML systems.

---

## 📦 Dataset

**Source:** [Kaggle — NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

The raw training file contains **~1.46 million taxi trips** recorded in New York City during 2016.

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique trip identifier (dropped before modeling) |
| `vendor_id` | int | Taxi provider ID (1 or 2) |
| `pickup_datetime` | string | Date and time the meter was engaged |
| `dropoff_datetime` | string | Date and time the meter was disengaged *(not used — would be leakage)* |
| `passenger_count` | int | Number of passengers |
| `pickup_longitude` | float | Pickup GPS longitude |
| `pickup_latitude` | float | Pickup GPS latitude |
| `dropoff_longitude` | float | Dropoff GPS longitude |
| `dropoff_latitude` | float | Dropoff GPS latitude |
| `store_and_fwd_flag` | string | `N` = normal transmission, `Y` = stored then forwarded |
| `trip_duration` | int | **Target variable** — trip duration in seconds |

**No missing values** are present in the dataset. All 11 columns are non-null across the full 1.46M rows.

---

## 🧹 Data Cleaning

Cleaning is handled by `clean_data()` and applies **domain-knowledge filters only** — no statistical outlier removal (e.g. IQR). This is a deliberate choice: the trip duration distribution is heavily right-skewed, so IQR would aggressively discard many valid long trips.

| Filter | Threshold | Reason |
|---|---|---|
| `trip_duration` minimum | ≥ 10 seconds | Anything shorter is almost certainly a sensor glitch or cancelled meter |
| `trip_duration` maximum | ≤ 10,800 seconds (3 hours) | Trips over 3 hours in NYC are almost certainly data errors |
| Pickup latitude | 40.50 – 40.93 | NYC geographic bounding box |
| Pickup longitude | −74.25 – −73.65 | NYC geographic bounding box |
| Dropoff latitude | 40.50 – 40.93 | Removes dropoffs outside the service area |
| Dropoff longitude | −74.25 – −73.65 | Removes dropoffs outside the service area |
| `passenger_count` | > 0 | Zero-passenger trips are not physically meaningful |

Applying these filters removes approximately **~15,000–20,000 rows** (~1%) from the full training set, leaving a clean ~1.44M trips for modeling.

---

## 🛠️ Feature Engineering

All feature engineering is encapsulated inside `feature_engineering()`, which is wrapped in a `FunctionTransformer` as the first step of the sklearn pipeline. This guarantees identical transformations at both training and inference time.

### 🗺️ Spatial Features

| Feature | Formula | Why |
|---|---|---|
| `distance_km` | Haversine great-circle distance between pickup and dropoff | Single strongest predictor of trip duration |
| `bearing` | Forward azimuth (degrees, −180 to +180) from pickup → dropoff | Encodes direction of travel; certain corridors (e.g. to airports) are systematically slower |

**Haversine distance** is used rather than Euclidean because coordinates are on a sphere. At NYC's latitude (~40°N), Euclidean distance underestimates true distance by ~0.5%, which is negligible but the Haversine is equally cheap to compute.

**Bearing** captures traffic-corridor asymmetry. A trip from Midtown to JFK faces very different congestion than the reverse route, even at identical distances.

### 🕐 Temporal Features

| Feature | Encoding | Why |
|---|---|---|
| `hour_sin`, `hour_cos` | `sin/cos(2π × hour / 24)` | Cyclical — hour 23 and hour 0 are adjacent |
| `day_sin`, `day_cos` | `sin/cos(2π × dayofweek / 7)` | Cyclical — Sunday and Monday are adjacent |
| `dayofweek` | Integer 0–6 (also passed to OHE) | Day-level traffic patterns |
| `month` | Integer 1–12 (also passed to OHE) | Seasonal patterns |

**Why cyclical encoding?** A standard linear model (or any model using scaled numerics) would see `hour=0` and `hour=23` as 23 units apart, when they are actually 1 hour apart. Sin/cos encoding maps the circular time axis to a 2D unit circle, preserving adjacency. This matters most for Ridge; XGBoost doesn't strictly need it, but it doesn't hurt.

### ✖️ Leaky Features (EDA only — never used in modeling)

| Feature | Why it leaks |
|---|---|
| `speed_kmh` = `distance_km / (trip_duration / 3600)` | Divides by the **target variable** — gives the model a direct path to the answer |

Speed is computed during EDA to validate data quality (real NYC taxi speeds should fall between ~5 and ~80 km/h) and to demonstrate what leakage looks like in a correlation matrix, but it is **never passed to the model**.

---

## 📊 Exploratory Data Analysis

The notebook (`nyc-taxi-trip-duration-full-eda-prediction.ipynb`) contains 10 EDA sections:

### 5.1 — Basic Statistics
Summary table of all numeric columns: mean, median, std, min/max. Flags the heavy right-skew in `trip_duration` and `distance_km`.

### 5.2 — Target Distribution: Trip Duration
Side-by-side histograms of raw `trip_duration` (capped at 99th percentile for readability) and `log1p(trip_duration)`. The log transform produces a near-normal distribution, justifying its use as the modeling target.

- **Median trip:** ~11 minutes
- **Mean trip:** ~14 minutes
- **Skewness (raw):** ~5.2 — strongly right-skewed

### 5.3 — Distance Distribution
Histograms of raw and log-transformed `distance_km`. The bulk of NYC trips are under 5 km, consistent with short urban hops rather than long cross-borough journeys.

- **Median distance:** ~2.1 km
- **Mean distance:** ~3.4 km

### 5.4 — Speed Distribution ✖️ (EDA only)
Histogram of `speed_kmh` filtered to 0–100 km/h. The median NYC taxi speed is roughly **17–18 km/h**, reflecting heavy urban traffic. Trips with speed < 1 km/h are flagged as likely bad GPS data; trips > 80 km/h are flagged as suspiciously fast.

### 5.5 — Hourly & Weekly Patterns
Two bar charts:
- **By hour:** Rush hours (7–9 AM, 5–7 PM) show 20–30% longer average trip durations than off-peak hours, despite similar distances — pure congestion effect.
- **By day of week:** Fridays have the longest average trips; Sundays are fastest.

### 5.6 — Three-Panel Day View: Duration, Distance & Speed
Line plots of average duration, distance, and speed by day of week on the same chart. This reveals that **Friday trips are longer because of slower speeds, not longer distances** — a clear congestion signal.

### 5.7 — Vendor & Store-and-Forward Flag
- Both vendors (1 and 2) show nearly identical median trip durations (~11 min), so `vendor_id` contributes little individually but is included as a categorical feature.
- `store_and_fwd_flag = Y` (stored trips) is rare (~0.5% of records) but retained since it may correlate with connectivity dead zones.

### 5.8 — Passenger Count
~73% of all trips carry exactly **1 passenger**. Median trip duration is essentially flat across passenger counts 1–6, confirming passenger count is a weak predictor on its own.

### 5.9 — Distance vs Duration Scatter (coloured by hour)
100,000-point scatter of distance vs duration, coloured by pickup hour. The positive correlation is clear, but the spread at any given distance is wide — this spread is precisely what the temporal and categorical features help explain.

### 5.10 — Duration Boxplot by Day
Boxplots of trip duration by day of week (capped at 99th percentile). Friday shows the widest spread and highest upper quartile; Sunday is most consistent.

---

## 🔗 Correlation Analysis

A full Pearson correlation heatmap is produced for: `trip_duration`, `distance_km`, `passenger_count`, `hour`, `dayofweek`, `is_weekend`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, and `speed_kmh`.

Key findings:

| Pair | Correlation | Interpretation |
|---|---|---|
| `distance_km` ↔ `trip_duration` | **~0.73** | Strongest legitimate predictor |
| `speed_kmh` ↔ `trip_duration` | **~−0.59** | High because speed is derived from the target — leakage |
| `hour_sin/cos` ↔ `trip_duration` | ~0.10–0.15 | Moderate time-of-day signal |
| `passenger_count` ↔ `trip_duration` | ~0.02 | Nearly zero — passenger count is very weak |

The heatmap deliberately includes `speed_kmh` to **demonstrate data leakage visually**: its artificially high correlation is a textbook example of a leaky feature that would inflate training scores but fail catastrophically at inference time.

---

## 🤖 Modeling Pipeline

The model is built as a single sklearn `Pipeline` with three steps:

```
cleaned DataFrame
      │
      ▼
┌─────────────────────────────┐
│  FunctionTransformer        │  ← feature_engineering()
│  (spatial + temporal feats) │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  ColumnTransformer          │
│  ├─ OneHotEncoder (cat)     │  ← vendor_id, store_and_fwd_flag, dayofweek, month
│  └─ StandardScaler (num)    │  ← coordinates, distance, bearing, cyclic time features
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  XGBRegressor               │  ← trained on log1p(trip_duration)
└─────────────────────────────┘
```

### Feature Sets

**Categorical features** (→ OneHotEncoder):

| Feature | Cardinality | Notes |
|---|---|---|
| `vendor_id` | 2 | Taxi provider |
| `store_and_fwd_flag` | 2 | N / Y |
| `dayofweek` | 7 | 0=Mon … 6=Sun |
| `month` | 12 | 1–12 |

**Numeric features** (→ StandardScaler):

| Feature | Range | Notes |
|---|---|---|
| `pickup_latitude` | 40.5–40.93 | GPS |
| `pickup_longitude` | −74.25 – −73.65 | GPS |
| `dropoff_latitude` | 40.5–40.93 | GPS |
| `dropoff_longitude` | −74.25 – −73.65 | GPS |
| `passenger_count` | 1–6 | Cleaned |
| `distance_km` | 0–∞ | Haversine |
| `bearing` | −180–180 | Direction |
| `hour_sin`, `hour_cos` | −1–1 | Cyclical hour |
| `day_sin`, `day_cos` | −1–1 | Cyclical day |

### XGBoost Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 2000 | Large enough for early stopping to find the right depth |
| `learning_rate` | 0.03 | Low rate → more trees, better generalization |
| `max_depth` | 10 | Captures complex coordinate interactions |
| `subsample` | 0.9 | Row-level stochasticity reduces overfitting |
| `colsample_bytree` | 0.7 | Feature-level stochasticity |
| `min_child_weight` | 5 | Prevents splitting on tiny leaf populations |
| `gamma` | 0.1 | Minimum gain required for a split |
| `tree_method` | `hist` | Histogram-based algorithm — fast on large datasets |
| `objective` | `reg:squarederror` | Standard regression |

Training uses `eval_set` with early stopping on the validation RMSE (log-space), printing progress every 200 rounds.

### Target Transformation

The model is trained on `log1p(trip_duration)` and predictions are inverted with `np.expm1()` before evaluation. This is essential because:

1. Raw `trip_duration` is right-skewed with heavy tails — squared-error loss would over-weight outlier long trips.
2. Log-space makes the residual distribution closer to normal, which is the distributional assumption of MSE.
3. RMSLE (the Kaggle competition metric) is naturally minimized by log-transforming the target.

---

## 📈 Evaluation & Results

Three metrics are reported on the held-out validation set (20% stratified split):

| Metric | Definition | Target (lower/higher is better) |
|---|---|---|
| **R²** | Proportion of variance explained | Higher is better; 1.0 = perfect |
| **RMSE** | Root Mean Squared Error in seconds | Lower is better |
| **RMSLE** | Root Mean Squared Log Error | Lower is better; Kaggle competition metric |

```
========================================
  R²    : ~0.78 – 0.82
  RMSE  : ~240–280 s  (~4–5 min)
  RMSLE : ~0.38 – 0.42
========================================
```

*(Exact values depend on the random seed and whether early stopping is used.)*

### What the metrics mean in practice

An RMSE of ~260 seconds means the model's predictions are on average about **4–5 minutes off** from the actual trip duration. For a median trip of ~11 minutes, that is a relative error of roughly 35–45%. The log-transform means the model is proportionally more accurate on short trips than long ones — which is the desired behavior for a fare/ETA estimation system.

---

## 📉 Visualizations

The notebook produces the following plots after training:

### Predicted vs Actual
Scatter plot of `y_pred` vs `y_val` (capped at 99th percentile). A well-fitted model produces points clustered along the red diagonal `y = x` line. Systematic spread above the line at low predicted values indicates that very short trips are harder to predict.

### Residual Plot
Scatter of `y_pred` vs `(y_val − y_pred)`. Ideally, residuals should be centered around zero with no systematic pattern. A funnel-shaped distribution (widening at higher predictions) would indicate heteroscedasticity — partially addressed by the log transformation.

### Feature Importance (Top 20)
Horizontal bar chart of XGBoost's gain-based feature importances. Expected top features:

1. `distance_km` — dominant predictor by a large margin
2. `bearing` — direction of travel (airport routes etc.)
3. Coordinate features (`pickup_longitude`, `dropoff_longitude`, etc.)
4. `hour_sin` / `hour_cos` — time of day
5. `dayofweek` OHE columns — day-level traffic patterns

---

## 🎯 Key Design Decisions

**1. Feature engineering inside the pipeline.**
`FunctionTransformer(feature_engineering)` makes the pipeline a single serializable object. You call `pipeline.predict(raw_df)` at inference time and get predictions back — no preprocessing code to run separately.

**2. Domain-knowledge cleaning over statistical outlier removal.**
IQR-based outlier removal would discard 25–50% of long trips (they are not errors, just rare). The bounding box and duration thresholds are grounded in what is physically possible for a NYC taxi.

**3. Log-transform the target, not the features.**
`distance_km` is right-skewed too, but we do not log it — XGBoost handles skewed feature distributions natively via its split criterion. Log-transforming the target is beneficial regardless of model type because it changes the loss landscape.

**4. Cyclical sin/cos encoding for time.**
Even though XGBoost doesn't need it (trees find the right splits regardless), cyclic encoding is included so that swapping to a linear model (e.g. Ridge) later requires zero feature engineering changes.

**5. `remainder='drop'` in ColumnTransformer.**
Explicitly drops any columns not in `NUMERIC_FEATURES` or `CATEGORICAL_FEATURES`. This prevents raw columns like `vendor_id` (before OHE) or intermediate engineered columns from accidentally passing through to the model in their raw form.

**6. No `dropoff_datetime` used.**
Even though the raw dataset contains `dropoff_datetime`, using it would be pure data leakage — the dropoff time is a direct function of the trip duration we are trying to predict.

---

## ⚠️ Data Leakage Warning

This project explicitly identifies and avoids **data leakage**. The most important leaky feature is:

```python
# ✖️ LEAKY — DO NOT USE IN MODELING
df['speed_kmh'] = df['distance_km'] / (df['trip_duration'] / 3600)
```

`speed_kmh` is computed using `trip_duration`, the target variable. Including it in the model would give the model a near-direct path to the answer during training, producing misleadingly high training scores that completely collapse at inference time (when `trip_duration` is unknown). It is computed in the notebook **only for EDA** and is never passed to the pipeline.

Similarly, `dropoff_datetime` would be leaky because `trip_duration = dropoff_datetime - pickup_datetime`.

---

## 📄 License

This project is released for educational and portfolio purposes. The dataset is provided by Kaggle under their [competition rules](https://www.kaggle.com/c/nyc-taxi-trip-duration/rules).
