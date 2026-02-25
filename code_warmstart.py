import pandas as pd
import numpy as np
import os

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred, squared=False)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


def approach1(train, test): # direct
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear', 'passenger_count']
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
        ]
        , remainder = 'passthrough'
    )

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', Ridge())
    ])

    model = pipeline.fit(train[train_features], train.log_trip_duration)
    predict_eval(model, train, train_features, "train")
    predict_eval(model, test, train_features, "test")


def prepare_data(train):
    train.drop(columns=['id'], inplace=True)

    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['dayofweek'] = train.pickup_datetime.dt.dayofweek
    train['month'] = train.pickup_datetime.dt.month
    train['hour'] = train.pickup_datetime.dt.hour
    train['dayofyear'] = train.pickup_datetime.dt.dayofyear

    train['log_trip_duration'] = np.log1p(train.trip_duration)


if __name__ == '__main__':
    root_dir = 'project-nyc-taxi-trip-duration'
    train = pd.read_csv(os.path.join(root_dir, 'split_sample/train.csv'))
    test = pd.read_csv(os.path.join(root_dir, 'split_sample/val.csv'))

    prepare_data(train)
    prepare_data(test)

    approach1(train, test)



# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import FunctionTransformer, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
#
# def calculate_distance(lon1, lat1, lon2, lat2):
#     # Radius of the Earth in km
#     R = 6371.0
#
#     # Convert degrees to radians
#     phi1 = np.radians(lat1)
#     phi2 = np.radians(lat2)
#     delta_phi = np.radians(lat2 - lat1)
#     delta_lambda = np.radians(lon2 - lon1)
#
#     # Haversine formula
#     a = np.sin(delta_phi / 2)**2 + \
#         np.cos(phi1) * np.cos(phi2) * \
#         np.sin(delta_lambda / 2)**2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#
#     distance = R * c
#     return distance # in kilometers
#
# def parepare_data(df):
#     df.drop('id', axis=1, inplace=True)
#     df = df[df['trip_duration'] < 4 * 3600]
#     X["pickup_datetime"] = pd.to_datetime(X["pickup_datetime"])
#     X['log_trip_duration'] = np.log1p(X.trip_duration)
#
#
# def feature_engineering(X):
#     X = X.copy()
#
#     X["distance_km"] = calculate_distance(
#         X["pickup_longitude"],
#         X["pickup_latitude"],
#         X["dropoff_longitude"],
#         X["dropoff_latitude"]
#     )
#
#     X["hour"] = X["pickup_datetime"].dt.hour
#     X["day_of_week"] = X["pickup_datetime"].dt.day_name()
#     X["is_weekend"] = X["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
#     X["month"] = X["pickup_datetime"].dt.month
#
#     X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
#     X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
#     X['dayofweek'] = X.pickup_datetime.dt.dayofweek
#     X['month'] = X.pickup_datetime.dt.month
#     X['hour'] = X.pickup_datetime.dt.hour
#     X['dayofyear'] = X.pickup_datetime.dt.dayofyear
#
#
#     X = X.drop(["pickup_datetime"], axis=1)
#
#     return X
#
# def predect():
#     y_pred = modeling_pipeline.predict(X_val)
#     print(r2_score(y_val, y_pred))
#
#
# def abdata(df):
#
#     X = df.drop('trip_duration', axis=1)
#     y = np.log1p(df['trip_duration'])
#
#     numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count',
#               'distance_km', 'is_weekend']
#     categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear', 'passenger_count', 'store_and_fwd_flag', 'day_of_week']
#     feature_transformer = FunctionTransformer(feature_engineering)
#
#     processors = ColumnTransformer(
#         transformers=[
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
#             ('num', StandardScaler(),numeric_features)
#         ]
#     )
#
#     modeling_pipeline = Pipeline(steps=[
#         ('features', feature_transformer),
#         ('processor', processors),
#         ('model', Ridge(alpha=[1.0, 0.1, 0.01, 0.001]))
#     ])
#
#     modeling_pipeline.fit(X_train, y_train)
#
#     predect()
#
# def save_model():
#     pass
#     # import joblib
#     # joblib.dump(modeling_pipeline, "model.pkl")
#     # model = joblib.load("model.pkl")
#     # preds = model.predict(test_df)
#
# if __name__ == "__main__":
#     df = pd.read_csv('split/train.csv')
#
#     abdata(df)
#     save_model()
#