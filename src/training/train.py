

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pickle
from ..util.transformer.transformer import FeatureEngineer

import sklearn


def find_haul_type(duration):
    """Find haul class depending on flight duration"""
    if ((duration >= 0) and (duration < 3)):
        return "short_haul"
    if ((duration >= 3) and (duration < 6)):
        return "medium_haul"
    if ((duration >= 6) and (duration < 16)):
        return "long_haul"
    if (duration >= 16):
        return "ultra_long_haul"

    return "invalid"


def train():
    print("loading")
    print(sklearn.__version__)
    print(pd.__version__)
    df_full = pd.read_csv("src/data/flight_price_prediction.csv")

    # df_full["route"] = df_full["source_city"]+"_"+df_full["destination_city"]

    df = df_full[df_full["class"] == "Economy"]
    df = df[df["price"] < df["price"].quantile(0.99)]
    # df["days_bucket"] = pd.cut(df["days_left"], bins=[0, 2, 5, 10, 50, 100], labels=[
    #                            "last_min", "very_close", "mid", "early", "far"])

    # df["route_days"] = df["route"]+"_"+df["days_bucket"].astype(str)
    # df["airline_days"] = df["airline"]+"_"+df["days_bucket"].astype(str)

    print(df.info())
    print(df.describe())

    df["haul_type"] = df.apply(
        lambda row: find_haul_type(row["duration"]), axis=1)

    print(df["airline"].unique())
    print(df["source_city"].unique())
    print(df["stops"].unique())
    print(df["destination_city"].unique())

    target = "price"

    preprocessor = ColumnTransformer(transformers=[
        ("scale_numbers", StandardScaler(), ["days_left"]),

        # ("ordinal_day_bucket_encoder", OrdinalEncoder(categories=[
        #  ["last_min", "very_close", "mid", "early", "far", ]]), ["days_bucket"]),
        # ("ordinal_stop_encoder", OrdinalEncoder(
        #     categories=[["zero", "one", "two_or_more"]]), ["stops"]),
        ("cat_encoding", OneHotEncoder(handle_unknown='ignore'), [
            "airline",
            "source_city",
            "destination_city",
            "departure_time",
            "arrival_time",
            "route_days",
            # "airline_days",
            "stops",
            # "days_bucket",
            "haul_type",
            "route"])
    ],
        remainder="drop")

    model_xgboost = XGBRegressor(
        objective='reg:squarederror',
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    param_grid = {
        "sample_weight": 4
    }
    pipeline = Pipeline(steps=([
        ("feature_engineer", FeatureEngineer()),
        ("preprocess", preprocessor),
        ("model", model_xgboost)
    ]))

    X_full = df.drop([target], axis=1)
    y_full = np.log(df[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=2/7, random_state=0)

    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_test = np.exp(y_test)
    print(np.round(mean_absolute_percentage_error(y_pred, y_test), 2))
    print(np.round(root_mean_squared_log_error(y_pred, y_test), 2))
    print(np.round(mean_absolute_error(y_pred, y_test), 2))

    joblib.dump(filename="src/models/economy_pipeline.joblib",
                value=pipeline)
    # with open('backend/models/economy_pipeline.pkl', 'wb') as f:
    #     pickle.dump(pipeline, f)


if __name__ == "__main__":
    train()
