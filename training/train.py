
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import logging


from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error


def run_grid_search_cv_and_get_best_hp(var_model, param_grid):
    grid_search_cv = GridSearchCV(
        var_model, param_grid, n_jobs=2, cv=3, scoring="r2", refit=True)
    grid_search_cv.fit(X_valid, y_valid)
    best_hp = grid_search_cv.best_params_
    res = {
        'best_hp': best_hp,
        'cv': grid_search_cv
    }
    return res


def calculate_model_metrics(var_model, X_val, y_val):

    model_name = type(var_model).__name__

    model_prediction_log = var_model.predict(X_val)
    model_prediction = np.expm1(model_prediction_log)

    # model_prediction = np.exp(model_prediction_log)
    # model_prediction = np.exp(model_prediction_log)

    # model_prediction = np.round(model_prediction, 3)

    y_val = np.expm1(y_val)

    model_mae = round(mean_absolute_error(y_val, model_prediction), 2)
    model_rmse = round(root_mean_squared_error(y_val, model_prediction), 2)
    model_r2_score = round(r2_score(y_val, model_prediction), 2)
    model_mape = round(mean_absolute_percentage_error(
        y_val, model_prediction), 2)
    model_rmsle = round(root_mean_squared_log_error(
        y_val, model_prediction), 2)
    res = {
        'metrics': {
            'name': model_name,
            'mae': model_mae,
            'rmse': model_rmse,
            'r2_score': model_r2_score,
            'mape': model_mape,
            'rmsle': model_rmsle
        },
        'prediction': model_prediction,
    }
    return res


df_original = pd.read_csv("data/flight_price_prediction.csv")
df_original = df_original.sample(frac=0.1, random_state=42)

df = df_original.loc[df_original['class'] == "Economy"]

logger = logging.getLogger(__name__)

logging.basicConfig(filename='logs/metrics.log',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'
                    )


print(df.info())

print(df.describe())


airlines = df["airline"].unique()

print(airlines)

df["airline"] = df["airline"].astype("category")

df["flight"] = df["flight"].astype("category")

source_cities = df["source_city"].unique()

print(source_cities)

destination_cities = df["destination_city"].unique()

print(destination_cities)
departure_times = df["departure_time"].unique()


stops = df["stops"].unique()

print(stops)

df["route"] = df["source_city"]+"_"+df["destination_city"]
df["route_and_departure"] = df["route"]+df["departure_time"]


df["duration_bin"] = pd.cut(df["duration"], bins=5)
df["route_and_departure"] = df["route_and_departure"].astype("category")

df["source_city"] = df["source_city"].astype("category")
df["destination_city"] = df["destination_city"].astype("category")
df["route"] = df["route"].astype("category")

class_categories = df["class"].unique()

print(class_categories)


print(departure_times)

print(df.isnull().any(axis=1).sum())


time_categories = ["Early_Morning", "Morning",
                   "Afternoon", "Evening", "Night", "Late_Night"]

# class_categories = ["Economy", "Business"]

stop_categories = ["two_or_more", "one", "zero"]


# time_encoder = OrdinalEncoder(categories=[time_categories])

time_encoder = OneHotEncoder(categories=[time_categories])
airline_encoder = OneHotEncoder(categories=[airlines])


class_encoder = OrdinalEncoder(categories=[class_categories])

stop_encoder = OrdinalEncoder(categories=[stop_categories])


df["departure_time_encoded"] = time_encoder.fit(
    df[["departure_time"]])


df["departure_time_encoded"] = df["departure_time"].astype("category")

df["arrival_time_encoded"] = time_encoder.fit(
    df[["arrival_time"]])
df["arrival_time_encoded"] = df["arrival_time"].astype("category")

df["class_encoded"] = class_encoder.fit_transform(df[["class"]])

df["stop_encoded"] = stop_encoder.fit_transform(df[["stops"]])

df["route_daysleft"] = df["route"].astype(
    "string")+'_'+df["days_left"].astype("string")

df["route_daysleft"] = df["route_daysleft"].astype("category")


full_data = df.copy(deep=True)

target = "price"

useful_features = [
    "departure_time_encoded",
    "arrival_time_encoded",
    "stop_encoded",
    "duration",
    "days_left",
    "airline",
    #    "source_city",
    #    "destination_city",
    #    "route_and_departure",
    # "duration_bin",
    "flight",
    "route"
]

X_full = full_data[useful_features]
y_full = np.log(full_data[target])

print(X_full.info())

print(y_full.head())

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.3, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=2/7, random_state=0)

X_train_original = X_train.copy()
X_valid_original = X_valid.copy()
X_test_original = X_test.copy()

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

X_train.info()

xgboost_param_grid = {'learning_rate': [0.05, 0.10, .25],
                      'min_child_weight': [1, 2, 3],
                      'gamma': [0.5, 1, 2],
                      'subsample': [0.6, 0.8, .4],
                      'colsample_bytree': [0.006, 0.08, 0.9],
                      'eta': [0.3, 0.01],
                      'max_depth': [2, 4, 6]}

xgboost_cv_res = run_grid_search_cv_and_get_best_hp(
    XGBRegressor(enable_categorical=True), xgboost_param_grid)

xgboost_grid_search_cv = xgboost_cv_res['cv']
xgboost_best_hp = xgboost_cv_res['best_hp']

print(xgboost_best_hp)

model_xgboost = XGBRegressor(
    colsample_bytree=xgboost_best_hp['colsample_bytree'],
    gamma=xgboost_best_hp['gamma'],
    enable_categorical=True,
    learning_rate=xgboost_best_hp["learning_rate"],
    max_depth=xgboost_best_hp["max_depth"],
    min_child_weight=xgboost_best_hp["min_child_weight"],
    subsample=xgboost_best_hp["subsample"], n_jobs=1, cv=3, eta=xgboost_best_hp["eta"], scoring="neg_root_mean_squared_error", verbose=1, refit=True)

model_xgboost.fit(X_train, y_train)

train_res = calculate_model_metrics(model_xgboost, X_train, y_train)

xgboost_train_metrics = train_res["metrics"]

print(
    f"{xgboost_train_metrics['name']} train data Mean Absolute Error: {xgboost_train_metrics['mae']}")
print(
    f"{xgboost_train_metrics['name']} train data Root Mean Squared Error: {xgboost_train_metrics['rmse']}")
print(
    f"{xgboost_train_metrics['name']} train data Root Mean Squared Logarithimic Error: {xgboost_train_metrics['rmsle']}")

print(
    f"{xgboost_train_metrics['name']} train data R2 score: {xgboost_train_metrics['r2_score']}")

print(
    f"{xgboost_train_metrics['name']} train data Mean Absolute Percentage Error: {xgboost_train_metrics['mape']}")

test_res = calculate_model_metrics(model_xgboost, X_test, y_test)
xgboost_test_metrics = test_res["metrics"]

print(
    f"{xgboost_test_metrics['name']} test data Mean Absolute Error: {xgboost_test_metrics['mae']}")
print(
    f"{xgboost_test_metrics['name']} test data Root Mean Squared Error: {xgboost_test_metrics['rmse']}")
print(
    f"{xgboost_test_metrics['name']} test data Root Mean Logarithimic Squared Error: {xgboost_test_metrics['rmsle']}")

print(
    f"{xgboost_test_metrics['name']} test data R2 score: {xgboost_test_metrics['r2_score']}")


print(
    f"{xgboost_test_metrics['name']} test data Mean Absolute Percentage Error: {xgboost_test_metrics['mape']}")

logger.info(xgboost_test_metrics)
