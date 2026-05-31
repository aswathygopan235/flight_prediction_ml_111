# Flight Price Prediction

## Intro

This is an app which predicts the price of economy class tickets based on cities, arrival time, departure time, number of stops

##### UI app

[Try UI app](https://flight-prediction-ml-111-963074352354.europe-west2.run.app/ "link to ui app")

##### view OpenAPI docs

[Go to api docs](https://flight-prediction-ml-111.onrender.com/docs/ "Link api docs")

## Introduction

This is an app which predicts the price of economy class tickets based on cities, arrival time, departure time, number of stops

## Dataset

- **airline**   : The name of the airline within has 6 valid cities

- **flight**: A code associated with the flight

- **source_city** : The starting point of the journey

- **departure_time**: The time slot in which the flight leaves. Can fall into 5 categories : Early Morning, Morning, Afternoon, Night, Late Night

- **stops**: Nunber of connections in between. It can be zero (Direct Flight), one or two or more.

- **arrival_time**:The time slot in which the flight arrives. Can fall into 5 categories : Early Morning, Morning, Afternoon, Night, Late Night

- destination_city: The destination city which has 6  valid cities

- **duration**: The overall time taken for the trip represented on hours

- **days_left**: The number of day in between the date of booking and the trip

- **price**: The target value to predict the price of ticket

## Model

  This is a regression problem where a single value is predicted. [XGboost](https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html "XGBoost Regression") is one of the boosting techinque where the simple decision trees serve as *weak learners*. As with any other boosting methods, a gradient boost tree model is build in stages which optimises for a artbitrary loss function.

## Feature Engineering

  Some field were combined to reduce redundancy. New features were calculated.

  *route* was created by combining *source_city*  and *destination_city*
  *days_bucket* splits the number of days till trip into categories like "last_min", "very_close", "mid", "early", "far".
  *route_days* combined *route* and *days_bucket*
  *airline_days* joins *airline* and *days_bucket*
  *haul_type* depends upon the length of the journey, which can be short_haul, medium_haul,long_haul,ultra_long_haul

### Inputs

  [Standard scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html "Standard Scaler docs") is applied to numerical *days_left* field. 
  *airline*, *source_city*, *destination_city*, *departure_time*, *arrival_time*, *route_days*, *stops*, *haul_type*, *route* are all passed through [One Hot Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html "One Hot Encoder").

### Output

  price is the target value.

### How was it processed

[Pipeline](./pipeline.html)

## Metrics

| Metric                         | Value    |
| ------------------------------ | -------- |
| Mean Absolute Percentage Error | *21%*    |
| Root Mean Squared Log Error    | *26%*    |
| Mean Absolute Error            | *1212.8* |

> The model prediction can be off ±1212.8.  Overall the value can be off by ±21%. So a ticket with original price of 1000 might be predited in the range of 790 to 1021

## References

### Dataset credit

[Dataset - Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction "Kaggle Flight Price prediction")