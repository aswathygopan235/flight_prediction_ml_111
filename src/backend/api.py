from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Self

from fastapi.exceptions import RequestValidationError


import numpy as np
import pandas as pd

import joblib
import logging
import sys
import xgboost as xgb
from pydantic import BaseModel, field_validator, ValidationError, BeforeValidator, model_validator
from typing import Optional

from xgboost import XGBRegressor
import pickle
from src.util.transformer.transformer import FeatureEngineer
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import Annotated


def is_valid_city(value: str) -> str:
    """Validate against set of cities"""
    valid_city = ['Delhi', 'Mumbai', 'Bangalore',
                  'Kolkata', 'Hyderabad', 'Chennai']
    if value not in valid_city:
        raise ValueError("Not in set of approved cities")
    return value


def is_valid_airline(value: str) -> str:
    """Validate against set of airline"""
    valid_airline = [
        'SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
    if value not in valid_airline:
        raise ValueError("Not in set of approved airline")
    return value


class Flight(BaseModel):
    airline: Annotated[str, BeforeValidator(is_valid_airline)]
    source_city: Annotated[str, BeforeValidator(is_valid_city)]
    destination_city: Annotated[str, BeforeValidator(is_valid_city)]
    stops: str
    departure_time: str
    arrival_time: str
    haul_type: str
    days_left: int
    route: Optional[str] = None
    days_bucket: Optional[str] = None

    @model_validator(mode="before")
    def verify_city(self) -> Self:
        if (self["source_city"] == self["destination_city"]):
            raise ValueError("Source and destination city must not be same")
        return self


app = FastAPI()
# model = XGBRegressor(enable_categorical=True)
# model.load_model('models/economy_pipeline.json')
# print(model)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Override"""
    txt = []
    for err in exc.errors():

        field = err["loc"][0]

        if (len(err["loc"]) > 1):
            field = err["loc"][1]

        row = {"field": field, "msg": err["msg"]}
        txt.append(row)
        print(err["loc"])

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=jsonable_encoder(
            {"status": False, "errors": txt}),
    )


model = joblib.load("src/models/economy_pipeline.joblib")

# with open('backend/models/economy_pipeline.pkl', 'rb') as f:
#     model = pickle.load(f)
origins = [
    "http://0.0.0.0:8000/",
    "http://localhost:8000/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]

)

# logger = logging.getLogger('uvicorn.error')
# logger.setLevel(logging.DEBUG)


@app.get("/")
def home():
    """Function is home index"""
    return ("API running")


@app.post("/predict")
async def predict(data: Flight):
    """API to predict"""
    sample_data = pd.DataFrame([{
        "airline": data.airline,
        "source_city": data.source_city,
        "destination_city": data.destination_city,
        "route": data.route,
        "departure_time": data.departure_time,
        "arrival_time": data.arrival_time,
        "days_left": data.days_left,
        "stops": data.stops,
        "haul_type": data.haul_type,
        "days_bucket": data.days_bucket
    }])

    # print(sample_data)

    # sample_data = [{"airline": data.airline,
    #                "source_city": data.source_city,
    #                 "destination_city": data.destination_city,
    #                 "route": data.route,
    #                 "departure_time": data.departure_time,
    #                 "arrival_time": data.arrival_time,
    #                 "days_left": data.days_left,
    #                 "stops": data.stops,
    #                 "haul_type": data.haul_type,
    #                 "days_bucket": data.days_bucket

    #                 }]
    # flight_data = pd.DataFrame(data=to_dict)

    # sample_data = [[
    #     data.airline,
    #     data.source_city,
    #     data.destination_city,
    #     data.route,
    #     data.departure_time,
    #     data.arrival_time,
    #     data.days_left,
    #     data.stops,
    #     data.haul_type

    # ]]

#     route = inp[0]["source_city"]+"_"+inp[0]["destination_city"]
#     inp[0]["route"] = route
#     # features = list(data.keys)
#     # print(data)
# #
#     res = np.column_stack((inp.keys(), inp.values()))

    # print(flight_data["route"])

#     # inp = np.array(data.features)

#     print(res.shape)

    # feat = np.array()

    # prediction = model.predict([[data.airline, data.arrival_time,
    #                             data.departure_time, data.days_left, data.source_city, data.destination_city, data.route, data.stops]])

    # sample_data = pd.DataFrame(data=data.model_dump())

    prediction = model.predict(sample_data)
    result = float(np.expm1(prediction[0]))
    print(type(result))
    print(result)
    d = {
        "result": np.round(result, 2)
    }
    return JSONResponse(content=jsonable_encoder(d))

# return data

# logger.debug('this is a debug message')
