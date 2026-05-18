
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FeatureEngineer(BaseEstimator, TransformerMixin):

    """Feature Engineering"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit to itself"""
        return self

    def transform(self, X):
        """Adding new columns"""
        X = X.copy()

        X["days_bucket"] = pd.cut(X["days_left"], bins=[0, 2, 5, 10, 50, 100],
                                  labels=[
            "last_min", "very_close", "mid", "early", "far"]).astype(str)

        X["route"] = X["source_city"]+"_"+X["destination_city"]
        X["airline_days"] = X["airline"]+"_"+X["days_bucket"]
        X["route_days"] = X["route"]+"_"+X["days_bucket"]

        return X
