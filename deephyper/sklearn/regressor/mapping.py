"""Mapping of available classifiers for automl.
"""

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

REGRESSORS = {
    "RandomForest": RandomForestRegressor,
    "Linear": LinearRegression,
    "AdaBoost": AdaBoostRegressor,
    "KNeighbors": KNeighborsRegressor,
    "MLP": MLPRegressor,
    "SVR": SVR,
    "XGBoost": XGBRegressor,
}
