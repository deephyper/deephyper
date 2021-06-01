"""Mapping of available classifiers for automl.
"""

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

CLASSIFIERS = {
    "RandomForest": RandomForestClassifier,
    "Logistic": LogisticRegression,
    "AdaBoost": AdaBoostClassifier,
    "KNeighbors": KNeighborsClassifier,
    "MLP": MLPClassifier,
    "SVC": SVC,
    "XGBoost": XGBClassifier,
}
