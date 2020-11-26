"""Mapping of available classifiers for automl.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)


CLASSIFIERS = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}
