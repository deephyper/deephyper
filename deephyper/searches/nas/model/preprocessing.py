import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def stdscaler(data):
    """
    Return:
        preprocessor:
    """
    preprocessor = Pipeline([
        ('stdscaler', StandardScaler()),
    ])
    return preprocessor
