import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def stdscaler(data):
    """
    Return:
        preprocessor, preproc_data:
    """
    preprocessor = Pipeline([
        ('stdscaler', StandardScaler()),
    ])
    return preprocessor
