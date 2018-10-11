import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def classic(X, y):
    req_data = np.concatenate((X, y), axis=1)
    preprocessor = Pipeline([('stdscaler', StandardScaler()), ('minmax', MinMaxScaler(feature_range=(0, 1)))])
    preproc_data = preprocessor.fit_transform(req_data)
    return preproc_data
