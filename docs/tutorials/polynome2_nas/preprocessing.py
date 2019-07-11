from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def minmaxstdscaler():
    """Use MinMaxScaler then StandardScaler.

    Returns:
        preprocessor:
    """

    preprocessor = Pipeline([
        ('minmaxscaler', MinMaxScaler()),
        ('stdscaler', StandardScaler()),
    ])
    return preprocessor
