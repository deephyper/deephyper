
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from deephyper.search.nas.model.preprocessing import minmaxstdscaler


class BaseRegressorPipeline:
    """Baseline regressor to evaluate the problem at stake.

    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from deephyper.baseline import BaseRegressorPipeline
    >>> from sklearn.datasets import load_boston
    >>> load_data = lambda : load_boston(return_X_y=True)
    >>> baseline_regressor = BaseRegressorPipeline(RandomForestRegressor(n_jobs=4), load_data)
    >>> baseline_regressor.run()
    """

    def __init__(
        self,
        clf=RandomForestRegressor(n_jobs=4),
        load_data_func=lambda: load_boston(return_X_y=True),
        seed=42,
    ):
        self.clf = clf
        self.seed = seed
        self.load_data_func = load_data_func

    def run(self):

        # loading the data
        try:
            (X_train, X_test), (y_train, y_test) = self.load_data_func()
        except:
            X, y = self.load_data_func()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=self.seed
            )

        # preprocessing the data
        preproc = minmaxstdscaler()
        X_train = preproc.fit_transform(X_train)
        X_test = preproc.transform(X_test)

        self.clf.fit(X_train, y_train)

        y_pred = self.clf.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        print("R2: ", r2)


if __name__ == "__main__":
    baseline = BaseRegressorPipeline()
    baseline.run()
