import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from deephyper.nas.preprocessing import minmaxstdscaler
from deephyper.baseline import BasePipeline


class BaseRegressorPipeline(BasePipeline):
    """Baseline regressor to evaluate the problem at stake.

    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from deephyper.baseline import BaseRegressorPipeline
    >>> from sklearn.datasets import load_boston
    >>> load_data = lambda : load_boston(return_X_y=True)
    >>> baseline_regressor = BaseRegressorPipeline(RandomForestRegressor(n_jobs=4), load_data)
    >>> baseline_regressor.run() # doctest:+ELLIPSIS
    r2_score on Train:...
    r2_score on Test:...
    """

    def __init__(
        self,
        clf=RandomForestRegressor(n_jobs=4, random_state=42),
        load_data_func=lambda: load_boston(return_X_y=True),
        preproc=minmaxstdscaler(),
        seed=42,
    ):
        super().__init__(
            clf=clf, load_data_func=load_data_func, preproc=preproc, seed=seed
        )

    def evaluate(self, metric=r2_score):
        return super().evaluate(metric=metric)


if __name__ == "__main__":
    baseline = BaseRegressorPipeline()
    baseline.run()
