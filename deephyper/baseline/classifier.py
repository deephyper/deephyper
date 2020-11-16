import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from deephyper.baseline.base import BasePipeline
from deephyper.nas.preprocessing import minmaxstdscaler


class BaseClassifierPipeline(BasePipeline):
    """Baseline classifier to evaluate the problem at stake.

    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from deephyper.baseline import BaseClassifierPipeline
    >>> from sklearn.datasets import load_digits
    >>> load_data = lambda : load_digits(return_X_y=True)
    >>> baseline_classifier = BaseClassifierPipeline(KNeighborsClassifier(), load_data)
    >>> baseline_classifier.run() # doctest:+ELLIPSIS
    accuracy_score on Train:...
    accuracy_score on Test:...
    """

    def __init__(
        self,
        clf=KNeighborsClassifier(n_jobs=4),
        load_data_func=lambda: load_breast_cancer(return_X_y=True),
        preproc=minmaxstdscaler(),
        seed=42,
    ):
        super().__init__(
            clf=clf, load_data_func=load_data_func, preproc=preproc, seed=seed
        )

    def evaluate(self, metric=accuracy_score):
        return super().evaluate(metric=metric)


if __name__ == "__main__":
    baseline = BaseClassifierPipeline()
    baseline.run()
