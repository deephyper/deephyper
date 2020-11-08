from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deephyper.nas.preprocessing import minmaxstdscaler
import numpy as np


class BasePipeline:
    """Baseline Pipeline.
    """

    def __init__(
        self,
        clf=None,
        load_data_func=lambda: load_breast_cancer(return_X_y=True),
        preproc=minmaxstdscaler(),
        seed=42,
    ):
        self.clf = clf
        self.seed = seed
        self.load_data_func = load_data_func
        self.preproc = preproc

    def load_data(self):
        try:
            (X_train, y_train), (X_test, y_test) = self.load_data_func()
        except:
            X, y = self.load_data_func()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=self.seed
            )

        return X_train, X_test, y_train, y_test

    def run(self):

        # loading the data
        X_train, _, y_train, _ = self.load_data()

        # preprocessing the data
        X_train = self.preproc.fit_transform(X_train)

        self.clf.fit(X_train, y_train)

        self.evaluate()

    def evaluate(self, metric=None):
        assert callable(metric), "The given metric should be a callable!"

        X_train, X_test, y_train, y_test = self.load_data()

        X_train = self.preproc.transform(X_train)
        X_test = self.preproc.transform(X_test)

        y_pred = self.clf.predict(X_train)
        score_train = metric(y_train, y_pred)

        y_pred = self.clf.predict(X_test)
        score_test = metric(y_test, y_pred)

        metric_name = metric.__name__

        print(f"{metric_name} on Train: ", score_train)
        print(f"{metric_name} on Test: ", score_test)
