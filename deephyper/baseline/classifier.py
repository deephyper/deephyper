from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


class BaseClassifierPipeline:
    """Baseline classifier to evaluate the problem at stake.

    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from deephyper.baseline import BaseClassifierPipeline
    >>> from sklearn.datasets import load_digits
    >>> load_data = lambda : load_digits(return_X_y=True)
    >>> baseline_classifier = BaseClassifier(KNeighborsClassifier(), load_data)
    >>> baseline_classifier.run()
    """

    def __init__(
        self,
        clf=KNeighborsClassifier(),
        load_data_func=lambda: load_digits(return_X_y=True),
        seed=42,
    ):
        self.clf = clf
        self.seed = seed
        self.load_data_func = load_data_func

    def run(self):

        try:
            (X_train, X_test), (y_train, y_test) = self.load_data_func()
        except:
            X, y = self.load_data_func()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=self.seed
            )

        self.clf.fit(X_train, y_train)

        y_pred = self.clf.predict(X_test)
        acc_not_weighted = accuracy_score(y_test, y_pred)

        print("Not Weighted Accuracy: ", acc_not_weighted)


if __name__ == "__main__":
    baseline = BaseClassifierPipeline()
    baseline.run()
