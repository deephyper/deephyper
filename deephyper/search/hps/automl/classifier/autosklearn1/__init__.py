"""
The problem to use with the ``--problem`` argument is ``deephyper.search.hps.automl.classifier.autosklearn1.Problem``. For the ``--run`` argument you can wrap the ``autosklearn1.run`` function as the following code and then use this new function.

.. code-block:: python

    import numpy as np

    from deephyper.search.hps.automl.classifier import autosklearn1


    def load_data():
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True)
        print(np.shape(X))
        print(np.shape(y))
        return X, y


    def run(config):
        return autosklearn1.run(config, load_data)
"""
from deephyper.search.hps.automl.classifier.autosklearn1.problem import Problem
from deephyper.search.hps.automl.classifier.autosklearn1.run import run
