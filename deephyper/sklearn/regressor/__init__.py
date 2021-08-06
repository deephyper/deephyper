from deephyper.sklearn.regressor import mapping
from deephyper.sklearn.regressor.autosklearn1.problem import Problem
from deephyper.sklearn.regressor.autosklearn1.run import run

clf_names = ", ".join(mapping.REGRESSORS.keys())

__doc__ = f"Sub-package providing automl tools for a set of regressors: {clf_names}."

__doc__ += """
AutoML searches are executed with the ``deephyper.search.hps.ambs`` algorithm only. We provide ready to go problems, and run functions for you to use it easily. The following piece of code is an example of a provided problem definition.

.. code-block:: python
    :caption: Example Problem

    import ConfigSpace as cs
    from deephyper.problem import HpProblem


    Problem = HpProblem(seed=45)

    regressor = Problem.add_hyperparameter(
        name="regressor",
        value=["RandomForest", "Linear", "AdaBoost", "KNeighbors", "MLP", "SVR", "XGBoost"],
    )

    # n_estimators
    n_estimators = Problem.add_hyperparameter(
        name="n_estimators", value=(1, 2000, "log-uniform")
    )

    cond_n_estimators = cs.OrConjunction(
        cs.EqualsCondition(n_estimators, regressor, "RandomForest"),
        cs.EqualsCondition(n_estimators, regressor, "AdaBoost"),
    )

    Problem.add_condition(cond_n_estimators)

    # max_depth
    max_depth = Problem.add_hyperparameter(name="max_depth", value=(2, 100, "log-uniform"))

    cond_max_depth = cs.EqualsCondition(max_depth, regressor, "RandomForest")

    Problem.add_condition(cond_max_depth)

    # n_neighbors
    n_neighbors = Problem.add_hyperparameter(name="n_neighbors", value=(1, 100))

    cond_n_neighbors = cs.EqualsCondition(n_neighbors, regressor, "KNeighbors")

    Problem.add_condition(cond_n_neighbors)

    # alpha
    alpha = Problem.add_hyperparameter(name="alpha", value=(1e-5, 10.0, "log-uniform"))

    cond_alpha = cs.EqualsCondition(alpha, regressor, "MLP")

    Problem.add_condition(cond_alpha)

    # C
    C = Problem.add_hyperparameter(name="C", value=(1e-5, 10.0, "log-uniform"))

    cond_C = cs.EqualsCondition(C, regressor, "SVR")

    Problem.add_condition(cond_C)

    # kernel
    kernel = Problem.add_hyperparameter(
        name="kernel", value=["linear", "poly", "rbf", "sigmoid"]
    )

    cond_kernel = cs.EqualsCondition(kernel, regressor, "SVR")

    Problem.add_condition(cond_kernel)

    # gamma
    gamma = Problem.add_hyperparameter(name="gamma", value=(1e-5, 10.0, "log-uniform"))

    cond_gamma = cs.OrConjunction(
        cs.EqualsCondition(gamma, kernel, "rbf"),
        cs.EqualsCondition(gamma, kernel, "poly"),
        cs.EqualsCondition(gamma, kernel, "sigmoid"),
    )

    Problem.add_condition(cond_gamma)


The problem to use with the ``--problem`` argument is ``deephyper.sklearn.regressor.Problem``. For the ``--run`` argument you can wrap the ``deephyper.sklearn.regressor.run`` function as the following code and then use this new function:

.. code-block:: python

    import numpy as np

    from deephyper.sklearn.regressor import run as sklearn_run


    def load_data():
        from sklearn.datasets import load_boston

        X, y = load_boston(return_X_y=True)
        print(np.shape(X))
        print(np.shape(y))
        return X, y


    def run(config):
        return sklearn_run(config, load_data)
"""
