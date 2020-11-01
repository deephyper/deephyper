from deephyper.search.hps.automl.classifier import mapping

clf_names = ", ".join(mapping.CLASSIFIERS.keys())

__doc__ = f"Sub-package providing automl tools for a set of classifiers: {clf_names}."

__doc__ += """
AutoML searches are executed with the ``deephyper.search.hps.ambs`` algorithm only. We provide ready to go problems, and run functions for you to use it easily. The following piece of code is an exammple of a problem definition used in :ref:`autosklearn1`.

.. code-block:: python

    import ConfigSpace as cs
    from deephyper.problem import HpProblem


    Problem = HpProblem(seed=45)

    classifier = Problem.add_hyperparameter(
        name="classifier",
        value=["RandomForest", "Logistic", "AdaBoost", "KNeighbors", "MLP", "SVC", "XGBoost"],
    )

    # n_estimators
    n_estimators = Problem.add_hyperparameter(
        name="n_estimators", value=(1, 2000, "log-uniform")
    )

    cond_n_estimators = cs.OrConjunction(
        cs.EqualsCondition(n_estimators, classifier, "RandomForest"),
        cs.EqualsCondition(n_estimators, classifier, "AdaBoost"),
    )

    Problem.add_condition(cond_n_estimators)

    # max_depth
    max_depth = Problem.add_hyperparameter(name="max_depth", value=(2, 100, "log-uniform"))

    cond_max_depth = cs.EqualsCondition(max_depth, classifier, "RandomForest")

    Problem.add_condition(cond_max_depth)

    # n_neighbors
    n_neighbors = Problem.add_hyperparameter(name="n_neighbors", value=(1, 100))

    cond_n_neighbors = cs.EqualsCondition(n_neighbors, classifier, "KNeighbors")

    Problem.add_condition(cond_n_neighbors)

    # alpha
    alpha = Problem.add_hyperparameter(name="alpha", value=(1e-5, 10.0, "log-uniform"))

    cond_alpha = cs.EqualsCondition(alpha, classifier, "MLP")

    Problem.add_condition(cond_alpha)

    # C
    C = Problem.add_hyperparameter(name="C", value=(1e-5, 10.0, "log-uniform"))

    cond_C = cs.OrConjunction(
        cs.EqualsCondition(C, classifier, "Logistic"),
        cs.EqualsCondition(C, classifier, "SVC"),
    )

    Problem.add_condition(cond_C)

    # kernel
    kernel = Problem.add_hyperparameter(
        name="kernel", value=["linear", "poly", "rbf", "sigmoid"]
    )

    cond_kernel = cs.EqualsCondition(kernel, classifier, "SVC")

    Problem.add_condition(cond_kernel)

    # gamma
    gamma = Problem.add_hyperparameter(name="gamma", value=(1e-5, 10.0, "log-uniform"))

    cond_gamma = cs.OrConjunction(
        cs.EqualsCondition(gamma, kernel, "rbf"),
        cs.EqualsCondition(gamma, kernel, "poly"),
        cs.EqualsCondition(gamma, kernel, "sigmoid"),
    )

    Problem.add_condition(cond_gamma)
"""
