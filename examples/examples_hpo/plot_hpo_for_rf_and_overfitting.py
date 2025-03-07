r"""
Hyperparameter optimization and overfitting
===========================================

In this example, you will learn how to treat the choice of a learning method as just another
hyperparameter. We consider the `Random Forest (RF) <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
and `Gradient Boosting (GB) <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_
classifiers from `Scikit-Learn <https://scikit-learn.org/stable/>`_ on the Airlines dataset.

Each classifier has both unique and shared hyperparameters.
We use `ConfigSpace <https://automl.github.io/ConfigSpace/latest/>`_, a Python package for defining conditional hyperparameters and more, to model them.

By using, the objective of hyperparameter properly, and considering hyperparameter optimization as an optimized model selection method, you will also learn how to fight overfitting.
"""
# %%
# Installation and imports
# ------------------------
#
# Installing dependencies with the :ref:`pip installation <install-pip>` is recommended. It requires **Python >= 3.10**.
#
# .. code-block:: bash
#
#     %%bash
#     pip install "deephyper[ray] openml==0.15.1"

# .. dropdown:: Import statements
from inspect import signature

import ConfigSpace as cs
import matplotlib.pyplot as plt
import numpy as np
import openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state, resample

from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem

WIDTH_PLOTS = 8
HEIGHT_PLOTS = WIDTH_PLOTS / 1.618

# %%
# We start by creating a function which loads the data of interest. Here we use the `"Airlines" dataset from
# OpenML <https://www.openml.org/search?type=data&sort=runs&id=1169&status=active>`_ where the task is to
# predict whether a given flight will be delayed, given the information of the scheduled departure.


# .. dropdown:: Loading the data
def load_data(
    random_state=42,
    verbose=False,
    test_size=0.33,
    valid_size=0.33,
    categoricals_to_integers=False,
):
    """Load the "Airlines" dataset from OpenML.

    Args:
        random_state (int, optional): A numpy `RandomState`. Defaults to 42.
        verbose (bool, optional): Print informations about the dataset. Defaults to False.
        test_size (float, optional): The proportion of the test dataset out of the whole data. Defaults to 0.33.
        valid_size (float, optional): The proportion of the train dataset out of the whole data without the test data. Defaults to 0.33.
        categoricals_to_integers (bool, optional): Convert categoricals features to integer values. Defaults to False.

    Returns:
        tuple: Numpy arrays as, `(X_train, y_train), (X_valid, y_valid), (X_test, y_test)`.
    """
    random_state = (
        np.random.RandomState(random_state) if type(random_state) is int else random_state
    )

    dataset = openml.datasets.get_dataset(
        dataset_id=1169,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )

    if verbose:
        print(
            f"This is dataset '{dataset.name}', the target feature is "
            f"'{dataset.default_target_attribute}'"
        )
        print(f"URL: {dataset.url}")
        print(dataset.description[:500])

    X, y, categorical_indicator, ft_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    # encode categoricals as integers
    if categoricals_to_integers:
        for ft_ind, ft_name in enumerate(ft_names):
            if categorical_indicator[ft_ind]:
                labenc = LabelEncoder().fit(X[ft_name])
                X[ft_name] = labenc.transform(X[ft_name])
                n_classes = len(labenc.classes_)
            else:
                n_classes = -1
            categorical_indicator[ft_ind] = (
                categorical_indicator[ft_ind],
                n_classes,
            )

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )

    # relative valid_size on Train set
    r_valid_size = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=r_valid_size,
        shuffle=True,
        random_state=random_state,
    )

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


# %%
# Then, we create a mapping to record the classification algorithms of interest:


CLASSIFIERS = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}

# %%
# Create a baseline code to test the accuracy of each candidate model with its default hyperparameters:


# .. dropdown:: Evaluate baseline models
def evaluate_baseline():
    rs_clf = check_random_state(42)
    rs_data = check_random_state(42)

    ratio_test = 0.33
    ratio_valid = (1 - ratio_test) * 0.33

    train, valid, test = load_data(
        random_state=rs_data,
        test_size=ratio_test,
        valid_size=ratio_valid,
        categoricals_to_integers=True,
    )

    for clf_name, clf_class in CLASSIFIERS.items():
        print("Scoring model:", clf_name)

        clf = clf_class(random_state=rs_clf)

        clf.fit(*train)

        acc_train = clf.score(*train)
        acc_valid = clf.score(*valid)
        acc_test = clf.score(*test)

        print(f"\tAccuracy on Training: {acc_train:.3f}")
        print(f"\tAccuracy on Validation: {acc_valid:.3f}")
        print(f"\tAccuracy on Testing: {acc_test:.3f}\n")


evaluate_baseline()

# %%
# The accuracy values show that the RandomForest classifier with default hyperparameters results in overfitting
# and therefore poor generalization (i.e., high accuracy on training data but not on the validation or test data).
# On the contrary GradientBoosting does not show any sign of overfitting and has a better accuracy on the validation
# and testing set, which shows a better generalization than RandomForest (for the default hyperparameters).
#
# Then, we optimize the hyperparameters, where we seek to find the best classifier and its corresponding best hyperparameters
# to improve the accuracy on the vaidation and test data.
# We create a ``load_subsampled_data`` function to load and return subsampled training and validation data in order to
# speed up the evaluation of candidate models and hyperparameters:


def load_subsampled_data(verbose=0, subsample=True, random_state=None):
    # In this case passing a random state is critical to make sure
    # that the same data are loaded all the time and that the test set
    # is not mixed with either the training or validation set.
    # It is important to not avoid setting a global seed for safety reasons.
    random_state = np.random.RandomState(random_state)

    # Proportion of the test set on the full dataset
    ratio_test = 0.33

    # Proportion of the valid set on "dataset \ test set"
    # here we want the test and validation set to have same number of elements
    ratio_valid = (1 - ratio_test) * 0.33

    # The 3rd result is ignored with "_" because it corresponds to the test set
    # which is not interesting for us now.
    (X_train, y_train), (X_valid, y_valid), _ = load_data(
        random_state=42,
        test_size=ratio_test,
        valid_size=ratio_valid,
        categoricals_to_integers=True,
    )

    # Uncomment the next line if you want to sub-sample the training data to speed-up
    # the search, "n_samples" controls the size of the new training data
    if subsample:
        X_train, y_train = resample(X_train, y_train, n_samples=int(1e4))

    if verbose:
        print(f"X_train shape: {np.shape(X_train)}")
        print(f"y_train shape: {np.shape(y_train)}")
        print(f"X_valid shape: {np.shape(X_valid)}")
        print(f"y_valid shape: {np.shape(y_valid)}")

    return (X_train, y_train), (X_valid, y_valid)


print("Without subsampling")
_ = load_subsampled_data(verbose=1, subsample=False)
print()
print("With subsampling")
_ = load_subsampled_data(verbose=1)


# %%
# Then, we create a ``run`` function to train and evaluate a given hyperparameter configuration. This function has to return a scalar value (typically, a validation accuracy) that is maximized by the hyperparameter optimization algorithm.


# .. dropdown:: Utility function that filters a dictionnary based on the signature of an object
def filter_parameters(obj, config: dict) -> dict:
    """Filter the incoming configuration dict based on the signature of obj.

    Args:
        obj (Callable): the object for which the signature is used.
        config (dict): the configuration to filter.

    Returns:
        dict: the filtered configuration dict.
    """
    sig = signature(obj)
    clf_allowed_params = list(sig.parameters.keys())
    clf_params = {(k[2:] if k.startswith("p:") else k): v for k, v in config.items()}
    clf_params = {
        k: v
        for k, v in clf_params.items()
        if (k in clf_allowed_params and (v not in ["nan", "NA"]))
    }
    return clf_params


# %%
def run(job) -> float:
    config = job.parameters.copy()
    config["random_state"] = check_random_state(42)

    (X_train, y_train), (X_valid, y_valid) = load_subsampled_data(subsample=True)

    clf_class = CLASSIFIERS[config["classifier"]]

    # keep parameters possible for the current classifier
    config["n_jobs"] = 4
    clf_params = filter_parameters(clf_class, config)

    try:  # good practice to manage the fail value yourself...
        clf = clf_class(**clf_params)

        clf.fit(X_train, y_train)

        fit_is_complete = True
    except Exception:
        fit_is_complete = False

    if fit_is_complete:
        y_pred = clf.predict(X_valid)
        acc = accuracy_score(y_valid, y_pred)
    else:
        acc = "F_fit_failed"

    return acc


# %%
# Then, we create the ``HpProblem`` to define the search space of hyperparameters for each model.
#
# The first hyperparameter is ``"classifier"``, the selected model.
#
# Then, we use ``Condition`` and ``Forbidden`` to define constraints on the hyperparameters.
#
# Default values are very important when adding ``Condition`` and ``Forbidden`` clauses.
# Otherwise, the creation of the problem can fail if the default configuration is not acceptable.

problem = HpProblem()

classifier = problem.add_hyperparameter(
    ["RandomForest", "GradientBoosting"], "classifier", default_value="RandomForest"
)

# For both
problem.add_hyperparameter((1, 1000, "log-uniform"), "n_estimators")
problem.add_hyperparameter((1, 50), "max_depth")
problem.add_hyperparameter((2, 10), "min_samples_split")
problem.add_hyperparameter((1, 10), "min_samples_leaf")
criterion = problem.add_hyperparameter(
    ["friedman_mse", "squared_error", "gini", "entropy"],
    "criterion",
    default_value="gini",
)

# GradientBoosting
loss = problem.add_hyperparameter(["log_loss", "exponential"], "loss")
learning_rate = problem.add_hyperparameter((0.01, 1.0), "learning_rate")
subsample = problem.add_hyperparameter((0.01, 1.0), "subsample")

gradient_boosting_hp = [loss, learning_rate, subsample]
for hp_i in gradient_boosting_hp:
    problem.add_condition(cs.EqualsCondition(hp_i, classifier, "GradientBoosting"))

forbidden_criterion_rf = cs.ForbiddenAndConjunction(
    cs.ForbiddenEqualsClause(classifier, "RandomForest"),
    cs.ForbiddenInClause(criterion, ["friedman_mse", "squared_error"]),
)
problem.add_forbidden_clause(forbidden_criterion_rf)

forbidden_criterion_gb = cs.ForbiddenAndConjunction(
    cs.ForbiddenEqualsClause(classifier, "GradientBoosting"),
    cs.ForbiddenInClause(criterion, ["gini", "entropy"]),
)
problem.add_forbidden_clause(forbidden_criterion_gb)

problem

# %%
# Then, we create an ``Evaluator`` object using the ``ray`` backend to distribute the evaluation of the run-function defined previously.

evaluator = Evaluator.create(
    run,
    method="ray",
    method_kwargs={
        "num_cpus_per_task": 1,
        "callbacks": [TqdmCallback()],
    },
)

print("Number of workers: ", evaluator.num_workers)

# %%
# Finally, you can define a Bayesian optimization search called ``CBO`` (for Centralized Bayesian Optimization) and link to it the defined ``problem`` and ``evaluator``.

max_evals = 100

search = CBO(
    problem,
    evaluator,
    acq_func="UCBd",
    acq_func_optimizer="mixedga",
    acq_optimizer_freq=1,
    multi_point_strategy="qUCBd",
    objective_scaler="identity",
    random_state=42,
)
results = search.search(max_evals=max_evals)

# %%
# Once the search is over, a file named ``results.csv`` is saved in the current directory.
# The same dataframe is returned by the ``search.search(...)`` call.
# It contains the hyperparameters configurations evaluated during the search and their corresponding ``objective``
# value (i.e, validation accuracy), ``timestamp_submit`` the time when the evaluator submitted the configuration
# to be evaluated and ``timestamp_gather`` the time when the evaluator received the configuration once evaluated
# (both are relative times with respect to the creation of the ``Evaluator`` instance).

results

# %%

# .. dropdown:: Plot results from hyperparameter optimization
fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
plot_search_trajectory_single_objective_hpo(results, mode="max", ax=ax)
_ = plt.title("Search Trajectory")

# Remember that these results only used a subsample of the training data!
# The baseline with the full dataset reached about the same performance, 0.64 in validation accuracy.

# %%
# Then, we can now look at the Top-3 configuration of hyperparameters.

results.nlargest(n=3, columns="objective")

# %%
# Let us define a test to evaluate the best configuration on the training, validation and test data sets.

def evaluate_config(config):
    config["random_state"] = check_random_state(42)

    rs_data = check_random_state(42)

    ratio_test = 0.33
    ratio_valid = (1 - ratio_test) * 0.33

    train, valid, test = load_data(
        random_state=rs_data,
        test_size=ratio_test,
        valid_size=ratio_valid,
        categoricals_to_integers=True,
    )

    print("Scoring model:", config["p:classifier"])
    clf_class = CLASSIFIERS[config["p:classifier"]]
    config["n_jobs"] = 4
    clf_params = filter_parameters(clf_class, config)

    clf = clf_class(**clf_params)

    clf.fit(*train)

    acc_train = clf.score(*train)
    acc_valid = clf.score(*valid)
    acc_test = clf.score(*test)

    print(f"\tAccuracy on Training: {acc_train:.3f}")
    print(f"\tAccuracy on Validation: {acc_valid:.3f}")
    print(f"\tAccuracy on Testing: {acc_test:.3f}")


config = results.iloc[results.objective.argmax()][:-2].to_dict()
print(f"Best config is:\n {config}")
evaluate_config(config)

# %%
# In conclusion, compared to the default configuration, we can see the accuracy improvement 
# from 0.619 to 0.666 on test data and we can also see the reduction of overfitting between 
# the training and  the validation/test data sets. It was 0.879 training accuracy to 0.619 test accuracy 
# for baseline RandomForest). It is now 0.750 training accuracy to 0.666 test accuracy with the best 
# hyperparameters that selected the RandomForest classifier.
