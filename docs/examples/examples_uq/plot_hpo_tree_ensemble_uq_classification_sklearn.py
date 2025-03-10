r"""
Hyperparameter Optimized Ensemble of Random Decision Trees with Uncertainty for Classification
==============================================================================================

**Author(s)**: Romain Egele.

In this tutorial, you will learn about how to use hyperparameter optimization to generate ensemble of `Scikit-Learn <https://scikit-learn.org/stable/>`_ models that can be used for uncertainty quantification.
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
#     pip install "deephyper[ray]"

# %%

# .. dropdown:: Import statements
import pathlib
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import make_moons
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

WIDTH_PLOTS = 8
HEIGHT_PLOTS = WIDTH_PLOTS / 1.618

# %%
# Synthetic data generation
# -------------------------
# 
# For the data, we use the :func:`sklearn.datasets.make_moons` functionality from Scikit-Learn to have a synthetic binary-classification problem with two moons.
# The input data :math:`x` are two dimensionnal and the target data :math:`y` are binary values.
# We randomly flip 10% of the labels to generate artificial noise that should later be estimated by what we call "aleatoric uncertainty" (a.k.a., intrinsic random noise).

# .. dropdown:: Loading synthetic data
def flip_binary_labels(y, ratio, random_state=None):
    """Increase the variance of P(Y|X) by ``ratio``"""
    y_flipped = np.zeros(np.shape(y))
    y_flipped[:] = y[:]
    rs = np.random.RandomState(random_state)
    idx = np.arange(len(y_flipped))
    idx = rs.choice(idx, size=int(ratio * len(y_flipped)), replace=False)
    y_flipped[idx] = 1 - y_flipped[idx]
    return y_flipped


def load_data(noise=0.1, n=1_000, ratio_flipped=0.1, test_size=0.33, valid_size=0.33, random_state=42):
    rng = np.random.RandomState(random_state)
    max_int = np.iinfo(np.int32).max

    test_size = int(test_size * n)
    valid_size = int(valid_size * n)

    X, y = make_moons(n_samples=n, noise=noise, shuffle=True, random_state=rng.randint(max_int))
    X = X - np.mean(X, axis=0)

    y = flip_binary_labels(y, ratio=ratio_flipped, random_state=rng.randint(max_int))
    y = y.astype(np.int64)

    train_X, test_X, train_y, test_y = train_test_split(
        X, 
        y, 
        test_size=test_size,
        random_state=rng.randint(max_int),
        stratify=y,
    )

    train_X, valid_X, train_y, valid_y = train_test_split(
        train_X,
        train_y, 
        test_size=valid_size, 
        random_state=rng.randint(max_int), 
        stratify=train_y,
    )

    return (train_X, train_y), (valid_X, valid_y), (test_X, test_y)

(x, y), (vx, vy), (tx, ty) = load_data()

_ = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS), tight_layout=True)
_ = plt.scatter(
    x[:, 0].reshape(-1), x[:, 1].reshape(-1), c=y, label="train", alpha=0.8
)
_ = plt.scatter(
    vx[:, 0].reshape(-1),
    vx[:, 1].reshape(-1),
    c=vy,
    marker="s",
    label="valid",
    alpha=0.8,
)
_ = plt.ylabel("$x1$", fontsize=12)
_ = plt.xlabel("$x0$", fontsize=12)
_ = plt.legend(loc="upper center", ncol=3, fontsize=12)

# %%
# Training a Decision Tree
# ------------------------
# 
# We focus on the class of random decision tree models. 
# We define a function that trains and evaluate a random decision tree from given parameters ``job.parameters``.
# These parameters will be optimized in the next steps by DeepHyper.
#
# The score we minimize with respect to hyperparameters $\theta$ is the validation log loss (a.k.a., binary cross entropy) as we want to have calibrated uncertainty estimates of :math:`P(Y|X=x)` and :math:`1-P(Y|X=x)`:
#
# .. math::
#
#     L_\text{BCE}(x, y;\theta) = y \cdot \log\left(p(y|x;\theta)\right) + (1 - y) \cdot \log\left(1 - p(y|x\theta)\right)
#
# where :math:`p(y|x;\theta)` is the predited probability of a tree with hyperparameters :math:`\theta`.

# .. dropdown:: Plot decision boundary
def plot_decision_boundary_decision_tree(dataset, labels, model, steps=1000, color_map="viridis", ax=None):
    color_map = plt.get_cmap(color_map)
    # Define region of interest by data limits
    xmin, xmax = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    ymin, ymax = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels_predicted = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels_predicted[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, z, cmap=color_map, alpha=0.5)

    # Get predicted labels on training data and plot
    ax.scatter(
        dataset[:, 0],
        dataset[:, 1],
        c=labels,
        # cmap=color_map,
        lw=0,
    )

# %%
# The ``run`` function takes a ``job`` object as input suggested by DeepHyper.
# We use it to pass the ``job.parameters`` and create the decision tree ``model``. 
# Then, we fit the model on the data on compute its log-loss score on the validation dataset.
# In case of unexpected error we return a special value ``F_fit`` so that our hyperparameter optimization can learn to avoid these unexepected failures.
# We checkpoint the model on disk as ``model_*.pkl`` files.
# Finally, we return all of our scores, the ``"objective"`` is the value maximized by DeepHyper. Other scores are returned as metadata for further analysis (e.g., overfitting, underfitting, etc.).
hpo_dir = "hpo_sklearn_classification"
model_checkpoint_dir = os.path.join(hpo_dir, "models")


def run(job, model_checkpoint_dir=".", verbose=True, show_plots=False):

    (x, y), (vx, vy), (tx, ty) = load_data()

    model = DecisionTreeClassifier(**job.parameters)

    try:
        model.fit(x, y)
        vy_pred_proba = model.predict_proba(vx)
        val_cce = log_loss(vy, vy_pred_proba)
    except:
        return "F_fit"

    # Saving the model
    with open(os.path.join(model_checkpoint_dir, f"model_{job.id}.pkl"), "wb") as f:
        pickle.dump(model, f)

    if verbose:
        print(f"{job.id}: {val_cce=:.3f}")

    if show_plots:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(WIDTH_PLOTS, HEIGHT_PLOTS*2), tight_layout=True)
        plot_decision_boundary_decision_tree(tx, ty, model, steps=1000, color_map="viridis", ax=axes[0])
        disp = CalibrationDisplay.from_predictions(ty, model.predict_proba(tx)[:, 1], ax=axes[1])

    test_cce = log_loss(ty, model.predict_proba(tx))
    test_acc = accuracy_score(ty, model.predict(tx))

    # The score is negated for maximization
    # The score is -Categorical Cross Entropy/LogLoss
    return {
        "objective": -val_cce,
        "metadata": {"test_cce": test_cce, "test_acc": test_acc},
    }

# %%
# It is important to note that we did not fix the random state of the random decision tree.
# The hyperparameter optimization takes into consideration the fact that the observed objective is noisy and of course this can be tuned.
# For example, as the default surrogate model of DeepHyper is itself a randomized forest, increasing the number of samples in leaf nodes would have the effect of averaging out the prediction of the surrogate.
#
# Also, the point of ensembling randomized decision trees is to build a model with lower variance (i.e., variability of the score when fitting it) than its base estimators.

# %%
# Hyperparameter search space
# ---------------------------
#
# We define the hyperparameter search space for decision trees.
# This tells to DeepHyper the hyperparameter values it can use for the optimization.
# To define these hyperparameters we look at the `DecisionTreeClassifier API Reference <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_.
from deephyper.hpo import HpProblem


def create_hpo_problem():

    problem = HpProblem()

    problem.add_hyperparameter(["gini", "entropy", "log_loss"], "criterion")
    problem.add_hyperparameter(["best", "random"], "splitter")
    problem.add_hyperparameter((10, 1000, "log-uniform"), "max_depth", default_value=1000)
    problem.add_hyperparameter((2, 20), "min_samples_split", default_value=2)
    problem.add_hyperparameter((1, 20), "min_samples_leaf", default_value=1)
    problem.add_hyperparameter((0.0, 0.5), "min_weight_fraction_leaf", default_value=0.0)

    return problem

problem = create_hpo_problem()
problem

# %%
# Evaluation of the baseline
# --------------------------
# 
# We previously defined ``default_value=...`` for each hyperparameter. These values corresponds to the default hyperparameters used in Scikit-Learn. We now test them to have a base performance.
from deephyper.evaluator import RunningJob


def evaluate_decision_tree(problem):

    model_checkpoint_dir = "models_sklearn_test"
    pathlib.Path(model_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    default_parameters = problem.default_configuration
    print(f"{default_parameters=}")
    
    output = run(
        RunningJob(id="test", parameters=default_parameters),
        model_checkpoint_dir=model_checkpoint_dir,
        show_plots=True,
    )
    return output

baseline_output = evaluate_decision_tree(problem)
baseline_output

# %%
# The accuracy is great, but the uncertainty is not well calibrated.

# %%
# Hyperparameter Optimization
# ---------------------------
#
# In DeepHyper, instead of just performing sequential Bayesian optimization we provide asynchronous parallelisation for
# Bayesian optimization (and other methods). This allows to execute multiple evaluation function in parallel to collect 
# observations of objectives faster.
#
# In this example, we will focus on using centralized Bayesian optimization (CBO). In this setting, we have one main process that runs the
# Bayesian optimization algorithm and we have multiple worker processes that run evaluation functions. The class we use for this is
# :class:`deephyper.hpo.CBO`.
#
# Let us start by explaining import configuration parameters of :class:`deephyper.hpo.CBO`:
# 
# - ``initial_points``: is a list of initial hyperparameter configurations to test, we add the baseline hyperparameters as we want to be at least better than this configuration.
# - ``surrogate_model_*``: are parameters related to the surrogate model we use, here ``"ET"`` is an alias for the Extremely Randomized Trees regression model.
# - ``multi_point_strategy``: is the strategy we use for parallel suggestion of hyperparameters, here we use the ``qUCBd`` that will sample for each new parallel configuration a different :math:`\kappa^j_i` value from an exponential with mean :math:`\kappa_i` where :math:`j` is the index in the current generated parallel batch and :math:`i` is the iteration of the Bayesian optimization loop. ``UCB`` corresponds to the Upper Confidence Bound acquisition function. Finally the ``"d"`` postfix in ``qUCBd`` means that we will only consider the epistemic component of the uncertainty returned by the surrogate model.
# - ``acq_optimizer_*``: are parameters related to optimization of the previously defined acquisition function.
# - ``kappa`` and ``scheduler``: are the parameters that define the schedule of :math:`\kappa^j_i` previously mentionned.
# - ``objective_scaler``: is a parameter that can be used to rescale the observed objectives (e.g., identity, min-max, log).
search_kwargs = {
    "initial_points": [problem.default_configuration],
    "n_initial_points": 2 * len(problem) + 1,  # Number of initial random points
    "surrogate_model": "ET",  # Use Extra Trees as surrogate model
    "surrogate_model_kwargs": {
        "n_estimators": 50,  # Relatively small number of trees in the surrogate to make it "fast"
        "min_samples_split": 8,  # Larger number to avoid small leaf nodes (smoothing the objective response)
    },
    "multi_point_strategy": "qUCBd",  # Multi-point strategy for asynchronous batch generations (explained later)
    "acq_optimizer": "sampling",  # Use random sampling for the acquisition function optimizer
    "filter_duplicated": False,  # Deactivate filtration of duplicated new points
    "kappa": 10.0,  # Initial value of exploration-exploitation parameter for the acquisition function
    "scheduler": {  # Scheduler for the exploration-exploitation parameter "kappa"
        "type": "periodic-exp-decay",  # Periodic exponential decay
        "period": 50,  # Period over which the decay is applied. It is useful to escape local solutions.
        "kappa_final": 0.001,  # Value of kappa at the end of each "period"
    },
    "objective_scaler": "identity",
    "random_state": 42,  # Random seed
}

# %% 
# Then we can run the optimization.

from deephyper.hpo import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback


def run_hpo(problem):

    pathlib.Path(model_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator.create(
        run,
        method="ray",
        method_kwargs={
            "num_cpus_per_task": 1,
            "run_function_kwargs": {
                "model_checkpoint_dir": model_checkpoint_dir,
                "verbose": False,
            },
            "callbacks": [TqdmCallback()]
        },
    )
    search = CBO(
        problem,
        evaluator,
        log_dir=hpo_dir,
        **search_kwargs,
    )

    results = search.search(max_evals=1_000)

    return results

results = run_hpo(problem)

# %%
# Analysis of the results
# -----------------------
#
# The results of the HPO is a dataframe.
# The columns starting with ``p:`` are the hyperparameters.
# The columns starting with ``m:`` are the metadata.
# There are also special columns: ``objective``, ``job_id`` and ``job_status``.

results

# %%
# Evolution of the objective
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use :func:`deephyper.analysis.hpo.plot_search_trajectory_single_objective_hpo` to look at the evolution of the objective during the search.

# .. dropdown:: Plot search trajectory
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo


_, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS), tight_layout=True)
_ = plot_search_trajectory_single_objective_hpo(results, mode="min", ax=ax)
ax.axhline(-baseline_output["objective"], linestyle="--", color="red", label="baseline")
ax.set_yscale("log")

# %%
# The dashed red horizontal line corresponds to the baseline performance.

# %%
# Worker utilization
# ~~~~~~~~~~~~~~~~~~
#
# We use :func:`deephyper.analysis.hpo.plot_worker_utilization` to look at the number of active workers over the search.

# .. dropdown:: Plot worker utilization
from deephyper.analysis.hpo import plot_worker_utilization

_, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS), tight_layout=True)
_ = plot_worker_utilization(results, ax=ax)

# %% 
# Best decision tree
# ~~~~~~~~~~~~~~~~~~~
#
# Then, we look indivudualy at the performance of the top 5 models by using :func:`deephyper.analysis.hpo.parameters_from_row`:
from deephyper.analysis.hpo import parameters_from_row


topk_rows = results.nlargest(5, "objective").reset_index(drop=True)

for i, row in topk_rows.iterrows():
    parameters = parameters_from_row(row)
    obj = row["objective"]
    print(f"Top-{i+1} -> {obj=:.3f}: {parameters}")
    print()

# %%
# If we just plot the decision boundary and calibration plots of the best model we can
# observe a significant improvement over the baseline with log-loss values around 0.338 when it
# was previously around 6.

best_job = topk_rows.iloc[0]
hpo_dir = "hpo_sklearn_classification"
model_checkpoint_dir = os.path.join(hpo_dir, "models")
with open(os.path.join(model_checkpoint_dir, f"model_0.{best_job.job_id}.pkl"), "rb") as f:
    best_model = pickle.load(f)

# %%

# .. dropdown:: Plot decision boundary and calibration
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(WIDTH_PLOTS, HEIGHT_PLOTS*2), tight_layout=True)
plot_decision_boundary_decision_tree(tx, ty, best_model, steps=1000, color_map="viridis", ax=axes[0])
disp = CalibrationDisplay.from_predictions(ty, best_model.predict_proba(tx)[:, 1], ax=axes[1])

# %% 
# Ensemble of decision trees
# --------------------------
#
# We now move to ensembling checkpointed models and we start by importing utilities from :module:`deephyper.ensemble` and `deephyper.predictor`.
from deephyper.ensemble import EnsemblePredictor
from deephyper.ensemble.aggregator import MixedCategoricalAggregator
from deephyper.ensemble.loss import CategoricalCrossEntropy 
from deephyper.ensemble.selector import GreedySelector, TopKSelector
from deephyper.predictor.sklearn import SklearnPredictorFileLoader

# %%

# .. dropdown:: Plot decision boundary and uncertainty
def plot_decision_boundary_and_uncertainty(
    dataset, labels, model, steps=1000, color_map="viridis", s=5
):

    fig, axs = plt.subplots(
        3, sharex="all", sharey="all", figsize=(WIDTH_PLOTS, HEIGHT_PLOTS * 2), tight_layout=True,
    )

    # Define region of interest by data limits
    xmin, xmax = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    ymin, ymax = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    y_pred = model.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    y_pred_proba = y_pred["loc"]
    y_pred_aleatoric = y_pred["uncertainty_aleatoric"]
    y_pred_epistemic = y_pred["uncertainty_epistemic"]

    # Plot decision boundary in region of interest

    # 1. MODE
    color_map = plt.get_cmap("viridis")
    z = y_pred_proba[:, 1].reshape(xx.shape)

    cont = axs[0].contourf(xx, yy, z, cmap=color_map, vmin=0, vmax=1, alpha=0.5)

    # Get predicted labels on training data and plot
    axs[0].scatter(
        dataset[:, 0],
        dataset[:, 1],
        c=labels,
        cmap=color_map,
        s=s,
        lw=0,
    )
    plt.colorbar(cont, ax=axs[0], label="Probability of class 1")

    # 2. ALEATORIC
    color_map = plt.get_cmap("plasma")
    z = y_pred_aleatoric.reshape(xx.shape)

    cont = axs[1].contourf(xx, yy, z, cmap=color_map, vmin=0, vmax=0.69, alpha=0.5)

    # Get predicted labels on training data and plot
    axs[1].scatter(
        dataset[:, 0],
        dataset[:, 1],
        c=labels,
        cmap=color_map,
        s=s,
        lw=0,
    )
    plt.colorbar(cont, ax=axs[1], label="Aleatoric uncertainty")

    # 3. EPISTEMIC
    z = y_pred_epistemic.reshape(xx.shape)

    cont = axs[2].contourf(xx, yy, z, cmap=color_map, vmin=0, vmax=0.69, alpha=0.5)

    # Get predicted labels on training data and plot
    axs[2].scatter(
        dataset[:, 0],
        dataset[:, 1],
        c=labels,
        cmap=color_map,
        s=s,
        lw=0,
    )
    plt.colorbar(cont, ax=axs[2], label="Epistemic uncertainty")


# %%
# We define a function that will create an ensemble with TopK or Greedy selection strategies.
# This function also has a parameter ``k`` that sets the number of unique member in the ensemble.
def create_ensemble_from_checkpoints(ensemble_selector: str = "topk", k=20):

    # 0. Load data
    _, (vx, vy), _ = load_data()

    # !1.3 SKLEARN EXAMPLE
    predictor_files = SklearnPredictorFileLoader.find_predictor_files(
        model_checkpoint_dir
    )
    predictor_loaders = [SklearnPredictorFileLoader(f) for f in predictor_files]
    predictors = [p.load() for p in predictor_loaders]

    # 2. Build an ensemble
    ensemble = EnsemblePredictor(
        predictors=predictors,
        aggregator=MixedCategoricalAggregator(
            uncertainty_method="entropy",
            decomposed_uncertainty=True,
        ),
        # You can specify parallel backends for the evaluation of the ensemble
        evaluator={
            "method": "ray",
            "method_kwargs": {"num_cpus_per_task": 1},
        },
    )
    y_predictors = ensemble.predictions_from_predictors(
        vx, predictors=ensemble.predictors
    )

    # Use TopK or Greedy/Caruana
    if ensemble_selector == "topk":
        selector = TopKSelector(
            loss_func=CategoricalCrossEntropy(),
            k=20,
        )
    elif ensemble_selector == "greedy":
        selector = GreedySelector(
            loss_func=CategoricalCrossEntropy(),
            aggregator=MixedCategoricalAggregator(),
            k=20,
            k_init=5,
            max_it=100,
            early_stopping=False,
            bagging=True,
            eps_tol=1e-5,
        )
    else:
        raise ValueError(f"Unknown ensemble_selector: {ensemble_selector}")

    selected_predictors_indexes, selected_predictors_weights = selector.select(
        vy, y_predictors
    )
    print(f"{selected_predictors_indexes=}")
    print(f"{selected_predictors_weights=}")

    ensemble.predictors = [ensemble.predictors[i] for i in selected_predictors_indexes]
    ensemble.weights = selected_predictors_weights

    return ensemble

# %%
# We start by testing the Topk strategy.
ensemble = create_ensemble_from_checkpoints("topk")
ty_pred = ensemble.predict(tx)["loc"]
cce = log_loss(ty, ty_pred)
acc = accuracy_score(ty, np.argmax(ty_pred, axis=1))
print(f"Test scores: {cce=:.3f}, {acc=:.3f}")

# %%

# .. dropdown:: Plot decision boundary and uncertainty for ensemble
plot_decision_boundary_and_uncertainty(tx, ty, ensemble, steps=1000, color_map="viridis")

# %%

# .. dropdown:: Plot calibration for ensemble
fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS), tight_layout=True)
disp = CalibrationDisplay.from_predictions(ty, ty_pred[:, 1], ax=ax)

# %%
# We do the same for the Greedy strategy.
ensemble = create_ensemble_from_checkpoints("greedy")
ty_pred = ensemble.predict(tx)["loc"]
cce = log_loss(ty, ty_pred)
acc = accuracy_score(ty, np.argmax(ty_pred, axis=1))
print(f"Test scores: {cce=:.3f}, {acc=:.3f}")

# %%
# sphinx_gallery_thumbnail_number = 8

# .. dropdown:: Plot decision boundary and uncertainty for ensemble
plot_decision_boundary_and_uncertainty(tx, ty, ensemble, steps=1000, color_map="viridis")

# %%

# .. dropdown:: Plot calibration for ensemble
fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS), tight_layout=True)
disp = CalibrationDisplay.from_predictions(ty, ty_pred[:, 1], ax=ax)

# %%
# In conclusion, the improvement over the default hyperparameters is significant.
# 
# For CCE, we improved from about 6 to 0.4.
# 
# For Accuracy, we improved from 0.82 to 0.87.
#
# Not only that we have disentangled uncertainty estimates. The epistemic uncertainty is informative of locations where we are missing data and the aleatoric uncertainty is informative of the noise level in the labels.
