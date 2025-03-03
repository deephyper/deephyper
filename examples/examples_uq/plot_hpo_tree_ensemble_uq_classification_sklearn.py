r"""
Hyperparameter Optimization of Decision Tree and Ensemble with Uncertainty Quantification for Classification
============================================================================================================

**Author(s)**: Romain Egele.

In this tutorial, you will learn about how to use hyperparameter optimization to generate ensemble of `Scikit-Learn <https://scikit-learn.org/stable/>`_ models that can be used for uncertainty quantification.
"""
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
# Loading synthetic data
# ----------------------
# 
# For the data, we use the :func:`sklearn.datasets.make_moons` functionality from Scikit-Learn to have a synthetic binary-classification problem with two moons.
# We randomly flip 10% of the labels to generate artificial noise that should later be estimated by what we call "aleatoric uncertainty" (a.k.a., intrinsic random noise).

# .. dropdown:: Load and plot synthetic data
def flip_binary_labels(y, ratio, random_state=None):
    """Increase the variance of P(Y|X) by ``ratio``"""
    y_flipped = np.zeros(np.shape(y))
    y_flipped[:] = y[:]
    rs = np.random.RandomState(random_state)
    idx = np.arange(len(y_flipped))
    idx = rs.choice(idx, size=int(ratio * len(y_flipped)), replace=False)
    y_flipped[idx] = 1 - y_flipped[idx]
    return y_flipped


def load_data(random_state=42):
    noise = 0.1
    n = 1_000
    ratio_flipped = 0.1  # 10% of the labels are flipped

    rng = np.random.RandomState(random_state)
    max_int = np.iinfo(np.int32).max

    # Moons
    make_dataset = lambda n, seed: make_moons(
        n_samples=n,
        noise=noise,
        shuffle=True,
        random_state=seed,
    )

    X, y = make_dataset(n, rng.randint(max_int))
    center = np.mean(X, axis=0)
    X = X - center

    y = flip_binary_labels(y, ratio=ratio_flipped, random_state=rng.randint(max_int))
    y = y.astype(np.int64)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.33, random_state=rng.randint(max_int), stratify=y,
    )

    train_X, valid_X, train_y, valid_y = train_test_split(
        train_X, train_y, test_size=0.33, random_state=rng.randint(max_int), stratify=train_y,
    )

    return (train_X, train_y), (valid_X, valid_y), (test_X, test_y)

(x, y), (vx, vy), (tx, ty) = load_data()

_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
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
# In this tutorial, we focus on the class of random decision tree models. 
# We now define a function that trains and evaluate such a model from given parameters ``job.parameters``.
# These parameters will be optimized in the next steps by DeepHyper.
#
# The score we optimize the validation log loss (a.k.a., binary cross entropy) as we want to have calibrated uncertainty estimates.

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
hpo_dir = "hpo_sklearn_classification"
model_checkpoint_dir = os.path.join(hpo_dir, "models")


def run(job, model_checkpoint_dir=".", verbose=True, show_plots=False):

    (x, y), (vx, vy), (tx, ty) = load_data()

    model = DecisionTreeClassifier(**job.parameters)

    if verbose:
        print(model)

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
        fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
        plot_decision_boundary_decision_tree(
            tx, ty, model, steps=1000, color_map="viridis", ax=ax
        )

        fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
        disp = CalibrationDisplay.from_predictions(ty, model.predict_proba(tx)[:, 1], ax=ax)

    test_cce = log_loss(ty, model.predict_proba(tx))
    test_acc = accuracy_score(ty, model.predict(tx))

    # The score is negated for maximization
    # The score is -Categorical Cross Entropy/LogLoss
    return {
        "objective": -val_cce,
        "metadata": {"test_cce": test_cce, "test_acc": test_acc},
    }

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
    print(f"{output=}")

evaluate_decision_tree(problem)

# %%
# The accuracy is great, but the uncertainty is not well calibrated.

# %%
# Hyperparameter Optimization
# ---------------------------

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
        initial_points=[problem.default_configuration],
        kappa=5.0,
        scheduler={"type": "periodic-exp-decay", "period": 50, "kappa_final": 0.0001},
        objective_scaler="identity",
    )

    # results = search.search(max_evals=1_000)
    results = search.search(max_evals=200)

    return results

results = run_hpo(problem)

# %%
# Analysis of the results
# -----------------------

results

# %%
# Evolution of the objective
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# .. dropdown:: Plot search trajectory
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo


_, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plot_search_trajectory_single_objective_hpo(results, ax=ax)

# %%
# Worker utilization
# ~~~~~~~~~~~~~~~~~~

# .. dropdown:: Plot worker utilization
from deephyper.analysis.hpo import plot_worker_utilization

_, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plot_worker_utilization(results, ax=ax)

# %% 
# The best tecision tree
# ~~~~~~~~~~~~~~~~~~~~~~

from deephyper.analysis.hpo import parameters_from_row

topk_rows = results.nlargest(5, "objective").reset_index(drop=True)

for i, row in topk_rows.iterrows():
    parameters = parameters_from_row(row)
    obj = row["objective"]
    print(f"Top-{i+1} -> {obj=:.3f}: {parameters}")
    print()

best_job = topk_rows.iloc[0]

hpo_dir = "hpo_sklearn_classification"
model_checkpoint_dir = os.path.join(hpo_dir, "models")

with open(os.path.join(model_checkpoint_dir, f"model_0.{best_job.job_id}.pkl"), "rb") as f:
    best_model = pickle.load(f)

# %%

# .. dropdown:: Plot decision boundary and calibration
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
plot_decision_boundary_decision_tree(tx, ty, best_model, steps=1000, color_map="viridis", ax=axes[0])

fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
disp = CalibrationDisplay.from_predictions(ty, best_model.predict_proba(tx)[:, 1], ax=axes[1])

# %% 
# Ensemble of decision trees
# --------------------------
from deephyper.ensemble import EnsemblePredictor
from deephyper.ensemble.aggregator import MixedCategoricalAggregator
from deephyper.ensemble.loss import CategoricalCrossEntropy 
from deephyper.ensemble.selector import GreedySelector, TopKSelector
from deephyper.predictor.sklearn import SklearnPredictorFileLoader

# %%

# .. dropdown:: Make plot with decision boundary and uncertainty
def plot_decision_boundary_and_uncertainty(
    dataset, labels, model, steps=1000, color_map="viridis", s=5
):

    fig, axs = plt.subplots(
        3, sharex="all", sharey="all", figsize=(WIDTH_PLOTS, HEIGHT_PLOTS * 2)
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
def create_ensemble_from_checkpoints(ensemble_selector: str = "topk"):

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
            uncertainty_method="entropy", decomposed_uncertainty=True
        ),
        # ! You can specify parallel backends for the evaluation of the ensemble
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

ensemble = create_ensemble_from_checkpoints("topk")

# %%
plot_decision_boundary_and_uncertainty(tx, ty, ensemble, steps=1000, color_map="viridis")

ty_pred = ensemble.predict(tx)["loc"]

cce = log_loss(ty, ty_pred)
acc = accuracy_score(ty, np.argmax(ty_pred, axis=1))

print(f"{cce=:.3f}, {acc=:.3f}")

# %%
plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
disp = CalibrationDisplay.from_predictions(ty, ty_pred[:, 1])

# %%
ensemble = create_ensemble_from_checkpoints("greedy")

# %%
plot_decision_boundary_and_uncertainty(tx, ty, ensemble, steps=1000, color_map="viridis")

ty_pred = ensemble.predict(tx)["loc"]

cce = log_loss(ty, ty_pred)
acc = accuracy_score(ty, np.argmax(ty_pred, axis=1))

print(f"{cce=:.3f}, {acc=:.3f}")

# %%
# The improvement over the default hyperparameters is significant.
# 
# For CCE, we moved from about 6 to 0.4.
# 
# For Accuracy, we moved from 0.82 to 0.87.

# %%
plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
disp = CalibrationDisplay.from_predictions(ty, ty_pred[:, 1])