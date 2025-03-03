r"""
Hyperparameter Optimization of Decision Tree and Ensemble with Uncertainty Quantification for Classification
============================================================================================================

In this tutorial, we will see how to use Hyperparameter optimization to generate ensemble of models that can be used for uncertainty quantification.

**Author(s)**: Romain Egele.
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
# Data loading
# ------------
# 
# For the data, we use the :func:`sklearn.datasets.make_moons` functionality from Scikit-Learn to have a binary-classification problem.
# 
# In addition, we randomly flip 10% of the labels to generate artificial noise (later corresponding to aleatoric uncertainty).

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
# Training and Scoring of Decision Tree
# -------------------------------------
# 
# The class of model we use in this tutorial is Decision Tree.
# 
# In this part, we will see how to train and evaluate such models.

# .. dropdown:: Utility to plot decision boundary
def plot_decision_boundary_decision_tree(dataset, labels, model, steps=1000, color_map="viridis"):
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

    ax = plt.gca()
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
        plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
        plot_decision_boundary_decision_tree(
            tx, ty, model, steps=1000, color_map="viridis"
        )

        plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
        disp = CalibrationDisplay.from_predictions(ty, model.predict_proba(tx)[:, 1])

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
# 
# We now define the hyperparameter optimization search space for decision trees.

from deephyper.hpo import HpProblem


def create_hpo_problem():

    print(f"--> PID: {os.getpid()}")
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
# Now, we evaluate the baseline Decision Tree model by test `default_value` hyperparameters.

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
        method="loky",
        method_kwargs={
            "num_workers": 4,
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

if __name__ == "__main__":
    results = run_hpo(problem)

