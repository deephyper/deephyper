r"""
Neural Architecture Search and Deep Ensemble with Uncertainty Quantification for Regression (Pytorch)
=====================================================================================================

**Author(s)**: Romain Egele, Brett Eiffert.

In this tutorial, you will learn how to perform **Neural Architecture Search (NAS)** and use it to construct a diverse deep ensemble with disentangled **aleatoric** and **epistemic uncertainty**.
 
NAS is the idea of automatically optimizing the architecture of deep neural networks to solve a given task. Here, we will use **hyperparameter optimization (HPO)** algorithms to guide the NAS process.

Specifically, in this tutorial you will learn how to:

1.	**Define a customizable PyTorch module** that exposes neural architecture hyperparameters.
2.	**Define constraints** on the neural architecture hyperparameters to reduce redundancies and improve efficiency of the optimization.

This tutorial will provide a hands-on approach to leveraging NAS for robust regression models with well-calibrated uncertainty estimates.
"""
# .. dropdown:: Import statements
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


WIDTH_PLOTS = 8
HEIGHT_PLOTS = WIDTH_PLOTS / 1.618

# %%
# Synthetic data generation
# -------------------------
# 
# We generate synthetic data from a 1D scalar function :math:`Y = f(X) + \epsilon(X)`, where :math:`X,Y` are random variables with support :math:`\mathbb{R}`.
# 
# The training data are drown uniformly from :math:`X \sim U([-30,-15] \cup [15,30])` with:
# 
# .. math::
# 
#     f(x) = \cos(x/2) + 2 \cdot \sin(x/10) + x/100
# 
# and :math:`\epsilon(X) \sim \mathcal{N}(0, \sigma(X))` with:
#
# - :math:`\sigma(x) = 0.5` if :math:`x \in [-30,-15]`
# - :math:`\sigma(x) = 1.0` if :math:`x \in [15,30]`

# .. dropdown:: Loading synthetic data
def load_data(
    developement_size=500,
    test_size=200,
    random_state=42,
    x_min=-50,
    x_max=50,
):
    rs = np.random.RandomState(random_state)

    def f(x):
        return np.cos(x / 2) + 2 * np.sin(x / 10) + x / 100

    x_1 = rs.uniform(low=-30, high=-15.0, size=developement_size // 2)
    eps_1 = rs.normal(loc=0.0, scale=0.5, size=developement_size // 2)
    y_1 = f(x_1) + eps_1

    x_2 = rs.uniform(low=15.0, high=30.0, size=developement_size // 2)
    eps_2 = rs.normal(loc=0.0, scale=1.0, size=developement_size // 2)
    y_2 = f(x_2) + eps_2

    x = np.concatenate([x_1, x_2], axis=0)
    y = np.concatenate([y_1, y_2], axis=0)

    test_X = np.linspace(x_min, x_max, test_size)
    test_y = f(test_X)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    train_X, valid_X, train_y, valid_y = train_test_split(
        x, y, test_size=0.33, random_state=random_state
    )

    test_X = test_X.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    return (train_X, train_y), (valid_X, valid_y), (test_X, test_y)


(train_X, train_y), (valid_X, valid_y), (test_X, test_y) = load_data()


x_lim, y_lim = 50, 7
_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plt.scatter(train_X, train_y, s=5, label="Training")
_ = plt.scatter(valid_X, valid_y, s=5, label="Validation")
_ = plt.plot(test_X, test_y, linestyle="--", color="gray", label="Test")
_ = plt.fill_between([-30, -15], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.25)
_ = plt.fill_between([15, 30], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.25)
_ = plt.xlim(-x_lim, x_lim)
_ = plt.ylim(-y_lim, y_lim)
_ = plt.legend()
_ = plt.xlabel(r"$x$")
_ = plt.ylabel(r"$f(x)$")
_ = plt.grid(which="both", linestyle=":")

# %%
# Configurable neural network with uncertainty
# --------------------------------------------
#
# We define a configurable Pytorch module to be able to explore:
#
# - the number of layers
# - the number of units per layer
# - the activation function per layer
# - the dropout rate
# - the output layer
#
# The output of this module will be a Gaussian distribution :math:`\mathcal{N}(\mu_\theta(x), \sigma_\theta(x))`, where :math:`\theta` represent the concatenation of the weights and the hyperparameters of our model.
#
# The uncertainty :math:`\sigma_\theta(x)` estimated by the network is an estimator of :math:`V_Y[Y|X=x]` therefore corresponding
# to aleatoric uncertainty (a.k.a., intrinsic noise).

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DeepNormalRegressor(nn.Module):
    def __init__(
        self,
        n_inputs,
        layers,
        n_units_mean=64,
        n_units_std=64,
        std_offset=1e-3,
        softplus_factor=0.05,
        loc=0,
        scale=1.0,
    ):
        super().__init__()

        layers_ = []
        prev_n_units = n_inputs
        for n_units, activation, dropout_rate in layers:
            linear_layer = nn.Linear(prev_n_units, n_units)
            if activation == "relu":
                activation_layer = nn.ReLU()
            elif activation == "sigmoid":
                activation_layer = nn.Sigmoid()
            elif activation == "tanh":
                activation_layer = nn.Tanh()
            elif activation == "swish":
                activation_layer = nn.SiLU()
            elif activation == "mish":
                activation_layer = nn.Mish()
            elif activation == "gelu":
                activation_layer = nn.GELU()
            elif activation == "silu":
                activation_layer = nn.SiLU()
            dropout_layer = nn.Dropout(dropout_rate)

            layers_.extend([linear_layer, activation_layer, dropout_layer])

            prev_n_units = n_units

        # Shared parameters
        self.shared_layer = nn.Sequential(
            *layers_,
        )

        # Mean parameters
        self.mean_layer = nn.Sequential(
            nn.Linear(prev_n_units, n_units_mean),
            nn.ReLU(),
            nn.Linear(n_units_mean, 1),
        )

        # Standard deviation parameters
        self.std_layer = nn.Sequential(
            nn.Linear(prev_n_units, n_units_std),
            nn.ReLU(),
            nn.Linear(n_units_std, 1),
            nn.Softplus(beta=1.0, threshold=20.0),  # enforces positivity
        )

        self.std_offset = std_offset
        self.softplus_factor = softplus_factor
        self.loc = loc
        self.scale = scale

    def forward(self, x):
        # Shared embedding
        shared = self.shared_layer(x)

        # Parametrization of the mean
        mu = self.mean_layer(shared) + self.loc

        # Parametrization of the standard deviation
        sigma = self.std_offset + self.std_layer(self.softplus_factor * shared) * self.scale

        return torch.distributions.Normal(mu, sigma)

# %%
# Hyperparameter search space
# ---------------------------
#
# We define the hyperparameter space that includes both **neural architecture** and **training hyperparameters**.
#
# Without having a good heuristic on training hyperparameters given the neural architecture hyperparameter search space 
# it is important to define them jointly with the neural architecture hyperparameters as they can have strong interactions. 
#
# In the definition of the hyperparameter space, we add constraints using :class:`ConfigSpace.GreaterThanCondition` to
# represent when an hyperparameter is active. In this example, "active" means it actually influence the code execution of
# the trained model.

from ConfigSpace import GreaterThanCondition
from deephyper.hpo import HpProblem


def create_hpo_problem(max_num_layers=10):
    problem = HpProblem()

    # Neural Architecture Hyperparameters
    num_layers = problem.add_hyperparameter((1, max_num_layers), "num_layers", default_value=5)

    conditions = []
    for i in range(max_num_layers):

        # Adding the hyperparameters that impact each layer of the model
        layer_i_units = problem.add_hyperparameter((16, 200), f"layer_{i}_units", default_value=64)
        layer_i_activation = problem.add_hyperparameter(
            ["relu", "sigmoid", "tanh", "swish", "mish", "gelu", "silu"],
            f"layer_{i}_activation",
            default_value="relu",
        )
        layer_i_dropout_rate = problem.add_hyperparameter(
            (0.0, 0.25), f"layer_{i}_dropout_rate", default_value=0.0
        )

        # Adding the constraints to define when these hyperparameters are active
        if i > 0:
            conditions.extend(
                [
                    GreaterThanCondition(layer_i_units, num_layers, i),
                    GreaterThanCondition(layer_i_activation, num_layers, i),
                    GreaterThanCondition(layer_i_dropout_rate, num_layers, i),
                ]
            )

    problem.add_conditions(conditions)

    # Hyperparameters of the output layers
    problem.add_hyperparameter((16, 200), "n_units_mean", default_value=64)
    problem.add_hyperparameter((16, 200), "n_units_std", default_value=64)
    problem.add_hyperparameter((1e-8, 1e-2, "log-uniform"), "std_offset", default_value=1e-3)
    problem.add_hyperparameter((0.01, 1.0), "softplus_factor", default_value=0.05)

    # Training Hyperparameters
    problem.add_hyperparameter((1e-5, 1e-1, "log-uniform"), "learning_rate", default_value=2e-3)
    problem.add_hyperparameter((8, 256, "log-uniform"), "batch_size", default_value=128)
    problem.add_hyperparameter((0.1, 0.99), "lr_scheduler_factor", default_value=0.1)
    problem.add_hyperparameter((10, 100), "lr_scheduler_patience", default_value=20)

    return problem

problem = create_hpo_problem()
problem

# %%
# Loss and Metric
# ---------------
#
# For the loss we will use the Gaussian negative log-likelihood to evalute the quality of the 
# predicted distribution :math:`\mathcal{N}(\mu_\theta(x), \sigma_\theta(x))` using with formula:
#
# .. math::
#
#     L_\text{NLL}(x, y;\theta) = \frac{1}{2}\left(\log\left(\sigma_\theta^{2}(x)\right) + \frac{\left(y-\mu_{\theta}(x)\right)^{2}}{\sigma_{\theta}^{2}(x)}\right) + \text{cst}
# 
# As complementary metric, we use the squared error to evaluate the quality of the mean predictions :math:`\mu_\theta(x)`:
#
# .. math::
#
#     L_\text{SE}(x, y;\theta) = (\mu_\theta(x)-y)^2
#
def nll(y, rv_y):
    """Negative log likelihood for Pytorch distribution.

    Args:
        y: true data.
        rv_y: learned (predicted) probability distribution.
    """
    return -rv_y.log_prob(y)


def squared_error(y_true, rv_y):
    """Squared error for Pytorch distribution.

    Args:
        y: true data.
        rv_y: learned (predicted) probability distribution.
    """
    y_pred = rv_y.mean
    return (y_true - y_pred) ** 2

# %%
# Training loop
# -------------
#
# In our training loop, we make sure to collect training and validation learning curves for better analysis.
#
# We also add a mechanism to checkpoint weights of the model based on the best observed validation loss.
#
# Finally, we add an early stopping mechanism to save computing resources.

# .. dropdown:: Training loop
def train_one_step(model, optimizer, x_batch, y_batch):
    model.train()
    optimizer.zero_grad()
    y_dist = model(x_batch)

    loss = torch.mean(nll(y_batch, y_dist))
    mse = torch.mean(squared_error(y_batch, y_dist))

    loss.backward()
    optimizer.step()

    return loss, mse


def train(
    job,
    model,
    optimizer,
    x_train,
    x_val,
    y_train,
    y_val,
    n_epochs,
    batch_size,
    scheduler=None,
    patience=200,
    progressbar=True,
):
    data_train = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    checkpointed_state_dict = model.state_dict()
    checkpointed_val_loss = np.inf

    train_loss, val_loss = [], []
    train_mse, val_mse = [], []

    tqdm_bar = tqdm(total=n_epochs, disable=not progressbar)

    for epoch in range(n_epochs):
        batch_losses_t, batch_losses_v, batch_mse_t, batch_mse_v = [], [], [], []

        for batch_x, batch_y in data_train:
            b_train_loss, b_train_mse = train_one_step(model, optimizer, batch_x, batch_y)

            model.eval()
            y_dist = model(x_val)
            b_val_loss = torch.mean(nll(y_val, y_dist))
            b_val_mse = torch.mean(squared_error(y_val, y_dist))

            batch_losses_t.append(b_train_loss.detach().numpy())
            batch_mse_t.append(b_train_mse.detach().numpy())
            batch_losses_v.append(b_val_loss.detach().numpy())
            batch_mse_v.append(b_val_mse.detach().numpy())

        train_loss.append(np.mean(batch_losses_t))
        val_loss.append(np.mean(batch_losses_v))
        train_mse.append(np.mean(batch_mse_t))
        val_mse.append(np.mean(batch_mse_v))

        if scheduler is not None:
            scheduler.step(val_loss[-1])

        tqdm_bar.update(1)
        tqdm_bar.set_postfix(
            {
                "train_loss": f"{train_loss[-1]:.3f}",
                "val_loss": f"{val_loss[-1]:.3f}",
                "train_mse": f"{train_mse[-1]:.3f}",
                "val_mse": f"{val_mse[-1]:.3f}",
            }
        )

        # Checkpoint weights if they improve
        if val_loss[-1] < checkpointed_val_loss:
            checkpointed_val_loss = val_loss[-1]
            checkpointed_state_dict = model.state_dict()

        # Early discarding
        job.record(budget=epoch+1, objective=-val_loss[-1])
        if job.stopped():
            break

        if len(val_loss) > (patience + 1) and val_loss[-patience - 1] < min(val_loss[-patience:]):
            break

    # Reload the best weights
    model.load_state_dict(checkpointed_state_dict)

    return train_loss, val_loss, train_mse, val_mse

# %%
# Evaluation function
# -------------------
#
# The evaluation function (often called ``run``-function in DeepHyper) is the function that 
# receives suggested parameters as inputs ``job.parameters`` and returns an ``"objective"`` 
# that we want to maximize.

max_n_epochs = 1_000

def run(job, model_checkpoint_dir=".", verbose=False):
    (x, y), (vx, vy), (tx, ty) = load_data()

    y_mu = np.mean(y)
    y_std = np.std(y)

    inputs = x.shape[1]

    # Create the model based on neural architecture hyperparameters
    num_layers = job.parameters["num_layers"]
    model = DeepNormalRegressor(
        inputs,
        layers=[
            (
                job.parameters[f"layer_{i}_units"],
                job.parameters[f"layer_{i}_activation"],
                job.parameters[f"layer_{i}_dropout_rate"],
            )
            for i in range(num_layers)
        ],
        loc=y_mu,
        scale=y_std,
    )

    if verbose:
        print(model)

    # Initialize training loop based on training hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=job.parameters["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=job.parameters["lr_scheduler_factor"],
        patience=job.parameters["lr_scheduler_patience"],
    )

    x = torch.from_numpy(x).float()
    vx = torch.from_numpy(vx).float()
    tx = torch.from_numpy(tx).float()

    y = torch.from_numpy(y).float()
    vy = torch.from_numpy(vy).float()
    ty = torch.from_numpy(ty).float()

    try:
        train_losses, val_losses, train_mse, val_mse = train(
            job,
            model,
            optimizer,
            x,
            vx,
            y,
            vy,
            n_epochs=max_n_epochs,
            batch_size=job.parameters["batch_size"],
            scheduler=scheduler,
            progressbar=verbose,
        )
    except Exception:
        return "F_fit"

    ty_pred = model(tx)
    test_loss = torch.mean(nll(ty, ty_pred)).detach().numpy()
    test_mse = torch.mean(squared_error(ty, ty_pred)).detach().numpy()

    # Saving the model's state (i.e., weights)
    torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, f"model_{job.id}.pt"))

    return {
        "objective": -val_losses[-1],
        "metadata": {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "test_loss": test_loss,
            "test_mse": test_mse,
        },
    }

# %% 
# Evaluation of the baseline
# --------------------------
#
# We evaluate the default configuration of hyperparameters that we call "baseline" using the same evaluation function.
# This allows to test the evaluation function.

from deephyper.evaluator import RunningJob

baseline_dir = "nas_baseline_regression"

def evaluate_baseline():
    model_checkpoint_dir = os.path.join(baseline_dir, "models")
    pathlib.Path(model_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    problem = create_hpo_problem()

    default_parameters = problem.default_configuration
    print(f"{default_parameters=}\n")

    result = run(
        RunningJob(parameters=default_parameters),
        model_checkpoint_dir=model_checkpoint_dir,
        verbose=True,
    )
    return result

baseline_results = evaluate_baseline()

# %%
# Then, we look at the learning curves of our baseline model returned by the evaluation function.
#
# These curves display a good learning behaviour:
# 
# - the training and validation curves follow each other closely and are decreasing.
# - a clear convergence plateau is reached at the end of the training.

# .. dropdown:: Make learning curves plot
_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))

x_values = np.arange(1, len(baseline_results["metadata"]["train_loss"]) + 1)
_ = plt.plot(
    x_values,
    baseline_results["metadata"]["train_loss"],
    label="Training",
)
_ = plt.plot(
    x_values,
    baseline_results["metadata"]["val_loss"],
    label="Validation",
)

_ = plt.xlim(x_values.min(), x_values.max())
_ = plt.grid(which="both", linestyle=":")
_ = plt.legend()
_ = plt.xlabel("Epochs")
_ = plt.ylabel("NLL")


# %%
# In addition, we look at the predictions by reloading the checkpointed weights.
#
# We first need to recreate the torch module and then we update its state using the checkpointed weights.

weights_path = os.path.join(baseline_dir, "models",  "model_0.0.pt")

y_mu, y_std = np.mean(train_y), np.std(train_y)
inputs = train_X.shape[1]


parameters = problem.default_configuration

num_layers = parameters["num_layers"]
torch_module = DeepNormalRegressor(
    inputs,
    layers=[
        (
            parameters[f"layer_{i}_units"],
            parameters[f"layer_{i}_activation"],
            parameters[f"layer_{i}_dropout_rate"],
        )
        for i in range(num_layers)
    ],
    loc=y_mu,
    scale=y_std,
)

torch_module.load_state_dict(torch.load(weights_path, weights_only=True))
torch_module.eval()

y_pred = torch_module.forward(torch.from_numpy(test_X).float())
y_pred_mean = y_pred.loc.detach().numpy()
y_pred_std = y_pred.scale.detach().numpy()

# %%

# .. dropdown:: Make prediction plot
_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plt.scatter(train_X, train_y, s=5, label="Training")
_ = plt.scatter(valid_X, valid_y, s=5, label="Validation")
_ = plt.plot(test_X, test_y, linestyle="--", color="gray", label="Test")

_ = plt.plot(test_X, y_pred_mean, label=r"$\mu(x)$")
kappa = 1.96
_ = plt.fill_between(
    test_X.reshape(-1),
    (y_pred_mean - kappa * y_pred_std).reshape(-1),
    (y_pred_mean + kappa * y_pred_std).reshape(-1),
    alpha=0.25,
    label=r"$\sigma_\text{al}(x)$",
)

_ = plt.fill_between([-30, -15], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.15)
_ = plt.fill_between([15, 30], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.15)
_ = plt.xlim(-x_lim, x_lim)
_ = plt.ylim(-y_lim, y_lim)
_ = plt.legend(ncols=2)
_ = plt.xlabel(r"$x$")
_ = plt.ylabel(r"$f(x)$")
_ = plt.grid(which="both", linestyle=":")

# %%
# Neural architecture search
# --------------------------
#
# We will now use Bayesian opimization to perform neural architecture search. 
# The sequential Bayesian optimization algorithm can be described by the following pseudo-code:
#
# Sequential Bayesian optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# **Algorithm**: Bayesian Optimization (a.k.a., Efficient Global Optimization (EGO))
#
#   Inputs
#     :math:`\texttt{thetaSpace}`: a hyperparameter space
#
#     :math:`\texttt{nInitial}`: the number of initial hyperparameter configurations
#
#     :math:`\texttt{f}`: a function that returns the objective of the learning workflow
#
#   Outputs
#     :math:`\texttt{thetaStar}` the recommended hyperparameter configuration
#
#   :math:`\texttt{thetaArray}, \texttt{objArray} \gets` New empty arrays of hyperparameter configurations and objectives
#   :math:`\texttt{model} \gets` New surrogate model
#
#   Loop until stopping criteria is not valid
#
#     If Length of :math:`\texttt{thetaArray} < \texttt{nInitial}` then
#
#       :math:`\texttt{theta} \gets` Sample hyperparameter configuration from :math:`\texttt{thetaSpace}`
#
#     Else
#
#       Update :math:`\texttt{model}` with :math:`\texttt{thetaArray}, \texttt{objArray}`
#
#       :math:`\texttt{theta} \gets` Returns :math:`\texttt{theta}` in :math:`\texttt{thetaSpace}` that maximizes 
#       the acquisition function for the current :math:`\texttt{model}`
#
#     :math:`\texttt{obj} \gets` Returns the objective of learning workflow :math:`\texttt{f}(\texttt{theta})`
#
#     :math:`\texttt{thetaArray}  \gets` Concatenate :math:`\texttt{thetaArray}` with :math:`[\texttt{theta}]`
#
#     :math:`\texttt{objArray}  \gets` Concatenate :math:`\texttt{objArray}` with :math:`[\texttt{obj}]`
#
#     :math:`\texttt{thetaStar} \gets` Update recommendation
#
# Parallel Bayesian optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In DeepHyper, instead of just performing sequential Bayesian optimization we provide asynchronous parallelisation for
# Bayesian optimization (and other methods). This allows to execute multiple evaluation function in parallel to collect observations of objectives
# faster.
#
# In this example, we will focus on using centralized Bayesian optimization (CBO). In this setting, we have one main process that runs the
# Bayesian optimization algorithm and we have multiple worker processes that run evaluation functions. The class we use for this is
# :class:`deephyper.hpo.CBO`.
#
# Let us start by explaining import configuration parameters of :class:`deephyper.hpo.CBO`:
# 
# - ``initial_points``: is a list of initial hyperparameter configurations to test, we add the baseline hyperparameters as we want to be at least better than this configuration.
# - ``surrogate_model_*``: are parameters related to the surrogate model we use, here ``"ET"`` is an alias for the Extremely Randomized Trees regression model.
# - ``multi_point_strategy``: is the strategy we use for parallel suggestion of hyperparameters, here we use the `qUCBd` that will sample for each new parallel configuration a different :math:`\kappa^j_i` value from an exponential with mean :math:`\kappa_i` where :math:`j` is the index in the current generated parallel batch and :math:`i` is the iteration of the Bayesian optimization loop. ``UCB`` corresponds to the Upper Confidence Bound acquisition function:
# 
# .. math::
#     
#     \alpha_\text{UCB}(\theta;\kappa) = \mu_\text{ET}(\theta) + \kappa \cdot \sigma_\text{ET}(\theta)
#
# where :math:`\mu_\text{ET}(\theta)` and :math:`\sigma_\text{ET}^2(\theta)` are respectively estimators of :math:`E_C[C|\Theta=\theta]` and :math:`V_C[C|\Theta=\theta]` with :math:`C` the random variable describing the objective (or cost) and :math:`\Theta` the random variable describing the hyperparameters alone.
#
# Finally the ``"d"`` postfix in ``"qUCBd"`` means that we will only consider the epistemic component of the uncertainty returned by the surrogate model.
# Thanks to the law of total variance we have the following decomposition:
#
# .. math::
# 
#     V_C[C|\Theta=\theta] = E_\text{tree}\left[V_C[C|\Theta=\theta;\text{tree}\right] + V_\text{tree}\left[E_C[C|\Theta=\theta;\text{tree}]\right]
# 
# Then, we define :math:`\sigma_{\text{ET},\text{ep}}(\theta)` as the empirical estimate of :math:`V_\text{tree}\left[E_C[C|\Theta=\theta;\text{tree}]\right]`.
# Then, we define :math:`\alpha_\text{qUCBd}(\theta;\kappa^j_i)` as:
#
# .. math::
#     
#     \alpha_\text{qUCBd}(\theta;\kappa^j_i) = \mu_\text{ET}(\theta) + \kappa^j_i \cdot \sigma_{\text{ET},\text{ep}}(\theta)
#
# Interestingly the same trick will be used later to decompose the uncertainty of the deep ensemble.
#
# - ``acq_optimizer_*```: are parameters related to optimization of the previously defined acquisition function.
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
    "acq_optimizer": "mixedga",  # Use continuous Genetic Algorithm for the acquisition function optimizer
    "acq_optimizer_freq": 1,  # Frequency of the acquisition function optimizer (1 = each new batch generation) increasing this value can help amortize the computational cost of acquisition function optimization
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
# Then, we create the search instance.
#
# For this we pass the hyperparameter ``problem``, the ``evaluator`` and also a ``stopper`` (optional).
#
# The ``problem`` is the instance of :class:`deephyper.hpo.HpProblem` that we defined in previous sections.
#
# The ``evaluator`` is a subclass of :class:`deephyper.evaluator.Evaluator` that provides a ``.submit(...)`` method and a ``.gather(...)`` method to
# submit and gather asynchronous evaluations.
# 
# The ``stopper`` is an optional parameter that allows to use an early-discarding (a.k.a., multi-fidelity) strategy to stop early low performing evaluations.
# In our case we will use the Asynchronous Successive Halving (ASHA) early-discarding strategy.
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO
from deephyper.stopper import SuccessiveHalvingStopper


hpo_dir = "nas_regression"


def run_neural_architecture_search(problem, max_evals):
    model_checkpoint_dir = os.path.join(hpo_dir, "models")
    pathlib.Path(model_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator.create(
        run,
        method="ray",  
        method_kwargs={
            "num_cpus": 8,
            "num_cpus_per_task": 1,
            "run_function_kwargs": {
                "model_checkpoint_dir": model_checkpoint_dir,
                "verbose": False,
            },
            "callbacks": [TqdmCallback()],
        },
    )

    stopper = SuccessiveHalvingStopper(min_steps=1, max_steps=max_n_epochs)
    search = CBO(problem, evaluator, log_dir=hpo_dir, stopper=stopper, **search_kwargs)

    results = search.search(max_evals=max_evals)

    return results

# %% 
# As the search can take some time to finalize we provide a mechanism that checks if results were already computed and skip the search if it is the case.
max_evals = 200

hpo_results = None
hpo_results_path = os.path.join(hpo_dir, "results.csv")
if os.path.exists(hpo_results_path):
    print("Reloading results...")
    hpo_results = pd.read_csv(hpo_results_path)

if hpo_results is None or len(hpo_results) < max_evals:
    print("Running neural architecture search...")
    hpo_results = run_neural_architecture_search(problem, max_evals)

# %%
# Analysis of the results
# -----------------------
# 
# We will now look at the results of the search globally in term of evolution of the objective and worker's activity.

from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo
from deephyper.analysis.hpo import plot_worker_utilization


fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(WIDTH_PLOTS, HEIGHT_PLOTS),
)

_ = plot_search_trajectory_single_objective_hpo(
    hpo_results,
    mode="min",
    x_units="seconds",
    ax=axes[0],
)
axes[0].set_yscale("log")

_ = plot_worker_utilization(
    hpo_results,
    profile_type="submit/gather",
    ax=axes[1],
)

plt.tight_layout()

# %%
# Then, we split results between successful and failed results if there are some.
from deephyper.analysis.hpo import filter_failed_objectives


hpo_results, hpo_results_failed = filter_failed_objectives(hpo_results)

hpo_results


# %%
# We look at the learning curves of the best model and observe improvements in both training and validation loss:

# .. dropdown: Make learning curves plot
x_values = np.arange(1, len(baseline_results["metadata"]["train_loss"]) + 1)
_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plt.plot(
    x_values,
    baseline_results["metadata"]["train_loss"],
    linestyle=":",
    label="Baseline Training",
)
_ = plt.plot(
    x_values,
    baseline_results["metadata"]["val_loss"],
    linestyle=":",
    label="Baseline Validation",
)

i_max = hpo_results["objective"].argmax()
train_loss = json.loads(hpo_results.iloc[i_max]["m:train_loss"])
val_loss = json.loads(hpo_results.iloc[i_max]["m:val_loss"])
x_values = np.arange(1, len(train_loss) + 1)
_ = plt.plot(
    x_values,
    train_loss,
    alpha=0.8,
    linestyle="--",
    label="Best Training",
)
_ = plt.plot(
    x_values,
    val_loss,
    alpha=0.8,
    linestyle="--",
    label="Best Validation",
)
_ = plt.xlim(x_values.min(), x_values.max())
_ = plt.grid(which="both", linestyle=":")
_ = plt.legend()
_ = plt.xlabel("Epochs")
_ = plt.ylabel("NLL")


# %%
# Finally, we look at predictions of this best model and observe that it manage to predict much better than the baseline one the right range. 
from deephyper.analysis.hpo import parameters_from_row


hpo_dir = "nas_regression"
model_checkpoint_dir = os.path.join(hpo_dir, "models")
job_id = hpo_results.iloc[i_max]["job_id"]
file_name = f"model_0.{job_id}.pt"

y_mu, y_std = np.mean(train_y), np.std(train_y)
inputs = train_X.shape[1]

weights_path = os.path.join(model_checkpoint_dir, file_name)
parameters = parameters_from_row(hpo_results.iloc[i_max])

num_layers = parameters["num_layers"]
torch_module = DeepNormalRegressor(
    inputs,
    layers=[
        (
            parameters[f"layer_{i}_units"],
            parameters[f"layer_{i}_activation"],
            parameters[f"layer_{i}_dropout_rate"],
        )
        for i in range(num_layers)
    ],
    loc=y_mu,
    scale=y_std,
)

torch_module.load_state_dict(torch.load(weights_path, weights_only=True))
torch_module.eval()

y_pred = torch_module.forward(torch.from_numpy(test_X).float())
y_pred_mean = y_pred.loc.detach().numpy()
y_pred_std = y_pred.scale.detach().numpy()

# %%

# .. dropdown:: Make prediction plot
kappa = 1.96
_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plt.scatter(train_X, train_y, s=5, label="Training")
_ = plt.scatter(valid_X, valid_y, s=5, label="Validation")
_ = plt.plot(test_X, test_y, linestyle="--", color="gray", label="Test")

_ = plt.plot(test_X, y_pred_mean, label=r"$\mu(x)$")
_ = plt.fill_between(
    test_X.reshape(-1),
    (y_pred_mean - kappa * y_pred_std).reshape(-1),
    (y_pred_mean + kappa * y_pred_std).reshape(-1),
    alpha=0.25,
    label=r"$\sigma_\text{al}(x)$",
)
_ = plt.fill_between([-30, -15], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.25)
_ = plt.fill_between([15, 30], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.25)
_ = plt.xlim(-x_lim, x_lim)
_ = plt.ylim(-y_lim, y_lim)
_ = plt.legend(ncols=2)
_ = plt.xlabel(r"$x$")
_ = plt.ylabel(r"$f(x)$")
_ = plt.grid(which="both", linestyle=":")

# %% 
# Deep ensemble
# -------------

from deephyper.ensemble import EnsemblePredictor
from deephyper.ensemble.aggregator import MixedNormalAggregator
from deephyper.ensemble.loss import NormalNegLogLikelihood
from deephyper.ensemble.selector import GreedySelector, TopKSelector
from deephyper.predictor.torch import TorchPredictor

class NormalTorchPredictor(TorchPredictor):
    def __init__(self, torch_module):
        super().__init__(torch_module)

    def pre_process_inputs(self, X):
        X = super().pre_process_inputs(X).float()
        return X

    def post_process_predictions(self, y):
        y = {
            "loc": y.loc.detach().numpy(),
            "scale": y.scale.detach().numpy(),
        }
        return y


# %%
hpo_dir = "nas_regression"
model_checkpoint_dir = os.path.join(hpo_dir, "models")

y_mu, y_std = np.mean(train_y), np.std(train_y)
inputs = train_X.shape[1]

y_predictors = []
job_id_predictors = []

for file_name in tqdm(os.listdir(model_checkpoint_dir)):
    if not file_name.endswith(".pt"):
        continue

    weights_path = os.path.join(model_checkpoint_dir, file_name)
    job_id = int(file_name[6:-3].split(".")[-1])

    row = hpo_results[hpo_results["job_id"] == job_id]
    if len(row) == 0:
        continue
    assert len(row) == 1
    row = row.iloc[0]
    parameters = parameters_from_row(row)

    num_layers = parameters["num_layers"]
    torch_module = DeepNormalRegressor(
        inputs,
        layers=[
            (
                parameters[f"layer_{i}_units"],
                parameters[f"layer_{i}_activation"],
                parameters[f"layer_{i}_dropout_rate"],
            )
            for i in range(num_layers)
        ],
        loc=y_mu,
        scale=y_std,
    )

    try:
        torch_module.load_state_dict(torch.load(weights_path, weights_only=True))
    except RuntimeError:
        continue

    torch_module.eval()

    predictor = NormalTorchPredictor(torch_module)
    y_pred = predictor.predict(valid_X)
    y_predictors.append(y_pred)
    job_id_predictors.append(job_id)

# %%
# Ensemble selection
# ------------------
k = 50

# Use TopK or Greedy/Caruana
# selector = TopKSelector(
#     loss_func=NormalNegLogLikelihood(),
#     k=k,
# )

selector = GreedySelector(
    loss_func=NormalNegLogLikelihood(),
    aggregator=MixedNormalAggregator(),
    k=k,
    max_it=100,
    k_init=3,
    early_stopping=False,
    with_replacement=True,
    bagging=True,
    verbose=True,
)

selected_predictors_indexes, selected_predictors_weights = selector.select(
    valid_y,
    y_predictors,
)

print(f"{selected_predictors_indexes=}")
print(f"{selected_predictors_weights=}")

selected_predictors_job_ids = np.array(job_id_predictors)[selected_predictors_indexes]
selected_predictors_job_ids

print(f"{selected_predictors_job_ids=}")

# %%
# Evaluation of the ensemble
# --------------------------
predictors = []

hpo_dir = "nas_regression"
model_checkpoint_dir = os.path.join(hpo_dir, "models")

y_mu, y_std = np.mean(train_y), np.std(train_y)
inputs = train_X.shape[1]

for job_id in selected_predictors_job_ids:
    file_name = f"model_0.{job_id}.pt"

    weights_path = os.path.join(model_checkpoint_dir, file_name)

    row = hpo_results[hpo_results["job_id"] == job_id].iloc[0]
    parameters = parameters_from_row(row)

    num_layers = parameters["num_layers"]
    torch_module = DeepNormalRegressor(
        inputs,
        layers=[
            (
                parameters[f"layer_{i}_units"],
                parameters[f"layer_{i}_activation"],
                parameters[f"layer_{i}_dropout_rate"],
            )
            for i in range(num_layers)
        ],
        loc=y_mu,
        scale=y_std,
    )

    torch_module.load_state_dict(torch.load(weights_path, weights_only=True))
    torch_module.eval()
    predictor = NormalTorchPredictor(torch_module)
    predictors.append(predictor)

ensemble = EnsemblePredictor(
    predictors=predictors,
    weights=selected_predictors_weights,
    aggregator=MixedNormalAggregator(decomposed_scale=True),
)

y_pred = ensemble.predict(test_X)

# %%
# Aleatoric Uncertainty
# ---------------------

# .. dropdown:: Make aleatoric uncertainty plot
kappa = 1.96
_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plt.scatter(train_X, train_y, s=5, label="Training")
_ = plt.scatter(valid_X, valid_y, s=5, label="Validation")
_ = plt.plot(test_X, test_y, linestyle="--", color="gray", label="Test")
_ = plt.plot(test_X, y_pred["loc"], label=r"$\mu(x)$")
_ = plt.fill_between(
    test_X.reshape(-1),
    (y_pred["loc"] - kappa * y_pred["scale_aleatoric"]).reshape(-1),
    (y_pred["loc"] + kappa * y_pred["scale_aleatoric"]).reshape(-1),
    alpha=0.25,
    label=r"$\sigma_\text{al}(x)$",
)
_ = plt.fill_between([-30, -15], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.25)
_ = plt.fill_between([15, 30], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.25)
_ = plt.xlim(-x_lim, x_lim)
_ = plt.ylim(-y_lim, y_lim)
_ = plt.legend(ncols=2)
_ = plt.xlabel(r"$x$")
_ = plt.ylabel(r"$f(x)$")
_ = plt.grid(which="both", linestyle=":")

# %%
# Epistemic uncertainty
# ---------------------

# .. dropdown:: Make epistemic uncertainty plot
kappa = 1.96
_ = plt.figure(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
_ = plt.scatter(train_X, train_y, s=5, label="Training")
_ = plt.scatter(valid_X, valid_y, s=5, label="Validation")
_ = plt.plot(test_X, test_y, linestyle="--", color="gray", label="Test")
_ = plt.plot(test_X, y_pred["loc"], label=r"$\mu(x)$")
_ = plt.fill_between(
    test_X.reshape(-1),
    (y_pred["loc"] - kappa * y_pred["scale_epistemic"]).reshape(-1),
    (y_pred["loc"] + kappa * y_pred["scale_epistemic"]).reshape(-1),
    alpha=0.25,
    color="red",
    label=r"$\sigma_\text{ep}(x)$",
)
_ = plt.fill_between([-30, -15], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.25)
_ = plt.fill_between([15, 30], [-y_lim, -y_lim], [y_lim, y_lim], color="gray", alpha=0.25)
_ = plt.xlim(-x_lim, x_lim)
_ = plt.ylim(-y_lim, y_lim)
_ = plt.legend(ncols=2)
_ = plt.xlabel(r"$x$")
_ = plt.ylabel(r"$f(x)$")
_ = plt.grid(which="both", linestyle=":")
