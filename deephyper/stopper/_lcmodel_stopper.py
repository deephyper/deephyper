import sys
from functools import partial
from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, BarkerMH
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from deephyper.stopper._stopper import Stopper


# Learning curves models
@jax.jit
def f_lin2(z, rho):
    return rho[1] * z + rho[0]


@jax.jit
def f_pow3(z, rho):
    return rho[0] - rho[1] * z ** rho[2]


@jax.jit
def f_mmf4(z, rho):
    return (rho[0] * rho[1] + rho[2] * jnp.power(z, rho[3])) / (
        rho[1] + jnp.power(z, rho[3])
    )


@jax.jit
def f_vapor3(z, rho):
    return rho[0] + rho[1] / z + rho[2] * np.log(z)


@jax.jit
def f_logloglin2(z, rho):
    return jnp.log(rho[0] * jnp.log(z) + rho[1])


@jax.jit
def f_hill3(z, rho):
    ymax, eta, kappa = rho
    return ymax * (z**eta) / (kappa * eta + z**eta)


@jax.jit
def f_logpow3(z, rho):
    return rho[0] / (1 + (z / jnp.exp(rho[1])) ** rho[2])


@jax.jit
def f_pow4(z, rho):
    return rho[2] - (rho[0] * z + rho[1]) ** (-rho[3])


@jax.jit
def f_exp4(z, rho):
    return rho[2] - jnp.exp(-rho[0] * (z ** rho[3]) + rho[1])


@jax.jit
def f_janoschek4(z, rho):
    return rho[0] - (rho[0] - rho[1]) * jnp.exp(-rho[2] * (z ** rho[3]))


@jax.jit
def f_weibull4(z, rho):
    return rho[0] - (rho[0] - rho[1]) * jnp.exp(-((rho[2] * z) ** rho[3]))


@jax.jit
def f_ilog2(z, rho):
    return rho[1] - (rho[0] / jnp.log(z + 1))


@jax.jit
def f_arctan3(z, rho):
    return 2 / jnp.pi * jnp.arctan(rho[0] * jnp.pi / 2 * z + rho[1]) - rho[2]


# Utility to estimate parameters of learning curve model
# The combination of "partial" and "static_argnums" is necessary
# with the "f" lambda function passed as argument
@partial(jax.jit, static_argnums=(1,))
def residual_least_square(rho, f, z, y):
    """Residual for least squares."""
    y_pred = f(z, rho)
    y_pred = jnp.where(y_pred == 0.0, y_pred, 0.0)
    return y_pred - y


@partial(jax.jit, static_argnums=(1,))
def jac_residual_least_square(rho, f, z, y):
    """Jacobian of the residual for least squares."""
    return jax.jacfwd(residual_least_square, argnums=0)(rho, f, z, y)


def fit_learning_curve_model_least_square(
    z_train,
    y_train,
    f_model,
    f_model_nparams,
    max_trials_ls_fit=10,
    random_state=None,
    verbose=0,
):
    """The learning curve model is assumed to be modeled by 'f' with
    interface f(z, rho).
    """

    random_state = check_random_state(random_state)

    results = []
    mse_hist = []

    rho_init = np.zeros((f_model_nparams,))

    for i in range(max_trials_ls_fit):
        if verbose:
            print(f"Least-Square fit - trial {i+1}/{max_trials_ls_fit}: ", end="")

        rho_init[:] = random_state.randn(f_model_nparams)[:]

        try:
            res_lsq = least_squares(
                residual_least_square,
                rho_init,
                args=(f_model, z_train, y_train),
                method="lm" if len(z_train) >= f_model_nparams else "trf",
                jac=jac_residual_least_square,
            )
        except ValueError:
            continue

        mse_res_lsq = np.mean(res_lsq.fun**2)
        mse_hist.append(mse_res_lsq)
        results.append(res_lsq.x)

        if verbose:
            print(f"mse={mse_res_lsq:.3f}")

        if mse_res_lsq < 1e-8:
            break

    i_best = np.nanargmin(mse_hist)
    res = results[i_best]
    return res


def prob_model(
    z,
    y,
    f=None,
    rho_mu_prior=None,
    rho_sigma_prior=1.0,
    y_sigma_prior=1.0,
    num_obs=None,
):
    rho = numpyro.sample("rho", dist.Normal(rho_mu_prior, rho_sigma_prior))
    y_sigma = numpyro.sample(
        "sigma", dist.Exponential(y_sigma_prior)
    )  # introducing noise
    y_mu = f(z[:num_obs], rho)
    numpyro.sample("obs", dist.Normal(y_mu, y_sigma), obs=y[:num_obs])


@partial(jax.jit, static_argnums=(0,))
def predict_moments_from_posterior(f, X, posterior_samples):
    vf_model = jax.vmap(f, in_axes=(None, 0))
    posterior_mu = vf_model(X, posterior_samples)
    mean_mu = jnp.mean(posterior_mu, axis=0)
    std_mu = jnp.std(posterior_mu, axis=0)
    return mean_mu, std_mu


class BayesianLearningCurveRegressor(BaseEstimator, RegressorMixin):
    """Probabilistic model for learning curve regression.

    Args:
        f_model (callable, optional): The model function to use. Defaults to `f_power3` for a Power-Law with 3 parameters.
        f_model_nparams (int, optional): The number of parameters of the model. Defaults to `3`.
        max_trials_ls_fit (int, optional): The number of least-square fits that should be tried. Defaults to `10`.
        mcmc_kernel (str, optional): The MCMC kernel to use. It should be a string in the following list: `["NUTS", "BarkerMH"]`. Defaults to `"NUTS"`.
        mcmc_num_warmup (int, optional): The number of warmup steps in MCMC. Defaults to `200`.
        mcmc_num_samples (int, optional): The number of samples in MCMC. Defaults to `1_000`.
        random_state (int, optional): A random state. Defaults to `None`.
        verbose (int, optional): Wether or not to use the verbose mode. Defaults to `0` to deactive it.
        batch_size (int, optional): The expected maximum length of the X, y arrays (used in the `fit(X, y)` method) in order to preallocate memory and compile the code only once. Defaults to `100`.
        min_max_scaling (bool, optional): Wether or not to use min-max scaling in [0,1] for `y` values. Defaults to False.
    """

    def __init__(
        self,
        f_model=f_pow3,
        f_model_nparams=3,
        max_trials_ls_fit=10,
        mcmc_kernel="NUTS",
        mcmc_num_chains=1,
        mcmc_num_warmup=200,
        mcmc_num_samples=1_000,
        random_state=None,
        verbose=0,
        batch_size=1_000,
        min_max_scaling=False,
    ):
        self.f_model = f_model
        self.f_model_nparams = f_model_nparams
        self.mcmc_kernel = mcmc_kernel
        self.mcmc_num_chains = mcmc_num_chains
        self.mcmc_num_warmup = mcmc_num_warmup
        self.mcmc_num_samples = mcmc_num_samples
        self.max_trials_ls_fit = max_trials_ls_fit
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.rho_mu_prior_ = np.zeros((self.f_model_nparams,))

        self.batch_size = batch_size
        self.X_ = np.zeros((self.batch_size,))
        self.y_ = np.zeros((self.batch_size,))

        self.min_max_scaling = min_max_scaling

    def fit(self, X, y, update_prior=True):
        check_X_y(X, y, ensure_2d=False)

        # !Trick for performance to avoid performign JIT again and again
        # !This will fix the shape of inputs of the model for numpyro
        # !see https://github.com/pyro-ppl/numpyro/issues/441
        num_samples = len(X)
        assert num_samples <= self.batch_size
        self.X_[:num_samples] = X[:]
        self.y_[:num_samples] = y[:]
        self.X_[num_samples:] = 0.0
        self.y_[num_samples:] = 0.0

        # Min-Max Scaling
        if not (self.min_max_scaling):
            self.y_min_ = 0
            self.y_max_ = 1
        else:
            self.y_min_ = self.y_[:num_samples].min()
            self.y_max_ = self.y_[:num_samples].max()
            if abs(self.y_min_ - self.y_max_) <= 1e-8:  # avoid division by zero
                self.y_max_ = self.y_min_ + 1
            self.y_[:num_samples] = (self.y_[:num_samples] - self.y_min_) / (
                self.y_max_ - self.y_min_
            )

        if update_prior:
            self.rho_mu_prior_[:] = fit_learning_curve_model_least_square(
                self.X_,
                self.y_,
                f_model=self.f_model,
                f_model_nparams=self.f_model_nparams,
                max_trials_ls_fit=self.max_trials_ls_fit,
                random_state=self.random_state,
                verbose=self.verbose,
            )[:]

            if self.verbose:
                print(f"rho_mu_prior: {self.rho_mu_prior_}")

        if not (hasattr(self, "kernel_")):
            target_accept_prob = 0.8
            step_size = 0.05
            if self.mcmc_kernel == "NUTS":
                self.kernel_ = NUTS(
                    model=lambda z, y: prob_model(
                        z,
                        y,
                        f=self.f_model,
                        rho_mu_prior=self.rho_mu_prior_,
                        num_obs=num_samples,
                    ),
                    target_accept_prob=target_accept_prob,
                    step_size=step_size,
                )
            elif self.mcmc_kernel == "BarkerMH":
                self.kernel_ = BarkerMH(
                    model=lambda z, y: prob_model(
                        z,
                        y,
                        f=self.f_model,
                        rho_mu_prior=self.rho_mu_prior_,
                        num_obs=num_samples,
                    ),
                    target_accept_prob=target_accept_prob,
                    step_size=step_size,
                )
            else:
                raise ValueError(f"Unknown MCMC kernel: {self.mcmc_kernel}")

            self.mcmc_ = MCMC(
                self.kernel_,
                num_chains=self.mcmc_num_chains,
                num_warmup=self.mcmc_num_warmup,
                num_samples=self.mcmc_num_samples,
                progress_bar=self.verbose,
            )

        seed = self.random_state.randint(low=0, high=2**31)
        rng_key = jax.random.PRNGKey(seed)
        self.mcmc_.run(rng_key, z=self.X_, y=self.y_)

        if self.verbose:
            self.mcmc_.print_summary()

        return self

    def predict(self, X, return_std=True):
        posterior_samples = self.predict_posterior_samples(X)

        mean_mu = jnp.mean(posterior_samples, axis=0)

        if return_std:
            std_mu = jnp.std(posterior_samples, axis=0)
            return mean_mu, std_mu

        return mean_mu

    def predict_posterior_samples(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X, ensure_2d=False)

        posterior_samples = self.mcmc_.get_samples()
        vf_model = jax.vmap(self.f_model, in_axes=(None, 0))
        posterior_mu = vf_model(X, posterior_samples["rho"])

        # Inverse Min-Max Scaling
        posterior_mu = posterior_mu * (self.y_max_ - self.y_min_) + self.y_min_

        return posterior_mu

    def prob(self, X, condition):
        """Compute the approximate probability of P(cond(m(X_i), y_i))
        where m is the current fitted model and cond a condition.

        Args:
            X (np.array): An array of inputs.
            condition (callable): A function defining the condition to test.

        Returns:
            array: an array of shape X.
        """
        posterior_mu = self.predict_posterior_samples(X)

        prob = jnp.mean(condition(posterior_mu), axis=0)

        return prob


def area_learning_curve(z, f, z_max) -> float:
    assert len(z) == len(f)
    assert z[-1] <= z_max
    area = 0
    for i in range(1, len(z)):
        # z: is always monotinic increasing but not f!
        area += (z[i] - z[i - 1]) * f[i - 1]
    if z[-1] < z_max:
        area += (z_max - z[-1]) * f[-1]
    return area


class LCModelStopper(Stopper):
    """Stopper based on learning curve extrapolation (LCE) to evaluate if the iterations of the learning algorithm
    should be stopped.

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1

        * - Single-Objective
          - Multi-Objectives
          - Failures
        * - ✅
          - ❌
          - ❌

    The LCE is based on a parametric learning curve model (LCM) which is modeling the score as a function of the number of training steps. Training steps can correspond to the number of training epochs, the number of training batches, the number of observed samples or any other quantity that is iterated through during the training process. The LCE is based on the following steps:

    1. An early stopping condition is always checked first. If the early stopping condition is met, the LCE is not applied.
    2. Then, some safeguard conditions are checked to ensure that the LCE can be applied (number of observed steps must be greater or equal to the number of parameters of the LCM).
    3. If the LCM cannot be fitted (number of observed steps is less than number of parameters of the model), then the last observed step is compared to hitorical performance of others at the same step to check if it is a low-performing outlier (outlier in the direction of performing worse!) using the IQR criterion.
    4. If the LCM can be fitted, a least square fit is performed to estimate the parameters of the LCM.
    5. The probability of the current LC to perform worse than the best observed score at the maximum iteration is computed using Monte-Carlo Markov Chain (MCMC).

    To use this stopper, you need to install the following dependencies:

    .. code-block:: bash

        $ jax>=0.3.25
        $ numpyro

    Args:
        max_steps (int): The maximum number of training steps which can be performed.
        min_steps (int, optional): The minimum number of training steps which can be performed. Defaults to ``4``. It is better to have at least as many steps as the number of parameters of the fitted learning curve model. For example, if ``lc_model="mmf4"`` then ``min_steps`` should be at least ``4``.
        lc_model (str, optional): The parameteric learning model to use. It should be a string in the following list: ``["lin2", "loglin2", "loglin3", "loglin4", "pow3","mmf4", "vapor3", "logloglin2", "hill3", "logpow3", "pow4", "exp4", "janoschek4", "weibull4", "ilog2"]``. The number in the name corresponds to the number of parameters of the parametric model. Defaults to ``"mmf4"``.
        min_done_for_outlier_detection (int, optional): The minimum number of observed scores at the same step to check for if it is a lower-bound outlier. Defaults to ``10``.
        iqr_factor_for_outlier_detection (float, optional): The IQR factor for outlier detection. The higher it is the more inclusive the condition will be (i.e. if set very large it is likely not going to detect any outliers). Defaults to ``1.5``.
        prob_promotion (float, optional): The threshold probabily to stop the iterations. If the current learning curve has a probability greater than ``prob_promotion`` to be worse that the best observed score accross all evaluations then the current iterations are stopped. Defaults to ``0.9`` (i.e. probability of 0.9 of being worse).
        early_stopping_patience (float, optional): The patience of the early stopping condition. If it is an ``int`` it is directly corresponding to a number of iterations. If it is a ``float`` then it is corresponding to a proportion between [0,1] w.r.t. ``max_steps``. Defaults to ``0.25`` (i.e. 25% of ``max_steps``).
        objective_returned (str, optional): The returned objective. It can be a value in ``["last", "max", "alc"]`` where ``"last"`` corresponds to the last observed score, ``"max"`` corresponds to the maximum observed score and ``"alc"`` corresponds to the area under the learning curve. Defaults to "last".
        random_state (int or np.RandomState, optional): The random state of estimation process. Defaults to ``None``.

    Raises:
        ValueError: parameters are not valid.
    """

    def __init__(
        self,
        max_steps: int,
        min_steps: int = 1,
        lc_model="mmf4",
        min_obs_to_fit_lc_model=4,
        min_done_for_outlier_detection=10,
        iqr_factor_for_outlier_detection=1.5,
        prob_promotion=0.9,
        early_stopping_patience=0.25,
        reduction_factor=3,
        objective_returned="last",
        random_state=None,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.min_steps = min_steps

        lc_model = "f_" + lc_model
        self._f_model = getattr(sys.modules[__name__], lc_model)
        self._f_model_nparams = int(lc_model[-1])
        self._min_obs_to_fit_lc_model = min_obs_to_fit_lc_model
        self._reduction_factor = reduction_factor

        self.min_done_for_outlier_detection = min_done_for_outlier_detection
        self.iqr_factor_for_outlier_detection = iqr_factor_for_outlier_detection

        self.prob_promotion = prob_promotion
        if type(early_stopping_patience) is int:
            self.early_stopping_patience = early_stopping_patience
        elif type(early_stopping_patience) is float:
            self.early_stopping_patience = int(early_stopping_patience * self.max_steps)
        else:
            raise ValueError("early_stopping_patience must be int or float")
        self.objective_returned = objective_returned

        self._rung = 0

        self._random_state = random_state
        self._batch_size = 100
        self.lc_model = None

        self._lc_objectives = []

    def _refresh_lc_model(self):
        batch_has_increased = False
        if self._batch_size < len(self.observed_budgets):
            self._batch_size += 100
            batch_has_increased = True

        if self.lc_model is None or batch_has_increased:
            self.lc_model = BayesianLearningCurveRegressor(
                f_model=self._f_model,
                f_model_nparams=self._f_model_nparams,
                random_state=self._random_state,
                batch_size=self._batch_size,
            )

    def _compute_halting_step(self):
        return (self.min_steps - 1) * self._reduction_factor**self._rung

    def _retrieve_best_objective(self) -> float:
        search_id, _ = self.job.id.split(".")
        objectives = []

        for obj in self.job.storage.load_out_from_all_jobs(search_id):
            if isinstance(obj, Number):
                objectives.append(obj)

        if len(objectives) > 0:
            return np.max(objectives)
        else:
            return np.max(self.observations[1])

    def _get_competiting_objectives(self, rung) -> list:
        search_id, _ = self.job.id.split(".")
        values = self.job.storage.load_metadata_from_all_jobs(
            search_id, f"_completed_rung_{rung}"
        )
        # Filter out non numerical values (e.g., "F" for failed jobs)
        values = [v for v in values if isinstance(v, Number)]
        return values

    def observe(self, budget: float, objective: float):
        super().observe(budget, objective)
        self._budget = self.observed_budgets[-1]
        self._lc_objectives.append(self.objective)
        self._objective = self._lc_objectives[-1]

        # For Early-Stopping based on Patience
        if (
            not (hasattr(self, "_local_best_objective"))
            or self._objective > self._local_best_objective
        ):
            self._local_best_objective = self._objective
            self._local_best_step = self.step

        halting_step = self._compute_halting_step()
        if self._budget >= halting_step:
            self.job.storage.store_job_metadata(
                self.job.id, f"_completed_rung_{self._rung}", self._objective
            )

    def stop(self) -> bool:
        # Enforce Pre-conditions Before Learning-Curve based Early Discarding
        if super().stop():
            self.infos_stopped = "max steps reached"
            return True

        if self.step - self._local_best_step >= self.early_stopping_patience:
            self.infos_stopped = "early stopping"
            return True

        # This condition will enforce the stopper to stop the evaluation at the first step
        # for the first evaluation (The FABOLAS method does the same, bias the first samples with
        # small budgets)
        self.best_objective = self._retrieve_best_objective()

        halting_step = self._compute_halting_step()

        if self.step < self.min_steps:
            if self.step >= halting_step:
                self._rung += 1
            return False

        if self.step < self._min_obs_to_fit_lc_model:
            if self.step >= halting_step:
                competing_objectives = self._get_competiting_objectives(self._rung)
                if len(competing_objectives) > self.min_done_for_outlier_detection:
                    q1 = np.quantile(
                        competing_objectives,
                        q=0.25,
                    )
                    q3 = np.quantile(
                        competing_objectives,
                        q=0.75,
                    )
                    iqr = q3 - q1
                    # lower than the minimum of a box plot
                    if (
                        self._objective
                        < q1 - self.iqr_factor_for_outlier_detection * iqr
                    ):
                        self.infos_stopped = "outlier"
                        return True
                self._rung += 1

            return False

        # Check if the halting budget condition is met
        if self.step < halting_step:
            return False

        # Check if the evaluation should be stopped based on LC-Model

        # Fit and predict the performance of the learning curve model
        self._refresh_lc_model()

        z_train = self.observed_budgets
        y_train = self._lc_objectives
        z_train, y_train = np.asarray(z_train), np.asarray(y_train)
        self.lc_model.fit(z_train, y_train, update_prior=True)

        # Check if the configuration is promotable based on its predicted objective value
        p = self.lc_model.prob(
            X=[self.max_steps], condition=lambda y_hat: y_hat <= self.best_objective
        )[0]

        # Return whether the configuration should be stopped
        if p <= self.prob_promotion:
            self._rung += 1
        else:
            self.infos_stopped = f"prob={p:.3f}"

            return True

    @property
    def objective(self):
        if self.objective_returned == "last":
            return self.observations[-1][-1]
        elif self.objective_returned == "max":
            return max(self.observations[-1])
        elif self.objective_returned == "alc":
            z, y = self.observations
            return area_learning_curve(z, y, z_max=self.max_steps)
        else:
            raise ValueError("objective_returned must be one of 'last', 'best', 'alc'")


def test_bayesian_lce_model():
    import cProfile
    from pstats import SortKey

    import time
    import matplotlib.pyplot as plt
    import numpy as np

    def f(z):
        return f_pow3(z, [1, -1, 0.125])

    z = np.arange(1, 1000)

    y = f(z)
    # y = y + rng.normal(0, 0.01, size=y.shape)

    t_start = time.time()
    with cProfile.Profile() as pr:
        for r in range(20):
            model = BayesianLearningCurveRegressor(batch_size=100, verbose=0)
            for i in range(1, 20):
                model.fit(z[:i], y[:i])
                y_pred, y_std = model.predict(z)
                y_min, y_max = y_pred - y_std, y_pred + y_std

        pr.print_stats(SortKey.TIME)

    t_end = time.time()
    duration = t_end - t_start

    print(f"duration: {duration:.3f} sec")

    plt.figure()
    plt.plot(z, y, label="f_pow3")
    plt.plot(z, y_pred, label="$\\hat{y}$", color="C2")
    plt.fill_between(z, y_min, y_max, color="C2", alpha=0.5)
    plt.legend()
    plt.show()


# if __name__ == "__main__":
#     test_bayesian_lce_model()
#     rho = np.ones((3,))
#     z = np.arange(1000)
#     y = np.zeros((1000,))

#     out = jac_residual_least_square(rho, f_pow3, z, y)
#     out = jac_residual_least_square(rho, f_pow3, z, y)
#     out = jac_residual_least_square(rho, f_pow3, z, y)
