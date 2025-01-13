import sys
from functools import partial

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


LC_MODELS = [
    "lin2",
    "pow3",
    "mmf4",
    "vapor3",
    "logloglin2",
    "hill3",
    "logpow3",
    "pow4",
    "exp4",
    "janoschek4",
    "weibull4",
    "ilog2",
    "arctan3",
]


# Learning curves models
@jax.jit
def f_lin2(z, rho):
    return rho[1] * z + rho[0]


@jax.jit
def f_pow3(z, rho):
    return rho[0] - rho[1] * z ** rho[2]


@jax.jit
def f_mmf4(z, rho):
    return (rho[0] * rho[1] + rho[2] * jnp.power(z, rho[3])) / (rho[1] + jnp.power(z, rho[3]))


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
    """The learning curve model is assumed to be modeled by 'f' with interface f(z, rho)."""
    random_state = check_random_state(random_state)

    results = []
    mse_hist = []

    rho_init = np.zeros((f_model_nparams,))

    for i in range(max_trials_ls_fit):
        if verbose:
            print(f"Least-Square fit - trial {i + 1}/{max_trials_ls_fit}: ", end="")

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
    y_sigma = numpyro.sample("sigma", dist.Exponential(y_sigma_prior))  # introducing noise
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
        f_model (callable, optional):
            The model function to use. Defaults to `f_power3` for a Power-Law with 3 parameters.
        f_model_nparams (int, optional):
            The number of parameters of the model. Defaults to `3`.
        max_trials_ls_fit (int, optional):
            The number of least-square fits that should be tried. Defaults to `10`.
        mcmc_kernel (str, optional):
            The MCMC kernel to use. It should be a string in the following
            list: `["NUTS", "BarkerMH"]`. Defaults to `"NUTS"`.
        mcmc_num_warmup (int, optional):
            The number of warmup steps in MCMC. Defaults to `200`.
        mcmc_num_samples (int, optional):
            The number of samples in MCMC. Defaults to `1_000`.
        random_state (int, optional):
            A random state. Defaults to `None`.
        verbose (int, optional):
            Wether or not to use the verbose mode. Defaults to `0` to deactive it.
        batch_size (int, optional):
            The expected maximum length of the X, y arrays (used in the `fit
            (X, y)` method) in order to preallocate memory and compile the
            code only once. Defaults to `100`.
        min_max_scaling (bool, optional):
            Wether or not to use min-max scaling in [0,1] for `y` values. Defaults to False.
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
        """Fit the model.

        Args:
            X (np.ndarray):
                A 1-D array of inputs.
            y (_type_):
                A 1-D array of targets.
            update_prior (bool, optional):
                A boolean indicating if the prior distribution should be
                updated using least-squares before running the Bayesian
                inference. Defaults to ``True``.

        Raises:
            ValueError: if input arguments are invalid.
        """
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
        """Predict the mean and standard deviation of the model.

        Args:
            X (np.ndarray):
                A 1-D array of inputs.
            return_std (bool, optional):
                A boolean indicating if the standard-deviation representing
                uncertainty in the prediction should be returned. Defaults to
                ``True``.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The mean prediction with shape ``(len(X),)`` and the standard
                deviation with shape ``(len(X),)`` if ``return_std`` is
                ``True``.
        """
        posterior_samples = self.predict_posterior_samples(X)

        mean_mu = jnp.mean(posterior_samples, axis=0)

        if return_std:
            std_mu = jnp.std(posterior_samples, axis=0)
            return mean_mu, std_mu

        return mean_mu

    def predict_posterior_samples(self, X):
        """Predict the posterior samples of the model.

        Args:
            X (np.ndarray): a 1-D array of inputs.

        Returns:
            np.ndarray:
                A 2-D array of shape (n_samples, len(X)) where n_samples is
                the number of samples and len(X) is the length of the input
                array.
        """
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
        """Compute the approximate probability of P(cond(m(X_i), y_i)).

        Where m is the current fitted model and cond a condition.

        Args:
            X (np.array): An array of inputs.
            condition (callable): A function defining the condition to test.

        Returns:
            array: an array of shape X.
        """
        posterior_mu = self.predict_posterior_samples(X)

        prob = jnp.mean(condition(posterior_mu), axis=0)

        return prob

    @staticmethod
    def get_parametrics_model_func(name):
        """Return the function of the learning curve model given its name.

        Should be one of ``
        ["lin2", "pow3", "mmf4", "vapor3", "logloglin2", "hill3", "logpow3", "pow4",
        "exp4", "janoschek4", "weibull4", "ilog2", "arctan3"]`` where the
        integer suffix indicates the number of parameters of the model.

        Args:
            name (str): The name of the learning curve model.

        Returns:
            callable:
                A function with signature ``f(x, rho)`` of the learning curve
                model where ``x`` is a possible input of the model and
                ``rho`` is a 1-D array for the parameters of the model with
                length equal to the number of parameters of the model
                (e.g., it is of length ``3`` for ``"pow3"``).
        """
        return getattr(sys.modules[__name__], f"f_{name}")
