import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from joblib import Parallel, delayed

from deephyper.stopper._stopper import Stopper


# Budget allocation models
def b_lin2(z, nu=[1, 1]):
    return nu[1] * (z - 1) + nu[0]


def b_exp2(z, nu=[1, 2]):
    return nu[0] * jnp.power(nu[1], z - 1)


# Learning curves models
def f_lin2(z, b, rho):
    return rho[1] * b(z) + rho[0]


def f_loglin2(z, b, rho):
    Z = jnp.log(z)
    Y = rho[1] * Z + rho[0]
    y = jnp.exp(Y)
    return -y  # !maximization


def f_loglin3(z, b, rho):
    Z = jnp.log(z)
    Y = rho[2] * jnp.power(Z, 2) + rho[1] * Z + rho[0]
    y = jnp.exp(Y)
    return -y  # !maximization


def f_loglin4(z, b, rho):
    Z = jnp.log(z)
    Y = rho[3] * jnp.power(Z, 3) + rho[2] * jnp.power(Z, 2) + rho[1] * Z + rho[0]
    y = jnp.exp(Y)
    return -y  # !maximization


def f_pow3(z, b, rho):
    return rho[0] - rho[1] * b(z) ** -rho[2]


def f_mmf4(z, b, rho):
    return (rho[0] * rho[1] + rho[2] * jnp.power(b(z), rho[3])) / (
        rho[1] + jnp.power(b(z), rho[3])
    )


# Utility to estimate parameters of learning curve model
# The combination of "partial" and "static_argnums" is necessary
# with the "f" lambda function passed as argument
@partial(jax.jit, static_argnums=(1,))
def residual_least_square(rho, f, z, y):
    """Residual for least squares."""
    return f(z, rho) - y


def fit_learning_curve_model_least_square(
    f, nparams, z_train, y_train, use_jac=True, max_trials=100, random_state=None
):
    """The learning curve model is assumed to be modeled by 'f' with
    interface f(z, rho).
    """

    random_state = check_random_state(random_state)

    z_train = np.asarray(z_train)
    y_train = np.asarray(y_train)

    # compute the jacobian
    # using the true jacobian is important to avoid problems
    # with numerical errors and approximations! indeed the scale matters
    # a lot when approximating with finite differences
    def fun_wrapper(rho, f, z, y):
        return np.array(residual_least_square(rho, f, z, y))

    jac_residual = partial(jax.jit, static_argnums=(1,))(
        jax.jacfwd(residual_least_square, argnums=0)
    )

    def jac_wrapper(rho, f, z, y):
        return np.array(jac_residual(rho, f, z, y))

    results = []
    mse_hist = []

    for _ in range(max_trials):

        rho_init = random_state.randn(nparams)

        try:
            res_lsq = least_squares(
                fun_wrapper,
                rho_init,
                args=(f, z_train, y_train),
                method="lm",
                jac=jac_wrapper if use_jac else "2-point",
            )
        except ValueError:
            continue

        mse_res_lsq = np.mean(res_lsq.fun**2)
        mse_hist.append(mse_res_lsq)
        results.append(res_lsq.x)

    i_best = np.nanargmin(mse_hist)
    res = results[i_best]
    return res


def fit_ensemble_members(
    f,
    nparams,
    z_train,
    y_train,
    ensemble_size=10,
    max_trials_per_fit=5,
    n_jobs=-1,
    random_state=None,
):

    random_state = check_random_state(random_state)
    random_states = random_state.randint(low=0, high=2**32, size=ensemble_size)

    def f_wrapper(seed):
        return fit_learning_curve_model_least_square(
            f,
            nparams,
            z_train,
            y_train,
            max_trials=max_trials_per_fit,
            random_state=seed,
        )

    rho_hat_list = Parallel(n_jobs=n_jobs)(
        delayed(f_wrapper)(seed) for seed in random_states
    )
    return np.array(rho_hat_list)


def predict_ensemble_members(f, z, rho, return_std=True, n_jobs=-1):
    def f_wrapper(rho):
        return f(z, rho)

    y_list = Parallel(n_jobs=n_jobs)(delayed(f_wrapper)(rho_i) for rho_i in rho)
    y_list = np.array(y_list)

    mean_y_last = np.nanmean(y_list[:, -1])
    std_y_last = np.nanstd(y_list[:, -1])

    # remove outliers in case some least-squares estimations were not stable
    kappa = 3
    upper_bound = mean_y_last + kappa * std_y_last
    lower_bound = mean_y_last - kappa * std_y_last
    selection = (lower_bound < y_list[:, -1]) & (y_list[:, -1] < upper_bound)

    mean_y = np.nanmean(y_list[selection], axis=0)
    if return_std:
        std_y = np.nanstd(y_list[selection], axis=0)
        return mean_y, std_y
    return mean_y


class LearningCurveRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        f_model=f_mmf4,
        f_model_num_params=4,
        b_model=b_lin2,
        ensemble_size=15,
        max_trials_per_member_fit=5,
        n_jobs=-1,
        random_state=None,
    ):
        self.b_model = b_model
        self.f_model = lambda z, rho: f_model(z, self.b_model, rho)
        self.f_nparams = f_model_num_params
        self.ensemble_size = ensemble_size
        self.max_trials_per_member_fit = max_trials_per_member_fit
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)

    def fit(self, X, y):

        check_X_y(X, y, ensure_2d=False)

        self.rho_hat_ = fit_ensemble_members(
            self.f_model,
            self.f_nparams,
            X,
            y,
            ensemble_size=self.ensemble_size,
            max_trials_per_fit=self.max_trials_per_member_fit,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        assert len(self.rho_hat_) == self.ensemble_size

        return self

    def predict(self, X, return_std=True):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X, ensure_2d=False)

        pred = predict_ensemble_members(
            self.f_model, X, self.rho_hat_, return_std=return_std, n_jobs=self.n_jobs
        )

        return pred


def u_log_from_curve(b, y, z_max, delta=1e-5):
    assert len(b) == len(y)
    assert len(b) > 1

    values = [0]
    for i in range(1, len(b)):
        z = i + 1
        delta_yi = y[i] - y[i - 1]
        delta_bi = b[i] - b[i - 1]
        value_i = (z / z_max) * (np.log(delta_yi) - np.log(delta_bi) - np.log(delta))
        values.append(value_i)

    return values


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
    """Stopper based on a learning curve model."""

    def __init__(
        self,
        max_steps: int,
        min_steps: int = 1,
        lc_model="loglin3",
        maximize_utility=False,
        kappa=1.96,
        delta=1e-4,
        random_state=None,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.min_steps = min_steps

        lc_model = "f_" + lc_model
        lc_model_num_params = int(lc_model[-1])
        lc_model = getattr(sys.modules[__name__], lc_model)
        self.lc_model = LearningCurveRegressor(
            f_model=lc_model,
            f_model_num_params=lc_model_num_params,
            ensemble_size=20,
            max_trials_per_member_fit=10,
            random_state=random_state,
        )
        self.kappa = kappa
        self.delta = delta

        self.maximize_utility = maximize_utility
        self.min_obs_to_fit = lc_model_num_params

        self._rung = 0

        # compute the step at which to stop based on steps allocation policy
        max_rung = np.floor(
            np.log(self.max_steps / self.min_steps) / np.log(self.min_obs_to_fit)
        )
        self.max_steps_ = self.min_steps * self.min_obs_to_fit**max_rung
        self._step_max_utility = self.max_steps_

    def _compute_halting_step(self):
        return self.min_steps * self.min_obs_to_fit**self._rung

    def _fit_and_predict_lc_model_performance(self):
        """Estimate the LC Model and Predict the performance at b.

        Returns:
            (step, objective): a tuple of (step, objective) at which the estimation was made.
        """

        # By default (no utility used) predict at the last possible step.
        z_pred = self.max_steps
        z_opt = self.max_steps_

        z_train, y_train = self.observations
        z_train, y_train = np.asarray(z_train), np.asarray(y_train)

        self.lc_model.fit(z_train, y_train)

        if self.maximize_utility:
            z_range = np.arange(self.min_steps, self.max_steps + 1)
            mean_pred, std_pred = self.lc_model.predict(z_range)
            ucb_pred = mean_pred[-1] + self.kappa * std_pred[-1]
            scores = u_log_from_curve(
                z_range, mean_pred, z_max=self.max_steps, delta=self.delta
            )
            idx_max = np.nanargmax(scores)
            z_opt = z_range[idx_max]
            self._step_max_utility = z_pred
        else:
            mean_pred, std_pred = self.lc_model.predict([z_pred])
            ucb_pred = mean_pred[0] + self.kappa * std_pred[0]

        return z_opt, ucb_pred

    def _retrieve_best_objective(self) -> float:
        search_id, _ = self.job.id.split(".")
        objectives = []
        for obj in self.job.storage.load_out_from_all_jobs(search_id):
            try:
                objectives.append(float(obj))
            except ValueError:
                pass
        if len(objectives) > 0:
            return np.max(objectives)
        else:
            return None

    def _get_competiting_objectives(self, rung) -> list:
        search_id, _ = self.job.id.split(".")
        values = self.job.storage.load_metadata_from_all_jobs(
            search_id, f"completed_rung_{rung}"
        )
        values = [float(v) for v in values]
        return values

    def observe(self, budget: float, objective: float):
        super().observe(budget, objective)
        self._budget = self.observed_budgets[-1]
        self._objective = self.observed_objectives[-1]

        halting_step = self._compute_halting_step()
        if self._budget >= halting_step:
            self.job.storage.store_job_metadata(
                self.job.id, f"completed_rung_{self._rung}", str(self._objective)
            )

    def stop(self) -> bool:

        if not (hasattr(self, "best_objective")):
            # print("START")
            self.best_objective = self._retrieve_best_objective()

        # Enforce Pre-conditions Before Learning-Curve based Early Discarding
        if super().stop():
            return True

        # This condition will enforce the stopper to stop the evaluation at the first step
        # for the first evaluation (The FABOLAS method does the same, bias the first samples with
        # small budgets)
        if self.best_objective is None:
            return True

        halting_step = self._compute_halting_step()
        if self.step < max(self.min_steps, self.min_obs_to_fit):

            if self.step >= halting_step:
                # TODO: make fixed parameter accessible
                competing_objectives = self._get_competiting_objectives(self._rung)
                if len(competing_objectives) > 10:
                    q_objective = np.quantile(competing_objectives, q=0.33)
                    if self._objective < q_objective:
                        return True
                self._rung += 1

            return False

        # Check if the halting budget condition is met
        if self.step < halting_step and self.step < self._step_max_utility:
            return False

        # Check if the evaluation should be stopped based on LC-Model

        # Fit and predict the performance of the learning curve model
        z_opt, y_pred = self._fit_and_predict_lc_model_performance()
        # print(f"{z_opt=} - {y_pred=} - best={self.best_objective}")

        # Check if the configuration is promotable based on its predicted objective value
        promotable = (self.best_objective is None or y_pred > self.best_objective) and (
            self.step < z_opt
        )

        # Return whether the configuration should be stopped
        if promotable:
            self._rung += 1

            if self.step >= self.max_steps_:
                return True

        return not (promotable)
