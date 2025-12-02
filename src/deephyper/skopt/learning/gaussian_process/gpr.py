import warnings

import numpy as np
import sklearn
from packaging import version
from scipy.linalg import cho_solve, solve_triangular
from sklearn.gaussian_process import (
    GaussianProcessRegressor as sk_GaussianProcessRegressor,
)
from sklearn.utils import check_array

from .kernels import RBF, ConstantKernel, Sum, WhiteKernel


def _param_for_white_kernel_in_Sum(kernel, kernel_str=""):
    """Check if a WhiteKernel exists in a Sum Kernel
    and if it does return the corresponding key in
    `kernel.get_params()`
    """
    if kernel_str != "":
        kernel_str = kernel_str + "__"

    if isinstance(kernel, Sum):
        for param, child in kernel.get_params(deep=False).items():
            if isinstance(child, WhiteKernel):
                return True, kernel_str + param
            else:
                present, child_str = _param_for_white_kernel_in_Sum(
                    child, kernel_str + param
                )
                if present:
                    return True, child_str

    return False, "_"


class GaussianProcessRegressor(sk_GaussianProcessRegressor):
    """GaussianProcessRegressor that allows noise tunability.

    This implementation is based on Algorithm 2.1 of *Gaussian Processes for
    Machine Learning* (GPML) by Rasmussen and Williams.

    In addition to the standard scikit-learn estimator API,
    `GaussianProcessRegressor`:

        * Allows prediction without prior fitting (based on the GP prior).
        * Provides an additional method `sample_y(X)` to evaluate samples drawn
        from the GPR (prior or posterior) at given inputs.
        * Exposes a method `log_marginal_likelihood(theta)` that can be used
        externally for alternative hyperparameter selection strategies,
        such as Markov chain Monte Carlo.

    Args:
        kernel (Kernel):  
            The kernel specifying the covariance function of the GP. If None,
            the default kernel ``1.0 * RBF(1.0)`` is used. Kernel hyperparameters
            are optimized during fitting.

        alpha (float or array-like, optional):  
            Value added to the diagonal of the kernel matrix during fitting
            (default: 1e-10). Larger values correspond to higher assumed noise
            levels and improve numerical stability. If an array is provided, it
            must match the number of training samples and is interpreted as a
            per-sample noise level. Equivalent to adding a `WhiteKernel` with
            `c = alpha`. Provided mainly for convenience and consistency with
            Ridge regression.

        optimizer (str or callable, optional):  
            Optimizer for kernel parameter tuning (default: `"fmin_l_bfgs_b"`).  
            A string selects one of the built-in optimizers.  
            A callable must follow the signature:

            .. code-block:: python
            
                def optimizer(obj_func, initial_theta, bounds):
                    # obj_func: objective to maximize; accepts theta and
                    #           optionally eval_gradient
                    # initial_theta: initial hyperparameter state
                    # bounds: box constraints for theta
                    return theta_opt, func_min

            If None, kernel parameters are kept fixed.

            Available built-in optimizers:
                * `"fmin_l_bfgs_b"`

        n_restarts_optimizer (int, optional):  
            Number of optimizer restarts to maximize the log-marginal
            likelihood (default: 0). The first run uses the kernel's initial
            parameters; additional runs start from random log-uniform samples.
            If > 0, all parameter bounds must be finite. A value of 0 implies a
            single run.

        normalize_y (bool, optional):  
            Whether to normalize target values to have zero mean (default:
            False). Should be enabled when the target mean deviates significantly
            from zero. Note: this effectively alters the GP prior using the data,
            which contradicts the likelihood principle; thus the default is False.

        copy_X_train (bool, optional):  
            If True (default), a persistent copy of the training data is stored.
            If False, only a reference is kept, so external modification of the
            data may affect predictions.

        random_state (int or numpy.random.RandomState, optional):  
            Random generator used for initialization. If an integer is provided,
            it sets the seed. Defaults to NumPy's global RNG.

        noise (str, optional):  
            If set to `"gaussian"`, the model assumes that `y` is a noisy estimate
            of the latent function `f(x)` with Gaussian noise.

    Attributes:
        X_train_ (array-like of shape (n_samples, n_features)):  
            Training feature values.

        y_train_ (array-like of shape (n_samples, [n_output_dims])):  
            Training target values.

        kernel_ (Kernel):  
            The kernel used for prediction, identical in structure to the input
            kernel but with optimized hyperparameters.

        L_ (array-like of shape (n_samples, n_samples)):  
            Lower-triangular Cholesky decomposition of the kernel matrix evaluated
            at ``X_train_``.

        alpha_ (array-like of shape (n_samples,)):  
            Dual coefficients of training data points in kernel space.

        log_marginal_likelihood_value_ (float):  
            Log-marginal likelihood evaluated at ``self.kernel_.theta``.

        noise_ (float):  
            Estimated Gaussian noise level. Only relevant when ``noise="gaussian"``.
    """

    def __init__(
        self,
        kernel=None,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
        noise=None,
    ):
        self.noise = noise
        super(GaussianProcessRegressor, self).__init__(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state,
        )

    def fit(self, X, y):
        """Fit Gaussian process regression model.

        Args:
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values

        Returns:
        self
            Returns an instance of self.
        """
        if isinstance(self.noise, str) and self.noise != "gaussian":
            raise ValueError("expected noise to be 'gaussian', got %s" % self.noise)

        if self.kernel is None:
            self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
        if self.noise == "gaussian":
            self.kernel = self.kernel + WhiteKernel()
        elif self.noise:
            self.kernel = self.kernel + WhiteKernel(
                noise_level=self.noise, noise_level_bounds="fixed"
            )
        super(GaussianProcessRegressor, self).fit(X, y)

        self.noise_ = None

        if self.noise:
            # The noise component of this kernel should be set to zero
            # while estimating K(X_test, X_test)
            # Note that the term K(X, X) should include the noise but
            # this (K(X, X))^-1y is precomputed as the attribute `alpha_`.
            # (Notice the underscore).
            # This has been described in Eq 2.24 of
            # http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
            # Hence this hack
            if isinstance(self.kernel_, WhiteKernel):
                self.kernel_.set_params(noise_level=0.0)

            else:
                white_present, white_param = _param_for_white_kernel_in_Sum(
                    self.kernel_
                )

                # This should always be true. Just in case.
                if white_present:
                    noise_kernel = self.kernel_.get_params()[white_param]
                    self.noise_ = noise_kernel.noise_level
                    self.kernel_.set_params(
                        **{white_param: WhiteKernel(noise_level=0.0)}
                    )

        # Precompute arrays needed at prediction
        L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        self.K_inv_ = L_inv.dot(L_inv.T)

        # Fix deprecation warning #462
        sklearn_version = version.parse(sklearn.__version__)
        if sklearn_version >= version.parse("0.23.0"):
            self.y_train_std_ = self._y_train_std
            self.y_train_mean_ = self._y_train_mean
        elif sklearn_version >= version.parse("0.19.0"):
            self.y_train_mean_ = self._y_train_mean
            self.y_train_std_ = 1
        else:
            self.y_train_mean_ = self.y_train_mean
            self.y_train_std_ = 1

        return self

    def predict(
        self,
        X,
        return_std=False,
        return_cov=False,
        return_mean_grad=False,
        return_std_grad=False,
    ):
        """Predict output for X.

        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True),
        the gradient of the mean and the standard-deviation with respect to X
        can be optionally provided.

        Args:
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated.

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        return_mean_grad : bool, default: False
            Whether or not to return the gradient of the mean.
            Only valid when X is a single point.

        return_std_grad : bool, default: False
            Whether or not to return the gradient of the std.
            Only valid when X is a single point.

        Returns:
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.

        y_mean_grad : shape = (n_samples, n_features)
            The gradient of the predicted mean

        y_std_grad : shape = (n_samples, n_features)
            The gradient of the predicted std.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance."
            )

        if return_std_grad and not return_std:
            raise ValueError("Not returning std_gradient without returning " "the std.")

        X = check_array(X)
        if X.shape[0] != 1 and (return_mean_grad or return_std_grad):
            raise ValueError("Not implemented for n_samples > 1")

        if not hasattr(self, "X_train_"):  # Not fit; predict based on GP prior
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = self.kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = self.kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

        else:  # Predict based on GP posterior
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            # undo normalisation
            y_mean = self.y_train_std_ * y_mean + self.y_train_mean_

            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6
                # undo normalisation
                y_cov = y_cov * self.y_train_std_**2
                return y_mean, y_cov

            elif return_std:
                K_inv = self.K_inv_

                # Compute variance of predictive distribution
                y_var = self.kernel_.diag(X)
                y_var -= np.einsum("ki,kj,ij->k", K_trans, K_trans, K_inv)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn(
                        "Predicted variances smaller than 0. "
                        "Setting those variances to 0."
                    )
                    y_var[y_var_negative] = 0.0
                # undo normalisation
                y_var = y_var * self.y_train_std_**2
                y_std = np.sqrt(y_var)

            if return_mean_grad:
                grad = self.kernel_.gradient_x(X[0], self.X_train_)
                grad_mean = np.dot(grad.T, self.alpha_)
                # undo normalisation
                grad_mean = grad_mean * self.y_train_std_
                if return_std_grad:
                    grad_std = np.zeros(X.shape[1])
                    if not np.allclose(y_std, grad_std):
                        grad_std = -np.dot(K_trans, np.dot(K_inv, grad))[0] / y_std
                        # undo normalisation
                        grad_std = grad_std * self.y_train_std_**2
                    return y_mean, y_std, grad_mean, grad_std

                if return_std:
                    return y_mean, y_std, grad_mean
                else:
                    return y_mean, grad_mean

            else:
                if return_std:
                    return y_mean, y_std
                else:
                    return y_mean
