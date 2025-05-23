import numpy as np
import warnings

from scipy.stats import norm


def gaussian_acquisition_1D(
    X,
    model,
    y_opt=None,
    acq_func="LCB",
    acq_func_kwargs=None,
    return_grad=True,
):
    """A wrapper around the acquisition function that is called by fmin_l_bfgs_b.

    This is because lbfgs allows only 1-D input.
    """
    return _gaussian_acquisition(
        np.expand_dims(X, axis=0),
        model,
        y_opt,
        acq_func=acq_func,
        acq_func_kwargs=acq_func_kwargs,
        return_grad=return_grad,
    )


def _gaussian_acquisition(
    X,
    model,
    y_opt=None,
    acq_func="LCB",
    return_grad=False,
    acq_func_kwargs=None,
):
    """Wrapper so that the output of this function can be
    directly passed to a minimizer.
    """
    # Check inputs
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            "X is {}-dimensional, however," " it must be 2-dimensional.".format(X.ndim)
        )

    # Check if the deterministic acquisition function variant should be used
    deterministic = False
    if "d" == acq_func[-1]:
        deterministic = True
        acq_func = acq_func[:-1]

    if acq_func_kwargs is None:
        acq_func_kwargs = dict()
    xi = acq_func_kwargs.get("xi", 0.01)
    kappa = acq_func_kwargs.get("kappa", 1.96)

    # Evaluate acquisition function
    per_second = acq_func.endswith("ps")
    if per_second:
        model, time_model = model.estimators_

    if acq_func in ["LCB"]:
        func_and_grad = gaussian_lcb(
            X, model, kappa, return_grad, deterministic=deterministic
        )

        if return_grad:
            acq_vals, acq_grad = func_and_grad
        else:
            acq_vals = func_and_grad

    elif acq_func in ["EI", "PI", "EIps", "PIps"]:
        if acq_func in ["EI", "EIps"]:
            func_and_grad = gaussian_ei(
                X, model, y_opt, xi, return_grad, deterministic=deterministic
            )
        else:
            func_and_grad = gaussian_pi(
                X, model, y_opt, xi, return_grad, deterministic=deterministic
            )

        if return_grad:
            acq_vals = -func_and_grad[0]
            acq_grad = -func_and_grad[1]
        else:
            acq_vals = -func_and_grad

        if acq_func in ["EIps", "PIps"]:
            if return_grad:
                mu, std, mu_grad, std_grad = time_model.predict(
                    X, return_std=True, return_mean_grad=True, return_std_grad=True
                )
            else:
                mu, std = time_model.predict(X, return_std=True)

            # acq = acq / E(t)
            inv_t = np.exp(-mu + 0.5 * std**2)
            acq_vals *= inv_t

            # grad = d(acq_func) * inv_t + (acq_vals *d(inv_t))
            # inv_t = exp(g)
            # d(inv_t) = inv_t * grad(g)
            # d(inv_t) = inv_t * (-mu_grad + std * std_grad)
            if return_grad:
                acq_grad *= inv_t
                acq_grad += acq_vals * (-mu_grad + std * std_grad)

    elif acq_func in ["MES"]:
        acq_vals = -gaussian_mes(X, model, deterministic=deterministic)
        if return_grad:
            raise NotImplementedError(
                "Gradient not implemented for MES acquisition function."
            )
    else:
        raise ValueError("Acquisition function not implemented.")

    if return_grad:
        return acq_vals, acq_grad

    return acq_vals


def gaussian_lcb(X, model, kappa=1.96, return_grad=False, deterministic=False):
    """Use the lower confidence bound to estimate the acquisition
    values.

    The trade-off between exploitation and exploration is left to
    be controlled by the user through the parameter ``kappa``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Values where the acquisition function should be computed.

    model : sklearn estimator that implements predict with ``return_std``
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    kappa : float, default 1.96 or 'inf'
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        If set to 'inf', the acquisition function will only use the variance
        which is useful in a pure exploration setting.
        Useless if ``method`` is not set to "LCB".

    return_grad : boolean, optional
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns:
    -------
    values : array-like, shape (X.shape[0],)
        Acquisition function values computed at X.

    grad : array-like, shape (n_samples, n_features)
        Gradient at X.
    """
    # Compute posterior.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True, return_std_grad=True
            )

            if kappa == "inf":
                return -std, -std_grad
            return mu - kappa * std, mu_grad - kappa * std_grad

        else:
            if deterministic:
                mu, std_al, std_ep = model.predict(
                    X, return_std=True, disentangled_std=True
                )
                std = std_ep
            else:
                mu, std = model.predict(X, return_std=True)
            if kappa == "inf":
                return -std
            return mu - kappa * std


def gaussian_pi(X, model, y_opt=0.0, xi=0.01, return_grad=False, deterministic=False):
    """Use the probability of improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)` form a gaussian with a
    certain mean and standard deviation approximated by the model.

    The PI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 1``, if ``f(x) < y_opt`` and ``u(f(x)) = 0``,
    if``f(x) > y_opt``.

    This means that the PI condition does not care about how "better" the
    predictions are than the previous values, since it gives an equal reward
    to all of them.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Values where the acquisition function should be computed.

    model : sklearn estimator that implements predict with ``return_std``
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    y_opt : float, default 0
        Previous minimum value which we would like to improve upon.

    xi : float, default=0.01
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    return_grad : boolean, optional
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns:
    -------
    values : [array-like, shape=(X.shape[0],)
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True, return_std_grad=True
            )
        else:
            if deterministic:
                mu, std_al, std_ep = model.predict(
                    X, return_std=True, disentangled_std=True
                )
                std = std_ep
            else:
                mu, std = model.predict(X, return_std=True)

    # check dimensionality of mu, std so we can divide them below
    if (mu.ndim != 1) or (std.ndim != 1):
        raise ValueError(
            "mu and std are {}-dimensional and {}-dimensional, "
            "however both must be 1-dimensional. Did you train "
            "your model with an (N, 1) vector instead of an "
            "(N,) vector?".format(mu.ndim, std.ndim)
        )

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    values[mask] = norm.cdf(scaled)

    if return_grad:
        if not np.all(mask):
            return values, np.zeros_like(std_grad)

        # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
        # improve_grad is the gradient of t wrt x.
        improve_grad = -mu_grad * std - std_grad * improve
        improve_grad /= std**2

        return values, improve_grad * norm.pdf(scaled)

    return values


def gaussian_ei(X, model, y_opt=0.0, xi=0.01, return_grad=False, deterministic=False):
    """Use the expected improvement to calculate the acquisition values.

    The conditional probability `P(y=f(x) | x)` form a gaussian with a certain
    mean and standard deviation approximated by the model.

    The EI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 0``, if ``f(x) > y_opt`` and ``u(f(x)) = y_opt - f(x)``,
    if``f(x) < y_opt``.

    This solves one of the issues of the PI condition by giving a reward
    proportional to the amount of improvement got.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Values where the acquisition function should be computed.

    model : sklearn estimator that implements predict with ``return_std``
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    y_opt : float, default 0
        Previous minimum value which we would like to improve upon.

    xi : float, default=0.01
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    return_grad : boolean, optional
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns:
    -------
    values : array-like, shape=(X.shape[0],)
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True, return_std_grad=True
            )

        else:
            if deterministic:
                mu, std_al, std_ep = model.predict(
                    X, return_std=True, disentangled_std=True
                )
                std = std_ep
            else:
                mu, std = model.predict(X, return_std=True)

    # check dimensionality of mu, std so we can divide them below
    if (mu.ndim != 1) or (std.ndim != 1):
        raise ValueError(
            "mu and std are {}-dimensional and {}-dimensional, "
            "however both must be 1-dimensional. Did you train "
            "your model with an (N, 1) vector instead of an "
            "(N,) vector?".format(mu.ndim, std.ndim)
        )

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore

    if return_grad:
        if not np.all(mask):
            return values, np.zeros_like(std_grad)

        # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
        # improve_grad is the gradient of t wrt x.
        improve_grad = -mu_grad * std - std_grad * improve
        improve_grad /= std**2
        cdf_grad = improve_grad * pdf
        pdf_grad = -improve * cdf_grad
        exploit_grad = -mu_grad * cdf - pdf_grad
        explore_grad = std_grad * pdf + pdf_grad

        grad = exploit_grad + explore_grad
        return values, grad

    return values


def gaussian_mes(X, model, k_samples=10, deterministic=False):
    """Use the max-value entropy to calculate the acquisition values.
    Article: https://arxiv.org/abs/1703.01968
    Source implementation: https://github.com/zi-w/Max-value-Entropy-Search/blob/master/acFuns/evaluateMES.m

    The conditional probability `P(y=f(x) | x)` form a gaussian with a certain
    mean and standard deviation approximated by the model.

    The EI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 0``, if ``f(x) > y_opt`` and ``u(f(x)) = y_opt - f(x)``,
    if``f(x) < y_opt``.

    This solves one of the issues of the PI condition by giving a reward
    proportional to the amount of improvement got.

    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Values where the acquisition function should be computed.

    model : sklearn estimator that implements predict with ``return_std``
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.

    y_opt : float, default 0
        Previous minimum value which we would like to improve upon.

    xi : float, default=0.01
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    return_grad : boolean, optional
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.

    Returns:
    -------
    values : array-like, shape=(X.shape[0],)
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if deterministic:
            mu, std_al, std_ep = model.predict(
                X, return_std=True, disentangled_std=True
            )
            std = std_ep
        else:
            mu, std = model.predict(X, return_std=True)

        # MES is defined for a maximization problem but the model prediction are for minimization
        # we map minimization to maximization by multiplying by -1
        mu *= -1

    # check dimensionality of mu, std so we can divide them below
    if (mu.ndim != 1) or (std.ndim != 1):
        raise ValueError(
            "mu, std_al, std_ep are {}-dimensional and {}-dimensional, "
            "however both must be 1-dimensional. Did you train "
            "your model with an (N, 1) vector instead of an "
            "(N,) vector?".format(mu.ndim, std.ndim)
        )

    values = np.zeros_like(mu)
    eps = 1e-10
    std = np.maximum(std, eps)
    for _ in range(k_samples):
        y_sample = norm.rvs(loc=mu, scale=std)
        gamma = (np.max(y_sample) - mu) / std
        pdfgamma = np.maximum(norm.pdf(gamma), eps)
        cdfgamma = np.maximum(norm.cdf(gamma), eps)
        values += 0.5 * gamma * pdfgamma / cdfgamma - np.log(cdfgamma)
    values /= k_samples

    return values
