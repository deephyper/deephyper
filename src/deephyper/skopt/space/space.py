"""Submodule for Skopt space definition."""

import numbers

import ConfigSpace as CS
import numpy as np
import scipy.stats as ss
import yaml
from ConfigSpace.util import deactivate_inactive_hyperparameters
from scipy.stats import gaussian_kde
from sklearn.utils import check_random_state

from deephyper.core.utils.joblib_utils import Parallel, delayed
from deephyper.core.utils import CaptureSTD

from .transformers import (
    CategoricalEncoder,
    Identity,
    LabelEncoder,
    LogN,
    Normalize,
    Pipeline,
    StringEncoder,
    ToInteger,
)


# helper class to be able to print [1, ..., 4] instead of [1, '...', 4]
class _Ellipsis:
    def __repr__(self):
        return "..."


def _transpose_list_array(x):
    """Utility to transpose a list matrix."""
    n_dims = len(x)
    assert n_dims > 0
    n_samples = len(x[0])
    rows = [None] * n_samples
    for i in range(n_samples):
        r = [None] * n_dims
        for j in range(n_dims):
            r[j] = x[j][i]
        rows[i] = r
    return rows


def check_dimension(dimension, transform=None):
    """Turn a provided dimension description into a dimension object.

    Checks that the provided dimension falls into one of the
    supported types. For a list of supported types, look at
    the documentation of ``dimension`` below.

    If ``dimension`` is already a ``Dimension`` instance, return it.

    Args:
        dimension (Dimension):
            Search space Dimension.
            Each search dimension can be defined either as

            - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
              dimensions),
            - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
              dimensions),
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
              `Categorical`).

        transform (str): One of
            "identity", "normalize", "string", "label", "onehot" optional

            - For `Categorical` dimensions, the following transformations are
              supported.

              - "onehot" (default) one-hot transformation of the original space.
              - "label" integer transformation of the original space
              - "string" string transformation of the original space.
              - "identity" same as the original space.

            - For `Real` and `Integer` dimensions, the following transformations
              are supported.

              - "identity", (default) the transformed space is the same as the
                original space.
              - "normalize", the transformed space is scaled to be between 0 and 1.

    Returns:
            dimension (Dimension): Dimension instance.
    """
    if isinstance(dimension, Dimension):
        return dimension

    if not isinstance(dimension, (list, tuple, np.ndarray)):
        raise ValueError("Dimension has to be a list or tuple.")

    # A `Dimension` described by a single value is assumed to be
    # a `Categorical` dimension. This can be used in `BayesSearchCV`
    # to define subspaces that fix one value, e.g. to choose the
    # model type, see "sklearn-gridsearchcv-replacement.py"
    # for examples.
    if len(dimension) == 1:
        return Categorical(dimension, transform=transform)

    if len(dimension) == 2:
        if any(
            [isinstance(d, (str, bool)) or isinstance(d, np.bool_) for d in dimension]
        ):
            return Categorical(dimension, transform=transform)
        elif all([isinstance(dim, numbers.Integral) for dim in dimension]):
            return Integer(*dimension, transform=transform)
        elif any([isinstance(dim, numbers.Real) for dim in dimension]):
            return Real(*dimension, transform=transform)
        else:
            raise ValueError(
                "Invalid dimension {}. Read the documentation for"
                " supported types.".format(dimension)
            )

    if len(dimension) == 3:
        if any([isinstance(dim, int) for dim in dimension[:2]]) and dimension[2] in [
            "uniform",
            "log-uniform",
        ]:
            return Integer(*dimension, transform=transform)
        elif any(
            [isinstance(dim, (float, int)) for dim in dimension[:2]]
        ) and dimension[2] in ["uniform", "log-uniform"]:
            return Real(*dimension, transform=transform)
        else:
            return Categorical(dimension, transform=transform)

    if len(dimension) == 4:
        if (
            any([isinstance(dim, int) for dim in dimension[:2]])
            and dimension[2] == "log-uniform"
            and isinstance(dimension[3], int)
        ):
            return Integer(*dimension, transform=transform)
        elif (
            any([isinstance(dim, (float, int)) for dim in dimension[:2]])
            and dimension[2] == "log-uniform"
            and isinstance(dimension[3], int)
        ):
            return Real(*dimension, transform=transform)

    if len(dimension) > 3:
        return Categorical(dimension, transform=transform)

    raise ValueError(
        "Invalid dimension {}. Read the documentation for " "supported types.".format(
            dimension
        )
    )


class Dimension:
    """Base class for search space dimensions."""

    prior = None

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        Args:
        n_samples : int or None
            The number of samples to be drawn.

        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.
        """
        rng = check_random_state(random_state)
        samples = self._rvs.rvs(size=n_samples, random_state=rng)
        return self.inverse_transform(samples)

    def transform(self, X):
        """Transform samples form the original space to a warped space."""
        return self.transformer.transform(X)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the original space."""
        return self.transformer.inverse_transform(Xt)

    def set_transformer(self):
        """Setter for the transformer."""
        raise NotImplementedError

    @property
    def size(self):
        """Dimensionality of sampel from the dimension before the transform/preprocessing."""
        return 1

    @property
    def transformed_size(self):
        """Dimensionality of samples from the dimension after the transform/preprocessing."""
        return 1

    @property
    def bounds(self):
        """Bounds before the transform/preprocessing."""
        raise NotImplementedError

    @property
    def is_constant(self):
        """Test if the dimension is a constant."""
        raise NotImplementedError

    @property
    def transformed_bounds(self):
        """Bounds after the transform/preprocessing."""
        raise NotImplementedError

    @property
    def name(self):
        """Name of the dimension."""
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str) or value is None:
            self._name = value
        else:
            raise ValueError("Dimension's name must be either string or None.")


def _uniform_inclusive(loc=0.0, scale=1.0):
    # like scipy.stats.distributions but inclusive of `high`
    # XXX scale + 1. might not actually be a float after scale if
    # XXX scale is very large.
    return ss.uniform(loc=loc, scale=np.nextafter(scale, scale + 1.0))


def _normal_inclusive(loc=0.0, scale=1.0, lower=-2, upper=2):
    assert lower <= upper
    a, b = (lower - loc) / scale, (upper - loc) / scale
    return ss.truncnorm(a, b, loc=loc, scale=scale)


class Real(Dimension):
    """Search space dimension that can take on any real value.

    Parameters
    ----------
    low : float
        Lower bound (inclusive).

    high : float
        Upper bound (inclusive).

    prior : "uniform" or "log-uniform", default="uniform"
        Distribution to use when sampling random points for this dimension.

        - If `"uniform"`, points are sampled uniformly between the lower
          and upper bounds.
        - If `"log-uniform"`, points are sampled uniformly between
          `log(lower, base)` and `log(upper, base)` where log
          has base `base`.

    base : int
        The logarithmic base to use for a log-uniform prior.
        - Default 10, otherwise commonly 2.

    transform : "identity", "normalize", optional
        The following transformations are supported.

        - "identity", (default) the transformed space is the same as the
          original space.
        - "normalize", the transformed space is scaled to be between
          0 and 1.

    name : str or None
        Name associated with the dimension, e.g., "learning rate".

    dtype : str or dtype, default=float
        float type which will be used in inverse_transform,
        can be float.

    """

    def __init__(
        self,
        low,
        high,
        prior="uniform",
        base=10,
        transform=None,
        name=None,
        dtype=float,
        loc=None,
        scale=None,
    ):
        if high <= low:
            raise ValueError(
                "the lower bound {} has to be less than the" " upper bound {}".format(
                    low, high
                )
            )
        if prior not in ["uniform", "log-uniform", "normal"]:
            raise ValueError(
                "prior should be 'normal', 'uniform' or 'log-uniform'" " got {}".format(
                    prior
                )
            )
        self.low = low
        self.high = high
        self.prior = prior
        self.base = base
        self.log_base = np.log10(base)
        self.name = name
        self.dtype = dtype
        self.loc = loc
        self.scale = scale
        self._rvs = None
        self.transformer = None
        self.transform_ = transform
        if isinstance(self.dtype, str) and self.dtype not in [
            "float",
            "float16",
            "float32",
            "float64",
        ]:
            raise ValueError(
                "dtype must be 'float', 'float16', 'float32'"
                "or 'float64'"
                " got {}".format(self.dtype)
            )
        elif isinstance(self.dtype, type) and not np.issubdtype(
            self.dtype, np.floating
        ):
            raise ValueError(
                "dtype must be a np.floating subtype;" " got {}".format(self.dtype)
            )

        if transform is None:
            transform = "identity"
        self.set_transformer(transform)

    def set_transformer(self, transform="identity"):
        """Define rvs and transformer spaces.

        Args:
        transform : str
           Can be 'normalize' or 'identity'

        """
        self.transform_ = transform

        if self.transform_ not in ["normalize", "identity"]:
            raise ValueError(
                "transform should be 'normalize' or 'identity'" " got {}".format(
                    self.transform_
                )
            )

        # XXX: The _rvs is for sampling in the transformed space.
        # The rvs on Dimension calls inverse_transform on the points sampled
        # using _rvs
        if self.transform_ == "normalize":
            # set upper bound to next float after 1. to make the numbers
            # inclusive of upper edge
            self._rvs = _uniform_inclusive(0.0, 1.0)
            assert self.prior in ["uniform", "log-uniform"]
            if self.prior == "uniform":
                self.transformer = Pipeline(
                    [Identity(), Normalize(self.low, self.high)]
                )
            else:
                self.transformer = Pipeline(
                    [
                        LogN(self.base),
                        Normalize(
                            np.log10(self.low) / self.log_base,
                            np.log10(self.high) / self.log_base,
                        ),
                    ]
                )
        else:
            if self.prior == "uniform":
                self._rvs = _uniform_inclusive(self.low, self.high - self.low)
                self.transformer = Identity()
            elif self.prior == "normal":
                self._rvs = _normal_inclusive(self.loc, self.scale, self.low, self.high)
                self.transformer = Identity()
            else:
                self._rvs = _uniform_inclusive(
                    np.log10(self.low) / self.log_base,
                    np.log10(self.high) / self.log_base
                    - np.log10(self.low) / self.log_base,
                )
                self.transformer = LogN(self.base)

    def __eq__(self, other):
        """Test if the dimension is equal to an other by testing if types, bounds, prior, and other parameters are equal."""
        return (
            type(self) is type(other)
            and np.allclose([self.low], [other.low])
            and np.allclose([self.high], [other.high])
            and self.prior == other.prior
            and self.transform_ == other.transform_
            and self.loc == other.loc
            and self.scale == other.scale
        )

    def __repr__(self):
        """String representation of the dimension."""
        return "Real(low={}, high={}, prior='{}', transform='{}', loc='{}', scale='{}')".format(
            self.low, self.high, self.prior, self.transform_, self.loc, self.scale
        )

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the original space."""
        inv_transform = super(Real, self).inverse_transform(Xt)
        if isinstance(inv_transform, list):
            inv_transform = np.array(inv_transform)

        if self.dtype == float or self.dtype == "float":
            # necessary, otherwise the type is converted to a numpy type
            return getattr(inv_transform, "tolist")()
        else:
            return inv_transform

    @property
    def bounds(self):
        """Bounds before the transform/preprocessing."""
        return (self.low, self.high)

    @property
    def is_constant(self):
        """Tet if the dimension is constant."""
        return self.low == self.high

    def __contains__(self, point):
        """Test if a value is contained in the support of the dimension."""
        if isinstance(point, list):
            point = np.array(point)
        if point == np.nan:
            return True
        else:
            return self.low <= point <= self.high

    @property
    def transformed_bounds(self):
        """Bounds after the transform/preprocessing."""
        if self.transform_ == "normalize":
            return 0.0, 1.0
        else:
            if self.prior == "uniform":
                return self.low, self.high
            else:
                return (
                    np.log10(self.low) / self.log_base,
                    np.log10(self.high) / self.log_base,
                )

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Args:
        a : float
            First point.

        b : float
            Second point.
        """
        if not (a in self and b in self):
            raise RuntimeError(
                "Can only compute distance for values within "
                "the space, not %s and %s." % (a, b)
            )
        return abs(a - b)

    def update_prior(self, X, y, q=0.9):
        """Fit a Kernel Density Estimator to the data to increase density of samples around regions of interest instead of uniform random-sampling."""
        X = np.array(X)
        y = np.array(y)

        y_ = np.quantile(y, q)  # threshold
        X_low = X[y <= y_]

        # It is possible that fitting the Gaussian Kernel Density Estimator
        # triggers an error, for example if all values of X_low are the same.
        # In this case, we fall back to uniform sampling or we reuse the last
        # fitted self._kde.
        try:
            kde = gaussian_kde(X_low)
            self._kde = kde
        except np.linalg.LinAlgError:
            pass

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        Args:
            n_samples : int or None
                The number of samples to be drawn.
            random_state : int, RandomState instance, or None (default)
                Set random state to something other than None for reproducible
                results.
        """
        rng = check_random_state(random_state)

        if hasattr(self, "_kde"):
            samples = self._kde.resample(n_samples, rng).reshape(-1)
            samples = np.clip(samples, self.low, self.high)
        else:
            samples = self._rvs.rvs(size=n_samples, random_state=rng)

        return self.inverse_transform(samples)


class Integer(Dimension):
    """Search space dimension that can take on integer values.

    Args:
        low : int
            Lower bound (inclusive).

        high : int
            Upper bound (inclusive).

        prior : "uniform" or "log-uniform", default="uniform"
            Distribution to use when sampling random integers for
            this dimension.

            - If `"uniform"`, integers are sampled uniformly between the lower
            and upper bounds.
            - If `"log-uniform"`, integers are sampled uniformly between
            `log(lower, base)` and `log(upper, base)` where log
            has base `base`.

        base : int
            The logarithmic base to use for a log-uniform prior.

            - Default 10, otherwise commonly 2.

        transform : "identity", "normalize", optional
            The following transformations are supported.

            - "identity", (default) the transformed space is the same as the
            original space.
            - "normalize", the transformed space is scaled to be between
            0 and 1.

        name : str or None
            Name associated with dimension, e.g., "number of trees".

        dtype : str or dtype, default=np.int64
            integer type which will be used in inverse_transform,
            can be int, np.int16, np.uint32, np.int32, np.int64 (default).
            When set to int, `inverse_transform` returns a list instead of
            a numpy array
    """

    def __init__(
        self,
        low,
        high,
        prior="uniform",
        base=10,
        transform=None,
        name=None,
        dtype=np.int64,
        loc=None,
        scale=None,
    ):
        if high <= low:
            raise ValueError(
                "the lower bound {} has to be less than the" " upper bound {}".format(
                    low, high
                )
            )
        if prior not in ["uniform", "log-uniform"]:
            raise ValueError(
                "prior should be 'uniform' or 'log-uniform'" " got {}".format(prior)
            )
        self.low = low
        self.high = high
        self.prior = prior
        self.base = base
        self.log_base = np.log10(base)
        self.name = name
        self.dtype = dtype
        self.transform_ = transform
        self._rvs = None
        self.transformer = None
        self.loc = loc
        self.scale = scale

        if isinstance(self.dtype, str) and self.dtype not in [
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ]:
            raise ValueError(
                "dtype must be 'int', 'int8', 'int16',"
                "'int32', 'int64', 'uint8',"
                "'uint16', 'uint32', or"
                "'uint64', but got {}".format(self.dtype)
            )
        elif isinstance(self.dtype, type) and self.dtype not in [
            int,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]:
            raise ValueError(
                "dtype must be 'int', 'np.int8', 'np.int16',"
                "'np.int32', 'np.int64', 'np.uint8',"
                "'np.uint16', 'np.uint32', or"
                "'np.uint64', but got {}".format(self.dtype)
            )

        if transform is None:
            transform = "identity"
        self.set_transformer(transform)

    def set_transformer(self, transform="identity"):
        """Define _rvs and transformer spaces.

        Args:
        transform : str
           Can be 'normalize' or 'identity'

        """
        self.transform_ = transform

        if transform not in ["normalize", "identity"]:
            raise ValueError(
                "transform should be 'normalize' or 'identity'" " got {}".format(
                    self.transform_
                )
            )

        if self.transform_ == "normalize":
            self._rvs = _uniform_inclusive(0.0, 1.0)
            assert self.prior in ["uniform", "log-uniform"]
            if self.prior == "uniform":
                self.transformer = Pipeline(
                    [Identity(), Normalize(self.low, self.high, is_int=True)]
                )
            else:
                self.transformer = Pipeline(
                    [
                        LogN(self.base),
                        Normalize(
                            np.log10(self.low) / self.log_base,
                            np.log10(self.high) / self.log_base,
                        ),
                    ]
                )
        else:
            if self.prior == "uniform":
                self._rvs = ss.randint(self.low, self.high + 1)
                self.transformer = Identity()
            elif self.prior == "normal":
                self._rvs = _normal_inclusive(self.loc, self.scale, self.low, self.high)
                self.transformer = ToInteger()
            else:
                self._rvs = _uniform_inclusive(
                    np.log10(self.low) / self.log_base,
                    np.log10(self.high) / self.log_base
                    - np.log10(self.low) / self.log_base,
                )
                self.transformer = LogN(self.base)

    def __eq__(self, other):
        """Test if the dimension is equal to an other by testing if types and bounds are all equal."""
        return (
            type(self) is type(other)
            and np.allclose([self.low], [other.low])
            and np.allclose([self.high], [other.high])
        )

    def __repr__(self):
        """String representation of the dimension."""
        return "Integer(low={}, high={}, prior='{}', transform='{}')".format(
            self.low, self.high, self.prior, self.transform_
        )

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the original space."""
        # The concatenation of all transformed dimensions makes Xt to be
        # of type float, hence the required cast back to int.
        inv_transform = super(Integer, self).inverse_transform(Xt)
        if isinstance(inv_transform, list):
            inv_transform = np.array(inv_transform)
        inv_transform = np.clip(inv_transform, self.low, self.high)

        # PB nan is a float cannot be converted to int
        if any(np.isnan(inv_transform)):
            nan_values = np.isnan(inv_transform)
            inv_transform[nan_values] = np.round(inv_transform[nan_values])
            return inv_transform

        if self.dtype == int or self.dtype == "int":
            # necessary, otherwise the type is converted to a numpy type
            return getattr(np.round(inv_transform).astype(self.dtype), "tolist")()
        else:
            return np.round(inv_transform).astype(self.dtype)

    @property
    def bounds(self):
        """Bounds before transform/preprocessing."""
        return (self.low, self.high)

    @property
    def is_constant(self):
        """Test if the dimension is a constant."""
        return self.low == self.high

    def __contains__(self, point):
        """Test if the value is contained in the support of the dimension."""
        if isinstance(point, list):
            point = np.array(point)
        if point == np.nan:
            return True
        else:
            return self.low <= point <= self.high

    @property
    def transformed_bounds(self):
        """Bounds after the transform/preprocessing."""
        if self.transform_ == "normalize":
            return 0.0, 1.0
        else:
            if self.prior == "uniform":
                return self.low, self.high
            else:
                return (
                    np.log10(self.low) / self.log_base,
                    np.log10(self.high) / self.log_base,
                )

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Args:
        a : int
            First point.

        b : int
            Second point.
        """
        if not (a in self and b in self):
            raise RuntimeError(
                "Can only compute distance for values within "
                "the space, not %s and %s." % (a, b)
            )
        return abs(a - b)


class Categorical(Dimension):
    """Search space dimension that can take on categorical values.

    Args:
        categories (list, shape=(n_categories,)):
            Sequence of possible categories.

        prior (list, shape=(categories,), default=None):
            Prior probabilities for each category. By default all categories
            are equally likely.

        transform (str, Optional):
            - ``"identity"``, the transformed space is the same as the original
            space.
            - ``"string"``,  the transformed space is a string encoded
            representation of the original space.
            - ``"label"``, the transformed space is a label encoded
            representation (integer) of the original space.
            - ``"onehot"``, the transformed space is a one-hot encoded
            representation of the original space.

        name (str, Optional):
            Name associated with dimension, e.g., ``"colors"``.
    """

    def __init__(self, categories, prior=None, transform=None, name=None):
        self.categories = tuple(categories)

        self.name = name

        if transform is None:
            transform = "onehot"
        self.transform_ = transform
        self.transformer = None
        self._rvs = None
        self.prior = prior

        if prior is None:
            self.prior_ = np.tile(1.0 / len(self.categories), len(self.categories))
        else:
            self.prior_ = prior
        self.set_transformer(transform)

    def set_transformer(self, transform="onehot"):
        """Define _rvs and transformer spaces.

        Args:
            transform (str): Can be a value in ``['normalize', 'onehot', 'string', 'label', 'identity']``.

        """
        self.transform_ = transform
        if transform not in ["identity", "onehot", "string", "normalize", "label"]:
            raise ValueError(
                "Expected transform to be 'identity', 'string',"
                "'label' or 'onehot' got {}".format(transform)
            )
        if transform == "onehot":
            self.transformer = CategoricalEncoder()
            self.transformer.fit(self.categories)
        elif transform == "string":
            self.transformer = StringEncoder()
            self.transformer.fit(self.categories)
        elif transform == "label":
            self.transformer = LabelEncoder()
            self.transformer.fit(self.categories)
        elif transform == "normalize":
            self.transformer = Pipeline(
                [
                    LabelEncoder(list(self.categories)),
                    Normalize(0, len(self.categories) - 1, is_int=True),
                ]
            )
        else:
            if all(isinstance(x, (int, np.integer)) for x in self.categories):
                self.transformer = Identity(type_func=lambda x: int(x))
            else:
                self.transformer = Identity()
            self.transformer.fit(self.categories)
        if transform == "normalize":
            self._rvs = _uniform_inclusive(0.0, 1.0)
        else:
            # XXX check that sum(prior) == 1
            self._rvs = ss.rv_discrete(
                values=(range(len(self.categories)), self.prior_)
            )

    def __eq__(self, other):
        """Test if the dimension is equal to an other by checking if types, categories and priors are equal."""
        return (
            type(self) is type(other)
            and self.categories == other.categories
            and np.allclose(self.prior_, other.prior_)
        )

    def __repr__(self):
        """String representation of the dimension."""
        if len(self.categories) > 7:
            cats = self.categories[:3] + (_Ellipsis(),) + self.categories[-3:]
        else:
            cats = self.categories

        if self.prior is not None and len(self.prior) > 7:
            prior = self.prior[:3] + [_Ellipsis()] + self.prior[-3:]
        else:
            prior = self.prior

        return "Categorical(categories={}, prior={}, transform={})".format(
            cats, prior, self.transform_
        )

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the original space."""
        # The concatenation of all transformed dimensions makes Xt to be
        # of type float, hence the required cast back to int.
        inv_transform = super(Categorical, self).inverse_transform(Xt)
        if isinstance(inv_transform, list):
            inv_transform = np.array(inv_transform)
        return inv_transform

    def rvs(self, n_samples=None, random_state=None):
        """Sample elements from the dimension."""
        choices = self._rvs.rvs(size=n_samples, random_state=random_state)

        if isinstance(choices, numbers.Integral):
            return self.categories[choices]
        elif self.transform_ == "normalize" and isinstance(choices, float):
            return self.inverse_transform([(choices)])
        elif self.transform_ == "normalize":
            return self.inverse_transform(list(choices))
        else:
            return [self.categories[c] for c in choices]

    @property
    def transformed_size(self):
        """Cardinality of the dimension after applying transform/preprocessing."""
        if self.transform_ == "onehot":
            size = len(self.categories)
            # when len(categories) == 2, CategoricalEncoder outputs a
            # single value
            return size if size != 2 else 1
        return 1

    @property
    def bounds(self):
        """Bounds of before applying transform/preprocessing."""
        return self.categories

    @property
    def is_constant(self):
        """Test if the current dimension is a constant (with only 1 element it its support)."""
        return len(self.categories) <= 1

    def __contains__(self, point):
        """Test if a value is contained among current categories."""
        return point in self.categories

    @property
    def transformed_bounds(self):
        """Bounds after applying transform/preprocessing."""
        if self.transformed_size == 1:
            N = len(self.categories)
            if self.transform_ == "label":
                return 0.0, float(N - 1)
            elif self.transform_ == "identity":
                return min(self.categories), max(self.categories)
            else:
                return 0.0, 1.0
        else:
            return [(0.0, 1.0) for i in range(self.transformed_size)]

    def distance(self, a, b):
        """Compute distance between category `a` and `b`.

        As categories have no order the distance between two points is one
        if a != b and zero otherwise.

        Args:
        a : category
            First category.

        b : category
            Second category.
        """
        if not (a in self and b in self):
            raise RuntimeError(
                "Can only compute distance for values within"
                " the space, not {} and {}.".format(a, b)
            )
        return 1 if a != b else 0


def _sample_dimension(dim, i, n_samples, random_state, out):
    """Wrapper to sample dimension for joblib parallelization."""
    out[0][:, i] = dim.rvs(n_samples=n_samples, random_state=random_state)


class Space:
    """Initialize a search space from given specifications.

    Parameters
    ----------
    dimensions : list, shape=(n_dims,)
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

        .. note::
            The upper and lower bounds are inclusive for `Integer`
            dimensions.
    """

    def __init__(self, dimensions, model_sdv=None, config_space=None):
        # attribute used when a generative model is used to sample
        self.model_sdv = model_sdv

        # attribute use when a config space is used to sample
        assert config_space is None or isinstance(config_space, CS.ConfigurationSpace)
        self.config_space = config_space

        self.dimensions = [check_dimension(dim) for dim in dimensions]

    def __eq__(self, other):
        """Check if a space is equal to an other by checking if their dimensions are all equal."""
        return all([a == b for a, b in zip(self.dimensions, other.dimensions)])

    def __repr__(self):
        """String representation of the space."""
        if len(self.dimensions) > 31:
            dims = self.dimensions[:15] + [_Ellipsis()] + self.dimensions[-15:]
        else:
            dims = self.dimensions
        return "Space([{}])".format(",\n       ".join(map(str, dims)))

    def __iter__(self):
        """Iter over the ``dimensions`` of the ``Space``."""
        return iter(self.dimensions)

    @property
    def dimension_names(self):
        """Names of all the dimensions in the search-space."""
        index = 0
        names = []
        for dim in self.dimensions:
            if dim.name is None:
                names.append("X_%d" % index)
            else:
                names.append(dim.name)
            index += 1
        return names

    @property
    def is_real(self):
        """Returns true if all dimensions are Real."""
        return all([isinstance(dim, Real) for dim in self.dimensions])

    @classmethod
    def from_yaml(cls, yml_path, namespace=None):
        """Create a ``Space`` from yaml configuration file.

        Args:
            yml_path (str): Full path to yaml configuration file, example YaML below:
                Space:

                - Integer:
                low: -5
                high: 5
                - Categorical:
                categories:
                - a
                - b
                - Real:
                low: 1.0
                high: 5.0
                prior: log-uniform

            namespace : str, default=None
            Namespace within configuration file to use, will use first
            namespace if not provided

        Returns:
            space (Space): Instantiated Space object.
        """
        with open(yml_path, "rb") as f:
            config = yaml.safe_load(f)

        dimension_classes = {
            "real": Real,
            "integer": Integer,
            "categorical": Categorical,
        }

        # Extract space options for configuration file
        if isinstance(config, dict):
            if namespace is None:
                options = next(iter(config.values()))
            else:
                options = config[namespace]
        elif isinstance(config, list):
            options = config
        else:
            raise TypeError("YaML does not specify a list or dictionary")

        # Populate list with Dimension objects
        dimensions = []
        for option in options:
            key = next(iter(option.keys()))
            # Make configuration case insensitive
            dimension_class = key.lower()
            values = {k.lower(): v for k, v in option[key].items()}
            if dimension_class in dimension_classes:
                # Instantiate Dimension subclass and add it to the list
                dimension = dimension_classes[dimension_class](**values)
                dimensions.append(dimension)

        space = cls(dimensions=dimensions)

        return space

    def rvs(self, n_samples=1, random_state=None, n_jobs=1):
        """Draw random samples.

        The samples are in the original space. They need to be transformed
        before being passed to a model or minimizer by ``space.transform()``.

        Args:
            n_samples (int, default=1): Number of samples to be drawn from the space.
            random_state (int or RandomState, Optional): Set random state to something other than None for reproducible results.
            n_jobs (int): the number of parallel processes to use to perform sampling.

        Returns:
            points (list of lists, shape=(n_points, n_dims)): Points sampled from the space.
        """
        rng = check_random_state(random_state)
        if self.config_space:
            req_points = []

            hps_names = list(self.config_space.keys())

            if self.model_sdv is None:
                confs = self.config_space.sample_configuration(n_samples)

                if n_samples == 1:
                    confs = [confs]
            else:
                with CaptureSTD():
                    confs = self.model_sdv.sample(n_samples)

                sdv_names = confs.columns

                new_hps_names = list(set(hps_names) - set(sdv_names))

                # randomly sample the new hyperparameters
                for name in new_hps_names:
                    hp = self.config_space[name]
                    rvs = []
                    rvs = hp.sample_value(n_samples, rng)
                    confs[name] = rvs

                # reoder the column names
                confs = confs[hps_names]

                confs = confs.to_dict("records")
                for idx, conf in enumerate(confs):
                    cf = deactivate_inactive_hyperparameters(conf, self.config_space)
                    confs[idx] = dict(cf)

            for idx, conf in enumerate(confs):
                point = []
                point_as_dict = dict(conf)
                for i, hps_name in enumerate(hps_names):
                    # If the parameter is inactive due to some conditions then we attribute the
                    # lower bound value to break symmetries and enforce the same representation.
                    if hps_name in point_as_dict:
                        val = conf[hps_name]
                    else:
                        val = self.dimensions[i].bounds[0]
                    point.append(val)
                req_points.append(point)

            return req_points
        else:
            if self.model_sdv is None:
                # Regular sampling without transfer learning from flat search space
                # Joblib parallel optimization
                # Draw

                columns = np.zeros((n_samples, len(self.dimensions)), dtype="O")
                random_states = rng.randint(
                    low=0, high=2**31, size=len(self.dimensions)
                )
                Parallel(n_jobs=n_jobs, verbose=0, require="sharedmem")(
                    delayed(_sample_dimension)(
                        dim,
                        i,
                        n_samples,
                        np.random.RandomState(random_states[i]),
                        [columns],
                    )
                    for i, dim in enumerate(self.dimensions)
                )

                return columns.tolist()
            else:
                with CaptureSTD():
                    confs = self.model_sdv.sample(n_samples)  # sample from SDV

                columns = []
                for dim in self.dimensions:
                    if dim.name in confs.columns:
                        columns.append(confs[dim.name].values.tolist())
                    else:
                        columns.append(dim.rvs(n_samples=n_samples, random_state=rng))

                # Transpose
                return _transpose_list_array(columns)

    def set_transformer(self, transform):
        """Sets the transformer of all dimension objects to ``transform``.

        Args:
            transform (str or list of str): Sets all transformer,, when `transform`  is a string. Otherwise, transform must be a list with strings with the same length as `dimensions`
        """
        # Transform
        for j in range(self.n_dims):
            if isinstance(transform, list):
                self.dimensions[j].set_transformer(transform[j])
            else:
                self.dimensions[j].set_transformer(transform)

    def set_transformer_by_type(self, transform, dim_type):
        """Sets the transformer of ``dim_type`` objects to ``transform``.

        Args:
            transform (str): Sets all transformer of type ``dim_type`` to ``transform``
            dim_type (type): Can be ``deephyper.skopt.space.Real``, ``deephyper.skopt.space.Integer`` or ``deephyper.skopt.space.Categorical``.
        """
        # Transform
        for j in range(self.n_dims):
            if isinstance(self.dimensions[j], dim_type):
                self.dimensions[j].set_transformer(transform)

    def get_transformer(self):
        """Returns all transformers as list."""
        return [self.dimensions[j].transform_ for j in range(self.n_dims)]

    def transform(self, X):
        """Transform samples from the original space into a warped space.

        Note: this transformation is expected to be used to project samples
              into a suitable space for numerical optimization.

        Args:
            X (list of lists, shape=(n_samples, n_dims)): The samples to transform.

        Returns:
            Xt (array of floats, shape=(n_samples, transformed_n_dims)): The transformed samples.
        """
        # Pack by dimension
        columns = [list() for _ in self.dimensions]

        for i in range(len(X)):
            for j in range(self.n_dims):
                columns[j].append(X[i][j])

        # Transform
        for j in range(self.n_dims):
            columns[j] = self.dimensions[j].transform(columns[j])

        # Repack as an array
        Xt = np.hstack([np.asarray(c).reshape((len(X), -1)) for c in columns])

        return Xt

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back to the original space.

        Args:
            Xt (array of floats, shape=(n_samples, transformed_n_dims)): The samples to inverse transform.

        Returns:
            X (list of lists, shape=(n_samples, n_dims)): The original samples.
        """
        # Inverse transform
        columns = []
        start = 0
        Xt = np.asarray(Xt)
        for j in range(self.n_dims):
            dim = self.dimensions[j]
            offset = dim.transformed_size

            if offset == 1:
                columns.append(dim.inverse_transform(Xt[:, start]))
            else:
                columns.append(dim.inverse_transform(Xt[:, start : start + offset]))

            start += offset

        # Transpose
        return _transpose_list_array(columns)

    @property
    def n_dims(self):
        """The dimensionality of the original space."""
        return len(self.dimensions)

    @property
    def transformed_n_dims(self):
        """The dimensionality of the warped space."""
        return sum([dim.transformed_size for dim in self.dimensions])

    @property
    def bounds(self):
        """The dimension bounds, in the original space."""
        b = []

        for dim in self.dimensions:
            if dim.size == 1:
                b.append(dim.bounds)
            else:
                b.extend(dim.bounds)

        return b

    def __contains__(self, point):
        """Check that `point` is within the bounds of the space."""
        for component, dim in zip(point, self.dimensions):
            if component not in dim:
                return False
        return True

    def __getitem__(self, dimension_names):
        """Lookup and return the search-space dimension with the given name.

        This allows for dict-like lookup of dimensions, for example:
        `space['foo']` returns the dimension named 'foo' if it exists,
        otherwise `None` is returned.

        It also allows for lookup of a list of dimension-names, for example:
        `space[['foo', 'bar']]` returns the two dimensions named
        'foo' and 'bar' if they exist.

        Args:
        dimension_names : str or list(str)
            Name of a single search-space dimension (str).
            List of names for search-space dimensions (list(str)).

        Returns:
        dims tuple (index, Dimension), list(tuple(index, Dimension)), \
                (None, None)
            A single search-space dimension with the given name,
            or a list of search-space dimensions with the given names.
        """

        def _get(dimension_name):
            """Helper-function for getting a single dimension."""
            index = 0
            # Get the index of the search-space dimension using its name.
            for dim in self.dimensions:
                if dimension_name == dim.name:
                    return (index, dim)
                elif dimension_name == index:
                    return (index, dim)
                index += 1
            return (None, None)

        if isinstance(dimension_names, (str, int)):
            # Get a single search-space dimension.
            dims = _get(dimension_name=dimension_names)
        elif isinstance(dimension_names, (list, tuple)):
            # Get a list of search-space dimensions.
            # Note that we do not check whether the names are really strings.
            dims = [_get(dimension_name=name) for name in dimension_names]
        else:
            msg = (
                "Dimension name should be either string or"
                "list of strings, but got {}."
            )
            raise ValueError(msg.format(type(dimension_names)))

        return dims

    @property
    def transformed_bounds(self):
        """The dimension bounds, in the warped space."""
        b = []

        for dim in self.dimensions:
            if dim.transformed_size == 1:
                b.append(dim.transformed_bounds)
            else:
                b.extend(dim.transformed_bounds)

        return b

    @property
    def is_categorical(self):
        """Space contains exclusively categorical dimensions."""
        return all([isinstance(dim, Categorical) for dim in self.dimensions])

    @property
    def is_partly_categorical(self):
        """Space contains any categorical dimensions."""
        return any([isinstance(dim, Categorical) for dim in self.dimensions])

    @property
    def n_constant_dimensions(self):
        """Returns the number of constant dimensions which have zero degree of freedom, e.g. an Integer dimensions with (0., 0.) as bounds."""
        n = 0
        for dim in self.dimensions:
            if dim.is_constant:
                n += 1
        return n

    def distance(self, point_a, point_b):
        """Compute distance between two points in this space.

        Args:
        point_a : array
            First point.

        point_b : array
            Second point.
        """
        distance = 0.0
        for a, b, dim in zip(point_a, point_b, self.dimensions):
            distance += dim.distance(a, b)

        return distance

    def update_prior(self, X, y, q=0.9):
        """Update the prior of the dimensions.

        Instead of doing random-sampling, a kernel density estimation is fit on the region of interest and
        sampling is performed from this distribution.
        """
        y = np.array(y)

        for i, dim in enumerate(self.dimensions):
            Xi = [x[i] for x in X]
            if hasattr(dim, "update_prior"):
                dim.update_prior(Xi, y, q=q)

    def deactivate_inactive_dimensions(self, x):
        """When ConfigSpace is used, it will return the "lower" bound of inactive parameters."""
        x = x[:]
        if self.config_space is not None:
            x_dict = {k: v for k, v in zip(self.dimension_names, x)}
            x_dict = dict(
                deactivate_inactive_hyperparameters(x_dict, self.config_space)
            )
            for i, hps_name in enumerate(self.dimension_names):
                # If the parameter is inactive due to some conditions then we attribute the
                # lower bound value to break symmetries and enforce the same representation.
                x[i] = x_dict.get(hps_name, self.dimensions[i].bounds[0])
        return x
