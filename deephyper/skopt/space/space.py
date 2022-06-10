import numbers
import numpy as np
import yaml
import sys

from scipy.stats.distributions import randint
from scipy.stats.distributions import rv_discrete
from scipy.stats.distributions import uniform, truncnorm

from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version


if type(sp_version) is not tuple:  # Version object since sklearn>=2.3.x
    if hasattr(sp_version, "release"):
        sp_version = sp_version.release
    else:
        sp_version = sp_version._version.release


from .transformers import CategoricalEncoder
from .transformers import StringEncoder
from .transformers import LabelEncoder
from .transformers import Normalize
from .transformers import Identity
from .transformers import LogN
from .transformers import Pipeline
from .transformers import ToInteger


import ConfigSpace as CS
from ConfigSpace.util import deactivate_inactive_hyperparameters

from sklearn.impute import SimpleImputer


# helper class to be able to print [1, ..., 4] instead of [1, '...', 4]
class _Ellipsis:
    def __repr__(self):
        return "..."


def _transpose_list_array(x):
    """Transposes a list matrix"""

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

    Parameters
    ----------
    dimension : Dimension
        Search space Dimension.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    transform : "identity", "normalize", "string", "label", "onehot" optional
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

    Returns
    -------
    dimension : Dimension
        Dimension instance.
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
        "Invalid dimension {}. Read the documentation for "
        "supported types.".format(dimension)
    )


class Dimension(object):
    """Base class for search space dimensions."""

    prior = None

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        Parameters
        ----------
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
        """Inverse transform samples from the warped space back into the
        original space.
        """
        return self.transformer.inverse_transform(Xt)

    def set_transformer(self):
        raise NotImplementedError

    @property
    def size(self):
        return 1

    @property
    def transformed_size(self):
        return 1

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def is_constant(self):
        raise NotImplementedError

    @property
    def transformed_bounds(self):
        raise NotImplementedError

    @property
    def name(self):
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
    return uniform(loc=loc, scale=np.nextafter(scale, scale + 1.0))


def _normal_inclusive(loc=0.0, scale=1.0, lower=-2, upper=2):
    assert lower <= upper
    a, b = (lower - loc) / scale, (upper - loc) / scale
    return truncnorm(a, b, loc=loc, scale=scale)


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
                "the lower bound {} has to be less than the"
                " upper bound {}".format(low, high)
            )
        if prior not in ["uniform", "log-uniform", "normal"]:
            raise ValueError(
                "prior should be 'normal', 'uniform' or 'log-uniform'"
                " got {}".format(prior)
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

        Parameters
        ----------
        transform : str
           Can be 'normalize' or 'identity'

        """
        self.transform_ = transform

        if self.transform_ not in ["normalize", "identity"]:
            raise ValueError(
                "transform should be 'normalize' or 'identity'"
                " got {}".format(self.transform_)
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
        return "Real(low={}, high={}, prior='{}', transform='{}', loc='{}', scale='{}')".format(
            self.low, self.high, self.prior, self.transform_, self.loc, self.scale
        )

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
        original space.
        """
        inv_transform = super(Real, self).inverse_transform(Xt)
        if isinstance(inv_transform, list):
            inv_transform = np.array(inv_transform)

        # PB commenting clip
        # print(inv_transform)
        # inv_transform = np.clip(inv_transform, self.low, self.high).astype(self.dtype)

        if self.dtype == float or self.dtype == "float":
            # necessary, otherwise the type is converted to a numpy type
            return getattr(inv_transform, "tolist", lambda: value)()
        else:
            return inv_transform

    @property
    def bounds(self):
        return (self.low, self.high)

    @property
    def is_constant(self):
        return self.low == self.high

    def __contains__(self, point):
        if isinstance(point, list):
            point = np.array(point)
        if point == np.nan:
            return True
        else:
            return self.low <= point <= self.high

    @property
    def transformed_bounds(self):
        if self.transform_ == "normalize":
            return 0.0, 1.0
        else:
            if self.prior == "uniform":
                return self.low, self.high
            else:
                return np.log10(self.low), np.log10(self.high)

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Parameters
        ----------
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


class Integer(Dimension):
    """Search space dimension that can take on integer values.

    Parameters
    ----------
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
                "the lower bound {} has to be less than the"
                " upper bound {}".format(low, high)
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

        Parameters
        ----------
        transform : str
           Can be 'normalize' or 'identity'

        """
        self.transform_ = transform

        if transform not in ["normalize", "identity"]:
            raise ValueError(
                "transform should be 'normalize' or 'identity'"
                " got {}".format(self.transform_)
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
                self._rvs = randint(self.low, self.high + 1)
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
        return (
            type(self) is type(other)
            and np.allclose([self.low], [other.low])
            and np.allclose([self.high], [other.high])
        )

    def __repr__(self):
        return "Integer(low={}, high={}, prior='{}', transform='{}')".format(
            self.low, self.high, self.prior, self.transform_
        )

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
        original space.
        """
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
            return getattr(
                np.round(inv_transform).astype(self.dtype), "tolist", lambda: value
            )()
        else:
            return np.round(inv_transform).astype(self.dtype)

    @property
    def bounds(self):
        return (self.low, self.high)

    @property
    def is_constant(self):
        return self.low == self.high

    def __contains__(self, point):
        if isinstance(point, list):
            point = np.array(point)
        if point == np.nan:
            return True
        else:
            return self.low <= point <= self.high

    @property
    def transformed_bounds(self):
        if self.transform_ == "normalize":
            return 0.0, 1.0
        else:
            if self.prior == "uniform":
                return self.low, self.high
            else:
                return np.log10(self.low), np.log10(self.high)

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Parameters
        ----------
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

    Parameters
    ----------
    categories : list, shape=(n_categories,)
        Sequence of possible categories.

    prior : list, shape=(categories,), default=None
        Prior probabilities for each category. By default all categories
        are equally likely.

    transform : "onehot", "string", "identity", "label", default="onehot"

        - "identity", the transformed space is the same as the original
          space.
        - "string",  the transformed space is a string encoded
          representation of the original space.
        - "label", the transformed space is a label encoded
          representation (integer) of the original space.
        - "onehot", the transformed space is a one-hot encoded
          representation of the original space.

    name : str or None
        Name associated with dimension, e.g., "colors".

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

        Parameters
        ----------
        transform : str
           Can be 'normalize', 'onehot', 'string', 'label', or 'identity'

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
            self.transformer = Identity()
            self.transformer.fit(self.categories)
        if transform == "normalize":
            self._rvs = _uniform_inclusive(0.0, 1.0)
        else:
            # XXX check that sum(prior) == 1
            self._rvs = rv_discrete(values=(range(len(self.categories)), self.prior_))

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.categories == other.categories
            and np.allclose(self.prior_, other.prior_)
        )

    def __repr__(self):
        if len(self.categories) > 7:
            cats = self.categories[:3] + (_Ellipsis(),) + self.categories[-3:]
        else:
            cats = self.categories

        if self.prior is not None and len(self.prior) > 7:
            prior = self.prior[:3] + [_Ellipsis()] + self.prior[-3:]
        else:
            prior = self.prior

        return "Categorical(categories={}, prior={})".format(cats, prior)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
        original space.
        """
        # The concatenation of all transformed dimensions makes Xt to be
        # of type float, hence the required cast back to int.
        inv_transform = super(Categorical, self).inverse_transform(Xt)
        if isinstance(inv_transform, list):
            inv_transform = np.array(inv_transform)
        return inv_transform

    def rvs(self, n_samples=None, random_state=None):
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
        if self.transform_ == "onehot":
            size = len(self.categories)
            # when len(categories) == 2, CategoricalEncoder outputs a
            # single value
            return size if size != 2 else 1
        return 1

    @property
    def bounds(self):
        return self.categories

    @property
    def is_constant(self):
        return len(self.categories) <= 1

    def __contains__(self, point):
        return point in self.categories

    @property
    def transformed_bounds(self):
        if self.transformed_size == 1:
            N = len(self.categories)
            if self.transform_ == "label":
                return 0.0, float(N - 1)
            else:
                return 0.0, 1.0
        else:
            return [(0.0, 1.0) for i in range(self.transformed_size)]

    def distance(self, a, b):
        """Compute distance between category `a` and `b`.

        As categories have no order the distance between two points is one
        if a != b and zero otherwise.

        Parameters
        ----------
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


class Space(object):
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

    def __init__(self, dimensions, model_sdv=None):

        # attributes used when a ConfigurationSpace from ConfigSpace is given
        self.is_config_space = False
        self.config_space_samples = None
        self.config_space_explored = False

        self.imp_const = SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value=-1000
        )
        self.imp_const_inv = SimpleImputer(
            missing_values=-1000, strategy="constant", fill_value=np.nan
        )

        # attribute used when a generative model is used to sample
        self.model_sdv = model_sdv

        self.hps_names = []

        if isinstance(dimensions, CS.ConfigurationSpace):
            self.is_config_space = True
            self.config_space = dimensions
            self.hps_type = {}

            hps = self.config_space.get_hyperparameters()
            cond_hps = self.config_space.get_all_conditional_hyperparameters()

            space = []
            for x in hps:
                self.hps_names.append(x.name)
                if isinstance(x, CS.hyperparameters.CategoricalHyperparameter):
                    categories = list(x.choices)
                    prior = list(x.probabilities)
                    if x.name in cond_hps:
                        categories.append("NA")

                        # remove p from prior
                        p = 1 / len(categories)
                        pi = p / (len(categories) - 1)
                        prior = [prior_i - pi for prior_i in prior]
                        prior.append(p)
                    param = Categorical(categories, prior=prior, name=x.name)
                    space.append(param)
                    self.hps_type[x.name] = "Categorical"
                elif isinstance(x, CS.hyperparameters.OrdinalHyperparameter):
                    vals = list(x.sequence)
                    if x.name in cond_hps:
                        vals.append("NA")
                    param = Categorical(vals, name=x.name)
                    space.append(param)
                    self.hps_type[x.name] = "Categorical"
                elif isinstance(x, CS.hyperparameters.UniformIntegerHyperparameter):
                    prior = "uniform"
                    if x.log:
                        prior = "log-uniform"
                    param = Integer(x.lower, x.upper, prior=prior, name=x.name)
                    space.append(param)
                    self.hps_type[x.name] = "Integer"
                elif isinstance(x, CS.hyperparameters.NormalIntegerHyperparameter):
                    # TODO
                    prior = "uniform"
                    if x.log:
                        prior = "log-uniform"
                    param = Integer(x.lower, x.upper, prior=prior, name=x.name)
                    space.append(param)
                    self.hps_type[x.name] = "Integer"
                    # raise ValueError("NormalIntegerHyperparameter not implemented")
                elif isinstance(x, CS.hyperparameters.UniformFloatHyperparameter):
                    prior = "uniform"
                    if x.log:
                        prior = "log-uniform"
                    param = Real(x.lower, x.upper, prior=prior, name=x.name)
                    space.append(param)
                    self.hps_type[x.name] = "Real"
                elif isinstance(x, CS.hyperparameters.NormalFloatHyperparameter):
                    prior = "normal"
                    if x.log:
                        raise ValueError(
                            "Unsupported 'log' transformation for NormalFloatHyperparameter."
                        )
                    param = Real(
                        x.lower,
                        x.upper,
                        prior=prior,
                        name=x.name,
                        loc=x.mu,
                        scale=x.sigma,
                    )
                    space.append(param)
                    self.hps_type[x.name] = "Real"
                else:
                    raise ValueError("Unknown Hyperparameter type.")
            dimensions = space
        self.dimensions = [check_dimension(dim) for dim in dimensions]

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.dimensions, other.dimensions)])

    def __repr__(self):
        if len(self.dimensions) > 31:
            dims = self.dimensions[:15] + [_Ellipsis()] + self.dimensions[-15:]
        else:
            dims = self.dimensions
        return "Space([{}])".format(",\n       ".join(map(str, dims)))

    def __iter__(self):
        return iter(self.dimensions)

    @property
    def dimension_names(self):
        """
        Names of all the dimensions in the search-space.
        """
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
        """
        Returns true if all dimensions are Real
        """
        return all([isinstance(dim, Real) for dim in self.dimensions])

    @classmethod
    def from_yaml(cls, yml_path, namespace=None):
        """Create Space from yaml configuration file

        Parameters
        ----------
        yml_path : str
            Full path to yaml configuration file, example YaML below:
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

        Returns
        -------
        space : Space
           Instantiated Space object

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

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        The samples are in the original space. They need to be transformed
        before being passed to a model or minimizer by `space.transform()`.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to be drawn from the space.

        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        points : list of lists, shape=(n_points, n_dims)
           Points sampled from the space.
        """

        rng = check_random_state(random_state)
        if self.is_config_space:
            req_points = []

            hps_names = self.config_space.get_hyperparameter_names()

            if self.model_sdv is None:
                confs = self.config_space.sample_configuration(n_samples)

                if n_samples == 1:
                    confs = [confs]
            else:
                confs = self.model_sdv.sample(n_samples)

                sdv_names = confs.columns

                new_hps_names = list(set(hps_names) - set(sdv_names))

                # randomly sample the new hyperparameters
                for name in new_hps_names:
                    hp = self.config_space.get_hyperparameter(name)
                    rvs = []
                    for i in range(n_samples):
                        v = hp._sample(rng)
                        rv = hp._transform(v)
                        rvs.append(rv)
                    confs[name] = rvs

                # reoder the column names
                confs = confs[hps_names]

                confs = confs.to_dict("records")
                for idx, conf in enumerate(confs):
                    cf = deactivate_inactive_hyperparameters(conf, self.config_space)
                    confs[idx] = cf.get_dictionary()

                    # TODO: remove because debug instructions
                    # check if other conditions are not met; generate valid 1-exchange neighbor; need to test and develop the logic
                    # print('conf invalid...generating valid 1-exchange neighbor')
                    # neighborhood = get_one_exchange_neighbourhood(cf,1)
                    # for new_config in neighborhood:
                    #     print(new_config)
                    #     print(new_config.is_valid_configuration())
                    #     confs[idx] = new_config.get_dictionary()

            for idx, conf in enumerate(confs):
                point = []
                for hps_name in hps_names:
                    val = np.nan
                    if self.hps_type[hps_name] == "Categorical":
                        val = "NA"
                    if hps_name in conf.keys():
                        val = conf[hps_name]
                    point.append(val)
                req_points.append(point)

            return req_points
        else:
            if self.model_sdv is None:
                # Draw
                columns = []
                for dim in self.dimensions:
                    columns.append(dim.rvs(n_samples=n_samples, random_state=rng))

                # Transpose
                return _transpose_list_array(columns)
            else:
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
        """Sets the transformer of all dimension objects to `transform`

        Parameters
        ----------
        transform : str or list of str
           Sets all transformer,, when `transform`  is a string.
           Otherwise, transform must be a list with strings with
           the same length as `dimensions`
        """
        # Transform
        for j in range(self.n_dims):
            if isinstance(transform, list):
                self.dimensions[j].set_transformer(transform[j])
            else:
                self.dimensions[j].set_transformer(transform)

    def set_transformer_by_type(self, transform, dim_type):
        """Sets the transformer of `dim_type` objects to `transform`

        Parameters
        ----------
        transform : str
           Sets all transformer of type `dim_type` to `transform`
        dim_type : type
            Can be `deephyper.skopt.space.Real`, `deephyper.skopt.space.Integer` or
             `deephyper.skopt.space.Categorical`
        """
        # Transform
        for j in range(self.n_dims):
            if isinstance(self.dimensions[j], dim_type):
                self.dimensions[j].set_transformer(transform)

    def get_transformer(self):
        """Returns all transformers as list"""
        return [self.dimensions[j].transform_ for j in range(self.n_dims)]

    def transform(self, X):
        """Transform samples from the original space into a warped space.

        Note: this transformation is expected to be used to project samples
              into a suitable space for numerical optimization.

        Parameters
        ----------
        X : list of lists, shape=(n_samples, n_dims)
            The samples to transform.

        Returns
        -------
        Xt : array of floats, shape=(n_samples, transformed_n_dims)
            The transformed samples.
        """
        # Pack by dimension
        columns = []
        for dim in self.dimensions:
            columns.append([])

        for i in range(len(X)):
            for j in range(self.n_dims):
                columns[j].append(X[i][j])

        # Transform
        for j in range(self.n_dims):
            columns[j] = self.dimensions[j].transform(columns[j])

        # Repack as an array
        Xt = np.hstack([np.asarray(c).reshape((len(X), -1)) for c in columns])

        if False and self.is_config_space:
            self.imp_const.fit(Xt)
            Xtt = self.imp_const.transform(Xt)
            Xt = Xtt

        return Xt

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back to the
           original space.

        Parameters
        ----------
        Xt : array of floats, shape=(n_samples, transformed_n_dims)
            The samples to inverse transform.

        Returns
        -------
        X : list of lists, shape=(n_samples, n_dims)
            The original samples.
        """

        Xt = self.imp_const_inv.fit_transform(Xt)

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
        """
        Lookup and return the search-space dimension with the given name.

        This allows for dict-like lookup of dimensions, for example:
        `space['foo']` returns the dimension named 'foo' if it exists,
        otherwise `None` is returned.

        It also allows for lookup of a list of dimension-names, for example:
        `space[['foo', 'bar']]` returns the two dimensions named
        'foo' and 'bar' if they exist.

        Parameters
        ----------
        dimension_names : str or list(str)
            Name of a single search-space dimension (str).
            List of names for search-space dimensions (list(str)).

        Returns
        -------
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
        """Space contains exclusively categorical dimensions"""
        return all([isinstance(dim, Categorical) for dim in self.dimensions])

    @property
    def is_partly_categorical(self):
        """Space contains any categorical dimensions"""
        return any([isinstance(dim, Categorical) for dim in self.dimensions])

    @property
    def n_constant_dimensions(self):
        """Returns the number of constant dimensions which have zero degree of
        freedom, e.g. an Integer dimensions with (0., 0.) as bounds.
        """
        n = 0
        for dim in self.dimensions:
            if dim.is_constant:
                n += 1
        return n

    def distance(self, point_a, point_b):
        """Compute distance between two points in this space.

        Parameters
        ----------
        point_a : array
            First point.

        point_b : array
            Second point.
        """
        distance = 0.0
        for a, b, dim in zip(point_a, point_b, self.dimensions):
            distance += dim.distance(a, b)

        return distance
