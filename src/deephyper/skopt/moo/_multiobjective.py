import abc

import numpy as np

from ..utils import is_listlike


class MoScalarFunction(abc.ABC):
    """Abstract class representing a scalarizing function.

    Args:
        n_objectives (int, optional): Number of objective functions. Defaults to ``1``.
        weight (float or 1-D array, optional): Array of weights for each objective function. Defaults to ``None``.
        weight_sampling_periode (int, optional): Sampling periode for the weight vector. Defaults to ``5``.
        utopia_point (float or 1-D array, optional): Array of reference values for each objective function. Defaults to ``None``.
        random_state (int, optional): Random seed. Defaults to ``None``.
    """

    def __init__(
        self,
        n_objectives: int = 1,
        weight=None,
        weight_sampling_periode: int = 1,
        utopia_point=None,
        random_state=None,
    ):
        self._seed = None
        if type(random_state) is int:
            self._seed = random_state
            self._rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self._rng = random_state
        else:
            self._rng = np.random.RandomState()

        if type(n_objectives) is not int:
            raise ValueError("Parameter 'n_objectives' shoud be an integer value!")
        self._n_objectives = n_objectives

        self._utopia_point = None
        if utopia_point is not None:
            self._check_shape(utopia_point)
            self._utopia_point = np.asarray(utopia_point)

        # Record the passed weight vector.
        self.weight = weight
        self.weight_sampling_periode = weight_sampling_periode
        self.counter_weight_sampling = -1
        self._weight = None
        self.update_weight()

    def update_weight(self):
        self.counter_weight_sampling += 1
        if self.counter_weight_sampling % self.weight_sampling_periode != 0:
            return

        if self.weight == "random" or self.weight is None:
            # Uniformly sample from the probability simplex using Remark 1.3
            # from https://arxiv.org/pdf/1909.06406.pdf
            # and the inverse exponential CDF: F_inv(p) = -log(1 - p).
            # Can be checked with plot similar to Fig. 3 in http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
            self._weight = -np.log(1.0 - self._rng.rand(self._n_objectives))
            self._weight /= np.sum(self._weight)
            # self._weight = self._rng.dirichlet([1.0 for _ in range(self._n_objectives)])
        elif self.weight == "uniform":
            self._weight = np.ones(self._n_objectives) / self._n_objectives
        elif is_listlike(self.weight):
            self._check_shape(self.weight)
            self._weight = np.asarray(self.weight)
        else:
            raise ValueError(f"Invalid weight value: {self.weight}")

    def _check_shape(self, y):
        """Check if the shape of y is consistent with the object."""
        if not (
            (np.ndim(y) == 0 and self._n_objectives == 1)
            or (np.ndim(y) == 1 and np.shape(y)[0] == self._n_objectives)
        ):
            raise ValueError(
                f"Expected y to be a scalar or 1-D array of length {self._n_objectives}"
            )

    def scalarize(self, y):
        """Convert the input array (or scalar) into a scalar value.

        Args:
            yi (scalar or 1-D array): The input array or scalar to be scalarized.

        Returns:
            float: The converted scalar value.
        """
        y = np.asarray(y)
        return self._scalarize(y)

    def normalize(self, yi):
        """Compute normalization constants based on the history of evaluated objective values.

        Args:
            yi (array): Array of evaluated objective values.

        Raises:
            ValueError: Raised if yi is not a list of scalars each of length _n_objectives.
        """
        if np.ndim(yi) != 2:
            raise ValueError(f"Expected yi to be a 2D-array but is {yi}!")

        self._utopia_point = np.min(yi, axis=0)

    @abc.abstractmethod
    def _scalarize(self, y):
        """Scalarization to be implemented.

        Args:
            y: Array of length _n_objectives.

        Returns:
            float: Converted scalar value.
        """


class MoLinearFunction(MoScalarFunction):
    """This scalarizing function linearly combines the individual objective values (after automatically scaling them in [0, 1]).

    Args:
        n_objectives (int, optional): Number of objective functions. Defaults to ``1``.
        weight (float or 1-D array, optional): Array of weights for each objective function. Defaults to ``None``.
        weight_sampling_periode (int, optional): Sampling periode for the weight vector. Defaults to ``5``.
        utopia_point (float or 1-D array, optional): Array of reference values for each objective function. Defaults to ``None``.
        random_state (int, optional): Random seed. Defaults to ``None``.
    """

    def __init__(
        self,
        n_objectives: int = 1,
        weight=None,
        weight_sampling_periode: int = 1,
        utopia_point=None,
        random_state=None,
    ):
        super().__init__(
            n_objectives, weight, weight_sampling_periode, utopia_point, random_state
        )

    def _scalarize(self, y):
        return np.dot(self._weight, y)


class MoChebyshevFunction(MoScalarFunction):
    """This scalarizing function computes a weighted infinity-norm of the individual objective values (after automatically scaling them in [0, 1]).

    Args:
        n_objectives (int, optional): Number of objective functions. Defaults to ``1``.
        weight (float or 1-D array, optional): Array of weights for each objective function. Defaults to ``None``.
        weight_sampling_periode (int, optional): Sampling periode for the weight vector. Defaults to ``5``.
        utopia_point (float or 1-D array, optional): Array of reference values for each objective function. Defaults to ``None``.
        random_state (int, optional): Random seed. Defaults to ``None``.
    """

    def __init__(
        self,
        n_objectives: int = 1,
        weight=None,
        weight_sampling_periode: int = 1,
        utopia_point=None,
        random_state=None,
    ):
        super().__init__(
            n_objectives, weight, weight_sampling_periode, utopia_point, random_state
        )

    def _scalarize(self, y):
        return np.max(np.multiply(self._weight, np.abs(y)))


class MoPBIFunction(MoScalarFunction):
    """This scalarizing function computes the projection of the objective vector along a reference vector and adds a penalty term to minimize deviations from the projected point to the attainable objective set. See https://doi.org/10.1109/TEVC.2007.892759

    Args:
        n_objectives (int, optional): Number of objective functions. Defaults to ``1``.
        weight (float or 1-D array, optional): Array of weights for each objective function. Defaults to ```None``.
        weight_sampling_periode (int, optional): Sampling periode for the weight vector. Defaults to ``5``.
        utopia_point (float or 1-D array, optional): Array of reference values for each objective function. Defaults to ``None``.
        random_state (int, optional): Random seed. Defaults to ``None``.
        penalty (float, optional): Value of penalty parameter. Defaults to ``100.0``.
    """

    def __init__(
        self,
        n_objectives: int = 1,
        weight=None,
        weight_sampling_periode: int = 1,
        utopia_point=None,
        random_state=None,
        penalty: float = 5.0,
    ):
        self._penalty = np.abs(penalty) if np.isreal(penalty) else 5.0
        super().__init__(
            n_objectives, weight, weight_sampling_periode, utopia_point, random_state
        )

    def update_weight(self):
        super().update_weight()
        self._weightnorm = np.linalg.norm(self._weight) ** 2

    def _scalarize(self, y):
        d1 = np.dot(self._weight, y) / self._weightnorm
        d2 = np.linalg.norm(y - (d1 * self._weight), 1)
        return d1 + (self._penalty * d2)


class MoAugmentedChebyshevFunction(MoScalarFunction):
    """This scalarizing function computes a sum of weighted infinity- and 1-norms of the individual objective values (after automatically scaling them in [0, 1]).

    Args:
        n_objectives (int, optional): Number of objective functions. Defaults to ``1``.
        weight (float or 1-D array, optional): Array of weights for each objective function. Defaults to ``None``.
        weight_sampling_periode (int, optional): Sampling periode for the weight vector. Defaults to ``5``.
        utopia_point (float or 1-D array, optional): Array of reference values for each objective function. Defaults to ``None``.
        random_state (int, optional): Random seed. Defaults to ``None``.
        penalty (float, optional): Value of weight given to 1-norm. Defaults to ``0.001``.
    """

    def __init__(
        self,
        n_objectives: int = 1,
        weight=None,
        weight_sampling_periode: int = 1,
        utopia_point=None,
        random_state=None,
        alpha: float = 0.001,
    ):
        self._alpha = np.abs(alpha) if np.isreal(alpha) else 0.001
        super().__init__(
            n_objectives, weight, weight_sampling_periode, utopia_point, random_state
        )

    def _scalarize(self, y):
        y = np.multiply(self._weight, np.abs(y))
        return np.max(y) + (self._alpha * np.linalg.norm(y, 1))


class MoQuadraticFunction(MoScalarFunction):
    """This scalarizing function quadratically combines the individual objective values (after automatically scaling them in [0, 1]). It can be interpreted a smoother version of `MoChebyshevFunction`.

    Args:
        n_objectives (int, optional): Number of objective functions. Defaults to ``1``.
        weight (float or 1-D array, optional): Array of weights for each objective function. Defaults to ``None``.
        weight_sampling_periode (int, optional): Sampling periode for the weight vector. Defaults to ``5``.
        utopia_point (float or 1-D array, optional): Array of reference values for each objective function. Defaults to ``None``.
        random_state (int, optional): Random seed. Defaults to ``None``.
        penalty (float, optional): Value of smoothness parameter. Larger values make it less smooth. Defaults to ``10.0``.
    """

    def __init__(
        self,
        n_objectives: int = 1,
        weight=None,
        weight_sampling_periode: int = 1,
        utopia_point=None,
        random_state=None,
        alpha: float = 10.0,
    ):
        self._alpha = np.abs(alpha) if np.isreal(alpha) else 10.0
        super().__init__(
            n_objectives, weight, weight_sampling_periode, utopia_point, random_state
        )

    def update_weight(self):
        super().update_weight()
        U, _, _ = np.linalg.svd(self._weight.reshape(-1, 1), full_matrices=True)
        self._Q = U.dot(
            np.diag([self._alpha if j > 0 else 1.0 for j in range(self._n_objectives)])
        ).dot(U.T)

    def _scalarize(self, y):
        return y.T.dot(self._Q).dot(y)
