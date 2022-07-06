import abc
import warnings
import numpy as np
from numbers import Number
from deephyper.skopt.utils import is_listlike
from deephyper.skopt.utils import is_2Dlistlike

class MoScalarFunction:
    def __init__(
        self,
        n_objectives = 1,
        utopia_point = None,
        random_state = None,
    ):
        self._seed = None
        if type(random_state) is int:
            self._seed = random_state
            self._rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self._rng = random_state
        else:
            self._rng = np.random.RandomState()

        if not (type(n_objectives) is int):
            raise ValueError("Parameter 'n_objectives' shoud be an integer value!")
        self._n_objectives = n_objectives
        self._utopia_point = None
        if utopia_point is not None:
            if is_listlike(utopia_point) or (
                isinstance(utopia_point, np.ndarray)
                and np.ndim(utopia_point) == 1
                and np.shape(utopia_point)[0] == self._n_objectives
            ):
                self._utopia_point = np.asarray(utopia_point)
            else:
                raise ValueError(
                        f"expected utopia_point to be a list or array length {self._n_objectives}"
                    )


    def scalarize(self, y): 
        if not (np.ndim(y) == 1 and np.shape(y)[0] == self._n_objectives):
            raise ValueError(f"expected y to be an array of length {self._n_objectives}")     
        return self._scalarize(y)


    def is_utopia_point_initialized(self):
        return self._utopia_point is not None


    def is_utopia_point_valid(self, y_samples):
        return np.all(self._utopia_point <= np.min(y_samples, axis=0))
        

    def normalize(self, y_samples):
        if not (is_listlike(y_samples) and len(y_samples) > 0):
            raise ValueError(f"expected y_samples to be a non-empty list")
        
        for y in y_samples:
            if not (np.ndim(y) == 1 and np.shape(y)[0] == self._n_objectives):
                raise ValueError(f"expected y to be an array of length {self._n_objectives}")
        
        y_samples = np.asarray(y_samples)
        y_max = np.max(y_samples, axis=0)
        y_min = np.min(y_samples, axis=0)
        self._utopia_point = y_min - 100.0*(y_max - y_min)
        

    @abc.abstractmethod
    def _scalarize(self, yi):
        """Scalarization to be implemented.

        Args:
            yi: vector of length _n_objectives
        
        Returns:
            f: scalar
        """


class MoLinearFunction(MoScalarFunction):
    def __init__(
        self,
        n_objectives = 1,
        utopia_point = None,
        random_state = None,
    ):
        super().__init__(n_objectives, utopia_point, random_state)
        self._weight = self._rng.rand(self._n_objectives)
        self._weight /= np.sum(self._weight)
    

    def _scalarize(self, yi):
        return np.dot(self._weight, yi)

class MoChebyshevFunction(MoScalarFunction):
    def __init__(
        self,
        n_objectives = 1,
        utopia_point = None,
        random_state = None,
    ):
        super().__init__(n_objectives, utopia_point, random_state)
        self._weight = self._rng.rand(self._n_objectives)
        # self._weight /= np.sum(self._weight)
    
    def _scalarize(self, yi):
        if self._utopia_point is None:
            raise ValueError(f"expected _utopia_point to be an array of length _n_objectives")

        yi = yi - self._utopia_point
        if np.any(yi < 0):
            print(
                "encountered objective value smaller than estimated utopia point!"
            )
        return np.max(self._weight * np.abs(yi))

class MoPBIFunction(MoScalarFunction):
    def __init__(
        self,
        n_objectives = 1,
        utopia_point = None,
        random_state = None,
        penalty: float = 1e4,
    ):
        super().__init__(n_objectives, utopia_point, random_state)
        self._weight = self._rng.rand(self._n_objectives)
        self._weightnorm = np.linalg.norm(self._weight)**2
        # self._weight /= np.sum(self._weight)
        if np.isreal(penalty):
            self._penalty = np.abs(penalty)
    
    def _scalarize(self, yi):
        # print(f"{self._weight=}, {self._utopia_point=}")
        if self._utopia_point is None:
            raise ValueError(f"expected _utopia_point to be an array of length _n_objectives")

        yi = yi - self._utopia_point
        if np.any(yi < 0):
            print(
                "encountered objective value smaller than estimated utopia point!"
            )
        d1 = np.dot(self._weight, yi)/self._weightnorm
        d2 = np.linalg.norm(yi - (d1 * self._weight), 1)
        return d1 + (self._penalty*d2)