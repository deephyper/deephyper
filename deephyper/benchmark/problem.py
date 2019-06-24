from collections import OrderedDict
from pprint import pformat
import inspect
from inspect import signature

from deephyper.core.exceptions.problem import *


class Problem:
    """Representation of a problem.

    Attribute:
        space (OrderedDict): represents the search space of the problem.
    """

    def __init__(self):
        self._space = OrderedDict()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'Problem\n{pformat({k:v for k,v in self._space.items()}, indent=2)}'

    def add_dim(self, p_name, p_space):
        """Add a dimension to the search space.

        Args:
            p_name (str): name of the parameter/dimension.
            p_space (Object): space corresponding to the new dimension.
        """
        self._space[p_name] = p_space

    @property
    def space(self):
        dims = list(self._space.keys())
        dims.sort()
        space = OrderedDict(**{d: self._space[d] for d in dims})
        return space


class HpProblem(Problem):
    """Problem specification for Hyperparameter Optimization
    """

    def __init__(self):
        super().__init__()
        # * starting points
        self.references = []

    def __repr__(self):
        prob = super().__repr__()
        start = f'{pformat({k:v for k,v in enumerate(self.starting_point_asdict)})}'
        return prob + '\n\nStarting Point\n' + start

    def add_dim(self, p_name, p_space):
        """Add a dimension to the search space.

        >>> from deephyper.benchmark import HpProblem
        >>> Problem = HpProblem()
        >>> Problem.add_dim('nunits', (10, 20))
        >>> Problem
        Problem
        {'nunits': (10, 20)}
        <BLANKLINE>
        Starting Point
        {}

        Args:
            p_name (str): name of the parameter/dimension.
            p_space (tuple(int, int) or tuple(float, float) or list(Object,)): space corresponding to the new dimension.
        """
        if not type(p_name) is str:
            raise SpaceDimNameOfWrongType(p_name)

        if not type(p_space) is tuple \
                and not type(p_space) is list:
            raise SpaceDimValueOfWrongType(p_space)

        super().add_dim(p_name, p_space)

    def add_starting_point(self, **dims):
        """Add a new starting point to the problem.

        >>> from deephyper.benchmark import HpProblem
        >>> Problem = HpProblem()
        >>> Problem.add_dim('nunits', (10, 20))
        >>> Problem.add_starting_point(nunits=10)
        >>> Problem
        Problem
        {'nunits': (10, 20)}
        <BLANKLINE>
        Starting Point
        {0: {'nunits': 10}}

        Args:
            dims (dict): dictionnary where all keys are dimensions of our search space and the corresponding value is a specific element of this dimension.

        Raises:
            SpaceNumDimMismatch: Raised when 2 set of keys doesn't have the same number of keys for a given Problem.
            SpaceDimNameMismatch: Raised when 2 set of keys are not corresponding for a given Problem.
            SpaceDimValueNotInSpace: Raised when a dimension value of the space is in the coresponding dimension's space.
        """

        if len(dims) != len(self.space):
            raise SpaceNumDimMismatch(dims, self.space)

        if not all(d in self.space for d in dims):
            raise SpaceDimNameMismatch(dims, self.space)

        for dim, value in zip(dims, dims.values()):
            if type(self.space[dim]) is list:
                if not value in self.space[dim]:
                    raise SpaceDimValueNotInSpace(value, dim, self.space[dim])
            else:  # * type(self.space[dim]) is tuple
                if value < self.space[dim][0] \
                        or value > self.space[dim][1]:
                    raise SpaceDimValueNotInSpace(value, dim, self.space[dim])

        self.references.append([dims[d] for d in self.space])

    @property
    def starting_point(self):
        """Starting point(s) of the search space.

        Returns:
            list(list): list of starting points where each point is a list of values. Values are indexed in the same order as the order of creation of space's dimensions.
        """
        return self.references

    @property
    def starting_point_asdict(self):
        """Starting point(s) of the search space.

        Returns:
            list(dict): list of starting points where each point is a dict of values. Each key are correspnding to dimensions of the space.
        """
        return [{k: v for k, v in zip(list(self.space.keys()), p)} for p in self.references]


class NaProblem(Problem):
    """A Neural Architecture Problem specification for Neural Architecture Search.

    Args:
        regression (bool): if ``True`` the problem is defined as a ``regression`` problem, if ``False`` the problem is defined as a ``classification`` problem.
    """

    def __init__(self, regression: bool):
        super().__init__()

        self._space['regression'] = regression
        self._space['metrics'] = list()

    def __repr__(self):
        kind = 'REGRESSION' if self._space['regression'] else 'CLASSIFICATION'

        preprocessing = None if self._space.get(
            'preprocessing') is None else module_location(self._space['preprocessing']['func'])

        hps = "".join(
            [f"\n        * {h}: {self._space['hyperparameters'][h]}" for h in self._space['hyperparameters']])

        metrics = "".join(
            [f"\n        * {m}" for m in self._space['metrics']])

        post = None if self._space.get('posttraining') is None else pformat(
            self._space['posttraining'])

        objective = self._space['objective']
        if not type(objective) is str:
            objective = module_location(objective)

        out = (f"Problem is a {kind}:\n"
               f"    - search space   : {module_location(self._space['create_structure']['func'])}\n"
               f"    - data loading   : {module_location(self._space['load_data']['func'])}\n"
               f"    - preprocessing  : {preprocessing}\n"
               f"    - hyperparameters: {hps}\n"
               f"    - loss           : {self._space['loss']}\n"
               f"    - metrics        : {metrics}\n"
               f"    - objective      : {objective}\n"
               f"    - post-training  : {post}")

        return out

    def load_data(self, func: callable, **kwargs):

        if not callable(func):
            raise ProblemLoadDataIsNotCallable(func)

        self.add_dim('load_data', {
            'func': func,
            'kwargs': kwargs
        })

    def search_space(self, func: callable, **kwargs):
        """Set a search space for neural architecture search.

        Args:
            func (Callable): an object which has to be a callable and which is returning a ``Structure`` instance.

        Raises:
            SearchSpaceBuilderIsNotCallable: raised when the ``func`` parameter is not a callable.
            SearchSpaceBuilderMissingParameter: raised when either of ``(input_shape, output_shape)`` are missing parameters of ``func``.
            SearchSpaceBuilderMissingDefaultParameter: raised when either of ``(input_shape, output_shape)`` are missing a default value.
        """

        if not callable(func):
            raise SearchSpaceBuilderIsNotCallable(func)

        sign_func = signature(func)
        if not 'input_shape' in sign_func.parameters:
            raise SearchSpaceBuilderMissingParameter('input_shape')

        if not 'output_shape' in sign_func.parameters:
            raise SearchSpaceBuilderMissingParameter('output_shape')

        if isinstance(sign_func.parameters['input_shape'].default, inspect._empty):
            raise SearchSpaceBuilderMissingDefaultParameter('input_shape')

        if sign_func.parameters['output_shape'].default is inspect._empty:
            raise SearchSpaceBuilderMissingDefaultParameter('output_shape')

        self.add_dim('create_structure', {
            'func': func,
            'kwargs': kwargs
        })

    def preprocessing(self, func: callable):
        """Define how to preprocess your data.

        Args:
            func (callable): a function which returns a preprocessing scikit-learn ``Pipeline``.
        """

        if not callable(func):
            raise ProblemPreprocessingIsNotCallable(func)

        super().add_dim('preprocessing', {
            'func': func
        })

    def hyperparameters(self, **kwargs):
        """Define hyperparameters used to evaluate generated architectures.
        """
        if self._space.get('hyperparameters') is None:
            self._space['hyperparameters'] = dict()
        self._space['hyperparameters'].update(kwargs)

    def loss(self, loss):
        """Define the loss used to train generated architectures.

        Args:
            loss (str): a string indicating a specific loss function.
        """
        if not type(loss) is str and \
                not callable(loss):
            raise RuntimeError(
                f'The loss should be either a str or a callable when it\'s of type {type(loss)}')

        self._space['loss'] = loss

    def metrics(self, metrics: list):
        """Define a list of metrics for the training of generated architectures.

        Args:
            metrics (list(str|callable)): If ``str`` the metric should be defined in Keras or in DeepHyper. If ``callable`` it should take 2 arguments ``(y_pred, y_true)`` which are a prediction and a true value respectively.
        """

        self._space['metrics'] = metrics

    def objective(self, objective):
        """Define the objective you want to maximize for the search.

        Args:
            objective (str|callable): if (str) then it should be a defined metrics. if (callable) it should return a scalar value (i.e. float) and it will take a dictionnary parameter.
        """
        if not type(objective) is str and \
                not callable(objective):
            raise RuntimeError(
                'The objective should be of type (str|callable) when it\'s of type {type(objective)!}')

        self._space['objective'] = objective

    def post_training(self, num_epochs: int, metrics: list, model_checkpoint, early_stopping):
        self.add_dim('post_train', {
            'num_epochs': num_epochs,
            'metrics': metrics,
            'model_checkpoint': model_checkpoint,
            'early_stopping': early_stopping
        })

def module_location(attr):
    return f'{attr.__module__}.{attr.__name__}'