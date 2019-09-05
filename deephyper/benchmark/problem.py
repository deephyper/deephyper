from collections import OrderedDict
from pprint import pformat
import inspect
from inspect import signature

from deephyper.core.exceptions.problem import *


class Problem:
    """Representation of a problem.
    """

    def __init__(self, seed=None, **kwargs):
        self._space = OrderedDict()
        self.seed = seed

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

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
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

    >>> from deephyper.benchmark import NaProblem
    >>> from deephyper.benchmark.nas.linearReg.load_data import load_data
    >>> from deephyper.search.nas.model.baseline.simple import create_search_space
    >>> from deephyper.search.nas.model.preprocessing import minmaxstdscaler
    >>> Problem = NaProblem()
    >>> Problem.load_data(load_data)
    >>> Problem.preprocessing(minmaxstdscaler)
    >>> Problem.search_space(create_search_space)
    >>> Problem.hyperparameters(
    ...     batch_size=100,
    ...     learning_rate=0.1,
    ...     optimizer='adam',
    ...     num_epochs=10,
    ...     callbacks=dict(
    ...        EarlyStopping=dict(
    ...             monitor='val_r2',
    ...             mode='max',
    ...             verbose=0,
    ...             patience=5
    ...         )
    ...     )
    ... )
    >>> Problem.loss('mse')
    >>> Problem.metrics(['r2'])
    >>> Problem.objective('val_r2__last')
    >>> Problem.post_training(
    ...        num_epochs=1000,
    ...        metrics=['r2'],
    ...        callbacks=dict(
    ...         ModelCheckpoint={
    ...             'monitor': 'val_r2',
    ...             'mode': 'max',
    ...             'save_best_only': True,
    ...             'verbose': 1
    ...         },
    ...         EarlyStopping={
    ...             'monitor': 'val_r2',
    ...             'mode': 'max',
    ...             'verbose': 1,
    ...             'patience': 10
    ...         },
    ...         TensorBoard={
    ...             'log_dir':'tb_logs',
    ...             'histogram_freq':1,
    ...             'batch_size':64,
    ...             'write_graph':True,
    ...             'write_grads':True,
    ...             'write_images':True,
    ...             'update_freq':'epoch'
    ...         })
    ... )

    Args:
        regression (bool): if ``True`` the problem is defined as a ``regression`` problem, if ``False`` the problem is defined as a ``classification`` problem.
    """

    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)

        self._space['metrics'] = list()
        self._space['hyperparameters'] = dict(verbose=1)

    def __repr__(self):

        preprocessing = None if self._space.get(
            'preprocessing') is None else module_location(self._space['preprocessing']['func'])

        hps = "".join(
            [f"\n        * {h}: {self._space['hyperparameters'][h]}" for h in self._space['hyperparameters']])

        metrics = "".join(
            [f"\n        * {m}" for m in self._space['metrics']])

        post = None if self._space.get('post_train') is None else "".join(
            [f"\n        * {k}: {self._space['post_train'][k]}" for k in self._space['post_train']])

        objective = self._space['objective']
        if not type(objective) is str:
            objective = module_location(objective)

        out = ( f"Problem is:\n"
                f" * SEED = {self.seed} *\n"
                f"    - search space   : {module_location(self._space['create_search_space']['func'])}\n"
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
            func (callable): an object which has to be a callable and which is returning a ``Structure`` instance.

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

        self.add_dim('create_search_space', {
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
        """Define hyperparameters used to evaluate generated search_spaces.
        """
        if self._space.get('hyperparameters') is None:
            self._space['hyperparameters'] = dict()
        self._space['hyperparameters'].update(kwargs)

    def loss(self, loss):
        """Define the loss used to train generated search_spaces.

        Args:
            loss (str|callable): a string indicating a specific loss function.
        """
        if not type(loss) is str and \
                not callable(loss):
            raise RuntimeError(
                f'The loss should be either a str or a callable when it\'s of type {type(loss)}')

        self._space['loss'] = loss

    def metrics(self, metrics: list):
        """Define a list of metrics for the training of generated search_spaces.

        Args:
            metrics (list(str|callable)): If ``str`` the metric should be defined in Keras or in DeepHyper. If ``callable`` it should take 2 arguments ``(y_pred, y_true)`` which are a prediction and a true value respectively.
        """

        self._space['metrics'] = metrics

    def check_objective(self, objective):

        if not type(objective) is str and not callable(objective):
            raise WrongProblemObjective(objective)
        elif type(objective) is str:
            list_suffix = ['__min', '__max', '__last']
            for suffix in list_suffix:
                if suffix in objective:
                    objective = objective.replace(suffix, '')
                    break # only one suffix autorized
            objective = objective.replace('val_', '')
            possible_names = list()
            for val in ['loss'] + self._space['metrics']:
                if callable(val):
                    possible_names.append(val.__name__)
                else:
                    possible_names.append(val)
            if not(objective in possible_names):
                raise WrongProblemObjective(objective)

    def objective(self, objective):
        """Define the objective you want to maximize for the search.

        Args:
            objective (str|callable): The objective will be maximized. If ``objective`` is ``str`` then it should be either 'loss' or a defined metric. You can use the ``'val_'`` prefix when you want to select the objective on the validation set. You can use one of ``['__min', '__max', '__last']`` which respectively means you want to select the min, max or last value among all epochs. Using '__last' will save a consequent compute time because the evaluation will not compute metrics on validation set for all epochs but the last. If ``objective`` is callable it should return a scalar value (i.e. float) and it will take a ``dict`` parameter. The ``dict`` will contain keys corresponding to loss and metrics such as ``['loss', 'val_loss', 'r2', 'val_r2]``, it will also contains ``'n_parameters'`` which corresponds to the number of trainable parameters of the current model, ``'training_time'`` which corresponds to the time required to train the model. If this callable has a ``'__last'`` suffix then the evaluation will only compute validation loss/metrics for the last epoch. If this callable has contains 'with_pred' in its name then the ``dict`` will have two other keys ``['y_pred', 'y_true']`` where ``'y_pred`` corresponds to prediction of the model on validation set and ``'y_true'`` corresponds to real prediction.
        Raise:
            WrongProblemObjective: raised when the objective is of a wrong definition.
        """
        if not self._space.get('loss') is None and not self._space.get('metrics') is None:
            self.check_objective(objective)
        else:
            raise NaProblemError('.loss and .metrics should be defined before .objective!')

        self._space['objective'] = objective

    def post_training(self, num_epochs: int, metrics: list, callbacks: dict):
        """Choose settings to run a post-training.

        Args:
            num_epochs (int): the number of post-training epochs.
            metrics (list): list of post-training metrics.
            callbacks (dict): dict of ``keras.callbacks`` such as,

                * ModelCheckpoint (dict): ``tensorflow.keras.callbacks.ModelCheckpoint`` settings.

                    * ``'filepath'``: string, path to save the model file.

                    * ``monitor``: quantity to monitor.

                    * ``verbose``: verbosity mode, 0 or 1.

                    * ``save_best_only``: if ``save_best_only=True``, the latest best model according to the quantity monitored will not be overwritten.

                    * ``save_weights_only``: if True, then only the model's weights will be saved (``model.save_weights(filepath)``), else the full model is saved (``model.save(filepath)``).

                    * ``mode``: one of {auto, min, max}. If ``save_best_only=True``, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For ``val_acc``, this should be ``max``, for `val_loss` this should be ``min``, etc. In ``auto`` mode, the direction is automatically inferred from the name of the monitored quantity.

                    * ``period``: Interval (number of epochs) between checkpoints.

                * EarlyStopping (dict): ``tensorflow.keras.callbacks.EarlyStopping`` settings.

                    * ``monitor``: quantity to be monitored.

                    * ``min_delta``: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.

                    * ``patience``: number of epochs with no improvement after which training will be stopped.

                    * ``verbose``: verbosity mode.

                    * ``mode``: one of ``{'auto', 'min', 'max'}``. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.

                    * ``baseline``: Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline. restore_best_weights: whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.

        """
        self._space['post_train'] = {
            'num_epochs': num_epochs,
            'metrics': metrics,
            'callbacks': callbacks
        }

    @property
    def space(self):
        space = super().space
        space['seed'] = self.seed
        return space

def module_location(attr):
    return f'{attr.__module__}.{attr.__name__}'