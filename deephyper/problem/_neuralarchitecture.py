from collections import OrderedDict
from copy import deepcopy
from inspect import signature

import ConfigSpace.hyperparameters as csh
import tensorflow as tf
from deephyper.core.exceptions.problem import (
    NaProblemError,
    ProblemLoadDataIsNotCallable,
    ProblemPreprocessingIsNotCallable,
    SearchSpaceBuilderIsNotCallable,
    SearchSpaceBuilderMissingDefaultParameter,
    SearchSpaceBuilderMissingParameter,
    WrongProblemObjective,
)
from deephyper.nas.run._util import get_search_space, setup_data
from deephyper.problem import HpProblem


class NaProblem:
    """A Neural Architecture Problem specification for Neural Architecture Search.

    >>> from deephyper.problem import NaProblem
    >>> from deephyper.benchmark.nas.linearReg.load_data import load_data
    >>> from deephyper.nas.preprocessing import minmaxstdscaler
    >>> from deepspace.tabular import OneLayerSpace
    >>> Problem = NaProblem()
    >>> Problem.load_data(load_data)
    >>> Problem.preprocessing(minmaxstdscaler)
    >>> Problem.search_space(OneLayerSpace)
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
    """

    def __init__(self):
        self._space = OrderedDict()
        self._hp_space = HpProblem()
        self._space["metrics"] = []
        self._space["hyperparameters"] = dict(verbose=0)

    def __repr__(self):

        preprocessing = (
            None
            if self._space.get("preprocessing") is None
            else module_location(self._space["preprocessing"]["func"])
        )

        hps = "".join(
            [
                f"\n        * {h}: {self._space['hyperparameters'][h]}"
                for h in self._space["hyperparameters"]
            ]
        )

        if type(self._space["metrics"]) is list:
            metrics = "".join([f"\n        * {m}" for m in self._space["metrics"]])
        else:
            metrics = "".join(
                [f"\n        * {m[0]}: {m[1]}" for m in self._space["metrics"].items()]
            )

        objective = self._space["objective"]
        if not type(objective) is str:
            objective = module_location(objective)

        out = (
            f"Problem is:\n"
            f"    - search space   : {module_location(self._space['search_space']['class'])}\n"
            f"    - data loading   : {module_location(self._space['load_data']['func'])}\n"
            f"    - preprocessing  : {preprocessing}\n"
            f"    - hyperparameters: {hps}\n"
            f"    - loss           : {self._space['loss']}\n"
            f"    - metrics        : {metrics}\n"
            f"    - objective      : {objective}\n"
        )

        return out

    def load_data(self, func: callable, **kwargs):
        """Define the function loading the data.
        .. code-block:: python

            Problem.load_data(load_data, load_data_kwargs)

        This ``load_data`` callable can follow two different interfaces: Numpy arrays or generators.

        1. **Numpy arrays**:

        In the case of Numpy arrays, the callable passed to ``Problem.load_data(...)`` has to return the following tuple: ``(X_train, y_train), (X_valid, y_valid)``. In the most simple case where the model takes a single input, each of these elements is a Numpy array. Generally, ``X_train`` and ``y_train`` have to be of the same length (i.e., same ``array.shape[0]``) which is also the case for ``X_valid`` and ``y_valid``. Similarly, the shape of the elements of ``X_train`` and ``X_valid`` which is also the case for ``y_train`` and ``y_valid``. An example ``load_data`` function can be

        .. code-block:: python

            import numpy as np

            def load_data(N=100):
                X = np.zeros((N, 1))
                y = np.zeros((N,1))
                return (X, y), (X, y)


        It is also possible for the model to take several inputs. In fact, experimentaly it can be notices that separating some inputs with different inputs can significantly help the learning of the model. Also, sometimes different inputs may be of the "types" for example two molecular fingerprints. In this case, it can be very interesting to share the weights of the model to process these two inputs. In the case of multi-inputs models the ``load_data`` function will also return ``(X_train, y_train), (X_valid, y_valid)`` bu where ``X_train`` and ``X_valid`` are two lists of Numpy arrays. For example, the following is correct:

        .. code-block:: python

            import numpy as np

            def load_data(N=100):
                X = np.zeros((N, 1))
                y = np.zeros((N,1))
                return ([X, X], y), ([X, X], y)


        2. **Generators**:

        Returning generators with a single input:

        .. code-block:: python

            def load_data(N=100):

                tX, ty = np.zeros((N,1)), np.zeros((N,1))
                vX, vy = np.zeros((N,1)), np.zeros((N,1))

                def train_gen():
                    for x, y in zip(tX, ty):
                        yield ({"input_0": x}, y)

                def valid_gen():
                    for x, y in zip(vX, vy):
                        yield ({"input_0": x}, y)

                res = {
                    "train_gen": train_gen,
                    "train_size": N,
                    "valid_gen": valid_gen,
                    "valid_size": N,
                    "types": ({"input_0": tf.float64}, tf.float64),
                    "shapes": ({"input_0": (1, )}, (1, ))
                    }
                return res

        Returning generators with multiple inputs:

        .. code-block:: python

            def load_data(N=100):

                tX0, tX1, ty = np.zeros((N,1)), np.zeros((N,1)), np.zeros((N,1)),
                vX0, vX1, vy = np.zeros((N,1)), np.zeros((N,1)), np.zeros((N,1)),

                def train_gen():
                    for x0, x1, y in zip(tX0, tX1, ty):
                        yield ({
                            "input_0": x0,
                            "input_1": x1
                            }, y)

                def valid_gen():
                    for x0, x1, y in zip(vX0, vX1, vy):
                        yield ({
                            "input_0": x0,
                            "input_1": x1
                        }, y)

                res = {
                    "train_gen": train_gen,
                    "train_size": N,
                    "valid_gen": valid_gen,
                    "valid_size": N,
                    "types": ({"input_0": tf.float64, "input_1": tf.float64}, tf.float64),
                    "shapes": ({"input_0": (5, ), "input_1": (5, )}, (1, ))
                    }

                return res

        Args:
            func (callable): the load data function.
        """

        if not callable(func):
            raise ProblemLoadDataIsNotCallable(func)

        self._space["load_data"] = {"func": func, "kwargs": kwargs}

    def augment(self, func: callable, **kwargs):
        """
        :meta private:
        """

        if not callable(func):
            raise ProblemLoadDataIsNotCallable(func)

        self._space["augment"] = {"func": func, "kwargs": kwargs}

    def search_space(self, space_class, **kwargs):
        """Set a search space for neural architecture search.

        Args:
            space_class (KSearchSpace): an object of type ``KSearchSpace`` which has to implement the ``build()`` method.

        Raises:
            SearchSpaceBuilderMissingParameter: raised when either of ``(input_shape, output_shape)`` are missing parameters of ``func``.
        """

        sign = signature(space_class)
        if not "input_shape" in sign.parameters:
            raise SearchSpaceBuilderMissingParameter("input_shape")

        if not "output_shape" in sign.parameters:
            raise SearchSpaceBuilderMissingParameter("output_shape")

        self._space["search_space"] = {"class": space_class, "kwargs": kwargs}

    def add_hyperparameter(
        self, value, name: str = None, default_value=None
    ) -> csh.Hyperparameter:
        """Add hyperparameters to search the neural architecture search problem.

        >>> Problem.hyperparameters(
        ...     batch_size=problem.add_hyperparameter((32, 256), "batch_size")
        ...     )

        Args:
            value: a hyperparameter description.
            name: a name of the defined hyperparameter, the same as the current key.
            default_value (Optional): a default value of the hyperparameter.

        Returns:
            Hyperparameter: the defined hyperparameter.
        """
        return self._hp_space.add_hyperparameter(value, name, default_value)

    def preprocessing(self, func: callable):
        """Define how to preprocess your data.

        Args:
            func (callable): a function which returns a preprocessing scikit-learn ``Pipeline``.
        """

        if not callable(func):
            raise ProblemPreprocessingIsNotCallable(func)

        self._space["preprocessing"] = {"func": func}

    def hyperparameters(self, **kwargs):
        """Define hyperparameters used to evaluate generated architectures.

        Hyperparameters can be defined such as:

        .. code-block:: python

            Problem.hyperparameters(
                batch_size=256,
                learning_rate=0.01,
                optimizer="adam",
                num_epochs=20,
                verbose=0,
                callbacks=dict(...),
            )
        """
        if self._space.get("hyperparameters") is None:
            self._space["hyperparameters"] = dict()
        self._space["hyperparameters"].update(kwargs)

    def loss(self, loss, loss_weights=None, class_weights=None):
        """Define the loss used to train generated architectures.

        It can be a ``str`` corresponding to a Keras loss function:

        .. code-block:: python

            problem.loss("categorical_crossentropy")

        A custom loss function can also be defined:

        .. code-block:: python

            def NLL(y, rv_y):
                return -rv_y.log_prob(y)

            problem.loss(NLL)

        The loss can be automatically searched:

        .. code-block:: python

            problem.loss(
                problem.add_hyperparameter(
                    ["mae", "mse", "huber_loss", "log_cosh", "mape", "msle"], "loss"
                )
            )

        It is possible to define a different loss for each output:

        .. code-block:: python

            problem.loss(
                loss={"output_0": "mse", "output_1": "mse"},
                loss_weights={"output_0": 0.0, "output_1": 1.0},
            )

        Args:
            loss (str or callable orlist): a string indicating a specific loss function.
            loss_weights (list): Optional.
            class_weights (dict): Optional.
        """
        if not (type(loss) is csh.CategoricalHyperparameter):
            if not type(loss) is str and not callable(loss) and not type(loss) is dict:
                raise RuntimeError(
                    f"The loss should be either a str, dict or a callable when it's of type {type(loss)}"
                )

            if (
                type(loss) is dict
                and loss_weights is not None
                and len(loss) != len(loss_weights)
            ):
                raise RuntimeError(
                    f"The losses list (len={len(loss)}) and the weights list (len={len(loss_weights)}) should be of same length!"
                )

        self._space["loss"] = loss
        if loss_weights is not None:
            self._space["loss_weights"] = loss_weights

        if class_weights is not None:
            self._space["class_weights"] = class_weights

    def metrics(self, metrics=None):
        """Define a list of metrics for the training of generated architectures.

        A list of metrics can be defined to be monitored or used as an objective. It can be a keyword or a callable. For example, if it is a keyword:

        .. code-block:: python

            problem.metrics(["acc"])

        In case you need multiple metrics:

        .. code-block:: python

            problem.metrics["mae", "mse"]

        In case you want to use a custom metric:

        .. code-block:: python

            def sparse_perplexity(y_true, y_pred):
                cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                perplexity = tf.pow(2.0, cross_entropy)
                return perplexity

            problem.metrics([sparse_perplexity])

        Args:
            metrics (list(str or callable) or dict): If ``str`` the metric should be defined in Keras or in DeepHyper. If ``callable`` it should take 2 arguments ``(y_pred, y_true)`` which are a prediction and a true value respectively.
        """

        if metrics is None:
            metrics = []

        self._space["metrics"] = metrics

    def check_objective(self, objective):
        """
        :meta private:
        """

        if not type(objective) is str and not callable(objective):
            raise WrongProblemObjective(objective)

        # elif type(objective) is str:
        #     list_suffix = ["__min", "__max", "__last"]

        #     for suffix in list_suffix:
        #         if suffix in objective:
        #             objective = objective.replace(suffix, "")
        #             break  # only one suffix autorized

        #     objective = objective.replace("val_", "")
        #     possible_names = list()

        #     if type(self._space["metrics"]) is dict:
        #         metrics = list(self._space["metrics"].values())
        #         for k in self._space["metrics"].keys():
        #             objective = objective.replace(f"{k}_", "")

        #     else:  # assuming it s a list
        #         metrics = self._space["metrics"]

        #     for val in ["loss"] + metrics:
        #         if callable(val):
        #             possible_names.append(val.__name__)
        #         else:
        #             possible_names.append(val)

        #     if not (objective in possible_names):
        #         raise WrongProblemObjective(objective, possible_names)

    def objective(self, objective):
        """Define the objective you want to maximize for the search.

        If you want to use the validation accuracy at the last epoch:

        .. code-block:: python

            problem.objective("val_acc")

        .. note:: Be sure to define ``acc`` in the ``problem.metrics(["acc"])``.

        It can accept some prefix and suffix such as ``__min, __max, __last``:

        .. code-block:: python

            problem.objective("-val_acc__max")

        It can be a ``callable``:

        .. code-block:: python

            def myobjective(history: dict) -> float:
                return history["val_acc"][-1]

            problem.objective(myobjective)


        Args:
            objective (str or callable): The objective will be maximized. If ``objective`` is ``str`` then it should be either 'loss' or a defined metric. You can use the ``'val_'`` prefix when you want to select the objective on the validation set. You can use one of ``['__min', '__max', '__last']`` which respectively means you want to select the min, max or last value among all epochs. Using '__last' will save a consequent compute time because the evaluation will not compute metrics on validation set for all epochs but the last. If ``objective`` is callable it should return a scalar value (i.e. float) and it will take a ``dict`` parameter. The ``dict`` will contain keys corresponding to loss and metrics such as ``['loss', 'val_loss', 'r2', 'val_r2]``, it will also contains ``'n_parameters'`` which corresponds to the number of trainable parameters of the current model, ``'training_time'`` which corresponds to the time required to train the model, ``'predict_time'`` which corresponds to the time required to make a prediction over the whole validation set. If this callable has a ``'__last'`` suffix then the evaluation will only compute validation loss/metrics for the last epoch. If this callable has contains 'with_pred' in its name then the ``dict`` will have two other keys ``['y_pred', 'y_true']`` where ``'y_pred`` corresponds to prediction of the model on validation set and ``'y_true'`` corresponds to real prediction.
        Raise:
            WrongProblemObjective: raised when the objective is of a wrong definition.
        """
        if (
            not self._space.get("loss") is None
            and not self._space.get("metrics") is None
        ):
            self.check_objective(objective)
        else:
            raise NaProblemError(
                ".loss and .metrics should be defined before .objective!"
            )

        self._space["objective"] = objective

    @property
    def space(self):
        keys = list(self._space.keys())
        keys.sort()
        space = OrderedDict(**{d: self._space[d] for d in keys})
        return space

    def build_search_space(self, seed=None):
        """Build and return a search space object using the infered data shapes after loading data.

        Returns:
            KSearchSpace: A search space instance.
        """
        config = self.space
        input_shape, output_shape, _ = setup_data(config, add_to_config=False)

        search_space = get_search_space(config, input_shape, output_shape, seed=seed)
        return search_space

    def get_keras_model(self, arch_seq: list) -> tf.keras.Model:
        """Get a keras model object from a set of decisions in the current search space.
        Args:
            arch_seq (list): a list of int of floats describing a choice of operations for the search space defined in the current problem.
        """
        search_space = self.build_search_space()
        return search_space.sample(arch_seq)

    def gen_config(self, arch_seq: list, hp_values: list) -> dict:
        """Generate a ``dict`` configuration from the ``arch_seq`` and ``hp_values`` passed.

        Args:
            arch_seq (list): a valid embedding of a neural network described by the search space of the current ``NaProblem``.
            hp_values (list): a valid list of hyperparameters corresponding to the defined hyperparameters of the current ``NaProblem``.
        """
        config = deepcopy(self.space)

        # architecture DNA
        config["arch_seq"] = arch_seq

        # replace hp values in the config
        hp_names = self._hp_space._space.get_hyperparameter_names()

        for hp_name, hp_value in zip(hp_names, hp_values):

            if hp_name == "loss":
                config["loss"] = hp_value
            else:
                config["hyperparameters"][hp_name] = hp_value

        return config

    def extract_hp_values(self, config):
        """Extract the value of hyperparameters present in ``config`` based on the defined hyperparameters in the current ``NaProblem``"""
        hp_names = self.hyperparameter_names
        hp_values = []
        for hp_name in hp_names:
            if hp_name == "loss":
                hp_values.append(config["loss"])
            else:
                hp_values.append(config["hyperparameters"][hp_name])

        return hp_values

    @property
    def hyperparameter_names(self):
        """The list of hyperparameters names."""
        return self._hp_space.hyperparameter_names

    @property
    def default_hp_configuration(self):
        """The default configuration as a dictionnary."""
        return self._hp_space.default_configuration


def module_location(attr):
    """
    :meta private:
    """
    return f"{attr.__module__}.{attr.__name__}"
