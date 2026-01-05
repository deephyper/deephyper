import copy
import warnings

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

import deephyper.skopt
from deephyper.skopt.joblib import Parallel, delayed


def convert_to_skopt_dim(cs_hp, surrogate_model=None):
    if surrogate_model in ["RF", "ET", "GBRT", "HGBRT", "MF", "BT"]:
        # models not sensitive to the metric space such as trees
        surrogate_model_type = "rule_based"
    else:
        # models sensitive to the metric space such as GP, neural networks
        surrogate_model_type = "distance_based"

    if isinstance(cs_hp, csh.UniformIntegerHyperparameter):
        skopt_dim = deephyper.skopt.space.Integer(
            low=cs_hp.lower,
            high=cs_hp.upper,
            prior="log-uniform" if cs_hp.log else "uniform",
            name=cs_hp.name,
        )
    elif isinstance(cs_hp, csh.UniformFloatHyperparameter):
        skopt_dim = deephyper.skopt.space.Real(
            low=cs_hp.lower,
            high=cs_hp.upper,
            prior="log-uniform" if cs_hp.log else "uniform",
            name=cs_hp.name,
        )
    elif isinstance(cs_hp, csh.CategoricalHyperparameter):
        # the transform is important if we don't want the complexity of trees
        # to explode with categorical variables
        skopt_dim = deephyper.skopt.space.Categorical(
            categories=cs_hp.choices,
            name=cs_hp.name,
            transform="onehot" if surrogate_model_type == "distance_based" else "label",
        )
    elif isinstance(cs_hp, csh.OrdinalHyperparameter):
        categories = list(cs_hp.sequence)
        if all(
            isinstance(x, (int, np.integer)) or isinstance(x, (float, np.floating))
            for x in categories
        ):
            transform = "identity"
        else:
            transform = "label"
        skopt_dim = deephyper.skopt.space.Categorical(
            categories=categories, name=cs_hp.name, transform=transform
        )
    elif isinstance(cs_hp, csh.Constant):
        categories = [cs_hp.value]
        skopt_dim = deephyper.skopt.space.Categorical(
            categories=categories, name=cs_hp.name, transform="label"
        )
    elif isinstance(cs_hp, csh.NormalIntegerHyperparameter):
        skopt_dim = deephyper.skopt.space.Integer(
            low=cs_hp.lower,
            high=cs_hp.upper,
            prior="normal",
            name=cs_hp.name,
            loc=cs_hp.mu,
            scale=cs_hp.sigma,
        )
    elif isinstance(cs_hp, csh.NormalFloatHyperparameter):
        skopt_dim = deephyper.skopt.space.Real(
            low=cs_hp.lower,
            high=cs_hp.upper,
            prior="normal",
            name=cs_hp.name,
            loc=cs_hp.mu,
            scale=cs_hp.sigma,
        )
    else:
        raise TypeError(f"Cannot convert hyperparameter of type {type(cs_hp)}")

    return skopt_dim


def convert_to_skopt_space(cs_space, surrogate_model=None):
    """Convert a ConfigurationSpace to a scikit-optimize Space.

    Args:
        cs_space (ConfigurationSpace): the ``ConfigurationSpace`` to convert.
        surrogate_model (str, optional): the type of surrogate model/base estimator used to
            perform Bayesian optimization. Defaults to ``None``.

    Raises:
        TypeError: if the input space is not a ConfigurationSpace.

    Returns:
        deephyper.skopt.space.Space: a scikit-optimize Space.
    """
    # verify pre-conditions
    if not (isinstance(cs_space, cs.ConfigurationSpace)):
        raise TypeError("Input space should be of type ConfigurationSpace")

    sample_with_config_space = len(cs_space.conditions) > 0 or len(cs_space.forbidden_clauses) > 0

    # convert the ConfigSpace to deephyper.skopt.space.Space
    dimensions = []
    for hp in list(cs_space.values()):
        dimensions.append(convert_to_skopt_dim(hp, surrogate_model))

    skopt_space = deephyper.skopt.space.Space(
        dimensions, config_space=cs_space if sample_with_config_space else None
    )
    return skopt_space


def check_hyperparameter(parameter, name=None, default_value=None):
    """Check if the passed parameter is a valid description of an hyperparameter.

    :meta private:

    Args:
        parameter (str|Hyperparameter): an instance of ``ConfigSpace.hyperparameters.
            hyperparameter`` or a synthetic description (e.g., ``list``, ``tuple``).
        name (str): the name of the hyperparameter. Only required when the parameter is not a
            ``ConfigSpace.hyperparameters.hyperparameter``.
        default_value: a default value for the hyperparameter.

    Returns:
        Hyperparameter: the ConfigSpace hyperparameter instance corresponding to the ``parameter``
        description.
    """
    if isinstance(parameter, csh.Hyperparameter):
        return parameter

    if not isinstance(parameter, (list, tuple, np.ndarray, dict)):
        if isinstance(parameter, (int, float, str)):
            return csh.Constant(name=name, value=parameter)

        raise ValueError(
            "Shortcut definition of an hyper-parameter has to be a type in [list, tuple, array, "
            "dict, float, int, str]."
        )

    if type(name) is not str:
        raise ValueError("The 'name' of an hyper-parameter should be a string!")

    kwargs = {}
    if default_value is not None:
        kwargs["default_value"] = default_value

    if type(parameter) is tuple:  # Range of reals or integers
        if len(parameter) == 2:
            prior = "uniform"
        elif len(parameter) == 3:
            prior = parameter[2]
            assert prior in [
                "uniform",
                "log-uniform",
            ], (
                f"Prior has to be 'uniform' or 'log-uniform' when {prior} "
                f"was given for parameter '{name}'"
            )
            parameter = parameter[:2]

        log = prior == "log-uniform"

        if all([isinstance(p, int) for p in parameter]):
            return csh.UniformIntegerHyperparameter(
                name=name, lower=parameter[0], upper=parameter[1], log=log, **kwargs
            )
        elif any([isinstance(p, float) for p in parameter]):
            return csh.UniformFloatHyperparameter(
                name=name, lower=parameter[0], upper=parameter[1], log=log, **kwargs
            )
    elif type(parameter) is list:  # Categorical
        if any([isinstance(p, (str, bool)) or isinstance(p, np.bool_) for p in parameter]):
            return csh.CategoricalHyperparameter(name, choices=parameter, **kwargs)
        elif all([isinstance(p, (int, float)) for p in parameter]):
            return csh.OrdinalHyperparameter(name, sequence=parameter, **kwargs)
    elif type(parameter) is dict:  # Integer or Real distribution
        # Normal
        if "mu" in parameter and "sigma" in parameter:
            if type(parameter["mu"]) is float:
                return csh.NormalFloatHyperparameter(name=name, **parameter, **kwargs)
            elif type(parameter["mu"]) is int:
                return csh.NormalIntegerHyperparameter(name=name, **parameter, **kwargs)
            else:
                raise ValueError(
                    "Wrong hyperparameter definition! 'mu' should be either a float or an integer."
                )

    raise ValueError(
        f"Invalid dimension {name}: {parameter}. Read the documentation for supported types."
    )


class HpProblem:
    """Class to define an hyperparameter problem.

    >>> from deephyper.hpo import HpProblem
    >>> problem = HpProblem()

    Args:
        config_space (ConfigurationSpace, optional): In case the ``HpProblem`` is defined from a
        `ConfigurationSpace`.
    """

    def __init__(self, config_space=None, seed: int | None = None):
        if config_space is not None and not (isinstance(config_space, cs.ConfigurationSpace)):
            raise ValueError(
                "Parameter 'config_space' should be an instance of ConfigurationSpace!"
            )

        if config_space:
            self._space = copy.deepcopy(config_space)
        else:
            self._space = cs.ConfigurationSpace()
        if seed is not None:
            self._space.seed(seed)
        self.rng = check_random_state(seed)

        self.skopt_dims = {}
        self.references = []  # starting points

        self.constraint_fn = None
        self.sampling_fn = None

    def __str__(self):
        return repr(self)

    def __repr__(self):
        prob = repr(self._space)
        return prob

    def __len__(self):
        return len(self.hyperparameter_names)

    def __getitem__(self, hyperparameter_name):
        return self.space[hyperparameter_name]

    def add_hyperparameter(self, value, name: str = None, default_value=None) -> csh.Hyperparameter:
        """Add an hyperparameter to the ``HpProblem``.

        Hyperparameters can be added to a ``HpProblem`` with a short syntax:

        >>> problem.add_hyperparameter((0, 10), "discrete", default_value=5)
        >>> problem.add_hyperparameter((0.0, 10.0), "real", default_value=5.0)
        >>> problem.add_hyperparameter([0, 10], "categorical", default_value=0)

        Sampling distributions can be provided:

        >>> problem.add_hyperparameter((0.0, 10.0, "log-uniform"), "real", default_value=5.0)

        It is also possible to use `ConfigSpace Hyperparameters
        <https://automl.github.io/ConfigSpace/master/API-Doc.html#hyperparameters>`_:

        >>> import ConfigSpace.hyperparameters as csh
        >>> csh_hp = csh.UniformIntegerHyperparameter(
        ...     name='uni_int', lower=10, upper=100, log=False)
        >>> problem.add_hyperparameter(csh_hp)

        Args:
            value (tuple or list or ConfigSpace.Hyperparameter): a valid hyperparametr description.
            name (str): The name of the hyperparameter to add.
            default_value (float or int or str): A default value for the corresponding
                hyperparameter.

        Returns:
            ConfigSpace.Hyperparameter: a ConfigSpace ``Hyperparameter`` object corresponding to
            the ``(value, name, default_value)``.
        """
        if not (type(name) is str or name is None):
            raise TypeError(
                f"Dimension name: '{name}' is of type == {type(name)} when should be 'str'!"
            )
        csh_parameter = check_hyperparameter(value, name, default_value=default_value)
        self._space.add(csh_parameter)

        if isinstance(csh_parameter, csh.Hyperparameter):
            skopt_dim = convert_to_skopt_dim(csh_parameter, surrogate_model="ET")
            self.skopt_dims[skopt_dim.name] = skopt_dim

        return csh_parameter

    def add_hyperparameters(self, hp_list):
        """Add a list of hyperparameters.

        It can be useful when a list of ``ConfigSpace.Hyperparameter`` are defined and we need to
        add them to the ``HpProblem``.

        Args:
            hp_list (ConfigSpace.Hyperparameter): a list of ConfigSpace hyperparameters.

        Returns:
            list: The list of added hyperparameters.
        """
        return [self.add_hyperparameter(hp) for hp in hp_list]

    def add_forbidden_clause(self, clause):
        r"""Add a forbidden clause to the problem.

        Add a `forbidden clause <https://automl.github.io/ConfigSpace/master/API-Doc.html#forbidden-clauses>`_
        to the ``HpProblem``.

        For example if we want to optimize :math:`\frac{1}{x}` where :math:`x` cannot be equal to 0:

        >>> from deephyper.hpo import HpProblem
        >>> import ConfigSpace as cs
        >>> problem = HpProblem()
        >>> x = problem.add_hyperparameter((0.0, 10.0), "x")
        >>> problem.add_forbidden_clause(cs.ForbiddenEqualsClause(x, 0.0))

        Args:
            clause: a ConfigSpace forbidden clause.
        """
        self._space.add(clause)

    def add_condition(self, condition):
        """Add a condition to the problem.

        Add a `condition <https://automl.github.io/ConfigSpace/master/API-Doc.html#conditions>`_ to
        the ``HpProblem``.

        >>> from deephyper.hpo import HpProblem
        >>> import ConfigSpace as cs
        >>> problem = HpProblem()
        >>> x = problem.add_hyperparameter((0.0, 10.0), "x")
        >>> y = problem.add_hyperparameter((1e-4, 1.0), "y")
        >>> problem.add_condition(cs.LessThanCondition(y, x, 1.0))

        Args:
            condition: A ConfigSpace condition.
        """
        self._space.add(condition)

    def add_conditions(self, conditions: list) -> None:
        """Add a list of conditions to the problem.

        Args:
            conditions (list): A list of ConfigSpace conditions.
        """
        self._space.add(*conditions)

    def add(self, value, name=None, default_value=None) -> None:
        """Add a component to the configuration space.

        An added component can be an hyperparameter, a forbidden rule or a condition.
        """
        if name is not None:
            self.add_hyperparameter(value, name, default_value)
            return

        if isinstance(value, csh.Hyperparameter):
            skopt_dim = convert_to_skopt_dim(value, surrogate_model="ET")
            self.skopt_dims[skopt_dim.name] = skopt_dim

        self._space.add(value)

    def sample(
        self,
        size: int = 1,
        strict: bool = False,
        max_trials: int = 5,
        n_jobs: int = 1,
    ) -> list[dict]:
        """Sample a list of hyperparameter configuration.

        Args:
            size (int): The number of configurations to sample.
            strict (bool): If the returned number of samples should be strictly equal to
                ``size``. Defaults to ``False``.
            max_trials (int): The maximum number of sampling trials. Defaults to ``5``.
            n_jobs (int): The number of concurrent threads for sampling flat search space.

        Returns:
            list[dict]: the list of sampled configurations.
        """

        def _sample_dimension(dim, i, n_samples, random_state, out):
            """Wrapper to sample dimension for joblib parallelization."""
            out[0][:, i] = dim.rvs(n_samples=n_samples, random_state=random_state)

        def sample_fn(size: int) -> list[dict]:
            if self.sampling_fn is None:
                sample_with_config_space = (
                    len(self._space.conditions) > 0 or len(self._space.forbidden_clauses) > 0
                )
                if sample_with_config_space:
                    samples = self._space.sample_configuration(size=size)
                    samples = [dict(s) for s in samples]
                else:
                    # Regular sampling without transfer learning from flat search space
                    # Joblib parallel optimization
                    # Draw
                    n_columns = len(self.hyperparameter_names)
                    columns = np.zeros((size, n_columns), dtype="O")
                    random_states = self.rng.randint(
                        low=0,
                        high=np.iinfo(np.int32).max,
                        size=n_columns,
                    )
                    Parallel(n_jobs=n_jobs, verbose=0, require="sharedmem")(
                        delayed(_sample_dimension)(
                            self.skopt_dims[dim_name],
                            i,
                            size,
                            np.random.RandomState(random_states[i]),
                            [columns],
                        )
                        for i, dim_name in enumerate(self.hyperparameter_names)
                    )
                    df = pd.DataFrame(
                        {k: columns[:, i] for i, k in enumerate(self.hyperparameter_names)}
                    )
                    samples = df.to_dict(orient="records")
            else:
                samples = self.sampling_fn(size)
            return samples

        if self.constraint_fn is None:
            # Fast path: no constraint
            return sample_fn(size)

        accepted = []
        trials = 0
        batch_size = size

        while len(accepted) < size and trials < max_trials:
            # Sample a batch
            batch = sample_fn(size)

            # Convert batch into DataFrame only once
            df = pd.DataFrame(batch)

            # Apply constraint ---
            accept_mask = self.constraint_fn(df)

            df = df[accept_mask]
            accepted.extend(df.to_dict(orient="records"))

            trials += 1
            ratio_accept = float(accept_mask.sum() / batch_size)
            if ratio_accept <= 1e-3:
                batch_size = 2 * batch_size
                if batch_size > 100_000:
                    warnings.warn(
                        f"Constraint is hard to sample with {ratio_accept=}! "
                        "Consider setting a custom sampling_fn",
                        category=UserWarning,
                    )
            else:
                batch_size = int((size - len(accepted)) / ratio_accept + 0.5)

        # If constraints are too strict, return what we have (or raise)
        # You can choose to raise if you need strictly size samples
        if strict:
            accepted = accepted[:size]

            if len(accepted) < size:
                return RuntimeError(f"The number of samples is less than {size=}!")

        return accepted

    @property
    def space(self) -> cs.ConfigurationSpace:
        """The wrapped ConfigurationSpace object."""
        return self._space

    @property
    def hyperparameter_names(self):
        """The list of hyperparameters names."""
        return list(self._space.keys())

    def check_configuration(self, parameters: dict, raise_if_not_valid: bool = True) -> bool:
        """Check if a configuration is valid.

        Args:
            parameters (dict): the configuration of parameters to test.
            raise_if_not_valid (bool): indicate if an error is raised if the configuration of
                parameters is invalid.

        Raise:
            ValueError: if the configuration is invalid.
        """
        try:
            # Check is included in the init of Configuration
            cs.Configuration(self._space, parameters)
        except ValueError as e:
            if raise_if_not_valid:
                raise ValueError(str(e))
            return False
        if self.constraint_fn:
            if not self.constraint_fn(parameters):
                if raise_if_not_valid:
                    raise ValueError(
                        f"The {parameters=} are not valid with respect to the defined constraint_fn"
                    )
                else:
                    return False
        return True

    @property
    def default_configuration(self):
        """The default configuration as a dictionnary."""
        config = dict(self._space.get_default_configuration())
        for hp_name, hp in self._space.items():
            if hp_name not in config:
                if isinstance(hp, csh.CategoricalHyperparameter):
                    config[hp_name] = hp.choices[0]
                elif isinstance(hp, csh.OrdinalHyperparameter):
                    config[hp_name] = hp.sequence[0]
                elif isinstance(hp, csh.Constant):
                    config[hp_name] = hp.value
                elif isinstance(
                    hp,
                    (csh.UniformIntegerHyperparameter, csh.UniformFloatHyperparameter),
                ):
                    config[hp_name] = hp.lower
                else:
                    config[hp_name] = hp.default_value
        return config

    def to_json(self) -> dict:
        """Returns a dictionary of the space which can be saved as JSON."""
        d = self._space.to_serialized_dict()
        return d

    def set_seed(self, seed: int):
        """Set the random seed of the space."""
        self._space.seed(seed)
        self.rng = check_random_state(seed)

    def set_constraint_fn(self, fn: callable):
        """Set the constraint function.

        Example:

            .. code-block:: python

                pb = HpProblem()
                pb.add((0.0, 10.0), "x")
                pb.add((0.0, 10.0), "y")

                def constraint_fn(df: pd.DataFrame) -> pd.Series:
                    accept = df["x"] + df["y"] >= 10
                    return accept

                pb.set_constraint_fn(constraint_fn)

                samples = pb.sample(size=100)
                df = pd.DataFrame(samples)
                assert all(df["x"] + df["y"] >= 10)
        """
        self.constraint_fn = fn

    def set_sampling_fn(self, fn: callable):
        """Set the sampling function."""
        self.sampling_fn = fn
