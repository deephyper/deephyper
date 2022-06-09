import abc
import copy
import logging
import os
import pathlib
import signal

import numpy as np
import pandas as pd
from deephyper.core.exceptions import SearchTerminationError
from deephyper.core.utils._introspection import get_init_params_as_json
import yaml
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback


class Search(abc.ABC):
    """Abstract class which represents a search algorithm.

    Args:
        problem ([type]): [description]
        evaluator ([type]): [description]
        random_state ([type], optional): [description]. Defaults to None.
        log_dir (str, optional): [description]. Defaults to ".".
        verbose (int, optional): [description]. Defaults to 0.
    """

    def __init__(
        self, problem, evaluator, random_state=None, log_dir=".", verbose=0, **kwargs
    ):

        # get the __init__ parameters
        self._init_params = locals()
        self._call_args = []

        self._problem = copy.deepcopy(problem)

        # if a callable is directly passed wrap it around the serial evaluator
        if not (isinstance(evaluator, Evaluator)) and callable(evaluator):
            self._evaluator = Evaluator.create(
                evaluator,
                method="serial",
                method_kwargs={"callbacks": [TqdmCallback()]},
            )
        else:
            self._evaluator = evaluator
        self._seed = None

        if type(random_state) is int:
            self._seed = random_state
            self._random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self._random_state = random_state
        else:
            self._random_state = np.random.RandomState()

        # Create logging directory if does not exist
        self._log_dir = os.path.abspath(log_dir)
        pathlib.Path(log_dir).mkdir(parents=False, exist_ok=True)

        self._verbose = verbose

    def to_json(self):
        """Returns a json version of the search object."""
        json_self = {
            "search": {
                "type": type(self).__name__,
                "num_workers": self._evaluator.num_workers,
                **get_init_params_as_json(self),
            },
            "calls": self._call_args,
        }
        return json_self

    def dump_context(self):
        """Dumps the context in the log folder."""
        context = self.to_json()
        path_context = os.path.join(self._log_dir, "context.yaml")
        with open(path_context, "w") as file:
            yaml.dump(context, file)

    def terminate(self):
        """Terminate the search.

        Raises:
            SearchTerminationError: raised when the search is terminated with SIGALARM
        """
        logging.info("Search is being stopped!")
        raise SearchTerminationError

    def _set_timeout(self, timeout=None):
        def handler(signum, frame):
            self.terminate()

        signal.signal(signal.SIGALRM, handler)

        if np.isscalar(timeout) and timeout > 0:
            signal.alarm(timeout)

    def search(self, max_evals: int = -1, timeout: int = None):
        """Execute the search algorithm.

        Args:
            max_evals (int, optional): The maximum number of evaluations of the run function to perform before stopping the search. Defaults to ``-1``, will run indefinitely.
            timeout (int, optional): The time budget (in seconds) of the search before stopping. Defaults to ``None``, will not impose a time budget.

        Returns:
            DataFrame: a pandas DataFrame containing the evaluations performed or ``None`` if the search could not evaluate any configuration.
        """
        if timeout is not None:
            if type(timeout) is not int:
                raise ValueError(
                    f"'timeout' shoud be of type'int' but is of type '{type(timeout)}'!"
                )
            if timeout <= 0:
                raise ValueError(f"'timeout' should be > 0!")

        self._set_timeout(timeout)

        # save the search call arguments for the context
        self._call_args.append({"timeout": timeout, "max_evals": max_evals})
        # save the context in the log folder
        self.dump_context()
        # init tqdm callback
        if max_evals > 1:
            for cb in self._evaluator._callbacks:
                if isinstance(cb, TqdmCallback):
                    cb.set_max_evals(max_evals)

        try:
            self._search(max_evals, timeout)
        except SearchTerminationError:
            if "saved_keys" in dir(self):
                self._evaluator.dump_evals(saved_keys=self.saved_keys)
            else:
                self._evaluator.dump_evals()

        try:
            path_results = os.path.join(self._log_dir, "results.csv")
            df_results = pd.read_csv(path_results)
            return df_results
        except FileNotFoundError:
            return None

    @abc.abstractmethod
    def _search(self, max_evals, timeout):
        """Search algorithm to be implemented.

        Args:
            max_evals (int, optional): The maximum number of evaluations of the run function to perform before stopping the search. Defaults to -1, will run indefinitely.
            timeout (int, optional): The time budget of the search before stopping.Defaults to None, will not impose a time budget.
        """
