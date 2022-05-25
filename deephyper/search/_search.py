import abc
import ast
import copy
import logging
import os
import pathlib
import signal
import subprocess

import numpy as np
import pandas as pd
from deephyper.core.exceptions import SearchTerminationError
import yaml


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

        self._problem = copy.deepcopy(problem)
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

        self._context = {
            "env": self._get_env(),
            "search": {
                "type": type(self).__name__,
                "random_state": random_state,
                "num_workers": evaluator.num_workers,
                "evaluator": evaluator.get_infos(),
                "problem": problem.get_infos(),
            }
        }
    
    def _get_env(self):
        """Gives the environment of execution of a search.

        Returns:
            dict: contains the infos of the environment.
        """        
        pip_list_com = subprocess.run(['pip', 'list', '--format', 'json'], stdout=subprocess.PIPE)
        pip_list = ast.literal_eval(pip_list_com.stdout.decode('utf-8'))

        env = {
            "pip": pip_list,
        }
        return env

    def _add_call_log(self, call_args: dict = None):
        calls_log = self._context.get("calls", [])
        calls_log.append(call_args)
        self._context["calls"] = calls_log

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
                raise ValueError(f"'timeout' shoud be of type'int' but is of type '{type(timeout)}'!")
            if timeout <= 0:
                raise ValueError(f"'timeout' should be > 0!")

        self._add_call_log(
            {
                "max_evals": max_evals,
                "timeout": timeout,
            }
        )
        try:
            path_context = os.path.join(self._log_dir, "context.yaml")
            with open(path_context, "w") as file:
                yaml.dump(self._context, file)
        except FileNotFoundError:
            None

        self._set_timeout(timeout)

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
