import abc
import copy
import functools
import logging
import os
import pathlib
import time

import numpy as np
import pandas as pd
import yaml

from deephyper.core.exceptions import MaximumJobsSpawnReached, SearchTerminationError
from deephyper.core.utils._introspection import get_init_params_as_json
from deephyper.core.utils._timeout import terminate_on_timeout
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.skopt.moo import non_dominated_set


class Search(abc.ABC):
    """Abstract class which represents a search algorithm.

    Args:
        problem: object describing the search/optimization problem.
        evaluator: object describing the evaluation process.
        random_state (np.random.RandomState, optional): Initial random state of the search. Defaults to ``None``.
        log_dir (str, optional): Path to the directoy where results of the search are stored. Defaults to ``"."``.
        verbose (int, optional): Use verbose mode. Defaults to ``0``.
    """

    def __init__(
        self, problem, evaluator, random_state=None, log_dir=".", verbose=0, **kwargs
    ):
        # get the __init__ parameters
        self._init_params = locals()
        self._call_args = []

        self._problem = copy.deepcopy(problem)

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
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._path_results = os.path.join(self._log_dir, "results.csv")
        if os.path.exists(self._path_results):
            str_current_time = time.strftime("%Y%m%d-%H%M%S")
            path_results_renamed = self._path_results.replace(
                ".", f"_{str_current_time}."
            )
            logging.warning(
                f"Results file already exists, it will be renamed to {path_results_renamed}"
            )
            os.rename(
                self._path_results,
                path_results_renamed,
            )

        self._verbose = verbose

        # if a callable is directly passed wrap it around the serial evaluator
        self.check_evaluator(evaluator)

    def check_evaluator(self, evaluator):
        if not (isinstance(evaluator, Evaluator)):
            if callable(evaluator):
                self._evaluator = Evaluator.create(
                    evaluator,
                    method="serial",
                    method_kwargs={
                        "callbacks": [TqdmCallback()] if self._verbose else []
                    },
                )
            else:
                raise TypeError(
                    f"The evaluator shoud be an instance of deephyper.evaluator.Evaluator by is {type(evaluator)}!"
                )
        else:
            self._evaluator = evaluator

    def to_json(self):
        """Returns a json version of the search object."""
        json_self = {
            "search": {
                "type": type(self).__name__,
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

    def _set_timeout(self, timeout=None):
        """If the `timeout` parameter is valid. Run the search in an other thread and trigger a timeout when this thread exhaust the allocated time budget.

        Args:
            timeout (int, optional): The time budget (in seconds) of the search before stopping. Defaults to ``None``, will not impose a time budget.
        """

        if timeout is not None:
            if type(timeout) is not int:
                raise ValueError(
                    f"'timeout' shoud be of type'int' but is of type '{type(timeout)}'!"
                )
            if timeout <= 0:
                raise ValueError("'timeout' should be > 0!")

        if np.isscalar(timeout) and timeout > 0:
            self._evaluator.set_timeout(timeout)
            self._search = functools.partial(
                terminate_on_timeout, timeout, self._search
            )

    def search(
        self, max_evals: int = -1, timeout: int = None, max_evals_strict: bool = False
    ):
        """Execute the search algorithm.

        Args:
            max_evals (int, optional): The maximum number of evaluations of the run function to perform before stopping the search. Defaults to ``-1``, will run indefinitely.
            timeout (int, optional): The time budget (in seconds) of the search before stopping. Defaults to ``None``, will not impose a time budget.
            max_evals_strict (bool, optional): If ``True`` the search will not spawn more than ``max_evals`` jobs. Defaults to ``False``.

        Returns:
            DataFrame: a pandas DataFrame containing the evaluations performed or ``None`` if the search could not evaluate any configuration.
        """
        self._set_timeout(timeout)
        if max_evals_strict:
            self._evaluator.set_max_num_jobs_spawn(max_evals)

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
        except SearchTerminationError as exc:
            # Collect remaining jobs
            if max_evals_strict and isinstance(exc, MaximumJobsSpawnReached):
                self._evaluator.gather("ALL")

            if "saved_keys" in dir(self):
                self._evaluator.dump_evals(saved_keys=self.saved_keys)
            else:
                self._evaluator.dump_evals(log_dir=self._log_dir)

        if not (os.path.exists(self._path_results)):
            logging.warning(f"Could not find results file at {self._path_results}!")
            return None

        self.extend_results_with_pareto_efficient(self._path_results)

        df_results = pd.read_csv(self._path_results)

        return df_results

    @abc.abstractmethod
    def _search(self, max_evals, timeout, max_evals_strict=False):
        """Search algorithm to be implemented.

        Args:
            max_evals (int, optional): The maximum number of evaluations of the run function to perform before stopping the search. Defaults to -1, will run indefinitely.
            timeout (int, optional): The time budget of the search before stopping.Defaults to None, will not impose a time budget.
        """

    @property
    def search_id(self):
        """The identifier of the search used by the evaluator."""
        return self._evaluator._search_id

    def extend_results_with_pareto_efficient(self, df_path: str):
        """Extend the results DataFrame with a column ``pareto_efficient`` which is ``True`` if the point is Pareto efficient.

        Args:
            df (pd.DataFrame): the input results DataFrame.
        """
        df = pd.read_csv(df_path)

        # Check if Multi-Objective Optimization was performed to save the pareto front
        objective_columns = [col for col in df.columns if col.startswith("objective")]

        if len(objective_columns) > 1:
            if pd.api.types.is_string_dtype(df[objective_columns[0]]):
                mask_no_failures = ~df[objective_columns[0]].str.startswith("F")
            else:
                mask_no_failures = np.ones(len(df), dtype=bool)
            objectives = -df.loc[mask_no_failures, objective_columns].values.astype(
                float
            )
            mask_pareto_front = non_dominated_set(objectives)
            df["pareto_efficient"] = False
            df.loc[mask_no_failures, "pareto_efficient"] = mask_pareto_front
            df.to_csv(df_path, index=False)
