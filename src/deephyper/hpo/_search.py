import abc
import asyncio
import copy
import logging
import os
import pathlib
import time

import numpy as np
import pandas as pd
import yaml

from typing import List, Dict

from deephyper.core.exceptions import SearchTerminationError
from deephyper.core.exceptions import MaximumJobsSpawnReached
from deephyper.core.exceptions import TimeoutReached
from deephyper.core.utils._introspection import get_init_params_as_json
from deephyper.evaluator import Evaluator, HPOJob
from deephyper.evaluator.callback import TqdmCallback
from deephyper.skopt.moo import non_dominated_set


class Search(abc.ABC):
    """Abstract class which represents a search algorithm.

    Args:
        problem: object describing the search/optimization problem.
        evaluator: object describing the evaluation process.
        random_state (np.random.RandomState, optional): Initial random state of the search.
            Defaults to ``None``.
        log_dir (str, optional): Path to the directoy where results of the search are stored.
            Defaults to ``"."``.
        verbose (int, optional): Use verbose mode. Defaults to ``0``.
        stopper (Stopper, optional): a stopper to leverage multi-fidelity when evaluating the
            function. Defaults to ``None`` which does not use any stopper.
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state=None,
        log_dir=".",
        verbose=0,
        stopper=None,
        **kwargs,
    ):
        # TODO: stopper should be an argument passed here... check CBO and generalize
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

        self._verbose = verbose

        self.is_master = True

        # if a callable is directly passed wrap it around the serial evaluator
        self.check_evaluator(evaluator)

        # Set the search object in the evaluator to be able to call it within callbacks
        self._evaluator.search = self

        # Check if results already exist
        self._path_results = os.path.join(self._log_dir, "results.csv")
        if os.path.exists(self._path_results):
            str_current_time = time.strftime("%Y%m%d-%H%M%S")
            path_results_dirname = os.path.dirname(self._path_results)
            path_results_basename = os.path.basename(self._path_results)
            path_results_basename = path_results_basename.replace(".", f"_{str_current_time}.")
            path_results_renamed = os.path.join(path_results_dirname, path_results_basename)
            logging.warning(
                f"Results file already exists, it will be renamed to {path_results_renamed}"
            )
            os.rename(
                self._path_results,
                path_results_renamed,
            )
            evaluator._columns_dumped = None
            evaluator._start_dumping = False

        # Default setting is asynchronous
        self.gather_type = "BATCH"
        self.gather_batch_size = 1

        self._evaluator._stopper = stopper

        self.stopped = False

    def check_evaluator(self, evaluator):
        if not (isinstance(evaluator, Evaluator)):
            if callable(evaluator):
                # Pick the adapted evaluator depending if the passed function is a coroutine
                if asyncio.iscoroutinefunction(evaluator):
                    method = "serial"
                else:
                    method = "thread"

                self._evaluator = Evaluator.create(
                    evaluator,
                    method=method,
                    method_kwargs={"callbacks": [TqdmCallback()] if self._verbose else []},
                )
            else:
                raise TypeError(
                    f"The evaluator shoud be of type deephyper.evaluator.Evaluator or Callable but "
                    f"it is  {type(evaluator)}!"
                )
        else:
            self._evaluator = evaluator

        self._evaluator._job_class = HPOJob

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

    def _check_timeout(self, timeout=None):
        """Check the timeout parameter for the evaluator used by the search.

        Args:
            timeout (int, optional): The time budget (in seconds) of the search before stopping.
                Defaults to ``None``, will not impose a time budget.
        """
        if timeout is not None:
            if type(timeout) is not int:
                raise ValueError(
                    f"'timeout' shoud be of type'int' but is of type '{type(timeout)}'!"
                )
            if timeout <= 0:
                raise ValueError("'timeout' should be > 0!")

    def search(self, max_evals: int = -1, timeout: int = None, max_evals_strict: bool = False):
        """Execute the search algorithm.

        Args:
            max_evals (int, optional): The maximum number of evaluations of the run function to
                perform before stopping the search. Defaults to ``-1``, will run indefinitely.
            timeout (int, optional): The time budget (in seconds) of the search before stopping.
                Defaults to ``None``, will not impose a time budget.
            max_evals_strict (bool, optional): If ``True`` the search will not spawn more than
                ``max_evals`` jobs. Defaults to ``False``.

        Returns:
            DataFrame: A pandas DataFrame containing the evaluations performed or ``None`` if the
                search could not evaluate any configuration.

                This DataFrame contains the following columns:
                - ``p:HYPERPARAMETER_NAME``: for each hyperparameter of the problem.
                - ``objective``: for single objective optimization.
                - ``objective_0``, ``objective_1``, ...: for multi-objective optimization.
                - ``job_id``: the identifier of the job.
                - ``job_status``: the status of the job at the end of the search.
                - ``m:METADATA_NAME``: for each metadata of the problem. Some metadata are always
                    present like ``m:timestamp_submit`` and ``m:timestamp_gather`` which are the
                    timestamps of the submission and gathering of the job.
        """
        self.stopped = False
        self._check_timeout(timeout)
        if max_evals_strict:
            # TODO: should be replaced by a property with a setter?
            self._evaluator.set_maximum_num_jobs_submitted(max_evals)

        # save the search call arguments for the context
        self._call_args.append({"timeout": timeout, "max_evals": max_evals})
        # save the context in the log folder
        self.dump_context()
        # init tqdm callback
        if max_evals > 1:
            for cb in self._evaluator._callbacks:
                if isinstance(cb, TqdmCallback):
                    cb.set_max_evals(max_evals)

        wait_all_running_jobs = True
        try:
            if np.isscalar(timeout) and timeout > 0:
                self._evaluator.timeout = timeout
            self._search(max_evals, timeout, max_evals_strict)
        except TimeoutReached:
            self.stopped = True
            wait_all_running_jobs = False
            logging.warning("Search is being stopped because the allowed timeout has been reached.")
        except MaximumJobsSpawnReached:
            self.stopped = True
            logging.warning(
                "Search is being stopped because the maximum number of spawned jobs has been "
                "reached."
            )
        except SearchTerminationError:
            self.stopped = True
            logging.warning("Search has been requested to be stopped.")

        # Collect remaining jobs
        logging.info("Collect remaining jobs...")
        if wait_all_running_jobs:
            while self._evaluator.num_jobs_submitted > self._evaluator.num_jobs_gathered:
                self._evaluator.gather("ALL")
                self.dump_jobs_done_to_csv()
        else:
            self._evaluator.gather_other_jobs_done()
            self.dump_jobs_done_to_csv()

        self._evaluator.close()

        if not (os.path.exists(self._path_results)):
            logging.warning(f"Could not find results file at {self._path_results}!")
            return None

        # Force dumping if all configurations were failed
        self.dump_jobs_done_to_csv(flush=True)

        self.extend_results_with_pareto_efficient_indicator()

        df_results = pd.read_csv(self._path_results)

        return df_results

    @property
    def search_id(self):
        """The identifier of the search used by the evaluator."""
        return self._evaluator._search_id

    def extend_results_with_pareto_efficient_indicator(self):
        """Extend the results DataFrame with Pareto-Front.

        A column ``pareto_efficient`` is added to the dataframe. It is ``True`` if the
        point is Pareto efficient.
        """
        if self.is_master:
            logging.info("Extends results with pareto efficient indicator...")
            df_path = self._path_results
            df = pd.read_csv(df_path)

            # Check if Multi-Objective Optimization was performed to save the pareto front
            objective_columns = [col for col in df.columns if col.startswith("objective")]

            if len(objective_columns) > 1:
                if pd.api.types.is_string_dtype(df[objective_columns[0]]):
                    mask_no_failures = ~df[objective_columns[0]].str.startswith("F")
                else:
                    mask_no_failures = np.ones(len(df), dtype=bool)
                objectives = -df.loc[mask_no_failures, objective_columns].values.astype(float)
                mask_pareto_front = non_dominated_set(objectives)
                df["pareto_efficient"] = False
                df.loc[mask_no_failures, "pareto_efficient"] = mask_pareto_front
                df.to_csv(df_path, index=False)

    def _search(self, max_evals, timeout, max_evals_strict=False):
        """Search algorithm logic.

        Args:
            max_evals (int): The maximum number of evaluations of the run function to perform
                before stopping the search. Defaults to -1, will run indefinitely.
            timeout (int): The time budget of the search before stopping. Defaults to ``None``,
                will not impose a time budget.
            max_evals_strict (bool, optional): Wether the number of submitted jobs should be
            strictly equal to ``max_evals``.
        """
        if max_evals_strict:

            def num_evals():
                return self._evaluator.num_jobs_submitted

        else:

            def num_evals():
                return self._evaluator.num_jobs_gathered

        # Update the number of evals in case the `search.search(...)` was previously called
        max_evals = max_evals if max_evals < 0 else max_evals + num_evals()

        n_ask = self._evaluator.num_workers

        while not self.stopped and (max_evals < 0 or num_evals() < max_evals):
            new_batch = self.ask(n_ask)

            logging.info(f"Submitting {len(new_batch)} configurations...")
            t1 = time.time()
            self._evaluator.submit(new_batch)
            logging.info(f"Submition took {time.time() - t1:.4f} sec.")

            logging.info("Gathering jobs...")
            t1 = time.time()

            new_results = self._evaluator.gather(self.gather_type, self.gather_batch_size)

            # Check if results are received from other search instances
            # connected to the same storage
            if isinstance(new_results, tuple) and len(new_results) == 2:
                local_results, other_results = new_results
                n_ask = len(local_results)
                new_results = local_results + other_results
                logging.info(
                    f"Gathered {len(local_results)} local job(s) and {len(other_results)} other "
                    f"job(s) in {time.time() - t1:.4f} sec."
                )
            else:
                n_ask = len(new_results)
                logging.info(f"Gathered {len(new_results)} job(s) in {time.time() - t1:.4f} sec.")

            logging.info("Dumping evaluations...")
            t1 = time.time()
            self.dump_jobs_done_to_csv()
            logging.info(f"Dumping took {time.time() - t1:.4f} sec.")

            logging.info(f"Telling {len(new_results)} new result(s)...")
            t1 = time.time()
            self.tell(new_results)
            logging.info(f"Telling took {time.time() - t1:.4f} sec.")

            # Test if search should be stopped due to timeout
            time_left = self._evaluator.time_left
            if time_left is not None and time_left <= 0:
                self.stopped = True

            # Test if search should be stopped because a callback requested it
            if any(
                (hasattr(c, "search_stopped") and c.search_stopped)
                for c in self._evaluator._callbacks
            ):
                self.stopped = True

    def ask(self, n: int = 1) -> List[Dict]:
        """Ask the search for new configurations to evaluate.

        Args:
            n (int, optional): The number of configurations to ask. Defaults to 1.

        Returns:
            List[Dict]: a list of hyperparameter configurations to evaluate.
        """
        logging.info(f"Asking {n} configuration(s)...")
        t1 = time.time()

        new_samples = self._ask(n)

        logging.info(f"Asking took {time.time() - t1:.4f} sec.")

        return new_samples

    @abc.abstractmethod
    def _ask(self, n: int = 1) -> List[Dict]:
        """Ask the search for new configurations to evaluate.

        Args:
            n (int, optional): The number of configurations to ask. Defaults to 1.

        Returns:
            List[Dict]: a list of hyperparameter configurations to evaluate.
        """

    def tell(self, results: List[HPOJob]):
        """Tell the search the results of the evaluations.

        Args:
            results (List[HPOJob]): a list of HPOJobs from which hyperparameters and objectives can
            be retrieved.
        """
        self._tell(results)

    @abc.abstractmethod
    def _tell(self, results: List[HPOJob]):
        """Tell the search the results of the evaluations.

        Args:
            results (List[HPOJob]): a list of HPOJobs from which hyperparameters and objectives can
                be retrieved.
        """

    def dump_jobs_done_to_csv(self, flush: bool = False):
        """Dump jobs completed to CSV in log_dir.

        Args:
            flush (bool, optional): Force the dumping if set to ``True``. Defaults to ``False``.
        """
        if self.is_master:
            self._evaluator.dump_jobs_done_to_csv(log_dir=self._log_dir, flush=flush)
