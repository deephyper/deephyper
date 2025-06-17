import abc
import asyncio
import copy
import csv
import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import yaml

from deephyper.evaluator import Evaluator, HPOJob, MaximumJobsSpawnReached
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo._problem import HpProblem
from deephyper.hpo._solution import (
    ArgMaxEstSelection,
    ArgMaxObsSelection,
    Solution,
    SolutionSelection,
)
from deephyper.analysis.hpo import get_mask_of_rows_without_failures
from deephyper.skopt.moo import non_dominated_set

__all__ = ["Search"]


def get_init_params_as_json(obj):
    """Get the parameters of an object in a json format.

    Args:
        obj (any): The object of which we want to know the ``__init__`` arguments.

    Returns:
        params (dict): Parameter names mapped to their values.
    """
    if hasattr(obj, "_init_params"):
        base_init_params = obj._init_params
        if "self" in base_init_params:
            base_init_params.pop("self")
    else:
        base_init_params = dict()
    params = dict()
    for k, v in base_init_params.items():
        if "__" not in k:
            if hasattr(v, "to_json"):
                params[k] = v.to_json()
            else:
                try:
                    params[k] = json.loads(json.dumps(v))
                except Exception:
                    params[k] = "NA"
    return params


class SearchHistory:
    """Represents the history of a search."""

    def __init__(
        self,
        problem: HpProblem,
        solution_selection: SolutionSelection,
    ):
        self.problem = problem
        self.solution_selection = solution_selection

        self.num_objective = None
        self.jobs: list[HPOJob] = []
        self.pareto_efficient = []
        self._csv_cursor = 0
        self._csv_columns = None
        self.solution_history = {}

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, idx) -> HPOJob:
        return self.jobs[idx]

    def extend(self, jobs: List[HPOJob]):
        if self.num_objective is None:
            obj = jobs[0].objective
            if isinstance(obj, (tuple, list)):
                self.num_objective = len(obj)
            else:
                self.num_objective = 1
            self.solution_selection.num_objective = self.num_objective
        self.jobs.extend(jobs)
        self.solution_selection.update(jobs)
        for job in jobs:
            self.solution_history[job.id] = self.solution

    @property
    def solution(self) -> Solution:
        return self.solution_selection.solution

    def _to_dict(self, jobs: List[HPOJob]) -> List[Dict[str, Any]]:
        results = []

        for job in jobs:
            # Prefix args with "p:"
            result = {f"p:{k}": v for k, v in job.args.items()}

            # Extract and process the objective
            obj = job.objective
            if isinstance(obj, (tuple, list)):
                if self.num_objective is None:
                    self.num_objective = len(obj)
                    self.solution_selection.num_objective = self.num_objective
                for i, val in enumerate(obj):
                    result[f"objective_{i}"] = val
            else:
                if self.num_objective is None:
                    self.num_objective = 1
                    self.solution_selection.num_objective = self.num_objective
                if self.num_objective > 1:
                    for i in range(self.num_objective):
                        result[f"objective_{i}"] = obj
                else:
                    result["objective"] = obj

            # Add job metadata
            result["job_id"] = int(job.id.split(".")[1])
            result["job_status"] = job.status.name

            # Add filtered metadata with "m:" prefix
            result.update({f"m:{k}": v for k, v in job.metadata.items() if not k.startswith("_")})

            # Optional Pareto-efficient tag
            if hasattr(job, "pareto_efficient"):
                result["pareto_efficient"] = job.pareto_efficient

            # Solution
            if self.num_objective == 1:
                result.update(
                    {f"sol.p:{k}": v for k, v in self.solution_history[job.id].parameters.items()}
                )
                result.update({"sol.objective": self.solution_history[job.id].objective})

            results.append(result)

        return results

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._to_dict(self.jobs))
        return df

    def to_csv_complete(self, path: str) -> pd.DataFrame:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return df

    def to_csv_partial(self, path: str, flush: bool = False):
        """Write results to CSV file.

        Args:
            path (str):

                Path of the CSV file.

            flush (bool, optional):

                A boolean that indicates if the CSV write in the case where ``partial=True`` should
                be flushed anyway. Otherwise if ``False`` it will not write to disk until there is
                a successful job. Defaults to ``False``.
        """
        resultsList = self._to_dict(self.jobs[self._csv_cursor :])

        if len(resultsList) > 0:
            started_dumping = self._csv_cursor > 0
            file_mode = "a" if started_dumping else "w"

            if not (started_dumping):
                for result in resultsList:
                    # Waiting to start receiving non-failed jobs before dumping results
                    is_single_obj_and_has_success = (
                        "objective" in result and type(result["objective"]) is not str
                    )
                    is_multi_obj_and_has_success = (
                        "objective_0" in result and type(result["objective_0"]) is not str
                    )
                    if is_single_obj_and_has_success or is_multi_obj_and_has_success or flush:
                        self._csv_columns = result.keys()
                        break

            if self._csv_columns is not None:
                with open(os.path.join(path), file_mode) as fp:
                    writer = csv.DictWriter(fp, self._csv_columns, extrasaction="ignore")
                    if not (started_dumping):
                        writer.writeheader()
                    writer.writerows(resultsList)
                    self._csv_cursor += len(resultsList)

    def compute_pareto_efficiency(self):
        """Compute the Pareto-Front from the current history.

        A column ``pareto_efficient`` is added to the dataframe. It is ``True`` if the
        point is Pareto efficient.
        """
        logging.info("Computing pareto efficient indicator...")
        df = self.to_dataframe()

        # Check if Multi-Objective Optimization was performed to save the pareto front
        objective_columns = [col for col in df.columns if col.startswith("objective")]

        if len(objective_columns) > 1:
            _, mask_no_failures = get_mask_of_rows_without_failures(df, objective_columns[0])
            objectives = -df.loc[mask_no_failures, objective_columns].values.astype(float)
            mask_pareto_front = non_dominated_set(objectives)

            self.pareto_efficient = np.zeros((len(self.jobs),), dtype=bool)
            self.pareto_efficient[mask_no_failures] = mask_pareto_front

            for job, pf in zip(self.jobs, self.pareto_efficient):
                job.pareto_efficient = pf


class Search(abc.ABC):
    """Abstract class which represents a search algorithm.

    Args:
        problem:
            object describing the search/optimization problem.

        evaluator:
            object describing the evaluation process.

        random_state (np.random.RandomState, optional):
            Initial random state of the search. Defaults to ``None``.

        log_dir (str, optional):
            Path to the directoy where results of the search are stored. Defaults to ``"."``.

        verbose (int, optional):
            Use verbose mode. Defaults to ``0``.

        stopper (Stopper, optional):
            a stopper to leverage multi-fidelity when evaluating the function. Defaults
            to ``None`` which does not use any stopper.

        checkpoint_history_to_csv (bool, optional):
            wether the results from progressively collected evaluations should be checkpointed
            regularly to disc as a csv. Defaults to ``True``.

        solution_selection (Literal["argmax_obs", "argmax_est"] | SolutionSelection, optional):
            the solution selection strategy. It can be a string where ``"argmax_obs"`` would
            select the argmax of observed objective values, and ``"argmax_est"`` would select the
            argmax of estimated objective values (through a predictive model).
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state=None,
        log_dir=".",
        verbose=0,
        stopper=None,
        checkpoint_history_to_csv=True,
        solution_selection: Literal["argmax_obs", "argmax_est"] | SolutionSelection = "argmax_obs",
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

        self._verbose = verbose
        self.checkpoint_history_to_csv = checkpoint_history_to_csv

        self.is_master = True

        # if a callable is directly passed wrap it around the serial evaluator
        self.check_evaluator(evaluator)

        # Set the search object in the evaluator to be able to call it within callbacks
        self._evaluator.search = self

        # Check if results already exist
        self._path_results = os.path.join(self._log_dir, "results.csv")
        if os.path.exists(self._path_results) and self.checkpoint_history_to_csv:
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

        # Default setting is asynchronous
        self.gather_type = "BATCH"
        self.gather_batch_size = 1

        self._evaluator._stopper = stopper

        self.stopped = False

        # Related to management of history of results
        if type(solution_selection) is str:
            if solution_selection == "argmax_obs":
                solution_selection = ArgMaxObsSelection()
            elif solution_selection == "argmax_est":
                solution_selection = ArgMaxEstSelection(
                    problem, random_state=self._random_state.randint(0, 2**31)
                )
            else:
                raise ValueError(
                    f"{solution_selection=} should be in ['argmax_obs', 'argmax_est'] when a str."
                )
        elif isinstance(solution_selection, SolutionSelection):
            pass
        else:
            raise ValueError(
                f"{solution_selection=} should be a str or an instance of SolutionSelection"
            )
        self.history = SearchHistory(self._problem, solution_selection=solution_selection)

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

    def search(
        self,
        max_evals: int = -1,
        timeout: Optional[int | float] = None,
        max_evals_strict: bool = False,
    ) -> pd.DataFrame:
        """Execute the search algorithm.

        Args:
            max_evals (int, optional): The maximum number of evaluations of the run function to
                perform before stopping the search. Defaults to ``-1``, will run indefinitely.

            timeout (int, optional): The time budget (in seconds) of the search before stopping.
                Defaults to ``None``, will not impose a time budget.

            max_evals_strict (bool, optional): If ``True`` the search will not spawn more than
                ``max_evals`` jobs. Defaults to ``False``.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the evaluations performed or ``None`` if the
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

        try:
            if isinstance(timeout, (int, float)):
                if timeout > 0:
                    self._evaluator.timeout = timeout
                else:
                    timeout = None
            elif timeout is None:
                pass
            else:
                raise ValueError(f"{timeout=} but is should be an int, float or None")
            self._search(max_evals, timeout, max_evals_strict)
        except MaximumJobsSpawnReached:
            self.stopped = True
            logging.warning(
                "Search is being stopped because the maximum number of spawned jobs has been "
                "reached."
            )

        # Collect remaining jobs
        logging.info("Collect remaining jobs...")
        last_results = []
        while self._evaluator.num_jobs_submitted > self._evaluator.num_jobs_gathered:
            results = self._evaluator.gather("ALL")
            if isinstance(results, tuple) and len(results) == 2:
                results = results[0] + results[1]
            last_results += results

        results = self._evaluator.close()
        last_results += results

        self.history.extend(last_results)

        if len(self.history) == 0:
            logging.warning("No results found in search history")
            return None

        self.dump_jobs_done_to_csv(flush=True)

        self.history.compute_pareto_efficiency()

        if self.checkpoint_history_to_csv:
            df_results = self.history.to_csv_complete(os.path.join(self._log_dir, "results.csv"))
        else:
            df_results = self.history.to_dataframe()

        return df_results

    @property
    def search_id(self):
        """The identifier of the search used by the evaluator."""
        return self._evaluator._search_id

    def _search(self, max_evals: int, timeout: Optional[int], max_evals_strict=False):
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

            self.history.extend(new_results)

            logging.info("Dumping evaluations...")
            t1 = time.time()
            self.dump_jobs_done_to_csv()
            logging.info(f"Dumping took {time.time() - t1:.4f} sec.")

            self.tell(new_results)

            # Test if search should be stopped due to timeout
            time_left = self._evaluator.time_left
            if time_left is not None and time_left <= 0:
                logging.info(
                    f"Searching time remaining is {time_left:.3f} <= 0 stopping the search..."
                )
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
        logging.info(f"Telling {len(results)} new result(s)...")
        t1 = time.time()
        self._tell(results)
        logging.info(f"Telling took {time.time() - t1:.4f} sec.")

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
        if self.is_master and self.checkpoint_history_to_csv:
            path = os.path.join(self._log_dir, "results.csv")
            self.history.to_csv_partial(path, flush=flush)
