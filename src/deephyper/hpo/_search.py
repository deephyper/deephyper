import abc
import copy
import csv
import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Literal, Optional
from inspect import iscoroutinefunction

import numpy as np
import pandas as pd

from deephyper.analysis.hpo import (
    get_mask_of_rows_without_failures,
    read_results_from_csv,
)
from deephyper.evaluator import Evaluator, HPOJob, MaximumJobsSpawnReached, JobStatus
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo._problem import HpProblem
from deephyper.hpo._solution import (
    ArgMaxEstSelection,
    ArgMaxObsSelection,
    Solution,
    SolutionSelection,
)
from deephyper.stopper import Stopper
from deephyper.skopt.moo import non_dominated_set

__all__ = ["Search", "SearchHistory"]

logger = logging.getLogger(__name__)


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
        solution_selection: Optional[SolutionSelection] = None,
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

    def set_num_objective(self, job):
        obj = job.objective
        if isinstance(obj, (tuple, list)):
            self.num_objective = len(obj)
        else:
            self.num_objective = 1
        if isinstance(self.solution_selection, SolutionSelection):
            self.solution_selection.num_objective = self.num_objective

    def extend(self, jobs: List[HPOJob]):
        # Do nothing if input list is empty
        if len(jobs) == 0:
            return

        if self.num_objective is None:
            self.set_num_objective(jobs[0])
        self.jobs.extend(jobs)

        if isinstance(self.solution_selection, SolutionSelection):
            self.solution_selection.update(jobs)
            for job in jobs:
                self.solution_history[job.id] = self.solution

    @property
    def solution(self) -> Solution | None:
        if isinstance(self.solution_selection, SolutionSelection):
            return self.solution_selection.solution
        else:
            return None

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
            if isinstance(self.solution_selection, SolutionSelection):
                if self.num_objective == 1:
                    sol = dict(self.solution_history[job.id])
                    parameters = sol.pop("parameters")
                    objective = sol.pop("objective")
                    if parameters is not None and objective is not None:
                        result.update({f"sol.p:{k}": v for k, v in parameters.items()})
                        result.update({"sol.objective": objective})
                        result.update({f"sol.{k}": v for k, v in sol.items() if v is not None})

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
        logger.info("Computing pareto efficient indicator...")
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
    """Base class search/optimization algorithms.

    Args:
        problem:
            object describing the search/optimization problem.

        random_state (np.random.RandomState, optional):
            Initial random state of the search. Defaults to ``None``.

        log_dir (str, optional):
            Path to the directoy where results of the search are stored. Defaults to ``"."``.

        verbose (int, optional):
            Use verbose mode. Defaults to ``0``.

        stopper (Stopper, optional):
            a stopper to leverage multi-fidelity when evaluating the function. Defaults
            to ``None`` which does not use any stopper.

        checkpoint_history_to_csv (bool):
            wether the results from progressively collected evaluations should be checkpointed
            regularly to disc as a csv. Defaults to ``True``.

        solution_selection (Literal["argmax_obs", "argmax_est"] | SolutionSelection, optional):
            the solution selection strategy. It can be a string where ``"argmax_obs"`` would
            select the argmax of observed objective values, and ``"argmax_est"`` would select the
            argmax of estimated objective values (through a predictive model).

        checkpoint_restart (bool): ...
    """

    def __init__(
        self,
        problem,
        random_state=None,
        log_dir: str = ".",
        verbose: int = 0,
        stopper: Optional[Stopper] = None,
        checkpoint_history_to_csv: bool = True,
        solution_selection: Optional[
            Literal["argmax_obs", "argmax_est"] | SolutionSelection
        ] = None,
        checkpoint_restart: bool = False,
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

        # Create logger directory if does not exist
        self._log_dir = os.path.abspath(log_dir)
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._verbose = verbose
        self.checkpoint_history_to_csv = checkpoint_history_to_csv
        self.checkpoint_restart = checkpoint_restart

        self.is_master = True

        # Check if results already exist
        self._path_results = os.path.join(self._log_dir, "results.csv")
        if (
            os.path.exists(self._path_results)
            and self.checkpoint_history_to_csv
            and not self.checkpoint_restart
        ):
            str_current_time = time.strftime("%Y%m%d-%H%M%S")
            path_results_dirname = os.path.dirname(self._path_results)
            path_results_basename = os.path.basename(self._path_results)
            path_results_basename = path_results_basename.replace(".", f"_{str_current_time}.")
            path_results_renamed = os.path.join(path_results_dirname, path_results_basename)
            logger.warning(
                f"Results file already exists, it will be renamed to {path_results_renamed}"
            )
            os.rename(
                self._path_results,
                path_results_renamed,
            )

        # Default setting is asynchronous
        self.gather_type = "BATCH"
        self.gather_batch_size = 1

        self._stopper = stopper

        self.stopped = False

        # Related to management of history of results
        if type(solution_selection) is str:
            if solution_selection == "argmax_obs":
                solution_selection = ArgMaxObsSelection()
            elif solution_selection == "argmax_est":
                solution_selection = ArgMaxEstSelection(
                    problem, random_state=self._random_state.randint(0, np.iinfo(np.int32).max)
                )
            else:
                raise ValueError(
                    f"{solution_selection=} should be in ['argmax_obs', 'argmax_est'] when a str."
                )
        elif isinstance(solution_selection, SolutionSelection):
            pass
        elif solution_selection is None:
            pass
        else:
            raise ValueError(
                f"{solution_selection=} should be a str or an instance of SolutionSelection"
            )
        self.history = SearchHistory(self._problem, solution_selection=solution_selection)

    def check_evaluator(self, evaluator):
        """Check if the input is a callable, an evaluator or else.

        :meta: private
        """
        if not (isinstance(evaluator, Evaluator)):
            if callable(evaluator):
                # Pick the adapted evaluator depending if the passed function is a coroutine
                if iscoroutinefunction(evaluator):
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

    def save_params(self, filename: str = "params.json"):
        """Save the search parameters to a JSON file in the log folder.

        Args:
            filename: Name of JSON file where search parameters are saved. Default is `params.json`.
        """
        if not filename.endswith(".json"):
            print("Invalid file type. File must be a JSON file.")
            return

        search_params = self.get_params()
        json_path = os.path.join(self._log_dir, filename)

        with open(json_path, "w") as f:
            json.dump(search_params, f, indent=2, sort_keys=True)

    def get_params(self) -> dict[str, Any]:
        """Get parameters used for the search object.

        Returns:
            A dictionary of the search parameters.
        """
        dict_self = {
            "search": {
                "type": type(self).__name__,
                **get_init_params_as_json(self),
            },
            "calls": self._call_args,
        }

        return dict_self

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
        evaluator,
        max_evals: int = -1,
        timeout: Optional[int | float] = None,
        max_evals_strict: bool = False,
    ) -> pd.DataFrame:
        """Execute the search algorithm.

        Args:
            evaluator:
                object describing the evaluation process.

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
        logger.info(f"Starting search with {type(self).__name__}")

        # Configure evaluator
        # if a callable is directly passed wrap it around the serial evaluator
        self.check_evaluator(evaluator)

        # Set the search object in the evaluator to be able to call it from Evaluator's callbacks
        self._evaluator.search = self

        # Set the stopper for the evaluator
        self._evaluator._stopper = self._stopper

        # Log problem and evaluator parameters
        logger.info(f"Search's problem: {self._problem}")
        logger.info(
            f"Search's evaluator: {type(self._evaluator).__name__} with "
            f"{self._evaluator.num_workers} worker(s)"
        )

        # Log remaining parameters
        params_dict = self.get_params()["search"]
        del params_dict["problem"]
        logger.info(
            f"Search's other parameters: {json.dumps(params_dict, indent=None, sort_keys=True)}"
        )  # noqa: E501

        self.stopped = False
        self._check_timeout(timeout)
        if max_evals_strict:
            # TODO: should be replaced by a property with a setter?
            self._evaluator.set_maximum_num_jobs_submitted(max_evals)

        # Reload checkpoint
        self.reload_checkpoint()

        # Save the search call arguments for the context
        self._call_args.append({"timeout": timeout, "max_evals": max_evals})
        if timeout is not None:
            logger.info(f"Running the search for {max_evals=} and {timeout=:.2f}")
        else:
            logger.info(f"Running the search for {max_evals=} and unlimited time...")

        # Init tqdm callback
        if max_evals > 1:
            for cb in self._evaluator._callbacks:
                if isinstance(cb, TqdmCallback):
                    cb.set_max_evals(max_evals)

        t_start_search = time.monotonic()
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
            logger.warning(
                "Search is being stopped because the maximum number of spawned jobs has been "
                "reached."
            )

        # Collect remaining jobs
        logger.info("Collect remaining jobs...")
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
            logger.warning("No results found in search history")
            return None

        self.dump_jobs_done_to_csv(flush=True)

        self.history.compute_pareto_efficiency()

        if self.checkpoint_history_to_csv:
            df_results = self.history.to_csv_complete(os.path.join(self._log_dir, "results.csv"))
        else:
            df_results = self.history.to_dataframe()

        logger.info(
            f"The search completed after {len(df_results)} evaluation(s) "
            f"and {time.monotonic() - t_start_search:.2f} sec."
        )

        return df_results

    @property
    def search_id(self):
        """The identifier of the search used by the evaluator."""
        return self._evaluator._search_id

    def _search(
        self, max_evals: int, timeout: Optional[int | float], max_evals_strict: bool = False
    ):
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

            logger.info(f"Submitting {len(new_batch)} configurations...")
            t1 = time.monotonic()
            self._evaluator.submit(new_batch)
            logger.info(f"Submition took {time.monotonic() - t1:.4f} sec.")

            logger.info("Gathering jobs...")
            t1 = time.monotonic()

            new_results = self._evaluator.gather(self.gather_type, self.gather_batch_size)

            # Check if results are received from other search instances
            # connected to the same storage
            if isinstance(new_results, tuple) and len(new_results) == 2:
                local_results, other_results = new_results
                n_ask = len(local_results)
                new_results = local_results + other_results
                logger.info(
                    f"Gathered {len(local_results)} local job(s) and {len(other_results)} other "
                    f"job(s) in {time.monotonic() - t1:.4f} sec."
                )
            else:
                n_ask = len(new_results)
                logger.info(
                    f"Gathered {len(new_results)} job(s) in {time.monotonic() - t1:.4f} sec."
                )

            # Tell comes before history.extend
            # Because the optimizer state needs to be updated to selection solutions
            # Try tell, if tell fails, execute finally then propagate error
            try:
                self.tell([(config, obj) for config, obj in new_results])
            except:
                raise
            finally:
                self.history.extend(new_results)

                logger.info("Dumping evaluations...")
                t1 = time.monotonic()
                self.dump_jobs_done_to_csv()
                logger.info(f"Dumping took {time.monotonic() - t1:.4f} sec.")

            # Test if search should be stopped due to timeout
            time_left = self._evaluator.time_left
            if time_left is not None and time_left <= 0:
                logger.info(
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
        logger.info(f"Asking {n} configuration(s)...")
        t1 = time.monotonic()

        new_samples = self._ask(n)

        logger.info(f"Asking took {time.monotonic() - t1:.4f} sec.")

        return new_samples

    @abc.abstractmethod
    def _ask(self, n: int = 1) -> list[dict[str, Optional[str | int | float]]]:
        """Ask the search for new configurations to evaluate.

        Args:
            n (int, optional): The number of configurations to ask. Defaults to 1.

        Returns:
            List[Dict]: a list of hyperparameter configurations to evaluate.
        """

    def tell(
        self,
        results: list[
            tuple[
                dict[str, Optional[str | int | float]], str | int | float | tuple[str | int | float]
            ]
        ],
    ):
        """Tell the search the results of the evaluations.

        Args:
            results (list[tuple[dict[str, Optional[str | int | float]], str | int | float]]):
                a dictionary containing the results of the evaluations.
        """
        logger.info(f"Telling {len(results)} new result(s)...")
        t1 = time.monotonic()
        self._tell(results)
        logger.info(f"Telling took {time.monotonic() - t1:.4f} sec.")

    @abc.abstractmethod
    def _tell(
        self,
        results: list[
            tuple[
                dict[str, Optional[str | int | float]], str | int | float | tuple[str | int | float]
            ]
        ],
    ):
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

    def reload_checkpoint(self):
        if not self.is_master:
            return

        assert self._evaluator is not None

        if not self.checkpoint_restart:
            return

        if os.path.exists(self._path_results):
            logging.info("Loading previous results.csv checkpoint")
            # Load previous results
            df = read_results_from_csv(self._path_results)

            evaluator = self._evaluator
            search_id = evaluator._search_id
            storage = evaluator._storage

            job_ids_storage = storage.load_all_job_ids(search_id)
            if len(job_ids_storage) == 0:
                # The storage is not Persistent so we reset the job counter
                for _ in range(df["job_id"].max() + 1):
                    storage.create_new_job(search_id)

            p_columns = [col for col in df.columns if col.startswith("p:")]
            p_metadata = [col for col in df.columns if col.startswith("m:")]
            p_objective = list(sorted([col for col in df.columns if col.startswith("objective")]))
            jobs = []
            for i, row in df.iterrows():
                job_id = f"{search_id}.{row.job_id}"
                # Set inputs
                job = HPOJob(
                    job_id,
                    {k[2:]: v for k, v in row[p_columns].to_dict().items()},
                    evaluator.run_function,
                    storage,
                )
                # Set outputs
                objective = row[p_objective].tolist()
                if len(objective) == 1:
                    objective = objective[0]
                job.set_output(
                    {
                        "objective": objective,
                        "metadata": {k[2:]: v for k, v in row[p_metadata].to_dict().items()},
                    }
                )
                # Set status
                job.status = JobStatus[row["job_status"]]
                jobs.append(job)

                evaluator.job_id_gathered.append(job_id)
            self.history.extend(jobs)

            x = [(config, obj) for config, obj in jobs]
            self.tell(x)
