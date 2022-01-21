import logging
import os
import pathlib
import signal
import time

import numpy as np
import pandas as pd
import ray
import skopt
from deephyper.core.exceptions import SearchTerminationError


@ray.remote(num_cpus=1)
class DList:
    """Distributed List"""

    def __init__(self) -> None:
        self._list_x = []  # vector of hyperparameters
        self._list_y = []  # objective values
        self._keys_infos = []  # keys
        self._list_infos = []  # values

    def append_keys_infos(self, k: list):
        self._keys_infos.extend(k)

    def get_keys_infos(self) -> list:
        return self._keys_infos

    def append(self, x, y, infos):
        self._list_x.append(x)
        self._list_y.append(y)
        self._list_infos.append(infos)

    def length(self):
        return len(self._list_x)

    def value(self):
        return self._list_x, self._list_y

    def infos(self):
        list_infos = np.array(self._list_infos).T
        infos = {k: v for k, v in zip(self._keys_infos, list_infos)}
        return self._list_x, self._list_y, infos


@ray.remote
class Worker:
    """Worker

    Args:
        id (int): id of the worker.
        history (ObjectRef of DList): distributed list.
        problem (HpProblem): problem.
        random_state (int): random seed.
        run_function (callable): run-function.
        log_dir (...): ...
        verbose (...): ...
    """

    def __init__(
        self,
        id,
        history,
        timestamp,
        problem,
        run_function,
        random_state,
        log_dir,
        verbose,
        acq_optimizer="sampling"
    ) -> None:
        self._id = id
        self._history = history  # history of [(x, y)...] configurations

        self._problem = problem
        self._run_function = run_function
        self._rng = np.random.RandomState(random_state)
        self._log_dir = log_dir
        self._verbose = verbose

        self._opt = None
        self._opt_space = self._problem.space
        self._opt_kwargs = dict(
            dimensions=self._opt_space,
            base_estimator="RF",
            acq_func="LCB",
            acq_optimizer=acq_optimizer,
            acq_optimizer_kwargs={"n_points": 10000},
            n_initial_points=1,
            random_state=random_state,
        )
        self._timestamp = timestamp  # Recorded time of when this worker interface was created.

    def _setup_optimizer(self):
        # if self._fitted:
        #     self._opt_kwargs["n_initial_points"] = 0
        self._opt = skopt.Optimizer(**self._opt_kwargs)

    def search(self, max_evals, timeout):
        
        def handler(signum, frame): pass

        signal.signal(signal.SIGALRM, handler)

        if self._opt is None:
            self._setup_optimizer()

        x = self._opt.ask()
        y = self._run_function(self.to_dict(x))
        infos = [self._id]
        if self._id == 0:  # only the first worker updates the keys of infos values
            ray.get(self._history.append_keys_infos.remote(["worker_id"]))

        # code to manage the profile decorator
        profile_keys = ["objective", "timestamp_start", "timestamp_end"]
        if isinstance(y, dict) and all(k in y for k in profile_keys):
            profile = y
            y = profile["objective"]
            timestamp_start = profile["timestamp_start"] - self._timestamp
            timestamp_end = profile["timestamp_end"] - self._timestamp
            infos.extend([timestamp_start, timestamp_end])

            if self._id == 0:  # only the first worker updates the keys of infos values
                ray.get(self._history.append_keys_infos.remote(profile_keys[1:]))

        y = -y  #! we do maximization

        ray.get(self._history.append.remote(x, y, infos))

        while max_evals < 0 or ray.get(self._history.length.remote()) < max_evals:

            # collect x, y from other nodes (history)
            hist_X, hist_y = ray.get(self._history.value.remote())

            # fit optimizer
            self._opt.Xi = []
            self._opt.yi = []
            self._opt.sampled = hist_X  # avoid duplicated samples
            self._opt.tell(hist_X, hist_y)

            # ask next configuration
            x = self._opt.ask()
            y = self._run_function(self.to_dict(x))
            infos = [self._id]

            # code to manage the profile decorator
            profile_keys = ["objective", "timestamp_start", "timestamp_end"]
            if isinstance(y, dict) and all(k in y for k in profile_keys):
                profile = y
                y = profile["objective"]
                timestamp_start = profile["timestamp_start"] - self._timestamp
                timestamp_end = profile["timestamp_end"] - self._timestamp
                infos.extend([timestamp_start, timestamp_end])

            y = -y  #! we do maximization

            # update shared history
            ray.get(self._history.append.remote(x, y, infos))

    def to_dict(self, x: list) -> dict:
        """Transform a list of hyperparameter values to a ``dict`` where keys are hyperparameters names and values are hyperparameters values.

        :meta private:

        Args:
            x (list): a list of hyperparameter values.

        Returns:
            dict: a dictionnary of hyperparameter names and values.
        """
        res = {}
        hps_names = self._problem.hyperparameter_names
        for i in range(len(x)):
            res[hps_names[i]] = x[i]
        return res


class DMBS:
    """Distributed Model-Based Search based on the `Scikit-Optimized Optimizer <https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html#skopt.Optimizer>`_.

    Args:
        problem (HpProblem): Hyperparameter problem describing the search space to explore.
        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.
        random_state (int, optional): Random seed. Defaults to ``None``.
        log_dir (str, optional): Log directory where search's results are saved. Defaults to ``"."``.
        verbose (int, optional): Indicate the verbosity level of the search. Defaults to ``0``.
    """

    def __init__(
        self,
        problem,
        run_function,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        num_workers: int = 1,
        resources_per_worker: dict = None,
    ):

        self._history = None
        self._problem = problem
        self._run_function = run_function

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

        self._num_workers = num_workers
        self._resources_per_worker = (
            {"num_cpus": 1} if resources_per_worker is None else resources_per_worker
        )
        self._workers_refs = None
        self._timestamp = time.time()

    def terminate(self):
        """Terminate the search.

        Raises:
            SearchTerminationError: raised when the search is terminated with SIGALARM
        """
        logging.info("Search is being stopped!")

        if self._workers_refs is not None:
            for ref in self._workers_refs:
                ray.kill(ref)

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
            DataFrame: a pandas DataFrame containing the evaluations performed.
        """
        if timeout is not None:
            if type(timeout) is not int:
                raise ValueError(
                    f"'timeout' shoud be of type'int' but is of type '{type(timeout)}'!"
                )
            if timeout <= 0:
                raise ValueError(f"'timeout' should be > 0!")

        self._set_timeout(timeout)

        try:
            self._search(max_evals, timeout)
        except ray.exceptions.RayActorError as ex: pass

        path_results = os.path.join(self._log_dir, "results.csv")
        results = self.gather_results()
        results.to_csv(path_results)
        return results

    def _search(self, max_evals, timeout):

        # shared history
        if self._history is None:
            self._history = DList.remote()

        # initialize remote workers
        create_worker = lambda id: Worker.options(**self._resources_per_worker).remote(
            id,
            self._history,
            self._timestamp,
            self._problem,
            self._run_function,
            self._random_state.randint(0, 2 ** 32),  # upper bound is exclusive
            self._log_dir,
            self._verbose,
        )
        self._workers_refs = [create_worker(i) for i in range(self._num_workers)]
        search_refs = [w.search.remote(max_evals, timeout) for w in self._workers_refs]

        # run the search process for each worker
        search_done, search_processing = ray.wait(search_refs, num_returns=1)

        # terminate other workers as soon as the first is done because it means
        # we reached the correct number of evaluations
        for search_ref in search_processing:
                ray.kill(self._workers_refs[search_refs.index(search_ref)]) 

    def gather_results(self):
        x_list, y_list, infos_dict = ray.get(self._history.infos.remote())
        x_list = np.array(x_list)
        y_list = -np.array(y_list)

        results = {
            hp_name: x_list[:, i]
            for i, hp_name in enumerate(self._problem.hyperparameter_names)
        }
        results.update(dict(objective=y_list, **infos_dict))
        results = pd.DataFrame(results)
        return results
