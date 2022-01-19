import os
import pathlib
import time

import numpy as np
import pandas as pd
import ray
import skopt


@ray.remote(num_cpus=1)
class DList:
    """Distributed List"""

    def __init__(self) -> None:
        self._list_x = []
        self._list_y = []
        self._list_infos = []

    def append(self, x, y, infos):
        self._list_x.append(x)
        self._list_y.append(y)
        self._list_infos.append(infos)

    def length(self):
        return len(self._list_x)

    def value(self):
        return self._list_x, self._list_y

    def infos(self):
        return self._list_x, self._list_y, self._list_infos


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
        self, id, history, problem, run_function, random_state, log_dir, verbose
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
            acq_optimizer="softmax_sampling",
            acq_optimizer_kwargs={
                "n_points": 10000
            },
            n_initial_points=1,
            random_state=random_state,
        )
        self._timestamp = (
            time.time()
        )  # Recorded time of when this worker interface was created.

    def _setup_optimizer(self):
        # if self._fitted:
        #     self._opt_kwargs["n_initial_points"] = 0
        self._opt = skopt.Optimizer(**self._opt_kwargs)

    def search(self, max_evals, timeout):

        if self._opt is None:
            self._setup_optimizer()

        x = self._opt.ask()
        duration = time.perf_counter()
        y = -self._run_function(self.to_dict(x))
        duration = time.perf_counter() - duration
        elapsed_sec = time.time() - self._timestamp
        infos = [duration, elapsed_sec]

        ray.get(self._history.append.remote(x, y, infos))

        while max_evals < 0 or ray.get(self._history.length.remote()) < max_evals:

            # collect x, y from other nodes (history)
            hist_X, hist_y = ray.get(self._history.value.remote())

            # fit optimizer
            self._opt.Xi = []
            self._opt.yi = []
            self._opt.tell(hist_X, hist_y)

            # ask next configuration
            x = self._opt.ask()
            duration = time.perf_counter()
            y = -self._run_function(self.to_dict(x))
            duration = time.perf_counter() - duration
            elapsed_sec = time.time() - self._timestamp
            infos = [duration, elapsed_sec]

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
        resources_per_worker: dict = None
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
        self._resources_per_worker = {"num_cpus": 1} if resources_per_worker is None else resources_per_worker

    def search(self, max_evals, timeout):

        # shared history
        if self._history is None:
            self._history = DList.remote()

        # initialize remote workers
        create_worker = lambda id: Worker.options(**self._resources_per_worker).remote(
            id,
            self._history,
            self._problem,
            self._run_function,
            self._random_state.randint(0, 2 ** 32),  # upper bound is exclusive
            self._log_dir,
            self._verbose,
        )
        workers = [create_worker(i) for i in range(self._num_workers)]

        # run the search process for each worker
        ray.get([w.search.remote(max_evals, timeout) for w in workers])

        # return the results
        results = self.gather_results()

        return results

    def gather_results(self):
        x_list, y_list, infos_list = ray.get(self._history.infos.remote())
        x_list = np.array(x_list)
        y_list = -np.array(y_list)
        infos_list = np.transpose(infos_list)
        results = {
            hp_name: x_list[:, i]
            for i, hp_name in enumerate(self._problem.hyperparameter_names)
        }
        results.update(
            {
                "objective": y_list,
                "duration": infos_list[0],
                "elapsed_sec": infos_list[1],
            }
        )
        results = pd.DataFrame(results)
        return results


# for test
def run_test(config):
    return config["x"]


if __name__ == "__main__":
    # problem definition
    from deephyper.problem import HpProblem

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    # initialize ray
    ray.init(num_cpus=8)

    search = DMBS(problem, run_test, random_state=42)

    results = search.search(max_evals=100, timeout=None)
    results.to_csv("results.csv")

    print(results)
