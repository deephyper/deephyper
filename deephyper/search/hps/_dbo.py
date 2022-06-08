import logging
import os
import pathlib
import pickle
import signal
import time

import numpy as np
import pandas as pd
import deephyper.skopt
import ConfigSpace as CS

# avoid initializing mpi4py when importing
import mpi4py
import yaml

mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI

from deephyper.core.exceptions import SearchTerminationError
from deephyper.problem._hyperparameter import convert_to_skopt_space
from deephyper.core.utils._introspection import get_init_params_as_json
from sklearn.ensemble import GradientBoostingRegressor

TAG_INIT = 20
TAG_DATA = 30


MAP_acq_func = {
    "UCB": "LCB",
}


class History:
    """History"""

    def __init__(self) -> None:
        self._list_x = []  # vector of hyperparameters
        self._list_y = []  # objective values
        self._keys_infos = []  # keys
        self._list_infos = []  # values
        self.n_buffered = 0

    def append_keys_infos(self, k: list):
        self._keys_infos.extend(k)

    def get_keys_infos(self) -> list:
        return self._keys_infos

    def append(self, x, y, infos):
        self._list_x.append(x)
        self._list_y.append(y)
        self._list_infos.append(infos)
        self.n_buffered += 1

    def extend(self, x: list, y: list, infos: dict):
        self._list_x.extend(x)
        self._list_y.extend(y)

        infos = np.array([v for v in infos.values()], dtype="O").T.tolist()
        self._list_infos.extend(infos)
        self.n_buffered += len(x)

    def length(self):
        return len(self._list_x)

    def value(self):
        return self._list_x[:], self._list_y[:]

    def infos(self, k=None):
        list_infos = np.array(self._list_infos, dtype="O").T
        if k is not None:
            if k == 0:
                infos = {key: [] for key in self._keys_infos}
                return [], [], infos
            else:
                infos = {
                    key: val[-k:] for key, val in zip(self._keys_infos, list_infos)
                }
                return self._list_x[-k:], self._list_y[-k:], infos
        else:
            infos = {key: val for key, val in zip(self._keys_infos, list_infos)}
            return self._list_x, self._list_y, infos

    def reset_buffer(self):
        self.n_buffered = 0


# TODO: bring all parameters of the surrogate_model in surrogate_model_kwargs


class DBO:
    """Distributed Bayesian Optimization Search.

    Args:
        problem (HpProblem): Hyperparameter problem describing the search space to explore.
        run_function (callable): A callable instance which represents the black-box function we want to evaluate.
        random_state (int, optional): Random seed. Defaults to ``None``.
        log_dir (str, optional): Log directory where search's results are saved. Defaults to ``"."``.
        verbose (int, optional): Indicate the verbosity level of the search. Defaults to ``0``.
        comm (optional): The MPI communicator to use. Defaults to ``None``.
        run_function_kwargs (dict): Keyword arguments to pass to the run-function. Defaults to ``None``.
        n_jobs (int, optional): Parallel processes per rank to use for optimization updates (e.g., model re-fitting). Defaults to ``1``.
        surrogate_model (str, optional): Type of the surrogate model to use. ``"DUMMY"`` can be used of random-search, ``"GP"`` for Gaussian-Process (efficient with few iterations such as a hundred sequentially but bottleneck when scaling because of its cubic complexity w.r.t. the number of evaluations), "``"RF"`` for the Random-Forest regressor (log-linear complexity with respect to the number of evaluations). Defaults to ``"RF"``.
        lazy_socket_allocation (bool, optional): If `True` then MPI communication socket are initialized only when used for the first time, otherwise the initialization is forced when creating the instance. Defaults to ``False``.
        sync_communication (bool, optional): If `True`  workers communicate synchronously, otherwise workers communicate asynchronously. Defaults to ``False``.
        sync_communication_freq (int, optional): Manage the frequency at which workers should communicate their results in the case of synchronous communication. Defaults to ``10``.
        checkpoint_file (str): Name of the file in ``log_dir`` where results are checkpointed. Defaults to ``"results.csv"``.
        checkpoint_freq (int): Frequency at which results are checkpointed. Defaults to ``1``.
        acq_func (str): Acquisition function to use. If ``"UCB"`` then the upper confidence bound is used, if ``"EI"`` then the expected-improvement is used, if ``"PI"`` then the probability of improvement is used, if ``"gp_hedge"`` then probabilistically choose one of the above.
        acq_optimizer (str): Method use to optimise the acquisition function. If ``"sampling"`` then random-samples are drawn and infered for optimization, if ``"lbfgs"`` gradient-descent is used. Defaults to ``"auto"``.
        kappa (float): Exploration/exploitation value for UCB-acquisition function, the higher the more exploration, the smaller the more exploitation. Defaults to ``1.96`` which corresponds to a 95% confidence interval.
        xi (float): Exploration/exploitation value for EI and PI-acquisition functions, the higher the more exploration, the smaller the more exploitation. Defaults to ``0.001``.
        sample_max_size (int): Maximum size of the number of samples used to re-fit the surrogate model. Defaults to ``-1`` for infinite sample size.
        sample_strategy (str): Sub-sampling strategy to re-fit the surrogate model. If ``"quantile"`` then sub-sampling is performed based on the quantile of the collected objective values. Defaults to ``"quantile"``.

    """

    def __init__(
        self,
        problem,
        run_function,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        comm=None,
        run_function_kwargs: dict = None,
        n_jobs: int = 1,
        surrogate_model: str = "RF",
        surrogate_model_kwargs: dict = None,
        n_initial_points: int = 10,
        lazy_socket_allocation: bool = False,
        communication_batch_size=2048,
        sync_communication: bool = False,
        sync_communication_freq: int = 10,
        checkpoint_file: str = "results.csv",
        checkpoint_freq: int = 1,
        acq_func: str = "UCB",
        acq_optimizer: str = "auto",
        kappa: float = 1.96,
        xi: float = 0.001,
        sample_max_size: int = -1,
        sample_strategy: str = "quantile",
    ):

        # get the __init__ parameters
        self._init_params = locals()
        self._call_args = []

        self._problem = problem
        self._run_function = run_function
        self._run_function_kwargs = (
            {} if run_function_kwargs is None else run_function_kwargs
        )

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

        # mpi
        if not MPI.Is_initialized():
            MPI.Init_thread()
        self._comm = comm if comm else MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        self._communication_batch_size = communication_batch_size
        logging.info(f"DMBSMPI has {self._size} worker(s)")

        # force socket allocation with dummy message to reduce overhead
        if not lazy_socket_allocation:
            logging.info("Initializing communication...")
            ti = time.time()
            logging.info("Sending to all...")
            t1 = time.time()
            req_send = [
                self._comm.isend(None, dest=i, tag=TAG_INIT)
                for i in range(self._size)
                if i != self._rank
            ]
            MPI.Request.waitall(req_send)
            logging.info(f"Sending to all done in {time.time() - t1:.4f} sec.")

            logging.info("Receiving from all...")
            t1 = time.time()
            req_recv = [
                self._comm.irecv(source=i, tag=TAG_INIT)
                for i in range(self._size)
                if i != self._rank
            ]
            MPI.Request.waitall(req_recv)
            logging.info(f"Receiving from all done in {time.time() - t1:.4f} sec.")
            logging.info(
                f"Initializing communications done in {time.time() - ti:.4f} sec."
            )

        # sync communication management
        self._sync_communication = sync_communication
        self._sync_communication_freq = sync_communication_freq

        # checkpointing
        self._checkpoint_size = 0
        self._checkpoint_file = checkpoint_file
        self._checkpoint_freq = checkpoint_freq

        # set random state for given rank
        self._rank_seed = self._random_state.randint(
            low=0, high=2**32, size=self._size
        )[self._rank]

        self._timestamp = time.time()

        self._history = History()

        if acq_optimizer == "auto":
            if acq_func == "qUCB":
                acq_optimizer = "sampling"
            else:
                acq_optimizer = "boltzmann_sampling"

        if acq_func == "qUCB":
            kappa = self._random_state.exponential(kappa, size=self._size)[self._rank]
            acq_func = "UCB"

        # check if it is possible to convert the ConfigSpace to standard skopt Space
        if (
            isinstance(self._problem.space, CS.ConfigurationSpace)
            and len(self._problem.space.get_forbiddens()) == 0
            and len(self._problem.space.get_conditions()) == 0
        ):
            self._opt_space = convert_to_skopt_space(self._problem.space)
        else:
            self._opt_space = self._problem.space

        self._opt = None
        self._opt_kwargs = dict(
            dimensions=self._opt_space,
            base_estimator=self._get_surrogate_model(
                surrogate_model,
                surrogate_model_kwargs,
                n_jobs,
            ),
            acq_func=MAP_acq_func.get(acq_func, acq_func),
            acq_func_kwargs={"xi": xi, "kappa": kappa},
            acq_optimizer=acq_optimizer,
            acq_optimizer_kwargs={
                "n_points": 10000,
                "boltzmann_gamma": 1,
                # "boltzmann_psucc": 1/self._size,
                "n_jobs": n_jobs,
            },
            n_initial_points=n_initial_points,
            random_state=self._rank_seed,
            sample_max_size=sample_max_size,
            sample_strategy=sample_strategy,
        )

    def to_json(self):
        """Returns a json version of the search object."""
        json_self = {
            "search": {
                "type": type(self).__name__,
                "num_workers": self._size,
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

    def send_all(self, x, y, infos):
        logging.info("Sending to all...")
        t1 = time.time()

        data = (x, y, infos)
        data = MPI.pickle.dumps(data)

        # batched version
        if self._communication_batch_size > 0:

            size_processed = 0
            while size_processed < self._size:

                batch_size = min(
                    self._size - size_processed, self._communication_batch_size
                )

                req_send = [
                    self._comm.Isend(data, dest=i, tag=TAG_DATA)
                    for i in range(size_processed, size_processed + batch_size)
                    if i != self._rank
                ]
                MPI.Request.waitall(req_send)

                size_processed += batch_size
                logging.info(f"Processed {size_processed/self._size*100:.2f}%")

        # not batched
        else:

            req_send = [
                self._comm.Isend(data, dest=i, tag=TAG_DATA)
                for i in range(self._size)
                if i != self._rank
            ]
            MPI.Request.waitall(req_send)

        logging.info(f"Sending to all done in {time.time() - t1:.4f} sec.")

    def recv_any(self):
        logging.info("Receiving from any...")
        t1 = time.time()

        n_received = 0
        received_any = self._size > 1

        # batched version
        if self._communication_batch_size > 0:

            while received_any:

                received_any = False
                size_processed = 0

                while size_processed < self._size:

                    batch_size = min(
                        self._size - size_processed, self._communication_batch_size
                    )

                    req_recv = [
                        self._comm.irecv(source=i, tag=TAG_DATA)
                        for i in range(size_processed, size_processed + batch_size)
                        if i != self._rank
                    ]

                    # asynchronous
                    for i, req in enumerate(req_recv):
                        try:
                            done, data = req.test()
                            if done:
                                received_any = True
                                n_received += 1
                                x, y, infos = data
                                self._history.append(x, y, infos)
                            else:
                                req.cancel()
                        except pickle.UnpicklingError as e:
                            logging.error(f"UnpicklingError for request {i}")

                    size_processed += batch_size
                    logging.info(f"Processed {size_processed/self._size*100:.2f}%")

        else:

            while received_any:

                received_any = False
                req_recv = [
                    self._comm.irecv(source=i, tag=TAG_DATA)
                    for i in range(self._size)
                    if i != self._rank
                ]

                # asynchronous
                for i, req in enumerate(req_recv):
                    try:
                        done, data = req.test()
                        if done:
                            received_any = True
                            n_received += 1
                            x, y, infos = data
                            self._history.append(x, y, infos)
                        else:
                            req.cancel()
                    except pickle.UnpicklingError as e:
                        logging.error(f"UnpicklingError for request {i}")
        logging.info(
            f"Received {n_received} configurations in {time.time() - t1:.4f} sec."
        )

    def broadcast(self, X: list, Y: list, infos: list):
        logging.info("Broadcasting to all...")
        t1 = time.time()
        data = self._comm.allgather((X, Y, infos))
        n_received = 0

        for i in range(self._size):
            if i != self._rank:
                self._history.extend(*data[i])
                n_received += len(data[i][0])

        logging.info(
            f"Broadcast received {n_received} configurations in {time.time() - t1:.4f} sec."
        )

    def broadcast_to_root(self, X: list, Y: list, infos: list):
        logging.info("Broadcasting to root all...")
        t1 = time.time()

        if self._rank == 0:
            data = self._comm.gather((X, Y, infos), root=0)
            n_received = 0

            for i in range(self._size):
                if i != self._rank:
                    self._history.extend(*data[i])
                    n_received += len(data[i][0])

            logging.info(
                f"Broadcast to root received {n_received} configurations in {time.time() - t1:.4f} sec."
            )
        else:
            self._comm.gather((X, Y, infos), root=0)
            logging.info(f"Broadcast to root done in {time.time() - t1:.4f} sec.")

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

        # save the search call arguments for the context
        self._call_args.append({"timeout": timeout, "max_evals": max_evals})
        # save the context in the log folder
        self.dump_context()

        try:
            self._search(max_evals, timeout)
        except SearchTerminationError:
            logging.error("Processing SearchTerminationError")
            if not (self._sync_communication):
                self.recv_any()

        if self._rank == 0:
            return self.checkpoint()
        else:
            return None

    def _setup_optimizer(self):
        # if self._fitted:
        #     self._opt_kwargs["n_initial_points"] = 0
        self._opt = deephyper.skopt.Optimizer(**self._opt_kwargs)

    def _search(self, max_evals, timeout):

        if self._opt is None:
            self._setup_optimizer()

        logging.info("Asking 1 configuration...")
        t1 = time.time()
        x = self._opt.ask()
        logging.info(f"Asking took {time.time() - t1:.4f} sec.")

        logging.info("Executing the run-function...")
        t1 = time.time()
        y = self._run_function(self.to_dict(x), **self._run_function_kwargs)
        logging.info(f"Execution took {time.time() - t1:.4f} sec.")

        infos = [self._rank]
        self._history.append_keys_infos(["worker_rank"])

        # code to manage the @profile decorator
        profile_keys = ["objective", "timestamp_start", "timestamp_end"]
        if isinstance(y, dict) and all(k in y for k in profile_keys):
            profile = y
            y = profile["objective"]
            timestamp_start = profile["timestamp_start"] - self._timestamp
            timestamp_end = profile["timestamp_end"] - self._timestamp
            infos.extend([timestamp_start, timestamp_end])

            self._history.append_keys_infos(profile_keys[1:])

        y = -y  #! we do maximization

        self._history.append(x, y, infos)

        if self._sync_communication:
            if self._history.n_buffered % self._sync_communication_freq == 0:
                self.broadcast(*self._history.infos(k=self._history.n_buffered))
                self._history.reset_buffer()
        else:
            self.send_all(x, y, infos)
            self._history.reset_buffer()

        while max_evals < 0 or self._history.length() < max_evals:

            # collect x, y from other nodes (history)
            if not (self._sync_communication):
                self.recv_any()

            hist_X, hist_y = self._history.value()
            n_new = len(hist_X) - len(self._opt.Xi)

            logging.info("Fitting the optimizer...")
            t1 = time.time()
            self._opt.tell(hist_X[-n_new:], hist_y[-n_new:])
            logging.info(f"Fitting took {time.time() - t1:.4f} sec.")

            # ask next configuration
            logging.info("Asking 1 configuration...")
            t1 = time.time()
            x = self._opt.ask()
            logging.info(f"Asking took {time.time() - t1:.4f} sec.")

            logging.info("Executing the run-function...")
            t1 = time.time()
            y = self._run_function(self.to_dict(x), **self._run_function_kwargs)
            logging.info(f"Execution took {time.time() - t1:.4f} sec.")
            infos = [self._rank]

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
            self._history.append(x, y, infos)

            # checkpointing
            if self._rank == 0 and self._history.length() % self._checkpoint_freq == 0:
                self.checkpoint()

            if self._sync_communication:
                if self._history.n_buffered % self._sync_communication_freq == 0:
                    self.broadcast(*self._history.infos(k=self._history.n_buffered))
                    self._history.reset_buffer()
            else:
                self.send_all(x, y, infos)
                self._history.reset_buffer()

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

    def gather_results(self):
        x_list, y_list, infos_dict = self._history.infos()
        x_list = np.transpose(np.array(x_list))
        y_list = -np.array(y_list)

        results = {
            hp_name: x_list[i]
            for i, hp_name in enumerate(self._problem.hyperparameter_names)
        }
        results.update(dict(objective=y_list, **infos_dict))

        results = pd.DataFrame(data=results, index=list(range(len(y_list))))
        return results

    def _get_surrogate_model(
        self,
        name: str,
        surrogate_model_kwargs: dict = None,
        n_jobs: int = 1,
    ):
        """Get a surrogate model from Scikit-Optimize.

        Args:
            name (str): name of the surrogate model.
            n_jobs (int): number of parallel processes to distribute the computation of the surrogate model.

        Raises:
            ValueError: when the name of the surrogate model is unknown.
        """
        accepted_names = ["RF", "ET", "GBRT", "DUMMY", "GP"]
        if not (name in accepted_names):
            raise ValueError(
                f"Unknown surrogate model {name}, please choose among {accepted_names}."
            )

        if name == "RF":
            default_kwargs = dict(
                n_estimators=100,
                min_samples_leaf=3,
                n_jobs=n_jobs,
                max_features="auto",
                random_state=self._rank_seed,
            )
            if surrogate_model_kwargs is not None:
                default_kwargs.update(surrogate_model_kwargs)
            surrogate = deephyper.skopt.learning.RandomForestRegressor(**default_kwargs)
        elif name == "ET":
            default_kwargs = dict(
                n_estimators=100,
                min_samples_leaf=3,
                n_jobs=n_jobs,
                max_features="log2",
                random_state=self._rank_seed,
            )
            if surrogate_model_kwargs is not None:
                default_kwargs.update(surrogate_model_kwargs)
            surrogate = deephyper.skopt.learning.ExtraTreesRegressor(**default_kwargs)
        elif name == "GBRT":
            default_kwargs = dict(
                n_jobs=n_jobs,
                random_state=self._rank_seed,
            )
            if surrogate_model_kwargs is not None:
                default_kwargs.update(surrogate_model_kwargs)
            gbrt = GradientBoostingRegressor(n_estimators=30, loss="quantile")
            surrogate = deephyper.skopt.learning.GradientBoostingQuantileRegressor(
                base_estimator=gbrt, **default_kwargs
            )
        else:  # for DUMMY and GP
            surrogate = name

        return surrogate

    def fit_surrogate(self, df):
        """Fit the surrogate model of the search from a checkpointed Dataframe.

        Args:
            df (str|DataFrame): a checkpoint from a previous search.

        Example Usage:

        >>> search = CBO(problem, evaluator)
        >>> search.fit_surrogate("results.csv")
        """
        if type(df) is str and df[-4:] == ".csv":
            df = pd.read_csv(df, index_col=0)
        assert isinstance(df, pd.DataFrame)

        self._fitted = True

        if self._opt is None:
            self._setup_optimizer()

        hp_names = self._problem.hyperparameter_names
        try:
            x = df[hp_names].values.tolist()
            y = df.objective.tolist()
        except KeyError:
            raise ValueError(
                "Incompatible dataframe 'df' to fit surrogate model of CBO."
            )

        self._opt.tell(x, [-yi for yi in y])

    def checkpoint(self):
        """Dump evaluations to a CSV file.``"""
        logging.info("Checkpointing starts...")

        path_results = os.path.join(self._log_dir, self._checkpoint_file)
        results = self.gather_results()
        results.to_csv(path_results)

        logging.info("Checkpointing done")

        return results
