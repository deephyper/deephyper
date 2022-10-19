import logging
from multiprocessing.sharedctypes import Value


# avoid initializing mpi4py when importing
import numpy as np
from deephyper.search.hps._cbo import CBO
from deephyper.evaluator._distributed import distributed
from deephyper.evaluator import Evaluator, SerialEvaluator
from deephyper.evaluator.callback import TqdmCallback


MAP_acq_func = {
    "UCB": "LCB",
}


class DBO(CBO):
    """Distributed Bayesian Optimization Search.

    Args:
        problem (HpProblem): Hyperparameter problem describing the search space to explore.
        run_function (callable): A callable instance which represents the black-box function we want to evaluate.
        random_state (int, optional): Random seed. Defaults to ``None``.
        log_dir (str, optional): Log directory where search's results are saved. Defaults to ``"."``.
        verbose (int, optional): Indicate the verbosity level of the search. Defaults to ``0``.
        comm (optional): The MPI communicator to use. Defaults to ``None``.
        run_function_kwargs (dict): Keyword arguments to pass to the run-function. Defaults to ``None``.
        n_jobs (int, optional): Parallel processes per rank to use for optimization updates (e.g., model re-fitting). Not used in ``surrogate_model`` if passed as own sklearn regressor. Defaults to ``1``.
        surrogate_model (Union[str,sklearn.base.RegressorMixin], optional): Type of the surrogate model to use. Can be a value in ``["RF", "GP", "ET", "GBRT", "DUMMY"]`` or a sklearn regressor. ``"DUMMY"`` can be used of random-search, ``"GP"`` for Gaussian-Process (efficient with few iterations such as a hundred sequentially but bottleneck when scaling because of its cubic complexity w.r.t. the number of evaluations), "``"RF"`` for the Random-Forest regressor (log-linear complexity with respect to the number of evaluations). Defaults to ``"RF"``.
        n_initial_points (int, optional): Number of collected objectives required before fitting the surrogate-model. Defaults to ``10``.
        initial_point_generator (str, optional): Sets an initial points generator. Can be either ``["random", "sobol", "halton", "hammersly", "lhs", "grid"]``. Defaults to ``"random"``.
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
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        surrogate_model="RF",
        acq_func: str = "UCB",
        acq_optimizer: str = "auto",
        kappa: float = 1.96,
        xi: float = 0.001,
        n_points: int = 10000,
        filter_duplicated: bool = True,
        update_prior: bool = False,
        multi_point_strategy: str = "cl_max",
        n_jobs: int = 1,
        n_initial_points: int = 10,
        initial_point_generator: str = "random",
        initial_points=None,
        sync_communication: bool = False,
        filter_failures: str = "mean",
        max_failures: int = 100,
        moo_scalarization_strategy: str = "Chebyshev",
        moo_scalarization_weight=None,
        **kwargs,
    ):
        # get the __init__ parameters
        _init_params = locals()

        self.check_evaluator(evaluator)

        if type(random_state) is int:
            random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            random_state = random_state
        else:
            random_state = np.random.RandomState()

        # set random state for given rank
        random_state = np.random.RandomState(
            random_state.randint(low=0, high=2 ** 32, size=self._evaluator.size)[
                self._evaluator.rank
            ]
        )

        if acq_optimizer == "auto":
            if acq_func[0] == "q":
                acq_optimizer = "sampling"
            else:
                acq_optimizer = "boltzmann_sampling"

        if acq_func[0] == "q":
            kappa = random_state.exponential(kappa, size=self._evaluator.size)[
                self._evaluator.rank
            ]
            xi = random_state.exponential(xi, size=self._evaluator.size)[
                self._evaluator.rank
            ]
            acq_func = acq_func[1:]

        if self._evaluator.rank == 0:
            super().__init__(
                problem,
                evaluator,
                random_state,
                log_dir,
                verbose,
                surrogate_model,
                acq_func,
                acq_optimizer,
                kappa,
                xi,
                n_points,
                filter_duplicated,
                update_prior,
                multi_point_strategy,
                n_jobs,
                n_initial_points,
                initial_point_generator,
                initial_points,
                sync_communication,
                filter_failures,
                max_failures,
                moo_scalarization_strategy,
                moo_scalarization_weight,
                **kwargs,
            )
        self._evaluator.comm.Barrier()
        if self._evaluator.rank != 0:
            super().__init__(
                problem,
                evaluator,
                random_state,
                log_dir,
                verbose,
                surrogate_model,
                acq_func,
                acq_optimizer,
                kappa,
                xi,
                n_points,
                filter_duplicated,
                update_prior,
                multi_point_strategy,
                n_jobs,
                n_initial_points,
                initial_point_generator,
                initial_points,
                sync_communication,
                filter_failures,
                max_failures,
                moo_scalarization_strategy,
                moo_scalarization_weight,
                **kwargs,
            )
        self._evaluator.comm.Barrier()

        # Replace CBO _init_params by DBO _init_params
        self._init_params = _init_params

        logging.info(
            f"DBO has {self._evaluator.num_total_workers} worker(s) with {self._evaluator.num_workers} local worker(s) per rank"
        )

    def check_evaluator(self, evaluator):
        super().check_evaluator(evaluator)

        if not (isinstance(evaluator, Evaluator)) and callable(evaluator):
            self._evaluator = distributed(backend="mpi")(SerialEvaluator)(evaluator)
            if self._evaluator.rank == 0:
                self._evaluator._callbacks.append(TqdmCallback())
        else:
            if not ("Distributed" in type(evaluator).__name__):
                raise ValueError(
                    "The evaluator must is not distributed! Use deephyper.evaluator.distributed(backend)(evaluator_class)!"
                )

            self._evaluator = evaluator
