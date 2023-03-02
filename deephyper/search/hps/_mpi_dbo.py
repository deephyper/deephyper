import logging

import mpi4py
import numpy as np
import scipy.stats

# !To avoid initializing MPI when module is imported (MPI is optional)
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI  # noqa: E402

from deephyper.evaluator import Evaluator  # noqa: E402
from deephyper.evaluator.callback import TqdmCallback  # noqa: E402
from deephyper.evaluator.storage import Storage  # noqa: E402
from deephyper.search.hps._cbo import CBO  # noqa: E402
from deephyper.stopper import Stopper  # noqa: E402

MAP_acq_func = {
    "UCB": "LCB",
}


class MPIDistributedBO(CBO):
    """Distributed Bayesian Optimization Search using MPI to launch parallel search instances.

    Args:
        problem (HpProblem): Hyperparameter problem describing the search space to explore.
        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.
        random_state (int, optional): Random seed. Defaults to ``None``.
        log_dir (str, optional): Log directory where search's results are saved. Defaults to ``"."``.
        verbose (int, optional): Indicate the verbosity level of the search. Defaults to ``0``.
        surrogate_model (Union[str,sklearn.base.RegressorMixin], optional): Surrogate model used by the Bayesian optimization. Can be a value in ``["RF", "GP", "ET", "GBRT", "DUMMY"]`` or a sklearn regressor. ``"RF"`` is for Random-Forest which is the best compromise between speed and quality when performing a lot of parallel evaluations, i.e., reaching more than hundreds of evaluations. ``"GP"`` is for Gaussian-Process which is the best choice when maximizing the quality of iteration but quickly slow down when reaching hundreds of evaluations, also it does not support conditional search space. ``"ET"`` is for Extra-Tree, faster than random forest but with worse mean estimate and poor uncertainty quantification capabilities. ``"GBRT"`` is for Gradient-Boosting Regression Tree, it has better mean estimate than other tree-based method worse uncertainty quantification capabilities and slower than ``"RF"``. Defaults to ``"RF"``.
        acq_func (str, optional): Acquisition function used by the Bayesian optimization. Can be a value in ``["UCB", "EI", "PI", "gp_hedge"]``. Defaults to ``"UCB"``.
        acq_optimizer (str, optional): Method used to minimze the acquisition function. Can be a value in ``["sampling", "lbfgs"]``. Defaults to ``"auto"``.
        kappa (float, optional): Manage the exploration/exploitation tradeoff for the "UCB" acquisition function. Defaults to ``1.96`` which corresponds to 95% of the confidence interval.
        xi (float, optional): Manage the exploration/exploitation tradeoff of ``"EI"`` and ``"PI"`` acquisition function. Defaults to ``0.001``.
        n_points (int, optional): The number of configurations sampled from the search space to infer each batch of new evaluated configurations.
        filter_duplicated (bool, optional): Force the optimizer to sample unique points until the search space is "exhausted" in the sens that no new unique points can be found given the sampling size ``n_points``. Defaults to ``True``.
        multi_point_strategy (str, optional): Definition of the constant value use for the Liar strategy. Can be a value in ``["cl_min", "cl_mean", "cl_max", "qUCB"]``. All ``"cl_..."`` strategies follow the constant-liar scheme, where if $N$ new points are requested, the surrogate model is re-fitted $N-1$ times with lies (respectively, the minimum, mean and maximum objective found so far; for multiple objectives, these are the minimum, mean and maximum of the individual objectives) to infer the acquisition function. Constant-Liar strategy have poor scalability because of this repeated re-fitting. The ``"qUCB"`` strategy is much more efficient by sampling a new $kappa$ value for each new requested point without re-fitting the model, but it is only compatible with ``acq_func == "UCB"``. Defaults to ``"cl_max"``.
        n_jobs (int, optional): Number of parallel processes used to fit the surrogate model of the Bayesian optimization. A value of ``-1`` will use all available cores. Not used in ``surrogate_model`` if passed as own sklearn regressor. Defaults to ``1``.
        n_initial_points (int, optional): Number of collected objectives required before fitting the surrogate-model. Defaults to ``10``.
        initial_point_generator (str, optional): Sets an initial points generator. Can be either ``["random", "sobol", "halton", "hammersly", "lhs", "grid"]``. Defaults to ``"random"``.
        initial_points (List[Dict], optional): A list of initial points to evaluate where each point is a dictionnary where keys are names of hyperparameters and values their corresponding choice. Defaults to ``None`` for them to be generated randomly from the search space.
        sync_communcation (bool, optional): Performs the search in a batch-synchronous manner. Defaults to ``False`` for asynchronous updates.
        filter_failures (str, optional): Replace objective of failed configurations by ``"min"`` or ``"mean"``. If ``"ignore"`` is passed then failed configurations will be filtered-out and not passed to the surrogate model. For multiple objectives, failure of any single objective will lead to treating that configuration as failed and each of these multiple objective will be replaced by their individual ``"min"`` or ``"mean"`` of past configurations. Defaults to ``"mean"`` to replace by failed configurations by the running mean of objectives.
        max_failures (int, optional): Maximum number of failed configurations allowed before observing a valid objective value when ``filter_failures`` is not equal to ``"ignore"``. Defaults to ``100``.
        moo_scalarization_strategy (str, optional): Scalarization strategy used in multiobjective optimization. Can be a value in ``["Linear", "Chebyshev", "AugChebyshev", "PBI", "Quadratic", "rLinear", "rChebyshev", "rAugChebyshev", "rPBI", "rQuadratic"]``. Defaults to ``"Chebyshev"``.
        moo_scalarization_weight (list, optional): Scalarization weights to be used in multiobjective optimization with length equal to the number of objective functions. Defaults to ``None``.
        scheduler (dict, callable, optional): a method to manage the the value of ``kappa, xi`` with iterations. Defaults to ``None`` which does not use any scheduler.
        objective_scaler (str, optional): a way to map the objective space to some other support for example to normalize it. Defaults to ``"auto"`` which automatically set it to "identity" for any surrogate model except "RF" which will use "minmaxlog".
        stopper (Stopper, optional): a stopper to leverage multi-fidelity when evaluating the function. Defaults to ``None`` which does not use any stopper.
        comm (Comm, optional): communicator used with MPI. Defaults to ``None`` for  ``COMM_WORLD``.
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
        update_prior: bool = False,  # TODO: check what this is doing?
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
        scheduler=None,
        objective_scaler="auto",
        stopper: Stopper = None,
        comm: MPI.Comm = None,
        **kwargs,
    ):
        # get the __init__ parameters
        _init_params = locals()

        if not MPI.Is_initialized():
            MPI.Init_thread()

        self.comm = comm if comm else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.check_evaluator(evaluator)

        if type(random_state) is int:
            random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            random_state = random_state
        else:
            random_state = np.random.RandomState()

        if acq_optimizer == "auto":
            if acq_func[0] == "q":
                acq_optimizer = "sampling"
            elif acq_func[0] == "b":
                acq_optimizer == "boltzmann_sampling"
            else:
                acq_optimizer = "sampling"

        if acq_func[0] == "q":
            kappa = scipy.stats.expon.rvs(
                size=self.size, scale=kappa, random_state=random_state
            )[self._evaluator.rank]
            xi = scipy.stats.expon.rvs(
                size=self.size, scale=xi, random_state=random_state
            )[self._evaluator.rank]
            acq_func = acq_func[1:]
        elif acq_func[0] == "b":
            acq_func[0] = acq_func[1:]

        # set random state for given rank
        random_state = np.random.RandomState(
            random_state.randint(low=0, high=2**32, size=self.size)[self.rank]
        )

        if self.rank == 0:
            super().__init__(
                problem=problem,
                evaluator=evaluator,
                random_state=random_state,
                log_dir=log_dir,
                verbose=verbose,
                surrogate_model=surrogate_model,
                acq_func=acq_func,
                acq_optimizer=acq_optimizer,
                kappa=kappa,
                xi=xi,
                n_points=n_points,
                filter_duplicated=filter_duplicated,
                update_prior=update_prior,
                multi_point_strategy=multi_point_strategy,
                n_jobs=n_jobs,
                n_initial_points=n_initial_points,
                initial_point_generator=initial_point_generator,
                initial_points=initial_points,
                sync_communication=sync_communication,
                filter_failures=filter_failures,
                max_failures=max_failures,
                moo_scalarization_strategy=moo_scalarization_strategy,
                moo_scalarization_weight=moo_scalarization_weight,
                scheduler=scheduler,
                objective_scaler=objective_scaler,
                stopper=stopper,
                **kwargs,
            )
        self.comm.Barrier()
        if self.rank > 0:
            super().__init__(
                problem=problem,
                evaluator=evaluator,
                random_state=random_state,
                log_dir=log_dir,
                verbose=verbose,
                surrogate_model=surrogate_model,
                acq_func=acq_func,
                acq_optimizer=acq_optimizer,
                kappa=kappa,
                xi=xi,
                n_points=n_points,
                filter_duplicated=filter_duplicated,
                update_prior=update_prior,
                multi_point_strategy=multi_point_strategy,
                n_jobs=n_jobs,
                n_initial_points=n_initial_points,
                initial_point_generator=initial_point_generator,
                initial_points=initial_points,
                sync_communication=sync_communication,
                filter_failures=filter_failures,
                max_failures=max_failures,
                moo_scalarization_strategy=moo_scalarization_strategy,
                moo_scalarization_weight=moo_scalarization_weight,
                scheduler=scheduler,
                objective_scaler=objective_scaler,
                stopper=stopper,
                **kwargs,
            )

        self.comm.Barrier()

        # Replace CBO _init_params by DBO _init_params
        self._init_params = _init_params

        logging.info(
            f"DBO rank {self.rank} has {self._evaluator.num_workers} local worker(s)"
        )

    def check_evaluator(self, evaluator):

        if not (isinstance(evaluator, Evaluator)):

            if callable(evaluator):
                self._evaluator = self.bootstrap_evaluator(
                    run_function=evaluator,
                    evaluator_type="serial",
                    storage_type="redis",
                    comm=self.comm,
                    root=0,
                )

            else:
                raise TypeError(
                    f"The evaluator shoud be an instance of deephyper.evaluator.Evaluator by is {type(evaluator)}!"
                )
        else:
            self._evaluator = evaluator

    @staticmethod
    def bootstrap_evaluator(
        run_function,
        evaluator_type: str = "serial",
        evaluator_kwargs: dict = None,
        storage_type: str = "redis",
        storage_kwargs: dict = None,
        comm=None,
        root=0,
    ):
        comm = comm if comm is None else MPI.COMM_WORLD
        rank = comm.Get_rank()

        evaluator_kwargs = evaluator_kwargs if evaluator_kwargs else {}
        storage_kwargs = storage_kwargs if storage_kwargs else {}

        storage = Storage.create(storage_type, storage_kwargs).connect()
        search_id = None
        if rank == root:
            search_id = storage.create_new_search()
        search_id = comm.bcast(search_id)

        callbacks = []
        if "callbacks" in evaluator_kwargs:
            callbacks = evaluator_kwargs["callbacks"]

        if rank == root and not (any(isinstance(cb, TqdmCallback) for cb in callbacks)):
            callbacks.append(TqdmCallback())

        evaluator_kwargs["callbacks"] = callbacks

        # all processes are using the same search_id
        evaluator_kwargs["storage"] = storage
        evaluator_kwargs["search_id"] = search_id

        evaluator = Evaluator.create(
            run_function,
            method=evaluator_type,
            method_kwargs=evaluator_kwargs,
        )

        # all ranks synchronise with timestamp on root rank
        evaluator.timestamp = comm.bcast(evaluator.timestamp)

        # replace dump_evals of evaluator by empty function to avoid concurrent writtings in file
        if rank != root:

            def dumps_evals(*args, **kwargs):
                pass

            evaluator.dump_evals = dumps_evals

        return evaluator
