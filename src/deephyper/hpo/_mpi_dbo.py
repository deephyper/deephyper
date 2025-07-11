import asyncio
import logging
from typing import Optional

import numpy as np
import scipy.stats

from deephyper.evaluator import Evaluator, HPOJob
from deephyper.evaluator.callback import TqdmCallback
from deephyper.evaluator.mpi import MPI
from deephyper.evaluator.storage import Storage
from deephyper.hpo._cbo import CBO, AcqFuncKwargs, AcqOptimizerKwargs, SurrogateModelKwargs
from deephyper.stopper import Stopper

__all__ = ["MPIDistributedBO"]


class MPIDistributedBO(CBO):
    """Distributed Bayesian Optimization Search using MPI to launch parallel search instances.

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1

        * - Single-Objective
          - Multi-Objectives
          - Failures
        * - ✅
          - ✅
          - ✅

    Args:
        problem (HpProblem): Hyperparameter problem describing the search space to explore.

        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.

        random_state (int, optional): Random seed. Defaults to ``None``.

        log_dir (str, optional): Log directory where search's results are saved. Defaults to
            ``"."``.

        verbose (int, optional): Indicate the verbosity level of the search. Defaults to ``0``.

        stopper (Stopper, optional): a stopper to leverage multi-fidelity when evaluating the
            function. Defaults to ``None`` which does not use any stopper.

        checkpoint_history_to_csv (bool, optional):
            wether the results from progressively collected evaluations should be checkpointed
            regularly to disc as a csv. Defaults to ``True``.

        surrogate_model (Union[str,sklearn.base.RegressorMixin], optional): Surrogate model used by
            the Bayesian optimization. Can be a value in ``["RF", "GP", "ET", "GBRT",
            "DUMMY"]`` or a sklearn regressor. ``"ET"`` is for Extremely Randomized Trees which is
            the best compromise between speed and quality when performing a lot of parallel
            evaluations, i.e., reaching more than hundreds of evaluations. ``"GP"`` is for Gaussian-
            Process which is the best choice when maximizing the quality of iteration but quickly
            slow down when reaching hundreds of evaluations, also it does not support conditional
            search space. ``"RF"`` is for Random-Forest, slower than extremely randomized trees but
            with better mean estimate and worse epistemic uncertainty quantification capabilities.
            ``"GBRT"`` is for Gradient-Boosting Regression Tree, it has better mean estimate than
            other tree-based method worse uncertainty quantification capabilities and slower than
            ``"RF"``. Defaults to ``"ET"``.

        surrogate_model_kwargs (dict, optional): Additional parameters to pass to the surrogate
            model. Defaults to ``None``.

        acq_func (str, optional): Acquisition function used by the Bayesian optimization. Can be a
            value in ``["UCB", "EI", "PI", "gp_hedge"]``. Defaults to ``"UCB"``.

        acq_func_kwargs (dict, optional):
            A dictionnary of parameters for the acquisition function:

            - ``"kappa"`` (float)
                Manage the exploration/exploitation tradeoff for the ``"UCB"`` acquisition function.
                Defaults to ``1.96`` which corresponds to 95% of the confidence interval.

            - ``"xi"`` (float)
                Manage the exploration/exploitation tradeoff of ``"EI"`` and ``"PI"``
                acquisition function. Defaults to ``0.001``.

        acq_optimizer (str, optional):
            Method used to minimze the acquisition function. Can be a value in
            ``["sampling", "lbfgs", "ga", "mixedga"]``. Defaults to ``"auto"``.

        acq_optimizer_kwargs (dict, optional):
            A dictionnary of parameters for the acquisition function optimizer:

            - ``"acq_optimizer_freq"`` (int)
                Frequency of optimization calls for the acquisition function. Defaults
                to ``10``, using optimizer every ``10`` surrogate model updates.

            - ``"n_points"`` (int)
                The number of configurations sampled from the search space to infer each
                batch of new evaluated configurations.

            - ``"filter_duplicated"`` (bool)
                Force the optimizer to sample unique points until the search space is "exhausted"
                in the sens that no new unique points can be found given the sampling size
                ``n_points``. Defaults to ``True``.

            - ``"n_jobs"`` (int)
                Number of parallel processes used when possible. Defaults to ``1``.

            - ``"filter_failures"`` (str)
                Replace objective of failed configurations by ``"min"`` or ``"mean"``. If
                ``"ignore"`` is passed then failed configurations will be filtered-out and not
                passed to the surrogate model. For multiple objectives,
                failure of any single objective will lead to treating that configuration as failed
                and each of these multiple objective will be replaced by their individual ``"min"``
                or ``"mean"`` of past configurations. Defaults to ``"min"`` to replace failed
                configurations objectives by the running min of all objectives.

            - ``"max_failures"`` (int)
                Maximum number of failed configurations allowed before observing a valid objective
                value when ``filter_failures`` is not equal to ``"ignore"``. Defaults to ``100``.

        multi_point_strategy (str, optional): Definition of the constant value use for the Liar
            strategy. Can be a value in ``["cl_min", "cl_mean", "cl_max", "qUCB", "qUCBd"]``. All
            ``"cl_..."`` strategies follow the constant-liar scheme, where if $N$ new points are
            requested, the surrogate model is re-fitted $N-1$ times with lies (respectively, the
            minimum, mean and maximum objective found so far; for multiple objectives, these are
            the minimum, mean and maximum of the individual objectives) to infer the acquisition
            function. Constant-Liar strategy have poor scalability because of this repeated re-
            fitting. The ``"qUCB"`` strategy is much more efficient by sampling a new $kappa$ value
            for each new requested point without re-fitting the model.

        n_initial_points (int, optional): Number of collected objectives required before fitting
            the surrogate-model. Defaults to ``None`` that will use ``2 * N + 1`` where ``N`` is
            the number of parameters in the ``problem``.

        initial_point_generator (str, optional): Sets an initial points generator. Can be either
            ``["random", "sobol", "halton", "hammersly", "lhs", "grid"]``. Defaults to ``"random"``.

        initial_points (List[Dict], optional): A list of initial points to evaluate where each
            point is a dictionnary where keys are names of hyperparameters and values their
            corresponding choice. Defaults to ``None`` for them to be generated randomly from
            the search space.

        moo_lower_bounds (list, optional): List of lower bounds on the interesting range of
            objective values. Must be the same length as the number of obejctives. Defaults to
            ``None``, i.e., no bounds. Can bound only a single objective by providing ``None``
            for all other values. For example, ``moo_lower_bounds=[None, 0.5, None]`` will explore
            all tradeoffs for the objectives at index 0 and 2, but only consider scores for
            objective 1 that exceed 0.5.

        moo_scalarization_strategy (str, optional): Scalarization strategy used in multiobjective
            optimization. Can be a value in ``["Linear", "Chebyshev", "AugChebyshev", "PBI",
            "Quadratic"]``. Defaults to ``"Chebyshev"``. Typically, randomized methods should be
            used to capture entire Pareto front, unless there is a known target solution a priori.
            Additional details on each scalarization can be found in :mod:`deephyper.skopt.moo`.

        moo_scalarization_weight (list, optional): Scalarization weights to be used in
            multiobjective optimization with length equal to the number of objective functions.
            Defaults to ``None`` for randomized weights. Only set if you want to fix the
            scalarization weights for a multiobjective HPS.

        scheduler (dict, callable, optional): a function to manage the value of ``kappa, xi`` with
            iterations. Defaults to ``None`` which does not use any scheduler. The periodic
            exponential decay scheduler can be used with  ``scheduler={"type":
            "periodic-exp-decay", "period": 30, "rate": 0.1}``. The scheduler can also be a
            callable function with signature ``scheduler(i, eta_0, **kwargs)`` where ``i`` is the
            current iteration, ``eta_0`` is the initial value of ``[kappa, xi]`` and ``kwargs`` are
            other fixed parameters of the function. Instead of fixing the decay ``"rate"`` the
            final ``kappa`` or ``xi`` can be used ``{"type": "periodic-exp-decay", "period": 25,
            "kappa_final": 1.96}``.

        objective_scaler (str, optional): a way to map the objective space to some other support
            for example to normalize it. Defaults to ``"auto"`` which automatically set it to
            "identity" for any surrogate model except "RF" which will use "quantile-uniform".

        comm (Comm, optional): communicator used with MPI. Defaults to ``None`` for  ``COMM_WORLD``.
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        stopper: Stopper = None,
        checkpoint_history_to_csv: bool = True,
        surrogate_model="ET",
        surrogate_model_kwargs: Optional[SurrogateModelKwargs] = None,
        acq_func: str = "UCBd",
        acq_func_kwargs: Optional[AcqFuncKwargs] = None,
        acq_optimizer: str = "mixedga",
        acq_optimizer_kwargs: Optional[AcqOptimizerKwargs] = None,
        multi_point_strategy: str = "cl_max",
        n_initial_points: int = None,
        initial_point_generator: str = "random",
        initial_points=None,
        moo_lower_bounds=None,
        moo_scalarization_strategy: str = "Chebyshev",
        moo_scalarization_weight=None,
        objective_scaler="minmax",
        comm: MPI.Comm = None,
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

        acq_func_kwargs = {} if acq_func_kwargs is None else acq_func_kwargs
        self._acq_func_kwargs = AcqFuncKwargs(**acq_func_kwargs).model_dump()

        if acq_func[0] == "q":
            self._acq_func_kwargs["kappa"] = scipy.stats.expon.rvs(
                size=self.size,
                scale=self._acq_func_kwargs["kappa"],
                random_state=random_state,
            )[self.rank]
            self._acq_func_kwargs["xi"] = scipy.stats.expon.rvs(
                size=self.size,
                scales=self._acq_func_kwargs["xi"],
                random_state=random_state,
            )[self.rank]
            acq_func = acq_func[1:]

        # set random state for given rank
        random_state = np.random.RandomState(
            random_state.randint(low=0, high=2**31, size=self.size)[self.rank]
        )

        if self.rank == 0:
            logging.info(f"MPIDistributedBO has {self.size} rank(s)")
            super().__init__(
                problem=problem,
                evaluator=evaluator,
                random_state=random_state,
                log_dir=log_dir,
                verbose=verbose,
                stopper=stopper,
                checkpoint_history_to_csv=checkpoint_history_to_csv,
                surrogate_model=surrogate_model,
                surrogate_model_kwargs=surrogate_model_kwargs,
                acq_func=acq_func,
                acq_func_kwargs=acq_func_kwargs,
                acq_optimizer=acq_optimizer,
                acq_optimizer_kwargs=acq_func_kwargs,
                multi_point_strategy=multi_point_strategy,
                n_initial_points=n_initial_points,
                initial_point_generator=initial_point_generator,
                initial_points=initial_points,
                moo_lower_bounds=moo_lower_bounds,
                moo_scalarization_strategy=moo_scalarization_strategy,
                moo_scalarization_weight=moo_scalarization_weight,
                objective_scaler=objective_scaler,
            )
        self.comm.Barrier()
        if self.rank > 0:
            super().__init__(
                problem=problem,
                evaluator=evaluator,
                random_state=random_state,
                log_dir=log_dir,
                verbose=verbose,
                stopper=stopper,
                checkpoint_history_to_csv=False,
                surrogate_model=surrogate_model,
                surrogate_model_kwargs=surrogate_model_kwargs,
                acq_func=acq_func,
                acq_func_kwargs=acq_func_kwargs,
                acq_optimizer=acq_optimizer,
                acq_optimizer_kwargs=acq_func_kwargs,
                multi_point_strategy=multi_point_strategy,
                n_initial_points=n_initial_points,
                initial_point_generator=initial_point_generator,
                initial_points=initial_points,
                moo_lower_bounds=moo_lower_bounds,
                moo_scalarization_strategy=moo_scalarization_strategy,
                moo_scalarization_weight=moo_scalarization_weight,
                objective_scaler=objective_scaler,
            )

        self.comm.Barrier()
        self.is_master = self.rank == 0

        # Replace CBO _init_params by DBO _init_params
        self._init_params = _init_params

        logging.info(
            f"MPIDistributedBO rank {self.rank} has {self._evaluator.num_workers} local worker(s)"
        )

    def check_evaluator(self, evaluator):
        if not (isinstance(evaluator, Evaluator)):
            if callable(evaluator):
                # Pick the adapted evaluator depending if the passed function is a coroutine
                if asyncio.iscoroutinefunction(evaluator):
                    evaluator_type = "serial"
                else:
                    evaluator_type = "thread"

                self._evaluator = self.bootstrap_evaluator(
                    run_function=evaluator,
                    evaluator_type=evaluator_type,
                    storage_type="redis",
                    comm=self.comm,
                    root=0,
                )

            else:
                raise TypeError(
                    f"The evaluator shoud be an instance of deephyper.evaluator.Evaluator "
                    f"by is {type(evaluator)}!"
                )
        else:
            self._evaluator = evaluator

        self._evaluator._job_class = HPOJob

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

        return evaluator
