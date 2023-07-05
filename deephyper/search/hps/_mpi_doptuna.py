import functools
import logging
import os
import time
from typing import Union

import ConfigSpace.hyperparameters as csh
import mpi4py
import numpy as np
import optuna
import pandas as pd
from optuna.study import MaxTrialsCallback
from optuna.trial import FrozenTrial, TrialState

# !To avoid initializing MPI when module is imported (MPI is optional)
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI  # noqa: E402

from deephyper.core.exceptions import SearchTerminationError  # noqa: E402
from deephyper.core.utils._timeout import terminate_on_timeout  # noqa: E402
from deephyper.evaluator import RunningJob  # noqa: E402
from deephyper.search import Search  # noqa: E402


def optuna_suggest_from_hp(trial, cs_hp):
    name = cs_hp.name
    if isinstance(cs_hp, csh.UniformIntegerHyperparameter):
        value = trial.suggest_int(name, cs_hp.lower, cs_hp.upper, log=cs_hp.log)
    elif isinstance(cs_hp, csh.UniformFloatHyperparameter):
        value = trial.suggest_float(name, cs_hp.lower, cs_hp.upper, log=cs_hp.log)
    elif isinstance(cs_hp, csh.CategoricalHyperparameter):
        value = trial.suggest_categorical(name, cs_hp.choices)
    elif isinstance(cs_hp, csh.OrdinalHyperparameter):
        value = trial.suggest_categorical(name, cs_hp.sequence)
    else:
        raise TypeError(f"Cannot convert hyperparameter of type {type(cs_hp)}")

    return name, value


def optuna_suggest_from_configspace(trial, cs_space):
    config = {}
    for cs_hp in cs_space.get_hyperparameters():
        name, value = optuna_suggest_from_hp(trial, cs_hp)
        config[name] = value
    return config


class CheckpointSaverCallback:
    def __init__(self, log_dir=".", states=(TrialState.COMPLETE,)) -> None:
        self._log_dir = log_dir
        self._states = states

    def __call__(self, study: optuna.study.Study, trial: FrozenTrial) -> None:
        all_trials = study.get_trials(deepcopy=False, states=self._states)
        # n_complete = len(all_trials)

        pd.DataFrame([t.user_attrs["results"] for t in all_trials]).to_csv(
            os.path.join(self._log_dir, "results.csv")
        )


# Supported samplers
supported_samplers = ["TPE", "CMAES", "NSGAII", "DUMMY", "BOTORCH"]
supported_pruners = ["NOP", "SHA", "HB", "MED"]


class MPIDistributedOptuna(Search):
    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        sampler: Union[str, optuna.samplers.BaseSampler] = "TPE",
        pruner: Union[str, optuna.pruners.BasePruner] = "NOP",
        n_objectives: int = 1,
        study_name: str = None,
        storage: Union[str, optuna.storages.BaseStorage] = None,
        comm: MPI.Comm = None,
        **kwargs,
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        optuna.logging.set_verbosity(
            optuna.logging.DEBUG if self._verbose else optuna.logging.ERROR
        )

        self._evaluator = evaluator

        # get the __init__ parameters
        _init_params = locals()

        if not MPI.Is_initialized():
            MPI.Init_thread()

        self.comm = comm if comm else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1

        # Set different random state for given rank
        self._random_state = np.random.RandomState(
            self._random_state.randint(low=0, high=2**31, size=self.size)[self.rank]
        )

        # Setup the sampler
        if isinstance(sampler, optuna.samplers.BaseSampler):
            pass
        elif isinstance(sampler, str):
            sampler_seed = self._random_state.randint(2**31)
            if sampler == "TPE":
                sampler = optuna.samplers.TPESampler(seed=sampler_seed)
            elif sampler == "CMAES":
                sampler = optuna.samplers.CmaEsSampler(seed=sampler_seed)
            elif sampler == "NSGAII":
                sampler = optuna.samplers.NSGAIISampler(seed=sampler_seed)
            elif sampler == "DUMMY":
                sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
            elif sampler == "BOTORCH":
                from optuna.integration import BoTorchSampler

                sampler = BoTorchSampler(seed=sampler_seed)
            else:
                raise ValueError(
                    f"Requested unknown sampler {sampler} should be one of {supported_samplers}"
                )
        else:
            raise TypeError(
                f"Sampler is of type {type(sampler)} but must be a str or optuna.samplers.BaseSampler!"
            )

        # Setup the pruner
        if isinstance(pruner, optuna.pruners.BasePruner):
            pass
        elif isinstance(pruner, str):
            if pruner == "NOP":
                pruner = optuna.pruners.NopPruner()
            elif pruner == "SHA":
                pruner = optuna.pruners.SuccessiveHalvingPruner()
            elif pruner == "HB":
                pruner = optuna.pruners.HyperbandPruner()
            elif pruner == "MED":
                pruner = optuna.pruners.MedianPruner()
            else:
                raise ValueError(
                    f"Requested unknown pruner {pruner} should be one of {supported_pruners}"
                )
        else:
            raise TypeError(
                f"Pruner is of type {type(pruner)} but must be a str or optuna.pruners.BasePruner!"
            )
        self.pruner = pruner

        self._n_objectives = n_objectives

        study_params = dict(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )

        if self.rank == 0:
            if self._n_objectives > 1:
                study_params["directions"] = [
                    "maximize" for _ in range(self._n_objectives)
                ]
            else:
                study_params["direction"] = "maximize"

        self.timestamp = None

        # Root rank creates study and initilize the timestamp
        if self.rank == 0:
            self.timestamp = time.time()
            self.study = optuna.create_study(**study_params)

        if self.size > 1:
            self.timestamp = comm.bcast(self.timestamp)

        # Other ranks load the previously created study
        if self.rank > 0:
            self.study = optuna.load_study(**study_params)

        self._init_params = _init_params

        logging.info(f"MPIDistributedOptuna has {self.size} rank(s)")
        logging.info(f"MPIDistributedOptuna rank {self.rank} has 1 local worker(s)")

    def search(self, max_evals=None, timeout=-1):
        if timeout == -1:
            timeout = None
        return self._search(max_evals, timeout)

    def _search(self, max_evals, timeout):
        def objective_wrapper(trial):
            config = optuna_suggest_from_configspace(trial, self._problem.space)
            if self.pruner is None:
                output = self._evaluator(RunningJob(id=trial.number, parameters=config))
            else:
                output = self._evaluator(
                    RunningJob(id=trial.number, parameters=config), optuna_trial=trial
                )

            data = {f"p:{k}": v for k, v in config.items()}
            if isinstance(output["objective"], list) or isinstance(
                output["objective"], tuple
            ):
                for i, obj in enumerate(output["objective"]):
                    data[f"objective_{i}"] = obj
            else:
                data["objective"] = output["objective"]
            data["job_id"] = trial.number
            data.update({f"m:{k}": v for k, v in output["metadata"].items()})
            trial.set_user_attr("results", data)

            if data.get("m:stopped", False):
                raise optuna.TrialPruned()

            return output["objective"]

        def optimize_wrapper(duration):
            callbacks = []

            if max_evals > 0:
                callbacks.append(MaxTrialsCallback(max_evals))

            if self.rank == 0:
                callbacks += [
                    CheckpointSaverCallback(
                        self._log_dir,
                        states=(
                            optuna.trial.TrialState.COMPLETE,
                            optuna.trial.TrialState.PRUNED,
                            optuna.trial.TrialState.FAIL,
                        ),
                    )
                ]

            self.study.optimize(
                objective_wrapper,
                n_trials=max_evals if max_evals > 0 else None,
                timeout=duration,
                callbacks=callbacks,
            )

        print(f"{self.rank=}, {max_evals=}, {timeout=}", flush=True)
        if timeout is None:
            logging.info(f"Running without timeout and max_evals={max_evals}")
            optimize_wrapper(None)
        else:
            logging.info(f"Running with timeout={timeout} and max_evals={max_evals}")
            optimize = functools.partial(
                terminate_on_timeout, timeout, optimize_wrapper
            )
            try:
                optimize(timeout)
            except SearchTerminationError:
                pass

        if self.rank == 0:
            all_trials = self.study.get_trials(
                deepcopy=True,
                states=[
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.PRUNED,
                    optuna.trial.TrialState.FAIL,
                ],
            )

            results = pd.DataFrame([t.user_attrs["results"] for t in all_trials])
            results.to_csv(os.path.join(self._log_dir, "results.csv"))
            return results
        else:
            return None
