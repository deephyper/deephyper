from collections import deque
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ConfigSpace.util import deactivate_inactive_hyperparameters

from deephyper.hpo._problem import HpProblem
from deephyper.hpo._search import Search
from deephyper.hpo._solution import SolutionSelection
from deephyper.hpo.utils import get_inactive_value_of_hyperparameter
from deephyper.stopper._stopper import Stopper

__all__ = ["RegularizedEvolution"]


class RegularizedEvolution(Search):
    """Regularized evolution algorithm.

    This implementation is an example for the Search API to implement new search algorithms.

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1

        * - Single-Objective
          - Multi-Objectives
          - Failures
        * - ✅
          - ❌
          - ✅

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
            a stopper to leverage multi-fidelity when evaluating the function. Defaults to
            ``None`` which does not use any stopper.

        checkpoint_history_to_csv (bool, optional):
            wether the results from progressively collected evaluations should be checkpointed
            regularly to disc as a csv. Defaults to ``True``.

        solution_selection (Literal["argmax_obs", "argmax_est"] | SolutionSelection, optional):
            the solution selection strategy. It can be a string where ``"argmax_obs"`` would
            select the argmax of observed objective values, and ``"argmax_est"`` would select the
            argmax of estimated objective values (through a predictive model).

        population_size (int, optional):
            The size of the population. Defaults to ``100``.

        sample_size (int, optional):
            The number of samples to draw from the population. Defaults to ``10``.

        max_trials_rejection_sampling (int):
            The maximum number of trials for rejection sampling with resolving contraints. Defaults
            to ``-1``.
    """

    def __init__(
        self,
        problem: HpProblem,
        random_state: int | np.random.RandomState | None = None,
        log_dir: str = ".",
        verbose: int = 0,
        stopper: Stopper | None = None,
        checkpoint_history_to_csv: bool = True,
        solution_selection: Literal["argmax_obs", "argmax_est"] | SolutionSelection | None = None,
        population_size: int = 100,
        sample_size: int = 10,
        max_trials_rejection_sampling: int = -1,
        init_population: list[tuple[dict, Any]] | None = None,
    ):
        super().__init__(
            problem,
            random_state,
            log_dir,
            verbose,
            stopper,
            checkpoint_history_to_csv,
            solution_selection,
        )
        self._problem.space.seed(self._random_state.randint(0, np.iinfo(np.int32).max))
        assert population_size > sample_size, "population_size must be greater than sample_size"
        self.population_size = population_size
        self.sample_size = sample_size
        if init_population is None:
            init_population = []
        self._population: deque[tuple[dict, Any]] = deque(
            init_population, maxlen=self.population_size
        )
        self._max_trials_rejection_sampling = max_trials_rejection_sampling

    def _ask(self, n: int = 1) -> list[dict[str, Any]]:
        """Ask the search for new configurations to evaluate.

        Args:
            n (int, optional): The number of configurations to ask. Defaults to 1.

        Returns:
            List[Dict]: a list of hyperparameter configurations to evaluate.
        """
        # Random sampling
        if len(self._population) < self.population_size:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)

                new_samples = self._problem.sample(size=n)

            if not (isinstance(new_samples, list)):
                new_samples = [new_samples]

            for i, sample in enumerate(new_samples):
                sample = dict(sample)

                self._set_inactive(sample)

                new_samples[i] = sample

        # Regularized evolution
        else:
            new_samples = []
            for i in range(n):
                # Get a sample of parents from the population
                samples_idxs = self._random_state.choice(
                    self.population_size, size=self.sample_size, replace=False
                )

                samples = [self._population[i] for i in samples_idxs]

                # Select the parent
                parent_sample = max(samples, key=lambda x: x[1])[0]

                # Produce the child
                n_trials = 0
                child_sample = self._mutate(parent_sample)

                def is_not_max_trials():
                    return (
                        self._max_trials_rejection_sampling < 0
                        or n_trials < self._max_trials_rejection_sampling
                    )

                while is_not_max_trials() and not self._problem.is_feasible(child_sample):
                    child_sample = self._mutate(parent_sample)
                    n_trials += 1

                # If we can't produce a feasible child for _max_trials_rejection_sampling
                # Then we resample a new fresh child
                if is_not_max_trials():
                    new_samples.append(child_sample)
                else:
                    try:
                        child_sample = self._problem.sample(size=1, strict=True)[0]
                    except RuntimeError:
                        raise RuntimeError(
                            "Could not resolve constraints through rejection sampling!"
                        )

        return new_samples

    def _tell(
        self, results: list[tuple[dict[str, Optional[str | int | float]], str | int | float]]
    ):
        """Tell the search the results of the evaluations.

        Args:
            results (list[tuple[dict[str, Optional[str | int | float]], str | int | float]]):
                a dictionary containing the results of the evaluations.
        """
        for config, obj in results:
            # Do not add failures to population
            if isinstance(obj, str):
                continue
            self._population.append((config, obj))

    def _mutate(self, parent_sample: dict) -> dict:
        space = self._problem.space
        child_sample = parent_sample.copy()
        active_hyperparameter_names = list(
            space.get_active_hyperparameters(
                deactivate_inactive_hyperparameters(child_sample, space)
            )
        )
        hp_name = self._random_state.choice(active_hyperparameter_names)
        hp = space[hp_name]
        hp_value = hp.rvs(size=None, random_state=space.random)

        child_sample[hp_name] = hp_value
        child_sample = dict(deactivate_inactive_hyperparameters(child_sample, space))

        self._set_inactive(child_sample)

        return child_sample

    def _set_inactive(self, sample: dict):
        space = self._problem.space
        for hp_name in self._problem.hyperparameter_names:
            # If the parameter is inactive due to some conditions then we attribute the
            # lower bound value to break symmetries and enforce the same repsresentation.
            if hp_name not in sample:
                sample[hp_name] = get_inactive_value_of_hyperparameter(space[hp_name])

            # Make sure to have JSON serializable values
            if type(sample[hp_name]).__module__ == np.__name__:
                sample[hp_name] = sample[hp_name].tolist()
