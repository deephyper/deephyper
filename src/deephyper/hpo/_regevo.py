from collections import deque
from typing import Dict, List

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.util import deactivate_inactive_hyperparameters

from deephyper.evaluator import HPOJob
from deephyper.hpo._search import Search


def get_inactive_value_of_hyperparameter(hp):
    """Return a value when the hyperparameter is considered inactive."""
    if isinstance(hp, NumericalHyperparameter):
        return hp.lower
    elif isinstance(hp, CategoricalHyperparameter):
        return hp.choices[0]
    elif isinstance(hp, OrdinalHyperparameter):
        return hp.sequence[0]
    elif isinstance(hp, Constant):
        return hp.value
    else:
        raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")


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
        problem: object describing the search/optimization problem.
        evaluator: object describing the evaluation process.
        random_state (np.random.RandomState, optional): Initial random state of the search.
            Defaults to ``None``.
        log_dir (str, optional): Path to the directoy where results of the search are stored.
            Defaults to ``"."``.
        verbose (int, optional): Use verbose mode. Defaults to ``0``.
        stopper (Stopper, optional): a stopper to leverage multi-fidelity when evaluating the
            function. Defaults to ``None`` which does not use any stopper.
        population_size (int, optional): The size of the population. Defaults to ``100``.
        sample_size (int, optional): The number of samples to draw from the population. Defaults
            to ``10``.
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state=None,
        log_dir=".",
        verbose=0,
        stopper=None,
        population_size=100,
        sample_size=10,
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose, stopper)
        self._problem.space.seed(self._random_state.randint(0, 2**31))
        assert population_size > sample_size, "population_size must be greater than sample_size"
        self.population_size = population_size
        self.sample_size = sample_size
        self._population = deque(maxlen=self.population_size)

    def _ask(self, n: int = 1) -> List[Dict]:
        """Ask the search for new configurations to evaluate.

        Args:
            n (int, optional): The number of configurations to ask. Defaults to 1.

        Returns:
            List[Dict]: a list of hyperparameter configurations to evaluate.
        """
        space = self._problem.space

        # Random sampling
        if len(self._population) < self.population_size:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)

                new_samples = space.sample_configuration(size=n)

            if not (isinstance(new_samples, list)):
                new_samples = [new_samples]

            for i, sample in enumerate(new_samples):
                sample = dict(sample)
                for hp_name in self._problem.hyperparameter_names:
                    # If the parameter is inactive due to some conditions then we attribute the
                    # lower bound value to break symmetries and enforce the same representation.
                    if hp_name not in sample:
                        sample[hp_name] = get_inactive_value_of_hyperparameter(space[hp_name])

                    # Make sure to have JSON serializable values
                    if type(sample[hp_name]).__module__ == np.__name__:
                        sample[hp_name] = sample[hp_name].tolist()

                new_samples[i] = sample

        # Regularized evolution
        else:
            new_samples = []
            for i in range(n):
                samples_idxs = self._random_state.choice(
                    self.population_size, size=self.sample_size, replace=False
                )

                samples = [self._population[i] for i in samples_idxs]

                parent_sample = max(samples, key=lambda x: x[1])[0]

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

                for hp_name in self._problem.hyperparameter_names:
                    # If the parameter is inactive due to some conditions then we attribute the
                    # lower bound value to break symmetries and enforce the same representation.
                    if hp_name not in child_sample:
                        child_sample[hp_name] = get_inactive_value_of_hyperparameter(
                            self._problem.space[hp_name]
                        )

                    # Make sure to have JSON serializable values
                    if type(child_sample[hp_name]).__module__ == np.__name__:
                        child_sample[hp_name] = child_sample[hp_name].tolist()

                new_samples.append(child_sample)

        return new_samples

    def _tell(self, results: List[HPOJob]):
        """Tell the search the results of the evaluations.

        Args:
            results (List[HPOJob]): a dictionary containing the results of the evaluations.
        """
        for config, obj in results:
            # Do not add failures to population
            if isinstance(obj, str):
                continue
            self._population.append((config, obj))
