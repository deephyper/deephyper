from typing import Dict, List, Literal

import numpy as np

from deephyper.evaluator import HPOJob
from deephyper.hpo._search import Search
from deephyper.hpo._solution import SolutionSelection
from deephyper.hpo.utils import get_inactive_value_of_hyperparameter

__all__ = ["RandomSearch"]


class RandomSearch(Search):
    """Random search algorithm used as an example for the API to implement new search algorithms.

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
        problem:
            object describing the search/optimization problem.

        evaluator:
            object describing the evaluation process.

        random_state (np.random.RandomState, optional):
            Initial random state of the search. Defaults to ``None``.

        log_dir (str, optional):
            Path to the directoy where results of the search are stored. Defaults to ``"."``.

        verbose (int, optional):
            Use verbose mode. Defaults to ``0``.

        stopper (Stopper, optional):
            a stopper to leverage multi-fidelity when evaluating the
            function. Defaults to ``None`` which does not use any stopper.

        checkpoint_history_to_csv (bool, optional):
            wether the results from progressively collected evaluations should be checkpointed
            regularly to disc as a csv. Defaults to ``True``.

        solution_selection (Literal["argmax_obs", "argmax_est"] | SolutionSelection, optional):
            the solution selection strategy. It can be a string where ``"argmax_obs"`` would
            select the argmax of observed objective values, and ``"argmax_est"`` would select the
            argmax of estimated objective values (through a predictive model).
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state=None,
        log_dir=".",
        verbose=0,
        stopper=None,
        checkpoint_history_to_csv: bool = True,
        solution_selection: Literal["argmax_obs", "argmax_est"] | SolutionSelection = "argmax_obs",
    ):
        super().__init__(
            problem,
            evaluator,
            random_state,
            log_dir,
            verbose,
            stopper,
            checkpoint_history_to_csv,
            solution_selection,
        )
        self._problem.space.seed(self._random_state.randint(0, 2**31))

    def _ask(self, n: int = 1) -> List[Dict]:
        """Ask the search for new configurations to evaluate.

        Args:
            n (int, optional): The number of configurations to ask. Defaults to 1.

        Returns:
            List[Dict]: a list of hyperparameter configurations to evaluate.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            new_samples = self._problem.space.sample_configuration(size=n)

        if not (isinstance(new_samples, list)):
            new_samples = [new_samples]

        for i, sample in enumerate(new_samples):
            sample = dict(sample)
            for hp_name in self._problem.hyperparameter_names:
                # If the parameter is inactive due to some conditions then we attribute the
                # lower bound value to break symmetries and enforce the same representation.
                if hp_name not in sample:
                    sample[hp_name] = get_inactive_value_of_hyperparameter(
                        self._problem.space[hp_name]
                    )

                # Make sure to have JSON serializable values
                if type(sample[hp_name]).__module__ == np.__name__:
                    sample[hp_name] = sample[hp_name].tolist()

            new_samples[i] = sample
        return new_samples

    def _tell(self, results: List[HPOJob]):
        """Tell the search the results of the evaluations.

        Args:
            results (List[HPOJob]): a dictionary containing the results of the evaluations.
        """
        for config, obj in results:
            pass
