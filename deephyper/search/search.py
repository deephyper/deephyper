import argparse
import logging
import signal

from deephyper.core.exceptions import SearchTerminationError
from deephyper.evaluator.evaluate import EVALUATORS
import numpy as np
import pandas as pd


class Search:
    def __init__(
        self, problem, evaluator, seed=None, log_dir=".", verbose=0, **kwargs
    ):
        self._problem = problem
        self._evaluator = evaluator
        self._random_state = np.random.RandomState(seed)
        self._log_dir = log_dir
        self._verbose = verbose

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

    def search(self, max_evals: int=-1, timeout: int=None):
        """Execute the search algorithm.

        Args:
            max_evals (int, optional): The maximum number of evaluations of the run function to perform before stopping the search. Defaults to -1, will run indefinitely.
            timeout (int, optional): The time budget of the search before stopping.Defaults to None, will not impose a time budget.

        Returns:
            DataFrame: a pandas DataFrame containing the evaluations performed.
        """

        self._set_timeout(timeout)

        try:
            self._search(max_evals, timeout)
        except SearchTerminationError:
            if "saved_keys" in dir(self):
                self._evaluator.dump_evals(saved_keys=self.saved_keys)
            else:
                self._evaluator.dump_evals()

        df_results = pd.read_csv("results.csv")
        return df_results

    def _search(self, max_evals, timeout):
        raise NotImplementedError