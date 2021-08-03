import logging
import signal

from deephyper.core.exceptions import SearchTerminationError

class Search:
    def __init__(
        self,
        problem,
        evaluator,
        random_state=None,
        log_dir=".",
        verbose=0,
        **kwargs
    ):
        self._problem = problem
        self._evaluator = evaluator
        self._random_state = random_state
        self._log_dir = log_dir
        self._verbose = verbose

    def terminate(self):
        logging.info("Search is being stopped!")
        raise SearchTerminationError

    def _set_timeout(self, timeout=None):

        def handler(signum, frame):
            self.terminate()

        signal.signal(signal.SIGALRM, handler)

        if type(timeout) is int:
            signal.alarm(timeout)

    def search(self, max_evals=-1, timeout=None):

        self._set_timeout(timeout)

        try:
            self._search(max_evals, timeout)
        except SearchTerminationError:
            self._evaluator.dump_evals()

    def _search(self, max_evals, timeout):
        raise NotImplementedError

