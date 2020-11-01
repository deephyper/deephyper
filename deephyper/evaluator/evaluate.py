import csv
from collections import OrderedDict
from contextlib import suppress as dummy_context
from math import isnan
import numpy as np
from numpy import integer, floating, ndarray, bool_
import json
import uuid
import logging
import os
import sys
import time
import types

import skopt

from deephyper.evaluator import runner
from deephyper.core.exceptions import DeephyperRuntimeError

logger = logging.getLogger(__name__)


class Encoder(json.JSONEncoder):
    """
    Enables JSON dump of numpy data
    """

    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return obj.hex
        elif isinstance(obj, integer):
            return int(obj)
        elif isinstance(obj, floating):
            return float(obj)
        elif isinstance(obj, bool_):
            return bool(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, types.FunctionType):
            return f"{obj.__module__}.{obj.__name__}"
        elif isinstance(obj, skopt.space.Dimension):
            return str(obj)
        else:
            return super(Encoder, self).default(obj)


class Evaluator:
    """The goal off the evaluator module is to have a set of objects which can helps us to run our task on different environments and with different system settings/properties.

        Args:
            run_function (Callable): the function to execute, it must be a callable and should not be defined in the `__main__` module.
            cache_key (Callable, str, optional): A way of defining how to use cached results. When an evaluation is done a corresponding uuid is generated, if an new evaluation as a already known uuid then the past result will be re-used. You have different ways to define how to generate an uuid. If `None` then the whole  parameter `dict` (corresponding to the evaluation) will be serialized and used as a uuid (it may raise an exception if it's not serializable, please use the `encoder` parameter to use a custom `json.JSONEncoder`). If callable then the parameter dict will be passed to the callable which must return an uuid. If `'uuid'` then the `uuid.uuid4()` will be used for every new evaluation. Defaults to None.

        Raises:
            DeephyperRuntimeError: raised if the `cache_key` parameter is not None, a callable or equal to 'uuid'.
            DeephyperRuntimeError: raised if the `run_function` parameter is from the`__main__` module.
    """

    FAIL_RETURN_VALUE = np.finfo(np.float32).min
    PYTHON_EXE = os.environ.get("DEEPHYPER_PYTHON_BACKEND", sys.executable)
    KERAS_BACKEND = os.environ.get("KERAS_BACKEND", "tensorflow")
    os.environ["KERAS_BACKEND"] = KERAS_BACKEND
    assert os.path.isfile(PYTHON_EXE)

    def __init__(
        self,
        run_function,
        cache_key=None,
        encoder=Encoder,
        seed=None,
        num_workers=None,
        **kwargs,
    ):
        self.encoder = encoder  # dict --> uuid
        self.pending_evals = {}  # uid --> Future
        self.finished_evals = OrderedDict()  # uid --> scalar
        self.requested_evals = []  # keys
        self.key_uid_map = {}  # map keys to uids
        self.uid_key_map = {}  # map uids to keys
        self.seed = seed
        self.seed_high = 2 ** 32  # exclusive

        self.stats = {"num_cache_used": 0}

        self.transaction_context = dummy_context
        self._start_sec = time.time()
        self.elapsed_times = {}

        self._run_function = run_function
        self.num_workers = num_workers

        if (cache_key is not None) and (cache_key != "to_dict"):
            if callable(cache_key):
                self._gen_uid = cache_key
            elif cache_key == "uuid":
                self._gen_uid = lambda d: uuid.uuid4()
            else:
                raise DeephyperRuntimeError(
                    'The "cache_key" parameter of an Evaluator must be a callable!'
                )
        else:
            self._gen_uid = lambda d: self.encode(d)

        moduleName = self._run_function.__module__
        if moduleName == "__main__":
            raise DeephyperRuntimeError(
                f'Evaluator will not execute function " {run_function.__name__}" because it is in the __main__ module.  Please provide a function imported from an external module!'
            )

    @staticmethod
    def create(
        run_function,
        cache_key=None,
        method="subprocess",
        redis_address=None,
        num_workers=None,
        **kwargs,
    ):
        available_methods = [
            "balsam",
            "subprocess",
            "processPool",
            "threadPool",
            "__mpiPool",
            "ray",
        ]

        if not method in available_methods:
            raise DeephyperRuntimeError(
                f'The method "{method}" is not a valid method for an Evaluator!'
            )

        if method == "balsam":
            from deephyper.evaluator._balsam import BalsamEvaluator

            Eval = BalsamEvaluator(run_function, cache_key=cache_key, **kwargs)
        elif method == "subprocess":
            from deephyper.evaluator._subprocess import SubprocessEvaluator

            Eval = SubprocessEvaluator(run_function, cache_key=cache_key, **kwargs)
        elif method == "processPool":
            from deephyper.evaluator._processPool import ProcessPoolEvaluator

            Eval = ProcessPoolEvaluator(run_function, cache_key=cache_key, **kwargs)
        elif method == "threadPool":
            from deephyper.evaluator._threadPool import ThreadPoolEvaluator

            Eval = ThreadPoolEvaluator(run_function, cache_key=cache_key, **kwargs)
        elif method == "__mpiPool":
            from deephyper.evaluator._mpiWorkerPool import MPIWorkerPool

            Eval = MPIWorkerPool(run_function, cache_key=cache_key, **kwargs)
        elif method == "ray":
            from deephyper.evaluator._ray_evaluator import RayEvaluator

            Eval = RayEvaluator(
                run_function, cache_key=cache_key, redis_address=redis_address, **kwargs
            )

        # Override the number of workers if passed as an argument
        if not (num_workers is None) and type(num_workers) is int:
            Eval.num_workers = num_workers

        return Eval

    def encode(self, x):
        if not isinstance(x, dict):
            raise ValueError(f"Expected dict, but got {type(x)}")
        return json.dumps(x, cls=self.encoder)

    def _elapsed_sec(self):
        return time.time() - self._start_sec

    def decode(self, key):
        """from JSON string to x (list)
        """
        x = json.loads(key)
        if not isinstance(x, dict):
            raise ValueError(f"Expected dict, but got {type(x)}")
        return x

    def add_eval(self, x):
        if (
            x.get("seed") is not None or self.seed is not None
        ):  # numpy seed fixed in Search.__init__
            x["seed"] = np.random.randint(
                0, self.seed_high
            )  # must be between (0, 2**32-1)

        key = self.encode(x)
        self.requested_evals.append(key)
        uid = self._gen_uid(x)
        if uid in self.key_uid_map.values():
            self.stats["num_cache_used"] += 1
            logger.info(f"UID: {uid} already evaluated; skipping execution")
        else:
            future = self._eval_exec(x)
            logger.info(f"Submitted new eval of {x}")
            future.uid = uid
            self.pending_evals[uid] = future
            self.uid_key_map[uid] = key
        self.key_uid_map[key] = uid
        return uid

    def add_eval_batch(self, XX):
        uids = []
        with self.transaction_context():
            for x in XX:
                uid = self.add_eval(x)
                uids.append(uid)
        return uids

    def _eval_exec(self, x):
        raise NotImplementedError

    def wait(self, futures, timeout=None, return_when="ANY_COMPLETED"):
        raise NotImplementedError

    @staticmethod
    def _parse(run_stdout):
        fail_return_value = Evaluator.FAIL_RETURN_VALUE
        y = fail_return_value
        for line in run_stdout.split("\n"):
            if "DH-OUTPUT:" in line.upper():
                try:
                    y = float(line.split()[-1])
                except ValueError:
                    logger.exception("Could not parse DH-OUTPUT line:\n" + line)
                    y = fail_return_value
                break
        if isnan(y):
            y = fail_return_value
        return y

    @property
    def _runner_executable(self):
        funcName = self._run_function.__name__
        moduleName = self._run_function.__module__
        assert moduleName != "__main__"
        module = sys.modules[moduleName]
        modulePath = os.path.dirname(os.path.abspath(module.__file__))
        runnerPath = os.path.abspath(runner.__file__)
        runner_exec = " ".join(
            (self.PYTHON_EXE, runnerPath, modulePath, moduleName, funcName)
        )
        return runner_exec

    def await_evals(self, uids, timeout=None):
        """Waiting for a collection of tasks.

        Args:
            uids (list(uid)): the list of X values that we are waiting to finish.
            timeout (float, optional): waiting time if a float, or infinite waiting time if None

        Returns:
            list: list of results from awaited task.
        """
        keys = [self.uid_key_map[uid] for uid in uids]
        futures = {
            uid: self.pending_evals[uid] for uid in set(uids) if uid in self.pending_evals
        }
        logger.info(f"Waiting on {len(futures)} evals to finish...")

        logger.info(f"Blocking on completion of {len(futures)} pending evals")
        self.wait(futures.values(), timeout=timeout, return_when="ALL_COMPLETED")
        # TODO: on TimeoutError, kill the evals that did not finish; return infinity
        for uid in futures:
            y = futures[uid].result()
            self.elapsed_times[uid] = self._elapsed_sec()
            del self.pending_evals[uid]
            self.finished_evals[uid] = y
        for (key, uid) in zip(keys, uids):
            y = self.finished_evals[uid]
            # same printing required in get_finished_evals because of logs parsing
            x = self.decode(key)
            logger.info(f"Requested eval x: {x} y: {y}")
            try:
                self.requested_evals.remove(key)
            except ValueError:
                pass
            yield (x, y)

    def get_finished_evals(self, timeout=0.5):
        futures = self.pending_evals.values()
        try:
            waitRes = self.wait(futures, timeout=timeout, return_when="ANY_COMPLETED")
        except TimeoutError:
            pass
        else:
            for future in waitRes.done + waitRes.failed:
                uid = future.uid
                y = future.result()
                logger.info(f"New eval finished: {uid} --> {y}")
                self.elapsed_times[uid] = self._elapsed_sec()
                del self.pending_evals[uid]
                self.finished_evals[uid] = y

        for key in self.requested_evals[:]:
            uid = self.key_uid_map[key]
            if uid in self.finished_evals:
                self.requested_evals.remove(key)
                x = self.decode(key)
                y = self.finished_evals[uid]
                logger.info(f"Requested eval x: {x} y: {y}")
                yield (x, y)

    @property
    def counter(self):
        return len(self.finished_evals) + len(self.pending_evals)

    def num_free_workers(self):
        num_evals = len(self.pending_evals)
        logger.debug(f"{num_evals} pending evals; {self.num_workers} workers")
        return max(self.num_workers - num_evals, 0)

    def convert_for_csv(self, val):
        if type(val) is list:
            return str(val)
        else:
            return val

    def dump_evals(self, saved_key: str = None, saved_keys: list = None):
        """Dump evaluations to 'results.csv' file.

        If both arguments are set to None, then all keys for all points will be added to the CSV file. Keys are columns and values are used to fill rows.


        Args:
            saved_key (str, optional): If is a key corresponding to an element of points which should be a list. Defaults to None.
            saved_keys (list|callable, optional): If is a list of key corresponding to elements of points' dictonnaries then it will add them to the CSV file. If callable such as:
                >>> def saved_keys(self, val: dict):
                >>>     res = {
                >>>         "learning_rate": val["hyperparameters"]["learning_rate"],
                >>>         "batch_size": val["hyperparameters"]["batch_size"],
                >>>         "ranks_per_node": val["hyperparameters"]["ranks_per_node"],
                >>>         "arch_seq": str(val["arch_seq"]),
                >>>     }
                >>> return res.
            Then it will add the result to the CSV file. Defaults to None.
        """
        if not self.finished_evals:
            return

        resultsList = []

        for key, uid in self.key_uid_map.items():
            if uid not in self.finished_evals:
                continue

            if saved_key is None and saved_keys is None:
                result = self.decode(key)
            elif type(saved_key) is str:
                result = {str(i): v for i, v in enumerate(self.decode(key)[saved_key])}
            elif type(saved_keys) is list:
                decoded_key = self.decode(key)
                result = {k: self.convert_for_csv(decoded_key[k]) for k in saved_keys}
            elif callable(saved_keys):
                decoded_key = self.decode(key)
                result = saved_keys(decoded_key)
            result["objective"] = self.finished_evals[uid]
            result["elapsed_sec"] = self.elapsed_times[uid]
            resultsList.append(result)

        with open("results.csv", "w") as fp:
            columns = resultsList[0].keys()
            writer = csv.DictWriter(fp, columns)
            writer.writeheader()
            writer.writerows(resultsList)
