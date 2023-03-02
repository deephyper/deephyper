import copy
from collections.abc import MutableMapping

from typing import Hashable

from deephyper.evaluator.storage import Storage, MemoryStorage
from deephyper.evaluator._run_function_utils import standardize_run_function_output
from deephyper.stopper._stopper import Stopper


class Job:
    """Represents an evaluation executed by the ``Evaluator`` class.

    Args:
        id (Any): unique identifier of the job. Usually an integer.
        config (dict): argument dictionnary of the ``run_function``.
        run_function (callable): function executed by the ``Evaluator``
    """

    # Job status states.
    READY = 0
    RUNNING = 1
    DONE = 2

    def __init__(self, id, config: dict, run_function):
        self.id = id
        self.rank = None
        self.config = copy.deepcopy(config)
        self.run_function = run_function
        self.status = self.READY
        self.output = {
            "objective": None,
            "metadata": {"timestamp_submit": None, "timestamp_gather": None},
        }
        self.observations = None

    def __repr__(self) -> str:
        if self.rank is not None:
            return f"Job(id={self.id}, rank={self.rank}, status={self.status}, config={self.config})"
        else:
            return f"Job(id={self.id}, status={self.status}, config={self.config})"

    def __getitem__(self, index):
        cfg = copy.deepcopy(self.config)
        return (cfg, self.objective)[index]

    @property
    def result(self):
        return self.objective

    @property
    def objective(self):
        """Objective returned by the run-function."""
        return self.output["objective"]

    @property
    def metadata(self):
        """Metadata of the job stored in the output of run-function."""
        return self.output["metadata"]

    def set_output(self, output):
        output = standardize_run_function_output(output)
        self.output["objective"] = output["objective"]
        self.output["metadata"].update(output["metadata"])
        self.observations = output.get("observations", None)

    def create_running_job(self, storage, stopper):
        stopper = copy.deepcopy(stopper)
        rjob = RunningJob(self.id, self.config, storage, stopper)
        if stopper is not None and hasattr(stopper, "job"):
            stopper.job = rjob
        return rjob


class RunningJob(MutableMapping):
    def __init__(
        self,
        id: Hashable = None,
        parameters: dict = None,
        storage: Storage = None,
        stopper: Stopper = None,
    ) -> None:
        self.id = id
        self.parameters = parameters

        if storage is None:
            self.storage = MemoryStorage()
            search_id = self.storage.create_new_search()
            self.id = self.storage.create_new_job(search_id)
        else:
            self.storage = storage

        self.stopper = stopper
        self.obs = None

    # @property
    # def config(self):
    #     return self.parameters

    def __getitem__(self, key):

        if key == "job_id":
            return int(self.id.split(".")[-1])

        return self.parameters[key]

    def __setitem__(self, key, value):

        if key == "job_id":
            raise KeyError("Cannot change the 'job_id' of a running job.")

        self.parameters[key] = value

    def __delitem__(self, key):
        del self.parameters[key]

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)

    # def __getitem__(self, k):
    #     """This method is present to simulate the behaviour of a dict. It is used in the ``run_function`` of the ``Evaluator`` class."""

    #     if k == "job_id":
    #         return int(self.id.split(".")[-1])

    #     return self.parameters[k]

    # def get(self, k, default=None):
    #     """This method is present to simulate the behaviour of a dict. It is used in the ``run_function`` of the ``Evaluator`` class."""
    #     return self.parameters.get(k, default)

    # def __contains__(self, k):
    #     """This method is present to simulate the behaviour of a dict. It is used in the ``run_function`` of the ``Evaluator`` class."""
    #     return k in self.parameters

    # def keys(self):
    #     """This method is present to simulate the behaviour of a dict. It is used in the ``run_function`` of the ``Evaluator`` class."""
    #     return self.parameters.keys()

    # def values(self):
    #     """This method is present to simulate the behaviour of a dict. It is used in the ``run_function`` of the ``Evaluator`` class."""
    #     return self.parameters.values()

    # def items(self):
    #     """This method is present to simulate the behaviour of a dict. It is used in the ``run_function`` of the ``Evaluator`` class."""
    #     return self.parameters.items()

    def record(self, budget: float, objective: float):
        if self.stopper:
            self.stopper.observe(budget, objective)
        else:
            self.obs = objective

    def stopped(self):
        if self.stopper:
            return self.stopper.stop()
        else:
            return False

    @property
    def objective(self):
        """If the RunningJob is using a Stopper then it will return observations from the it. Otherwise it will simply return the last objective value recorded."""
        if self.stopper:
            return self.stopper.objective
        else:
            return self.obs
