import copy
from collections.abc import MutableMapping

from typing import Hashable, Callable

from deephyper.evaluator.storage import Storage, MemoryStorage
from deephyper.evaluator._run_function_utils import standardize_run_function_output
from deephyper.stopper._stopper import Stopper


class JobContext:
    search = None


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

    def __init__(self, id, args: dict, run_function: Callable):
        self.id = id
        self.rank = None
        self.args = copy.deepcopy(args)
        self.run_function = run_function
        self.status = self.READY
        self.output = None
        self.metadata = dict()
        self.observations = None
        self.context = JobContext()

    def __repr__(self) -> str:
        if self.rank is not None:
            return f"Job(id={self.id}, rank={self.rank}, status={self.status}, args={self.args})"
        else:
            return f"Job(id={self.id}, status={self.status}, args={self.args})"

    @property
    def parameters(self):
        return self.args

    @property
    def result(self):
        return self.output

    def set_output(self, output):
        self.output = output

    def create_running_job(self, storage, stopper):
        stopper = copy.deepcopy(stopper)
        rjob = RunningJob(self.id, self.args, storage, stopper)
        if stopper is not None and hasattr(stopper, "job"):
            stopper.job = rjob
        return rjob


class HPOJob(Job):
    def __init__(self, id, args: dict, run_function: Callable):
        self.id = id
        self.rank = None
        self.args = copy.deepcopy(args)
        self.run_function = run_function
        self.status = self.READY
        self.output = {"objective": None, "metadata": dict()}
        self.metadata = dict()
        self.observations = None
        self.context = JobContext()

    def __getitem__(self, index):
        args = copy.deepcopy(self.args)
        return (args, self.objective)[index]

    @property
    def objective(self):
        """Objective returned by the run-function."""
        return self.output["objective"]

    @property
    def metadata(self):
        """Metadata of the job stored in the output of run-function."""
        return self.output.get("metadata", dict())

    @metadata.setter
    def metadata(self, value):
        self.output["metadata"] = value

    def set_output(self, output):
        output = standardize_run_function_output(output)
        self.output.update(output)


class RunningJob(MutableMapping):
    """A RunningJob is adapted Job object that is passed to the run-function as input.

    Args:
        id (Hashable, optional): The identifier of the job in the Storage. Defaults to None.
        parameters (dict, optional): The dictionnary of hyperparameters suggested. Defaults to None.
        storage (Storage, optional): The storage client used for the search. Defaults to None.
        stopper (Stopper, optional): The stopper object used for the evaluation. Defaults to None.
    """

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

    def record(self, budget: float, objective: float):
        """Records the current ``budget`` and ``objective`` values in the object and
        pass it to the stopper if one is being used.

        Args:
            budget (float): the budget used.
            objective (float): the objective value obtained.
        """
        if self.stopper:
            self.stopper.observe(budget, objective)
        else:
            self.obs = objective

    def stopped(self) -> bool:
        """Returns True if the RunningJob is using a Stopper and it is stopped. Otherwise it will return False."""
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
