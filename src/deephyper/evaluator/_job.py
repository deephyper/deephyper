import copy
from collections.abc import MutableMapping
from enum import Enum
from numbers import Number
from typing import Any, Callable, Hashable, Union

import numpy as np

from deephyper.evaluator.storage import MemoryStorage, Storage
from deephyper.stopper._stopper import Stopper


class JobContext:
    search = None


class JobStatus(Enum):
    """Represents the job status states."""

    READY = 0
    RUNNING = 1
    DONE = 2
    CANCELLING = 3  # The job has been requested to be cancelled but is still running.
    CANCELLED = 4


class Job:
    """Represents an evaluation executed by the ``Evaluator`` class.

    Args:
        id (Any): unique identifier of the job. Usually an integer.
        args (dict): argument dictionnary of the ``run_function``.
        run_function (callable): function executed by the ``Evaluator``
    """

    def __init__(
        self,
        id: Hashable,
        args: dict,
        run_function: Callable,
        storage: Storage,
    ):
        self.id = id
        self.args = copy.deepcopy(args)
        self.run_function = run_function
        self.output = None
        self.metadata = dict()
        self.context = JobContext()
        self.storage = storage

    def __repr__(self) -> str:
        return f"Job(id={self.id}, status={self.status}, args={self.args})"

    def set_output(self, output: Any):
        if isinstance(output, dict) and isinstance(output.get("metadata"), dict):
            self.metadata.update(output.pop("metadata"))
            self.output = output.get("output")
        else:
            self.output = output

    def create_running_job(self, stopper):
        stopper = copy.deepcopy(stopper)
        rjob = RunningJob(self.id, self.args, self.storage, stopper)
        if stopper is not None and hasattr(stopper, "job"):
            stopper.job = rjob
        return rjob

    @property
    def status(self) -> JobStatus:
        return JobStatus(self.storage.load_job_status(self.id))

    @status.setter
    def status(self, job_status: JobStatus):
        self.storage.store_job_status(self.id, job_status.value)


class HPOJob(Job):
    def __init__(self, id, args: dict, run_function: Callable, storage: Storage):
        super().__init__(id, args, run_function, storage)

    def __getitem__(self, index):
        args = copy.deepcopy(self.args)
        return (args, self.objective)[index]

    @property
    def objective(self):
        """Objective returned by the run-function."""
        return self.output["objective"]

    @staticmethod
    def standardize_output(
        output: Union[str, float, tuple, list, dict],
    ) -> dict:
        """Transform the output of the run-function to its standard form.

        Possible return values of the run-function are:

        >>> 0
        >>> 0, 0
        >>> "F_something"
        >>> {"objective": 0 }
        >>> {"objective": (0, 0), "metadata": {...}}

        Args:
            output (Union[str, float, tuple, list, dict]): the output of the run-function.

        Returns:
            dict: standardized output of the function.
        """
        # Start by checking if the format output/metadata was used
        metadata = dict()
        if isinstance(output, dict) and "output" in output:
            metadata.update(output.pop("metadata", dict()))
            output = output["output"]

        # output returned a single objective value
        if np.isscalar(output):
            if isinstance(output, str):
                output = {"objective": output}
            elif isinstance(output, Number):
                output = {"objective": float(output)}
            else:
                raise TypeError(
                    "When a scalar type, the output of run-function cannot be of type "
                    f"{type(output)} it should either be a string or a number."
                )

        # output only returned objective values as tuple or list
        elif isinstance(output, (tuple, list)):
            output = {"objective": output}

        elif isinstance(output, dict):
            other_metadata = output.pop("metadata", dict())
            if not isinstance(other_metadata, dict):
                TypeError(
                    "The metadata returned by the run-function should be a dictionnary but are: "
                    f"{other_metadata}."
                )
            metadata.update(other_metadata)

            if "objective" not in output:
                raise ValueError(
                    "The output of the run-function should have a key 'objective' when it is a "
                    "dictionnary."
                )

        else:
            raise TypeError(f"The output of the run-function cannot be of type {type(output)}")

        return output, metadata

    def set_output(self, output):
        output, metadata = HPOJob.standardize_output(output)
        self.output = output
        self.metadata.update(metadata)


class RunningJob(MutableMapping):
    """A RunningJob is an adapted Job object that is passed to the run-function as input.

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

    def __repr__(self) -> str:
        return f"RunningJob(id={self.id}, status={self.status}, parameters={self.parameters})"

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

    @property
    def status(self):
        return JobStatus(self.storage.load_job_status(self.id))

    def record(self, budget: float, objective: float):
        """Records the current ``budget`` and ``objective`` values in the object.

        These are passed to the stopper if one is being used.

        Args:
            budget (float): the budget used.
            objective (float): the objective value obtained.
        """
        if self.stopper:
            self.stopper.observe(budget, objective)
        else:
            self.obs = objective

    def stopped(self) -> bool:
        """Returns True if the RunningJob is using a Stopper and it is stopped.

        Otherwise it will return False.
        """
        if self.stopper:
            return self.stopper.stop()
        else:
            return False

    @property
    def objective(self):
        """If the RunningJob is using a Stopper then it will return observations from the it.

        Otherwise it will simply return the last objective value recorded.
        """
        if self.stopper:
            return self.stopper.objective
        else:
            return self.obs
