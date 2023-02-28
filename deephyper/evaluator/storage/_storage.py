import abc
import importlib
import logging
from typing import Any, Dict, Hashable, List, Tuple, TypeVar

StorageType = TypeVar("StorageType", bound="Storage")

STORAGES = {
    "memory": "_memory_storage.MemoryStorage",
    "redis": "_redis_storage.RedisStorage",
}


class Storage(abc.ABC):
    """An abstract interface representing a storage client."""

    def __init__(self) -> None:
        self.connected = False

    def connect(self) -> StorageType:
        """Connect the storage client to the storage service."""
        self._connect()
        return self

    @staticmethod
    def create(method: str = "memory", method_kwargs: Dict = None) -> StorageType:
        """Static method allowing the creation of a storage client.

        Args:
            method (str, optional): the type of storage client in ``["memory", "redis"]``. Defaults to "memory".
            method_kwargs (Dict, optional): the client keyword-arguments parameters. Defaults to None.

        Raises:
            ValueError: if the type of requested storage client is not valid.

        Returns:
            Storage: the created storage client.
        """

        method_kwargs = method_kwargs if method_kwargs else {}

        logging.info(
            f"Creating Storage(method={method}, method_kwargs={method_kwargs}..."
        )

        if method not in STORAGES.keys():
            val = ", ".join(STORAGES)
            raise ValueError(
                f'The method "{method}" is not a valid method for an Evaluator!'
                f" Choose among the following evalutor types: "
                f"{val}."
            )

        # create the evaluator
        mod_name, attr_name = STORAGES[method].split(".")
        mod = importlib.import_module(f"deephyper.evaluator.storage.{mod_name}")
        storage_cls = getattr(mod, attr_name)
        storage = storage_cls(**method_kwargs)

        logging.info("Creation done")

        return storage

    @abc.abstractmethod
    def _connect(self):
        """Connect the storage client to the storage service."""

    @abc.abstractmethod
    def create_new_search(self) -> Hashable:
        """Create a new search in the store and returns its identifier.

        Returns:
            Hashable: The identifier of the search.
        """

    @abc.abstractmethod
    def create_new_job(self, search_id: Hashable) -> Hashable:
        """Creates a new job in the store and returns its identifier.

        Args:
            search_id (Hashable): The identifier of the search in which a new job
            is created.

        Returns:
            Hashable: The created identifier of the job.
        """

    @abc.abstractmethod
    def store_search_value(
        self, search_id: Hashable, key: Hashable, value: Any
    ) -> None:
        """Stores the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """

    @abc.abstractmethod
    def load_search_value(self, search_id: Hashable, key: Hashable) -> Any:
        """Loads the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to access the value.
        """

    @abc.abstractmethod
    def store_job(self, job_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores the value corresponding to key for job_id.

        Args:
            job_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """

    @abc.abstractmethod
    def store_job_in(
        self, job_id: Hashable, args: Tuple = None, kwargs: Dict = None
    ) -> None:
        """Stores the input arguments of the executed job.

        Args:
            job_id (Hashable): The identifier of the job.
            args (Optional[Tuple], optional): The positional arguments. Defaults to None.
            kwargs (Optional[Dict], optional): The keyword arguments. Defaults to None.
        """

    @abc.abstractmethod
    def store_job_out(self, job_id: Hashable, value: Any) -> None:
        """Stores the output value of the executed job.

        Args:
            job_id (Hashable): The identifier of the job.
            value (Any): The value to store.
        """

    @abc.abstractmethod
    def store_job_metadata(self, job_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores other metadata related to the execution of the job.

        Args:
            job_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the metadata of the given job.
            value (Any): The value to store.
        """

    @abc.abstractmethod
    def load_all_search_ids(self) -> List[Hashable]:
        """Loads the identifiers of all recorded searches.

        Returns:
            List[Hashable]: A list of identifiers of all the recorded searches.
        """

    @abc.abstractmethod
    def load_all_job_ids(self, search_id: Hashable) -> List[Hashable]:
        """Loads the identifiers of all recorded jobs in the search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Hashable]: A list of identifiers of all the jobs.
        """

    @abc.abstractmethod
    def load_search(self, search_id: Hashable) -> dict:
        """Loads the data of a search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            dict: The corresponding data of the search.
        """

    @abc.abstractmethod
    def load_job(self, job_id: Hashable) -> dict:
        """Loads the data of a job.

        Args:
            job_id (Hashable): The identifier of the job.

        Returns:
            dict: The corresponding data of the job.
        """

    @abc.abstractmethod
    def load_metadata_from_all_jobs(
        self, search_id: Hashable, key: Hashable
    ) -> List[Any]:
        """Loads a given metadata value from all jobs.

        Args:
            search_id (Hashable): The identifier of the search.
            key (Hashable): The identifier of the value.

        Returns:
            List[Any]: A list of all the retrieved metadata values.
        """

    @abc.abstractmethod
    def load_out_from_all_jobs(self, search_id: Hashable) -> List[Any]:
        """Loads the output value from all jobs.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Any]: A list of all the retrieved output values.
        """

    @abc.abstractmethod
    def load_jobs(self, job_ids: List[Hashable]) -> dict:
        """Load all data from a given list of jobs' identifiers.

        Args:
            job_ids (list): The list of job identifiers.

        Returns:
            dict: A dictionnary of the retrieved values where the keys are the identifier of jobs.
        """
