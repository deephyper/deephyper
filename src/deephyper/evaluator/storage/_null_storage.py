from typing import Any, Dict, Hashable, List, Tuple

from deephyper.evaluator.storage._storage import Storage


class NullStorage(Storage):
    """Storage client for "null" storage. This backend does not store any data."""

    def __init__(self) -> None:
        super().__init__()
        self._job_counter = 0
        self.connected = True

    def _connect(self):
        pass

    def create_new_search(self) -> Hashable:
        """Create a new search in the store and returns its identifier.

        Returns:
            Hashable: The identifier of the search.
        """
        return "0"

    def create_new_job(self, search_id: Hashable) -> Hashable:
        """Creates a new job in the store and returns its identifier.

        Args:
            search_id (Hashable): The identifier of the search in which a new job
            is created.

        Returns:
            Hashable: The created identifier of the job.
        """
        partial_id = self._job_counter
        partial_id = f"{partial_id}"  # converting to str
        job_id = f"{search_id}.{partial_id}"
        self._job_counter += 1
        return job_id

    def store_job(self, job_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores the value corresponding to key for job_id.

        Args:
            job_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """

    def store_job_in(self, job_id: Hashable, args: Tuple = None, kwargs: Dict = None) -> None:
        """Stores the input arguments of the executed job.

        Args:
            job_id (Hashable): The identifier of the job.
            args (Optional[Tuple], optional): The positional arguments. Defaults to None.
            kwargs (Optional[Dict], optional): The keyword arguments. Defaults to None.
        """

    def store_job_out(self, job_id: Hashable, value: Any) -> None:
        """Stores the output value of the executed job.

        Args:
            job_id (Hashable): The identifier of the job.
            value (Any): The value to store.
        """

    def store_job_metadata(self, job_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores other metadata related to the execution of the job.

        Args:
            job_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the metadata of the given job.
            value (Any): The value to store.
        """

    def load_all_search_ids(self) -> List[Hashable]:
        """Loads the identifiers of all recorded searches.

        Returns:
            List[Hashable]: A list of identifiers of all the recorded searches.
        """
        return ["0"]

    def load_all_job_ids(self, search_id: Hashable) -> List[Hashable]:
        """Loads the identifiers of all recorded jobs in the search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Hashable]: A list of identifiers of all the jobs.
        """
        return [f"0.{i}" for i in range(self._job_counter)]

    def load_search(self, search_id: Hashable) -> dict:
        """Loads the data of a search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            dict: The corresponding data of the search.
        """
        return None

    def load_job(self, job_id: Hashable) -> dict:
        """Loads the data of a job.

        Args:
            job_id (Hashable): The identifier of the job.

        Returns:
            dict: The corresponding data of the job.
        """
        return None

    def store_search_value(self, search_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """

    def load_search_value(self, search_id: Hashable, key: Hashable) -> Any:
        """Loads the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to access the value.
        """
        return None

    def load_metadata_from_all_jobs(self, search_id: Hashable, key: Hashable) -> List[Any]:
        """Loads a given metadata value from all jobs.

        Args:
            search_id (Hashable): The identifier of the search.
            key (Hashable): The identifier of the value.

        Returns:
            List[Any]: A list of all the retrieved metadata values.
        """
        return []

    def load_out_from_all_jobs(self, search_id: Hashable) -> List[Any]:
        """Loads the output value from all jobs.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Any]: A list of all the retrieved output values.
        """
        return []

    def load_jobs(self, job_ids: List[Hashable]) -> dict:
        """Load all data from a given list of jobs' identifiers.

        Args:
            job_ids (list): The list of job identifiers.

        Returns:
            dict: A dictionnary of the retrieved values where the keys are the identifier of jobs.
        """
        return {}

    def store_job_status(self, job_id: Hashable, job_status: int):
        """Stores the new job status.

        Args:
            job_id (Hashable): The job identifier.
            job_status (int): The status of the job.
        """

    def load_job_status(self, job_id: Hashable) -> int:
        """Loads the status of a job.

        Args:
            job_id (Hashable): The job identifier.

        Returns:
            int: The status of the job.
        """
        return 0
