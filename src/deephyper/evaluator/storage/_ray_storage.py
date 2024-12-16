import ray

from typing import Any, Dict, Hashable, List, Tuple

from deephyper.evaluator.storage._storage import Storage
from deephyper.evaluator.storage._memory_storage import MemoryStorage


class RayStorage(Storage):
    """Storage class using Ray actors.

    The RayStorage is wrapping the MemoryStorage class to be a Ray actor.

    Args:
        address (str, optional):
            Address of the Ray-head. Defaults to ``"auto"``, to connect to the
            local head node.
    """

    ray_storage_counter = 0

    def __init__(self, address="auto") -> None:
        super().__init__()

        self.address = address
        self.actor_name = f"{RayStorage.ray_storage_counter}"
        RayStorage.ray_storage_counter += 1
        self.memory_storage_actor = None

    def _connect(self):
        if self.memory_storage_actor is None:
            self.memory_storage_actor = (
                ray.remote(MemoryStorage)
                .options(name=self.actor_name, namespace="deephyper")
                .remote()
            )
        else:
            self.memory_storage_actor = ray.get_actor(self.actor_name)
        self.connected = True

    def __getstate__(self):
        state = {
            "connected": self.connected,
            "actor_name": self.actor_name,
            "address": self.address,
        }
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)

        if not ray.is_initialized():
            ray.init(address=self.address)

        self.memory_storage_actor = ray.get_actor(self.actor_name, namespace="deephyper")

    def create_new_search(self) -> Hashable:
        """Create a new search in the store and returns its identifier.

        Returns:
            Hashable: The identifier of the search.
        """
        return ray.get(self.memory_storage_actor.create_new_search.remote())

    def create_new_job(self, search_id: Hashable) -> Hashable:
        """Creates a new job in the store and returns its identifier.

        Args:
            search_id (Hashable): The identifier of the search in which a new job
            is created.

        Returns:
            Hashable: The created identifier of the job.
        """
        return ray.get(self.memory_storage_actor.create_new_job.remote(search_id))

    def store_search_value(self, search_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """
        ray.get(self.memory_storage_actor.store_search_value.remote(search_id, key, value))

    def load_search_value(self, search_id: Hashable, key: Hashable) -> Any:
        """Loads the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to access the value.
        """
        return ray.get(self.memory_storage_actor.load_search_value.remote(search_id, key))

    def store_job(self, job_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores the value corresponding to key for job_id.

        Args:
            job_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """
        ray.get(self.memory_storage_actor.store_job.remote(job_id, key, value))

    def store_job_in(self, job_id: Hashable, args: Tuple = None, kwargs: Dict = None) -> None:
        """Stores the input arguments of the executed job.

        Args:
            job_id (Hashable): The identifier of the job.
            args (Optional[Tuple], optional): The positional arguments. Defaults to None.
            kwargs (Optional[Dict], optional): The keyword arguments. Defaults to None.
        """
        ray.get(self.memory_storage_actor.store_job_in.remote(job_id, args, kwargs))

    def store_job_out(self, job_id: Hashable, value: Any) -> None:
        """Stores the output value of the executed job.

        Args:
            job_id (Hashable): The identifier of the job.
            value (Any): The value to store.
        """
        ray.get(self.memory_storage_actor.store_job_out.remote(job_id, value))

    def store_job_metadata(self, job_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores other metadata related to the execution of the job.

        Args:
            job_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the metadata of the given job.
            value (Any): The value to store.
        """
        ray.get(self.memory_storage_actor.store_job_metadata.remote(job_id, key, value))

    def load_all_search_ids(self) -> List[Hashable]:
        """Loads the identifiers of all recorded searches.

        Returns:
            List[Hashable]: A list of identifiers of all the recorded searches.
        """
        return ray.get(self.memory_storage_actor.load_all_search_ids.remote())

    def load_all_job_ids(self, search_id: Hashable) -> List[Hashable]:
        """Loads the identifiers of all recorded jobs in the search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Hashable]: A list of identifiers of all the jobs.
        """
        return ray.get(self.memory_storage_actor.load_all_job_ids.remote(search_id))

    def load_search(self, search_id: Hashable) -> dict:
        """Loads the data of a search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            dict: The corresponding data of the search.
        """
        return ray.get(self.memory_storage_actor.load_search.remote(search_id))

    def load_job(self, job_id: Hashable) -> dict:
        """Loads the data of a job.

        Args:
            job_id (Hashable): The identifier of the job.

        Returns:
            dict: The corresponding data of the job.
        """
        return ray.get(self.memory_storage_actor.load_job.remote(job_id))

    def load_metadata_from_all_jobs(self, search_id: Hashable, key: Hashable) -> List[Any]:
        """Loads a given metadata value from all jobs.

        Args:
            search_id (Hashable): The identifier of the search.
            key (Hashable): The identifier of the value.

        Returns:
            List[Any]: A list of all the retrieved metadata values.
        """
        return ray.get(self.memory_storage_actor.load_metadata_from_all_jobs.remote(search_id, key))

    def load_out_from_all_jobs(self, search_id: Hashable) -> List[Any]:
        """Loads the output value from all jobs.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Any]: A list of all the retrieved output values.
        """
        return ray.get(self.memory_storage_actor.load_out_from_all_jobs.remote(search_id))

    def load_jobs(self, job_ids: List[Hashable]) -> dict:
        """Load all data from a given list of jobs' identifiers.

        Args:
            job_ids (list): The list of job identifiers.

        Returns:
            dict: A dictionnary of the retrieved values where the keys are the identifier of jobs.
        """
        return ray.get(self.memory_storage_actor.load_jobs.remote(job_ids))

    def store_job_status(self, job_id: Hashable, job_status: int):
        """Stores the new job status.

        Args:
            job_id (Hashable): The job identifier.
            job_status (int): The status of the job.
        """
        ray.get(self.memory_storage_actor.store_job_status.remote(job_id, job_status))

    def load_job_status(self, job_id: Hashable) -> int:
        """Loads the status of a job.

        Args:
            job_id (Hashable): The job identifier.

        Returns:
            int: The status of the job.
        """
        return ray.get(self.memory_storage_actor.load_job_status.remote(job_id))
