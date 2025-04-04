import copy
import logging
from typing import Any, Dict, Hashable, List, Tuple

from deephyper.evaluator.mpi import MPI
from deephyper.evaluator.storage._mpi_win_mutable_mapping import MPIWinMutableMapping
from deephyper.evaluator.storage._storage import Storage


class MPIWinStorage(Storage):
    """Storage based on MPIWinMutableMapping that uses one-sided communication for shared memory."""

    def __init__(self, comm, size: int = 104857600, root: int = 0) -> None:
        logging.info("Creating MPIWinStorage ...")
        super().__init__()

        self.comm = MPI.COMM_WORLD
        self.root = root
        self._mapping = MPIWinMutableMapping(
            default_value={"search_id_counter": 0, "data": {}},
            comm=self.comm,
            size=size,
            root=root,
        )
        logging.info("MPIWinStorage created")

    def _connect(self):
        self.connected = True

    def __getstate__(self):
        state = {
            "_mapping_cache_id": self._mapping._cache_id,
            "connected": False,
        }
        return state

    def __setstate__(self, newstate):
        self._mapping = MPIWinMutableMapping.CACHE[newstate["_mapping_cache_id"]]
        self.root = self._mapping.root
        self.comm = self._mapping.comm
        self.connect()

    def create_new_search(self) -> Hashable:
        """Create a new search in the store and returns its identifier.

        Returns:
            Hashable: The identifier of the search.
        """
        with self._mapping():
            search_id = self._mapping.incr("search_id_counter") - 1
            search_id = f"{search_id}"
            self._mapping["data"][search_id] = {"job_id_counter": 0, "data": {}}

        return search_id

    def create_new_job(self, search_id: Hashable) -> Hashable:
        """Creates a new job in the store and returns its identifier.

        Args:
            search_id (Hashable): The identifier of the search in which a new job
            is created.

        Returns:
            Hashable: The created identifier of the job.
        """
        with self._mapping():
            partial_id = self._mapping.incr(f"data.{search_id}.job_id_counter") - 1
            partial_id = f"{partial_id}"
            job_id = f"{search_id}.{partial_id}"
            self._mapping["data"][search_id]["data"][partial_id] = {
                "status": 0,
                "in": None,
                "out": None,
                "metadata": {},
                "intermediate": {"budget": [], "objective": []},
            }

        return job_id

    def store_job(self, job_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores the value corresponding to key for job_id.

        Args:
            job_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """
        search_id, partial_id = job_id.split(".")
        with self._mapping():
            self._mapping["data"][search_id]["data"][partial_id][key] = value

    def store_job_in(self, job_id: Hashable, args: Tuple = None, kwargs: Dict = None) -> None:
        """Stores the input arguments of the executed job.

        Args:
            job_id (Hashable): The identifier of the job.
            args (Optional[Tuple], optional): The positional arguments. Defaults to None.
            kwargs (Optional[Dict], optional): The keyword arguments. Defaults to None.
        """
        self.store_job(job_id, key="in", value={"args": args, "kwargs": kwargs})

    def store_job_out(self, job_id: Hashable, value: Any) -> None:
        """Stores the output value of the executed job.

        Args:
            job_id (Hashable): The identifier of the job.
            value (Any): The value to store.
        """
        self.store_job(job_id, key="out", value=value)

    def store_job_metadata(self, job_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores other metadata related to the execution of the job.

        Args:
            job_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the metadata of the given job.
            value (Any): The value to store.
        """
        search_id, partial_id = job_id.split(".")
        with self._mapping():
            self._mapping["data"][search_id]["data"][partial_id]["metadata"][key] = value

    def load_all_search_ids(self) -> List[Hashable]:
        """Loads the identifiers of all recorded searches.

        Returns:
            List[Hashable]: A list of identifiers of all the recorded searches.
        """
        return list(self._mapping["data"].keys())

    def load_all_job_ids(self, search_id: Hashable) -> List[Hashable]:
        """Loads the identifiers of all recorded jobs in the search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Hashable]: A list of identifiers of all the jobs.
        """
        partial_ids = self._mapping["data"][search_id]["data"].keys()
        job_ids = [f"{search_id}.{p_id}" for p_id in partial_ids]
        return job_ids

    def load_search(self, search_id: Hashable) -> dict:
        """Loads the data of a search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            dict: The corresponding data of the search.
        """
        data = self._mapping["data"][search_id]["data"]
        return copy.deepcopy(data)

    def load_job(self, job_id: Hashable) -> dict:
        """Loads the data of a job.

        Args:
            job_id (Hashable): The identifier of the job.

        Returns:
            dict: The corresponding data of the job.
        """
        search_id, partial_id = job_id.split(".")
        data = self._mapping["data"][search_id]["data"][partial_id]
        return copy.deepcopy(data)

    def store_search_value(self, search_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """
        with self._mapping():
            self._mapping["data"][search_id][key] = value

    def load_search_value(self, search_id: Hashable, key: Hashable) -> Any:
        """Loads the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to access the value.
        """
        return self._mapping["data"][search_id][key]

    def load_metadata_from_all_jobs(self, search_id: Hashable, key: Hashable) -> List[Any]:
        """Loads a given metadata value from all jobs.

        Args:
            search_id (Hashable): The identifier of the search.
            key (Hashable): The identifier of the value.

        Returns:
            List[Any]: A list of all the retrieved metadata values.
        """
        search_id
        values = []
        for job_data_i in self._mapping["data"][search_id]["data"].values():
            value_i = job_data_i["metadata"].get(key, None)
            if value_i is not None:
                values.append(value_i)
        return values

    def load_out_from_all_jobs(self, search_id: Hashable) -> List[Any]:
        """Loads the output value from all jobs.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Any]: A list of all the retrieved output values.
        """
        values = []
        for job_data_i in self._mapping["data"][search_id]["data"].values():
            value_i = job_data_i["out"]
            if value_i is not None:
                values.append(value_i)
        return values

    def load_jobs(self, job_ids: List[Hashable]) -> dict:
        """Load all data from a given list of jobs' identifiers.

        Args:
            job_ids (list): The list of job identifiers.

        Returns:
            dict: A dictionnary of the retrieved values where the keys are the identifier of jobs.
        """
        data = {}
        for job_id in job_ids:
            search_id, partial_id = job_id.split(".")
            job_data = self._mapping["data"][search_id]["data"][partial_id]
            data[job_id] = job_data
        return data

    def store_job_status(self, job_id: Hashable, job_status: int):
        """Stores the new job status.

        Args:
            job_id (Hashable): The job identifier.
            job_status (int): The status of the job.
        """
        self.store_job(job_id=job_id, key="status", value=job_status)

    def load_job_status(self, job_id: Hashable) -> int:
        """Loads the status of a job.

        Args:
            job_id (Hashable): The job identifier.

        Returns:
            int: The status of the job.
        """
        return self.load_job(job_id)["status"]

    def __del__(self):
        del self._mapping
