import pickle

from typing import Any, Dict, Hashable, List, Tuple

import redis

from deephyper.evaluator.storage._storage import Storage


class RedisStorage(Storage):
    """Storage client for Redis.

    The Redis server should be started with the Redis-JSON module loaded.

    Args:
        host (str, optional): The host of the Redis server. Defaults to "localhost".
        port (int, optional): The port of the Redis server. Defaults to 6379.
        db (int, optional): The database of the Redis server. Defaults to 0.
    """

    def __init__(self, host="localhost", port=6379, db=0) -> None:
        super().__init__()

        self._host = host
        self._port = port
        self._db = db

        self._redis = None

    def _connect(self):
        self._redis = redis.Redis(
            host=self._host,
            port=self._port,
            db=self._db,
            charset="utf-8",
            decode_responses=True,
        )
        self.connected = True
        self._redis.setnx("search_id_counter", 0)

    def __getstate__(self):
        state = {
            "_host": self._host,
            "_port": self._port,
            "_db": self._db,
            "_redis": None,
            "connected": False,
        }
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)
        self.connect()

    def create_new_search(self) -> Hashable:
        """Create a new search in the store and returns its identifier.

        Returns:
            Hashable: The identifier of the search.
        """
        search_id_counter = self._redis.incr("search_id_counter", amount=1) - 1
        search_id = f"{search_id_counter}"  # converting to str
        self._redis.rpush("search_id_list", search_id)
        return search_id

    def create_new_job(self, search_id: Hashable) -> Hashable:
        """Creates a new job in the store and returns its identifier.

        Args:
            search_id (Hashable): The identifier of the search in which a new job
            is created.

        Returns:
            Hashable: The created identifier of the job.
        """
        partial_id = (
            self._redis.incr(f"search:{search_id}.job_id_counter", amount=1) - 1
        )
        partial_id = f"{partial_id}"  # converting to str
        job_id = f"{search_id}.{partial_id}"
        self._redis.rpush(f"search:{search_id}.job_id_list", job_id)
        self._redis.json().set(
            f"job:{job_id}", ".", {"in": None, "metadata": {}, "out": None}
        )
        return job_id

    def store_job(self, job_id: Hashable, key: Hashable, value: Any) -> None:
        """Stores the value corresponding to key for job_id.

        Args:
            job_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """
        self._redis.json().set(f"job:{job_id}", f".{key}", value)

    def store_job_in(
        self, job_id: Hashable, args: Tuple = None, kwargs: Dict = None
    ) -> None:
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
        self._redis.json().set(f"job:{job_id}", f".metadata.{key}", value)

    def load_all_search_ids(self) -> List[Hashable]:
        """Loads the identifiers of all recorded searches.

        Returns:
            List[Hashable]: A list of identifiers of all the recorded searches.
        """
        search_ids = self._redis.lrange("search_id_list", 0, -1)
        return search_ids

    def load_all_job_ids(self, search_id: Hashable) -> List[Hashable]:
        """Loads the identifiers of all recorded jobs in the search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Hashable]: A list of identifiers of all the jobs.
        """
        job_ids = self._redis.lrange(f"search:{search_id}.job_id_list", 0, -1)
        return job_ids

    def load_search(self, search_id: Hashable) -> dict:
        """Loads the data of a search.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            dict: The corresponding data of the search.
        """
        job_ids = self.load_all_job_ids(search_id)
        with self._redis.pipeline() as pipe:
            for job_id in job_ids:
                pipe.json().get(f"job:{job_id}", ".")
            data = pipe.execute()
        for i, job_id in enumerate(job_ids):
            data[i]["job_id"] = job_id
        return data

    def load_job(self, job_id: Hashable) -> dict:
        """Loads the data of a job.

        Args:
            job_id (Hashable): The identifier of the job.

        Returns:
            dict: The corresponding data of the job.
        """
        data = self._redis.json().get(f"job:{job_id}", ".")
        return data

    def store_search_value(
        self, search_id: Hashable, key: Hashable, value: Any
    ) -> None:
        """Stores the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to store the value.
            value (Any): The value to store.
        """
        key = f"{search_id}.{key}"
        value = pickle.dumps(value)
        self._redis.set(key, value)

    def load_search_value(self, search_id: Hashable, key: Hashable) -> Any:
        """Loads the value corresponding to key for search_id.

        Args:
            search_id (Hashable): The identifier of the job.
            key (Hashable): A key to use to access the value.
        """
        key = f"{search_id}.{key}"
        value = self._redis.get(key)
        value = pickle.loads(value)
        return value

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
        search_id
        jobs_ids = self.load_all_job_ids(search_id)
        values = []
        for job_id in jobs_ids:
            try:
                value = self._redis.json().get(f"job:{job_id}", f".metadata.{key}")
            except redis.exceptions.ResponseError:
                value = None

            if value is not None:
                values.append(value)
        return values

    def load_out_from_all_jobs(self, search_id: Hashable) -> List[Any]:
        """Loads the output value from all jobs.

        Args:
            search_id (Hashable): The identifier of the search.

        Returns:
            List[Any]: A list of all the retrieved output values.
        """
        jobs_ids = self.load_all_job_ids(search_id)
        values = []
        for job_id in jobs_ids:
            try:
                value = self._redis.json().get(f"job:{job_id}", ".out")
            except redis.exceptions.ResponseError:
                value = None

            if value is not None:
                values.append(value)
        return values

    def load_jobs(self, job_ids: List[Hashable]) -> dict:
        """Load all data from a given list of jobs' identifiers.

        Args:
            job_ids (list): The list of job identifiers.

        Returns:
            dict: A dictionnary of the retrieved values where the keys are the identifier of jobs.
        """
        redis_job_ids = map(lambda jid: f"job:{jid}", job_ids)
        data = self._redis.json().mget(redis_job_ids, ".")
        data = {k: v for k, v in zip(job_ids, data)}
        return data
