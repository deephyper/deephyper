import copy
import logging
import pickle
from collections.abc import MutableMapping
from typing import Hashable

import numpy as np

from deephyper.evaluator.mpi import MPI

# A good reference about one-sided communication with MPI
# https://enccs.github.io/intermediate-mpi/one-sided-concepts/


class MPIWinMutableMapping(MutableMapping):
    """Dict like object shared between MPI processes using one-sided communication.

    Args:
        default_value (dict):
            The default value of the mutable mapping at initialization.
            Defaults to ``None`` for empty dict.
        comm (MPI.Comm):
            An MPI communicator.
        size (int):
            The total size of the shared memory in bytes. Defaults to ``104857600`` for 100MB.
        root (int):
            The MPI rank where the shared memory window is hosted.
    """

    HEADER_SIZE = 8  # Reserve 8 bytes for size header

    # Use to share state when pickling arguments of function
    COUNTER = 0  # Counter of created instances
    CACHE = {}

    def __init__(
        self,
        default_value: dict = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        size: int = 104857600,
        root: int = 0,
    ):
        logging.info("Creating MPIWinMutableMapping ...")
        self.comm = comm
        self.root = root
        self.locked = False
        self._session_is_started = False
        self._session_is_ready_only = False

        # Allocate shared memory
        # Creates a window from already allocated MPI shared memory.
        logging.info("Creating MPI.Win ...")
        self.win = MPI.Win.Allocate_shared(size, 1, comm=comm)
        buf, itemsize = self.win.Shared_query(self.root)
        logging.info("MPI.Win created")

        # Use a NumPy array to interface with the shared memory
        self.shared_memory = np.ndarray(buffer=buf, dtype=np.byte, shape=(size,))

        # Local cache for the dictionary
        if default_value is None:
            self.local_dict = {}
        else:
            self.local_dict = copy.deepcopy(default_value)
        self._cache_id = MPIWinMutableMapping.COUNTER
        MPIWinMutableMapping.COUNTER += 1
        MPIWinMutableMapping.CACHE[self._cache_id] = self

        if self.comm.Get_rank() == self.root:
            self.lock()
            self._write_dict()
            self.unlock()

        self.comm.Barrier()  # Synchronize processes
        logging.info("MPIWinMutableMapping created")

    def _lazy_read_dict(self):
        """Performs the read if not in a session."""
        if not self._session_is_started:
            self._read_dict()

    def _read_dict(self):
        """Read the dictionnary state from the shared memory."""
        # Deserialize the dictionary from shared memory
        try:
            size = int.from_bytes(self.shared_memory[: self.HEADER_SIZE], byteorder="big")
            if size > 0:
                raw_data = self.shared_memory[self.HEADER_SIZE : self.HEADER_SIZE + size].tobytes()
                self.local_dict = pickle.loads(raw_data)
            else:
                self.local_dict = {}
        except Exception as e:
            logging.error(f"Error reading shared memory: {e}")
            self.local_dict = {}

    def _lazy_write_dict(self):
        """Performs the write if not in a session."""
        if not self._session_is_started:
            self._write_dict()

    def _write_dict(self):
        """Write the dictionnary state to the shared memory."""
        # Serialize the dictionary to shared memory
        serialized = pickle.dumps(self.local_dict)
        size = len(serialized)
        if size + self.HEADER_SIZE > self.shared_memory.size:
            raise ValueError("Shared memory is too small for the dictionary.")

        self.shared_memory[: self.HEADER_SIZE] = np.frombuffer(
            size.to_bytes(self.HEADER_SIZE, byteorder="big"), dtype=np.byte
        )
        self.shared_memory[self.HEADER_SIZE : self.HEADER_SIZE + size] = np.frombuffer(
            serialized, dtype=np.byte
        )
        self.shared_memory[self.HEADER_SIZE + size :] = 0

    def __getitem__(self, key):
        self.lock()
        self._lazy_read_dict()
        self.unlock()
        return self.local_dict[key]

    def __setitem__(self, key, value):
        self.lock()
        self._lazy_read_dict()
        self.local_dict[key] = value
        self._lazy_write_dict()
        self.unlock()

    def __delitem__(self, key):
        self.lock()
        self._lazy_read_dict()
        del self.local_dict[key]
        self._lazy_write_dict()
        self.unlock()

    def __iter__(self):
        self.lock()
        self._lazy_read_dict()
        self.unlock()
        return iter(self.local_dict)

    def __len__(self):
        self.lock()
        self._lazy_read_dict()
        self.unlock()
        return len(self.local_dict)

    def __repr__(self):
        self.lock()
        self._lazy_read_dict()
        self.unlock()
        return repr(self.local_dict)

    def __call__(self, ready_only: bool = False):
        self._session_is_ready_only = ready_only
        return self

    def __enter__(self):
        self.session_start()
        return self

    def __exit__(self, type, value, traceback):
        self.session_finish()

    def lock(self):
        """Acquire the lock. Blocking operation."""
        if not self.locked:
            self.win.Lock(self.root)
            self.locked = True

    def unlock(self):
        """Release the lock."""
        if self.locked:
            self.win.Unlock(self.root)
            self.locked = False

    def session_start(self, ready_only: bool = False):
        if self._session_is_started:
            raise RuntimeError("A session has already been started without being finished!")

        self._session_is_started = True
        self._session_is_ready_only = ready_only

        self.lock()

        self._read_dict()

    def session_finish(self):
        if not self._session_is_started:
            raise RuntimeError("No session has been started!")

        if not self._session_is_ready_only:
            self._write_dict()

        self.unlock()

        self._session_is_started = False
        self._session_is_ready_only = True

    # This can create a deadlock if not called by all processes!
    def __del__(self):
        self.win.Free()
        self.CACHE.pop(self._cache_id)

    def incr(self, key: Hashable, amount=1):
        """Atomic operator that increments and returns the resulting value."""
        keys = key.split(".")

        assert len(keys) > 0

        # Case where the key is at the root of the mapping
        if len(keys) == 1:
            self.lock()
            self._lazy_read_dict()
            self.local_dict[key] += amount
            self._lazy_write_dict()
            self.unlock()
            return self.local_dict[key]

        # Case where the key is JSON path of type "key0.key1.key2"
        else:
            self.lock()
            self._lazy_read_dict()
            mapping = self.local_dict
            for key in keys[:-1]:
                mapping = mapping[key]
            key = keys[-1]
            mapping[key] += amount
            self._lazy_write_dict()
            self.unlock()
            return mapping[key]
