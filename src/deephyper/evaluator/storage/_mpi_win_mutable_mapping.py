import copy
import logging
import pickle
import threading
import weakref
from collections.abc import MutableMapping
from contextlib import contextmanager
from typing import Hashable

import numpy as np

from deephyper.evaluator.mpi import MPI

# A good reference about one-sided communication with MPI
# https://enccs.github.io/intermediate-mpi/one-sided-concepts/


class MPIWinMutableMappingSession(MutableMapping):
    def __init__(self, d: "MPIWinMutableMapping", read_only: bool = False):
        self.d = d
        self.read_only = read_only

    def __enter__(self):
        """Enter context manager."""
        self.d.session_start(self.read_only)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.d.session_finish()

    def __getitem__(self, key):
        return self.d.__getitem__(key)

    def __setitem__(self, key, value):
        self.d.__setitem__(key, value)

    def __delitem__(self, key):
        self.d.__delitem__(key)

    def __iter__(self):
        return self.d.__iter__()

    def __len__(self):
        return self.d.__len__()

    def __repr__(self):
        return self.d.__repr__()

    def get(self, key, default=None):
        """Get value with default, avoiding KeyError."""
        return self.d.get(key, default)

    def keys(self):
        """Return a copy of keys."""
        return self.d.keys()

    def values(self):
        """Return a copy of values."""
        return self.d.values()

    def items(self):
        """Return a copy of items."""
        return self.d.items()


class MPIWinMutableMapping(MutableMapping):
    """Dict like object shared between MPI processes using one-sided communication.

    This implementation is designed for asynchronous access by multiple MPI ranks
    with proper synchronization and error handling.

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
        lock_timeout (float):
            Timeout for lock acquisition in seconds. Defaults to 30.0.
    """

    HEADER_SIZE = 8  # Reserve 8 bytes for size header

    # Instance tracking for proper cleanup
    _instances = weakref.WeakSet()
    _instances_lock = threading.Lock()

    def __init__(
        self,
        default_value: dict = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        size: int = 104857600,
        root: int = 0,
        lock_timeout: float = 30.0,
    ):
        logging.info(f"Creating MPIWinMutableMapping on rank {comm.Get_rank()}...")

        self.comm = comm
        self.root = root
        self.lock_timeout = lock_timeout
        self.locked = False
        self._closed = False
        self._session_is_started = False
        self._session_is_read_only = False
        self._local_lock = threading.RLock()  # For thread safety within process

        # Register this instance for cleanup tracking
        with self._instances_lock:
            self._instances.add(self)

        # Allocate memory (works on multiple nodes)
        logging.info(f"Allocating MPI.Win on rank {comm.Get_rank()}...")

        # Check if all processes are in the local shared comm
        local_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        try:
            if local_comm.Get_size() == self.comm.Get_size():
                logging.info("Using MPI.Win.Allocate_shared")
                self.win = MPI.Win.Allocate_shared(size, 1, comm=comm)
                buf, itemsize = self.win.Shared_query(self.root)
                self.shared_memory = np.ndarray(buffer=buf, dtype=np.byte, shape=(size,))
            else:
                logging.info("Using MPI.Win.Allocate")
                self.win = MPI.Win.Allocate(size, 1, comm=comm)
                self.shared_memory = np.empty((size,), dtype=np.byte)
        finally:
            local_comm.Free()

        logging.info(f"MPI.Win allocated on rank {comm.Get_rank()}")

        # Initialize local dictionary
        if default_value is None:
            self.local_dict = {}
        else:
            self.local_dict = copy.deepcopy(default_value)

        # Initialize shared memory from root process
        if self.comm.Get_rank() == self.root:
            try:
                self._acquire_lock()
                self._write_dict_unsafe()
            finally:
                self._release_lock()

        # Synchronize all processes
        self.comm.Barrier()
        logging.info(f"MPIWinMutableMapping created on rank {comm.Get_rank()}")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other  # identity-based equality

    def _check_not_closed(self):
        """Check if the object has been closed."""
        if self._closed:
            raise RuntimeError("MPIWinMutableMapping has been closed")

    def _acquire_lock(self):
        """Acquire the MPI lock with timeout."""
        if self.locked:
            return

        self._check_not_closed()

        # Use MPI lock - in practice you might want to implement timeout
        # MPI doesn't have native timeout, so this is a simplified version
        try:
            self.win.Lock(self.root)
            self.locked = True
        except Exception as e:
            logging.error(f"Failed to acquire MPI lock: {e}")
            raise

    def _release_lock(self):
        """Release the MPI lock."""
        if self.locked and not self._session_is_started:
            try:
                self.win.Unlock(self.root)
            except Exception as e:
                logging.error(f"Failed to release MPI lock: {e}")
                raise
            finally:
                self.locked = False

    @contextmanager
    def _mpi_lock(self):
        """Context manager for MPI lock."""
        with self._local_lock:  # Thread safety within process
            self._acquire_lock()
            try:
                yield
            finally:
                self._release_lock()

    def _lazy_read_dict(self):
        """Performs the read if not in a session."""
        if not self._session_is_started:
            self._read_dict_unsafe()

    def _read_dict_unsafe(self):
        """Read the dictionary state from shared memory. Must be called with lock held."""
        self._check_not_closed()

        try:
            # Perform MPI Get operation
            self.win.Get(self.shared_memory, target_rank=self.root)
            self.win.Flush(self.root)

            # Read size header
            size = int.from_bytes(self.shared_memory[: self.HEADER_SIZE], byteorder="big")

            if size > 0:
                if size > self.shared_memory.size - self.HEADER_SIZE:
                    raise ValueError(f"Invalid size in shared memory: {size}")

                raw_data = self.shared_memory[self.HEADER_SIZE : self.HEADER_SIZE + size].tobytes()
                self.local_dict = pickle.loads(raw_data)
            else:
                self.local_dict = {}

        except Exception as e:
            logging.error(f"Error reading shared memory: {e}")
            # Don't reset to empty dict - keep existing data
            raise

    def _lazy_write_dict(self):
        """Performs the write if not in a session."""
        if not self._session_is_started:
            self._write_dict_unsafe()

    def _write_dict_unsafe(self):
        """Write the dictionary state to shared memory. Must be called with lock held."""
        self._check_not_closed()

        try:
            # Pre-check size before serialization for very large dicts
            # This is a heuristic - actual size may vary
            estimated_size = sum(len(str(k)) + len(str(v)) for k, v in self.local_dict.items())
            if estimated_size > self.shared_memory.size // 2:  # Conservative estimate
                logging.warning("Dictionary may be too large for shared memory")

            # Serialize the dictionary
            serialized = pickle.dumps(self.local_dict)
            size = len(serialized)

            if size + self.HEADER_SIZE > self.shared_memory.size:
                raise ValueError(
                    f"Shared memory is too small for the dictionary. "
                    f"Required: {size + self.HEADER_SIZE}, Available: {self.shared_memory.size}"
                )

            # Write size header
            self.shared_memory[: self.HEADER_SIZE] = np.frombuffer(
                size.to_bytes(self.HEADER_SIZE, byteorder="big"), dtype=np.byte
            )

            # Write serialized data
            self.shared_memory[self.HEADER_SIZE : self.HEADER_SIZE + size] = np.frombuffer(
                serialized, dtype=np.byte
            )

            # Zero out remaining space for security
            self.shared_memory[self.HEADER_SIZE + size :] = 0

            # Perform MPI Put operation
            self.win.Put(self.shared_memory, target_rank=self.root)
            self.win.Flush(self.root)

        except Exception as e:
            logging.error(f"Error writing to shared memory: {e}")
            raise

    def __getitem__(self, key):
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            return self.local_dict[key]

    def __setitem__(self, key, value):
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            self.local_dict[key] = value
            self._lazy_write_dict()

    def __delitem__(self, key):
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            del self.local_dict[key]
            self._lazy_write_dict()

    def __iter__(self):
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            # Return a copy to avoid issues with concurrent modification
            return iter(list(self.local_dict.keys()))

    def __len__(self):
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            return len(self.local_dict)

    def __repr__(self):
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            return repr(self.local_dict)

    def get(self, key, default=None):
        """Get value with default, avoiding KeyError."""
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            return self.local_dict.get(key, default)

    def keys(self):
        """Return a copy of keys."""
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            return list(self.local_dict.keys())

    def values(self):
        """Return a copy of values."""
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            return list(self.local_dict.values())

    def items(self):
        """Return a copy of items."""
        self._check_not_closed()
        with self._mpi_lock():
            self._lazy_read_dict()
            return list(self.local_dict.items())

    def __call__(self, read_only: bool = False):
        """Prepare for context manager usage."""
        return MPIWinMutableMappingSession(self, read_only)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()

    def session_start(self, read_only: bool = False):
        """Start a session for multiple operations."""
        self._check_not_closed()

        if self._session_is_started:
            raise RuntimeError("A session has already been started without being finished!")

        with self._local_lock:
            self._session_is_started = True
            self._session_is_read_only = read_only

            self._acquire_lock()
            self._read_dict_unsafe()

    def session_finish(self):
        """Finish the current session."""
        self._check_not_closed()

        if not self._session_is_started:
            raise RuntimeError("No session has been started!")

        with self._local_lock:
            try:
                if not self._session_is_read_only:
                    self._write_dict_unsafe()
            finally:
                self._session_is_started = False
                self._session_is_read_only = False
                self._release_lock()

    def incr(self, key: Hashable, amount: float = 1) -> float:
        """Atomic increment operation that works with nested keys.

        Args:
            key: Either a simple key or a tuple of keys for nested access
            amount: Amount to increment by

        Returns:
            The new value after increment
        """
        self._check_not_closed()

        with self._mpi_lock():
            self._lazy_read_dict()

            # Handle nested keys passed as tuple
            if isinstance(key, tuple):
                keys = key
            elif isinstance(key, str) and "." in key:
                # Support legacy dot notation
                keys = key.split(".")
            else:
                keys = (key,)

            # Navigate to the nested location
            current = self.local_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Increment the final key
            final_key = keys[-1]
            if final_key not in current:
                current[final_key] = 0

            current[final_key] += amount
            result = current[final_key]

            self._lazy_write_dict()
            return result

    def close(self):
        """Explicitly close the MPI window. This is a collective operation."""
        if self._closed:
            return

        logging.info(f"Closing MPIWinMutableMapping on rank {self.comm.Get_rank()}")

        # End any active session
        if self._session_is_started:
            try:
                self.session_finish()
            except Exception as e:
                logging.warning(f"Error finishing session during close: {e}")

        # Release any held locks
        if self.locked:
            try:
                self._release_lock()
            except Exception as e:
                logging.warning(f"Error releasing lock during close: {e}")

        # Mark as closed
        self._closed = True

        # Remove from instance tracking
        with self._instances_lock:
            self._instances.discard(self)

        # Free the MPI window (collective operation)
        try:
            self.win.Free()
            logging.info(f"MPI window freed on rank {self.comm.Get_rank()}")
        except Exception as e:
            logging.error(f"Error freeing MPI window: {e}")

    @classmethod
    def close_all(cls):
        """Close all active instances. Collective operation."""
        with cls._instances_lock:
            instances = list(cls._instances)

        for instance in instances:
            try:
                instance.close()
            except Exception as e:
                logging.error(f"Error closing instance: {e}")

    def __del__(self):
        """Destructor - warns if not properly closed."""
        if not self._closed:
            logging.warning(
                "MPIWinMutableMapping was not explicitly closed. "
                "Call close() or use as context manager to avoid issues."
            )
