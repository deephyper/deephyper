"""This subpackage provides an interface to implement new storage clients. The base class defining this interface is the :class:`deephyper.evaluator.storage.Storage`. A storage in our langage is a synonym of memory. Different databases  or memory systems can be used through this interface (e.g., key-value storage, relational database, etc.).
"""

from deephyper.evaluator.storage._storage import Storage
from deephyper.evaluator.storage._memory_storage import MemoryStorage
from deephyper.evaluator.storage._null_storage import NullStorage

__all__ = ["Storage", "MemoryStorage", "NullStorage"]


# optional import for RedisStorage
try:
    from deephyper.evaluator.storage._redis_storage import RedisStorage  # noqa: F401

    __all__.append("RedisStorage")
except ImportError:
    pass

# optional import for RayStorage
try:
    from deephyper.evaluator.storage._ray_storage import RayStorage  # noqa: F401

    __all__.append("RayStorage")
except ImportError:
    pass
