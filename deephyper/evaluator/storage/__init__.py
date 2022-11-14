from deephyper.evaluator.storage._storage import Storage
from deephyper.evaluator.storage._memory_storage import MemoryStorage

__all__ = ["Storage", "MemoryStorage"]


# optional import for RedisStorage
try:
    from deephyper.evaluator.storage._redis_storage import RedisStorage  # noqa: F401

    __all__.append("RedisStorage")
except ImportError:
    pass
