from multiprocessing.managers import BaseManager

from deephyper.evaluator.storage._memory_storage import MemoryStorage


BaseManager.register("MemoryStorage", MemoryStorage)


def SharedMemoryStorage():
    """Creates a server process managing a MemoryStorage class and provides Proxy classes
    to processes to which it is passed. It is made to be used in combination with :class:`deephyper.evaluator.ProcessPoolEvaluator`.
    """
    manager = BaseManager()
    manager.start()
    return manager.MemoryStorage()
