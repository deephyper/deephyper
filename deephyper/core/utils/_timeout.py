import multiprocessing
import multiprocessing.pool

from deephyper.core.exceptions import SearchTerminationError


def terminate_on_timeout(timeout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(processes=1)
            results = pool.apply_async(func, args, kwargs)
            pool.close()
            try:
                return results.get(timeout)
            except multiprocessing.TimeoutError:
                raise SearchTerminationError(f"Search timeout expired after: {timeout}")
            finally:
                pool.terminate()

        return wrapper

    return decorator