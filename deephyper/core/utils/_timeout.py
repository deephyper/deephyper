import multiprocessing
import multiprocessing.pool


from deephyper.core.exceptions import SearchTerminationError


def terminate_on_timeout(timeout, func, *args, **kwargs):
    """High order function to wrap the call of a function in a thread to monitor its execution time."""

    pool = multiprocessing.pool.ThreadPool(processes=1)
    results = pool.apply_async(func, args, kwargs)
    pool.close()
    try:
        return results.get(timeout)
    except multiprocessing.TimeoutError:
        raise SearchTerminationError(f"Search timeout expired after: {timeout}")
    finally:
        pool.terminate()
