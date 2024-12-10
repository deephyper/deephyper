import logging
import multiprocessing
import multiprocessing.pool

from deephyper.core.exceptions import TimeoutReached


def terminate_on_timeout(timeout, func, *args, **kwargs):
    """High order function to wrap the call of a function in a thread to monitor its execution time.

    >>> import functools
    >>> f_timeout = functools.partial(terminate_on_timeout, 10, f)
    >>> f_timeout(1, b=2)

    Args:
        timeout (int): timeout in seconds.
        func (function): function to call.
        *args: positional arguments to pass to the function.
        **kwargs: keyword arguments to pass to the function.
    """
    pool = multiprocessing.pool.ThreadPool(processes=1)
    results = pool.apply_async(func, args, kwargs)
    pool.close()
    try:
        return results.get(timeout)
    except multiprocessing.TimeoutError:
        msg = f"Search timeout expired after {timeout} sec."
        logging.warning(msg)
        raise TimeoutReached(msg)
    finally:
        pool.terminate()
