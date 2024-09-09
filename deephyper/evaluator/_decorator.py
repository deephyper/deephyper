import functools
import os
import pickle
import psutil
import sys
import time

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import CancelledError

from deephyper.evaluator._run_function_utils import standardize_run_function_output


def register_inner_function_for_pickle(func):
    """Register former decorated function under a new name to be called in subprocess within
    the decorator.
    See: https://stackoverflow.com/questions/73146709/python-process-inside-decorator
    """
    prefix = "profiled_"
    func_name = func.__qualname__
    saved_name = prefix + func_name
    module_name = pickle.whichmodule(func, func_name)
    module = sys.modules[module_name]
    setattr(module, saved_name, func)
    func.__qualname__ = saved_name


# Example from https://github.com/dabeaz/python-cookbook/blob/master/src/9/defining_a_decorator_that_takes_an_optional_argument/example.py
def profile(
    _func=None,
    *,
    memory: bool = False,
    memory_limit: int = -1,
    memory_tracing_interval: float = 0.1,
    raise_exception: bool = False,
    register=True,
):
    """Decorator to use on a ``run_function`` to profile its execution-time and peak memory usage.

    By default, only the run-time is measured, for example by using the decorator as follows:

    .. code-block::

        @profile
        def run(config):
            ...
            return y

    If the ``memory`` argument is set to ``True``, the memory usage is also measured, for example by using the decorator as follows:

    .. code-block::

        @profile(memory=True)
        def run(config):
            ...
            return y

    If the ``memory_limit` is used then the call will be cancelled (when possible) if the memory usage exceeds the limit, for example by using the decorator as follows:

    .. code-block::

        @profile(memory=True, memory_limit=0.1 * 1024**3, memory_tracing_interval=0.01)
        def run(config):
            ...
            return y

    Args:
        memory (bool): If ``True``, the memory usage is measured. The measured memory, in bytes, accounts for the whole process. Defaults to ``False``.
        memory_limit (int): In bytes, if set to a positive integer, the memory usage is measured at regular intervals and the function is interrupted if the memory usage exceeds the limit. If set to ``-1``, only the peak memory is measured. If the executed function is busy outside of the Python interpretor, this mechanism will not work properly. Defaults to ``-1``.
        memory_tracing_interval (float): In seconds, the interval at which the memory usage is measured. Defaults to ``0.1``.
        register (bool): Register the called function to be pickalable and executed in a subprocess when the we use as decorator ``@profile``.
    Returns:
        function: a decorated function.
    """

    def decorator_profile(func):

        if register and memory:
            register_inner_function_for_pickle(func)

        @functools.wraps(func)
        def wrapper_profile(*args, **kwargs):
            timestamp_start = time.time()

            # Measure peak memory
            if memory:

                p = psutil.Process()  # get the current process

                output = None

                # with ThreadPoolExecutor(max_workers=1) as executor:
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(os.getpid)
                    pid = future.result()
                    p = psutil.Process(pid)

                    future = executor.submit(func, *args, **kwargs)

                    memory_peak = p.memory_info().rss

                    while not future.done():

                        # in bytes (not the peak memory but last snapshot)
                        memory_peak = max(p.memory_info().rss, memory_peak)

                        if memory_limit > 0 and memory_peak > memory_limit:
                            p.kill()
                            future.cancel()
                            output = "F_memory_limit_exceeded"

                            if raise_exception:
                                raise CancelledError(
                                    f"Memory limit exceeded: {memory_peak} > {memory_limit}"
                                )

                            break

                        time.sleep(memory_tracing_interval)

                    if output is None:
                        output = future.result()

            # Regular call without memory profiling
            else:

                output = func(*args, **kwargs)

            timestamp_end = time.time()

            output = standardize_run_function_output(output)
            metadata = {
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
            }

            if memory:
                metadata["memory"] = memory_peak

            metadata.update(output["metadata"])
            output["metadata"] = metadata

            return output

        return wrapper_profile

    if _func is None:
        return decorator_profile
    else:
        return decorator_profile(_func)


def slow_down(_func=None, *, rate=1):
    """Sleep given amount of seconds before calling the function"""

    def decorator_slow_down(func):
        @functools.wraps(func)
        def wrapper_slow_down(*args, **kwargs):
            time.sleep(rate)
            return func(*args, **kwargs)

        return wrapper_slow_down

    if _func is None:
        return decorator_slow_down
    else:
        return decorator_slow_down(_func)
