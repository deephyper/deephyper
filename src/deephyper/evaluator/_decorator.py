import asyncio
import functools
import pickle
import psutil
import os
import sys
import time

from concurrent.futures import ProcessPoolExecutor


def register_inner_function_for_pickle(func):
    """Register former decorated function under a new name.

    This is to be called in subprocess within the decorator.

    See: https://stackoverflow.com/questions/73146709/python-process-inside-decorator
    """
    prefix = "profiled_"
    func_name = func.__qualname__
    saved_name = prefix + func_name
    module_name = pickle.whichmodule(func, func_name)
    module = sys.modules[module_name]
    setattr(module, saved_name, func)
    func.__qualname__ = saved_name


# Example from
# https://github.com/dabeaz/python-cookbook/blob/master/src/9/defining_a_decorator_that_takes_an_optional_argument/example.py


def asyncio_run(func, *args, **kwargs):
    """Useful to run async function from subprocess."""
    if asyncio.iscoroutinefunction(func):
        return asyncio.run(func(*args, **kwargs))
    else:
        return func(*args, **kwargs)


def profile(  # noqa: D417
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

    If the ``memory`` argument is set to ``True``, the memory usage is also
    measured, for example by using the decorator as follows:

    .. code-block::

        @profile(memory=True)
        def run(config):
            ...
            return y

    If the ``memory_limit` is used then the call will be cancelled
    (when possible) if the memory usage exceeds the limit, for example by
    using the decorator as follows:

    .. code-block::

        @profile(memory=True, memory_limit=0.1 * 1024**3, memory_tracing_interval=0.01)
        def run(config):
            ...
            return y

    Args:
        memory (bool):
            If ``True``, the memory usage is measured. The measured memory, in
            bytes, accounts for the whole process. Defaults to ``False``.
        memory_limit (int):
            In bytes, if set to a positive integer, the memory usage is
            measured at regular intervals and the function is interrupted if
            the memory usage exceeds the limit. If set to ``-1``, only the
            peak memory is measured. If the executed function is busy outside
            of the Python interpretor, this mechanism will not work properly.
            Defaults to ``-1``.
        memory_tracing_interval (float):
            In seconds, the interval at which the memory usage is measured.
            Defaults to ``0.1``.
        register (bool):
            Register the called function to be pickalable and executed in a
            subprocess when the we use as decorator ``@profile``.

    Returns:
        function: a decorated function.
    """

    def decorator_profile(func):
        if register and memory:
            register_inner_function_for_pickle(func)

        @functools.wraps(func)
        async def async_wrapper_profile(*args, **kwargs):
            timestamp_start = time.time()

            if memory:
                p = psutil.Process()
                output = None

                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(os.getpid)
                    pid = future.result()
                    p = psutil.Process(pid)

                    asyncio_run_func = functools.partial(asyncio_run, func)

                    future = executor.submit(asyncio_run_func, *args, **kwargs)
                    memory_peak = p.memory_info().rss

                    while not future.done():
                        memory_peak = max(p.memory_info().rss, memory_peak)

                        if memory_limit > 0 and memory_peak > memory_limit:
                            p.kill()
                            future.cancel()
                            output = "F_memory_limit_exceeded"

                            if raise_exception:
                                raise MemoryError(
                                    f"Memory limit exceeded: {memory_peak} > {memory_limit}"
                                )

                            break

                        await asyncio.sleep(memory_tracing_interval)

                    if output is None:
                        output = future.result()
            else:
                output = await func(*args, **kwargs)

            timestamp_end = time.time()
            new_metadata = {
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
            }
            if memory:
                new_metadata["memory"] = memory_peak

            # Format correctly the output to return metadata
            if isinstance(output, dict):
                if "output" in output:
                    if "metadata" not in output:
                        output["metadata"] = {}
                else:
                    output = {"output": output, "metadata": {}}
            else:
                output = {"output": output, "metadata": {}}

            output["metadata"].update(new_metadata)
            return output

        @functools.wraps(func)
        def sync_wrapper_profile(*args, **kwargs):
            timestamp_start = time.time()

            if memory:
                p = psutil.Process()
                output = None

                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(os.getpid)
                    pid = future.result()
                    p = psutil.Process(pid)

                    future = executor.submit(func, *args, **kwargs)
                    memory_peak = p.memory_info().rss

                    while not future.done():
                        memory_peak = max(p.memory_info().rss, memory_peak)

                        if memory_limit > 0 and memory_peak > memory_limit:
                            p.kill()
                            future.cancel()
                            output = "F_memory_limit_exceeded"

                            if raise_exception:
                                raise MemoryError(
                                    f"Memory limit exceeded: {memory_peak} > {memory_limit}"
                                )

                            break

                        time.sleep(memory_tracing_interval)

                    if output is None:
                        output = future.result()
            else:
                output = func(*args, **kwargs)

            timestamp_end = time.time()
            new_metadata = {
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
            }
            if memory:
                new_metadata["memory"] = memory_peak

            # Format correctly the output to return metadata
            if isinstance(output, dict):
                if "output" in output:
                    if "metadata" not in output:
                        output["metadata"] = {}
                else:
                    output = {"output": output, "metadata": {}}
            else:
                output = {"output": output, "metadata": {}}

            output["metadata"].update(new_metadata)
            return output

        if asyncio.iscoroutinefunction(func):
            return async_wrapper_profile
        else:
            return sync_wrapper_profile

    if _func is None:
        return decorator_profile
    else:
        return decorator_profile(_func)


def slow_down(_func=None, *, rate=1):
    """Sleep given amount of seconds before calling the function."""

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
