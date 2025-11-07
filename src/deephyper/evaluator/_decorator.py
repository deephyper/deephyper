import asyncio
import functools
import inspect
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cloudpickle
import psutil


class CloudpickleProcessPoolExecutor(ProcessPoolExecutor):
    def _adjust_process_count(self):
        # identical to ProcessPoolExecutor except using cloudpickle
        super()._adjust_process_count()

    def _sendback_result(self, call_item, result_item):
        # no change — cloudpickle is used in the worker instead
        super()._sendback_result(call_item, result_item)


def cloudpickle_submit(executor, func, *args, **kwargs):
    # Use cloudpickle.dumps instead of pickle.dumps
    buf = io.BytesIO()
    cloudpickle.dump((func, args, kwargs), buf)
    return executor.submit(_cloudpickle_wrapper, buf.getvalue())


def _cloudpickle_wrapper(serialized):
    func, args, kwargs = cloudpickle.loads(serialized)
    return func(*args, **kwargs)


def _wrap_output(output, metadata):
    """Ensure consistent output format and attach metadata."""
    if not isinstance(output, dict) or "output" not in output:
        output = {"output": output, "metadata": {}}
    output["metadata"].update(metadata)
    return output


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

    It will add the ``m:timestamp_start``, ``m:timestamp_end`` and optionaly ``m:memory`` metadata
    columns to the results.

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

    If the ``memory_limit`` is used then the call will be cancelled
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
            pass
            # register_inner_function_for_pickle(func)

        @functools.wraps(func)
        async def async_wrapper_profile(*args, **kwargs):
            timestamp_start = time.monotonic()

            if memory:
                p = psutil.Process()
                output = None

                with CloudpickleProcessPoolExecutor(max_workers=1) as executor:
                    future = cloudpickle_submit(executor, os.getpid)
                    pid = future.result()
                    p = psutil.Process(pid)

                    asyncio_run_func = functools.partial(asyncio_run, func)

                    future = cloudpickle_submit(executor, asyncio_run_func, *args, **kwargs)
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

            timestamp_end = time.monotonic()
            metadata = {
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
            }
            if memory:
                metadata["memory"] = memory_peak

            return _wrap_output(output, metadata)

        @functools.wraps(func)
        def sync_wrapper_profile(*args, **kwargs):
            timestamp_start = time.monotonic()

            if memory:
                p = psutil.Process()
                output = None

                with CloudpickleProcessPoolExecutor(max_workers=1) as executor:
                    future = cloudpickle_submit(executor, os.getpid)
                    pid = future.result()
                    p = psutil.Process(pid)

                    future = cloudpickle_submit(executor, func, *args, **kwargs)
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

            timestamp_end = time.monotonic()
            metadata = {
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
            }
            if memory:
                metadata["memory"] = memory_peak

            return _wrap_output(output, metadata)

        if inspect.iscoroutinefunction(func):
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
