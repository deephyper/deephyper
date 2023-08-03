import functools
import time

# !info [why is it important to use "wraps"]
# !http://gael-varoquaux.info/programming/decoration-in-python-done-right-decorating-and-pickling.html

from deephyper.evaluator._run_function_utils import standardize_run_function_output


def profile(*args, **kwargs):
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

    Args:
        memory (bool): If ``True``, the memory usage is measured. Defaults to ``False``.

    Returns:
        function: a decorated function.
    """
    memory = kwargs.get("memory", False)

    def profile_inner(func):
        @functools.wraps(func)
        def wrapper(job, *args, **kwargs):
            timestamp_start = time.time()

            if memory:
                import tracemalloc

                tracemalloc.start()

            output = func(job, *args, **kwargs)

            if memory:
                _, memory_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

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

        return wrapper

    if len(args) > 0 and callable(args[0]):
        return profile_inner(args[0])
    else:
        return profile_inner
