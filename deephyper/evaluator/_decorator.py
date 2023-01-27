import time
from functools import wraps

# !info [why is it important to use "wraps"]
# !http://gael-varoquaux.info/programming/decoration-in-python-done-right-decorating-and-pickling.html

from deephyper.evaluator._run_function_utils import standardize_run_function_output


def profile(run_function):
    """Decorator to use on a ``run_function`` to profile its execution-time. It is to be used such as:

    .. code-block::

        @profile
        def run(config):
            ...
            return y

    Args:
        run_function (function): the function to decorate.

    Returns:
        function: a decorated function.
    """

    @wraps(run_function)
    def wrapper(job, *args, **kwargs):
        timestamp_start = time.time()
        output = run_function(job, *args, **kwargs)
        timestamp_end = time.time()

        output = standardize_run_function_output(output)
        metadata = {"timestamp_start": timestamp_start, "timestamp_end": timestamp_end}
        metadata.update(output["metadata"])
        output["metadata"] = metadata

        return output

    return wrapper
