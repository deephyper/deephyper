import time
from functools import wraps

# !info [why is it important to use "wraps"]
# !http://gael-varoquaux.info/programming/decoration-in-python-done-right-decorating-and-pickling.html


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
        job.storage.store_job_metadata(job.id, "timestamp_start", timestamp_start)
        results = run_function(job, *args, **kwargs)
        timestamp_end = time.time()
        job.storage.store_job_metadata(job.id, "timestamp_end", timestamp_end)
        return results

    return wrapper
