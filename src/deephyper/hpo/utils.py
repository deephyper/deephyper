"""Utilities from search algorithms."""

import logging
import multiprocessing
import multiprocessing.pool

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)

from deephyper.hpo._search import TimeoutReached

__all__ = [
    "get_inactive_value_of_hyperparameter",
    "terminate_on_timeout",
]


def get_inactive_value_of_hyperparameter(hp):
    """Return the value of an hyperparameter when considered inactive."""
    if isinstance(hp, NumericalHyperparameter):
        return hp.lower
    elif isinstance(hp, CategoricalHyperparameter):
        return hp.choices[0]
    elif isinstance(hp, OrdinalHyperparameter):
        return hp.sequence[0]
    elif isinstance(hp, Constant):
        return hp.value
    else:
        raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")


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
