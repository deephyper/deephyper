"""Set of Black-Box functions useful to build examples.
"""
import time
import numpy as np
from deephyper.evaluator import profile


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / d))
    term2 = -np.exp(s2 / d)
    y = term1 + term2 + a + np.exp(1)
    return y


@profile
def run_ackley(config, sleep_loc=2, sleep_scale=0.5):

    # to simulate the computation of an expensive black-box
    if sleep_loc > 0:
        t_sleep = np.random.normal(loc=sleep_loc, scale=sleep_scale)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -ackley(x)  # maximisation is performed
