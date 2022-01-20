import time


def profile(run_function):
    def wrapper(*args, **kwargs):
        timestamp_start = time.time()
        objective = run_function(*args, **kwargs)
        timestamp_end = time.time()
        return {
            "objective": objective,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
        }

    return wrapper


@profile
def run_ackley(config):
    return 0


# return
# before: 0
# after: {"objective": 0, "timestamp": ..., "duration": ...}

# timestamp_start_run_function
# timestamp_end_run_function
# timestamp_submit
# timestamp_gather
