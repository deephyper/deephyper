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
