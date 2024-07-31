import unittest
import time

from deephyper.evaluator import Evaluator, profile


# @profile(memory=True)
# @profile(memory=True, memory_limit=250 * (1024**2), memory_tracing_interval=0.01)
def run(job):
    x = []
    for i in range(100_000):
        # print(i)
        x.append([1] * 2000)
        time.sleep(0.01)

    return sum(x)


def _run_preprocessing(job):
    import numpy as np

    from sklearn.cluster import FeatureAgglomeration
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline

    n_features = 38  # * 2
    n_samples = 16
    X = np.random.rand(n_samples, n_features)

    pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("feat", FeatureAgglomeration(n_clusters=2)),
        ]
    )

    X = pipeline.fit_transform(X)

    return {"objective": 0, "metadata": {"X_shape": X.shape}}


@profile(memory=True, memory_limit=1024**3, memory_tracing_interval=0.01)
def run_preprocessing_1(job):
    return _run_preprocessing(job)


def run_preprocessing_2(job):
    return _run_preprocessing(job)


class TestMemoryLimit(unittest.TestCase):
    def test_memory_limit_with_profile_decorator(self):

        evaluator = Evaluator.create(run_preprocessing_1, method="serial")
        tasks = [{"x": i} for i in range(1)]
        evaluator.submit(tasks)
        jobs = evaluator.gather("ALL")
        result = [job.result for job in jobs]
        metadata = [job.metadata for job in jobs]
        assert result[0] == "F_memory_limit_exceeded"

    def test_memory_limit_with_profile_decorator_as_function(self):

        run_profiled = profile(
            memory=True,
            memory_limit=1024**3,
            memory_tracing_interval=0.01,
            register=False,  # !Required to make it work
        )(run_preprocessing_2)
        evaluator = Evaluator.create(
            run_profiled,
            method="serial",
        )
        tasks = [{"x": i} for i in range(1)]
        evaluator.submit(tasks)
        jobs = evaluator.gather("ALL")
        result = [job.result for job in jobs]
        metadata = [job.metadata for job in jobs]
        assert result[0] == "F_memory_limit_exceeded"


if __name__ == "__main__":
    test = TestMemoryLimit()
    # test.test_memory_limit_with_profile_decorator()
    test.test_memory_limit_with_profile_decorator_as_function()
