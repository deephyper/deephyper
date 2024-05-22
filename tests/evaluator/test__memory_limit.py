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


# @profile(memory=True, memory_limit=0.1 * 1024**3, memory_tracing_interval=0.01)
# @profile(memory=True)
def run_preprocessing(job):
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


class TestMemoryLimit(unittest.TestCase):
    def test_memory_limit(self):

        run_profiled = profile(
            # memory=True, memory_limit=250 * (1024**2), memory_tracing_interval=0.01
            memory=True,
            memory_limit=1 * (1024**3),
            memory_tracing_interval=0.01,
        )(run_preprocessing)
        evaluator = Evaluator.create(run_profiled, method="serial")
        tasks = [{"x": i} for i in range(1)]
        evaluator.submit(tasks)
        jobs = evaluator.gather("ALL")
        result = [job.result for job in jobs]
        metadata = [job.metadata for job in jobs]
        print(f"{result=}")
        print(f"{metadata=}")


if __name__ == "__main__":
    test = TestMemoryLimit()
    test.test_memory_limit()
