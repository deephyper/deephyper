import pytest
from deephyper.evaluator import Evaluator, profile


def _run_preprocessing(job):
    import numpy as np

    from sklearn.cluster import FeatureAgglomeration
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline

    n_features = 38 * 2
    n_samples = 32
    X = np.random.rand(n_samples, n_features)

    pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("feat", FeatureAgglomeration(n_clusters=2)),
        ]
    )

    X = pipeline.fit_transform(X)

    return {"objective": 0, "metadata": {"X_shape": X.shape}}


@profile(memory=True, memory_limit=1024**3, memory_tracing_interval=0.2)
def run_preprocessing_1_sync(job):
    return _run_preprocessing(job)


@profile(memory=True, memory_limit=1024**3, memory_tracing_interval=0.2)
async def run_preprocessing_1_async(job):
    return _run_preprocessing(job)


def run_preprocessing_2_sync(job):
    return _run_preprocessing(job)


async def run_preprocessing_2_async(job):
    return _run_preprocessing(job)


@pytest.mark.memory_profiling
def test_memory_limit_with_profile_decorator():
    # With sync function
    evaluator = Evaluator.create(
        run_preprocessing_1_sync,
        method="thread",
    )
    tasks = [{"x": i} for i in range(1)]
    evaluator.submit(tasks)
    jobs = evaluator.gather("ALL")
    result = [job.output for job in jobs]
    metadata = [job.metadata for job in jobs]
    assert result[0] == "F_memory_limit_exceeded"
    assert metadata[0]["memory"] > 1024**3
    evaluator.close()

    # With async function
    evaluator = Evaluator.create(
        run_preprocessing_1_async,
        method="serial",
    )
    tasks = [{"x": i} for i in range(1)]
    evaluator.submit(tasks)
    jobs = evaluator.gather("ALL")
    result = [job.output for job in jobs]
    metadata = [job.metadata for job in jobs]
    assert result[0] == "F_memory_limit_exceeded"
    assert metadata[0]["memory"] > 1024**3
    evaluator.close()


@pytest.mark.memory_profiling
def test_memory_limit_with_profile_decorator_as_function():
    # With sync function
    run_profiled = profile(
        memory=True,
        memory_limit=1024**3,
        memory_tracing_interval=0.2,
        register=False,  # !Required to make it work
    )(run_preprocessing_2_sync)
    evaluator = Evaluator.create(
        run_profiled,
        method="thread",
    )
    tasks = [{"x": i} for i in range(1)]
    evaluator.submit(tasks)
    jobs = evaluator.gather("ALL")
    result = [job.output for job in jobs]
    metadata = [job.metadata for job in jobs]
    assert result[0] == "F_memory_limit_exceeded"
    assert metadata[0]["memory"] > 1024**3
    evaluator.close()

    # With async function
    run_profiled = profile(
        memory=True,
        memory_limit=1024**3,
        memory_tracing_interval=0.2,
        register=False,  # !Required to make it work
    )(run_preprocessing_2_async)
    evaluator = Evaluator.create(
        run_profiled,
        method="serial",
    )
    tasks = [{"x": i} for i in range(1)]
    evaluator.submit(tasks)
    jobs = evaluator.gather("ALL")
    result = [job.output for job in jobs]
    metadata = [job.metadata for job in jobs]
    assert result[0] == "F_memory_limit_exceeded"
    assert metadata[0]["memory"] > 1024**3
    evaluator.close()
