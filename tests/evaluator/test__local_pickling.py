from deephyper.evaluator import Evaluator


def test_loky_evaluator_with_lambda():
    # With loky it should work
    # Serialization by value
    evaluator = Evaluator.create(
        lambda j: j["x"] + 1, method="loky", method_kwargs=dict(num_workers=4)
    )
    evaluator.submit([{"x": i} for i in range(10)])
    results = evaluator.gather("ALL")

    assert len(results) == 10
    results = list(sorted(results, key=lambda j: j.id))
    for i, j in enumerate(results):
        assert i + 1 == j.output
