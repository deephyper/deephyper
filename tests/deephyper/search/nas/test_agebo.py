import pytest
import numpy as np
from deephyper.benchmark.nas import linearReg
from deephyper.benchmark.nas import linearRegHybrid
from deephyper.evaluator import Evaluator
from deephyper.nas.run import run_debug_arch
from deephyper.search.nas import AgEBO


def test_agebo_without_hp():

    create_evaluator = lambda: Evaluator.create(run_debug_arch, method="serial")

    # ValueError: No hyperparameter space was defined for this problem
    with pytest.raises(ValueError):
        search = AgEBO(
            linearReg.Problem,
            create_evaluator(),
            random_state=42,
        )


def test_agebo_with_hp():

    create_evaluator = lambda: Evaluator.create(run_debug_arch, method="serial")

    search = AgEBO(
        linearRegHybrid.Problem,
        create_evaluator(),
        random_state=42,
    )

    res1 = search.search(max_evals=4)
    res1_array = res1[
        ["arch_seq", "batch_size", "learning_rate", "optimizer"]
    ].to_numpy()

    search = AgEBO(
        linearRegHybrid.Problem,
        create_evaluator(),
        random_state=42,
    )
    res2 = search.search(max_evals=4)
    res2_array = res2[
        ["arch_seq", "batch_size", "learning_rate", "optimizer"]
    ].to_numpy()

    assert np.array_equal(res1_array, res2_array)


if __name__ == "__main__":
    # test_agebo_without_hp()
    test_agebo_with_hp()
