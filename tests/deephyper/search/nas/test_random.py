import unittest

import pytest


@pytest.mark.slow
@pytest.mark.nas
class RandomTest(unittest.TestCase):
    def test_random_search(self):
        import numpy as np
        from deephyper.evaluator import Evaluator
        from deephyper.nas.run import run_debug_arch
        from deephyper.search.nas import Random

        import deephyper.test.nas.linearReg as linearReg

        create_evaluator = lambda: Evaluator.create(run_debug_arch, method="serial")

        search = Random(linearReg.Problem, create_evaluator(), random_state=42)

        res1 = search.search(max_evals=4)

        search = Random(linearReg.Problem, create_evaluator(), random_state=42)
        res2 = search.search(max_evals=4)

        assert np.array_equal(
            res1["p:arch_seq"].to_numpy(), res2["p:arch_seq"].to_numpy()
        )

    def test_random_search_with_hp(self):
        import numpy as np
        from deephyper.evaluator import Evaluator
        from deephyper.nas.run import run_debug_arch
        from deephyper.search.nas import Random

        import deephyper.test.nas.linearRegHybrid as linearRegHybrid

        create_evaluator = lambda: Evaluator.create(run_debug_arch, method="serial")

        search = Random(linearRegHybrid.Problem, create_evaluator(), random_state=42)

        res1 = search.search(max_evals=4)
        res1_array = res1[
            ["p:arch_seq", "p:batch_size", "p:learning_rate", "p:optimizer"]
        ].to_numpy()

        search = Random(linearRegHybrid.Problem, create_evaluator(), random_state=42)
        res2 = search.search(max_evals=4)
        res2_array = res2[
            ["p:arch_seq", "p:batch_size", "p:learning_rate", "p:optimizer"]
        ].to_numpy()

        assert np.array_equal(res1_array, res2_array)


if __name__ == "__main__":
    test = RandomTest()
    test.test_random_search()
