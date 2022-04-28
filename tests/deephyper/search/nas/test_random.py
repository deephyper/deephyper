import unittest

import pytest


@pytest.mark.nas
class RandomTest(unittest.TestCase):
    def test_random_search(self):
        import numpy as np
        from deephyper.benchmark.nas import linearReg
        from deephyper.evaluator import Evaluator
        from deephyper.nas.run import run_debug_arch
        from deephyper.search.nas import Random

        create_evaluator = lambda: Evaluator.create(run_debug_arch, method="serial")

        search = Random(linearReg.Problem, create_evaluator(), random_state=42)

        res1 = search.search(max_evals=4)

        search = Random(linearReg.Problem, create_evaluator(), random_state=42)
        res2 = search.search(max_evals=4)

        assert np.array_equal(res1["arch_seq"].to_numpy(), res2["arch_seq"].to_numpy())

    def test_random_search_with_hp(self):
        import numpy as np
        from deephyper.benchmark.nas import linearRegHybrid
        from deephyper.evaluator import Evaluator
        from deephyper.nas.run import run_debug_arch
        from deephyper.search.nas import Random

        create_evaluator = lambda: Evaluator.create(run_debug_arch, method="serial")

        search = Random(linearRegHybrid.Problem, create_evaluator(), random_state=42)

        res1 = search.search(max_evals=4)
        res1_array = res1[
            ["arch_seq", "batch_size", "learning_rate", "optimizer"]
        ].to_numpy()

        search = Random(linearRegHybrid.Problem, create_evaluator(), random_state=42)
        res2 = search.search(max_evals=4)
        res2_array = res2[
            ["arch_seq", "batch_size", "learning_rate", "optimizer"]
        ].to_numpy()

        assert np.array_equal(res1_array, res2_array)
