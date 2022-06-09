import unittest

import pytest


@pytest.mark.hps_fast_test
class HpProblemTest(unittest.TestCase):
    def test_add_good_dim(self):
        import ConfigSpace as cs
        import ConfigSpace.hyperparameters as csh
        from deephyper.problem import HpProblem

        pb = HpProblem()

        p0 = pb.add_hyperparameter((-10, 10), "p0")
        p0_csh = csh.UniformIntegerHyperparameter(
            name="p0", lower=-10, upper=10, log=False
        )
        assert p0 == p0_csh

        p1 = pb.add_hyperparameter((1, 100, "log-uniform"), "p1")
        p1_csh = csh.UniformIntegerHyperparameter(
            name="p1", lower=1, upper=100, log=True
        )
        assert p1 == p1_csh

        p2 = pb.add_hyperparameter((-10.0, 10.0), "p2")
        p2_csh = csh.UniformFloatHyperparameter(
            name="p2", lower=-10.0, upper=10.0, log=False
        )
        assert p2 == p2_csh

        p3 = pb.add_hyperparameter((1.0, 100.0, "log-uniform"), "p3")
        p3_csh = csh.UniformFloatHyperparameter(
            name="p3", lower=1.0, upper=100.0, log=True
        )
        assert p3 == p3_csh

        p4 = pb.add_hyperparameter([1, 2, 3, 4], "p4")
        p4_csh = csh.OrdinalHyperparameter(name="p4", sequence=[1, 2, 3, 4])
        assert p4 == p4_csh

        p5 = pb.add_hyperparameter([1.0, 2.0, 3.0, 4.0], "p5")
        p5_csh = csh.OrdinalHyperparameter(name="p5", sequence=[1.0, 2.0, 3.0, 4.0])
        assert p5 == p5_csh

        p6 = pb.add_hyperparameter(["cat0", "cat1"], "p6")
        p6_csh = csh.CategoricalHyperparameter(name="p6", choices=["cat0", "cat1"])
        assert p6 == p6_csh

        p7 = pb.add_hyperparameter({"mu": 0, "sigma": 1}, "p7")
        p7_csh = csh.NormalIntegerHyperparameter(name="p7", mu=0, sigma=1)
        assert p7 == p7_csh

        if cs.__version__ > "0.4.20":
            p8 = pb.add_hyperparameter(
                {"mu": 0, "sigma": 1, "lower": -5, "upper": 5}, "p8"
            )
            p8_csh = csh.NormalIntegerHyperparameter(
                name="p8", mu=0, sigma=1, lower=-5, upper=5
            )
            assert p8 == p8_csh

        p9 = pb.add_hyperparameter({"mu": 0.0, "sigma": 1.0}, "p9")
        p9_csh = csh.NormalFloatHyperparameter(name="p9", mu=0, sigma=1)
        assert p9 == p9_csh

    def test_kwargs(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()
        pb.add_hyperparameter(value=(-10, 10), name="dim0")

    def test_dim_with_wrong_name(self):
        from deephyper.core.exceptions.problem import SpaceDimNameOfWrongType
        from deephyper.problem import HpProblem

        pb = HpProblem()
        with pytest.raises(SpaceDimNameOfWrongType):
            pb.add_hyperparameter((-10, 10), 0)

    def test_config_space_hp(self):
        import ConfigSpace.hyperparameters as csh
        from deephyper.problem import HpProblem

        alpha = csh.UniformFloatHyperparameter(name="alpha", lower=0, upper=1)
        beta = csh.UniformFloatHyperparameter(name="beta", lower=0, upper=1)

        pb = HpProblem()
        pb.add_hyperparameters([alpha, beta])


@pytest.mark.nas
class TestNaProblem(unittest.TestCase):
    def test_search_space(self):

        from deephyper.nas.spacelib.tabular import OneLayerSpace
        from deephyper.problem import NaProblem

        pb = NaProblem()

        with pytest.raises(TypeError):
            pb.search_space(space_class="a")

        pb.search_space(OneLayerSpace)

    def test_full_problem(self):
        from deephyper.core.exceptions.problem import NaProblemError
        from deephyper.nas.preprocessing import minmaxstdscaler
        from deephyper.nas.spacelib.tabular import OneLayerSpace
        from deephyper.problem import NaProblem

        pb = NaProblem()

        def load_data(prop):
            return ([[10]], [1]), ([10], [1])

        pb.load_data(load_data, prop=1.0)

        pb.preprocessing(minmaxstdscaler)

        pb.search_space(OneLayerSpace)

        pb.hyperparameters(
            batch_size=64,
            learning_rate=0.001,
            optimizer="adam",
            num_epochs=10,
            loss_metric="mse",
        )

        with pytest.raises(NaProblemError):
            pb.objective("r2")

        pb.loss("mse")
        pb.metrics(["r2"])

        possible_objective = ["loss", "val_loss", "r2", "val_r2"]
        for obj in possible_objective:
            pb.objective(obj)
