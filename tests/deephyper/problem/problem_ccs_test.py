import sys
import unittest

import pytest

try:
    import cconfigspace as ccs
except ModuleNotFoundError:
    pass


@pytest.mark.skipif(
    "cconfigspace" not in sys.modules, reason="requires the CConfigSpace library"
)
class HpProblemTest(unittest.TestCase):
    def test_add_good_dim(self):
        from deephyper.problem._hyperparameter_ccs import HpProblem

        pb = HpProblem()

        # integer hyperparameter
        p0 = pb.add_hyperparameter((1, 10), "p0")
        p0_ccs = ccs.NumericalHyperparameter.int(name="p0", lower=1, upper=10)
        assert p0 == p0_ccs

        p0 = pb.add_hyperparameter((1, 10, "log-uniform"), "p0_log")
        p0_ccs = ccs.NumericalHyperparameter.int(name="p0_log", lower=1, upper=10)
        p0_ccs_dist = ccs.UniformDistribution.int(
            lower=1, upper=10, scale=ccs.ccs_scale_type.LOGARITHMIC
        )
        assert p0 == p0_ccs
        assert p0_ccs_dist == p0_ccs_dist

        # float hyperparameter
        p1 = pb.add_hyperparameter((1.0, 10.0), "p1")
        p1_ccs = ccs.NumericalHyperparameter.float(name="p1", lower=1, upper=10)
        assert p1 == p1_ccs

        p1 = pb.add_hyperparameter((1.0, 10.0, "log-uniform"), "p1_log")
        p1_ccs = ccs.NumericalHyperparameter.float(name="p1_log", lower=1, upper=10)
        p1_ccs_dist = ccs.UniformDistribution.float(
            lower=1, upper=10, scale=ccs.ccs_scale_type.LOGARITHMIC
        )
        assert p1 == p1_ccs
        assert p1_ccs_dist == p1_ccs_dist

        # ordinal hyperparameter
        p4 = pb.add_hyperparameter([1, 2, 3, 4], "p4")
        p4_ccs = ccs.OrdinalHyperparameter(name="p4", values=[1, 2, 3, 4])
        assert p4 == p4_ccs

        p5 = pb.add_hyperparameter([1.0, 2.0, 3.0, 4.0], "p5")
        p5_ccs = ccs.OrdinalHyperparameter(name="p5", values=[1.0, 2.0, 3.0, 4.0])
        assert p5 == p5_ccs

        p6 = pb.add_hyperparameter(["cat0", "cat1"], "p6")
        p6_ccs = ccs.CategoricalHyperparameter(name="p6", values=["cat0", "cat1"])
        assert p6 == p6_ccs

        # p7 = pb.add_hyperparameter({"mu": 0, "sigma": 1}, "p7")
        # p7_csh = csh.NormalIntegerHyperparameter(name="p7", mu=0, sigma=1)
        # assert p7 == p7_csh

        # if cs.__version__ > "0.4.20":
        #     p8 = pb.add_hyperparameter(
        #         {"mu": 0, "sigma": 1, "lower": -5, "upper": 5}, "p8"
        #     )
        #     p8_csh = csh.NormalIntegerHyperparameter(
        #         name="p8", mu=0, sigma=1, lower=-5, upper=5
        #     )
        #     assert p8 == p8_csh

        # p9 = pb.add_hyperparameter({"mu": 0.0, "sigma": 1.0}, "p9")
        # p9_csh = csh.NormalFloatHyperparameter(name="p9", mu=0, sigma=1)
        # assert p9 == p9_csh

    def test_kwargs(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()
        pb.add_hyperparameter(value=(-10, 10), name="dim0")

    # def test_dim_with_wrong_name(self):
    #     from deephyper.problem import HpProblem

    #     pb = HpProblem()
    #     with pytest.raises(SpaceDimNameOfWrongType):
    #         pb.add_hyperparameter((-10, 10), 0)

    # def test_add_good_reference(self):
    #     from deephyper.problem import HpProblem

    #     pb = HpProblem()
    #     pb.add_hyperparameter((-10, 10), "dim0")
    #     pb.add_starting_point(dim0=0)

    # def test_add_starting_points_with_too_many_dim(self):
    #     from deephyper.problem import HpProblem

    #     pb = HpProblem()
    #     pb.add_hyperparameter((-10, 10), "dim0")
    #     with pytest.raises(ValueError):
    #         pb.add_starting_point(dim0=0, dim1=2)

    # def test_add_starting_points_with_wrong_name(self):
    #     from deephyper.problem import HpProblem

    #     pb = HpProblem()
    #     pb.add_hyperparameter((-10, 10), "dim0")
    #     with pytest.raises(ValueError):
    #         pb.add_starting_point(dim1=0)

    # def test_add_starting_points_not_in_space_def(self):
    #     from deephyper.problem import HpProblem

    #     pb = HpProblem()
    #     pb.add_hyperparameter((-10, 10), "dim0")
    #     pb.add_hyperparameter((-10.0, 10.0), "dim1")
    #     pb.add_hyperparameter(["a", "b"], "dim2")

    #     with pytest.raises(ValueError):
    #         pb.add_starting_point(dim0=-11, dim1=0.0, dim2="a")

    #     with pytest.raises(ValueError):
    #         pb.add_starting_point(dim0=11, dim1=0.0, dim2="a")

    #     with pytest.raises(ValueError):
    #         pb.add_starting_point(dim0=0, dim1=-11.0, dim2="a")

    #     with pytest.raises(ValueError):
    #         pb.add_starting_point(dim0=0, dim1=11.0, dim2="a")

    #     with pytest.raises(ValueError):
    #         pb.add_starting_point(dim0=0, dim1=0.0, dim2="c")

    #     pb.add_starting_point(dim0=0, dim1=0.0, dim2="a")

    # def test_config_space_hp(self):
    #     import ConfigSpace.hyperparameters as csh
    #     from deephyper.problem import HpProblem

    #     alpha = csh.UniformFloatHyperparameter(name="alpha", lower=0, upper=1)
    #     beta = csh.UniformFloatHyperparameter(name="beta", lower=0, upper=1)

    #     pb = HpProblem()
    #     pb.add_hyperparameters([alpha, beta])
