import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import pytest

from deephyper.hpo import HpProblem


def test_add_good_dim():
    pb = HpProblem()

    p0 = pb.add_hyperparameter((-10, 10), "p0")
    p0_csh = csh.UniformIntegerHyperparameter(name="p0", lower=-10, upper=10, log=False)
    assert p0 == p0_csh

    pb.add((-10, 10), "p00")
    assert pb["p00"] == csh.UniformIntegerHyperparameter(name="p00", lower=-10, upper=10, log=False)

    p1 = pb.add_hyperparameter((1, 100, "log-uniform"), "p1")
    p1_csh = csh.UniformIntegerHyperparameter(name="p1", lower=1, upper=100, log=True)
    assert p1 == p1_csh

    pb.add((1, 100, "log-uniform"), "p11")
    assert pb["p11"] == csh.UniformIntegerHyperparameter(name="p11", lower=1, upper=100, log=True)

    p2 = pb.add_hyperparameter((-10.0, 10.0), "p2")
    p2_csh = csh.UniformFloatHyperparameter(name="p2", lower=-10.0, upper=10.0, log=False)
    assert p2 == p2_csh

    pb.add((-10.0, 10.0), "p22")
    assert pb["p22"] == csh.UniformFloatHyperparameter(
        name="p22", lower=-10.0, upper=10.0, log=False
    )

    p3 = pb.add_hyperparameter((1.0, 100.0, "log-uniform"), "p3")
    p3_csh = csh.UniformFloatHyperparameter(name="p3", lower=1.0, upper=100.0, log=True)
    assert p3 == p3_csh

    pb.add((1.0, 100.0, "log-uniform"), "p33")
    assert pb["p33"] == csh.UniformFloatHyperparameter(name="p33", lower=1.0, upper=100.0, log=True)

    p4 = pb.add_hyperparameter([1, 2, 3, 4], "p4")
    p4_csh = csh.OrdinalHyperparameter(name="p4", sequence=[1, 2, 3, 4])
    assert p4 == p4_csh

    pb.add([1, 2, 3, 4], "p44")
    assert pb["p44"] == csh.OrdinalHyperparameter(name="p44", sequence=[1, 2, 3, 4])

    p5 = pb.add_hyperparameter([1.0, 2.0, 3.0, 4.0], "p5")
    p5_csh = csh.OrdinalHyperparameter(name="p5", sequence=[1.0, 2.0, 3.0, 4.0])
    assert p5 == p5_csh

    pb.add([1.0, 2.0, 3.0, 4.0], "p55")
    assert pb["p55"] == csh.OrdinalHyperparameter(name="p55", sequence=[1.0, 2.0, 3.0, 4.0])

    p6 = pb.add_hyperparameter(["cat0", "cat1"], "p6")
    p6_csh = csh.CategoricalHyperparameter(name="p6", choices=["cat0", "cat1"])
    assert p6 == p6_csh

    pb.add(["cat0", "cat1"], "p66")
    assert pb["p66"] == csh.CategoricalHyperparameter(name="p66", choices=["cat0", "cat1"])

    p7 = pb.add_hyperparameter({"mu": 0, "sigma": 1, "lower": -10, "upper": 10}, "p7")
    p7_csh = csh.NormalIntegerHyperparameter(name="p7", mu=0, sigma=1, lower=-10, upper=10)
    assert p7 == p7_csh

    pb.add({"mu": 0, "sigma": 1, "lower": -10, "upper": 10}, "p77")
    assert pb["p77"] == csh.NormalIntegerHyperparameter(
        name="p77", mu=0, sigma=1, lower=-10, upper=10
    )

    p8 = pb.add_hyperparameter({"mu": 0, "sigma": 1, "lower": -5, "upper": 5}, "p8")
    p8_csh = csh.NormalIntegerHyperparameter(name="p8", mu=0, sigma=1, lower=-5, upper=5)
    assert p8 == p8_csh

    pb.add({"mu": 0, "sigma": 1, "lower": -5, "upper": 5}, "p88")
    assert pb["p88"] == csh.NormalIntegerHyperparameter(
        name="p88", mu=0, sigma=1, lower=-5, upper=5
    )

    p9 = pb.add_hyperparameter({"mu": 0.0, "sigma": 1.0, "lower": -10.0, "upper": 10.0}, "p9")
    p9_csh = csh.NormalFloatHyperparameter(name="p9", mu=0, sigma=1, lower=-10, upper=10)
    assert p9 == p9_csh

    pb.add({"mu": 0.0, "sigma": 1.0, "lower": -10.0, "upper": 10.0}, "p99")
    assert pb["p99"] == csh.NormalFloatHyperparameter(
        name="p99", mu=0, sigma=1, lower=-10, upper=10
    )


def test_kwargs():
    from deephyper.hpo import HpProblem

    pb = HpProblem()
    pb.add_hyperparameter(value=(-10, 10), name="dim0")

    pb.add(value=(-10, 10), name="dim1")


def test_dim_with_wrong_name():
    pb = HpProblem()
    with pytest.raises(TypeError):
        pb.add_hyperparameter((-10, 10), 0)

    with pytest.raises(TypeError):
        pb.add((-10, 10), 0)


def test_config_space_hp():
    alpha = csh.UniformFloatHyperparameter(name="alpha", lower=0, upper=1)
    beta = csh.UniformFloatHyperparameter(name="beta", lower=0, upper=1)

    pb = HpProblem()
    pb.add_hyperparameters([alpha, beta])

    pb = HpProblem()
    pb.add(alpha)
    pb.add(beta)

    pb = HpProblem()
    pb.add([alpha, beta])


def test_forbidden():
    n_layers_total = 32
    n_layers_pruned = 3
    pb = HpProblem()

    for i in range(n_layers_pruned):
        pb.add(
            (i, n_layers_total - n_layers_pruned + i),
            f"l{i}",
            default_value=i,
        )
        if i > 0:
            # Enforce li < li+1
            fb = cs.ForbiddenGreaterThanRelation(pb[f"l{i - 1}"], pb[f"l{i}"])
            pb.add(fb)
            fb = cs.ForbiddenEqualsRelation(pb[f"l{i - 1}"], pb[f"l{i}"])
            pb.add(fb)

    assert pb.hyperparameter_names == ["l0", "l1", "l2"]

    for s in pb.sample(10):
        assert s["l0"] < s["l1"] and s["l1"] < s["l2"]
