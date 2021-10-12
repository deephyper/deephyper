import pytest
from deephyper.core.exceptions.problem import (
    SpaceDimNameOfWrongType,
    SearchSpaceBuilderMissingParameter,
    SearchSpaceBuilderIsNotCallable,
    SearchSpaceBuilderMissingDefaultParameter,
    NaProblemError,
    WrongProblemObjective,
)


@pytest.mark.incremental
class TestHpProblem:
    def test_import(self):
        from deephyper.problem import HpProblem

    def test_create(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()

    def test_add_good_dim(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()
        pb.add_hyperparameter((-10, 10), "dim0")
        pb.add_hyperparameter((-10.0, 10.0), "dim1")
        pb.add_hyperparameter([1, 2, 3, 4], "dim2")
        pb.add_hyperparameter(["cat0", 1, "cat2", 2.0], "dim3")

    def test_kwargs(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()
        pb.add_hyperparameter(value=(-10, 10), name="dim0")

    def test_dim_with_wrong_name(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()
        with pytest.raises(SpaceDimNameOfWrongType):
            pb.add_hyperparameter((-10, 10), 0)

    def test_add_good_reference(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()
        pb.add_hyperparameter((-10, 10), "dim0")
        pb.add_starting_point(dim0=0)

    def test_add_starting_points_with_too_many_dim(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()
        pb.add_hyperparameter((-10, 10), "dim0")
        with pytest.raises(ValueError):
            pb.add_starting_point(dim0=0, dim1=2)

    def test_add_starting_points_with_wrong_name(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()
        pb.add_hyperparameter((-10, 10), "dim0")
        with pytest.raises(ValueError):
            pb.add_starting_point(dim1=0)

    def test_add_starting_points_not_in_space_def(self):
        from deephyper.problem import HpProblem

        pb = HpProblem()
        pb.add_hyperparameter((-10, 10), "dim0")
        pb.add_hyperparameter((-10.0, 10.0), "dim1")
        pb.add_hyperparameter(["a", "b"], "dim2")

        with pytest.raises(ValueError):
            pb.add_starting_point(dim0=-11, dim1=0.0, dim2="a")

        with pytest.raises(ValueError):
            pb.add_starting_point(dim0=11, dim1=0.0, dim2="a")

        with pytest.raises(ValueError):
            pb.add_starting_point(dim0=0, dim1=-11.0, dim2="a")

        with pytest.raises(ValueError):
            pb.add_starting_point(dim0=0, dim1=11.0, dim2="a")

        with pytest.raises(ValueError):
            pb.add_starting_point(dim0=0, dim1=0.0, dim2="c")

        pb.add_starting_point(dim0=0, dim1=0.0, dim2="a")

    def test_config_space_hp(self):
        from deephyper.problem import HpProblem
        import ConfigSpace.hyperparameters as csh

        alpha = csh.UniformFloatHyperparameter(name="alpha", lower=0, upper=1)
        beta = csh.UniformFloatHyperparameter(name="beta", lower=0, upper=1)

        pb = HpProblem()
        pb.add_hyperparameters([alpha, beta])


@pytest.mark.incremental
class TestNaProblem:
    def test_import(self):
        from deephyper.problem import NaProblem

    def test_create(self):
        from deephyper.problem import NaProblem

        NaProblem()

    def test_search_space(self):
        from deephyper.problem import NaProblem

        pb = NaProblem()

        with pytest.raises(SearchSpaceBuilderIsNotCallable):
            pb.search_space(func="a")

        def dummy(a, b):
            return

        with pytest.raises(SearchSpaceBuilderMissingParameter):
            pb.search_space(func=dummy)

        def dummy(input_shape=(1,), output_shape=(1,)):
            return

        pb.search_space(func=dummy)

    def test_full_problem(self):
        from deephyper.problem import NaProblem
        from deephyper.nas.preprocessing import minmaxstdscaler

        pb = NaProblem()

        def load_data(prop):
            return ([[10]], [1]), ([10], [1])

        pb.load_data(load_data, prop=1.0)

        pb.preprocessing(minmaxstdscaler)

        def search_space(input_shape=(1,), output_shape=(1,)):
            return

        pb.search_space(search_space)

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
