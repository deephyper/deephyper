import pytest

from deephyper.benchmark.problem import (SpaceDimNameMismatch,
                                         SpaceDimNameOfWrongType,
                                         SpaceNumDimMismatch,
                                         SpaceDimValueNotInSpace,
                                         SearchSpaceBuilderMissingParameter,
                                         SearchSpaceBuilderIsNotCallable,
                                         SearchSpaceBuilderMissingDefaultParameter,
                                         NaProblemError,
                                         WrongProblemObjective)


@pytest.mark.incremental
class TestProblem:
    def test_import(self):
        from deephyper.benchmark.problem import Problem

    def test_create(self):
        from deephyper.benchmark.problem import Problem
        pb = Problem()

    def test_add_dim(self):
        from deephyper.benchmark.problem import Problem
        pb = Problem()
        pb.add_dim(p_name='dim0', p_space=0)

    def test_space_attr(self):
        from deephyper.benchmark.problem import Problem
        pb = Problem()
        assert hasattr(pb, 'space')

    def test_dim0_exist_and_has_good_value(self):
        from deephyper.benchmark.problem import Problem
        pb = Problem()
        pb.add_dim(p_name='dim0', p_space=0)
        assert pb.space['dim0'] == 0

    def test_pos_args(self):
        from deephyper.benchmark.problem import Problem
        pb = Problem()
        pb.add_dim('dim0', 0)
        assert pb.space['dim0'] == 0


@pytest.mark.incremental
class TestHpProblem:
    def test_import(self):
        from deephyper.benchmark.problem import HpProblem

    def test_create(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()

    def test_add_good_dim(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        pb.add_dim('dim0', (-10, 10))

    def test_kwargs(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        pb.add_dim(p_name='dim0', p_space=(-10, 10))

    def test_dim_with_wrong_name(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        with pytest.raises(SpaceDimNameOfWrongType):
            pb.add_dim(0, (-10, 10))

    def test_add_good_reference(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        pb.add_dim(p_name='dim0', p_space=(-10, 10))
        pb.add_starting_point(dim0=0)

    def test_add_starting_points_with_too_many_dim(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        pb.add_dim(p_name='dim0', p_space=(-10, 10))
        with pytest.raises(SpaceNumDimMismatch):
            pb.add_starting_point(dim0=0, dim1=2)

    def test_add_starting_points_with_wrong_name(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        pb.add_dim(p_name='dim0', p_space=(-10, 10))
        with pytest.raises(SpaceDimNameMismatch):
            pb.add_starting_point(dim1=0)

    def test_add_starting_points_not_in_space_def(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        pb.add_dim(p_name='dim0', p_space=(-10, 10))
        pb.add_dim(p_name='dim1', p_space=(-10.0, 10.0))
        pb.add_dim(p_name='dim2', p_space=['a', 'b'])

        with pytest.raises(SpaceDimValueNotInSpace):
            pb.add_starting_point(dim0=-11, dim1=0.0, dim2='a')

        with pytest.raises(SpaceDimValueNotInSpace):
            pb.add_starting_point(dim0=11, dim1=0.0, dim2='a')

        with pytest.raises(SpaceDimValueNotInSpace):
            pb.add_starting_point(dim0=0, dim1=-11.0, dim2='a')

        with pytest.raises(SpaceDimValueNotInSpace):
            pb.add_starting_point(dim0=0, dim1=11.0, dim2='a')

        with pytest.raises(SpaceDimValueNotInSpace):
            pb.add_starting_point(dim0=0, dim1=0.0, dim2='c')

        pb.add_starting_point(dim0=0, dim1=0.0, dim2='a')


@pytest.mark.incremental
class TestNaProblem:
    def test_import(self):
        from deephyper.benchmark.problem import NaProblem

    def test_create(self):
        from deephyper.benchmark.problem import NaProblem

        NaProblem()

    def test_search_space(self):
        from deephyper.benchmark.problem import NaProblem

        pb = NaProblem()

        with pytest.raises(SearchSpaceBuilderIsNotCallable):
            pb.search_space(func='a')

        def dummy(a, b):
            return

        with pytest.raises(SearchSpaceBuilderMissingParameter):
            pb.search_space(func=dummy)

        def dummy(input_shape, output_shape):
            return

        with pytest.raises(SearchSpaceBuilderMissingDefaultParameter):
            pb.search_space(func=dummy)

        def dummy(input_shape=(1,), output_shape=(1,)):
            return

        pb.search_space(func=dummy)

    def test_full_problem(self):
        from deephyper.benchmark import NaProblem
        from deephyper.search.nas.model.preprocessing import minmaxstdscaler

        pb = NaProblem()

        def load_data(prop):
            return ([[10]], [1]), ([10], [1])

        pb.load_data(load_data, prop=1.)

        pb.preprocessing(minmaxstdscaler)

        def search_space(input_shape=(1,), output_shape=(1,)):
            return

        pb.search_space(search_space)

        pb.hyperparameters(
            batch_size=64,
            learning_rate=0.001,
            optimizer='adam',
            num_epochs=10,
            loss_metric='mse',
        )

        with pytest.raises(NaProblemError):
            pb.objective('r2')

        pb.loss('mse')
        pb.metrics(['r2'])

        possible_objective = ['loss', 'val_loss', 'r2', 'val_r2']
        for obj in possible_objective:
            pb.objective(obj)

        wrong_objective = ['mse', 'wrong', 'r2__last__max', 'val_mse']
        for obj in wrong_objective:
            with pytest.raises(WrongProblemObjective):
                pb.objective(obj)

        pb.post_training(
            num_epochs=2000,
            metrics=['mse', 'r2'],
            callbacks=dict(
                ModelCheckpoint={
                    'monitor': 'val_r2',
                    'mode': 'max',
                    'save_best_only': True,
                    'verbose': 1
                },
                EarlyStopping={
                    'monitor': 'val_r2',
                    'mode': 'max',
                    'verbose': 1,
                    'patience': 50
                })
        )
