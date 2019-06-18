import pytest

from deephyper.benchmark.problem import (SpaceDimNameMismatch,
                                         SpaceDimNameOfWrongType,
                                         SpaceNumDimMismatch,
                                         SpaceDimValueNotInSpace)


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
