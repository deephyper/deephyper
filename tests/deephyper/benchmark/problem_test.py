import pytest

from deephyper.benchmark.problem import (SpaceDimNameMismatch,
                                         SpaceDimNameOfWrongType,
                                         SpaceNumDimMismatch)


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
        pb.add_reference(dim0=0)

    def test_add_references_with_too_many_dim(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        pb.add_dim(p_name='dim0', p_space=(-10, 10))
        with pytest.raises(SpaceNumDimMismatch):
            pb.add_reference(dim0=0, dim1=2)

    def test_add_references_with_wrong_name(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        pb.add_dim(p_name='dim0', p_space=(-10, 10))
        with pytest.raises(SpaceDimNameMismatch):
            pb.add_reference(dim1=0)
