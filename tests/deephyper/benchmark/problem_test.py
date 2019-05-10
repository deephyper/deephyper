import pytest

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
        pb.add_dim('dim0', (-10, 10), 0)

    def test_kwargs(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        pb.add_dim(p_name='dim0', p_space=(-10, 10), default=0)

    def test_dim_with_wrong_name(self):
        from deephyper.benchmark.problem import HpProblem
        pb = HpProblem()
        with pytest.raises(AssertionError):
            pb.add_dim(0, (-10, 10), 0)
