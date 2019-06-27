Tests
*****

For automatic tests in deephyper we choosed to use the pytest framework: `pytest official website <https://docs.pytest.org/en/latest/index.html>`_.


Install
=======

::

    pip install -e '.[tests,docs]'

Run Tests
=========

This is the basic and simplest command line to run test.
All test marked as ``@pytest.mark.slow`` will be skipped::

    cd deephyper/tests/
    pytest

If you want to run tests marked as ``@pytest.mark.slow``::

    pytest --runslow

You can also run doctest tests::

    cd deephyper/docs/
    make doctest

Incremental Tests
=================

Sometimes you may have a testing situation which consists of a series of
test steps. If one step fails it makes no sense to execute further steps
as they are all expected to fail anyway and their tracebacks add no insight.
You can use the ``@pytest.mark.incremental`` decorator such as:

::

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

