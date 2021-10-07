Tests
*****

For automatic tests in DeepHyper we chose to use the `Pytest <https://docs.pytest.org/en/latest/index.html>`_ package.


Developer Installation
======================

From an activated Python virtual environment run the following commands:

.. code-block:: console

    $ git clone https://github.com/deephyper/deephyper.git
    $ cd deephyper/ && git checkout develop
    $ pip install -e ".[dev,analytics]"

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

.. code-block:: python

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

