import unittest

import numpy as np
import jax.numpy as jnp


class SpaceTest(unittest.TestCase):
    def test(self):
        from deephyper.space._space import (
            IntDimension,
            RealDimension,
            CatDimension,
            ConstDimension,
            Space,
        )
        from deephyper.space._constraint import InequalityConstraint, BooleanConstraint

        # Real space
        space = Space("Real space")
        space.add_dimension(RealDimension("x", 0, 1))
        space.add_dimension(RealDimension("y", 0, 1))
        print(space)

        assert space.is_mixed is False
        assert space.default_value == [0, 0]

        values = space.sample(1000)
        freq_of_values = np.mean(np.asarray(values).sum(axis=1) < 1)

        # Real space with constraints
        space = Space("Real space with constraints")
        space.add_dimension(RealDimension("x", 0, 1))
        space.add_dimension(RealDimension("y", 0, 1))

        # Inequality constraints
        # "x + y <= 1"
        space.add_constraint(
            InequalityConstraint("x + y - 1", lambda p: p["x"] + p["y"] - 1)
        )
        print(space)

        values = space.sample(1000)
        freq_of_values = np.mean(np.asarray(values).sum(axis=1) < 1)
        assert freq_of_values > 0.75

        # Real space with constraints
        space = Space("Real space with constraints")
        space.add_dimension(RealDimension("x", 0, 1))
        space.add_dimension(RealDimension("y", 0, 1))

        # Inequality constraints
        # "x <= y"
        space.add_constraint(InequalityConstraint("x - y", lambda p: p["x"] - p["y"]))
        print(space)

        values = space.sample(1000)
        values = np.asarray(values)
        freq_of_values = np.mean(values[:, 0] < values[:, 1])
        assert freq_of_values > 0.75

        # Real space with constraints
        space = Space("Real space with strict constraints")
        space.add_dimension(RealDimension("x", 0, 1))
        space.add_dimension(RealDimension("y", 0, 1))

        # Inequality constraints
        # "x <= y"
        space.add_constraint(
            InequalityConstraint("x - y", lambda p: p["x"] - p["y"], is_strict=True)
        )
        print(space)

        values = space.sample(1000)
        values = np.asarray(values)
        assert len(values) > 750
        freq_of_values = np.mean(values[:, 0] < values[:, 1])
        assert freq_of_values == 1

        # Mixed space
        space = Space("Mixed space")
        space.add_dimension(IntDimension("x_int", 0, 10))
        space.add_dimension(RealDimension("x_real", -10.0, 0.0))
        space.add_dimension(CatDimension("x_cat", ["a", "b", "c"]))
        print(space)

        # Default value of the space
        assert space.default_value == [0, -10.0, "a"]

        # Sample from the space
        values = space.sample(10)

        # Discrete space
        space = Space("Discrete space")
        space.add_dimension(IntDimension("x", 0, 10))
        space.add_dimension(IntDimension("y", 0, 10))

        # Boolean constraint
        # "x != y"
        space.add_constraint(BooleanConstraint("x != y", lambda p: p["x"] != p["y"]))

        print(space)

        values = space.sample(1000)
        values = np.asarray(values)
        freq_of_values = np.mean(values[:, 0] != values[:, 1])
        assert freq_of_values > 0.75

        # Constant space
        space = Space("Constant space")
        # space.add_dimension(IntDimension("x_int", low=0, high=10))
        space.add_dimension(ConstDimension("x_const", 0))
        space.add_dimension(ConstDimension("y_const", 1.0))
        space.add_dimension(ConstDimension("z_const", "a"))
        print(space)
        assert space.default_value == [0, 1.0, "a"]

        values = space.sample(10)
        assert values == [[0, 1.0, "a"]] * 10

        # Space mixed with constant and other variables
        space = Space("Constant space with other variables")
        space.add_dimension(IntDimension("x_int", low=0, high=10))
        space.add_dimension(ConstDimension("x_const", 0))
        space.add_dimension(ConstDimension("y_const", 1.0))
        space.add_dimension(ConstDimension("z_const", "a"))
        print(space)
        assert space.default_value == [0, 0, 1.0, "a"]

        values = space.sample(10)
        values = np.asarray(values, dtype="O")
        assert values[:, 1:].tolist() == [[0, 1.0, "a"]] * 10
        assert all((0 <= values[:, 0]) & (values[:, 0] <= 10))


class MochiSpaceTest(unittest.TestCase):
    def test_mercury_spec(self):
        from deephyper.space._space import Space, ConstDimension, CatDimension

        # Default parameters from MercurySpec
        # https://github.com/mochi-hpc/mochi-bedrock/blob/main/python/mochi/bedrock/spec.py
        auto_sm = [True, False]
        na_no_block = [True, False]
        no_bulk_eager = [True, False]
        request_post_init = 256
        request_post_incr = 256
        input_eager_size = 4080
        output_eager_size = 4080
        na_max_expected_size = 0
        na_max_unexpected_size = 0

        space = Space("Mercury Space")
        space.add_dimension(CatDimension("auto_sm", auto_sm, default_value=True))
        space.add_dimension(
            CatDimension("na_no_block", na_no_block, default_value=False)
        )
        space.add_dimension(
            CatDimension("no_bulk_eager", no_bulk_eager, default_value=False)
        )
        space.add_dimension(ConstDimension("request_post_init", request_post_init))
        space.add_dimension(ConstDimension("request_post_incr", request_post_incr))
        space.add_dimension(ConstDimension("input_eager_size", input_eager_size))
        space.add_dimension(ConstDimension("output_eager_size", output_eager_size))
        space.add_dimension(
            ConstDimension("na_max_expected_size", na_max_expected_size)
        )
        space.add_dimension(
            ConstDimension("na_max_unexpected_size", na_max_unexpected_size)
        )
        assert space.default_value == [True, False, False, 256, 256, 4080, 4080, 0, 0]

        values = space.sample(num_samples=10)
        values = np.asarray(values, dtype="O")
        assert np.all(np.isin(values[:, :3], [True, False]))
        assert np.all(values[:, 3:] == [256, 256, 4080, 4080, 0, 0])

    def test_scheduler_spec(self):
        from deephyper.space._space import (
            Space,
            IntDimension,
        )
        from deephyper.space._constraint import BooleanConstraint
        import jax.numpy as jnp

        M = 1  # Maximum number of schedulers
        N = 10  # Maximum number of pools

        space = Space("")
        space.add_dimension(IntDimension("num_schedulers", low=1, high=M))
        space.add_dimension(IntDimension("num_pools", low=1, high=N))

        for scheduler_m in range(M):
            for pool_n in range(N):
                space.add_dimension(
                    # value 0 would be "inactive"
                    IntDimension(
                        f"scheduler_{scheduler_m}_connexion_{pool_n}", low=0, high=N
                    )
                )

                # Constraints
                # 1. scheduler_m_connexion_n == 0 if n > num_pools
                space.add_constraint(
                    BooleanConstraint(
                        "c1",
                        lambda p: jnp.where(
                            pool_n > p["num_pools"],
                            p[f"scheduler_{scheduler_m}_connexion_{pool_n}"] == 0,
                            p[f"scheduler_{scheduler_m}_connexion_{pool_n}"] > 0,
                        ),
                        # lambda p: (
                        #     jnp.logical_or(
                        #         jnp.logical_and(
                        #             p[f"scheduler_{scheduler_m}_connexion_{pool_n}"]
                        #             == 0,
                        #             pool_n > p["num_pools"],
                        #         ),
                        #         jnp.logical_and(
                        #             p[f"scheduler_{scheduler_m}_connexion_{pool_n}"]
                        #             > 0,
                        #             pool_n <= p["num_pools"],
                        #         ),
                        #     )
                        # ),
                    )
                )

        # print(space)

        # values = space.sample(num_samples=10)
        # print(np.asarray(values, dtype="O"))


class HyperparameterOptimizationTest(unittest.TestCase):
    def test_variable_layers_with_decreasing_sizes_boolean_constraints(self):
        import numpyro.distributions as dist
        from deephyper.space._space import (
            Space,
            IntDimension,
        )
        from deephyper.space._constraint import BooleanConstraint

        max_num_layers = 10
        # space = Space("", seed=42)
        space = Space("")
        space.add_dimension(IntDimension("n", low=1, high=max_num_layers))
        for i in range(1, max_num_layers + 1):
            space.add_dimension(
                IntDimension(f"l{i}", low=0, high=100, default_value=32)
            )

        for i in range(1, max_num_layers + 1):
            space.add_constraint(
                BooleanConstraint(
                    f"l{i}_is_active | (n >= {i} and l{i} > 0) or (n < {i} and l{i} == 0)",
                    lambda p, i=i: (
                        ((p["n"] >= i) & (p[f"l{i}"] > 0))
                        | ((p["n"] < i) & (p[f"l{i}"] == 0))
                    ),
                    strength=10,
                    is_strict=True,
                )
            )

        for i in range(1, max_num_layers):
            space.add_constraint(
                BooleanConstraint(
                    f"(l{i} >= l{i+1}) or (l{i+1} == 0)",
                    lambda p, i=i: ((p[f"l{i}"] >= p[f"l{i+1}"]) | (p[f"l{i+1}"] == 0)),
                    strength=10,
                    is_strict=True,
                )
            )

        print(space)

        values = space.sample(num_samples=1_000)
        assert len(values) > 0

        print(jnp.asarray(values)[:10])

        print(set(jnp.asarray(values)[:, 0].tolist()))

    def test_int_parameters_sampled_as_continuous(self):
        import numpyro.distributions as dist
        from deephyper.space import (
            Space,
            IntDimension,
        )
        from deephyper.space._constraint import (
            EqualityConstraint,
            InequalityConstraint,
            BooleanConstraint,
        )

        max_num_layers = 5
        # space = Space("", seed=42)
        space = Space("")
        space.add_dimension(IntDimension("n", low=1, high=max_num_layers))
        for i in range(1, max_num_layers + 1):
            dim = IntDimension(f"l{i}", low=0, high=100, default_value=32)
            # dim.distribution = dist.Uniform(0, 100)
            space.add_dimension(dim)

        # for i in range(1, max_num_layers + 1):
        #     space.add_constraint(
        #         BooleanConstraint(
        #             f"l{i}_is_active | (n >= {i} and l{i} > 0) or (n < {i} and l{i} == 0)",
        #             lambda p, i=i: (
        #                 ((p["n"] >= i) & (p[f"l{i}"] > 0))
        #                 | ((p["n"] < i) & (p[f"l{i}"] == 0))
        #             ),
        #             strength=10,
        #             is_strict=True,
        #         )
        #     )

        # f(x) <= 0
        for j in range(1, max_num_layers):
            space.add_constraint(
                InequalityConstraint(
                    f"l{j+1} - l{j}",
                    lambda p, i=j: p[f"l{i+1}"] - p[f"l{i}"],
                    strength=1,
                    is_strict=True,
                )
            )

        print(space)

        values = space.sample(num_samples=1_000)
        assert len(values) > 0

        print(jnp.asarray(values)[:10])

        print(set(jnp.asarray(values)[:, 0].tolist()))


if __name__ == "__main__":
    # SpaceTest().test()
    # MochiSpaceTest().test_mercury_spec()
    # MochiSpaceTest().test_scheduler_spec()
    # HyperparameterOptimizationTest().test_variable_layers_with_decreasing_sizes_boolean_constraints()
    HyperparameterOptimizationTest().test_int_parameters_sampled_as_continuous()
    # test_conditions()
