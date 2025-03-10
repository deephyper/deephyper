import pandas as pd
from ConfigSpace import Configuration

from deephyper.hpo import HpProblem
from deephyper.hpo.gmm import GMMSampler

# TODO: add test with 1 type of variable at a time
# TODO: add test to sample from config space with conditions
# TODO: add test to sample from config space with forbiddens


def test_gmm_sampler_config_space_categorical_without_condition():
    problem = HpProblem()
    problem.add_hyperparameter([chr(97 + i) for i in range(26)], "x_cat")

    config_space = problem.space
    samples = config_space.sample_configuration(size=100)
    samples = [dict(s) for s in samples]

    df = pd.DataFrame(samples)

    gmm_sampler = GMMSampler(config_space)
    gmm_sampler.fit(df)

    assert gmm_sampler.categorical_cols == ["x_cat"]
    assert gmm_sampler.ordinal_cols == []
    assert gmm_sampler.numerical_cols == []

    df_new = gmm_sampler.sample(n_samples=10)
    for s in df_new.to_dict(orient="records"):
        Configuration(config_space, s)


def test_gmm_sampler_config_space_ordinal_without_condition():
    problem = HpProblem()
    problem.add_hyperparameter([i for i in range(10)], "x_ord")

    config_space = problem.space
    samples = config_space.sample_configuration(size=100)
    samples = [dict(s) for s in samples]

    df = pd.DataFrame(samples)

    gmm_sampler = GMMSampler(config_space)
    gmm_sampler.fit(df)

    assert gmm_sampler.categorical_cols == []
    assert gmm_sampler.ordinal_cols == ["x_ord"]
    assert gmm_sampler.numerical_cols == []

    df_new = gmm_sampler.sample(n_samples=10)
    for s in df_new.to_dict(orient="records"):
        Configuration(config_space, s)


def test_gmm_sampler_config_space_integer_without_condition():
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")

    config_space = problem.space
    samples = config_space.sample_configuration(size=100)
    samples = [dict(s) for s in samples]

    df = pd.DataFrame(samples)

    gmm_sampler = GMMSampler(config_space)
    gmm_sampler.fit(df)

    assert gmm_sampler.categorical_cols == []
    assert gmm_sampler.ordinal_cols == []
    assert gmm_sampler.numerical_cols == ["x_int"]

    df_new = gmm_sampler.sample(n_samples=10)
    for s in df_new.to_dict(orient="records"):
        Configuration(config_space, s)


def test_gmm_sampler_config_space_real_without_condition():
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x_real")

    config_space = problem.space
    samples = config_space.sample_configuration(size=100)
    samples = [dict(s) for s in samples]

    df = pd.DataFrame(samples)

    gmm_sampler = GMMSampler(config_space)
    gmm_sampler.fit(df)

    assert gmm_sampler.categorical_cols == []
    assert gmm_sampler.ordinal_cols == []
    assert gmm_sampler.numerical_cols == ["x_real"]

    df_new = gmm_sampler.sample(n_samples=10)
    for s in df_new.to_dict(orient="records"):
        Configuration(config_space, s)


def test_gmm_sampler_config_space_mixed_without_condition():
    problem = HpProblem()
    problem.add_hyperparameter([chr(97 + i) for i in range(26)], "x_cat")
    problem.add_hyperparameter([i for i in range(10)], "x_ord")
    problem.add_hyperparameter((0.0, 10.0), "x_real")
    problem.add_hyperparameter((0, 10), "x_int")

    config_space = problem.space
    samples = config_space.sample_configuration(size=100)
    samples = [dict(s) for s in samples]

    df = pd.DataFrame(samples)

    gmm_sampler = GMMSampler(config_space)
    gmm_sampler.fit(df)

    assert gmm_sampler.categorical_cols == ["x_cat"]
    assert gmm_sampler.ordinal_cols == ["x_ord"]
    assert (
        len(gmm_sampler.numerical_cols) == 2
        and "x_real" in gmm_sampler.numerical_cols
        and "x_int" in gmm_sampler.numerical_cols
    )

    df_new = gmm_sampler.sample(n_samples=10)
    for s in df_new.to_dict(orient="records"):
        Configuration(config_space, s)

    assert "int" in str(df_new.dtypes["x_int"])
    assert "float" in str(df_new.dtypes["x_real"])


if __name__ == "__main__":
    test_gmm_sampler_config_space_mixed_without_condition()
