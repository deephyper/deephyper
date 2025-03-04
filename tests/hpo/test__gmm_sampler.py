import pytest

import numpy as np
import pandas as pd

from deephyper.hpo._gmm_sampler import GMMSampler
from deephyper.hpo import HpProblem

from ConfigSpace import Configuration


def test_gmm_sampler_config_space_without_condition():
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

    assert gmm_sampler.categorical_cols == ["x_cat"]
    assert gmm_sampler.ordinal_cols == ["x_ord"]
    assert (
        len(gmm_sampler.numerical_cols) == 2
        and "x_real" in gmm_sampler.numerical_cols
        and "x_int" in gmm_sampler.numerical_cols
    )

    gmm_sampler.fit(df)

    df_new = gmm_sampler.sample(n_samples=100)
    for s in df_new.to_dict(orient="records"):
        conf = Configuration(config_space, s)

    assert "int" in str(df_new.dtypes["x_int"])
    assert "float" in str(df_new.dtypes["x_real"])


if __name__ == "__main__":
    test_gmm_sampler_config_space_without_condition()
