import functools

import numpy as np
import pytest

from deephyper.evaluator import RunningJob
from deephyper.hpo import CBO, HpProblem


@pytest.mark.jax
def test_bayesian_learning_curve_regression_without_noise():
    from deephyper.stopper.lce import BayesianLearningCurveRegressor

    f_pow3 = BayesianLearningCurveRegressor.get_parametrics_model_func("pow3")
    rho = np.asarray([0.0, -0.1, 0.1])
    f_pow3_fixed = functools.partial(f_pow3, rho=rho)

    x = np.arange(1, 101)
    y = f_pow3_fixed(x)
    x_train, y_train = x[:50], y[:50]

    lce_model = BayesianLearningCurveRegressor(
        f_model=f_pow3,
        f_model_nparams=3,
    )
    for i in range(3):  # a few trials
        lce_model.fit(x_train, y_train)

        y_pred, y_pred_std = lce_model.predict(x)

        mse = np.mean((y_pred - y) ** 2)
        if mse <= 1e-4:
            break
    assert mse <= 1e-4


@pytest.mark.jax
def test_bayesian_learning_curve_regression_with_noise():
    from deephyper.stopper.lce import BayesianLearningCurveRegressor

    f_pow3 = BayesianLearningCurveRegressor.get_parametrics_model_func("pow3")
    rho = np.asarray([0.0, -0.1, 0.1])
    f_pow3_fixed = functools.partial(f_pow3, rho=rho)

    x = np.arange(1, 101)
    y_ = f_pow3_fixed(x)
    y = y_ + np.random.normal(0, scale=0.01, size=100)
    x_train, y_train = x[:50], y[:50]

    lce_model = BayesianLearningCurveRegressor(
        f_model=f_pow3,
        f_model_nparams=3,
    )
    for i in range(3):  # a few trials
        lce_model.fit(x_train, y_train)

        y_pred, y_pred_std = lce_model.predict(x)

        mean_se = np.mean((y_pred - y_) ** 2)
        mean_std = np.mean(y_pred_std)
        if mean_se <= 1e-4:
            break
    assert mean_se <= 1e-4 and (0.01 - mean_std) <= 0.01

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(x, y, label="True")
    # plt.plot(x, y_pred, label="Pred")
    # plt.fill_between(x, y_pred - y_pred_std, y_pred + y_pred_std, alpha=0.2)
    # plt.legend()
    # plt.show()


def optimization_test_bayesian_lce_model_speed():
    import cProfile
    import time
    from pstats import SortKey

    import matplotlib.pyplot as plt
    import numpy as np

    from deephyper.stopper.lce import BayesianLearningCurveRegressor

    f_pow3 = BayesianLearningCurveRegressor.get_parametrics_model_func("pow3")

    def f(z):
        return f_pow3(z, [1, -1, 0.125])

    z = np.arange(1, 1000)

    y = f(z)
    # y = y + rng.normal(0, 0.01, size=y.shape)

    t_start = time.time()
    with cProfile.Profile() as pr:
        model = BayesianLearningCurveRegressor(batch_size=100, verbose=0)
        for r in range(5):
            print(f"{r=}")
            for i in range(1, 20):
                print(f"{i=}")
                model.fit(z[:i], y[:i])
                y_pred, y_std = model.predict(z)
                y_min, y_max = y_pred - y_std, y_pred + y_std

        pr.print_stats(SortKey.TIME)

    t_end = time.time()
    duration = t_end - t_start

    print(f"duration: {duration:.3f} sec")

    plt.figure()
    plt.plot(z, y, label="f_pow3")
    plt.plot(z, y_pred, label="$\\hat{y}$", color="C2")
    plt.fill_between(z, y_min, y_max, color="C2", alpha=0.5)
    plt.legend()
    plt.show()


def run(job: RunningJob) -> dict:
    from deephyper.stopper import LCModelStopper

    assert isinstance(job.stopper, LCModelStopper)

    from deephyper.stopper.lce import BayesianLearningCurveRegressor

    f_pow3 = BayesianLearningCurveRegressor.get_parametrics_model_func("pow3")
    rho = np.asarray([job.parameters["rho_0"], -0.1, 0.1])
    f_pow3_fixed = functools.partial(f_pow3, rho=rho)

    max_budget = 50
    objective_i = 0

    for budget_i in range(1, max_budget + 1):
        objective_i = f_pow3_fixed(budget_i).tolist()

        job.record(budget_i, objective_i)
        if job.stopped():
            break

    return {
        "objective": job.objective,
        "metadata": {"budget": budget_i, "stopped": budget_i < max_budget},
    }


@pytest.mark.slow
@pytest.mark.jax
def test_lce_stopper(tmp_path):
    """This test can take up to 5 mins."""
    from deephyper.stopper import LCModelStopper

    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "rho_0")

    stopper = LCModelStopper(max_steps=50, min_steps=1, lc_model="pow3")

    search = CBO(
        problem,
        run,
        acq_optimizer="ga",
        acq_optimizer_freq=1,
        stopper=stopper,
        random_state=42,
        log_dir=tmp_path,
    )

    results = search.search(max_evals=30)

    assert "m:budget" in results.columns
    assert "m:stopped" in results.columns
    assert "p:rho_0" in results.columns
    assert "objective" in results.columns

    budgets = np.sort(np.unique(results["m:budget"].to_numpy())).tolist()
    assert len(budgets) > 1
    assert results["m:budget"].sum() < 50 * 30


if __name__ == "__main__":
    # test_bayesian_learning_curve_regression_with_noise()
    test_lce_stopper(tmp_path="/tmp/deephyper_test")
