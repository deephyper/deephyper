import functools

import numpy as np
import pytest

from deephyper.evaluator import RunningJob
from deephyper.hpo import CBO, HpProblem
from deephyper.stopper import LCModelStopper


def test_bayesian_learning_curve_regression():
    import matplotlib.pyplot as plt

    from deephyper.stopper.lce import BayesianLearningCurveRegressor

    f_pow3 = BayesianLearningCurveRegressor.get_parametrics_model_func("pow3")
    rho = np.asarray([0.0, -0.1, 0.1])
    f_pow3_fixed = functools.partial(f_pow3, rho=rho)

    x = np.arange(1, 100)
    y = f_pow3_fixed(x)
    x_train, y_train = x[:50], y[:50]
    x_test, y_test = x[50:], y[50:]

    lce_model = BayesianLearningCurveRegressor(
        f_model=f_pow3,
        f_model_nparams=3,
    )
    lce_model.fit(x_train, y_train)

    y_pred, y_pred_std = lce_model.predict(x)

    plt.figure()
    plt.plot(x, y, label="True")
    plt.plot(x, y_pred, label="Pred")
    plt.fill_between(x, y_pred - y_pred_std, y_pred + y_pred_std, alpha=0.2)
    plt.legend()
    plt.show()


def run(job: RunningJob) -> dict:

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
def test_lce_stopper(tmp_path):

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
    # test_bayesian_learning_curve_regression()
    test_lce_stopper(".")
