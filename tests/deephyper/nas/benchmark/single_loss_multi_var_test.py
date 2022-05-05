import pytest


@pytest.mark.nas
def test_single_loss_multi_var():
    from deephyper.nas.run import run_base_trainer
    from deephyper.benchmark.nas.linearRegMultiVar import Problem

    config = Problem.space
    config["hyperparameters"]["verbose"] = 1

    # Baseline
    config["arch_seq"] = [0.5] * 20

    run_base_trainer(config)
