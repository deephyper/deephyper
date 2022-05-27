import pytest


@pytest.mark.nas
def test_multi_loss():
    from deephyper.nas.run import run_base_trainer
    from deephyper.benchmark.nas.linearRegMultiLoss import Problem

    config = Problem.space
    config["hyperparameters"]["verbose"] = 1

    # Baseline
    config["arch_seq"] = [1.0] * 19

    run_base_trainer(config)
