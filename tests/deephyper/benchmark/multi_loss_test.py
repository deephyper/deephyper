from deephyper.nas.run import run_base_trainer
from deephyper.benchmark.nas.linearRegMultiLoss import Problem


def test_multi_loss():
    print(Problem)

    config = Problem.space
    config["hyperparameters"]["verbose"] = 1

    # Baseline
    config["arch_seq"] = [1.0] * 19

    run_base_trainer(config)


if __name__ == "__main__":
    test_multi_loss()
