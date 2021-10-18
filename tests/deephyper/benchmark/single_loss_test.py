from deephyper.nas.run import run_base_trainer
from deephyper.benchmark.nas.linearReg import Problem


def test_single_loss():
    print(Problem)

    config = Problem.space
    config["hyperparameters"]["verbose"] = 1

    # Baseline
    config["arch_seq"] = [0.5]

    run_base_trainer(config)


if __name__ == "__main__":
    test_single_loss()
