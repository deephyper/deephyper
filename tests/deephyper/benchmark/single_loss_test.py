from deephyper.nas.run.alpha import run
from deephyper.benchmark.nas.linearReg import Problem


def test_single_loss():
    print(Problem)

    config = Problem.space
    config["hyperparameters"]["verbose"] = 1

    # Baseline
    config["arch_seq"] = [0.5]

    run(config)


if __name__ == "__main__":
    test_single_loss()
