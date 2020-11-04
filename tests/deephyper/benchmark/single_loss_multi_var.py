from deephyper.nas.run.alpha import run
from deephyper.benchmark.nas.linearRegMultiVar import Problem


def test_single_loss_multi_var():
    print(Problem)

    config = Problem.space
    config["hyperparameters"]["verbose"] = 1

    # Baseline
    config["arch_seq"] = [0.5] * 20

    run(config)


if __name__ == "__main__":
    test_single_loss_multi_var()
