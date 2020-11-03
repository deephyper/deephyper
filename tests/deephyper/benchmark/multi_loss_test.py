from deephyper.post.pipeline import train
from deephyper.nas.run.alpha import run
from deephyper.benchmark.nas.linearRegMultiLoss import Problem


def test_multi_loss():
    print(Problem)

    config = Problem.space
    config["hyperparameters"]["verbose"] = 1

    # Baseline
    config["arch_seq"] = [1.0] * 19

    run(config)


if __name__ == "__main__":
    test_multi_loss()
