from deephyper.nas.run import run_base_trainer
from deephyper.benchmark.nas.linearRegMultiVar import Problem


def test_single_loss_multi_var():
    print(Problem)

    config = Problem.space
    config["hyperparameters"]["verbose"] = 1

    # Baseline
    config["arch_seq"] = [0.5] * 20

    run_base_trainer(config)


if __name__ == "__main__":
    test_single_loss_multi_var()
