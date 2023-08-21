import pytest
import numpy as np


def load_data(dim=100):
    """
    Generate data for linear function -sum(x_i).

    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """
    rs = np.random.RandomState(42)
    size = 100000
    prop = 0.80
    a, b = 0, 100
    d = b - a
    x = np.array([a + rs.random(dim) * d for i in range(size)])
    y = np.array([[np.sum(v)] for v in x])

    sep_index = int(prop * size)
    train_X = x[:sep_index]
    train_y = y[:sep_index]

    valid_X = x[sep_index:]
    valid_y = y[sep_index:]

    print(f"train_X shape: {np.shape(train_X)}")
    print(f"train_y shape: [{np.shape(train_X)}, {np.shape(train_y)}]")
    print(f"valid_X shape: {np.shape(valid_X)}")
    print(f"valid_y shape: [{np.shape(valid_X)}, {np.shape(valid_y)}]")
    return (train_X, [train_y, train_X]), (valid_X, [valid_y, valid_X])


@pytest.mark.nas
def test_multi_loss():
    from deephyper.evaluator import RunningJob
    from deephyper.nas.run import run_base_trainer
    from deephyper.problem import NaProblem
    from deephyper.nas.spacelib.tabular import SupervisedRegAutoEncoderSpace

    Problem = NaProblem()
    Problem.load_data(load_data)
    Problem.search_space(SupervisedRegAutoEncoderSpace, num_layers=10)
    Problem.hyperparameters(
        batch_size=100, learning_rate=0.1, optimizer="adam", num_epochs=1
    )
    Problem.loss(
        loss={"output_0": "mse", "output_1": "mse"},
        loss_weights={"output_0": 0.0, "output_1": 1.0},
    )
    Problem.metrics({"output_0": ["r2", "mse"], "output_1": "mse"})
    Problem.objective("val_output_0_r2")

    config = Problem.space
    config["hyperparameters"]["verbose"] = 1

    # Baseline
    config["arch_seq"] = [0.5] * 19

    job = RunningJob(id=0, parameters=config)
    result = run_base_trainer(job)


if __name__ == "__main__":
    test_multi_loss()
