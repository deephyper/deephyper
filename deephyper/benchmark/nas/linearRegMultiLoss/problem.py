from deephyper.problem import NaProblem
from deephyper.benchmark.nas.linearRegMultiLoss.load_data import load_data
from deepspace.tabular import SupervisedRegAutoEncoderSpace


Problem = NaProblem()

Problem.load_data(load_data)

Problem.search_space(SupervisedRegAutoEncoderSpace, num_layers=10)

Problem.hyperparameters(
    batch_size=100, learning_rate=0.1, optimizer="adam", num_epochs=20
)

Problem.loss(
    loss={"output_0": "mse", "output_1": "mse"},
    loss_weights={"output_0": 0.0, "output_1": 1.0},
)

Problem.metrics({"output_0": ["r2", "mse"], "output_1": "mse"})

Problem.objective("val_output_0_r2")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
