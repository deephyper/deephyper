from deephyper.problem import NaProblem
from deephyper.benchmark.nas.linearRegMultiLoss.load_data import load_data
from deephyper.search.nas.model.baseline.simple_bi_model import create_search_space

from deephyper.search.nas.model.preprocessing import minmaxstdscaler

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

# Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space, num_layers=10)

Problem.hyperparameters(
    batch_size=100, learning_rate=0.1, optimizer="adam", num_epochs=20
)

Problem.loss(
    loss={"output_0": "mse", "output_1": "mse"},
    weights={"output_0": 0.0, "output_1": 1.0},
)

Problem.metrics({"output_0": ["r2", "mse"], "output_1": "mse"})

Problem.objective("val_output_0_r2")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
