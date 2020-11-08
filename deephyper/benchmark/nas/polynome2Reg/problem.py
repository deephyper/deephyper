from deephyper.problem import NaProblem
from deephyper.benchmark.nas.polynome2Reg.load_data import load_data
from deephyper.nas.space.dense_skipco import create_search_space
from deephyper.nas.preprocessing import minmaxstdscaler

Problem = NaProblem()

Problem.load_data(load_data)

Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space, num_layers=3)

Problem.hyperparameters(
    batch_size=100, learning_rate=0.1, optimizer="adam", num_epochs=10
)

Problem.loss("mse")

Problem.metrics(["r2"])

Problem.objective("val_r2")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
