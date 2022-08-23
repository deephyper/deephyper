from deephyper.nas.spacelib.tabular import OneLayerSpace
from deephyper.problem import NaProblem
from deephyper.test.nas.linearReg.load_data import load_data

Problem = NaProblem()

Problem.load_data(load_data)

Problem.search_space(OneLayerSpace)

Problem.hyperparameters(
    batch_size=Problem.add_hyperparameter((1, 100), "batch_size"),
    learning_rate=Problem.add_hyperparameter(
        (1e-4, 1e-1, "log-uniform"), "learning_rate"
    ),
    optimizer=Problem.add_hyperparameter(["adam", "nadam", "rmsprop"], "optimizer"),
    num_epochs=1,
)

Problem.loss("mse")

Problem.metrics(["r2"])

Problem.objective("val_r2")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)

    model = Problem.get_keras_model([1])
