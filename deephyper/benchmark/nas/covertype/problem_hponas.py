import traceback

from deephyper.benchmark.nas.covertype.load_data import load_data
from deephyper.problem import NaProblem
from deephyper.search.nas.model.baseline.dense_skipco import create_search_space


Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_search_space, num_layers=10, regression=False, bn=False)

Problem.hyperparameters(
    # batch_size=256,  # (128, 2048),
    batch_size=[32, 64, 128, 256, 512, 1024, 2048, 4096],
    learning_rate=(0.001, 0.1, "log-uniform"),
    optimizer="adam",
    num_epochs=20,
    verbose=0,
    callbacks=dict(CSVExtendedLogger=dict()),
)

Problem.loss("categorical_crossentropy")

Problem.metrics(["acc"])

Problem.objective("val_acc")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)

    # model = Problem.get_keras_model([4 for _ in range(20)])
