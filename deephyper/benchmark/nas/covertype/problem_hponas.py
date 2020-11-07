import traceback

from deephyper.benchmark.nas.covertype.load_data import load_data
from deephyper.problem import NaProblem
from deepspace.tabular import DenseSkipCoFactory


def create_search_space(input_shape=(54,), output_shape=(7,), num_layers=10, **kwargs):
    return DenseSkipCoFactory()(
        input_shape,
        output_shape,
        num_layers=num_layers,
        regression=False,
        bn=False,
        **kwargs
    )


Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_search_space, num_layers=10)

Problem.hyperparameters(
    batch_size=[32, 64, 128, 256, 512, 1024],
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
