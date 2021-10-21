from deephyper.benchmark.nas.covertype.load_data import load_data
from deephyper.problem import NaProblem
from deepspace.tabular import DenseSkipCoSpace

Problem = NaProblem()

Problem.load_data(load_data)

Problem.search_space(DenseSkipCoSpace, regression=False, bn=False, num_layers=10)

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
