from deephyper.benchmark.nas.ensembleReg.load_data import load_data
from deephyper.problem import NaProblem
from deepspace.tabular import OneLayerFactory


def create_search_space(input_shape, output_shape, **kwargs):
    return OneLayerFactory()(input_shape, output_shape, **kwargs)


Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=100,
    learning_rate=0.1,
    optimizer="adam",
    num_epochs=1,
    callbacks=dict(
        ModelCheckpoint=dict(
            monitor="val_r2",
            mode="max",
            save_best_only=True,
            verbose=0,
            filepath="model.h5",
            save_weights_only=False,
        )
    ),
)

Problem.loss("mse")

Problem.metrics(["r2"])

Problem.objective("val_r2")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)

    model = Problem.get_keras_model([1])
