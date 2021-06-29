from deephyper.nas.preprocessing import stdscaler
from deephyper.problem import NaProblem
from dh_project.cubic.load_data import load_data_train_valid
from dh_project.cubic.search_space import create_search_space


Problem = NaProblem(seed=42)

Problem.load_data(load_data_train_valid, random_state=42)

Problem.preprocessing(stdscaler)

Problem.search_space(create_search_space, num_layers=2)

Problem.hyperparameters(
    batch_size=4,
    learning_rate=1e-3,
    optimizer="adam",
    num_epochs=100,
    callbacks=dict(
        ReduceLROnPlateau=dict(monitor="val_loss", mode="min", verbose=0, patience=5),
        EarlyStopping=dict(monitor="val_loss", mode="min", verbose=0, patience=10),
        ModelCheckpoint=dict(
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
            filepath="model.h5",
            save_weights_only=False,
        ),
    ),
)

Problem.loss("mse")

Problem.metrics(["r2", "rmse"])

Problem.objective("-val_loss")  # we want to maximise the Log-Likelihood


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
