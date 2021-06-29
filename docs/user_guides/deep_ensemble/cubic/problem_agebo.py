"""
deephyper nas agebo --problem nas_ensemble.cubic.problem_agebo.Problem --max-evals 10
"""

from deephyper.nas.preprocessing import stdscaler
from deephyper.problem import NaProblem
from dh_project.cubic.load_data import load_data_train_valid
from dh_project.cubic.search_space import create_search_space


Problem = NaProblem(seed=42)

Problem.load_data(
    load_data_train_valid, random_state=42
)

Problem.preprocessing(stdscaler)

Problem.search_space(create_search_space, num_layers=2)

Problem.hyperparameters(
    batch_size=Problem.add_hyperparameter((1, 20), "batch_size"),
    learning_rate=Problem.add_hyperparameter(
        (1e-4, 0.1, "log-uniform"),
        "learning_rate",
    ),
    optimizer=Problem.add_hyperparameter(
        ["sgd", "rmsprop", "adagrad", "adam", "adadelta", "adamax", "nadam"],
        "optimizer",
    ),
    patience_ReduceLROnPlateau=Problem.add_hyperparameter(
        (3, 30), "patience_ReduceLROnPlateau"
    ),
    patience_EarlyStopping=Problem.add_hyperparameter(
        (3, 30), "patience_EarlyStopping"
    ),
    num_epochs=100,
    callbacks=dict(
        ReduceLROnPlateau=dict(monitor="val_loss", mode="min", verbose=0, patience=5),
        EarlyStopping=dict(
            monitor="val_loss", mode="min", verbose=0, patience=10  # or 'val_acc' ?
        ),
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

Problem.loss("mse") # the loss is the mean squared error

Problem.metrics(["r2", "rmse"]) # we want additionnal metrics for post-analysis

Problem.objective("-val_loss") # we want to maximize the negative of the validation MSE


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
