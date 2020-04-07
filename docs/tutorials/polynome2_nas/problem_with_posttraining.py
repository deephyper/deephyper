from nas_problems.polynome2.load_data import load_data
from nas_problems.polynome2.search_space import create_search_space
from deephyper.problem import NaProblem

Problem = NaProblem()

Problem.load_data(load_data, size=1000)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=128, learning_rate=0.001, optimizer="rmsprop", num_epochs=5
)

Problem.loss("mse")

Problem.metrics(["r2"])

Problem.objective("val_r2__last")

Problem.post_training(
    num_epochs=60,
    metrics=["r2"],
    model_checkpoint={
        "monitor": "val_r2",
        "mode": "max",
        "save_best_only": True,
        "verbose": 1,
    },
    early_stopping={"monitor": "val_r2", "mode": "max", "verbose": 1, "patience": 5},
)

if __name__ == "__main__":
    print(Problem)
