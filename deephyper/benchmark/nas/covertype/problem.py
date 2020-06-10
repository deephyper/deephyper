import traceback

from deephyper.benchmark.nas.covertype.load_data import load_data
from deephyper.problem import NaProblem
from deephyper.search.nas.model.baseline.dense_skipco import create_search_space

# from deephyper.search.nas.model.preprocessing import minmaxstdscaler

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

# Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space, num_layers=10, regression=False, bn=False)

Problem.hyperparameters(
    batch_size=256,
    learning_rate=0.01,
    optimizer="adam",
    num_epochs=100,
    verbose=0,
    callbacks=dict(CSVExtendedLogger=dict()),
)

Problem.loss("categorical_crossentropy")

Problem.metrics(["acc"])


# def alc(infos: dict) -> float:
#     return sum(infos["val_acc"])


# def bacc_with_pred(infos):
#     from sklearn.utils import class_weight
#     from sklearn import metrics
#     import numpy as np

#     y_pred = infos["y_pred"]
#     y_true = infos["y_true"]

#     try:
#         y_pred = np.argmax(y_pred, axis=1)
#         y_true = np.argmax(y_true, axis=1)

#         cw = class_weight.compute_class_weight("balanced", np.unique(y_true), y_true)
#         sw = np.array([cw[class_ - 1] for class_ in y_true])
#         bacc = metrics.accuracy_score(y_true, y_pred, sample_weight=sw)
#     except:
#         traceback.print_exc()
#         bacc = -1
#     return bacc


Problem.objective("val_acc")


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)

    # model = Problem.get_keras_model([4 for _ in range(20)])
